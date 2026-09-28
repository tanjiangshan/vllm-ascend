# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# 【文件职责】DeepSeek V4.1 的 Ascend 视觉塔（ViT）与对齐器（Aligner）实现。
#
# 【在模型中的位置】vl_model.py 组合本文件的视觉塔与语言模型：图像经 MM
# processor 预处理后成为 patch 序列 → ViT 提特征（全双向注意力 + 2D RoPE）
# → Aligner 做 r×r 空间合并下采样并投影到 hidden_size → 与 image_start/
# image_end/image_newline 哨兵向量拼成 image span → 替换进文本嵌入序列。
#
# 【多模态 ViT 与文本解码器的关键差异】
#   - 注意力是双向的（图像内无因果掩码），走 MMEncoderAttention 接口；
#   - 位置编码是 2D RoPE：把 h/w 两个坐标轴的频率交错排布（见
#     get_vision_cos_sin），使模型感知 patch 的二维空间位置；
#   - 视觉塔完整复制（不做 TP/DP 切分），因为视觉计算量占比小、切分
#     通信开销不划算。
#
# 【NPU 适配点】注意力经 MMEncoderAttention 抽象落到 Ascend 后端；权重名与
# HF checkpoint 完全一致，加载时无需重命名。
# =============================================================================
"""Ascend DeepSeek vision tower (ViT + aligner), replicated (no TP/DP sharding).

Ported from the official reference implementation
(deepseek-ai/DeepSeek-V4-Flash-Vision-Exp). Weight names match the HF
checkpoint so no renaming is needed at load time.
"""

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb


@lru_cache(8)
def get_vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """预计算 2D RoPE 的 cos/sin 表（进程级缓存）。

    参数:
        n_h/n_w: ViT 特征图的网格高/宽（patch 数）。
        dim: 每个头的 RoPE 维度（head_dim 的一半，因 cos/sin 各占一半）。
        theta: RoPE 基底（如 10000）。
    返回:
        (cos, sin): 各为 [n_h*n_w, 1, dim] 的 FP32 张量。
    原理: 2D RoPE——先算标准逆频率 inv_freq（dim/2 个频率）；再把每个 patch 的
        (h, w) 坐标分别与频率相乘，h 与 w 的频率分量在通道维上交错堆叠
        （stack 后 reshape 成 [-1, 2, 1] 再 flatten），使一半通道编码 h 位置、
        一半编码 w 位置。
    语法点: @lru_cache(8) 装饰器按参数缓存返回值（最多 8 组），不同分辨率
        网格的 cos/sin 只计算一次；返回的张量会被调用方 .to(device) 拷贝。
    """
    # 逆频率: 1 / theta^(2i/dim)，i 为通道对索引。
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    # hpos/wpos: 广播生成 [n_h, n_w] 的二维坐标网格。
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    # 交错堆叠 h/w 坐标: [n_h*n_w, 2, 1]，flatten 后每个 patch 得到 dim 个
    # 频率分量（h 分量与 w 分量交替）。
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float()
    freqs = (freqs * inv_freq).flatten(1)
    # unsqueeze(1) 插入头维: [tokens, 1, dim]，供各头广播使用。
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


class DeepseekV41PatchEmbed(nn.Module):
    """patch 嵌入层：把每个 RGB patch 展平后线性投影到 vision_dim。"""

    def __init__(self, config):
        """参数: config 需含 vision_patch_size / vision_dim。"""
        super().__init__()
        # 输入维度 = 3 通道 × patch_size² 像素（MM processor 已把图像切成
        # patch 并展平成 [num_patches, 3*p*p]）。
        self.proj = nn.Linear(3 * config.vision_patch_size**2, config.vision_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [num_patches, 3*patch_size²] → [num_patches, vision_dim]。"""
        return self.proj(x.flatten(1))


class DeepseekV41VisionAttention(nn.Module):
    """ViT 自注意力：融合 QKV 投影 + 2D RoPE + 双向编码器注意力。"""

    def __init__(self, config):
        """参数: config 需含 vision_n_heads / vision_dim。"""
        super().__init__()
        self.n_heads = config.vision_n_heads
        self.head_dim = config.vision_dim // config.vision_n_heads
        # 融合的 QKV 投影（一次矩阵乘代替三次）。
        self.wqkv = nn.Linear(config.vision_dim, 3 * config.vision_dim)
        self.wo = nn.Linear(config.vision_dim, config.vision_dim)
        # RoPE 应用器: neox 风格（前半/后半配对），FP32 计算保证数值精度。
        self.apply_rotary_emb = ApplyRotaryEmb(
            enforce_enable=True,
            is_neox_style=True,
            enable_fp32_compute=True,
        )
        # MMEncoderAttention: vLLM 多模态编码器注意力抽象，在 Ascend 上
        # 落到对应后端（无因果掩码，全双向）。
        self.attn = MMEncoderAttention(
            num_heads=self.n_heads,
            head_size=self.head_dim,
            scale=self.head_dim**-0.5,
        )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """一次视觉自注意力。

        参数:
            x: [num_patches, vision_dim] 同一图像的全部 patch。
            cos/sin: [num_patches, 1, rope_dim] 2D RoPE 表。
        返回:
            [num_patches, vision_dim] 注意力输出。
        算法步骤:
            1) wqkv 一次投影后 chunk 成 q/k/v，各 reshape 出头维；
            2) q/k 堆叠后统一施加 RoPE（一次调用处理两个张量）；
            3) MMEncoderAttention 做双向注意力（bsz 维用 unsqueeze 补齐）；
            4) wo 输出投影。
        """
        n = x.size(0)
        # 步骤1: 融合投影 → 三等分 → 各自变成 [n, n_heads, head_dim]。
        q, k, v = (t.view(n, self.n_heads, self.head_dim) for t in self.wqkv(x).chunk(3, dim=-1))
        # 步骤2: stack 成 [2, n, heads, dim] 一次旋转再 unbind 拆回 q/k。
        qk = self.apply_rotary_emb(torch.stack((q, k)), cos.squeeze(1), sin.squeeze(1))
        q, k = qk.unbind()
        # 步骤3: 编码器注意力需要 [bsz=1, seq, heads, dim] 布局。
        o = self.attn(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
        # 步骤4: 合并头维 + 输出投影。
        return self.wo(o.squeeze(0).reshape(n, -1))


class DeepseekV41VisionMLP(nn.Module):
    """ViT 前馈层: SwiGLU（w1 升维 2 倍 → SiluAndMul → w2 降回）。"""

    def __init__(self, config):
        """参数: config 需含 vision_dim / vision_inter_dim。"""
        super().__init__()
        # w1 输出 2*inter_dim：SiluAndMul 对前半 SiLU 再乘后半（SwiGLU 门控）。
        self.w1 = nn.Linear(config.vision_dim, 2 * config.vision_inter_dim, bias=False)
        self.w2 = nn.Linear(config.vision_inter_dim, config.vision_dim, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [num_patches, vision_dim] → 同形状输出。"""
        return self.w2(self.act_fn(self.w1(x)))


class DeepseekV41VisionBlock(nn.Module):
    """ViT Transformer 块: pre-norm 残差结构（先 norm 再计算，残差相加）。"""

    def __init__(self, config):
        """参数: config 需含 vision_dim / vision_n_heads / vision_inter_dim。"""
        super().__init__()
        # FP32 RMSNorm: 视觉层用 FP32 计算归一化以保证精度。
        self.norm1 = RMSNorm(config.vision_dim, eps=1e-6, dtype=torch.float32)
        self.attn = DeepseekV41VisionAttention(config)
        self.norm2 = RMSNorm(config.vision_dim, eps=1e-6, dtype=torch.float32)
        self.mlp = DeepseekV41VisionMLP(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """x: [num_patches, vision_dim]；cos/sin 为该图网格的 2D RoPE 表。"""
        # 注意力子层: x + Attn(Norm(x))。
        residual = x + self.attn(self.norm1(x), cos, sin)
        # MLP 子层: residual + MLP(Norm(residual))。
        return residual + self.mlp(self.norm2(residual))


class DeepseekV41ViT(nn.Module):
    """Ascend DeepSeek ViT: full bidirectional attention per image, 2D RoPE."""
    """【中文说明】视觉塔主体：patch 嵌入 → N 个 VisionBlock → 最终 RMSNorm。
    每张图独立做全双向注意力（图与图之间不交互）。"""

    def __init__(self, config):
        """参数: config 需含 vision_dim/n_heads/n_layers/rope_theta/patch_size。"""
        super().__init__()
        # rope_dim = head_dim 的一半（2D RoPE 中 h/w 各占一半通道）。
        self.rope_dim = config.vision_dim // config.vision_n_heads // 2
        self.rope_theta = config.vision_rope_theta
        self.patch_embed = DeepseekV41PatchEmbed(config)
        self.blocks = nn.ModuleList([DeepseekV41VisionBlock(config) for _ in range(config.vision_n_layers)])
        self.norm = RMSNorm(config.vision_dim, eps=1e-6, dtype=torch.float32)

    def forward(self, patches: torch.Tensor, n_vit_h: int, n_vit_w: int) -> torch.Tensor:
        """视觉塔前向。

        参数:
            patches: [num_patches, 3*patch_size²] 单张图的 patch 序列。
            n_vit_h/n_vit_w: 该图的 ViT 网格尺寸。
        返回:
            [num_patches, vision_dim] 归一化后的视觉特征。
        """
        x = self.patch_embed(patches)
        # 按网格尺寸取缓存的 2D RoPE 表并搬到当前设备。
        cos, sin = get_vision_cos_sin(n_vit_h, n_vit_w, self.rope_dim, self.rope_theta)
        cos = cos.to(device=x.device)
        sin = sin.to(device=x.device)
        # 逐块过 Transformer，同一图的 cos/sin 全层共享。
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class DeepseekV41Aligner(nn.Module):
    """Spatial merge (downsample_ratio x downsample_ratio) + MLP projector."""
    """【中文说明】对齐器：把相邻 downsample_ratio×downsample_ratio 个 ViT
    patch 特征拼成一帧，再经两层 MLP（GELU）投影到语言模型的 hidden_size。
    这是典型的多模态"空间合并 + 投影"对齐范式，可显著减少 LLM 侧的图像
    token 数量。"""

    def __init__(self, config):
        """参数: config 需含 vision_downsample_ratio / vision_dim / hidden_size。"""
        super().__init__()
        self.downsample_ratio = config.vision_downsample_ratio
        # 合并后每帧的输入维度 = vision_dim × r²（r×r 个 patch 拼接）。
        in_dim = config.vision_dim * self.downsample_ratio**2
        self.w1 = nn.Linear(in_dim, config.hidden_size)
        self.w2 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x: torch.Tensor, n_vit_h: int, n_vit_w: int) -> torch.Tensor:
        """空间合并 + 投影。

        参数:
            x: [n_vit_h*n_vit_w, vision_dim] ViT 输出特征。
            n_vit_h/n_vit_w: ViT 网格尺寸。
        返回:
            [ceil(n_h/r)*ceil(n_w/r), hidden_size] 合并后的图像 token。
        算法步骤:
            1) 还原成 [n_h, n_w, dim] 网格并转到通道在前 [C, H, W]；
            2) 右/下零填充到 r 的整数倍；
            3) F.unfold 以 r×r 不重叠滑窗取出每帧的 r² 个 patch（展开拼接）；
            4) w1 → GELU → w2 投影到语言空间。
        """
        r = self.downsample_ratio
        # 步骤1: [n, dim] → [n_h, n_w, dim] → [dim, n_h, n_w]。
        x = x.view(n_vit_h, n_vit_w, -1).permute(2, 0, 1)
        # 步骤2: 填充到 r 整数倍（负数取模技巧: -n % r = r - n%r）。
        x = F.pad(x, (0, -n_vit_w % r, 0, -n_vit_h % r))
        # 步骤3: unfold 提取 r×r 窗口: [dim, n_frames, r²]，转置成
        # [n_frames, r²*dim]（每帧 = 相邻 r² 个 patch 特征拼接）。
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        # 步骤4: 两层 MLP 投影（GELU 激活，区别于文本侧的 SwiGLU）。
        return self.w2(F.gelu(self.w1(x)))
