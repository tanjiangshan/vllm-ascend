# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ============================================================================
# 【模块职责】DeepSeek-V4 视觉塔（Vision Tower）的昇腾 NPU 实现，包含两部分：
#   1) DeepseekV4ViT     —— 视觉 Transformer 编码器（ViT）：对单张图像的 patch
#                           序列做双向（非因果）注意力编码，使用 2D RoPE
#                           （行/列坐标各自占用一半旋转频率）。
#   2) DeepseekV4Aligner —— 对齐器：把 ViT 输出的相邻 r×r 个 patch 特征拼接
#                           （空间合并/像素 unshuffling）后经两层 MLP 投影到
#                           语言模型 hidden_size，得到 LLM 可消费的图像 token。
# 本模块为“复制式”（replicated）实现：视觉塔不参与 TP/DP 切分，每卡持有完整
# 权重——视觉计算量相对语言主干很小，切分的通信开销得不偿失。
# 权重命名与 HF 官方 checkpoint（deepseek-ai/DeepSeek-V4-Flash-Vision-Exp）
# 完全一致，加载时无需重命名。
# ============================================================================
"""DeepSeek-V4 vision tower (ViT + aligner), replicated (no TP/DP sharding).

Ported from the official reference implementation
(deepseek-ai/DeepSeek-V4-Flash-Vision-Exp). Weight names match the HF
checkpoint so no renaming is needed at load time.
"""

# functools.lru_cache: 装饰器，按参数缓存函数返回值（最近最少使用淘汰），
# 这里缓存 RoPE 表——同一网格尺寸的多张图像只需计算一次。

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn


@lru_cache(8)
def get_vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """计算 2D RoPE 的 cos/sin 角度表（按网格尺寸缓存）。

    原理: ViT 中每个 patch 拥有 (行, 列) 二维坐标。本函数为行坐标和列坐标
    各分配 dim 个频率分量（交错排列），组合出每个 patch 位置的完整旋转角，
    再取 cos/sin 供 apply_rotary 使用。相比 1D 序列 RoPE，2D RoPE 能正确
    表达图像的空间平移等变性。

    Args:
        n_h: patch 网格行数。
        n_w: patch 网格列数。
        dim: 单个坐标方向（行或列）的 RoPE 维度（= head_dim // 2）。
        theta: RoPE 频率基底（如 10000.0，值越大低频分量越多）。

    Returns:
        (cos, sin): 形状均为 [n_h*n_w, 1, dim]；中间的 1 是 head 维占位，
        便于在 [num_tokens, num_heads, head_dim] 的 q/k 上广播。
    """
    # 步骤1: 逆频率 inv_freq[i] = theta^(-2i/dim)，共 dim 个频率（头低尾高）。
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    # 步骤2: 构造行/列坐标网格。hpos[i,j]=i（行号），wpos[i,j]=j（列号）。
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    # 步骤3: 行/列坐标按最后一维交错堆叠 -> [n_h*n_w, 2, 1]，乘逆频率后
    # flatten，得到每个 patch 的 [row 角度..., col 角度...]（交错排列）。
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float()
    freqs = (freqs * inv_freq).flatten(1)
    # 步骤4: cos/sin 并 unsqueeze(1) 增加 head 广播维度。
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """对输入张量最后一维施加 RoPE 旋转（half-split 非交错实现）。

    原理: 将特征向量按前后两半切开为 [x1; x2]，做二维旋转
    [x1*cos - x2*sin; x2*cos + x1*sin]，数学上等价于复数乘 e^{iθ}，
    使两向量的内积只依赖相对位置差。先升 float32 计算再转回原 dtype
    （bf16 直接算会累积数值误差）。

    Args:
        x: [num_tokens, num_heads, head_dim] 输入（q 或 k）。
        cos/sin: [num_tokens, 1, head_dim//2] 角度表（在 head 维上广播）。

    Returns:
        与 x 同形状、同 dtype 的旋转结果。
    """
    dtype = x.dtype
    # half-split 切分: x1 为前一半维度, x2 为后一半维度。
    x1, x2 = x.float().chunk(2, dim=-1)
    # 旋转公式（非交错模式，与 GPT-NeoX 的“前后两半”风格一致）。
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class DeepseekV4RMSNorm(nn.Module):
    """视觉塔专用 RMSNorm（独立实现，避免依赖 vLLM 通用层的 NPU 逻辑）。

    原理: RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight。相比 LayerNorm
    省去均值中心化，计算更轻。权重与中间计算均保持 float32，规避 bf16
    归一化的精度损失。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """初始化。

        Args:
            dim: 归一化的特征维度。
            eps: 方差下限（防除零）。
        """
        super().__init__()
        self.eps = eps
        # 可学习缩放权重，初始化为全 1，固定 float32 存储。
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向归一化。

        Args:
            x: [num_tokens, dim] 任意 batch 形状的输入。
        Returns:
            与输入同形状、同 dtype 的归一化结果。
        """
        dtype = x.dtype
        # 步骤1: 升 float32，计算 rsqrt(mean(x^2)+eps) 并逐元素缩放。
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        # 步骤2: 乘可学习权重后转回原 dtype。
        return (self.weight * x).to(dtype)


class DeepseekV4PatchEmbed(nn.Module):
    """patch 嵌入层：把展平后的图像 patch 像素线性投影成 ViT 特征。

    原理: mm_preprocess.load_image 已把图像切块并 reshape 为
    [num_patches, 3*p*p] 的展平像素向量；这里用一个
    Linear(3*p^2 -> vision_dim) 完成嵌入，数学上等价于
    Conv2d(k=p, stride=p)，但用 Linear 实现可与 HF 权重直接对齐。
    """

    def __init__(self, config):
        """初始化。

        Args:
            config: DeepseekV4Config，读取 vision_patch_size / vision_dim。
        """
        super().__init__()
        # 输入维度 3 * p^2 = RGB 三通道 × p × p 像素数。
        self.proj = nn.Linear(3 * config.vision_patch_size**2, config.vision_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """像素嵌入。

        Args:
            x: [num_patches, 3*p*p]（或 [num_patches, 3, p, p]）展平前的像素。
        Returns:
            [num_patches, vision_dim] 的 patch 特征。
        """
        # flatten(1) 保证最后一维是展平像素（兼容传入 4D patch 张量）。
        return self.proj(x.flatten(1))


class DeepseekV4VisionAttention(nn.Module):
    """视觉自注意力：单图像内双向全注意力（无因果掩码、无 KV Cache）。

    结构: 融合 wqkv 投影（一次 matmul 同时得 q/k/v）-> 2D RoPE ->
    F.scaled_dot_product_attention（PyTorch 自动调度 NPU 融合注意力核）->
    wo 输出投影。视觉序列短（每图数百 patch），无需 paged attention。
    """

    def __init__(self, config):
        """初始化。

        Args:
            config: 读取 vision_n_heads / vision_dim。
        """
        super().__init__()
        self.n_heads = config.vision_n_heads
        # 每头维度 = vision_dim / n_heads（整除约束）。
        self.head_dim = config.vision_dim // config.vision_n_heads
        # 融合 QKV: [vision_dim, 3*vision_dim]，输出可 chunk 成 q/k/v 三份。
        self.wqkv = nn.Linear(config.vision_dim, 3 * config.vision_dim)
        self.wo = nn.Linear(config.vision_dim, config.vision_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """双向自注意力前向。

        Args:
            x: [num_patches, vision_dim] 单张图像的 patch 特征序列。
            cos/sin: [num_patches, 1, rope_dim] 2D RoPE 角度表。
        Returns:
            [num_patches, vision_dim] 注意力输出。
        """
        n = x.size(0)
        # 步骤1: 融合投影 -> 按最后一维 chunk 成 3 份，用生成器表达式把每个
        # [n, 3*vision_dim] 的切片 reshape 为 [n, n_heads, head_dim]。
        # （生成器惰性求值，避免中间列表开销。）
        q, k, v = (t.view(n, self.n_heads, self.head_dim) for t in self.wqkv(x).chunk(3, dim=-1))
        # 步骤2: 仅对 q/k 施加 RoPE 位置旋转（v 不编码位置）。
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        # 步骤3: SDPA 要求 batch 维在前: [n, H, D] -> [H, n, D]。
        # is_causal 缺省 False —— ViT 是双向全注意力。
        o = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
        # 步骤4: 转回 [n, H, D] 并 flatten 头维度，过输出投影 wo。
        return self.wo(o.transpose(0, 1).reshape(n, -1))


class DeepseekV4VisionMLP(nn.Module):
    """视觉 MLP（SwiGLU 门控结构，与 LLM 的 FFN 同构）。

    原理: w1 一次投影出 2*inter 维，拆成 gate/up 两半，
    输出 = w2(silu(gate) * up)。相比普通 GELU MLP，SwiGLU 在同等参数量下
    表达能力更强，是 LLaMA/DeepSeek 系 FFN 的标准选择。
    """

    def __init__(self, config):
        """初始化。

        Args:
            config: 读取 vision_dim / vision_inter_dim。
        """
        super().__init__()
        # w1 输出 2*inter_dim，供 gate/up 各取一半。
        self.w1 = nn.Linear(config.vision_dim, 2 * config.vision_inter_dim, bias=False)
        self.w2 = nn.Linear(config.vision_inter_dim, config.vision_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU 前向。Args: x: [num_patches, vision_dim]。Returns: 同形状。"""
        # chunk(2) 把 w1 输出拆成 gate 与 up 两半。
        gate, up = self.w1(x).chunk(2, dim=-1)
        # silu(gate)*up 再经 w2 降维回 vision_dim。
        return self.w2(F.silu(gate) * up)


class DeepseekV4VisionBlock(nn.Module):
    """ViT 基础块: pre-norm 残差结构（x = x + Attn(Norm(x))，再 x = x + MLP(Norm(x))）。

    与 LLM decoder 层的残差组织一致，但注意力为双向且无 KV Cache。
    """

    def __init__(self, config):
        """初始化。Args: config: DeepseekV4Config。"""
        super().__init__()
        # pre-norm: 注意力与 MLP 前各接一个 RMSNorm（残差分支内归一化）。
        self.norm1 = DeepseekV4RMSNorm(config.vision_dim)
        self.attn = DeepseekV4VisionAttention(config)
        self.norm2 = DeepseekV4RMSNorm(config.vision_dim)
        self.mlp = DeepseekV4VisionMLP(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """前向。

        Args:
            x: [num_patches, vision_dim] 输入特征。
            cos/sin: 2D RoPE 角度表（传给注意力）。
        Returns:
            同形状的残差累加结果。
        """
        # 子块1: 双向注意力残差分支。
        x = x + self.attn(self.norm1(x), cos, sin)
        # 子块2: SwiGLU MLP 残差分支。
        return x + self.mlp(self.norm2(x))


class DeepseekV4ViT(nn.Module):
    """DeepSeek-V4 ViT: full bidirectional attention per image, 2D RoPE."""

    # 【中文补充】视觉编码器主体: patch 嵌入 -> N 个 VisionBlock -> 末端
    # RMSNorm。每张图像独立编码（图像间不互相注意），输入网格尺寸动态
    # 决定 RoPE 表（lru_cache 缓存复用）。

    def __init__(self, config):
        """初始化 ViT。

        Args:
            config: 读取 vision_dim / vision_n_heads / vision_n_layers /
                vision_rope_theta 等。
        """
        super().__init__()
        # 每个坐标方向（行/列）的 RoPE 维度 = head_dim / 2
        # （另一半频率留给另一坐标轴，实现 2D 位置编码）。
        self.rope_dim = config.vision_dim // config.vision_n_heads // 2
        self.rope_theta = config.vision_rope_theta
        self.patch_embed = DeepseekV4PatchEmbed(config)
        # ModuleList + 列表推导式: 按序堆叠 vision_n_layers 个 Block。
        self.blocks = nn.ModuleList([DeepseekV4VisionBlock(config) for _ in range(config.vision_n_layers)])
        self.norm = DeepseekV4RMSNorm(config.vision_dim)

    def forward(self, patches: torch.Tensor, n_vit_h: int, n_vit_w: int) -> torch.Tensor:
        """ViT 前向。

        Args:
            patches: [n_vit_h*n_vit_w, 3, p, p] 单张图像的 patch 像素。
            n_vit_h/n_vit_w: 该图像的 patch 网格行/列数。
        Returns:
            [n_vit_h*n_vit_w, vision_dim] 每个 patch 的视觉特征。
        """
        # 步骤1: patch 像素 -> 特征向量。
        x = self.patch_embed(patches)
        # 步骤2: 按网格尺寸生成 2D RoPE 表（lru_cache 跨图像复用同尺寸结果）。
        cos, sin = get_vision_cos_sin(n_vit_h, n_vit_w, self.rope_dim, self.rope_theta)
        # 步骤3: cos/sin 在 CPU 上计算，需搬到输入所在的 NPU 设备。
        cos = cos.to(device=x.device)
        sin = sin.to(device=x.device)
        # 步骤4: 逐层过 VisionBlock。
        for block in self.blocks:
            x = block(x, cos, sin)
        # 步骤5: 末端 RMSNorm。
        return self.norm(x)


class DeepseekV4Aligner(nn.Module):
    """Spatial merge (downsample_ratio x downsample_ratio) + MLP projector."""

    # 【中文补充】对齐器: 把 ViT 的 r×r 相邻 patch 特征拼接（空间合并，
    # 即像素 unshuffling 的特征版）成单个 LLM token 表示，再经两层 MLP
    #（GELU 激活）投影到语言模型 hidden_size。r = vision_downsample_ratio。

    def __init__(self, config):
        """初始化。

        Args:
            config: 读取 vision_dim / vision_downsample_ratio / hidden_size。
        """
        super().__init__()
        self.downsample_ratio = config.vision_downsample_ratio
        # 空间合并后单个 LLM token 的输入维度 = vision_dim * r^2
        # （r×r 个相邻 patch 的特征首尾拼接）。
        in_dim = config.vision_dim * self.downsample_ratio**2
        self.w1 = nn.Linear(in_dim, config.hidden_size)
        self.w2 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x: torch.Tensor, n_vit_h: int, n_vit_w: int) -> torch.Tensor:
        """空间合并 + 两层 MLP 投影。

        Args:
            x: [n_vit_h*n_vit_w, vision_dim] ViT 输出特征。
            n_vit_h/n_vit_w: ViT patch 网格尺寸。
        Returns:
            [n_llm_h*n_llm_w, hidden_size] 每个 LLM 图像 token 的嵌入，
            其中 n_llm_h = ceil(n_vit_h/r)、n_llm_w = ceil(n_vit_w/r)。
        """
        r = self.downsample_ratio
        # 步骤1: 一维序列还原为 2D 网格 [H, W, C]，permute 成 [C, H, W]
        # 以便用 F.pad/F.unfold 做空间操作。
        x = x.view(n_vit_h, n_vit_w, -1).permute(2, 0, 1)
        # 步骤2: 右/下边缘补零至 r 的整数倍，保证空间切分无遗漏。
        # 技巧: -n % r 得到“补到下一个 r 倍数”所需的元素个数。
        x = F.pad(x, (0, -n_vit_w % r, 0, -n_vit_h % r))
        # 步骤3: F.unfold 以步长 r 取 r×r 不重叠滑窗（像素 unshuffling），
        # 得到 [C*r*r, L]，其中 L = n_llm_h*n_llm_w；transpose 回 [L, C*r*r]。
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        # 步骤4: 两层 MLP（GELU 激活）投影到语言模型 hidden_size。
        return self.w2(F.gelu(self.w1(x)))
