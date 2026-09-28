# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash vision tower and multimodal processor.

GLM-5.3-Flash 的视觉塔（ViT）与多模态处理器注册。

视觉塔张量流（图像/视频 -> 视觉 token）：
  像素 [L, C*T*P*P]（L 个 patch，已由 processor 展平）
    -> Glm5NextVisionPatchEmbed（3D 卷积，stride=kernel，等效 patch 化）
    -> depth 个 Glm5NextVisionBlock（RMSNorm + 视觉注意力 + SwiGLU MLP）
    -> post_layernorm
    -> spatial_merge（Conv2d 下采样 2x2）+ Glm5NextPatchMerger（投影瓶颈）
    -> [num_vision_tokens, out_hidden_size] 视觉 token 嵌入
  视觉 token 嵌入随后替换输入序列中的 <|image|>/<|video|> 占位符，
  进入语言模型（多模态前向由上游 Glm4vForConditionalGeneration 驱动）。

位置编码：2D mRoPE——h/w 两轴各半 head_dim，按 spatial_merge 重排；
视频按帧重复 t 次。

NPU/工程适配点：
  - 视觉注意力用 MMEncoderAttention（选定的后端，NPU 上为 ASCEND）；
  - q/k 归一化用融合算子 fused_q_kv_rmsnorm（一次内核两个权重）；
  - 支持 ViT 数据并行（每卡完整塔、按 batch 分片）或张量并行；
  - encoder_metadata 预计算路径支持视觉 CUDA Graph（encode 阶段捕获）。
"""

from collections.abc import Mapping
from functools import cached_property, partial

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    parallel_state,
)
from vllm.distributed import utils as dist_utils
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.conv import Conv2dLayer, Conv3dLayer
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.models.glm4_1v import (
    Glm4vMultiModalProcessor,
    Glm4vProcessingInfo,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
)
from vllm.model_executor.models.vision import (
    get_vit_attn_backend,
    is_vit_use_data_parallel,
)
from vllm.models.common.ops import fused_q_kv_rmsnorm
from vllm.multimodal.parse import ImageSize, MultiModalDataItems
from vllm.v1.attention.backends.registry import AttentionBackendEnum


class Glm5NextVisionPatchEmbed(nn.Module):
    """视觉 patch 嵌入：3D 卷积把像素块投影为向量。

    用 kernel_size = stride = (temporal_patch, patch, patch) 的 Conv3d
    同时完成"切块 + 线性投影"（非重叠切块时卷积等效于逐 patch 线性层）。
    """

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 1,
        in_channels: int = 3,
        hidden_size: int = 1536,
    ) -> None:
        """初始化。

        参数：
            patch_size: 空间 patch 边长 P。
            temporal_patch_size: 时间 patch 大小 T（视频帧分组）。
            in_channels: 输入通道（RGB=3，或展开后的 C*T）。
            hidden_size: 输出嵌入维度。
        """
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = Conv3dLayer(
            in_channels,
            hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向：reshape 成卷积输入形状再做 3D 卷积。

        参数：
            x: [L, C*T*P*P] 已展平的像素（processor 输出）。

        返回：
            [L, hidden_size] patch 嵌入。
        """
        L, C = x.shape
        # [L, C] -> [L, C, T, P, P]（把展平的块还原成卷积空间形状）。
        x = x.view(L, -1, self.temporal_patch_size, self.patch_size, self.patch_size)
        x = self.proj(x).view(L, self.hidden_size)
        return x


class Glm5NextVisionMLP(nn.Module):
    """视觉 MLP：SwiGLU（gate/up 融合 GEMM + 限幅激活）+ down 投影。

    GLM-5.3-Flash 特点：与 GLM-OCR/GLM-4V 不同，对视觉 SwiGLU 的
    gate/up 做限幅（SiluAndMulWithClamp，swiglu_limit 来自配置）。
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        swiglu_limit: float,
        bias: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        """初始化。

        参数：
            in_features: 输入维度。
            hidden_features: 中间维度。
            swiglu_limit: SwiGLU 限幅值。
            bias: 是否带偏置。
            quant_config: 量化配置。
            prefix: 层名前缀。
        """
        super().__init__()
        # ViT 数据并行时禁用 TP（每卡完整权重、按数据分片）。
        use_data_parallel = is_vit_use_data_parallel()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=in_features,
            output_sizes=[hidden_features] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            disable_tp=use_data_parallel,
        )
        self.down_proj = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            disable_tp=use_data_parallel,
        )
        # GLM-5.3-Flash clamps the vision SwiGLU gate/up unlike GLM-OCR/GLM-4V.
        self.act_fn = SiluAndMulWithClamp(swiglu_limit=swiglu_limit)

    def forward(self, x: torch.Tensor):
        """前向：gate_up -> 限幅 SwiGLU -> down。

        参数：
            x: [L, in_features]。

        返回：
            [L, in_features]。
        """
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class Glm5NextVisionAttention(nn.Module):
    """视觉注意力：QKV 投影 + q/k RMSNorm + RoPE + 编码器注意力。

    特点：
      - q/k 各自做 RMSNorm（eps=1e-5，与块内 norm 不同），
        用融合算子 fused_q_kv_rmsnorm 一次完成；
      - RoPE 拼接 q/k 后一次施加（cat + chunk 减少 launch）；
      - MMEncoderAttention 按 cu_seqlens 做变长批注意力
        （NPU 上走 Ascend 后端）。
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        """初始化。

        参数：
            embed_dim: 嵌入维度（也是 head_dim * num_heads）。
            num_heads: 注意力头数。
            projection_size: QKV 投影总维度。
            quant_config: 量化配置。
            prefix: 层名前缀。
        """
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel()
        # 数据并行模式下 tp_size 视为 1（不切分）。
        self.tp_size = 1 if use_data_parallel else get_tensor_model_parallel_world_size()
        self.tp_rank = 0 if use_data_parallel else parallel_state.get_tensor_model_parallel_rank()
        # divide：带整除校验的除法（vLLM 工具）。
        self.hidden_size_per_attention_head = dist_utils.divide(projection_size, num_heads)
        self.num_attention_heads_per_partition = dist_utils.divide(num_heads, self.tp_size)

        self.head_dim = embed_dim // num_heads

        # q/k norm eps hard-coded 1e-5 — distinct from block/post norm eps.
        # q/k 归一化 eps 硬编码 1e-5——与块内/后置 norm 的 eps 不同。
        self.q_norm = RMSNorm(self.head_dim, eps=1e-5)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-5)

        self.qkv = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.hidden_size_per_attention_head,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj" if quant_config else f"{prefix}.qkv",
            disable_tp=use_data_parallel,
        )
        self.proj = RowParallelLinear(
            input_size=projection_size,
            output_size=embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
            bias=True,
            disable_tp=use_data_parallel,
        )

        self.attn = MMEncoderAttention(
            num_heads=self.num_attention_heads_per_partition,
            head_size=self.hidden_size_per_attention_head,
            scale=self.hidden_size_per_attention_head**-0.5,
            prefix=f"{prefix}.attn",
        )
        self.apply_rotary_emb = ApplyRotaryEmb(enforce_enable=True)

    def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """把 QKV 投影输出按最后一维三等分并重排成 [S, B, H, D]。

        参数：
            qkv: [S, B, 3*projection]。

        返回：
            (q, k, v)，各 [S, B, local_heads, head_dim]。
        语法点：生成器表达式 `(x.view(...) for x in ...)` 一次解包三个。
        """
        seq_len, bs, _ = qkv.shape
        q, k, v = qkv.chunk(3, dim=2)
        new_shape = (
            seq_len,
            bs,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        q, k, v = (x.view(*new_shape) for x in (q, k, v))
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """视觉注意力前向。

        参数：
            x: [S, B, embed_dim] 输入。
            cu_seqlens: [n+1] 每个视觉序列的累积长度（变长批注意力）。
            rotary_pos_emb_cos/sin: RoPE 的 cos/sin（按 patch 位置索引好）。
            max_seqlen: 批内最长序列（部分后端需要）。

        返回：
            [S, B, embed_dim] 注意力输出。
        """
        # 步骤1: QKV 投影并重排。
        x, _ = self.qkv(x)
        q, k, v = self.split_qkv(x)

        # P1: fused q/k RMSNorm (two distinct weights, one launch; fp32, bit-identical).
        # 步骤2: 融合 q/k RMSNorm——两个不同权重一次内核启动，fp32 计算
        # 与分别做 RMSNorm 位级一致。
        q_shape, k_shape = q.shape, k.shape
        q_flat = q.reshape(-1, self.head_dim)
        k_flat = k.reshape(-1, self.head_dim)
        q, k = fused_q_kv_rmsnorm(
            q_flat,
            k_flat,
            self.q_norm.weight,
            self.k_norm.weight,
            self.q_norm.variance_epsilon,
        )
        q = q.view(q_shape)
        k = k.view(k_shape)

        # 步骤3: [S, B, ...] -> [B, S, ...]（注意力后端的批主序）。
        # rearrange 为 einops 的模式重排语法："s b ... -> b s ..."。
        q, k, v = (rearrange(t, "s b ... -> b s ...").contiguous() for t in (q, k, v))
        # 步骤4: RoPE——q/k 沿 dim0 拼接后一次施加再拆开（省一次 launch）。
        if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            qk_concat = torch.cat([q, k], dim=0)
            qk_rotated = self.apply_rotary_emb(
                qk_concat,
                rotary_pos_emb_cos,
                rotary_pos_emb_sin,
            )
            q, k = torch.chunk(qk_rotated, 2, dim=0)

        # 步骤5: 变长批注意力（MMEncoderAttention，按 cu_seqlens 分段）。
        context_layer = self.attn(
            query=q,
            key=k,
            value=v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        # 步骤6: [B, S, H, D] -> [S, B, H*D]（还原 seq 主序）+ 输出投影。
        context_layer = rearrange(context_layer, "b s h d -> s b (h d)").contiguous()

        output, _ = self.proj(context_layer)
        return output


class Glm5NextVisionBlock(nn.Module):
    """视觉 Transformer 块：norm -> 注意力 -> norm（融合残差）-> MLP。

    残差结构：x = residual + mlp(norm2(attn_out))，其中 norm2 采用
    "融合残差"模式（返回 (normed, residual)）。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        swiglu_limit: float,
        norm_layer: partial[nn.Module] | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        """初始化。

        参数：
            dim: 嵌入维度。
            num_heads: 头数。
            mlp_hidden_dim: MLP 中间维度。
            swiglu_limit: SwiGLU 限幅。
            norm_layer: 归一化层工厂（partial(RMSNorm, eps=...)）；
                None 时默认 LayerNorm(eps=1e-6)。
            quant_config: 量化配置。
            prefix: 层名前缀。
        语法点：partial[nn.Module] 类型注解——偏函数工厂的类型。
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Glm5NextVisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.mlp = Glm5NextVisionMLP(
            dim,
            mlp_hidden_dim,
            swiglu_limit=swiglu_limit,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        """块前向：注意力子层 + MLP 子层（均带残差）。

        参数：
            x: [S, B, dim]。
            其余同 Glm5NextVisionAttention.forward。

        返回：
            [S, B, dim]。
        """
        # 子层1: 注意力（先 norm1）。
        x_attn = self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            max_seqlen=max_seqlen,
        )
        # 子层2: norm2 的融合残差模式——一次内核完成 norm 与残差加。
        x_fused_norm, residual = self.norm2(x, residual=x_attn)
        x = residual + self.mlp(x_fused_norm)
        return x


class Glm5NextPatchMerger(nn.Module):
    """Patch 合并器：把 2x2 相邻 patch 的特征融合为单 token 表示。

    结构：proj（线性）-> LayerNorm -> GELU -> gate_up SwiGLU（限幅）
    -> down 投影。context_dim 是 GLM-5.3-Flash 特有的瓶颈宽度
    （projection_intermediate_size，如 10240）。
    """

    def __init__(
        self,
        d_model: int,
        context_dim: int,
        swiglu_limit: float,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        prefix: str = "",
    ) -> None:
        """初始化。

        参数：
            d_model: 输入/输出维度（= out_hidden_size）。
            context_dim: SwiGLU 瓶颈宽度。
            swiglu_limit: SwiGLU 限幅。
            quant_config: 量化配置。
            bias: 偏置开关。
            prefix: 层名前缀。
        """
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel()
        self.hidden_size = d_model
        self.proj = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=bias,
            gather_output=True,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
            disable_tp=use_data_parallel,
        )
        self.post_projection_norm = nn.LayerNorm(self.hidden_size)
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[context_dim] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            disable_tp=use_data_parallel,
        )
        self.down_proj = RowParallelLinear(
            context_dim,
            self.hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            disable_tp=use_data_parallel,
        )
        # GLM-5.3-Flash also clamps the merger SwiGLU.
        self.act_fn = SiluAndMulWithClamp(swiglu_limit=swiglu_limit)
        self.extra_activation_func = nn.GELU()

    def forward(self, x: torch.Tensor):
        """前向：proj -> LN+GELU -> 限幅 SwiGLU -> down。

        参数：
            x: [L, d_model]（已做 2x2 空间合并后的特征）。

        返回：
            [L, d_model] 合并后的视觉 token 特征。
        """
        x, _ = self.proj(x)
        x = self.extra_activation_func(self.post_projection_norm(x))
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Glm5NextVisionTransformer(nn.Module):
    """GLM-5.3-Flash 视觉塔（ViT 主干）：patch 嵌入 + 块堆叠 + 空间合并。

    继承 nn.Module；内部组合 PatchEmbed、depth 个 VisionBlock、
    Conv2d 下采样与 PatchMerger。提供 rot_pos_emb（2D mRoPE）、
    prepare_encoder_metadata（CUDA Graph 预计算）与 load_weights
    （含 GLM-OCR/GLM-4V 堆叠权重重映射）。
    """

    # Stacked-weight remap for the GLM-OCR/GLM-4V vision checkpoint layout.
    # checkpoint 的分离 q/k/v 与 gate/up 重映射到运行时的融合层。
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".attn.q.": (".attn.qkv.", "q"),
            ".attn.k.": (".attn.qkv.", "k"),
            ".attn.v.": (".attn.qkv.", "v"),
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        }
    )

    def __init__(
        self,
        text_config,  # noqa: ANN001
        vision_config,
        norm_eps: float = 1e-6,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        """初始化视觉塔。

        参数：
            text_config: 文本配置（仅用于读取 swiglu_limit 兜底）。
            vision_config: Glm5NextVisionConfig 视觉配置。
            norm_eps: 块内 RMSNorm eps。
            quant_config: 量化配置。
            prefix: 层名前缀。
        """
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel()
        self.tp_size = 1 if use_data_parallel else get_tensor_model_parallel_world_size()

        patch_size = vision_config.patch_size
        temporal_patch_size = vision_config.temporal_patch_size
        in_channels = vision_config.in_channels
        depth = vision_config.depth
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads

        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.out_hidden_size = vision_config.out_hidden_size

        # SwiGLU 限幅：优先视觉配置，兜底文本配置；必须存在。
        swiglu_limit = vision_config.swiglu_limit
        if swiglu_limit is None:
            swiglu_limit = text_config.swiglu_limit
        assert swiglu_limit is not None, "GLM-5.3-Flash vision requires swiglu_limit (vision_config or text_config)"

        # Single construction pass — no abs-pos embeddings / post-conv norm (OCR delta).
        # 步骤1: patch 嵌入（无绝对位置嵌入、无卷积后 norm——与 OCR 版的差异）。
        self.patch_embed = Glm5NextVisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            hidden_size=self.hidden_size,
        )

        norm_layer = partial(RMSNorm, eps=norm_eps)
        head_dim = self.hidden_size // self.num_heads
        # 步骤2: 2D mRoPE——partial_rotary_factor=0.5（一半维度加旋转），
        # neox 风格。
        self.rotary_pos_emb = get_rope(
            head_size=head_dim,
            max_position=8192,
            is_neox_style=True,
            rope_parameters={"partial_rotary_factor": 0.5},
        )
        # 步骤3: depth 个视觉块。
        self.blocks = nn.ModuleList(
            [
                Glm5NextVisionBlock(
                    dim=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_hidden_dim=vision_config.intermediate_size,
                    swiglu_limit=swiglu_limit,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{layer_idx}",
                )
                for layer_idx in range(depth)
            ]
        )
        # GLM-5.3-Flash merger bottleneck width.
        # 步骤4: PatchMerger（瓶颈宽度用视觉配置的
        # projection_intermediate_size）。
        self.merger = Glm5NextPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=vision_config.projection_intermediate_size,
            swiglu_limit=swiglu_limit,
            quant_config=quant_config,
            bias=False,
            prefix=f"{prefix}.merger",
        )

        # 步骤5: 2x2 空间合并的卷积下采样（hidden->out_hidden 通道）。
        self.downsample = Conv2dLayer(
            in_channels=vision_config.hidden_size,
            out_channels=vision_config.out_hidden_size,
            kernel_size=vision_config.spatial_merge_size,
            stride=vision_config.spatial_merge_size,
        )
        self.post_layernorm = RMSNorm(vision_config.hidden_size, eps=vision_config.rms_norm_eps)

        # 步骤6: 按头维与 dtype 选定编码器注意力后端（NPU 上选 ASCEND）。
        self.attn_backend = get_vit_attn_backend(
            head_size=head_dim,
            dtype=torch.get_default_dtype(),
        )

    @property
    def dtype(self) -> torch.dtype:
        """塔的参数 dtype（取 patch_embed 卷积权重的 dtype）。"""
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        """塔所在设备（取 patch_embed 卷积权重的设备）。"""
        return self.patch_embed.proj.weight.device

    def rot_pos_emb(self, grid_thw: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """按 (t, h, w) 网格计算 2D mRoPE 的 cos/sin。

        参数：
            grid_thw: 每个视觉序列的 [帧数, 高格数, 宽格数] 列表。

        返回：
            (cos_combined, sin_combined, pos_ids)：
            cos/sin 形状 [总 patch 数, head_dim]（h/w 两轴索引拼接）。

        原理：
          - 每个 patch 的位置 = (行号, 列号)；
          - 为配合 spatial_merge（2x2 合并），把 h/w 网格重排为
            [h//m, w//m, m, m] 再 permute——使 2x2 邻域在展平后相邻；
          - 视频按帧数 t 重复同一空间位置。
        """
        pos_ids_per_grid = []
        for t, h, w in grid_thw:
            # 行/列位置网格：hpos [h,w] 行号，wpos [h,w] 列号。
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            # 按 spatial_merge_size 重排：相邻 2x2 的 patch 在展平后相邻。
            hpos_ids = (
                hpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            wpos_ids = (
                wpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            # 堆叠 (h, w) 并按帧数 t 重复。
            pos_ids_per_grid.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids_per_grid, dim=0)
        max_grid_size = max(max(h, w) for _, h, w in grid_thw)

        # 取最大网格尺寸的 cos/sin 表，再按 pos_ids 索引 gather。
        cos, sin = self.rotary_pos_emb.get_cos_sin(max_grid_size)

        pos_ids = pos_ids.to(cos.device, non_blocking=True)
        cos_combined = cos[pos_ids].flatten(1)
        sin_combined = sin[pos_ids].flatten(1)
        return cos_combined, sin_combined, pos_ids

    def compute_attn_mask_seqlen(
        self,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor | None:
        """计算批内最长序列长度（仅部分后端需要）。

        参数：
            cu_seqlens: [n+1] 累积序列长度。

        返回：
            max_seqlen 张量；后端不需要时 None。
        """
        max_seqlen = None
        if self.attn_backend in {
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.ROCM_AITER_FA,
            AttentionBackendEnum.TRITON_ATTN,
        }:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        return max_seqlen

    def prepare_encoder_metadata(
        self,
        grid_thw_list: list[list[int]],
        *,
        max_batch_size: int | None = None,
        max_frames_per_batch: int | None = None,
        max_seqlen_override: int | None = None,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Compute encoder metadata for eager and CUDA graph execution.

        为视觉编码器（eager 与 CUDA Graph 两种执行模式）预计算元数据。

        参数（语法点：* 之后 keyword-only）：
            grid_thw_list: 各视觉序列的 (t, h, w) 网格。
            max_batch_size: 图捕获的批大小上限（padding 目标）。
            max_frames_per_batch: 视频帧批上限（优先于 batch_size）。
            max_seqlen_override: 覆盖 max_seqlen（图捕获固定形状用）。
            device: 目标设备。

        返回：
            dict：rotary_pos_emb_cos/sin、sequence_lengths、max_seqlen、
            cu_seqlens——键名与 forward 的 encoder_metadata 一致。
        """
        if device is None:
            device = self.device

        metadata: dict[str, torch.Tensor | None] = {}

        # 步骤1: RoPE cos/sin（与 eager 路径完全一致的计算）。
        rotary_cos, rotary_sin, _ = self.rot_pos_emb(grid_thw_list)
        metadata["rotary_pos_emb_cos"] = rotary_cos
        metadata["rotary_pos_emb_sin"] = rotary_sin

        # 步骤2: cu_seqlens（numpy 累积和）：每帧的 patch 数 =
        # h*w，按帧数展开后 cumsum，首部补 0。
        grid_thw_np = np.array(grid_thw_list, dtype=np.int32)
        patches_per_frame = grid_thw_np[:, 1] * grid_thw_np[:, 2]
        cu_seqlens = np.repeat(patches_per_frame, grid_thw_np[:, 0]).cumsum(dtype=np.int32)
        cu_seqlens = np.concatenate([np.zeros(1, dtype=np.int32), cu_seqlens])

        # 步骤3: 图捕获 padding——序列数不足 pad_to 时用末值补齐
        # （空序列，注意力天然跳过）。
        pad_to = max_frames_per_batch if max_frames_per_batch is not None else max_batch_size
        if pad_to is not None:
            num_seqs = len(cu_seqlens) - 1
            if num_seqs < pad_to:
                cu_seqlens = np.concatenate(
                    [
                        cu_seqlens,
                        np.full(
                            pad_to - num_seqs,
                            cu_seqlens[-1],
                            dtype=np.int32,
                        ),
                    ]
                )

        # 步骤4: 各后端所需的序列长度张量与 cu_seqlens（按后端重算格式）。
        metadata["sequence_lengths"] = MMEncoderAttention.maybe_compute_seq_lens(self.attn_backend, cu_seqlens, device)

        if max_seqlen_override is not None:
            max_seqlen_val = max_seqlen_override
        else:
            max_seqlen_val = MMEncoderAttention.compute_max_seqlen(self.attn_backend, cu_seqlens)
        metadata["max_seqlen"] = torch.tensor(max_seqlen_val, dtype=torch.int32)

        metadata["cu_seqlens"] = MMEncoderAttention.maybe_recompute_cu_seqlens(
            self.attn_backend,
            cu_seqlens,
            self.hidden_size,
            self.tp_size,
            device,
        )

        return metadata

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor | list[list[int]],
        *,
        encoder_metadata: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """视觉塔前向：patch 化 -> Transformer 块 -> 空间合并 + merger。

        参数：
            x: [L, C*T*P*P] 展平像素。
            grid_thw: [n, 3] 或 list 的 (t, h, w) 网格。
            encoder_metadata: 预计算元数据（CUDA Graph 路径）；
                None 时现场重算（eager 路径）。

        返回：
            [num_vision_tokens, out_hidden_size] 视觉 token 特征。
        """
        # patchify
        # 步骤1: 转设备/dtype 并做 patch 嵌入。
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)

        if encoder_metadata is not None:
            # Encoder CUDA-graph path (PR #49852): rotary/cu_seqlens/max_seqlen are
            # precomputed by prepare_encoder_metadata (which uses rot_pos_emb exactly
            # as the eager rebuild does), so reuse them and skip the per-call CPU
            # rebuild (the low-GPU-util culprit on multimodal workloads).
            # 编码器 CUDA Graph 路径：rotary/cu_seqlens/max_seqlen 由
            # prepare_encoder_metadata 预计算（与 eager 重建完全一致），
            # 直接复用，跳过每次调用的 CPU 重建（多模态负载 GPU
            # 利用率低的罪魁祸首）。
            rotary_pos_emb_cos = encoder_metadata["rotary_pos_emb_cos"]
            rotary_pos_emb_sin = encoder_metadata["rotary_pos_emb_sin"]
            cu_seqlens = encoder_metadata["cu_seqlens"]
            max_seqlen = encoder_metadata["max_seqlen"]
        else:
            # eager 路径：现场计算 RoPE 与 cu_seqlens。
            if isinstance(grid_thw, list):
                grid_thw = torch.tensor(grid_thw, dtype=torch.int32)
            rotary_pos_emb_cos, rotary_pos_emb_sin, _ = self.rot_pos_emb(grid_thw)
            cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
                dim=0, dtype=torch.int32
            )
            cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])
            cu_seqlens = cu_seqlens.to(self.device, non_blocking=True)
            max_seqlen = self.compute_attn_mask_seqlen(cu_seqlens)

        # transformers
        # 步骤2: 堆叠视觉块（unsqueeze(1) 给出批维 B=1）。
        x = x.unsqueeze(1)
        for blk in self.blocks:
            x = blk(
                x,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                max_seqlen=max_seqlen,
            )

        # adapter
        # 步骤3: 后处理——RMSNorm -> 重排出 2x2 空间邻域 -> 卷积下采样
        # -> PatchMerger。
        x = self.post_layernorm(x)
        x = x.view(-1, self.spatial_merge_size, self.spatial_merge_size, x.shape[-1])
        x = x.permute(0, 3, 1, 2)
        x = self.downsample(x).view(-1, self.out_hidden_size)
        x = self.merger(x)
        return x

    def load_weights(self, weights) -> set[str]:
        """权重加载：AutoWeightsLoader + GLM-OCR/GLM-4V 堆叠权重映射。"""
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class Glm5NextProcessingInfo(Glm4vProcessingInfo):
    """Wires up the vLLM-native processor for the multimodal checkpoint.

    The checkpoint's ``processor_config.json`` declares a custom ``processor_class``
    and stores its image/video processor configs inline (no standalone
    ``preprocessor_config.json``), so ``AutoProcessor`` cannot resolve the
    config. We bypass it and build our own ``Glm5NextProcessor``
    (``vllm/transformers_utils/processors/glm5next.py``), a port of the
    training-side pipeline that no longer imports transformers' GLM processor
    classes. The port applies ``patch_expand_factor`` (checkpoint ships 1)
    inside ``smart_resize``'s spatial factor.

    为多模态 checkpoint 接线 vLLM 原生处理器（ProcessingInfo）。

    继承 Glm4vProcessingInfo（GLM-4V 的处理信息类，提供
    _get_vision_info 等几何推导接口）。本类把 get_hf_processor 换成
    vLLM 原生的 Glm5NextProcessor（processor.py）。
    """

    @cached_property
    def _glm5_next_hf_processor(self):
        """延迟构建并缓存的 Glm5NextProcessor 实例。

        语法点：@cached_property——首次访问时计算并缓存为实例属性。
        """
        from vllm_ascend.models.glm5next.processor import Glm5NextProcessor

        return Glm5NextProcessor.from_pretrained(self.ctx.model_config.model)

    def get_hf_processor(self, **kwargs: object):
        """返回处理器实例（覆盖父类，供 vLLM 多模态管道调用）。"""
        return self._glm5_next_hf_processor

    def _processor_pixel_budget(self, proc) -> tuple[int, int]:
        """从处理器的 token 预算推导 (min_pixels, max_pixels)。"""
        from vllm_ascend.models.glm5next.processor import _pixel_budget

        return _pixel_budget(
            proc.min_image_tokens,
            proc.max_image_tokens,
            proc.patch_size,
            proc.merge_size,
            proc.temporal_patch_size,
        )

    def _get_image_max_pixels(self) -> int:
        """图像最大像素：mm_kwargs 的 max_pixels 覆盖或处理器预算上限。

        语法点：海象运算符 := 在 if 中赋值并判断。
        """
        mm_kwargs = self.ctx.get_merged_mm_kwargs({})
        if (override := mm_kwargs.get("max_pixels")) is not None:
            return int(override)
        return self._processor_pixel_budget(self.get_hf_processor().image_processor)[1]

    def _get_video_max_pixels(self) -> int:
        """视频最大像素：同图像逻辑（视频处理器预算上限）。"""
        mm_kwargs = self.ctx.get_merged_mm_kwargs({})
        if (override := mm_kwargs.get("max_pixels")) is not None:
            return int(override)
        return self._processor_pixel_budget(self.get_hf_processor().video_processor)[1]

    def _get_vision_info(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int = 16,
        do_resize: bool = True,
        max_image_pixels: int = 28 * 28 * 2 * 30000,
    ) -> tuple[ImageSize, int]:
        """GLM-5.3-Flash canvas geometry for token budgeting and dummy inputs.

        GLM-5.3-Flash 的画布几何（用于 token 预算估算与 dummy 输入）。

        The inherited Glm4v path resolves the pixel budget from
        ``size.longest_edge`` and resizes with GLM-4V's ``smart_resize``. This
        checkpoint's ``processor_config.json`` ships the token-budget style
        (``min_image_tokens`` / ``max_image_tokens``) with no ``size`` key, and
        the alignment factor carries ``patch_expand_factor`` — resolve both
        from the vLLM-native processor so profiling matches runtime geometry.

        参数（语法点：* 之后 keyword-only）：
            image_width/height: 原始图像尺寸。
            num_frames: 帧数（图像默认 16，按 temporal_patch 取整）。
            do_resize: 是否做 smart_resize（False 用于已预处理输入）。
            max_image_pixels: 像素预算上限。

        返回：
            (preprocessed_size, num_vision_tokens)：
            预处理后的图像尺寸与视觉 token 数。
        """
        from vllm_ascend.models.glm5next.processor import smart_resize

        vision_config = self.get_hf_config().vision_config
        patch_size = vision_config.patch_size
        merge_size = vision_config.spatial_merge_size
        temporal_patch_size = vision_config.temporal_patch_size

        image_processor = self.get_hf_processor().image_processor
        # 对齐因子 = patch * merge * patch_expand（GLM 特有扩展因子）。
        factor = patch_size * merge_size * image_processor.patch_expand_factor
        # Keep the profiling search viable when the caller's budget is below
        # one aligned canvas of the requested duration.
        # 调用方预算低于一个对齐画布时抬高预算，保证 profiling 搜索可行。
        max_image_pixels = max(max_image_pixels, temporal_patch_size * factor * factor)

        if do_resize:
            # 帧数至少 temporal_patch_size（不足则取整）。
            t = num_frames if num_frames > temporal_patch_size else temporal_patch_size
            resized_height, resized_width = smart_resize(
                t=t,
                h=image_height,
                w=image_width,
                t_factor=temporal_patch_size,
                h_factor=factor,
                w_factor=factor,
                min_pixels=1,
                max_pixels=max_image_pixels,
            )
            preprocessed_size = ImageSize(width=resized_width, height=resized_height)
        else:
            preprocessed_size = ImageSize(width=image_width, height=image_height)

        # 网格与 token 数：帧上取整到 temporal_patch 的倍数，
        # patch 数 = t_grid * h_grid * w_grid，token 数再除以 merge^2。
        padded_num_frames = num_frames + (-num_frames % temporal_patch_size)
        grid_t = max(padded_num_frames // temporal_patch_size, 1)
        grid_h = preprocessed_size.height // patch_size
        grid_w = preprocessed_size.width // patch_size

        num_patches = grid_t * grid_h * grid_w
        num_vision_tokens = num_patches // (merge_size**2)

        return preprocessed_size, num_vision_tokens


class Glm5NextMultiModalProcessor(Glm4vMultiModalProcessor):
    """The vLLM-native ``Glm5NextProcessor`` extracts image/video features
    only and passes the prompt text through unchanged, so prompt expansion
    (image token repeat, video frame/timestamp structure) is owned by vLLM's
    prompt-update machinery — the inherited ``_get_prompt_updates`` builds
    the replacement content and the placeholder scan validates against
    exactly that.

    多模态处理器：prompt 展开由 vLLM 的 prompt-update 机制负责。

    原理：vLLM 原生 Glm5NextProcessor 只提取图像/视频特征、原文透传
    prompt 文本，因此"图像 token 重复、视频帧/时间戳结构"等 prompt 展开
    由 vLLM 的 prompt-update 机制完成——继承的 _get_prompt_updates 构建
    替换内容，占位符扫描也按其校验。
    """

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        """声明 HF 处理器不自行修改 prompt（展开交给 vLLM）。

        参数：
            prompt_text: 原始 prompt。
            mm_items: 多模态数据项。
            hf_processor_mm_kwargs: 处理器参数。
            tokenization_kwargs: 分词参数。

        返回：
            bool: 恒 False——Glm5NextProcessor 不做 prompt 更新。
        """
        return False
