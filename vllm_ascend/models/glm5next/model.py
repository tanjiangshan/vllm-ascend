# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# GLM-5.Next 主模型实现（昇腾 NPU 适配版）。
#
# 模型总体结构（Glm5NextForConditionalGeneration = 视觉塔 + 文本 LM）：
#   Glm5NextForConditionalGeneration (继承 Glm4vForConditionalGeneration)
#     ├── visual: Glm5NextVisionTransformer     # ViT 视觉塔（multimodal.py）
#     └── language_model: Glm5NextForCausalLM   # 文本语言模型
#           └── model: Glm5NextModel
#                 ├── embed_tokens              # 词嵌入（PP 首卡）
#                 ├── layers[i]: Glm5NextDecoderLayer
#                 │     ├── self_attn: Glm5NextLinearAttention (KDA 层)
#                 │     │            或 Glm5NextMLAAttention (MLA/稀疏层)
#                 │     ├── mlp: Glm5NextMoE 或 Glm5NextMLP
#                 │     └── mHC 超连接参数（hc_attn_fn/hc_ffn_fn 等）
#                 └── norm                       # 最终 RMSNorm（PP 末卡）
#
# 混合架构（IsHybrid）：
#   - layer_types 指定每层注意力类型："linear_attention"（KDA，常数复杂度、
#     循环状态）与 "deepseek_sparse_attention"（稀疏 MLA + KPool 索引器）
#     交错排布；
#   - mlp_layer_types 指定每层 MLP："dense" 或 "sparse"（MoE，默认 288 专家）。
#
# MHC（multi-head hyper-Connection，超连接）残差流：
#   - 把单条残差流扩展为 n 条（mhc_num_residual_streams=4），
#     层间用可学习的混合矩阵（含 sinkhorn 归一化）做信息交换，
#     经 hc_expand/hc_contract 进出；计算由 NPU 自定义算子
#     npu_hc_pre_v2 / npu_hc_post 完成。
#
# MTP 投机解码：num_nextn_predict_layers 个额外层（mtp.py 中
# Glm5NextMTP 复用本文件的 Glm5NextDecoderLayer）。
#
# 权重加载：load_weights 支持堆叠权重重映射（gate/up 融合、qkv 融合、
# KDA 六投影融合、索引器 wk+weights_proj 融合）与 FP8 索引器 WK 反量化。
# =============================================================================

from collections.abc import Iterable
from typing import Any, ClassVar, Literal

import torch
import torch_npu
from torch import nn
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    GateLinear,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    scaled_dequantize,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.deepseek_v2 import _get_moe_router_dtype
from vllm.model_executor.models.glm4_1v import (
    Glm4vDummyInputsBuilder,
    Glm4vForConditionalGeneration,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    init_vllm_registered_model,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_reduce_scatter,
    sp_shard,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from .attention import Glm5NextMLAAttention
from .config import Glm5NextConfig
from .kda import Glm5NextLinearAttention
from .multimodal import (
    Glm5NextMultiModalProcessor,
    Glm5NextProcessingInfo,
    Glm5NextVisionTransformer,
)
from .ops.mhc_ops import hc_contract, hc_expand


class Glm5NextMLP(nn.Module):
    """GLM-5.Next 稠密 MLP（SwiGLU 门控前馈，可选 clamp 限幅）。

    结构：gate_up_proj（融合 GEMM 出 gate|up 两半）-> SiluAndMul 激活
    -> down_proj。支持序列并行（SP：输入输出按 TP 组分片，权重复制，
    无需集合通信）与普通 TP（尾部 all-reduce）。
    swiglu_limit 非空时用 SiluAndMulWithClamp 对门控值限幅（GLM 特有）。
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel=False,
        prefix: str = "",
        swiglu_limit: float | None = None,
    ) -> None:
        """初始化稠密 MLP。

        参数：
            hidden_size: 隐藏维度 H。
            intermediate_size: 中间维度 I。
            hidden_act: 激活名（仅支持 "silu"）。
            quant_config: 量化配置。
            reduce_results: down_proj 是否做 TP all-reduce 输出。
            is_sequence_parallel: 序列并行模式（权重复制、按序列分片）。
            prefix: 层名前缀。
            swiglu_limit: SwiGLU 限幅值；None 表示不限幅。
        """
        super().__init__()

        # If is_sequence_parallel, the input and output tensors are sharded
        # across the ranks within the tp_group. In this case the weights are
        # replicated and no collective ops are needed.
        # Otherwise we use standard TP with an allreduce at the end.
        # 序列并行：输入输出按 tp_group 内各 rank 分片，权重复制、无集合通信；
        # 否则用标准 TP（down_proj 尾部 all-reduce）。
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. Only silu is supported for now.")

        self.swiglu_limit = swiglu_limit
        if self.swiglu_limit is not None:
            self.act_fn = SiluAndMulWithClamp(swiglu_limit=self.swiglu_limit)
        else:
            self.act_fn = SiluAndMul()

    def forward(self, x):
        """前向：gate_up GEMM -> SwiGLU（可选限幅）-> down GEMM。

        参数：
            x: [num_tokens, hidden_size]（SP 模式下为分片后的 [N/tp, H]）。

        返回：
            [num_tokens, hidden_size]。
        """
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Glm5NextMoE(nn.Module):
    """GLM-5.Next 的 MoE（混合专家）层。

    结构：gate 路由（sigmoid 评分 + 分组 top-k）+ n_shared_experts 个共享
    专家（Glm5NextMLP 实现）+ n_routed_experts 个路由专家（FusedMoEFactory
    构建的融合 MoE）。支持 EPLB（专家并行负载均衡冗余专家）、序列并行
    MoE 与 Ascend 内部路由模式（is_internal_router：gate 在 MoE runner
    内部计算，避免重复路由）。
    """

    def __init__(
        self,
        config: Glm5NextConfig,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        apply_routed_scale_to_output: bool = False,
    ):
        """初始化 MoE 层。

        参数：
            config: Glm5NextConfig（专家数/路由参数等）。
            parallel_config: 并行配置（EP/TP/EPLB/序列并行）。
            quant_config: 量化配置。
            prefix: 层名前缀。
            apply_routed_scale_to_output: 路由缩放因子作用在输出还是权重。
        """
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        self.routed_scaling_factor = config.routed_scaling_factor

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = config.n_shared_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {config.hidden_act}. Only silu is supported for now.")

        self.router_dtype = _get_moe_router_dtype(config)
        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            out_dtype=self.router_dtype,
            prefix=f"{prefix}.gate",
        )
        if config.topk_method == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(torch.empty(config.n_routed_experts, dtype=torch.float32))
        else:
            self.gate.e_score_correction_bias = None

        # Load balancing settings.
        # EPLB（专家并行负载均衡）设置：逻辑专家 + 冗余专家 = 物理专家，
        # 每个 EP rank 持有一段连续的物理专家区间。
        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = self.physical_expert_start + self.n_local_physical_experts

        # 共享专家：n_shared_experts 个专家并联成一个宽 MLP
        # （intermediate = moe_intermediate * n_shared），reduce_results=False
        # 让最终归约由融合 MoE 路径统一处理。
        swiglu_limit = config.swiglu_limit
        if config.n_shared_experts is None:
            self.shared_experts = None
        else:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts

            self.shared_experts = Glm5NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                is_sequence_parallel=self.is_sequence_parallel,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
                swiglu_limit=swiglu_limit,
            )

        # 路由专家：FusedMoEFactory 构建（融合路由 top-k + 分组选择 +
        # 专家 GEMM）。sigmoid 评分、renormalize、routed_scaling_factor、
        # EPLB 冗余专家、序列并行等全部透传。
        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_token,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.moe_renormalize,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func=config.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scale_to_output=apply_routed_scale_to_output,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            n_shared_experts=None,
            router_logits_dtype=self.gate.out_dtype,
            swiglu_limit=swiglu_limit,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        """MoE 前向：路由 + 共享专家 + 路由专家。

        参数：
            hidden_states: [num_tokens, hidden_size]。
            already_sequence_parallel: 调用方已做过序列分片时为 True
                （mHC 路径在层内已完成 sp 分片，避免重复切分）。

        返回：
            [num_tokens, hidden_size]（SP 模式下内部 all-gather 还原）。
        """
        num_tokens, hidden_dim = hidden_states.shape

        # Chunk the hidden states so they aren't replicated across TP ranks.
        # This avoids duplicate computation in self.experts.
        # 步骤1: 序列并行模式下先按序列分片，避免 hidden 在 TP 各卡重复，
        # 省去专家里的重复计算。
        if self.is_sequence_parallel and not already_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        # 步骤2: 路由 + 专家计算。
        if self.experts.is_internal_router:
            # The Ascend MoE runner owns the gate in this mode. Pass hidden
            # states through the router_logits slot so it can compute routing
            # exactly once inside the fused path.
            # Ascend 内部路由模式：MoE runner 拥有 gate。把 hidden 塞进
            # router_logits 槽位，让融合路径内恰好计算一次路由。
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=hidden_states,
            )
        else:
            router_logits, _ = self.gate(hidden_states)
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )

        # 步骤3: 序列并行模式下 all-gather 还原完整序列并裁回原 token 数。
        if self.is_sequence_parallel and not already_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(final_hidden_states, 0)
            final_hidden_states = final_hidden_states[:num_tokens]

        return final_hidden_states.view(num_tokens, hidden_dim)


class Glm5NextDecoderLayer(nn.Module):
    """GLM-5.Next 解码层：注意力（KDA 或 MLA）+ MLP/MoE + 可选 mHC 残差流。

    三种配置形态：
      1. mHC 层（标准 GLM-5.3-Flash 中间层）：n 条超连接残差流，
         层间通过 hc_pre/hc_post（NPU 算子）做混合；
      2. 非 mHC 层（70B 变体或 MTP 层）：常规单残差流
         input_layernorm -> attn -> post_attention_layernorm -> mlp；
      3. MTP 层（is_mtp_layer=True）：返回"未求和"的 (hidden, residual)
         对，供 shared_head 的 fused_add_rms_norm 单内核完成残差加+归一化。

    forward 返回四元组 (hidden_states, residual, post, comb)：
    post/comb 携带"延迟到下一层"的 mHC 状态（本层 ffn 前的 hc_post 输入），
    使相邻层的 hc_post 与下一层 hc_pre 融合成一次 hc_post_pre 调用。
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Glm5NextConfig,
        layer_idx: int,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        is_mtp_layer: bool = False,
        **kwargs,
    ) -> None:
        """初始化解码层。

        参数：
            vllm_config: vLLM 全局配置。
            config: Glm5NextConfig。
            layer_idx: 层下标（决定 KDA 还是 MLA、dense 还是 MoE）。
            prefix: 层名前缀。
            topk_indices_buffer: 稀疏索引共享缓冲（仅 MLA 层用）。
            is_mtp_layer: 是否 MTP 投机解码层（走非 mHC 路径）。
            **kwargs: 兼容扩展参数（吸收多余实参）。
        """
        super().__init__()

        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.is_moe = config.is_moe
        self.num_hidden_layers = config.num_hidden_layers
        self.rms_norm_eps = config.rms_norm_eps
        self.num_experts = config.n_routed_experts
        self.is_mtp_layer = is_mtp_layer
        self.mhc = config.mhc
        # 层类型判定：layer_types 标记为 linear_attention 则是 KDA 层。
        self.layer_kind = "kda" if config.is_kda_layer(layer_idx) else "mla"
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        # 步骤1: 按层类型构建注意力子层。
        if config.is_kda_layer(layer_idx):
            # KDA 线性注意力层（循环状态，无 KV cache）。
            self.self_attn = Glm5NextLinearAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            # MLA layers require the latent head dims, which are guaranteed set
            # on MLA configs; narrow away the `int | None`.
            # MLA 层：需要潜在头维度（MLA 配置下必定非空；断言收窄类型）。
            assert config.v_head_dim is not None
            assert config.kv_lora_rank is not None
            # 稀疏/稠密 MLA 注意力层；skip_rope=mla_nope（GLM 特有：
            # 部分配置完全不用 RoPE）。
            self.self_attn = Glm5NextMLAAttention(
                vllm_config=vllm_config,
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                max_position_embeddings=config.max_position_embeddings,
                cache_config=cache_config,
                quant_config=quant_config,  # keep MLA projections quantized when checkpoint weights are quantized
                prefix=f"{prefix}.self_attn",
                topk_indices_buffer=topk_indices_buffer,
                skip_rope=config.mla_nope,
            )

        # MTP layers sit past the base model's hidden layers (layer_idx >=
        # num_hidden_layers), so they're outside mlp_layer_types; default them
        # to the last base layer's MLP type (sparse/MoE for these checkpoints).
        # 步骤2: 按 mlp_layer_types 构建 MLP 子层。MTP 层下标越过主模型
        # 层数、不在 mlp_layer_types 里，默认沿用最后一层的类型
        # （这些 checkpoint 为 sparse/MoE）。
        mlp_layer_types = config.mlp_layer_types
        mlp_type = (
            mlp_layer_types[layer_idx]
            if layer_idx < len(mlp_layer_types)
            else (mlp_layer_types[-1] if mlp_layer_types else "sparse")
        )
        if self.is_moe and self.num_experts is not None and mlp_type == "sparse":
            self.mlp = Glm5NextMoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Glm5NextMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                is_sequence_parallel=self.is_sequence_parallel,
                reduce_results=not self.is_sequence_parallel,
                prefix=f"{prefix}.mlp",
                swiglu_limit=config.swiglu_limit,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Cached for the hot forward path (isinstance per layer per step).
        # 热路径缓存：避免每层每步做 isinstance 判断。
        self._mlp_is_moe = isinstance(self.mlp, Glm5NextMoE)
        # In SP, the attention output projection leaves a partial sum; the
        # decoder-layer reduce_scatter after attention completes it (DSv4 pattern).
        # MTP layers use the non-mHC path which has no sp_reduce_scatter, so
        # their o_proj must still reduce normally.
        # 序列并行下注意力输出投影只留部分和，层内 reduce_scatter 补全
        # （DeepSeek-V4 模式）；MTP 层走非 mHC 路径（无 sp_reduce_scatter），
        # 其 o_proj 必须正常 reduce。
        if self.is_sequence_parallel and not is_mtp_layer:
            self.self_attn.o_proj.reduce_results = False
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 步骤3: mHC（超连接）参数——n 条残差流的混合矩阵。
        # mix_hc = (2+n)*n 是 hc_pre_v2 内核的混合输出宽度
        # （n 条 post 混合 + n 条 res 混合 + 2 条组合系数）。
        # attn 侧与 ffn 侧各一组 (fn: [mix_hc, n*H], base: [mix_hc], scale: [3])。
        if self.mhc and not is_mtp_layer:
            # mhc config
            self.mhc_num_residual_streams = config.mhc_num_residual_streams
            self.mhc_tau = config.mhc_tau
            self.hc_eps = config.hc_eps
            self.mhc_sinkhorn_iterations = config.mhc_sinkhorn_iterations
            self.mhc_post_mult_value = config.mhc_post_mult_value

            n = config.mhc_num_residual_streams
            d_model = n * self.hidden_size
            mix_hc = (2 + n) * n

            self.n = n

            # attn hc
            self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, d_model, dtype=torch.float32))
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

            # ffn hc
            self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, d_model, dtype=torch.float32))
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        post: torch.Tensor | None = None,
        comb: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """解码层前向（mHC 与非 mHC 两条路径）。

        参数：
            positions: [num_tokens] token 位置。
            hidden_states: [num_tokens, hidden_size]（SP 分片时 [N/tp, H]）。
            residual: 残差流；mHC 时为 [N, n, H]（n 条流），否则 [N, H]。
            post: 上一层延迟的 hc_post post_mix 输入（[N, n, 1] FP32）。
            comb: 上一层延迟的 hc_post comb 输入（[N, n, n] FP32）。

        返回：
            (hidden_states, residual, post, comb)：
            非 mHC/最后层/ MTP 层后两项为 None；否则把本层 hc_post 状态
            延迟到下一层（与下一层 hc_pre 融合执行）。
        """
        # 70B or MTP layers: KDA + MoE without HC.
        # 路径 A：非 mHC（70B 变体或 MTP 层）——常规 pre-norm 残差。
        if not self.mhc or self.is_mtp_layer:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            attn_output = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
            )
            hidden_states, residual = self.post_attention_layernorm(attn_output, residual=residual)
            hidden_states = self.mlp(hidden_states)
            if self.is_mtp_layer:
                # Return the unsummed pair: the MTP caller feeds it straight
                # into shared_head's fused_add_rms_norm (one kernel instead of
                # a separate residual-add + norm). The sum itself is unchanged
                # (fp32-accumulated inside the fused kernel).
                # MTP 层返回"未求和"的 (mlp 输出, 残差) 二元组：调用方直接
                # 送进 shared_head 的 fused_add_rms_norm（单内核完成残差加
                # +归一化）。数值与先加后归一化一致（融合内核内 fp32 累加）。
                return hidden_states, residual, None, None
            hidden_states = residual + hidden_states
            return hidden_states, residual, None, None

        # mHC start. `post`/`comb` carry the previous layer's deferred
        # hc_post inputs (its ffn-pre outputs); apply the existing HcPost and
        # HcPre kernels before attention. Layer 0 has no incoming state.
        # 路径 B：mHC 超连接。post/comb 携带上一层延迟的 hc_post 输入
        # （该层 ffn 前的输出）；先做 hc_post（配上一层的 post/comb）与
        # hc_pre 再进注意力。第 0 层无输入状态，需先 expand 成 n 条流。
        x = hidden_states
        if post is None:
            if self.layer_idx == 0:
                # 第 0 层：把 [N, H] 复制扩展为 [N, n, H]。
                x = hc_expand(x, self.n)
            residual = x
            # hc_pre：对 n 条流做 sinkhorn 归一化的混合，产出
            # (post_mix, res_mix, layer_input)；layer_input 再过 RMSNorm。
            post, comb, x = self.hc_pre(
                x,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                norm_weight=self.input_layernorm.weight.data,
                norm_eps=self.input_layernorm.variance_epsilon,
            )
        else:
            residual, post, comb, x = self.hc_post_pre(
                x,
                residual,
                post,
                comb,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                norm_weight=self.input_layernorm.weight.data,
                norm_eps=self.input_layernorm.variance_epsilon,
            )

        # Attention needs the full token sequence; mHC above ran on the SP
        # shard. Gather for attention, scatter back afterward (DSv4 pattern).
        # 注意力需要完整 token 序列；上面的 mHC 在 SP 分片上运行。
        # 进注意力前 all-gather，出注意力后 reduce-scatter（DSv4 模式）。
        if self.is_sequence_parallel:
            x = sp_all_gather(x)[: positions.shape[0]]

        # 注意力计算（KDA 或 MLA，内部走 Ascend 算子）。
        x = self.self_attn(
            hidden_states=x,
            positions=positions,
        )

        if self.is_sequence_parallel:
            x = sp_reduce_scatter(x)

        # Apply post-attention mixing, pre-FFN mixing, then input RMSNorm.
        # 注意力后：hc_post（消费上一段的 post/comb）+ hc_pre（ffn 侧混合）
        # + post_attention RMSNorm，一次 hc_post_pre 融合调用。
        residual, post, comb, x = self.hc_post_pre(
            x,
            residual,
            post,
            comb,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            norm_weight=self.post_attention_layernorm.weight.data,
            norm_eps=self.post_attention_layernorm.variance_epsilon,
        )

        # Fully Connected
        # MLP/MoE：MoE 走 already_sequence_parallel（本层已 sp 分片）。
        if self._mlp_is_moe:
            x = self.mlp(x, already_sequence_parallel=self.is_sequence_parallel)
        else:
            x = self.mlp(x)

        # mHC end. The last mHC layer materializes its final hc_post (nothing
        # to fuse with) then contracts; every other layer defers its hc_post to
        # the next layer's pre, returning the state.
        # mHC 收尾：最后一层显式执行 hc_post（无下一层可融合）并收缩回
        # 单流 [N, H]；其余层把 hc_post 状态延迟给下一层并返回。
        if self.layer_idx == self.num_hidden_layers - 1:
            x = self.hc_post(x, residual, post, comb)
            x = hc_contract(x, self.n)
            return x, None, None, None

        return x, residual, post, comb

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ):
        """mHC 预混合（NPU 算子 npu_hc_pre_v2）+ 可选 RMSNorm。

        参数：
            x: [N, n, H] 输入流（或收缩形态，由内核解释）。
            hc_fn: [mix_hc, n*H] 可学习混合函数矩阵。
            hc_scale: [3] 缩放系数。
            hc_base: [mix_hc] 偏置。
            norm_weight: RMSNorm 权重（None 跳过归一化）。
            norm_eps: RMSNorm epsilon。

        返回：
            (post_mix, res_mix, layer_input)：
            post_mix [N, n, 1] 与 res_mix 供下一层 hc_post 用；
            layer_input 为进注意力/FFN 的归一化输入。
        """
        layer_input, post_mix, res_mix = torch.ops._C_ascend.npu_hc_pre_v2(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            self.n,
            self.mhc_sinkhorn_iterations,
            self.rms_norm_eps,
            self.hc_eps,
        )
        # HcPre uses 2 * sigmoid for post mixing; retain the model's scale.
        # 内核固定用 2*sigmoid 做后混合；若模型配置了不同的
        # mhc_post_mult_value 则在此补偿缩放。
        if self.mhc_post_mult_value != 2.0:
            post_mix = post_mix * (self.mhc_post_mult_value / 2.0)
        if norm_weight is not None:
            # torch_npu 的 npu_rms_norm：NPU 融合 RMSNorm 算子。
            layer_input = torch_npu.npu_rms_norm(layer_input, norm_weight, epsilon=norm_eps)[0]
        return post_mix.unsqueeze(-1), res_mix, layer_input

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):
        """mHC 后混合（NPU 算子 npu_hc_post）：把子层输出合回残差流。

        参数：
            x: 子层（注意力或 FFN）输出 [N, n, H]。
            residual: 当前残差流 [N, n, H]。
            post: hc_pre 产出的 post_mix。
            comb: hc_pre 产出的 res_mix/组合系数。

        返回：
            新残差流 [N, n, H]。
        （unsqueeze(0)/squeeze(0) 适配内核的批维要求。）
        """
        return torch.ops._C_ascend.npu_hc_post(
            x.unsqueeze(0), residual.unsqueeze(0), post.squeeze(-1).unsqueeze(0), comb.unsqueeze(0)
        ).squeeze(0)

    def hc_post_pre(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ):
        """hc_post 与 hc_pre 的融合调用（上一段的收尾 + 下一段的预备）。

        参数：同 hc_post + hc_pre。

        返回：
            (residual, post, comb, layer_input)：
            新残差流、延迟给下一段的 post/comb、归一化后的子层输入。
        """
        residual = self.hc_post(x, residual, post, comb)
        post, comb, layer_input = self.hc_pre(
            residual,
            hc_fn,
            hc_scale,
            hc_base,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )
        return residual, post, comb, layer_input


class Glm5NextModel(nn.Module):
    """GLM-5.Next 文本塔主干：嵌入 + 解码层堆叠 + 最终归一化。

    职责：
      - 管理 topk_indices_buffer（稀疏 MLA 的 top-k 共享缓冲，v32 配置）；
      - 构建解码层（make_layers，支持 PP 切片 start_layer..end_layer）；
      - 处理 PP 中间张量（mHC 状态 post/comb 跨 rank 传递）与
        序列并行 sp_shard/sp_all_gather；
      - 权重加载（堆叠映射 + 专家映射 + FP8 索引器 WK 反量化）。
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        """初始化文本主干。

        参数（语法点：* 表示 keyword-only）：
            vllm_config: vLLM 全局配置。
            prefix: 模块名前缀。
        """
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config

        self.vocab_size = config.vocab_size
        self.device = current_platform.device_type

        # 步骤1: v32（稀疏）配置时预分配 top-k 索引缓冲。
        # 宽度 = topk + 未满池余量(kpool-1)，再向上取整到 128 的倍数
        # （稀疏 MLA 内核按 BLOCK_N=128 分块，填充列保持掩码）。
        self.is_v32 = config.index_topk is not None
        if self.is_v32:
            topk_tokens = config.index_topk
            assert topk_tokens is not None
            # Reserve room for the incomplete pool tail.
            kpool = config.index_kpool
            assert kpool is not None
            buffer_width = topk_tokens + (kpool - 1 if kpool > 1 else 0)
            # Sparse MLA tiles top-k in 128 columns; padded slots remain masked.
            sparse_topk_block_n = 128
            buffer_width = ((buffer_width + sparse_topk_block_n - 1) // sparse_topk_block_n) * sparse_topk_block_n
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                buffer_width,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            # Full-MLA config (no kpool sparse indexer): no topk buffer.
            # 全 MLA 配置（无 kpool 稀疏索引器）：无 topk 缓冲。
            topk_indices_buffer = None

        # 步骤2: 词嵌入（PP 首卡持有，其余卡 PPMissingLayer 占位）。
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def get_layer(prefix: str):
            """层工厂：从层名前缀解析层号并构建 Glm5NextDecoderLayer。"""
            layer_idx = int(prefix.rsplit(".", 1)[1])
            return Glm5NextDecoderLayer(
                vllm_config=vllm_config,
                config=config,
                layer_idx=layer_idx,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
            )

        # 步骤3: 构建层堆叠（PP 下只激活本 rank 负责的层切片）。
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )
        # The active slice is fixed after construction; cache it so forward
        # doesn't rebuild the slice (a fresh list) every step.
        # 激活切片构造后固定；缓存避免 forward 每步重建切片（新列表）。
        self._active_layers = self.layers[self.start_layer : self.end_layer]

        # 步骤4: 最终 RMSNorm（PP 末卡持有）。
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        # 步骤5: 序列并行与 PP 互斥校验（sp_shard 非幂等，跨 PP 边界
        # 会二次分片已分片的张量）。
        self.is_sequence_parallel = vllm_config.parallel_config.use_sequence_parallel_moe
        if self.is_sequence_parallel and get_pp_group().world_size > 1:
            # SP shards the activations once per TP rank and sp_shard is not
            # idempotent, so crossing a PP boundary would re-shard an already
            # sharded tensor. Keep the two features mutually exclusive.
            raise NotImplementedError(
                "Sequence parallelism (use_sequence_parallel_moe) is not supported together with "
                "pipeline parallelism for GLM-5.3-Flash."
            )

        world_size = get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % world_size == 0, "num_attention_heads must be divisible by world_size"

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """词嵌入查表：[num_tokens] -> [num_tokens, hidden_size]。"""
        return self.embed_tokens(input_ids)

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        """Placeholder inputs for PP ranks > 0 (profiling / dummy runs).

        The tensors must mirror exactly what the previous rank sends. With mHC
        the per-token stream stays 2D while ``residual`` carries the ``n``
        hyper-connection streams, and the deferred hc_post state travels as
        ``post`` / ``comb`` (FP32, matching the hc_pre kernel outputs).
        Non-mHC configs only ship ``hidden_states`` / ``residual``.

        为 PP 首个之后的 rank 构造占位中间张量（profiling/假跑用）。

        参数：
            batch_size: token 数。
            dtype: 激活 dtype。
            device: 目标设备。
        """
        config = self.config
        if not config.mhc:
            return IntermediateTensors(
                {
                    "hidden_states": torch.zeros((batch_size, config.hidden_size), dtype=dtype, device=device),
                    "residual": torch.zeros((batch_size, config.hidden_size), dtype=dtype, device=device),
                }
            )
        n = config.mhc_num_residual_streams
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros((batch_size, config.hidden_size), dtype=dtype, device=device),
                "residual": torch.zeros((batch_size, n, config.hidden_size), dtype=dtype, device=device),
                "post": torch.zeros((batch_size, n, 1), dtype=torch.float32, device=device),
                "comb": torch.zeros((batch_size, n, n), dtype=torch.float32, device=device),
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """文本主干前向：嵌入 -> 逐层解码 -> 最终归一化。

        参数：
            input_ids: [num_tokens] token id（与 inputs_embeds 二选一）。
            positions: [num_tokens] 位置。
            intermediate_tensors: PP 上一 stage 的中间张量。
            inputs_embeds: 预计算的输入嵌入（多模态路径传入）。
            **kwargs: 兼容扩展。

        返回：
            [num_tokens, hidden_size]（PP 中间 rank 返回 IntermediateTensors）。
        """
        # 步骤1: 输入来源——首 rank 嵌入或 inputs_embeds；其余 rank 从
        # intermediate_tensors 恢复（含 mHC 的 post/comb 延迟状态）。
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
            post = None
            comb = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
            # Continue the previous rank's deferred mHC state so this rank's
            # first layer runs hc_post_pre, identical to the single-stage math.
            # 延续上一 rank 的 mHC 延迟状态，使本 rank 第一层执行
            # hc_post_pre——与单 stage 数值完全一致。
            post = intermediate_tensors["post"] if self.config.mhc else None
            comb = intermediate_tensors["comb"] if self.config.mhc else None

        full_num_tokens = positions.shape[0]
        # 步骤2: 序列并行：进入层堆叠前按 TP 分片激活。
        if self.is_sequence_parallel:
            hidden_states = sp_shard(hidden_states)

        # 步骤3: 逐层前向（mHC 状态在层间传递）。
        for layer in self._active_layers:
            hidden_states, residual, post, comb = layer(positions, hidden_states, residual, post, comb)

        if not get_pp_group().is_last_rank:
            # Ship the deferred mHC state produced by this rank's last layer so
            # the next stage can continue exactly where this one stopped.
            # 步骤4a: PP 中间 rank：把本 rank 最后一层产出的 mHC 延迟状态
            # 发给下一 stage，使其能无缝续算。
            tensors = {"hidden_states": hidden_states, "residual": residual}
            if self.config.mhc:
                tensors["post"] = post
                tensors["comb"] = comb
            return IntermediateTensors(tensors)

        # 步骤4b: 末 rank：SP 下 all-gather 还原 + 最终 RMSNorm。
        if self.is_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]

        hidden_states = self.norm(hidden_states)
        return hidden_states

    # Entries are (name, weight) or (name, weight, kwargs); the optional third
    # element carries per-weight loader arguments used by the fused FP8 paths.
    # 步骤说明见函数内注释。
    def load_weights(self, weights: Iterable[tuple[Any, ...]]) -> set[str]:
        """文本主干权重加载（堆叠映射 + 专家映射 + FP8 索引器反量化）。

        参数：
            weights: (name, tensor) 或 (name, tensor, kwargs) 迭代器；
                第三元素携带融合 FP8 路径的逐权重加载参数。

        返回：
            set[str]: 成功加载的参数名集合。

        堆叠映射（stacked_params_mapping）把 checkpoint 的独立权重融合进
        运行时的合并层：gate/up -> gate_up；q_a/kv_a -> fused_qkv_a；
        wk/weights_proj -> wk_weights_proj；KDA 六投影 -> in_proj_qkvbfg_a。
        """
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            # MLA: fuse q_a_proj and kv_a_proj_with_mqa
            (".fused_qkv_a_proj", ".q_a_proj", 0),
            (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
            # Indexer: fuse wk and weights_proj
            (".wk_weights_proj", ".wk", 0),
            (".wk_weights_proj", ".weights_proj", 1),
            # KDA: merge q, k, v, b, f_a, g_a projections into one GEMM
            (".in_proj_qkvbfg_a", ".q_proj", 0),
            (".in_proj_qkvbfg_a", ".k_proj", 1),
            (".in_proj_qkvbfg_a", ".v_proj", 2),
            (".in_proj_qkvbfg_a", ".b_proj", 3),
            (".in_proj_qkvbfg_a", ".f_a_proj", 4),
            (".in_proj_qkvbfg_a", ".g_a_proj", 5),
        ]
        if self.config.is_moe:
            # Params for weights, fp8 weight scales, fp8 activation scales
            # (param_name, weight_name, expert_id, shard_id)
            # MoE 模型：生成专家参数映射（权重/FP8 权重 scale/FP8 激活 scale）。
            expert_params_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.config.n_routed_experts,
            )
        else:
            expert_params_mapping = []
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # FP8 索引器 WK 的待配对缓冲：weight 与 scale 都到齐后反量化融合。
        _pending_wk_fp8: dict = {}

        # 步骤1: 逐权重循环加载。
        for args in weights:
            # 语法点：args[:2] 解包前两个元素；三元表达式取可选 kwargs。
            name, loaded_weight = args[:2]
            kwargs: dict = args[2] if len(args) > 2 else {}
            if "rotary_emb.inv_freq" in name:
                # RoPE inv_freq 非参数，运行时重建，跳过。
                continue

            # 步骤2: 跳过 MTP 投机层权重（由 mtp.py 的加载器负责）。
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                # ColossalAI 训练的模型可能带这些缓存张量，跳过。
                continue

            # Handle FP8 indexer WK: dequantize to BF16 for fusion with
            # weights_proj into wk_weights_proj.
            # 步骤3: FP8 索引器 WK——反量化为 BF16 再与 weights_proj 融合
            # 进 wk_weights_proj（见 _try_load_fp8_indexer_wk）。
            if _try_load_fp8_indexer_wk(
                name,
                loaded_weight,
                _pending_wk_fp8,
                params_dict,
                loaded_params,
            ):
                continue

            # 步骤4: 堆叠权重匹配（for-else 语法：循环未 break 才走 else）。
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                # 专家权重走下方 expert_params_mapping，此处先跳过，
                # 防止名字被改写两次（gate_proj -> gate_up_proj ->
                # gate_gate_up_proj）导致加载失败。
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                # QKV fusion: skip if fused module doesn't exist in model
                # QKV 融合：模型中不存在融合模块则跳过（部分配置无 q 低秩）。
                if param_name == ".fused_qkv_a_proj" and name_mapped not in params_dict:
                    continue
                name = name_mapped
                # Skip loading extra bias for GPTQ models.
                # 跳过 GPTQ 模型的额外 bias。
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                # 分片加载（shard_id 标明写到融合权重的哪一段）。
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # 步骤5: 专家权重匹配。
                for idx, (
                    param_name,
                    weight_name,
                    expert_id,
                    expert_shard_id,
                ) in enumerate(expert_params_mapping):
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        expert_id=expert_id,
                        shard_id=expert_shard_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    # 步骤6: 普通权重（非堆叠、非专家）。
                    # 跳过 GPTQ 额外 bias（KDA 层的 bias 除外）。
                    if name.endswith(".bias") and name not in params_dict and not self.config.is_linear_attn:  # noqa: E501
                        continue
                    # Remapping the name of FP8 kv-scale.
                    # 重映射 FP8 KV scale 的名字；无法映射则跳过。
                    remapped_name = maybe_remap_kv_scale_name(name, params_dict)
                    if remapped_name is None:
                        continue
                    name = remapped_name
                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    # 无自定义 loader 的参数用 default_weight_loader 直拷。
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight, **kwargs)
            loaded_params.add(name)
        return loaded_params


class Glm5NextForCausalLM(nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid):
    """GLM-5.Next 文本因果语言模型（顶层包装：主干 + LM 头 + 采样接口）。

    多继承的接口（protocol/mixin，语法点：无 __init__ 的标记类）：
      - HasInnerState: 模型拥有内部状态（KDA 循环状态缓存），
        vLLM 由此启用 mamba 状态缓存管理；
      - SupportsPP: 支持流水线并行；
      - MixtureOfExperts: MoE 模型接口；
      - IsHybrid: 混合注意力模型（KDA + attention 层共存），
        vLLM 由此做块大小对齐与显存核算。

    提供的类方法供调度器/运行时查询 mamba 状态的 dtype/形状/拷贝函数。
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        """初始化：构建 Glm5NextModel 主干 + LM 头 + logits 处理器。

        参数：
            vllm_config: vLLM 全局配置。
            prefix: 模块名前缀。
        """
        super().__init__()
        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.config = self.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.quant_config = quant_config
        self.model = Glm5NextModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        # LM 头只在 PP 末卡创建；logits_processor 应用 logit_scale 缩放。
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(self.config.vocab_size, scale=self.config.logit_scale)
        # 语法点：方法别名——把子模型方法暴露为本类属性（PP 占位张量用）。
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """词嵌入查表（透传给主干）。"""
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        """前向：透传给 Glm5NextModel 主干。

        参数：
            input_ids: [num_tokens] token id。
            positions: [num_tokens] 位置。
            intermediate_tensors: PP 中间张量。
            inputs_embeds: 多模态预计算嵌入。
            **kwargs: 兼容扩展。

        返回：
            末卡 [num_tokens, hidden_size]；中间卡 IntermediateTensors。
        """
        hidden_states = self.model(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)
        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        """KDA 状态缓存的 (conv_state_dtype, recurrent_state_dtype)。

        语法点：@classmethod + 前向引用类型注解 "VllmConfig"（字符串形式
        避免导入顺序问题）。
        """
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        """KDA 状态缓存的 (conv_state_shape, recurrent_state_shape)。

        由 TP 大小、KDA 头数/头维、卷积核宽与投机 token 数计算。
        """
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = vllm_config.speculative_config.num_speculative_tokens if vllm_config.speculative_config else 0
        return MambaStateShapeCalculator.kda_state_shape(
            tp_size,
            hf_config.linear_num_heads,
            hf_config.linear_head_dim,
            conv_kernel_size=hf_config.linear_conv_kernel_dim,
            num_spec=num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc, MambaStateCopyFunc, MambaStateCopyFunc]:
        """KDA 状态缓存的各种拷贝函数（如 preemption 回滚/恢复用）。"""
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """计算 LM logits：lm_head 投影（logit_scale 在 processor 内应用）。

        参数：
            hidden_states: [num_tokens, hidden_size]。

        返回：
            [num_tokens, vocab_size] logits。
        """
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """权重加载：委托 AutoWeightsLoader 自动递归匹配。"""
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


@MULTIMODAL_REGISTRY.register_processor(
    Glm5NextMultiModalProcessor,
    info=Glm5NextProcessingInfo,
    dummy_inputs=Glm4vDummyInputsBuilder,
)
# 语法点：@MULTIMODAL_REGISTRY.register_processor(...) 装饰器——把多模态
# 处理器（图像/视频输入的预处理与 token 展开规则）注册进 vLLM 的
# 多模态注册表，使该模型类可接受图像/视频输入。
class Glm5NextForConditionalGeneration(Glm4vForConditionalGeneration, HasInnerState, IsHybrid):
    """GLM-5.Next 多模态条件生成模型（视觉塔 + 文本 LM 的顶层包装）。

    继承 Glm4vForConditionalGeneration（GLM-4V 的多模odal包装，提供
    视觉特征注入与通用多模态 forward），并实现 HasInnerState/IsHybrid
    接口——文本模型是 hybrid mamba 模型，多模态包装必须声明相同接口，
    vLLM 才会按混合模型处理（对齐 mamba/attention 块大小、核算 mamba
    状态缓存）；mamba 状态类方法全部委托文本模型。

    结构：
      visual: Glm5NextVisionTransformer（multimodal.py）
      language_model: Glm5NextForCausalLM（经 init_vllm_registered_model 构建）
    """

    # The text model (KDA + dense-MLA + MoE) is a hybrid mamba model. The
    # multimodal wrapper must declare the same interfaces so vLLM treats it as
    # hybrid (auto-aligns mamba/attention block sizes, sizes the mamba state
    # cache); the mamba-state classmethods delegate to the text model.
    # 语法点：ClassVar[Literal[True]] —— 类级常量且类型限定为字面量 True。
    has_inner_state: ClassVar[Literal[True]] = True
    is_hybrid: ClassVar[Literal[True]] = True

    # 权重名重映射：GLM-OCR/GLM-4V 序列化约定 -> 本类模块命名。
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
            "model.visual.": "visual.",
        },
        # ModelSlim W8A8 checkpoints group the KDA forget-gate tensors under
        # ``forget_gate``; the runtime KDA module keeps those parameters flat.
        orig_to_new_substr={
            ".forget_gate.": ".",
            ".attn_hc.fn": ".hc_attn_fn",
            ".attn_hc.base": ".hc_attn_base",
            ".attn_hc.scale": ".hc_attn_scale",
            ".ffn_hc.fn": ".hc_ffn_fn",
            ".ffn_hc.base": ".hc_ffn_base",
            ".ffn_hc.scale": ".hc_ffn_scale",
        },
    )

    # NOTE: weight-prefix mapping is inherited from Glm4vForConditionalGeneration
    # (``model.visual.`` -> ``visual.``, ``model.language_model.`` ->
    # ``language_model.model.``, ``lm_head.`` -> ``language_model.lm_head.``),
    # matching the GLM-OCR / GLM-4V serialization convention. If the real
    # checkpoint's safetensors keys differ (e.g. ``language_model.model.`` with
    # no outer ``model.``), override ``hf_to_vllm_mapper`` accordingly.

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        """mamba 状态 dtype：委托 Glm5NextForCausalLM（函数内导入避免循环依赖）。"""
        from .model import Glm5NextForCausalLM

        return Glm5NextForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        """mamba 状态形状：委托 Glm5NextForCausalLM。"""
        from .model import Glm5NextForCausalLM

        return Glm5NextForCausalLM.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        """mamba 状态拷贝函数：委托 Glm5NextForCausalLM。"""
        from .model import Glm5NextForCausalLM

        return Glm5NextForCausalLM.get_mamba_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        """初始化多模态模型：视觉塔 + 文本 LM。

        参数：
            vllm_config: vLLM 全局配置。
            prefix: 模块名前缀。
        """
        # 语法点：显式调用祖父类 nn.Module 的 __init__
        # （跳过 Glm4vForConditionalGeneration 的构造以完全自定义）。
        super(Glm4vForConditionalGeneration, self).__init__()
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        assert multimodal_config is not None

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        # 视觉塔数据并行（每卡完整塔、按数据切分）或张量并行。
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = multimodal_config.is_multimodal_pruning_enabled()

        # 步骤1: 视觉塔。注意两点 GLM 特有修正：
        #  a) norm eps 必须取 VISION 子配置的 1e-6——顶层 config.rms_norm_eps
        #     会被 Glm5NextConfig.__getattribute__ 镜像成 text 的 1e-5；
        #  b) quant_config=None——fp8 checkpoint 中视觉塔权重是 BF16
        #     （无 weight_scale_inv），继承全局 fp8 配置会错误量化产生 NaN。
        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Glm5NextVisionTransformer(
                config.text_config,
                config.vision_config,
                # Read eps from the VISION sub-config, not the top-level
                # `config.rms_norm_eps`: Glm5NextConfig.__getattribute__ mirrors
                # the latter onto text_config (1e-5), silently ignoring the
                # vision tower's own (1e-6) rms_norm_eps.
                norm_eps=config.vision_config.rms_norm_eps,
                # Vision tower ships BF16 weights in this fp8 checkpoint (no
                # weight_scale_inv for visual.*), so it must NOT inherit the
                # global fp8 quant_config -- doing so incorrectly quantizes
                # the tower
                # and yields NaN image features. Mirrors the MLA/KDA proj
                # pattern (quant_config=None for BF16 submodules).
                quant_config=None,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Glm5NextForCausalLM"],
            )

        # The language model owns the PP contract; expose it on the multimodal
        # wrapper so PP ranks > 0 can materialize placeholder intermediate
        # tensors (the vision tower only executes on the first rank).
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

        # 步骤2: 文本 LM——通过 init_vllm_registered_model 按
        # Glm5NextForCausalLM 架构构建（复用 vLLM 注册表）。
        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Glm5NextForCausalLM"],
            )

        # The language model owns the PP contract; expose it on the multimodal
        # wrapper so PP ranks > 0 can materialize placeholder intermediate
        # tensors (the vision tower only executes on the first rank).
        # 文本 LM 拥有 PP 契约；在多模态包装上暴露该方法使 PP 首卡之外的
        # rank 能构造占位中间张量（视觉塔只在首卡执行）。
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    def load_weights(self, weights: Iterable[tuple[Any, ...]]) -> set[str]:
        """权重加载：AutoWeightsLoader + 双层权重名重映射。

        原理：视觉 merger 的 down_proj 已含导出时的旋转（QuaRot），
        忽略独立的 rot 张量避免二次旋转。映射管道：
        hf_to_vllm_mapper | WeightsMapper("rot." -> None)。
        （语法点：WeightsMapper 的 | 运算符重载组合两个映射器。）
        """
        # The visual merger's down_proj already contains the exported rotation.
        # Ignore the standalone QuaRot tensor to avoid applying it a second time.
        loader = AutoWeightsLoader(self)
        mapper = self.hf_to_vllm_mapper | WeightsMapper(orig_to_new_prefix={"rot.": None})
        return loader.load_weights(weights, mapper=mapper)

    def get_encoder_cudagraph_config(self):
        """视觉编码器 CUDA 图配置：过滤掉 GLM4V 专用的 pos_embeds 缓冲。

        本视觉塔不产生绝对位置嵌入缓冲，需从 buffer_keys 中剔除，
        否则图捕获时缺缓冲会失败。
        """
        # This vision tower does not produce the absolute position embedding
        # buffer used by GLM4V.
        config = super().get_encoder_cudagraph_config()
        config.buffer_keys = [k for k in config.buffer_keys if k != "pos_embeds"]
        return config


def get_spec_layer_idx_from_weight_name(config: Glm5NextConfig, weight_name: str) -> int | None:
    """从权重名解析 MTP 投机层下标；非 MTP 权重返回 None。

    参数：
        config: 模型配置（num_nextn_predict_layers / num_hidden_layers）。
        weight_name: checkpoint 权重名，如 "model.layers.45.xxx"。

    返回：
        int: MTP 层下标（num_hidden_layers + i）；或 None。
    兼容 "model.layers.N." 与 "layers.N." 两种前缀拼写。
    """
    if hasattr(config, "num_nextn_predict_layers") and (config.num_nextn_predict_layers > 0):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(f"model.layers.{layer_idx + i}.") or weight_name.startswith(
                f"layers.{layer_idx + i}."
            ):
                return layer_idx + i
    return None


def _try_load_fp8_indexer_wk(name, tensor, buf, params_dict, loaded_params):
    """加载 FP8 索引器 WK：weight 与 scale 配对后反量化融合。

    原理：checkpoint 中索引器的 wk 是 FP8（e4m3）+ weight_scale_inv，
    但运行时 wk 与 weights_proj 融合为 BF16 的 wk_weights_proj。
    因此先在 buf（按层前缀分组）里攒 weight/scale，两者齐备后：
      1. 由形状差推出分组量化的 block_size；
      2. scaled_dequantize 反量化为 BF16；
      3. 以 shard_id=0 写入融合权重 wk_weights_proj。

    参数：
        name: 权重名。
        tensor: 权重张量。
        buf: dict，层前缀 -> {"weight":..., "scale":...} 待配对缓冲。
        params_dict: 模型参数字典。
        loaded_params: 已加载参数名集合（成功融合时登记）。

    返回：
        bool: 本权重是否被此函数消费（True 则调用方跳过后续逻辑）。
    """
    # 只处理索引器 wk（且不是已融合的 wk_weights 名）。
    if "indexer.wk." not in name or "wk_weights" in name:
        return False
    is_weight = name.endswith(".weight") and tensor.dtype == torch.float8_e4m3fn
    is_scale = "weight_scale_inv" in name
    if not is_weight and not is_scale:
        return False
    # 按层前缀（"...indexer"）分组暂存。
    layer_prefix = name.rsplit(".wk.", 1)[0]
    entry = buf.setdefault(layer_prefix, {})
    entry["weight" if is_weight else "scale"] = tensor
    # 只到一半：等另一半到达。
    if "weight" not in entry or "scale" not in entry:
        return True

    # 步骤1: 取出配对并清缓冲。
    weight_fp8, scale_inv = entry["weight"], entry["scale"]
    del buf[layer_prefix]
    # 步骤2: 由 K 维长度比推出量化块大小（如 128）。
    block_size = weight_fp8.shape[1] // scale_inv.shape[1]
    # 步骤3: 分组反量化到 BF16。
    weight_bf16 = scaled_dequantize(
        weight_fp8,
        scale_inv,
        group_shape=GroupShape(block_size, block_size),
        out_dtype=torch.bfloat16,
    )

    # 步骤4: 以分片 0 写入融合权重（weights_proj 走正常路径写分片 1）。
    fused_name = f"{layer_prefix}.wk_weights_proj.weight"
    param = params_dict[fused_name]
    param.weight_loader(param, weight_bf16, 0)
    loaded_params.add(fused_name)
    return True
