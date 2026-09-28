# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 model adapters for vLLM 0.27 on Ascend.

vLLM owns Kimi's configuration, multimodal processor, weight mappings, and
model-level forward contract.  This module composes those upstream pieces with
the generic MLA/MoE implementation and the Ascend KDA backend.

【中文说明 —— 本文件在插件架构中的位置】
Kimi K3（KimiLinear / KimiK3 多模态）的昇腾 NPU 适配层。上游 vLLM 拥有
配置、多模态处理器、权重映射与前向契约；本文件把这些上游组件与
通用的 MLA/MoE 实现、昇腾 KDA 后端组装起来。

【Kimi K3 模型结构概览】
- 混合注意力：按层交错 KDA 线性注意力（DeltaNet 类，常数复杂度，
  由 AscendKimiK3DeltaAttention 承载）与无 RoPE 的 MLA 潜在注意力层；
- MoE：无辅助损失路由（e_score_correction_bias 偏置 + sigmoid 打分）、
  共享专家、可选"潜空间 MoE"（专家在降维空间中计算，省显存）；
- 注意力残差块（attn_res_block_size）：每 N 层把前缀和流快照进
  block_residual，层内用"学习式 softmax 残差混合"代替朴素相加
  （torch.ops._C_ascend.attn_res_fwd 融合算子，Kimi 特有的动态残差机制）；
- 序列并行（SP）：非注意力层在本地 token 分片上计算，
  进注意力前 all_gather、出来后 reduce_scatter；
- DSpark 投机解码支持：aux 隐状态捕获（物化/原始前缀和两种模式）。
"""

import math
from bisect import bisect_left
from collections.abc import Callable, Iterable, Sequence
from copy import copy
from typing import Any, TypeVar, cast

import torch
import vllm.envs as envs
from torch import nn
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.parallel_state import model_parallel_is_initialized
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import MixtureOfExperts
from vllm.model_executor.models.kimi_k25_vit import (
    KimiK25MultiModalProjector,
    MoonViT3dPretrainedModel,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    init_vllm_registered_model,
    make_layers,
    maybe_prefix,
)
from vllm.model_executor.models.vision import is_vit_use_data_parallel
# 序列并行原语（vLLM 上游通用实现）：注意这里导入的是 vllm.models.common，
# 与 vllm_ascend/models/common/ops/sequence_parallel.py 是"上游-插件"镜像关系。
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_reduce_scatter,
    sp_shard,
)
# 上游参考实现（AMD 版被当作硬件无关基线复用其结构契约）。
from vllm.models.kimi_k3.amd.linear import (
    KimiDecoderLayer as UpstreamKimiDecoderLayer,
)
from vllm.models.kimi_k3.amd.linear import KimiLinearForCausalLM as UpstreamKimiLinearForCausalLM
from vllm.models.kimi_k3.amd.linear import KimiLinearModel as UpstreamKimiLinearModel
from vllm.models.kimi_k3.amd.linear import (
    KimiMLAAttention as UpstreamKimiMLAAttention,
)
from vllm.models.kimi_k3.amd.linear import (
    KimiMLP,
    KimiRoutedOutputTransform,
)
from vllm.models.kimi_k3.amd.model import (
    KimiK3ForConditionalGeneration as UpstreamKimiK3ForConditionalGeneration,
)
from vllm.models.kimi_k3.common.mm_preprocess import (
    KimiK3DummyInputsBuilder,
    KimiK3MultiModalProcessor,
    KimiK3ProcessingInfo,
)
from vllm.models.kimi_k3.nvidia.model import (
    KimiLinearModel as UpstreamPackedKimiLinearModel,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.utils import mark_fused_preprocess_weights
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
# 昇腾 KDA（Kimi Delta Attention）后端：线性注意力的 NPU 实现。
from vllm_ascend.ops.kimi_kda import AscendKimiK3DeltaAttention  # type: ignore[import-untyped]
from vllm_ascend.ops.linear_op import KimiOProjMMReduceScatterOp
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DynamicLinearMethod
from vllm_ascend.utils import get_rotation_path, is_950
from vllm_ascend.worker.v2.pp_transport import (
    PPTransportDataType,
    add_pp_transport_tensors,
    get_pp_transport_tensors,
)
from vllm_ascend.worker.v2.pp_transport import (
    make_empty_intermediate_tensors as make_pp_empty_intermediate_tensors,
)


def _use_attn_res_prefill_cache() -> bool:
    """Enable weight caching only for explicit pure-prefill A5 requests."""
    """【中文说明】仅对 A5 上的显式纯 prefill 请求启用注意力残差权重缓存
    （所有层均为 prefill、无 decode、无投机掩码时才开启）。"""
    if not is_forward_context_available():
        return False
    metadata = get_forward_context().attn_metadata
    if not isinstance(metadata, dict) or not metadata:
        return False
    for layer_metadata in metadata.values():
        prefills = getattr(layer_metadata, "num_prefills", None)
        decodes = getattr(layer_metadata, "num_decodes", None)
        if type(prefills) is not int or type(decodes) is not int or prefills <= 0 or decodes != 0:
            return False
        if getattr(layer_metadata, "spec_sequence_masks", None) is not None:
            return False
    return is_950()


class AscendKimiRoutedOutputTransform(KimiRoutedOutputTransform):
    """Fuse the latent RMSNorm and MXFP8 activation quantization on A5."""
    """【中文说明】在 A5 上把"潜空间 RMSNorm + MXFP8 激活动态量化"融合为
    单个 NPU 算子（条件不满足时回退上游分步实现）。"""

    def __init__(self, norm: RMSNorm | None, up_proj: ReplicatedLinear) -> None:
        super().__init__(norm, up_proj)
        self._supports_mx_norm_fusion = get_current_hardware_profile().supports(
            HardwareCapability.DYNAMIC_MX_QUANT_FUSION
        ) and hasattr(torch.ops.npu, "npu_rms_norm_dynamic_mx_quant")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        scheme = getattr(getattr(self.up_proj, "quant_method", None), "quant_method", None)
        if (
            self.norm is not None
            and self._supports_mx_norm_fusion
            and isinstance(scheme, AscendW8A8MXFP8DynamicLinearMethod)
            and scheme.group_size == 32
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.shape[-1] % 64 == 0
            and getattr(self.norm, "bias", None) is None
        ):
            # 融合路径：一次 NPU 算子完成 RMSNorm + 动态 MX 量化，再走 up_proj。
            quantized, scale, _ = torch.ops.npu.npu_rms_norm_dynamic_mx_quant(
                hidden_states,
                self.norm.weight,
                epsilon=self.norm.variance_epsilon,
                scale_alg=scheme.dynamic_mx_quant_scale_alg,
                dst_type=torch.float8_e4m3fn,
            )
            hidden_states, _ = self.up_proj((quantized, scale))
            return hidden_states
        return super().forward(hidden_states)


class AscendKimiMLP(KimiMLP):
    """Keep TP-sharded dense weights compatible with sequence-sharded tokens."""
    """（dense MLP 的序列并行适配：TP 切分的权重 + 序列切分的 token 协同工作。）

    【原理】序列并行（SP）下，每个 rank 只持有部分 token；
    而 MLP 的 gate/up（列并行，输出维切分）+ down（行并行，输入维切分）
    要求所有 TP rank 在**相同 token** 上各算一部分再求和。
    因此 forward 前后包裹 sp_all_gather / sp_reduce_scatter：
    - 进入前 gather：拼回全序列 token（所有 rank 一致）；
    - 出来后 reduce_scatter：部分和规约 + 重新按序列切分。
    上游 KimiMLP 的 reduce_results 在 SP 下必须关闭（由 sp_reduce_scatter
    统一做规约，避免二次 all_reduce）。
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
        activation_situ_beta: float | None = None,
        activation_situ_linear_beta: float | None = None,
        use_sequence_parallel: bool = False,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            quant_config=quant_config,
            # SP 模式下关闭行并行的自动规约（交给 sp_reduce_scatter）。
            reduce_results=False if use_sequence_parallel else reduce_results,
            prefix=prefix,
            activation_situ_beta=activation_situ_beta,
            activation_situ_linear_beta=activation_situ_linear_beta,
        )
        self.use_sequence_parallel = use_sequence_parallel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [num_local_tokens, hidden]（SP 下是本 rank 的 token 分片）。"""
        if self.use_sequence_parallel:
            # All weight shards must operate on the same tokens. Reducing
            # different sequence shards would mix live and padding rows.
            # （所有权重分片必须作用于相同 token；对不同序列分片规约
            #   会把有效行与填充行混在一起。）
            x = sp_all_gather(x)
        x = super().forward(x)
        if self.use_sequence_parallel:
            x = sp_reduce_scatter(x)
        return x


_WeightT = TypeVar("_WeightT", bound=tuple[str, torch.Tensor] | tuple[str, torch.Tensor, dict[str, Any]])


def load_eplb_expert_weights(
    model: nn.Module,
    weights: Iterable[_WeightT],
    num_redundant_experts: int,
    map_weight_name: Callable[[str], str | None] | None = None,
) -> Iterable[_WeightT]:
    """Stream expert weights according to the complete initial EPLB layout.

    Upstream Kimi loaders stop at the first matching expert mapping, so simply
    adding redundant entries to that mapping would leave replicas uninitialized.
    Remap the first logical-expert-count slots for the upstream loader and load
    the remaining slots directly. MRV2 can place replicas in either range.
    """
    if not num_redundant_experts or not model.config.is_moe:
        yield from weights
        return

    params_dict = dict(model.named_parameters())
    expert_mapping = fused_moe_make_expert_params_mapping(
        model,
        ckpt_gate_proj_name="w1",
        ckpt_down_proj_name="w2",
        ckpt_up_proj_name="w3",
        num_experts=model.config.num_experts,
        num_redundant_experts=num_redundant_experts,
    )
    experts_unpacked = not any(name.endswith("w13_weight_packed") for name in params_dict)
    for args in weights:
        checkpoint_name = cast(str, args[0])
        name = map_weight_name(checkpoint_name) if map_weight_name else checkpoint_name
        handled = False
        if name is not None:
            if experts_unpacked and name.endswith(".weight_packed"):
                name = name.replace(".weight_packed", ".weight")
            for param_name, weight_name, expert_id, shard_id in expert_mapping:
                if weight_name not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if mapped_name not in params_dict:
                    continue
                handled = True
                if expert_id < model.config.num_experts:
                    # The upstream mapping is identity and still owns loaded
                    # parameter tracking and MTP checkpoint validation.
                    logical_id = weight_name.split(".", 2)[1]
                    physical_name = checkpoint_name.replace(f"experts.{logical_id}.", f"experts.{expert_id}.")
                    yield cast(_WeightT, (physical_name, *args[1:]))
                else:
                    param = params_dict[mapped_name]
                    param.weight_loader(param, args[1], mapped_name, expert_id=expert_id, shard_id=shard_id)
        if not handled:
            yield args


def is_moe_layer_idx(config, layer_idx: int) -> bool:
    """Whether the Kimi layer at ``layer_idx`` hosts routed experts."""
    """【中文说明】判断第 layer_idx 层是否为 MoE 层（承载路由专家）。"""
    return bool(
        config.is_moe
        and config.num_experts is not None
        and layer_idx >= config.first_k_dense_replace
        and layer_idx % config.moe_layer_freq == 0
    )


class AscendKimiMoE(nn.Module):
    """Kimi K3 MoE assembled from the standard vLLM MoE interfaces."""
    """（Kimi K3 的 MoE 层 —— 用 vLLM 标准接口组装。）

    【MoE 原理】
    - gate：路由器，对每个 token 打分选出 top-k 个专家（FP32 精度保数值稳定）；
    - e_score_correction_bias：无辅助损失路由（DeepSeek-V3 式）——训练时
      不需要 load-balancing loss，推理时用偏置修正专家得分缓解负载不均；
    - shared_experts：共享专家，所有 token 无条件经过（捕获公共知识）；
    - 分组路由（use_grouped_topk）：专家先分组、组内选优，控制专家多样性；
    - routed_scaling_factor：路由输出缩放（DeepSeek 惯例，补偿 top-k 摊薄）。
    - 潜空间 MoE（use_latent_moe，Kimi 特色）：专家不在原 hidden 空间计算，
      而是 down_proj 降到 moe_hidden_size → 专家计算 → norm → up_proj 升回，
      大幅减少专家权重显存（专家权重与 hidden 解耦）。
    """

    def __init__(
        self,
        *,
        config,
        vllm_config: VllmConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_sequence_parallel: bool = False,
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        moe_intermediate_size = config.moe_intermediate_size
        num_experts = config.num_experts
        num_experts_per_token = config.num_experts_per_token
        # MoE 必需配置存在性断言（缺失说明配置解析出错，尽早失败）。
        assert moe_intermediate_size is not None
        assert num_experts is not None
        assert num_experts_per_token is not None

        parallel_config = vllm_config.parallel_config
        num_redundant_experts = parallel_config.eplb_config.num_redundant_experts if parallel_config.enable_eplb else 0

        routed_expert_hidden_size = config.routed_expert_hidden_size
        # 潜空间 MoE 开关：配置了 routed_expert_hidden_size 即启用。
        self.use_latent_moe = routed_expert_hidden_size is not None
        self.moe_hidden_size = routed_expert_hidden_size or hidden_size
        self.latent_moe_use_norm = config.latent_moe_use_norm
        self.routed_scaling_factor = config.routed_scaling_factor
        self.num_shared_experts = config.num_shared_experts
        # situ 激活的专属超参（Kimi 自定义激活函数的 beta 系数），
        # 仅 hidden_act == "situ" 时传入，否则为 None。
        activation_situ_beta = config.activation_situ_beta if config.hidden_act == "situ" else None
        activation_situ_linear_beta = config.activation_situ_linear_beta if config.hidden_act == "situ" else None

        # 路由器：hidden → num_experts 的打分，FP32 输出。
        # GateLinear 是 vLLM 专用于 MoE gate 的线性层（保持 FP32、无量化）。
        self.gate = GateLinear(
            input_size=hidden_size,
            output_size=num_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        # 无辅助损失路由的得分修正偏置（可学习，FP32）。
        self.gate.e_score_correction_bias = nn.Parameter(torch.empty(num_experts, dtype=torch.float32))

        if self.num_shared_experts is not None:
            # 共享专家：实现为"中间层加宽的单个 MLP"（intermediate × num_shared），
            # 数学上等价于多个并联专家但只用一个 GEMM，reduce_results=False
            # 让输出加法在 FusedMoE 融合内核里完成。
            self.shared_experts = KimiMLP(
                hidden_size=hidden_size,
                intermediate_size=moe_intermediate_size * self.num_shared_experts,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
                activation_situ_beta=activation_situ_beta,
                activation_situ_linear_beta=activation_situ_linear_beta,
            )
        else:
            self.shared_experts = None

        # 潜空间投影只在昇腾量化（ascend）方案下参与量化；其他量化方案
        # （如 fp8）对这两个小投影保持 BF16，避免精度损失。
        latent_quant_config = quant_config if quant_config is not None and quant_config.get_name() == "ascend" else None
        self.routed_output_transform: KimiRoutedOutputTransform | None
        if self.use_latent_moe:
            # 进入潜空间：hidden → moe_hidden_size（降维）。
            self.routed_expert_down_proj = ReplicatedLinear(
                hidden_size,
                self.moe_hidden_size,
                bias=False,
                quant_config=latent_quant_config,
                prefix=f"{prefix}.routed_expert_down_proj",
            )
            # 潜空间归一化（可选，稳定专家输出分布）。
            self.routed_expert_norm = (
                RMSNorm(self.moe_hidden_size, eps=config.rms_norm_eps) if self.latent_moe_use_norm else None
            )
            # 升回原空间：moe_hidden_size → hidden。
            self.routed_expert_up_proj = ReplicatedLinear(
                self.moe_hidden_size,
                hidden_size,
                bias=False,
                quant_config=latent_quant_config,
                prefix=f"{prefix}.routed_expert_up_proj",
            )
            # 输出变换封装：把"norm + up_proj"打包成可整体调用的对象，
            # 交给 FusedMoEFactory 让融合内核在专家输出后统一应用
            # （昇腾版额外支持 A5 上的 norm + MXFP8 量化融合）。
            self.routed_output_transform = AscendKimiRoutedOutputTransform(
                self.routed_expert_norm,
                self.routed_expert_up_proj,
            )
        else:
            self.routed_expert_down_proj = None
            self.routed_expert_norm = None
            self.routed_expert_up_proj = None
            self.routed_output_transform = None

        # FusedMoEFactory：构建融合 MoE 内核（昇腾平台的 fused 实现），
        # 把共享专家/输入输出变换/路由参数全部挂进融合路径，
        # 一次 kernel 完成路由 + 专家 GEMM + 加权求和。
        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            num_experts=num_experts,
            top_k=num_experts_per_token,
            hidden_size=self.moe_hidden_size,
            intermediate_size=moe_intermediate_size,
            activation=config.hidden_act,
            activation_situ_beta=activation_situ_beta,
            activation_situ_linear_beta=activation_situ_linear_beta,
            renormalize=config.moe_renormalize,
            quant_config=quant_config,
            use_grouped_topk=config.use_grouped_topk,
            num_expert_group=config.num_expert_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func=config.moe_router_activation_func,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            routed_input_transform=self.routed_expert_down_proj,
            routed_output_transform=self.routed_output_transform,
            is_sequence_parallel=use_sequence_parallel,
            enable_eplb=parallel_config.enable_eplb,
            num_redundant_experts=num_redundant_experts,
        )

        # Use the factory's validated counts, including its effective EP size.
        moe_config = self.experts.moe_config
        self.n_routed_experts = num_experts
        self.n_logical_experts = moe_config.num_logical_experts
        self.n_physical_experts = moe_config.num_experts
        self.n_local_physical_experts = moe_config.num_local_experts
        self.n_redundant_experts = num_redundant_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """MoE 前向：路由打分 → 融合专家计算。输入/输出 [num_tokens, hidden]。"""
        num_tokens, hidden_size = hidden_states.shape
        # view(-1, hidden) 兜底展平（兼容 [B, S, H] 形状的调用方）。
        hidden_states = hidden_states.view(-1, hidden_size)
        router_logits, _ = self.gate(hidden_states)
        # 融合 MoE 内核：内部完成 top-k 选择、专家 GEMM、加权求和、
        # 共享专家相加、潜空间输入/输出变换、SP 处理。
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        return final_hidden_states.view(num_tokens, hidden_size)


class AscendKimiMLAAttention(UpstreamKimiMLAAttention):
    """Extend vLLM's generic Kimi MLA only for DSpark RoPE metadata."""
    """（扩展上游 Kimi MLA —— 仅为 DSpark 草稿补配 RoPE 元数据。）

    【MLA 多头潜在注意力原理（DeepSeek 式）】
    - Q 侧：可选低秩压缩（q_lora_rank），wq_a 降维 → norm → wq_b 升维到
      n_heads × (nope + rope) 两段；
    - KV 侧：压成单个 kv_lora_rank 潜向量（+ rope 段），KV Cache 只存
      压缩向量，推理时在线"升维"恢复各头的 K/V —— 显存远小于 MHA/GQA；
    - use_nope=True：Kimi K3 变体完全不用 RoPE（位置信息由 KDA 层与
      注意力残差结构提供）。
    【本类职责】上游构造已建好平台注册的 MLA 包装（含全部投影与加载器），
    这里只对**已存在**的昇腾注意力层做二次配置（scale/RoPE/双向开关），
    而不是用相同 prefix 再注册第二个包装（会破坏 static_forward_context
    的层名唯一性）。DSpark 草稿需要 RoPE 和双向解码，故这两个开关
    只在 DSpark 场景下生效。
    """

    def __init__(
        self,
        config,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        use_output_gate: bool,
        use_rope: bool,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        non_causal_multi_token_decode: bool = False,
        disable_mlapo: bool = False,
    ) -> None:
        """参数中 qk_*_head_dim/v_head_dim/kv_lora_rank 定义 MLA 的维度结构；
        use_rope/non_causal_multi_token_decode/disable_mlapo 是 DSpark 专用开关。
        """
        # copy(config)：浅拷贝配置后注入 mla_use_output_gate，
        # 避免污染调用方共享的原始 config 对象。
        upstream_config = copy(config)
        upstream_config.mla_use_output_gate = use_output_gate
        super().__init__(
            config=upstream_config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            use_nope=True,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        # 拿到上游已构建的注意力层（平台注册的 Ascend MLA 实现）。
        attention_layer = self._attention_layer
        if disable_mlapo:
            # MLAPO：MLA 输出投影的昇腾融合优化路径；DSpark 草稿禁用
            # （草稿层小批次下融合无收益），并标记权重已预处理防止重复变换。
            attention_layer.impl.enable_mlapo = False
            mark_fused_preprocess_weights(attention_layer.impl)
        if not use_rope and not non_causal_multi_token_decode:
            # 非草稿场景（目标模型的 nope MLA 层）：无需任何补配，直接返回。
            return

        # 为 DSpark 草稿构建 RoPE：
        rotary_emb = None
        if use_rope:
            rope_parameters = dict(config.rope_parameters)
            if rope_parameters["rope_type"] != "default":
                # 非 default 类型映射到 DeepSeek 的两种外推方案：
                # apply_yarn_scaling（默认）→ "deepseek_yarn"，否则线性缩放。
                rope_parameters["rope_type"] = (
                    "deepseek_yarn" if rope_parameters.get("apply_yarn_scaling", True) else "deepseek_llama_scaling"
                )
            # is_neox_style=False：GPT-J 式旋转（DeepSeek/Kimi 惯例，
            # 与 Llama 的 NeoX 交错式相对）。
            rotary_emb = get_rope(
                qk_rope_head_dim,
                max_position=config.max_position_embeddings,
                rope_parameters=rope_parameters,
                is_neox_style=False,
            )
            if rope_parameters["rope_type"] == "deepseek_yarn":
                # YaRN 的 mscale 补偿：长外推后注意力分数整体偏低，
                # 按 0.1·m·ln(s)+1 放大 scale（乘两次 = 平方进入 softmax 分母）。
                scaling_factor = float(rope_parameters["factor"])
                mscale_all_dim = float(rope_parameters.get("mscale_all_dim", 0.0))
                if scaling_factor > 1 and mscale_all_dim:
                    mscale = 0.1 * mscale_all_dim * math.log(scaling_factor) + 1.0
                    self.scaling *= mscale * mscale

        # The upstream Kimi module has already constructed the platform-
        # registered MLA wrapper, including all projections and weight loaders.
        # Configure that existing Ascend attention layer for DSpark instead of
        # constructing and registering a second wrapper with the same prefix.
        # （上游已构建平台注册的 MLA 包装；对已有的昇腾注意力层做 DSpark
        #   配置，而不是用相同 prefix 注册第二个包装。）
        # 把 scale / RoPE / 双向解码开关透传给底层 impl（真正的算子层）。
        attention_layer.scale = self.scaling
        attention_layer.non_causal_multi_token_decode = non_causal_multi_token_decode
        attention_layer.impl.scale = float(self.scaling)
        attention_layer.impl.rotary_emb = rotary_emb
        attention_layer.impl.use_mla_rope = use_rope

    # ---- 以下 @property 把深层属性代理到 self.mla_attn.mla_attn，
    # 使外部（如 kimi_k3_dspark 的 precompute_and_store_context_kv）
    # 可以像访问普通注意力层一样访问 fused_qkv_a_proj/impl/kv_cache 等。----

    @property
    def _attention_layer(self):
        # mla_attn 是上游 MLA 包装（MultiHeadLatentAttentionWrapper），
        # 其 .mla_attn 才是平台注册的注意力层本体。
        return self.mla_attn.mla_attn

    @property
    def is_vl_first_layer(self) -> bool:
        # 多模态首层标记：首层需要把视觉 token 纳入注意力布局。
        return self.mla_attn.is_vl_first_layer

    @property
    def layer_name(self) -> str:
        # 层名：static_forward_context 中注册的键（自定义算子按名寻址）。
        return self._attention_layer.layer_name

    @property
    def impl(self):
        # 注意力后端实现（Ascend MLA impl，含 fused_qkv_a_proj 等）。
        return self._attention_layer.impl

    @property
    def kv_cache(self):
        # 本层 KV cache 张量（MLA 压缩格式）。
        return self._attention_layer.kv_cache

    @property
    def kv_cache_dtype(self):
        return self._attention_layer.kv_cache_dtype

    @property
    def _k_scale(self):
        # KV cache 量化缩放因子（未量化时为 1.0）。
        return self._attention_layer._k_scale

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """MLA 注意力前向：positions [N]、hidden_states [N, H] → 输出 [N, H]。"""
        return self.mla_attn(positions, hidden_states)


class AscendKimiDecoderLayer(UpstreamKimiDecoderLayer):
    """Upstream Kimi decoder structure with Ascend attention backends."""
    """（上游 Kimi 解码层结构 + 昇腾注意力后端。）

    职责：按层号选择注意力形态（KDA 或 nope-MLA）、按配置选择
    MoE 或 dense MLP、搭建注意力残差块的打分投影。
    """

    def __init__(
        self,
        config,
        vllm_config: VllmConfig,
        prefix: str = "",
        use_sequence_parallel: bool = False,
    ) -> None:
        """Select KDA or no-RoPE MLA and configure the layer residual path."""
        # （选择 KDA 或无 RoPE MLA，并配置层残差路径。）
        # 显式初始化 nn.Module（跳过上游构造，自建组件）。
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        # 从 prefix 解析层号（"model.layers.N" 的最后一段）。
        self.layer_idx = int(prefix.rsplit(".", 1)[1])
        self.is_moe = config.is_moe
        self.use_sequence_parallel = use_sequence_parallel
        layer_idx = self.layer_idx
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        # 注意力形态选择：KDA 层用昇腾 DeltaAttention（线性注意力），
        # 其余层用无 RoPE 的 MLA（use_rope=False —— 目标模型不走 RoPE）。
        if config.is_kda_layer(layer_idx):
            self.self_attn = AscendKimiK3DeltaAttention(
                config,
                vllm_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            qk_nope_head_dim = config.qk_nope_head_dim
            qk_rope_head_dim = config.qk_rope_head_dim
            v_head_dim = config.v_head_dim
            kv_lora_rank = config.kv_lora_rank
            assert qk_nope_head_dim is not None
            assert qk_rope_head_dim is not None
            assert v_head_dim is not None
            assert kv_lora_rank is not None
            assert config.mla_use_nope is True
            self.self_attn = AscendKimiMLAAttention(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=kv_lora_rank,
                use_output_gate=bool(config.mla_use_output_gate),
                use_rope=False,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )

        # MLP 形态选择：MoE 层 = 全局 MoE 开启 && 有专家数 &&
        # 层号 ≥ first_k_dense_replace（前几层用 dense）&& 按 moe_layer_freq 间隔。
        self.is_moe_layer = is_moe_layer_idx(config, layer_idx)
        if self.is_moe_layer:
            self.block_sparse_moe = AscendKimiMoE(
                config=config,
                vllm_config=vllm_config,
                quant_config=quant_config,
                prefix=f"{prefix}.block_sparse_moe",
                use_sequence_parallel=use_sequence_parallel,
            )
            # mlp 别名指向 MoE（forward 中统一走 self.mlp）。
            self.mlp = self.block_sparse_moe
        else:
            self.mlp = AscendKimiMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=not use_sequence_parallel,
                prefix=f"{prefix}.mlp",
                use_sequence_parallel=use_sequence_parallel,
                activation_situ_beta=config.activation_situ_beta,
                activation_situ_linear_beta=config.activation_situ_linear_beta,
            )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # ---- 注意力残差块（attn_res）配置：每 block_size 层一个块，
        # 块首层负责把前缀和写入 block_residual 快照（见 forward_attn_residual）。----
        attn_res_block_size = config.attn_res_block_size
        self.use_attn_residuals = attn_res_block_size is not None
        if attn_res_block_size is not None:
            self.attn_res_block_size = attn_res_block_size
            # 块首层判定：layer_idx 整除 block_size（含第 0 层）。
            self.is_block_write_layer = layer_idx % attn_res_block_size == 0
            # 本层要写入的块下标。
            self.block_write_idx = layer_idx // attn_res_block_size
            # 本层可见的历史块数（cdiv = 向上取整除法）。
            self.prev_valid_blocks = cdiv(layer_idx, attn_res_block_size)
            # 注意力/MLP 两套残差混合打分的 norm + 投影（见 forward_attn_residual）。
            self.self_attention_res_norm = RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
            self.mlp_res_norm = RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
            # 打分投影：hidden → 1（每候选一个标量分）。
            self.self_attention_res_proj = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.self_attention_res_proj",
            )
            self.mlp_res_proj = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.mlp_res_proj",
            )

        if self.use_sequence_parallel:
            # SP 下注意力的 o_proj 不做 all_reduce（由 sp_reduce_scatter 统一处理）。
            self.self_attn.o_proj.reduce_results = False

        self.fuse_o_proj_mm_reduce_scatter = self._enable_o_proj_mm_reduce_scatter(vllm_config)

    def _enable_o_proj_mm_reduce_scatter(self, vllm_config: VllmConfig) -> bool:
        if not self.use_sequence_parallel or not self.use_attn_residuals:
            return False
        # Fusion uses each PP stage's existing TP group and returns the same
        # SP token shard as the separate projection and ReduceScatter.
        if not get_current_hardware_profile().supports(HardwareCapability.MM_REDUCE_SCATTER_AI_CPU_INFERENCE):
            return False
        if get_ascend_config().weight_nz_mode == 2:
            return False
        if vllm_config.lora_config is not None:
            return False
        o_proj = self.self_attn.o_proj
        if KimiOProjMMReduceScatterOp.unsupported_reason(o_proj) is not None:
            return False
        o_proj.custom_op = KimiOProjMMReduceScatterOp(o_proj)
        if isinstance(self.self_attn, AscendKimiMLAAttention):
            # The MLA custom op writes into a caller-owned output buffer. Its
            # token dimension must match the fused projection's TP shard.
            self.self_attn.mla_attn.output_token_shard_size = o_proj.tp_size
        return True

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        prefix_delta: torch.Tensor | None = None,
        **kwargs,
    ):
        if self.use_attn_residuals:
            assert residual is not None
            return self.forward_attn_residual(positions, hidden_states, residual, prefix_delta)
        assert prefix_delta is None
        return super().forward(positions, hidden_states, residual, **kwargs)

    def _run_self_attn(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Ascend attention returns its output instead of filling an AMD buffer.
        # （昇腾注意力直接返回输出，而 AMD 实现是写入调用方提供的缓冲。）
        return self.self_attn(positions=positions, hidden_states=hidden_states)

    def prepare_attn_residual(
        self,
        prefix_sum: torch.Tensor,
        block_residual: torch.Tensor,
        addend: torch.Tensor | None = None,
        return_materialized: bool = False,
        optimize_prefill: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """（注意力残差的"准备阶段" —— 对应旧版数据流的步骤1。）

        由一个融合算子完成：前缀和（可叠加 addend）+ 历史块的学习式
        残差混合 + 输入 RMSNorm；块首层同时在算子内部把当前前缀和
        快照写入 block_residual（对应旧版步骤2）。
        返回 (混合后的 hidden_states, 新前缀和, 可选的物化输出)。
        """
        # The previous MLP add is rounded to BF16 before it becomes either a
        # DSpark raw prefix or an AttnRes input. A block boundary stores that
        # same prefix, while the mixture reads only the older valid slots.
        return torch.ops._C_ascend.attn_res_fwd(
            prefix_sum,
            addend,
            block_residual,
            self.self_attention_res_proj.weight,
            self.self_attention_res_norm.weight,
            self.self_attention_res_norm.variance_epsilon,
            self.prev_valid_blocks,
            self.input_layernorm.weight,
            self.input_layernorm.variance_epsilon,
            self.block_write_idx if self.is_block_write_layer else -1,
            return_materialized,
            optimize_prefill=optimize_prefill,
        )

    def forward_attn_residual(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        prefix_delta: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run Kimi attention residuals with Ascend attention and MoE.

        Mirrors upstream vLLM main (#50592/#50593): `prefix_delta` is folded
        into the prefix stream before each attn-res mixture, the input and
        post-attention norms are fused into the mixture, and the layer
        returns ``(prefix_sum, block_residual, hidden_states)``.
        """
        """【中文说明】注意力残差版前向（对齐上游 vLLM main）：
        prefix_delta 在每次残差混合前并入前缀流，输入/注意力后 norm
        融合进混合算子；返回 (前缀和, 块缓冲, 隐状态)。
        """
        # The fused op returns the new prefix (`prefix + delta`) as its
        # second result, whereas upstream's `_apply_attn_res` mutates
        # `prefix_sum` in place; capture it explicitly to mirror upstream.
        # 步骤1：注意力前的残差混合（融合算子同时完成输入 norm 与块快照写入）。
        hidden_states, prefix_sum, _ = torch.ops._C_ascend.attn_res_fwd(
            hidden_states,
            prefix_delta,
            block_residual,
            self.self_attention_res_proj.weight,
            self.self_attention_res_norm.weight,
            self.self_attention_res_norm.variance_epsilon,
            self.prev_valid_blocks,
            self.input_layernorm.weight,
            self.input_layernorm.variance_epsilon,
            self.block_write_idx if self.is_block_write_layer else -1,
        )
        if self.is_block_write_layer:
            # 步骤2：块首层快照已写入，此后本层前缀和清空（从注意力输出重启）。
            prefix_sum = None

        if self.use_sequence_parallel:
            # SP：注意力需要全序列 token —— gather 后裁掉对齐填充的行。
            hidden_states = sp_all_gather(hidden_states)
            hidden_states = hidden_states[: positions.shape[0]]
        # 步骤3：注意力。
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
        )
        if self.use_sequence_parallel:
            hidden_states = sp_reduce_scatter(hidden_states)

        # 步骤3（续）：累加前缀和 —— 块首层从零起算，
        # 否则把注意力输出记为增量（delta 在下个残差点并入）。
        if prefix_sum is None:
            prefix_sum = hidden_states
            prefix_delta = None
        else:
            prefix_delta = hidden_states

        # 步骤4：MLP 前的残差混合（可见块数 = 历史块 + 本层刚写入的块）。
        mlp_valid_blocks = self.prev_valid_blocks + (1 if self.is_block_write_layer else 0)
        hidden_states, prefix_sum, _ = torch.ops._C_ascend.attn_res_fwd(
            prefix_sum,
            prefix_delta,
            block_residual,
            self.mlp_res_proj.weight,
            self.mlp_res_norm.weight,
            self.mlp_res_norm.variance_epsilon,
            mlp_valid_blocks,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.variance_epsilon,
        )
        # 步骤5：MLP（加回前缀和由外层下一残差点完成）。
        hidden_states = self.mlp(hidden_states)
        return prefix_sum, block_residual, hidden_states


class AscendKimiLinearModel(UpstreamKimiLinearModel):
    """Kimi text model assembled from the Ascend decoder layer."""
    """（Kimi 文本模型主体 —— 用昇腾解码层组装。）

    职责：词嵌入、解码层堆叠（make_layers 按 PP 切分）、最终 norm、
    注意力残差块的缓冲管理、SP/PP 并行、DSpark aux 隐状态捕获。
    """

    # The Ascend forward path carries cumulative auxiliary states across PP
    # via IntermediateTensors (see pp_transport.PPTransportDataType).
    # （昇腾前向通过 IntermediateTensors 跨 PP 携带累计 aux 状态。）
    supports_aux_hidden_states_over_pp = True
    # Upstream's reserve_aux_intermediate_tensor_slots / relay_aux_hidden_states
    # read this via getattr to name and forward the receive-buffer aux slots;
    # it must equal pp_transport's transport key prefix so both sides agree.
    # （该键前缀必须与 pp_transport 的传输键一致，收发两侧才能对上。）
    AUX_HIDDEN_STATE_KEY = "pp_transport_aux_hidden_states_"

    # packed_modules_mapping：声明"融合参数 ↔ 源参数"的映射，
    # 权重加载器据此把 checkpoint 里的多个源权重装配进融合参数。
    # 初始化自 NVIDIA 版（packed）实现的映射表，再补充 KDA 融合投影。
    packed_modules_mapping = {
        name: list(shards) for name, shards in UpstreamPackedKimiLinearModel.packed_modules_mapping.items()
    }
    # fused_bfg_proj：KDA 的 b/f_a/g 三个门投影融合成一个参数
    #（b=beta 门、f_a/g_a=遗忘/输入门增量，Kimi K3 KDA 六投影的一部分）。
    packed_modules_mapping["fused_bfg_proj"] = [
        "b_proj",
        "f_a_proj",
        "g_proj",
    ]
    # Legacy Qwen3 GQA DSpark checkpoints consume the materialized input
    # to each selected Kimi layer. MLA DSpark checkpoints consume the raw
    # prefix-sum stream used by upstream vLLM, so keep that as the default.
    # （旧版 Qwen3 GQA DSpark 消费"物化后"的输入；MLA DSpark 消费上游
    #   vLLM 的原始前缀和流，故默认 False。）
    dspark_aux_capture_materialized = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.vocab_size = config.vocab_size
        # Only the last PP stage loads the drafter. Select its aux contract
        # on every stage before upstream reserves receive buffers/relay slots.
        spec_config = vllm_config.speculative_config
        if spec_config is not None and spec_config.method == "dspark":
            draft_config = spec_config.draft_model_config.hf_config
            self.dspark_aux_capture_materialized = draft_config.model_type == "qwen3" and any(
                arch in ("DSparkDraftModel", "Qwen3DSparkModel") for arch in (draft_config.architectures or ())
            )
        # vLLM's generic MoE SP switch currently requires DP > 1. K3 also
        # needs the same rank-local token layout for the TP/EP, DP=1 topology
        # that FlashComm used before the standard SP operators were available.
        parallel_config = vllm_config.parallel_config
        # Physical expert slots appended by EPLB; load_weights needs the count
        # so the duplicated initial expert placements receive weights.
        # （EPLB 追加的物理专家槽位数量；load_weights 用它给
        #   复制的初始专家放置加载权重。）
        self.n_redundant_experts = (
            parallel_config.eplb_config.num_redundant_experts if parallel_config.enable_eplb else 0
        )
        # （vLLM 通用 MoE SP 开关目前要求 DP>1；K3 在 TP/EP、DP=1 拓扑下
        #   也需要相同的"rank 本地 token"布局，故在此自行判定开启。）
        # SP 开启条件：无 PP（或使用 v2 runner）&& EP 开启 && TP>1。
        self.use_sequence_parallel = (
            (parallel_config.pipeline_parallel_size == 1 or vllm_config.use_v2_model_runner)
            and parallel_config.enable_expert_parallel
            and parallel_config.tensor_parallel_size > 1
        )

        # PP 首卡才有词嵌入（中间卡的该层用 PPMissingLayer 占位，
        # 保持模块树结构完整但不分配显存）。
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def get_layer(prefix: str):
            return AscendKimiDecoderLayer(
                config,
                vllm_config,
                prefix,
                use_sequence_parallel=self.use_sequence_parallel,
            )

        # make_layers：构建层堆叠并按 PP rank 裁剪出本地负责的层区间
        #（start_layer..end_layer），返回 (起始, 结束, ModuleList)。
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )

        # PP 末卡才有最终 norm 与输出残差混合投影。
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.attn_res_block_size is not None:
                # 输出端的残差混合（所有块 + 前缀和参与最终混合）。
                self.output_attn_res_norm = RMSNorm(
                    config.hidden_size,
                    eps=config.rms_norm_eps,
                )
                self.output_attn_res_proj = ReplicatedLinear(
                    config.hidden_size,
                    1,
                    bias=False,
                    quant_config=None,
                    prefix=f"{prefix}.output_attn_res_proj",
                )
        else:
            self.norm = PPMissingLayer()
            if config.attn_res_block_size is not None:
                self.output_attn_res_norm = PPMissingLayer()
                self.output_attn_res_proj = PPMissingLayer()

        # TP 可整除性校验：注意力头数必须能被 TP world size 整除。
        world_size = get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % world_size == 0, "num_attention_heads must be divisible by world_size"

    def _cache_aux_pp_layout(self) -> None:
        # （缓存 aux 槽位的 PP 布局：非首卡调整槽位基址，末卡记录上游总数。）
        super()._cache_aux_pp_layout()
        if (
            self.config.attn_res_block_size is None
            or not self.dspark_aux_capture_materialized
            or not model_parallel_is_initialized()
        ):
            return
        pp = get_pp_group()
        if not pp.is_first_rank:
            # A materialized state at start_layer belongs to this stage,
            # unlike the raw state emitted by the preceding stage.
            self._aux_slot_base_cached = bisect_left(self.aux_hidden_state_layers, self.start_layer)
        if pp.is_last_rank:
            self._aux_upstream_total_cached = self._aux_slot_base_cached

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        if self.config.attn_res_block_size is None:
            return super().make_empty_intermediate_tensors(batch_size, dtype, device)
        # Materialized DSpark states are captured before a layer; raw states
        # are captured after the preceding layer. Handle a PP cut at either.
        # （物化态在层前捕获、原始态在前一层之后捕获；两种 PP 切口都要处理。）
        return make_pp_empty_intermediate_tensors(
            self,
            super().make_empty_intermediate_tensors,
            include_start_layer=not self.dspark_aux_capture_materialized,
        )(batch_size, dtype, device)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor] | tuple[str, torch.Tensor, dict[str, Any]]],
    ) -> set[str]:
        """Reuse upstream loading with Ascend KDA gates and EPLB replicas."""
        """（复用上游加载，把混合精度的 KDA 门权重路由到融合参数加载器，
        并处理 EPLB 冗余专家副本。）

        【原理】fused_bfg_proj 把 b/f_a/g 三个投影融合成一个参数：
        - b/g 走标准融合分片（shard_id 0/2）；
        - f_a 在部分 checkpoint 里叫 f_b_proj 且跨 TP 复制（shard_id=None，
          加载器按"复制式分片"处理，即各 rank 装同一份）。
        remap_mixed_gate_weights 是一个生成器：流式读取权重流，
        命中映射表的改名转发，未命中的原样通过 —— 不物化中间列表，
        内存友好。
        """
        params_dict = dict(self.named_parameters())
        # (源名片段, 目标名片段, 融合分片 id) 三元组表。
        gate_mapping = (
            (".b_proj.weight", ".fused_bfg_proj.weight", 0),
            (".f_a_proj.weight", ".fused_bfg_proj.f_a_weight", None),
            (".f_b_proj.weight", ".fused_bfg_proj.f_b_weight", None),
            (".g_proj.weight", ".fused_bfg_proj.weight", 2),
        )

        def remap_mixed_gate_weights():
            # args 可能是 (name, tensor) 或 (name, tensor, kwargs_dict)。
            for args in weights:
                name, loaded_weight = args[:2]
                # 逐条匹配映射表（for...else：全不匹配才原样透传）。
                for source, target, shard_id in gate_mapping:
                    if source not in name:
                        continue
                    mapped_name = name.replace(source, target)
                    # 只有目标参数真实存在才重映射（防误伤其他模块的同名片段）。
                    if mapped_name in params_dict:
                        kwargs = dict(args[2]) if len(args) > 2 else {}
                        kwargs["loaded_shard_id"] = shard_id
                        yield mapped_name, loaded_weight, kwargs
                        break
                else:
                    yield args

        return super().load_weights(
            load_eplb_expert_weights(self, remap_mixed_gate_weights(), self.n_redundant_experts)
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        """文本主干前向（含注意力残差块路径）。

        返回值三种形态：
        - PP 中间卡：IntermediateTensors（hidden_states + residual/block 形式）；
        - 普通路径：最终隐状态 [N, H]；
        - DSpark aux 捕获开启时：(隐状态, aux 隐状态列表)。
        """
        # 无注意力残差配置时直接走上游通用前向。
        if self.config.attn_res_block_size is None:
            return super().forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

        # 记录全量 token 数（SP 裁剪后需要用它在结尾 gather 时截断）。
        full_num_tokens = positions.shape[0]
        # PP 边界处理：首卡从嵌入起步，中间卡从上一段的中间张量恢复。
        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        # DSpark aux 捕获模式（物化/原始前缀和）；aux 隐状态经 PP 传输随中间张量到达。
        materialized_aux = self.dspark_aux_capture_materialized
        aux_hidden_states = get_pp_transport_tensors(intermediate_tensors, PPTransportDataType.AUX_HIDDEN_STATES)
        if self.use_sequence_parallel:
            # 可选的填充掩码优化：标记哪些行是 SP 对齐填充的假 token，
            # 供 MoE 内核跳过（省计算）。
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                # Every stage starts with a full-token padding mask, even
                # though only the first stage starts with full activations.
                forward_context.is_padding = sp_padding_mask(
                    forward_context.is_padding,
                    positions,
                )
            # Every stage starts from full-sequence tensors - the previous
            # stage closed its sequence-parallel region at the boundary -
            # and keeps its own shard afterwards, matching DeepSeek V4.
            # （SP：切成本 rank 的 token 分片；residual 与 aux 同步切片。）
            hidden_states = sp_shard(hidden_states)
            if residual is not None:
                residual = sp_shard(residual)
            aux_hidden_states = [sp_shard(aux) for aux in aux_hidden_states]
        # DSpark aux 隐状态捕获：物化模式在层间收集，原始模式在（首卡）层前收集。
        if not materialized_aux and get_pp_group().is_first_rank:
            aux_hidden_states = self._maybe_add_hidden_state(
                aux_hidden_states,
                self.start_layer,
                hidden_states,
                residual,
            )
        # 注意力残差缓冲：[N, num_blocks, H]，new_empty 不初始化
        #（未读的块会被块首层覆写，初始化是浪费）。
        attn_res_block_num = cdiv(
            self.end_layer,
            self.config.attn_res_block_size,
        )
        block_residual = hidden_states.new_empty(
            hidden_states.size(0),
            attn_res_block_num,
            hidden_states.size(1),
        )
        # PP 中间卡进入时：把上一段传来的残差填进块缓冲前几块。
        if residual is not None:
            block_residual[:, : residual.size(1), :].copy_(residual)
        residual = block_residual

        prefix_delta: torch.Tensor | None = None
        # 主循环：逐层前向（enumerate 的 start 参数让 layer_idx 与全局层号对齐）。
        for layer_idx, layer in enumerate(
            self.layers[self.start_layer : self.end_layer],
            start=self.start_layer,
        ):
            if materialized_aux and layer_idx in self.aux_hidden_state_layers:
                # Capture the layer input before the layer consumes it. This
                # fused call has no bank writeback, so it has no side effects
                # on the residual state.
                # （物化模式的 aux 捕获：在层消费前捕获其输入；
                #   该融合调用不回写块缓冲，对残差状态无副作用。）
                prefix_input = hidden_states + prefix_delta if prefix_delta is not None else hidden_states
                aux_hidden_states.append(
                    torch.ops._C_ascend.attn_res_fwd(
                        prefix_input,
                        None,
                        residual,
                        layer.self_attention_res_proj.weight,
                        layer.self_attention_res_norm.weight,
                        layer.self_attention_res_norm.variance_epsilon,
                        layer.prev_valid_blocks,
                        return_materialized=True,
                    )[2]
                )
            hidden_states, residual, prefix_delta = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                prefix_delta=prefix_delta,
            )
            # 原始模式的 aux 捕获：记录指定层的输出（前缀和/块缓冲）。
            if not materialized_aux and (layer_idx + 1) in self.aux_hidden_state_layers:
                self._maybe_add_hidden_state(
                    aux_hidden_states,
                    layer_idx + 1,
                    hidden_states + prefix_delta if prefix_delta is not None else hidden_states,
                    residual,
                )

        if not get_pp_group().is_last_rank and prefix_delta is not None:
            hidden_states = hidden_states + prefix_delta
        # PP 中间卡：把 hidden + 块缓冲（连同 aux）打包传给下一段。
        if not get_pp_group().is_last_rank:
            if self.use_sequence_parallel:
                # The next PP rank expects full-sequence tensors; close the
                # sequence-parallel region before crossing the boundary so the
                # upstream PP transport only sees replicated tensors.
                hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
                residual = sp_all_gather(residual)[:full_num_tokens]
                aux_hidden_states = [sp_all_gather(aux)[:full_num_tokens] for aux in aux_hidden_states]
            return add_pp_transport_tensors(
                IntermediateTensors({"hidden_states": hidden_states, "residual": residual}),
                PPTransportDataType.AUX_HIDDEN_STATES,
                aux_hidden_states,
            )

        # The loop above already captured aux for every ``(layer_idx + 1)``
        # in ``aux_hidden_state_layers``, including ``end_layer``; upstream
        # vLLM main does not add a boundary/final aux slot on top of it.
        # （末卡收尾：输出端残差混合 —— 全部块 + 前缀和。）
        hidden_states, _, _ = torch.ops._C_ascend.attn_res_fwd(
            hidden_states,
            prefix_delta,
            residual,
            self.output_attn_res_proj.weight,
            self.output_attn_res_norm.weight,
            self.output_attn_res_norm.variance_epsilon,
            attn_res_block_num,
        )
        if materialized_aux and self.end_layer in self.aux_hidden_state_layers:
            aux_hidden_states.append(hidden_states)
        if self.use_sequence_parallel:
            # SP 收尾：gather 回全序列。有 aux 时把它们与主隐状态在最后一维
            # 拼成一个张量一次 gather（减少通信次数），gather 后再 split 拆回。
            if aux_hidden_states:
                hidden_size = hidden_states.shape[-1]
                packed_hidden_states = torch.cat(
                    [hidden_states, *aux_hidden_states],
                    dim=-1,
                )
                packed_hidden_states = sp_all_gather(packed_hidden_states)
                # 截断到真实 token 数（去掉 SP 对齐填充的行）。
                packed_hidden_states = packed_hidden_states[:full_num_tokens]
                hidden_states, *aux_hidden_states = packed_hidden_states.split(
                    hidden_size,
                    dim=-1,
                )
            else:
                hidden_states = sp_all_gather(hidden_states)
                hidden_states = hidden_states[:full_num_tokens]
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class KimiMixtureOfExperts(MixtureOfExperts):
    """MoE bookkeeping shared by the Ascend Kimi K3 adapters."""

    moe_mlp_layers: list[AscendKimiMoE]

    def extract_moe_parameters(self) -> None:
        self.expert_weights: list[Sequence[torch.Tensor]] = []
        self.num_expert_groups = getattr(self.config, "num_expert_group", None) or 1
        self.moe_layers = [moe.experts for moe in self.moe_mlp_layers]
        # EPLB state and transfers operate within this PP stage.
        self.num_moe_layers = len(self.moe_layers)
        example_moe = self.moe_mlp_layers[-1] if self.moe_mlp_layers else None
        if example_moe is None:
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_shared_experts = 0
            self.num_redundant_experts = 0
        else:
            self.num_logical_experts = example_moe.n_logical_experts
            self.num_physical_experts = example_moe.n_physical_experts
            self.num_local_physical_experts = example_moe.n_local_physical_experts
            self.num_routed_experts = example_moe.n_routed_experts
            self.num_shared_experts = example_moe.num_shared_experts or 0
            self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        # Rebalancing and elastic EP reuse the allocated per-rank weight slots.
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_physical_experts = num_physical_experts
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()


class AscendKimiLinearForCausalLM(UpstreamKimiLinearForCausalLM, KimiMixtureOfExperts):
    """Causal-LM wrapper retaining vLLM 0.27 state/cache interfaces."""
    """（Causal-LM 顶层包装 —— 保留 vLLM 0.27 的状态/缓存接口。）

    注册名 "KimiLinearForCausalLM" / "KimiK3ForCausalLM"（兼容旧名）。
    组件：模型主体 + lm_head（PP 末卡）+ logits 处理器。
    """

    # 类属性引用模型的融合映射（同一份配置两处使用）。
    packed_modules_mapping = AscendKimiLinearModel.packed_modules_mapping

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.config = self.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = AscendKimiLinearModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        # lm_head 只在 PP 末卡存在。
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        # logit_scale：可选的输出缩放（部分 Kimi checkpoint 使用）。
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size,
            scale=getattr(self.config, "logit_scale", 1.0),
        )
        # Set MoE hyperparameters for the EPLB registration.
        self.set_moe_parameters()

    def set_moe_parameters(self) -> None:
        self.moe_mlp_layers = []
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            assert isinstance(layer, AscendKimiDecoderLayer)
            if isinstance(layer.mlp, AscendKimiMoE):
                self.moe_mlp_layers.append(layer.mlp)
        self.extract_moe_parameters()

    def set_dspark_aux_capture_materialized(self, enabled: bool) -> None:
        # DSpark 草稿的接口：切换 aux 隐状态捕获模式（物化/原始前缀和）。
        self.model.dspark_aux_capture_materialized = enabled


class AscendKimiK3MultiModalProjector(KimiK25MultiModalProjector):
    """Kimi projector with the optional ModelSlim output rotation."""
    """（Kimi 多模态投影器 —— 带可选的 ModelSlim 输出旋转。）

    把视觉塔输出的特征投影到文本 hidden 空间；当目标模型使用
    QuaRot 旋转量化（rotation_path 非空）时，额外加一个 rot_proj
    把视觉嵌入旋到与文本一致的隐空间基。
    """

    def __init__(
        self,
        config,
        *args,
        prefix: str = "",
        enable_rotation: bool = False,
        **kwargs,
    ) -> None:
        # *args/**kwargs 透传：兼容上游基类构造签名的变化。
        super().__init__(config, *args, prefix=prefix, **kwargs)
        self.rot_proj: ReplicatedLinear | None = None
        if enable_rotation:
            # 输出旋转：text_hidden_size → text_hidden_size 的方阵投影。
            output_size = config.text_hidden_size
            self.rot_proj = ReplicatedLinear(
                output_size,
                output_size,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.rot_proj",
            )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """视觉特征 → 文本空间嵌入；启用旋转时再过 rot_proj。"""
        hidden_states = super().forward(image_features)
        rot_proj = self.rot_proj
        if rot_proj is not None:
            # [0] 取输出元组的张量（ReplicatedLinear 返回 (out, bias)）。
            hidden_states = rot_proj(hidden_states)[0]
        return hidden_states


# 装饰器：向 vLLM 多模态注册表登记处理器三件套
#（处理器类 / ProcessingInfo / DummyInputsBuilder —— 后者为显存探测
#  生成虚拟输入，决定多模态场景下的 KV cache 预算）。
@MULTIMODAL_REGISTRY.register_processor(
    KimiK3MultiModalProcessor,
    info=KimiK3ProcessingInfo,
    dummy_inputs=KimiK3DummyInputsBuilder,
)
class AscendKimiK3ForConditionalGeneration(UpstreamKimiK3ForConditionalGeneration):
    """Upstream Kimi K3 multimodal wrapper with Ascend text/projector layers."""
    """（Kimi K3 多模态顶层包装 —— 上游包装器 + 昇腾文本/投影层。）

    注册名 "KimiK3ForConditionalGeneration"。组合：
    - vision_tower：MoonViT 3D 视觉塔（复用上游，可选数据并行 DP 复制）；
    - mm_projector：上面的投影器（含可选 QuaRot 旋转）；
    - language_model：通过 init_vllm_registered_model 递归构建
      AscendKimiLinearForCausalLM（按注册表解析，而非硬编码导入）。
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        model_config = vllm_config.model_config
        self.config = model_config.hf_config
        self.quant_config = vllm_config.quant_config
        multimodal_config = model_config.multimodal_config
        assert multimodal_config is not None

        # 视觉塔数据并行判定：注意力头少到 TP 切分不划算时，
        # 改用"每 rank 完整复制 + 按图轮转"的 DP 模式。
        self.use_data_parallel = is_vit_use_data_parallel(
            self.config.vision_config.num_attention_heads,
        )
        self.hidden_size = self.config.text_config.hidden_size
        self.device = current_platform.current_device()
        # 视觉塔通常保持 BF16 不量化（量化收益小、精度风险高）。
        vision_quant_config = self._maybe_ignore_quant_config(self.quant_config)

        # _mark_tower_model 上下文：向编译/注册系统标记"这是视觉塔"，
        # 让 TP/DSP 等并行策略对其特殊处理。
        with self._mark_tower_model(vllm_config, "image"):
            self.vision_tower = MoonViT3dPretrainedModel(
                self.config.vision_config,
                quant_config=vision_quant_config,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            # 显式搬运到 NPU 设备（量化时保持原 dtype，否则用模型 dtype）。
            if vision_quant_config is not None:
                self.vision_tower = self.vision_tower.to(device=self.device)
            else:
                self.vision_tower = self.vision_tower.to(
                    device=self.device,
                    dtype=model_config.dtype,
                )

            self.mm_projector = AscendKimiK3MultiModalProjector(
                self.config.vision_config,
                use_data_parallel=self.use_data_parallel,
                quant_config=vision_quant_config,
                prefix=maybe_prefix(prefix, "mm_projector"),
                # QuaRot 旋转配置存在时启用投影器输出旋转。
                enable_rotation=get_rotation_path(vllm_config) is not None,
            )
        if vision_quant_config is not None:
            self.mm_projector = self.mm_projector.to(device=self.device)
        else:
            self.mm_projector = self.mm_projector.to(
                device=self.device,
                dtype=model_config.dtype,
            )

        # 语言主干：按注册表构建（architectures 指定 KimiLinearForCausalLM，
        # 实际解析到本文件的 AscendKimiLinearForCausalLM）。
        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=self.config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["KimiLinearForCausalLM"],
            )
        # 把语言模型的 PP 中间张量构造器提升到本层（方法别名）。
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.language_model.make_empty_intermediate_tensors
        )
        # 媒体占位 token id：prompt 中该位置会被视觉嵌入替换。
        self.media_placeholder = self.config.media_placeholder_token_id

    def get_language_model(self) -> AscendKimiLinearForCausalLM:
        return self.language_model

    def set_dspark_aux_capture_materialized(self, enabled: bool) -> None:
        # DSpark 接口透传给语言主干。
        self.language_model.set_dspark_aux_capture_materialized(enabled)
