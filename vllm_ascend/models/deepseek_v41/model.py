# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# 【文件职责】DeepSeek V4.1 文本模型与"源共享混合 cache 图"的 Ascend 核心实现。
#
# 【类清单与结构】
#   DeepseekV41MLP               稠密 SwiGLU MLP（共享专家等）
#   DeepseekV41MoE               MoE 层：路由/共享专家/hash 路由/EPLB
#   DeepseekV41MixtureOfExperts  MoE 元数据混入类（继承 vLLM MixtureOfExperts）
#   init_attention_projections   函数式初始化 MLA 注意力投影（混入多个类）
#   DeepseekV41LayerRole         单层角色（frozen dataclass）
#   DeepseekV41Topology          全模型"源/消费"拓扑（frozen dataclass）
#   DeepseekV41SharedAttentionState 一次前向内 topk/候选块跨层交接
#   build_layer_plan             从配置构建并校验拓扑
#   AscendDeepseekV41SWACache    滑窗注意力 cache 层（继承 DeepseekV41CacheLayer）
#   DeepseekV41SWAAttention      SWA 注意力（目标+草稿共用投影结构）
#   DeepseekV41Attention         V4.1 注意力（cache 所有权判定 + 源共享）
#   DeepseekV41DecoderLayer      解码块（mHC + 注意力 + MoE + engram）
#   DeepseekV41Model             文本主干（继承 EagleModelMixin，支持 PP）
#   AscendDeepseekV41LLMForCausalLM 顶层因果 LM（权重加载重命名）
#
# 【核心原理】
# - MLA 多头潜在注意力: Q 侧低秩压缩（wq_a→q_norm→wq_b，q_lora_rank），
#   KV 压成单潜向量（wkv→kv_norm，head_dim），输出按 o_groups 组做低秩
#   分解（wo_a→wo_b，o_lora_rank）。KV cache 只需存 head_dim 潜向量。
# - 源共享混合 cache: 配置里的 kv_source_layer_ids/index_source_layer_ids
#   指定少数"源层"物理拥有全上下文 KV 与 indexer K cache；压缩层
#   (compress_ratio>0) 与滑窗层(SWA)复用源层 cache，大幅省显存。
# - 稀疏注意力(DSA): indexer 选 TopK token 位置（见 indexer.py），
#   主注意力只算这些位置；候选块机制跨层逐级缩小搜索空间。
# - mHC 矩阵超连接: 残差流复制 hc_mult 份；每层前后用可学习混合矩阵
#   （Sinkhorn 归一化保持 doubly-stochastic）重组流；最终 hc_collapse 折叠。
# - MoE: gate 路由 + 可选 hash 路由(tid2eid，按 token ID 直接查专家) +
#   共享专家 + EPLB 冗余专家均衡。
# - engram 记忆: n-gram 哈希 → INT8 嵌入表检索 → 门控写回残差（engram/ 子包）。
#
# 【NPU 适配点】
# - 注意力前向走 torch.ops.vllm.dsa_v41_forward 自定义算子，经
#   static_forward_context 按层名分发（ACL Graph 友好，地址固定）；
# - mHC 前后处理走 npu_hc_pre_v3 / npu_hc_post 融合算子；
# - wo_a 权重保持 ND 格式（skip_weight_nz_conversion），因为 DSA 的
#   o_proj 用 npu_transpose_batchmatmul 直接消费该权重；
# - 序列并行（sp_shard/all_gather/reduce_scatter）在 MoE 场景切 token 维。
# =============================================================================
"""DeepSeek V4.1 text model and source-shared hybrid-cache graph."""

from __future__ import annotations

import typing
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import vllm.envs as envs
from safetensors import safe_open
from torch import nn
from transformers import PretrainedConfig
from vllm.config import ParallelConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader, maybe_remap_kv_scale_name
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from vllm.model_executor.models.utils import PPMissingLayer, is_pp_missing_parameter, make_layers, maybe_prefix

# Upstream #56741 normalized the V4.1 model package name.
from vllm.models.deepseek_v41.common.engram import EngramLayout
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.dsa_v41 import (
    DeepseekV41CacheLayer,
)
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSlidingWindowMLASpec
from vllm_ascend.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_reduce_scatter,
    sp_shard,
)
from vllm_ascend.ops.dsa import AscendDeepseekSparseAttention, DSAModules
from vllm_ascend.ops.rope_dsv4 import ComplexExpRotaryEmbedding
from vllm_ascend.ops.triton.mul_add import muls_add_triton
from vllm_ascend.utils import (
    enable_custom_op,
    enable_dsa_cp,
    normalize_deepseek_v41_config,
)

from .compressor import DeepseekV41Compressor
from .engram import (
    create_engram_hash_state,
    engram_cpu_offload,
    engram_dead_mask,
    engram_enabled,
)
from .engram.embedding import (
    AscendParallelEngramEmbedding,
    preflight_engram_checkpoint,
)
from .engram.layer import AscendEngram
from .engram.parallel import gather_engram_hashes, get_engram_dp_size
from .indexer import DeepseekV41Indexer


class DeepseekV41MLP(nn.Module):
    """稠密 SwiGLU MLP 前馈层（供 MoE 的 shared_experts 使用）。

    结构: down_proj(SiluAndMul(gate_up_proj(x)))。gate_up_proj 是把 gate
    和 up 两个线性层融合成的单个 MergedColumnParallelLinear（输出 2 倍
    intermediate_size），减少 kernel 启动次数。
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        swiglu_limit: float | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel=False,
        prefix: str = "",
    ) -> None:
        """初始化稠密 MLP。

        参数:
            hidden_size: 输入/输出维度。
            intermediate_size: 中间维度（gate/up 各占一份）。
            hidden_act: 激活函数名（保留接口参数）。
            swiglu_limit: 可选的 SwiGLU 输出钳制上限（V4.1 特有）。
            quant_config: 量化配置。
            reduce_results: RowParallel 输出是否做 all-reduce（False 时把
                归约留给上层，如共享专家与路由专家结果相加后再归约）。
            is_sequence_parallel: 序列并行模式——输入输出按 TP 切分、权重
                复制，免除集合通信。
            prefix: 参数名前缀。
        """
        super().__init__()

        # If is_sequence_parallel, the input and output tensors are sharded
        # across the ranks within the tp_group. In this case the weights are
        # replicated and no collective ops are needed.
        # Otherwise we use standard TP with an allreduce at the end.
        # 【中文】序列并行时 disable_tp=True——权重不切分（复制），靠输入
        # 输出已按序列切分来并行；普通 TP 则按列/行切分权重并归约。
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
        # swiglu_limit 存在时用带钳制的 SiLU·mul 变体（限制门控幅值，
        # 提升低精度数值稳定性）。
        if swiglu_limit is not None:
            self.act_fn = SiluAndMulWithClamp(swiglu_limit)
        else:
            self.act_fn = SiluAndMul()

    def forward(self, x):
        """x: [tokens, hidden] → [tokens, hidden]。步骤: 融合投影 → SwiGLU → 降维。"""
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class DeepseekV41MoE(nn.Module):
    """V4.1 MoE 层：路由门控 + 可选 hash 直连路由 + 共享专家 + EPLB 支持。

    路由方式（互斥）:
        - hash 层（layer_idx < num_hash_layers 且非草稿层）: 直接用
          tid2eid 查表把 token ID 映射到专家，不走 softmax 打分；
        - 常规层: gate 线性层打分 + e_score_correction_bias 修正 +
          top-k 选择（norm_topk_prob 重归一化）。
    多模态变体: gate.bias_vl 是仅视觉 token 使用的路由偏置（文本 token
    走 tid2eid 或常规偏置），配合 image_sentinel_lo 区分图像哨兵 ID 段。
    EPLB（专家并行负载均衡）: 支持冗余物理专家，n_physical = 逻辑专家 +
    冗余专家，按 EP rank 均分驻留。
    """

    def __init__(
        self,
        config: PretrainedConfig,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        is_draft_layer: bool = False,
    ):
        """初始化 MoE 层。

        参数:
            config: V4.1 文本配置。
            parallel_config: 并行配置（EPLB/序列并行开关）。
            quant_config: 量化配置。
            prefix: 参数名前缀（最后一段数字被解析为层号）。
            is_draft_layer: 是否为 DSpark 草稿层（草稿层禁用 hash 路由）。
        """
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        # prefix 形如 "model.layers.5.mlp"，倒数第二段是层号。
        layer_idx = int(prefix.split(sep=".")[-2])
        self.layer_idx = layer_idx
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.5)
        self.swiglu_limit = getattr(config, "swiglu_limit", None)

        # EP（专家并行）分组信息：专家按 EP rank 划分驻留。
        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = config.n_shared_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        # 路由门控: hidden → n_routed_experts logits。权重复制（不切分），
        # 保证每个 rank 算出一致的路由决策。
        self.gate = ReplicatedLinear(
            config.hidden_size, config.n_routed_experts, bias=False, quant_config=None, prefix=f"{prefix}.gate"
        )
        # 语法点: 动态属性 precast_fp32_weight——提示加载器把门控权重
        # 预转换为 FP32（路由打分对精度敏感）。
        self.gate.precast_fp32_weight = True

        # Load balancing settings.
        # 【中文】EPLB 负载均衡设置: 逻辑专家数 + 冗余专家数 = 物理专家数。
        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        # 本 EP rank 驻留的物理专家区间。
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = self.physical_expert_start + self.n_local_physical_experts

        # mix_placement（Ascend 配置）: 共享专家与路由专家融合进同一个
        # FusedMoE kernel（作为附加专家槽位），此时独立的 shared_experts
        # 模块不再创建。
        self.is_fusion_moe_shared_experts_enabled = getattr(get_ascend_config(), "mix_placement", False)
        if config.n_shared_experts is None or self.is_fusion_moe_shared_experts_enabled:
            self.shared_experts = None
        else:
            # 独立共享专家: 一个加宽的稠密 MLP（intermediate = 单专家 × 数量），
            # reduce_results=False（与路由专家结果相加后再统一归约/缩放）。
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts

            self.shared_experts = DeepseekV41MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                swiglu_limit=self.swiglu_limit,
                quant_config=quant_config,
                is_sequence_parallel=self.is_sequence_parallel,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        # hash 路由判定: 层号 < num_hash_layers 且非草稿层。
        self.hash = layer_idx < config.num_hash_layers and not is_draft_layer
        self.gate.bias_vl = None
        if getattr(config, "vision_n_layers", 0) > 0:
            # 多模态模型: 视觉 token 的路由偏置（每个专家一个 FP32 标量）。
            self.gate.bias_vl = nn.Parameter(
                torch.empty(
                    config.n_routed_experts,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
        if self.hash:
            # Use zeros instead of empty to avoid garbage values causing
            # invalid memory access in dummy mode (--load-format="dummy")
            # 【中文】tid2eid: [vocab_size, num_experts_per_tok] 的查表——
            # token ID 直接映射到 top-k 专家组合。用零初始化而非空值，
            # 避免 dummy 模式下垃圾值引发非法内存访问。
            self.gate.tid2eid = nn.Parameter(
                torch.zeros(
                    config.vocab_size,
                    config.num_experts_per_tok,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
            # hash 路由层不需要 softmax 修正偏置（路由不走打分）。
            self.gate.e_score_correction_bias = None
        else:
            self.gate.tid2eid = None
            # 常规路由: 专家分数修正偏置（DeepSeek V3 式 noaux_tc 路由）。
            self.gate.e_score_correction_bias = nn.Parameter(torch.empty(config.n_routed_experts, dtype=torch.float32))

        # FusedMoEFactory: 构建融合 MoE 算子（路由+专家计算融合）。
        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func=getattr(config, "scoring_func", "softmax"),
            # Keep scaling outside the router path so the order matches
            # DeepSeek V4: normalize top-k weights, then scale routed output.
            # AITER applies routed_scaling_factor internally.
            # 【中文】缩放放在路由路径之外: 先归一化 top-k 权重、再对路由
            # 输出乘 routed_scaling_factor——顺序必须与 V4 一致。
            routed_scaling_factor=self.routed_scaling_factor,
            swiglu_limit=self.swiglu_limit,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            bias_vl=self.gate.bias_vl,
            # 图像哨兵 ID 段起点（低于此 ID 段的 token 被视为图像 token，
            # 路由时用 bias_vl 而非文本偏置）。
            image_sentinel_lo=getattr(config, "image_sentinel_base_id", 129257),
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            # 融合模式下共享专家作为附加专家槽位传入。
            n_shared_experts=config.n_shared_experts if self.is_fusion_moe_shared_experts_enabled else 0,
            hash_indices_table=self.gate.tid2eid,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        hidden_states_fp32: torch.Tensor | None = None,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        """MoE 前向。

        参数:
            hidden_states: [tokens, hidden] 输入（可能是 hc 多流的折叠输入）。
            input_ids: [tokens] 原始 token ID（hash 路由/视觉偏置需要）。
            hidden_states_fp32: [tokens, hidden] FP32 版输入（路由打分精度），
                缺省时内部由 hidden_states 转 FP32。
            already_sequence_parallel: 上层已完成 SP 切分则跳过再切。
        返回:
            [tokens, hidden] 共享专家与路由专家加权和。
        步骤:
            1) 展平 + 可选 SP 切片（token 维按 TP 均分避免重复计算）；
            2) 路由: 内置路由（is_internal_router）或显式 F.linear 打分；
            3) 融合 MoE 计算专家输出；
            4) 共享专家结果融合 + routed_scaling_factor 缩放
               （muls_add_triton 融合算子一次完成乘加）；
            5) SP all_gather 恢复 / TP 归约。
        """
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if hidden_states_fp32 is not None:
            hidden_states_fp32 = hidden_states_fp32.view(-1, hidden_dim)
        # Chunk the hidden states so they aren't replicated across TP ranks.
        # This avoids duplicate computation in self.experts.
        # TODO: We can replace the all_reduce at the end of attn with a
        # reduce_scatter instead of chunking here.
        # 【中文】步骤1: SP 模式下把 token 切到各 rank（否则每 rank 都对
        # 全量 token 计算专家，浪费算力）。
        if self.is_sequence_parallel and not already_sequence_parallel:
            hidden_states = sp_shard(hidden_states)
            if hidden_states_fp32 is not None:
                hidden_states_fp32 = sp_shard(hidden_states_fp32)

        if self.experts.is_internal_router:
            # In this case, the gate/router runs inside the FusedMoEFactory class
            # 【中文】步骤2a: 融合路由——路由器在 FusedMoE 内部运行，
            # router_input 直接给原始输入（FP32 优先）。
            router_input = hidden_states if hidden_states_fp32 is None else hidden_states_fp32
            fused_moe_out = self.experts(
                hidden_states=hidden_states,
                router_logits=router_input,
                input_ids=input_ids,
            )
        else:
            # 步骤2b: 显式路由——FP32 打分（softmax 路由精度要求）后把
            # logits 交给融合 MoE。
            # router_logits: (num_tokens, n_experts)
            router_input = hidden_states.float() if hidden_states_fp32 is None else hidden_states_fp32
            router_logits = F.linear(router_input, self.gate.weight)
            fused_moe_out = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                input_ids=input_ids,
            )

        # 步骤4: 结果融合。tuple 形态 = (shared_output, routed_output)；
        # 单张量形态 = 已在内部融合完毕。
        fused_moe_out_is_tuple = isinstance(fused_moe_out, tuple)
        if fused_moe_out_is_tuple:
            shared_output, final_hidden_states = fused_moe_out
            if self.shared_experts is None:
                assert shared_output is None

            if hidden_states.dtype != torch.float16:
                # 非 FP16: routed * scaling + shared（一个 Triton 融合乘加）。
                if self.shared_experts is not None:
                    final_hidden_states = muls_add_triton(
                        final_hidden_states, shared_output, self.routed_scaling_factor
                    )
                else:
                    final_hidden_states *= self.routed_scaling_factor
            elif self.shared_experts is not None:
                # FP16 溢出风险: 改为 shared + routed/scaling，把大数放在加数侧。
                final_hidden_states = muls_add_triton(
                    shared_output, final_hidden_states, 1.0 / self.routed_scaling_factor
                )
        else:
            final_hidden_states = fused_moe_out

        # 步骤5: 通信收尾——SP 恢复全量 token；或 legacy tuple 输出的 TP 归约
        # （上游 MoERunner 的张量输出已在其内部完成最终归约）。
        if self.is_sequence_parallel and not already_sequence_parallel:
            final_hidden_states = sp_all_gather(final_hidden_states)
            final_hidden_states = final_hidden_states[:num_tokens]
        elif self.tp_size > 1 and fused_moe_out_is_tuple:
            # Legacy tuple outputs are reduced here. Tensor outputs from the
            # upstream MoERunner have already gone through its final reduction.
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)


def get_spec_layer_idx_from_weight_name(config: PretrainedConfig, weight_name: str) -> int | None:
    """判断权重名是否属于投机解码(mtp)层。mtp.* 返回 0（有 spec 层标记），
    其余返回 None（目标模型权重）。"""
    if weight_name.startswith("mtp."):
        return 0
    return None


class DeepseekV41MixtureOfExperts(MixtureOfExperts):
    moe_mlp_layers: list[DeepseekV41MoE]
    """
    List of MoE MLP layers in the model.
    """

    # 【中文】MoE 元数据混入类（继承 vLLM 的 MixtureOfExperts 接口）：
    # 为顶层模型（目标 LM 与 DSpark 草稿）提供 EPLB 所需的专家统计信息。

    def extract_moe_parameters(self, example_moe: DeepseekV41MoE | None):
        """从样例 MoE 层提取专家统计；样例为 None（无 MoE 层）时全部清零。"""
        if example_moe is None:
            self.num_moe_layers = 0
            self.num_expert_groups = 0
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
            self.num_shared_experts = example_moe.n_shared_experts
            self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        """EPLB 重平衡后更新物理专家元数据并刷新专家映射表。

        原理: EPLB 会动态改变每个 rank 驻留的物理专家数；此方法把新的
        物理专家布局同步到自身与所有 MoE 层，再调用 update_expert_map
        重建逻辑→物理专家映射。
        """
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()


def init_attention_projections(self, config, quant_config, prefix, reduce_results):
    """函数式初始化 MLA 注意力投影（被多个注意力类"混入"调用）。

    原理（MLA 多头潜在注意力，V4.1 变体）:
        Query 侧: hidden --wq_a--> q_lora_rank（低秩压缩）→ q_norm →
                  --wq_b--> n_heads × head_dim（升维出各头 query）；
        KV 侧:    hidden --wkv--> head_dim（单潜向量，所有头共享）→ kv_norm。
                  KV cache 只需缓存该 head_dim 潜向量（ MLA 的核心省显存点）；
        Output 侧: 注意力输出按 o_groups 组分组，先 wo_a 压到
                  o_groups × o_lora_rank，再 wo_b 升回 hidden（分组低秩分解）。
    语法点: 这里的 self 不是本函数所在模块，而是调用方类实例——典型的
        "函数混入"（mixin function）写法，让 SWA/长上下文注意力共享投影定义。
    参数:
        config: V4.1 文本配置。
        quant_config: 量化配置。
        prefix: 参数名前缀。
        reduce_results: wo_b 是否做输出归约。
    """
    tp_size = get_tensor_model_parallel_world_size()
    self.dim = config.hidden_size
    self.n_heads = config.num_attention_heads
    self.n_local_heads = config.num_attention_heads // tp_size
    # q_lora/o_lora: query/输出侧低秩瓶颈维度。
    self.q_lora_rank = config.q_lora_rank
    self.o_lora_rank = config.o_lora_rank
    # head_dim: MLA 潜向量总维度 = nope(非旋转) + rope(旋转) 两部分。
    self.head_dim = config.head_dim
    self.rope_head_dim = config.qk_rope_head_dim
    self.nope_head_dim = config.head_dim - config.qk_rope_head_dim
    # o_proj 分组数（输出侧分组低秩分解的组数）。
    self.n_groups = config.o_groups
    self.n_local_groups = self.n_groups // tp_size
    # 滑窗注意力的窗口大小。
    self.window_size = config.sliding_window
    self.eps = config.rms_norm_eps
    self.norm_eps = config.rms_norm_eps
    self.scale = self.head_dim**-0.5
    # enable_dsa_cp: DSA 上下文并行开关（环境变量 VLLM_ASCEND_* 控制，见
    # vllm_ascend/utils.py）——开启后序列维被切分到多个 CP rank。
    self.enable_dsa_cp = enable_dsa_cp()

    # attn_sink: 注意力"汇聚"偏置（每个头一个标量），softmax 前加在分数上，
    # 用于缓解长序列注意力稀释。CP 模式每 rank 需要全部头，否则按 TP 切片。
    attn_sink_heads = self.n_heads if self.enable_dsa_cp else self.n_local_heads
    self.attn_sink = nn.Parameter(torch.empty(attn_sink_heads, dtype=torch.float32))
    # wq_a: hidden → q_lora_rank 的降维投影（复制的，不切分）。
    self.wq_a = ReplicatedLinear(
        self.dim,
        self.q_lora_rank,
        bias=False,
        quant_config=quant_config,
        prefix=f"{prefix}.wq_a",
        return_bias=False,
    )
    self.q_norm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
    # q_norm_without_weight: 无权重 RMSNorm（head_dim 维），DSA 算子内部
    # 对 query head 做归一时使用。
    self.q_norm_without_weight = RMSNorm(self.head_dim, eps=config.rms_norm_eps, has_weight=False)
    # wq_b: q_lora_rank → n_heads*head_dim 升维。CP 模式下复制（各 rank
    # 有全部头的数据），普通模式按列切分（每 rank 只算本地头）。
    wq_b_cls = ReplicatedLinear if self.enable_dsa_cp else ColumnParallelLinear
    self.wq_b = wq_b_cls(
        self.q_lora_rank,
        self.n_heads * self.head_dim,
        bias=False,
        quant_config=quant_config,
        prefix=f"{prefix}.wq_b",
        return_bias=False,
    )

    # wkv: hidden → head_dim 的 KV 潜向量投影（MLA 单潜向量设计，
    # K 与 V 共享同一潜空间，cache 只存这一个向量）。
    self.wkv = ReplicatedLinear(
        self.dim,
        self.head_dim,
        bias=False,
        quant_config=quant_config,
        prefix=f"{prefix}.wkv",
        return_bias=False,
    )
    self.kv_norm = RMSNorm(self.head_dim, self.norm_eps)
    # wo_a: 注意力输出 → (o_groups × o_lora_rank) 的分组降维（按列切分）。
    self.wo_a = ColumnParallelLinear(
        self.n_heads * self.head_dim // self.n_groups,
        self.n_groups * config.o_lora_rank,
        bias=False,
        quant_config=quant_config,
        prefix=f"{prefix}.wo_a",
        return_bias=False,
    )
    # Every DSA o_proj path consumes wo_a.weight directly via
    # npu_transpose_batchmatmul / npu_transpose_quant_batchmatmul,
    # so the weight must remain ND.
    # 【中文】NPU 适配: DSA 的 o_proj 路径用转置矩阵乘算子直接消费
    # wo_a.weight，因此该权重禁止转成 NZ 格式（昇腾矩阵乘的专用排布）。
    self.wo_a.skip_weight_nz_conversion = True
    # wo_b: (o_groups × o_lora_rank) → hidden 升维（按行切分）。
    self.wo_b = RowParallelLinear(
        self.n_groups * config.o_lora_rank,
        self.dim,
        bias=False,
        reduce_results=reduce_results,
        quant_config=quant_config,
        prefix=f"{prefix}.wo_b",
        return_bias=False,
    )


@dataclass(frozen=True)
class DeepseekV41LayerRole:
    """The attention and future Engram responsibilities of one backbone layer."""
    """【中文说明】单个主干层的"角色卡"——描述该层在源共享混合 cache 图中的
    职责。语法点: @dataclass(frozen=True) 自动生成 __init__/__repr__ 等，
    frozen=True 使实例不可变（可哈希、防误改）。
    字段:
        layer_idx: 层号。
        compress_ratio: 压缩比（0=滑窗层；>0=长上下文压缩层）。
        kv_source_layer: 本层读取长 KV 的源层号（None=自己拥有或不适用）。
        index_source_layer: 本层读取 indexer K 的源层号。
        is_kv_source: 本层是否物理拥有长 KV cache。
        is_index_source: 本层是否物理拥有 indexer K cache。
        is_candidate_source: 本层是否产出候选块（粗筛 TopK block）。
        uses_candidate_filter: 本层是否消费上游候选块过滤搜索空间。
        engram_slot: engram 表槽位（engram 层为 0..n-1，其余 None）。
    """

    layer_idx: int
    compress_ratio: int
    kv_source_layer: int | None
    index_source_layer: int | None
    is_kv_source: bool
    is_index_source: bool
    is_candidate_source: bool
    uses_candidate_filter: bool
    engram_slot: int | None

    @property
    def has_long_context(self) -> bool:
        """是否为长上下文层（compress_ratio>0，需要 YaRN 伸缩 RoPE）。"""
        return self.compress_ratio > 0


@dataclass(frozen=True)
class DeepseekV41Topology:
    """Validated, immutable model-wide source/consumer topology."""
    """【中文说明】全模型层拓扑（不可变）：所有层的角色 + 源层 ID 集合 + 候选
    块选择参数。构建后供注意力实现查询"我的 cache 在哪 / 我该读谁的"。"""

    layers: tuple[DeepseekV41LayerRole, ...]
    kv_source_layer_ids: tuple[int, ...]
    index_source_layer_ids: tuple[int, ...]
    candidate_source_layer_id: int
    candidate_topk_blocks: int
    candidate_block_size: int
    index_topk: int

    def layer(self, layer_idx: int) -> DeepseekV41LayerRole:
        """按层号取角色。"""
        return self.layers[layer_idx]

    def kv_consumers(self, source_layer: int) -> tuple[int, ...]:
        """返回读取指定源层长 KV 的全部消费层号。"""
        return tuple(role.layer_idx for role in self.layers if role.kv_source_layer == source_layer)

    def index_consumers(self, source_layer: int) -> tuple[int, ...]:
        """返回读取指定源层 indexer K 的全部消费层号。"""
        return tuple(role.layer_idx for role in self.layers if role.index_source_layer == source_layer)


class DeepseekV41SharedAttentionState:
    """Per-forward handoff between index sources and their consumer layers."""
    """【中文说明】一次前向内跨层共享的注意力状态：topk_indices（各层选出的
    TopK token 位置）与 candidates（候选块 ID）。源层（index/candidate
    source）先写入，消费层在同一前向中读取。"""

    def __init__(self, topk_indices, candidates):
        """参数: topk_indices [max_tokens, index_topk]；candidates
        [max_tokens, 1, candidate_topk_blocks]（均 INT32，模型级共享缓冲）。"""
        self.topk_indices = topk_indices
        self.candidates = candidates

    def reset(self):
        # Source layers overwrite the active rows before any consumer reads
        # them. Keeping the storage intact avoids replay depending on Python
        # state mutation and preserves a fixed address for ACL Graph.
        # 【中文】"重置"是空操作: 源层会在消费层读取前覆写有效行；保持
        # 存储不重建，让 ACL Graph（昇腾版 CUDA Graph）重放时不依赖
        # Python 侧状态变更，且张量地址固定。
        return None


def _latest_source(layer_idx: int, sources: tuple[int, ...]) -> int | None:
    """返回不大于 layer_idx 的最近源层号；无则 None。

    语法点: next((...), None) 取生成器第一个元素，无元素时给默认值。"""
    return next((source for source in reversed(sources) if source <= layer_idx), None)


def build_layer_plan(config: Any) -> DeepseekV41Topology:
    """Build and validate the V4.1 layer-sharing graph from a text config.

    ``config`` is the parsed text-model config. Extra compression ratios for
    speculative layers are allowed,
    but only the first ``num_hidden_layers`` entries describe the backbone.
    """
    """【中文说明】从文本配置构建并校验 V4.1 层共享图。config 为解析后的文本
    配置；compress_ratios 可含投机层附加项，但只有前 num_hidden_layers 项
    描述主干。步骤:
        1) 读取各配置字段并截取主干部分；
        2) 逐层计算 kv/index 源（最近的 <= 本层号的源层）；
        3) 组装每层角色；候选过滤策略继承自 index 源层
           （uses_candidate_filter = index 源在 candidate 源之后）。
    """

    num_layers = int(config.num_hidden_layers)
    ratios = tuple(config.compress_ratios)
    kv_sources = tuple(config.kv_source_layer_ids)
    index_sources = tuple(config.index_source_layer_ids)
    engram_layers = tuple(config.engram_layer_ids)
    candidate_source = int(config.candidate_source_layer_id)
    candidate_topk_blocks = int(config.candidate_topk_blocks)
    candidate_block_size = int(config.candidate_block_size)
    index_topk = int(config.index_topk)

    ratios = ratios[:num_layers]

    # engram 层号 → 槽位号 映射（每张 engram 表一个槽位）。
    engram_slots = {layer_idx: slot for slot, layer_idx in enumerate(engram_layers)}
    roles: list[DeepseekV41LayerRole] = []
    for layer_idx, ratio in enumerate(ratios):
        # 压缩层(ratio>0)才需要源层；滑窗层(ratio=0)源为 None。
        kv_source = _latest_source(layer_idx, kv_sources) if ratio else None
        index_source = _latest_source(layer_idx, index_sources) if ratio else None

        roles.append(
            DeepseekV41LayerRole(
                layer_idx=layer_idx,
                compress_ratio=ratio,
                kv_source_layer=kv_source,
                index_source_layer=index_source,
                is_kv_source=layer_idx in kv_sources,
                is_index_source=layer_idx in index_sources,
                is_candidate_source=layer_idx == candidate_source,
                # Consumer layers inherit the selection policy of their index
                # source.  For example, layer 26 reuses layer 24 TopK, and that
                # TopK was computed inside layer 20's candidate blocks.
                # 【中文】消费层继承其 index 源的选择策略: 例如第 26 层复用
                # 第 24 层的 TopK，而那份 TopK 是在第 20 层候选块内算出的。
                uses_candidate_filter=index_source is not None and index_source > candidate_source,
                engram_slot=engram_slots.get(layer_idx),
            )
        )

    return DeepseekV41Topology(
        layers=tuple(roles),
        kv_source_layer_ids=kv_sources,
        index_source_layer_ids=index_sources,
        candidate_source_layer_id=candidate_source,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
        index_topk=index_topk,
    )


class AscendDeepseekV41SWACache(DeepseekV41CacheLayer):
    """Ascend SWA cache registered with the V4.1 allocator."""
    """【中文说明】滑窗注意力的 KV cache 层。继承 Ascend 侧的
    DeepseekV41CacheLayer（vllm_ascend/attention/dsa_v41.py），重写点在于
    用 DSV4_BLOCK_SIZES 表选择 Ascend 专用的物理 block 大小，并构造带
    "deepseek_v41" 标记的滑窗 MLA 规格。

    NPU 适配点: 物理存储 block 大小与调度器逻辑 block_size 不一定相同——
    DSV4_BLOCK_SIZES 按 cache_config.block_size（128/64/32）给出
    [mla, swa, c4 状态, c128 状态] 四类 block 大小；这里取 [0][1] 即 SWA
    的物理 block。A5 硬件或压缩 cache 特性会给出不同表。
    """

    def __init__(self, head_dim, window_size, dtype, prefix, cache_config):
        """参数: head_dim=MLA 潜向量宽；window_size=滑窗 token 数；
        dtype=cache 数据类型；prefix=注册名；cache_config=读逻辑 block_size。"""
        from vllm_ascend.models.layer.attention.layer import DSV4_BLOCK_SIZES

        # 从 DSV4 block 尺寸表取 SWA 的物理 block 大小。
        block_size = DSV4_BLOCK_SIZES[cache_config.block_size][0][1]
        spec = AscendSlidingWindowMLASpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=head_dim,
            dtype=dtype,
            sliding_window=window_size,
            cache_dtype_str=cache_config.cache_dtype,
            model_version="deepseek_v41",
            alignment=None,
        )
        # 语法点: get_current_vllm_config() 在构造时取当前引擎配置上下文，
        # 使非显式传参的路径也能拿到正确配置。
        super().__init__(get_current_vllm_config(), prefix, spec)
        self.head_dim = head_dim
        self.window_size = window_size
        self.dtype = dtype
        self.block_size = block_size
        self.cache_config = cache_config


class DeepseekV41SWAAttention(nn.Module):
    """Projection and Ascend SWA execution shared by target and draft."""
    """【中文说明】滑窗注意力基类：目标模型与 DSpark 草稿共用。持有
    init_attention_projections 定义的 MLA 投影 + 复指数 RoPE +
    AscendDeepseekSparseAttention（DSA 稀疏注意力封装）。compress_ratio
    固定为 0（不做长上下文压缩），compressor/indexer 为 None。"""

    # SWA cache 层类（子类可替换，如草稿专用标记类）。
    swa_cache_cls = AscendDeepseekV41SWACache

    def __init__(
        self,
        vllm_config,
        config,
        max_position_embeddings=0,
        cache_config=None,
        quant_config=None,
        prefix="",
        topk_indices_buffer=None,
        reduce_results=True,
        need_gather_q_kv=False,
        *,
        use_yarn=False,
    ):
        """初始化 SWA 注意力。

        参数:
            vllm_config/config: 引擎与模型配置。
            max_position_embeddings: RoPE 原始最大位置。
            cache_config/quant_config: cache 与量化配置。
            prefix: 参数名前缀（倒数第二段为层号）。
            topk_indices_buffer: 模型级共享 TopK 缓冲（SWA 不使用）。
            reduce_results: wo_b 是否归约输出。
            need_gather_q_kv: CP+SP 组合时是否需要先聚合 q/kv。
            use_yarn: 是否启用 YaRN 远程伸缩（keyword-only 参数）。
        步骤:
            1) 层号解析 + MLA 投影初始化；
            2) ComplexExpRotaryEmbedding 复指数 RoPE（V4.1 专用，
               beta_fast/beta_slow 为 YaRN 的低频/高频插值系数）；
            3) 构建 SWA cache 层；
            4) DSAModules 打包全部子模块交给 AscendDeepseekSparseAttention。
        """
        super().__init__()
        self.layer_idx = int(prefix.split(".")[-2])
        init_attention_projections(self, config, quant_config, prefix, reduce_results)
        # SWA 层不做长上下文压缩。
        self.compress_ratio = 0
        self.compressor = None
        self.indexer = None
        # 步骤2: V4.1 复指数 RoPE——用复指数实现的高效旋转，支持 YaRN。
        # use_yarn 时用 compress_rope_theta（压缩后基底）代替常规 theta。
        self.rotary_emb = ComplexExpRotaryEmbedding(
            vllm_config=vllm_config,
            layername=f"{prefix}.attn",
            head_size=self.rope_head_dim,
            rotary_dim=self.rope_head_dim,
            max_position_embeddings=max_position_embeddings,
            is_neox_style=False,
            scaling_factor=config.rope_parameters["factor"],
            base=config.compress_rope_theta if use_yarn else config.rope_theta,
            beta_fast=config.rope_parameters["beta_fast"],
            beta_slow=config.rope_parameters["beta_slow"],
            original_max_position_embeddings=max_position_embeddings,
            apply_yarn_scaling=use_yarn,
            rope_groups=["default"],
        )
        kv_cache_dtype = kv_cache_dtype_str_to_dtype(vllm_config.cache_config.cache_dtype, vllm_config.model_config)
        # 步骤3: SWA cache 层（block 大小来自 DSV4 表）。
        swa_cache_layer = self.swa_cache_cls(
            head_dim=self.head_dim,
            window_size=self.window_size,
            dtype=kv_cache_dtype,
            prefix=f"{prefix}.swa_cache",
            cache_config=cache_config,
        )

        # 步骤4: DSAModules 是"模块集合" dataclass——把投影/归一/cache 打包
        # 给 DSA 算子，算子内部直接调用这些模块（避免重复传参）。
        dsa_modules = DSAModules(
            wq_a=self.wq_a,
            q_norm=self.q_norm,
            q_norm_without_weight=self.q_norm_without_weight,
            wq_b=self.wq_b,
            wkv=self.wkv,
            kv_norm=self.kv_norm,
            wo_a=self.wo_a,
            wo_b=self.wo_b,
            attn_sink=self.attn_sink,
            indexer=self.indexer,
            compressor=self.compressor,
            swa_cache_layer=swa_cache_layer,
        )

        # AscendDeepseekSparseAttention: DSA 稀疏注意力的 Ascend 封装
        # （继承上游 MultiHeadLatentAttentionWrapper），负责真正的
        # 注意力计算与 cache 写入。
        self.dsa_attn = AscendDeepseekSparseAttention(
            dim=self.dim,
            n_heads=self.n_heads,
            scale=self.scale,
            n_local_heads=self.n_local_heads,
            q_lora_rank=self.q_lora_rank,
            o_lora_rank=self.o_lora_rank,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            nope_head_dim=self.nope_head_dim,
            eps=self.eps,
            n_groups=self.n_groups,
            n_local_groups=self.n_local_groups,
            window_size=self.window_size,
            compress_ratio=self.compress_ratio,
            dsa_modules=dsa_modules,
            cache_config=cache_config,
            quant_config=quant_config,
            # prefix=f'{prefix}.attn',
            prefix=f"{prefix}",
            need_gather_q_kv=need_gather_q_kv,
        )


class DeepseekV41Attention(DeepseekV41SWAAttention):
    """V4.1 source-shared attention with explicit cache ownership."""
    """【中文说明】V4.1 主干注意力：在 SWA 基类上增加"源共享"扩展。
    继承 DeepseekV41SWAAttention，重写 __init__/forward。关键扩展:
        1) 依据 layer role 判定 cache 所有权——仅 kv 源层创建
           long_kv_cache（全上下文压缩 KV）与 compressor，仅 index 源层
           创建 indexer；消费层通过前缀引用源层 cache；
        2) 构建 CP 感知的 v41_impl 并注册到 static_forward_context；
        3) forward 极薄——全部逻辑在自定义算子 dsa_v41_forward 内。
    """

    swa_cache_cls = AscendDeepseekV41SWACache

    def __init__(
        self,
        vllm_config,
        config,
        max_position_embeddings=0,
        cache_config=None,
        quant_config=None,
        prefix="",
        topk_indices_buffer=None,
        reduce_results=True,
        need_gather_q_kv=False,
    ):
        """初始化 V4.1 注意力。

        步骤:
            1) 解析层号并从 build_layer_plan 得到本层角色与拓扑；
            2) 调父类构造（长上下文层启用 YaRN）；
            3) kv 源层: 创建 long_kv_cache + compressor；
            4) index 源层: 创建 indexer（含 INT8 k_cache）；
            5) 计算源层 cache 引用前缀（消费层用）；
            6) 构建 CP 感知实现并注册 static_forward_context。
        """
        layer_idx = int(prefix.split(".")[-2])
        # 每层都重建拓扑（轻量、不可变），获取自身角色。
        topology = build_layer_plan(config)
        role = topology.layer(layer_idx)
        super().__init__(
            vllm_config=vllm_config,
            config=config,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            topk_indices_buffer=topk_indices_buffer,
            reduce_results=reduce_results,
            need_gather_q_kv=need_gather_q_kv,
            # 长上下文层（压缩层）需要 YaRN 伸缩 RoPE。
            use_yarn=role.has_long_context,
        )
        block_size = vllm_config.cache_config.block_size
        self.role = role
        self.topology = topology
        self.shared_state = None
        self.prefix = prefix
        width = config.head_dim
        self.softmax_scale = width**-0.5
        # 步骤3: 仅 kv 源层拥有全上下文压缩 KV cache。
        # tokens_per_state=compress_ratio: 每 ratio 个 token 一条状态；
        # storage_block_size = block_size // ratio: 存储 block 相应缩小。
        if role.is_kv_source:
            self.long_kv_cache = DeepseekV41CacheLayer(
                vllm_config,
                f"{prefix}.long_kv_cache",
                AscendMLAAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=width,
                    dtype=torch.bfloat16,
                    tokens_per_state=role.compress_ratio,
                    model_version="deepseek_v41",
                    storage_block_size=block_size // role.compress_ratio,
                ),
            )
        # kv 源层同时拥有 compressor（把 KV 压成环形状态）。
        self.compressor = (
            DeepseekV41Compressor(config, role.compress_ratio, vllm_config, f"{prefix}.compressor")
            if role.is_kv_source
            else None
        )
        # 步骤4: index 源层拥有 indexer（owns_k=True，含 wk/k_cache）。
        self.indexer = (
            DeepseekV41Indexer(
                config,
                role.is_kv_source,
                vllm_config,
                f"{prefix}.indexer",
                role.compress_ratio,
                quant_config=quant_config,
            )
            if role.is_index_source
            else None
        )
        # 步骤5: 计算源层 cache 的引用前缀。消费层（role.has_long_context
        # 但非源）用这两个前缀在运行时定位源层的 long_kv/indexer.k_cache。
        root = prefix.rsplit(".layers.", 1)[0]
        source = f"{root}.layers.{role.kv_source_layer}.self_attn"
        self.long_kv_source_prefix = f"{source}.long_kv_cache" if role.has_long_context else None
        self.index_k_source_prefix = f"{source}.indexer.k_cache" if role.has_long_context else None
        self.index_source_layer = role.index_source_layer
        from vllm_ascend.attention.context_parallel.dsa_v41_cp import get_v41_cp_classes

        # 步骤6: CP(上下文并行)感知实现——真正的前向逻辑载体，按角色
        # （源/消费、压缩比、候选策略）分派不同的执行路径。
        self.v41_impl = get_v41_cp_classes()[1](
            prefix=prefix,
            role=role,
            topology=topology,
            long_kv_source_prefix=self.long_kv_source_prefix,
            index_k_source_prefix=self.index_k_source_prefix,
        )
        # 注册进静态前向上下文: torch.ops.vllm.dsa_v41_forward 按名字找到
        # 本层，这也是 ACL Graph 捕获的入口约定。
        self.v41_layer_name = f"{prefix}.v41_attn"
        context = vllm_config.compilation_config.static_forward_context
        context[self.v41_layer_name] = self

    def forward(self, positions, hidden_states, llama_4_scaling=None):
        """注意力前向（薄封装）。

        参数:
            positions: [tokens] 位置。
            hidden_states: [tokens, hidden] 输入。
            llama_4_scaling: 兼容接口的缩放参数（未用）。
        返回:
            [tokens, hidden] 注意力输出。
        原理: torch.ops.vllm.dsa_v41_forward 是注册的自定义算子——按
            v41_layer_name 从 static_forward_context 取回本层，真正逻辑
            在 self.v41_impl 中执行（按角色分派）。预分配输出张量保证
            图捕获地址稳定。
        """
        output = torch.empty_like(hidden_states)
        torch.ops.vllm.dsa_v41_forward(hidden_states, output, self.v41_layer_name)
        return output


class DeepseekV41DecoderLayer(nn.Module):
    """V4.1 block with the checkpoint's delayed mHC coefficient handoff."""
    """【中文说明】V4.1 解码块。结构: mHC 前混合 → input_layernorm → 注意力 →
    mHC 后混合 → mHC 前混合(FFN) → rms_norm_cast → MoE → mHC 后混合。
    "延迟 mHC 系数交接"（delayed mHC coefficient handoff）指：hc_pre 返回的
    pre-mix 系数不在本层内消费，而是传给下一层/最终 hc_collapse 使用，
    使混合系数的计算与使用解耦。

    mHC（矩阵超连接）原理: 残差流有 hc_mult 条并行支路；每层入口用可学习
    混合矩阵（Sinkhorn 迭代归一化到双随机矩阵）重排各支路信息，出口再把
    注意力/FFN 输出按组合系数加回支路。相比普通残差，超连接缓解了深层
    梯度衰减并允许层间信息路由。hc_* 参数: fn=混合矩阵, base/scale=门控
    偏置与缩放, sinkhorn_iters=归一化迭代数。
    """

    # 注意力类（DSpark 子类替换为草稿版）。
    attention_cls = DeepseekV41Attention

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config=None,
        topk_indices_buffer: torch.Tensor | None = None,
        is_draft_layer: bool = False,
    ) -> None:
        """初始化解码块。

        参数:
            vllm_config: 引擎配置。
            prefix: 层前缀（末段数字为层号）。
            config: 可选配置（缺省从 vllm_config 取并归一化）。
            topk_indices_buffer: 模型级共享 TopK 缓冲。
            is_draft_layer: 是否为草稿层（禁用 engram/hash 路由）。
        """
        super().__init__()

        if config is None:
            config = normalize_deepseek_v41_config(vllm_config.model_config.hf_config)
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        # YaRN 的原始最大位置（伸缩的基准长度）。
        max_position_embeddings = config.rope_parameters["original_max_position_embeddings"]
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        # 【中文】make_layers 传入的 prefix 末段就是层号。
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx
        self.norm_eps = config.rms_norm_eps
        self.use_sequence_parallel_moe = parallel_config.use_sequence_parallel_moe
        # enable_dsa_cp: DSA 上下文并行开关（见 utils，待废弃的环境变量项）。
        self.enable_dsa_cp = enable_dsa_cp()  # TODO: delete this when enable_dsa_cp is sunset.

        attn_cls = self.attention_cls

        # 注意力: 序列并行时 wo_b 不归约（留 TP 部分和给后面的 reduce-scatter），
        # CP+SP 组合时需要先聚合 q/kv。
        self.self_attn = attn_cls(
            vllm_config=vllm_config,
            config=config,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            topk_indices_buffer=topk_indices_buffer,
            reduce_results=not self.use_sequence_parallel_moe,
            need_gather_q_kv=self.use_sequence_parallel_moe and self.enable_dsa_cp,
        )

        # V4.1 所有层都是 MoE 层（无稠密层分支）。
        self.mlp = DeepseekV41MoE(
            config=config,
            parallel_config=parallel_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            is_draft_layer=is_draft_layer,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=self.norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=self.norm_eps)
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        # ---- mHC 超连接参数 ----
        self.hc_mult = hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        # 混合矩阵行数 = (2 + hc_mult) * hc_mult: 每层要产生"注意力前/后 +
        # FFN 前/后"多组混合系数（2 个额外的组合行用于输出加回）。
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.hidden_size
        # hc_attn_* / hc_ffn_*: 注意力子层与 FFN 子层各自的混合矩阵(FP32)、
        # 门控偏置 base 与缩放 scale。
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.use_sequence_parallel = vllm_config.parallel_config.use_sequence_parallel_moe
        # Leave the TP partial sums for the reduce-scatter below. The mHC
        # and MoE paths then stay sharded between attention calls.
        # 【中文】SP 模式: wo_b 不做 all-reduce，把 TP 部分和留给后面的
        # reduce-scatter；mHC 与 MoE 路径在两次注意力调用之间保持分片。
        if self.use_sequence_parallel:
            self.self_attn.wo_b.reduce_results = False
        # ---- engram 记忆模块（仅配置了 engram 的目标层）----
        has_engram = engram_enabled(config)
        self.engram: AscendEngram | None
        # 类型注解先声明再赋值: 显式标注 "属性: 类型" 是 Python 的
        # "裸注解语句"语法，仅提供类型信息不产生运行时行为。
        if has_engram and not is_draft_layer and self.layer_idx in config.engram_layer_ids:
            self.engram = AscendEngram(config)
        else:
            self.engram = None

    def rms_norm_cast(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize once and provide the exact FP32 routing input."""
        """【中文说明】一次归一化同时产出两种精度: 归一化结果（原 dtype，喂
        MoE 专家）与其 FP32 版本（喂路由器打分）。NPU 适配: 优先走融合
        算子 npu_rms_norm_cast（单 kernel 完成），退路是手动二次转换。"""
        if enable_custom_op():
            return torch.ops._C_ascend.npu_rms_norm_cast(
                hidden_states,
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.variance_epsilon,
            )
        hidden_states = self.post_attention_layernorm(hidden_states)
        return hidden_states, hidden_states.float()

    @staticmethod
    def hc_collapse(x, pre_mix):
        """mHC 折叠: 多支路按 pre_mix 权重求和成单流。

        参数: x [tokens, hc_mult, hidden]；pre_mix [tokens, hc_mult] FP32。
        返回: [tokens, hidden]（转回原 dtype）。
        语法点: @staticmethod——不访问实例状态，可直接类名调用。
        """
        return (pre_mix.unsqueeze(-1) * x.float()).sum(-2).to(x.dtype)

    def hc_pre(self, x, hc_fn, hc_scale, hc_base, pre_mix=None):
        """mHC 入口混合（融合算子版）。

        原理: npu_hc_pre_v3 在单个 NPU 算子里完成——RMS 归一、线性混合、
        Sinkhorn 双随机归一化（hc_sinkhorn_iters 次）、sigmoid 门控，输出
        (混合后输入, 后混合系数 post, 组合系数 comb, 下一层 pre_mix)。
        """
        return torch.ops._C_ascend.npu_hc_pre_v3(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            pre_mix,
            hc_mult=self.hc_mult,
            hc_sinkhorn_iters=self.hc_sinkhorn_iters,
            norm_eps=self.norm_eps,
            hc_eps=self.hc_eps,
        )

    def hc_post(self, x, residual, post, comb):
        """mHC 出口混合（融合算子版）: 把子层输出 x 按组合系数 comb 加回
        残差流、再按 post 系数重排各支路。unsqueeze(0) 补 batch 维以匹配
        算子的张量布局约定。"""
        return torch.ops._C_ascend.npu_hc_post(
            x.unsqueeze(0),
            residual.unsqueeze(0),
            post.unsqueeze(0),
            comb.unsqueeze(0),
        ).squeeze(0)

    def forward(
        self,
        positions,
        hidden_states,
        pre_mix,
        llama_4_scaling=None,
        input_ids=None,
    ):
        """解码块前向。

        参数:
            positions: [tokens] 位置。
            hidden_states: [tokens, hc_mult, hidden] 多支路残差流。
            pre_mix: [tokens, hc_mult] 上一层传来的混合系数。
            input_ids: [tokens] 原始 token ID（MoE hash 路由用）。
        返回:
            (hidden_states, ffn_pre): 更新后的多支路流与本层 FFN 的
            pre-mix 系数（供模型级 hc_collapse 使用——"延迟交接"）。
        步骤:
            1) 注意力子层: hc_pre 混合 → input_layernorm →（SP: all_gather
               恢复全 token）→ 注意力 →（SP: reduce_scatter）→ hc_post;
            2) FFN 子层: hc_pre 混合 → rms_norm_cast（双精度）→ MoE → hc_post。
        """
        use_sequence_parallel = getattr(self, "use_sequence_parallel", False)
        residual = hidden_states
        # ---- 步骤1: 注意力子层 ----
        x, attn_post, attn_comb, attn_pre = self.hc_pre(
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            pre_mix,
        )
        x = self.input_layernorm(x)
        if use_sequence_parallel:
            # 注意力需要全量 token（滑窗跨 SP 分片），先 all_gather 恢复。
            x = sp_all_gather(x)[: positions.shape[0]]
        x = self.self_attn(positions, x, llama_4_scaling)
        if use_sequence_parallel:
            # 计算完把 TP 部分和 reduce-scatter 回分片形态。
            x = sp_reduce_scatter(x)
        hidden_states = self.hc_post(x, residual, attn_post, attn_comb)

        # ---- 步骤2: FFN(MoE) 子层 ----
        residual = hidden_states
        x, ffn_post, ffn_comb, ffn_pre = self.hc_pre(
            hidden_states,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            attn_pre,
        )
        # 一次归一化同时拿 BF16 输入与 FP32 路由输入。
        x, x_fp32 = self.rms_norm_cast(x)
        x = self.mlp(
            x,
            input_ids=input_ids,
            hidden_states_fp32=x_fp32,
            already_sequence_parallel=use_sequence_parallel,
        )
        hidden_states = self.hc_post(x, residual, ffn_post, ffn_comb)
        return hidden_states, ffn_pre


class DeepseekV41Model(nn.Module, EagleModelMixin):
    """V4.1 backbone with delayed HC collapse and shared attention state."""
    """【中文说明】V4.1 文本主干。继承 nn.Module 与上游 EagleModelMixin
    （投机解码目标模型接口）。职责:
        - 词嵌入 + N 层解码块 + 最终 RMSNorm；
        - 模型级共享缓冲: topk_indices_buffer（稀疏注意力 TopK 位置）与
          candidate_indices_buffer（候选块 ID），经 DeepseekV41SharedAttentionState
          在源层/消费层之间交接；
        - engram 记忆装配: 布局解析、各层嵌入表创建、n-gram 哈希状态、
          全局旋转矩阵加载；
        - "延迟 HC 折叠": 多支路残差流贯穿全部层，最后由 hc_collapse 折叠。
    """

    decoder_layer_cls = DeepseekV41DecoderLayer

    def __init__(self, *, vllm_config, prefix=""):
        """构建主干模型。

        步骤:
            1) 归一化配置；为稀疏注意力预分配 TopK 缓冲 [max_tokens, topk]；
            2) PP 首层建词嵌入（VocabParallelEmbedding，按词表切分），
               其余 PP 级为 PPMissingLayer 占位；
            3) make_layers 构建全部解码层（按 PP 区间实例化本 rank 的层）；
            4) PP 末层建最终 norm；
            5) 候选块缓冲 + 共享注意力状态挂到每层；
            6) engram 装配: checkpoint 预检、逐槽位建嵌入表、旋转矩阵
               加载、n-gram 哈希状态绑定 SWA cache。
        """
        super().__init__()

        config = normalize_deepseek_v41_config(vllm_config.model_config.hf_config)
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = current_platform.device_type
        self.use_sequence_parallel_moe = vllm_config.parallel_config.use_sequence_parallel_moe

        self.vocab_size = config.vocab_size
        # 步骤1: TopK 位置缓冲——容量 = max_num_batched_tokens，每 token
        # 存 index_topk 个选中位置。ACL Graph 重放时地址固定。
        if hasattr(config, "index_topk"):
            topk_tokens = config.index_topk
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                topk_tokens,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            topk_indices_buffer = None

        # Expose at model level so spec_decode/llm_base_proposer can share
        # this buffer with the MTP draft via attribute replacement.
        # 【中文】模型级暴露: spec_decode 提议器通过与 MTP 草稿共享该缓冲
        # （属性替换）避免重复分配。
        self.topk_indices_buffer = topk_indices_buffer

        # 步骤2: PP 分级构建。is_first_rank 建词嵌入，否则 PPMissingLayer
        # （占位对象，权重加载与 forward 自动跳过）。
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()
        # 步骤3: make_layers 返回 (start_layer, end_layer, layers)——本 rank
        # 只实例化自己 PP 区间内的层。语法点: lambda 捕获 topk_indices_buffer。
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: self.decoder_layer_cls(vllm_config, prefix, topk_indices_buffer=topk_indices_buffer),
            prefix=f"{prefix}.layers",
        )
        # 是否有层需要把 token IDs 喂给 MoE（hash 路由或视觉偏置）。
        self.needs_moe_input_ids = any(
            layer.mlp.gate.tid2eid is not None or layer.mlp.gate.bias_vl is not None
            for layer in islice(self.layers, self.start_layer, self.end_layer)
        )

        # 步骤4: 最终 norm（仅 PP 末级）。
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.hc_mult = config.hc_mult
        # MTP 草稿消费的隐状态缓冲（forward 中按需填充）。
        self._mtp_hidden_buffer = None
        self.make_empty_intermediate_tensors = self._make_empty_intermediate_tensors
        self.use_sequence_parallel = vllm_config.parallel_config.use_sequence_parallel_moe
        # 步骤5: 候选块缓冲（全 -1 初始化=无效）+ 共享注意力状态，挂到
        # 每个解码层的 self_attn 上。
        topology = build_layer_plan(self.config)
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        candidate_buffer = torch.full(
            (max_tokens, 1, topology.candidate_topk_blocks),
            -1,
            dtype=torch.int32,
            device=self.topk_indices_buffer.device,
        )
        self.candidate_indices_buffer = candidate_buffer
        self.shared_attention_state = DeepseekV41SharedAttentionState(
            self.topk_indices_buffer,
            candidate_buffer,
        )
        for layer in self.layers:
            if isinstance(layer, DeepseekV41DecoderLayer):
                layer.self_attn.shared_state = self.shared_attention_state
        # ---- 步骤6: engram 记忆装配 ----
        # engram 权重的 checkpoint 根路径（INT8 表 + 组 32 缩放）。
        self.engram_root = vllm_config.model_config.model
        config = self.config
        self.engram_weight_root = self.engram_root
        # The table is INT8 with group-32 scales; whether it lives in host
        # memory is vLLM's EngramConfig choice.
        # 【中文】表本体是 INT8+组32缩放；是否驻留主机内存由 EngramConfig
        # （--engram-config，cpu_offload 选项）决定。
        cpu_offload = engram_cpu_offload(vllm_config)
        self.engram_dp_shared_memory = bool(vllm_config.engram_config and vllm_config.engram_config.dp_shared_memory)
        self.engram_layout = EngramLayout.from_config(config) if engram_enabled(config) else None
        if self.engram_layout is not None:
            # Complete head buckets per rank, laid out over TP and the
            # node-local EDP group (upstream's, not one built from EP hosts).
            # Fail on an unreadable checkpoint before the first table exists:
            # the allocation below is per-rank 24-51 GiB, and discovering a
            # missing index/key during weight iteration would mean paying for
            # it first.  `dummy` reads no checkpoint at all.
            # 【中文】预检 checkpoint: engram 表每 rank 高达 24-51 GiB，必须
            # 在分配前确认 checkpoint 可读（索引存在、键存在、INT8 表带
            # scale），否则分配后再失败就白付显存。dummy 加载模式跳过。
            if vllm_config.load_config.load_format != "dummy":
                preflight_engram_checkpoint(
                    self.engram_weight_root, config.engram_layer_ids, AscendParallelEngramEmbedding
                )
            # 逐槽位创建嵌入表并挂到对应 engram 层的 layer.engram.embed_tokens。
            # head_sizes 由布局的素数表展开（每头桶宽度）。
            for slot, (layer_id, rows) in enumerate(zip(config.engram_layer_ids, config.engram_num_embeddings)):
                head_sizes = tuple(size for order in self.engram_layout.primes[slot] for size in order)
                embed = AscendParallelEngramEmbedding(
                    rows,
                    config.engram_head_dim,
                    head_sizes,
                    slot,
                    cpu_offload=cpu_offload,
                    dp_shared_memory=self.engram_dp_shared_memory,
                )
                # 绑定 checkpoint 路径与键名；真正加载发生在权重到达时的回调。
                embed.bind_checkpoint(self.engram_weight_root, f"layers.{layer_id}.engram.embed.weight")
                self.layers[layer_id].engram.embed_tokens = embed
        self.engram_hash = None
        self._engram_input_buffers = None
        # engram 输入缓冲容量: 取 max(批 token 上限, 图捕获尺寸上限)。
        self._engram_max_tokens = max(
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.compilation_config.max_cudagraph_capture_size or 0,
        )
        # 全局旋转矩阵: 默认 32×32 单位阵；checkpoint 的 optional/quarot.safetensors
        # 里有训练时的 global_rotation，加载后替换（engram_gate 用）。
        self.register_buffer("engram_rotation", torch.eye(32), persistent=False)
        if engram_enabled(config):
            if vllm_config.load_config.load_format != "dummy":
                # 语法点: torch.device("cpu") 上下文——在其中创建的张量都
                # 落在 CPU，避免大矩阵先占 NPU 显存。
                with torch.device("cpu"):
                    with safe_open(Path(self.engram_root) / "optional/quarot.safetensors", framework="pt") as file:
                        rotation = file.get_tensor("global_rotation")
                    block = rotation[:32, :32].contiguous()
                self.engram_rotation.copy_(block)
            # Upstream owns the n-gram history; the adapter only hands it the
            # Ascend SWA slot metadata (see engram/hash_state.py).
            # 【中文】n-gram 历史由上游 NgramHashState 管理；Ascend 适配器
            # 只把 SWA slot cache 元数据交给它（见 engram/hash_state.py）。
            swa_cache_layer = self.layers[config.engram_layer_ids[0]].self_attn.dsa_attn.swa_cache_layer
            self.engram_hash = create_engram_hash_state(vllm_config, config, swa_cache_layer)

    def _make_empty_intermediate_tensors(self, batch_size, dtype, device):
        """构造 PP 通信用的空中间张量。

        注意形状含 hc_mult 维: [batch, hc_mult, hidden]——多支路残差流
        在 PP 级间也保持展开形态。"""
        return IntermediateTensors(
            {
                "hidden_states": torch.empty(
                    (batch_size, self.hc_mult, self.config.hidden_size), dtype=dtype, device=device
                )
            }
        )

    def embed_input_ids(self, input_ids):
        """词元 ID → 词嵌入 [tokens, hidden]。"""
        return self.embed_tokens(input_ids)

    def prepare_engram(
        self,
        input_ids,
        positions,
        lookback_token_ids=None,
        query_start_loc=None,
        slot_mapping=None,
        block_table=None,
    ):
        """Hash on device with upstream NgramHashState, then look up head shards.

        【中文说明】engram 的核心准备流程: 设备上计算 n-gram 哈希 → 查各
        层的头分片嵌入表 → 返回 {layer_id: 行向量} 与有效 token 掩码。

        Calls without the device metadata (dummy runs) participate with empty
        hashes. History is the current chunk, then the runner's prompt
        lookback window, then the slot cache the hash state fills itself.

        【中文】没有设备元数据的调用（dummy 运行）以空哈希参与。历史来源
        优先级: 当前 chunk → runner 的 prompt 回看窗口 → 哈希状态自填的
        slot cache。

        参数:
            input_ids: [tokens] 原始词元。
            positions: [tokens] 位置。
            lookback_token_ids: [n_reqs, lookback_depth] chunk 之前的
                prompt 历史 token（-1 填充）。
            query_start_loc: [n_reqs+1] 各请求的 chunk 起始偏移。
            slot_mapping/block_table: SWA cache 的写槽映射与块表。
        返回:
            (lookups, mask): lookups = {layer_id: [tokens, n_hash_cols*dim]}
            BF16 行向量; mask = [tokens] bool（True=参与 engram，图像 token
            除外）。

        步骤:
            1) 判定是否真正做哈希(hashing)——需要: 哈希状态就绪 + 元数据
               有效（query_start_loc>1 且块表非空; DP dummy batch 只有
               1 个元素且块表 0 行，此时哈希会越界访存，必须走 dummy 路径）；
            2) DP 场景即使本副本跳过哈希也要参与集合通信(participates)，
               除非走共享内存模式（无逐步集合通信）；
            3) hashing: 计算 dead 掩码（图像哨兵）→ 调哈希状态前向 →
               keep 掩码取反；
            4) participates 但非 hashing: 用 dummy_hashes（全 DEAD_ID、
               不更新历史）；
            5) 查表: DP gather 哈希 → 逐层 embed_gathered。
        """
        config = self.config
        if not engram_enabled(config):
            # 未配置 engram: 空查找表 + 空掩码。
            return {}, torch.empty(0, dtype=torch.bool, device=positions.device)
        device = positions.device
        hash_state = self.engram_hash
        hashing = (
            hash_state is not None
            and hash_state.ensure_cache()
            and query_start_loc is not None
            # A DP dummy batch (worker.execute_dummy_batch -> _dummy_run) has
            # one token and no requests: the runner hands over a single-element
            # query_start_loc and a zero-row block table. Hashing with that
            # metadata makes the request search clamp to -1, and the hash
            # kernel then indexes one row before the block table -- an address
            # the device faults on ("GM address ... exceeds 48 bits", seen when
            # a FULL decode graph replays on an idle rank). No request rows
            # means no hash rows: fall back to the dummy-hash path below, which
            # is what an idle replica does in every other graph mode.
            # 【中文】DP dummy batch（FULL decode 图在空闲 rank 上重放时）
            # 只有 1 个 token 且无请求: query_start_loc 单元素、块表 0 行。
            # 用这种元数据做哈希会让请求搜索钳到 -1，内核越界访存导致
            # 设备故障（"GM 地址超 48 位"）。此时必须退回 dummy 哈希路径。
            and query_start_loc.numel() > 1
            and block_table is not None
            and block_table.shape[0] > 0
        )
        # A DP-sharded lookup is collective, so a replica that skips the hash
        # still has to reach it: it participates with no valid rows and no
        # history update (upstream's dummy_hashes branch). Sharing has no
        # per-step collectives, so it opts out.
        # 【中文】DP 分片查表是集合操作: 跳过哈希的副本也必须参与（以
        # 空行、不更新历史的方式），否则其他副本的 all_gather 会挂起。
        # 共享内存模式无逐步集合通信，可以完全退出。
        participates = hashing or (
            self.engram_hash is not None and not self.engram_dp_shared_memory and get_engram_dp_size() > 1
        )
        hashes = None
        mask = torch.empty(0, dtype=torch.bool, device=device)
        if hashing:
            assert hash_state is not None
            # 图像哨兵 ID: 图像区域 token 不参与 n-gram（完整图像区被排除）。
            image_token_id = config.image_token_id
            image_pad_token_id = getattr(config, "image_pad_token_id", image_token_id + 1)
            dead = engram_dead_mask(input_ids, image_token_id, image_pad_token_id)
            if lookback_token_ids is None:
                # 无回看窗口时构造全 -1 的占位（-1 不会匹配任何 token）。
                lookback_token_ids = input_ids.new_full((query_start_loc.numel() - 1, hash_state.lookback_depth), -1)
            # 核心: 计算 [tokens, layers, hash 列] 的 INT32 n-gram 哈希。
            hashes = hash_state(
                input_ids,
                positions,
                query_start_loc,
                dead,
                lookback_token_ids,
                engram_dead_mask(lookback_token_ids, image_token_id, image_pad_token_id),
                slot_mapping,
                block_table,
            )
            # Engram.forward takes True=keep.
            # 【中文】engram 前向约定 True=保留，取反 dead 得到 keep 掩码。
            mask = ~dead
        elif participates:
            assert hash_state is not None
            # dummy 参与: 全 DEAD_ID 哈希 + 全 False 掩码（不更新历史）。
            hashes, mask = hash_state.dummy_hashes(input_ids)
        lookups = {}
        tables = [self.layers[layer_id].engram.embed_tokens for layer_id in config.engram_layer_ids]
        if participates:
            assert hashes is not None
            # One DP gather feeds every layer sharing the split table.
            # 【中文】一次 DP gather 服务所有共享分片的 engram 层。
            gathered = gather_engram_hashes(hashes, dp_shared_memory=self.engram_dp_shared_memory)
            for slot, (layer_id, table) in enumerate(zip(config.engram_layer_ids, tables)):
                # gathered[:, slot] 取该层的哈希列；flatten(1) 把
                # [tokens, cols, dim] 压成 [tokens, cols*dim] 行向量。
                lookups[layer_id] = table.embed_gathered(gathered[:, slot], hashes.shape[0]).flatten(1)
        else:
            # 不参与（单副本且未 hashing）: 空行占位，形状与正常一致。
            for layer_id, table in zip(config.engram_layer_ids, tables):
                lookups[layer_id] = torch.empty(
                    (0, table.n_hash_cols * table.dim),
                    dtype=torch.bfloat16,
                    device=device,
                )
        return lookups, mask.to(positions.device)

    def prepare_engram_inputs(
        self,
        input_ids,
        positions,
        padded_tokens=None,
        lookback_token_ids=None,
        query_start_loc=None,
        slot_mapping=None,
        block_table=None,
    ):
        """Synchronously refresh the rows read by this forward, before replay."""
        """【中文说明】ACL Graph 重放前同步刷新 engram 行缓冲: 哈希/查表结果
        拷入固定地址缓冲（图内只读缓冲，不做哈希计算），保证重放语义正确。

        参数:
            padded_tokens: 图捕获的固定 token 数（None=按实际 token 数）。
        返回:
            graph_inputs: {"engram_lookups": {layer: 缓冲}, "engram_mask": 掩码缓冲}。
        步骤:
            1) 先取（或惰性创建）固定地址缓冲；
            2) 同步执行 prepare_engram；
            3) 掩码与各行缓冲拷入——有效段拷值、剩余段清零。
        """
        graph_inputs = self.prepare_engram_graph_inputs(padded_tokens)
        if not graph_inputs["engram_lookups"]:
            # 无 engram 层: 直接返回空结构。
            return graph_inputs
        num_tokens = positions.shape[0]
        output_tokens = num_tokens if padded_tokens is None else padded_tokens
        lookups, mask = self.prepare_engram(
            input_ids,
            positions,
            lookback_token_ids,
            query_start_loc,
            slot_mapping,
            block_table,
        )
        buffers = graph_inputs["engram_lookups"]
        mask_buffer = graph_inputs["engram_mask"]
        # 掩码: 有效段拷贝，图填充段清零。
        mask_buffer[: mask.numel()].copy_(mask)
        mask_buffer[mask.numel() : output_tokens].zero_()
        for layer, values in lookups.items():
            # 行缓冲: 有效段拷贝，填充段清零（长度固定=output_tokens）。
            buffers[layer][: values.shape[0]].copy_(values)
            buffers[layer][values.shape[0] : output_tokens].zero_()
        return graph_inputs

    def prepare_engram_graph_inputs(self, padded_tokens=None):
        """Capture fixed-address buffers without CPU history or routing work."""
        """【中文说明】惰性创建 engram 的固定地址输入缓冲（ACL Graph 捕获用）:
        每层一个 [capacity, cols*dim] BF16 缓冲 + 一个 [capacity] bool 掩码。
        缓冲按 _engram_max_tokens（批上限与图捕获尺寸的较大者）分配，仅
        创建一次后续复用——图重放时地址不变。"""
        if not engram_enabled(self.config):
            return {"engram_lookups": {}, "engram_mask": self.engram_rotation.new_empty(0, dtype=torch.bool)}
        if self._engram_input_buffers is None:
            capacity = self._engram_max_tokens
            # 列数 = (max_ngram - 1) × n_heads（每个 n-gram 阶 × 每头一列）。
            columns = (self.config.engram_max_ngram_size - 1) * self.config.engram_n_heads
            device = self.engram_rotation.device
            # 语法点: 元组打包 (dict, tensor) 存为一个属性，解包使用。
            self._engram_input_buffers = (
                {
                    layer: torch.zeros(
                        (capacity, columns * self.layers[layer].engram.embed_tokens.dim),
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    for layer in self.config.engram_layer_ids
                },
                torch.zeros(capacity, dtype=torch.bool, device=device),
            )
        buffers, mask_buffer = self._engram_input_buffers
        return {"engram_lookups": buffers, "engram_mask": mask_buffer}

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds=None,
        engram_lookups=None,
        engram_mask=None,
        lookback_token_ids=None,
    ):
        """主干前向。

        参数:
            input_ids: [tokens] 词元 ID（多模态时含占位 ID）。
            positions: [tokens] 位置。
            intermediate_tensors: PP 上游中间张量（非首级）。
            inputs_embeds: 预合并的输入嵌入（优先于词嵌入查表）。
            engram_lookups/engram_mask: 图模式预计算的 engram 行/掩码
                （None 时同步现算——仅 eager 模式）。
            lookback_token_ids: engram 哈希的 prompt 回看历史。
        返回:
            [tokens, hidden] 最终隐状态（或 (hidden, aux_hidden_states) 元组
            ——DSpark 需要 aux 辅助隐状态时）。
        步骤:
            1) 取输入嵌入（inputs_embeds 优先）；engram 行就绪（现算或复用）；
            2) 重置共享注意力状态（空操作，语义见其 docstring）；
            3) SP 切片: 图缓冲先截到真实 token 数再按 TP 分片；
            4) 词嵌入复制成 hc_mult 支路；pre_mix 初始化 one-hot；
            5) 逐层执行: 按需抽取 aux 隐状态（DSpark 条件输入）→ engram
               门控写回 → 解码层；
            6) 最后 hc_collapse 折叠多支路 → 最终 norm；
            7) SP all_gather 恢复。
        """
        use_sequence_parallel = getattr(self, "use_sequence_parallel", False)
        # 步骤1: 输入嵌入（VL 模型传 inputs_embeds；文本模型查词表）。
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        if engram_lookups is None:
            # eager 路径: 同步计算 engram 行（图模式在重放前已刷新缓冲）。
            lookups, token_mask = self.prepare_engram(input_ids, positions, lookback_token_ids=lookback_token_ids)
        else:
            lookups, token_mask = engram_lookups, engram_mask
        # 步骤2: 重置跨层共享注意力状态（ACL Graph 语义: 不重建存储）。
        self.shared_attention_state.reset()
        full_num_tokens = positions.shape[0]
        # Slice capacity-sized graph buffers before SP splits the token axis.
        # 【中文】步骤3: 容量型图缓冲先截到本次真实 token 数，再进 SP
        # 切片（顺序不能反，否则切片基数错误）。
        token_mask = token_mask[:full_num_tokens]
        lookups = {layer_idx: lookup[:full_num_tokens] for layer_idx, lookup in lookups.items()}
        if use_sequence_parallel:
            # SP: 维护填充掩码（VLLM_MOE_SKIP_PADDING 环境变量控制是否跳过
            # 填充 token 的计算）后按 TP 分片所有 token 维张量。
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                forward_context.is_padding = sp_padding_mask(
                    forward_context.is_padding,
                    hidden_states,
                )
            hidden_states = sp_shard(hidden_states)
            input_ids = sp_shard(input_ids)
            token_mask = sp_shard(token_mask)
            lookups = {layer_idx: sp_shard(lookup) for layer_idx, lookup in lookups.items()}
        # 步骤4: 复制成 [tokens, hc_mult, hidden] 多支路；初始混合系数
        # one-hot（只信任第一支路——等价于普通残差的起点）。
        hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
        pre_mix = hidden_states.new_zeros(hidden_states.shape[0], self.hc_mult, dtype=torch.float32)
        pre_mix[:, 0] = 1.0
        last_layer = None
        aux_hidden_states = []
        moe_input_ids = input_ids
        if self.needs_moe_input_ids:
            # SP 填充产生的 -1 占位 ID 替换为 0（防 hash 查表越界）。
            moe_input_ids = torch.where(input_ids == -1, 0, input_ids)
        for layer in self.layers:
            last_layer = layer
            # DSpark consumes the residual stream entering its configured
            # target layers. The runner expresses checkpoint IDs as one-based.
            # 【中文】步骤5a: DSpark 消费进入其目标层的残差流（支路均值），
            # 供草稿模型 combine_hidden_states 使用；checkpoint 的层号是
            # 1-based，因此用 layer_idx+1 比较。
            if layer.layer_idx + 1 in self.aux_hidden_state_layers:
                aux_hidden_state = hidden_states.mean(dim=1)
                if use_sequence_parallel:
                    # SP 下残差流是分片的，先 all_gather 恢复全量 token。
                    aux_hidden_state = sp_all_gather(aux_hidden_state)[:full_num_tokens]
                aux_hidden_states.append(aux_hidden_state)
            # 步骤5b: engram 层——查表行经门控写回多支路残差流。
            if layer.engram is not None and token_mask.numel():
                n = hidden_states.shape[0]
                # Graph captures keep lookup buffers at static capacity; the
                # model's actual token dimension remains scheduler-dynamic.
                # 【中文】图捕获的查找缓冲是静态容量；模型实际 token 维
                # 由调度器动态决定，故按 n 截取。
                lookup = lookups[layer.layer_idx][:n]
                active_mask = token_mask[:n]
                # 原地更新前 n 行（engram 只改有效 token 的残差流）。
                hidden_states[:n] = layer.engram(
                    hidden_states[:n],
                    lookup,
                    active_mask,
                    self.engram_rotation,
                )
            # 步骤5c: 解码层（注意力 + MoE + mHC 前后混合）。
            hidden_states, pre_mix = layer(positions, hidden_states, pre_mix, None, input_ids=moe_input_ids)
        assert last_layer is not None, "Hyper-connection collapse requires at least one decoder layer"
        # 步骤6: 延迟 HC 折叠——用最后一层 FFN 的 pre_mix 加权求和各支路。
        hidden_states = last_layer.hc_collapse(hidden_states, pre_mix)
        if use_sequence_parallel:
            # 步骤7: SP 恢复全量 token（截掉填充部分）。
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
        hidden_states = self.norm(hidden_states)
        if aux_hidden_states:
            # DSpark 场景: 额外返回辅助隐状态列表（目标层残差流）。
            return hidden_states, aux_hidden_states
        return hidden_states


class AscendDeepseekV41LLMForCausalLM(nn.Module, DeepseekV41MixtureOfExperts, SupportsPP, SupportsLoRA, SupportsEagle3):
    """V4.1 顶层因果语言模型（Ascend）。

    多继承:
        nn.Module                  —— PyTorch 基类；
        DeepseekV41MixtureOfExperts —— MoE 元数据（EPLB）；
        SupportsPP/SupportsLoRA/SupportsEagle3 —— vLLM 能力接口: 流水线
        并行 / LoRA 微调 / EAGLE3 投机解码目标模型。
    组成: DeepseekV41Model 主干 + lm_head（PP 末级）+ logits_processor。
    """

    # gate/up 合并加载映射（与 MoE 的融合权重对应）。
    packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    model_cls = DeepseekV41Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        """构建顶层 LM: 主干 + lm_head + logits 处理器 + MoE 通信方法。"""
        super().__init__()
        config = normalize_deepseek_v41_config(vllm_config.model_config.hf_config)
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.model = self.model_cls(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        # lm_head 仅 PP 末级创建（其他级是占位）。
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        # Set MoE hyperparameters
        # 【中文】收集 MoE 超参数（专家数/层数，EPLB 与运行时需要）。
        self.num_moe_layers = self.config.num_hidden_layers
        self.set_moe_parameters()
        from vllm_ascend.ascend_forward_context import MoECommType
        from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method

        # 每种 MoE 通信类型（如 EP all-to-all）预建一个 Ascend 通信方法。
        self.moe_comm_methods = {kind: get_moe_comm_method(kind) for kind in MoECommType}

    # 需要 runner 传原始词元（engram 历史与 hash MoE 路由用）。
    requires_raw_input_tokens = True
    # 延迟加载的权重标记/前缀: engram 表与多模态/草稿部分延后加载。
    _DEFERRED_WEIGHT_MARKERS: tuple[str, ...] = ()
    _DEFERRED_WEIGHT_PREFIXES = ("aligner.", "vision.", "image_", "mtp.")

    def prepare_engram_inputs(
        self,
        input_ids,
        positions,
        padded_tokens=None,
        lookback_token_ids=None,
        query_start_loc=None,
        slot_mapping=None,
        block_table=None,
    ):
        """engram 图输入同步刷新入口，委托主干（见 DeepseekV41Model）。"""
        return self.model.prepare_engram_inputs(
            input_ids,
            positions,
            padded_tokens,
            lookback_token_ids,
            query_start_loc,
            slot_mapping,
            block_table,
        )

    def prepare_engram_graph_inputs(self, padded_tokens=None):
        """engram 固定地址缓冲获取，委托主干。"""
        return self.model.prepare_engram_graph_inputs(padded_tokens)

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        engram_lookups=None,
        engram_mask=None,
        lookback_token_ids=None,
    ):
        """顶层前向，委托主干并透传 engram 相关参数（语义见主干 forward）。"""
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            engram_lookups=engram_lookups,
            engram_mask=engram_mask,
            lookback_token_ids=lookback_token_ids,
        )

    @property
    def token_lookback_depth(self) -> int:
        """Tokens before a chunk start the engram hash needs; the runner passes
        them as ``lookback_token_ids``."""
        """【中文说明】engram 哈希需要"chunk 起点之前"的 token 数；runner 按
        此值填充 lookback_token_ids 缓冲。无 engram 时为 0。"""
        engram_hash = self.model.engram_hash
        return engram_hash.lookback_depth if engram_hash is not None else 0

    @classmethod
    def _is_milestone_weight(cls, name):
        """判断权重是否"里程碑"权重（非延迟加载类）。

        原理: engram 表按流式分片加载（embedding.py 的索引读），与其余
        权重的常规迭代加载分离；视觉(mtp/aligner/vision/image_)也延后。
        语法点: @classmethod + 类属性访问（cls._DEFERRED_*）。
        """
        return not name.startswith(cls._DEFERRED_WEIGHT_PREFIXES) and not any(
            marker in name for marker in cls._DEFERRED_WEIGHT_MARKERS
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """权重加载入口: 过滤 engram 表权重后走常规加载。

        原理: engram 表不在常规迭代里加载（量太大且走独立索引流）；
        无 engram 配置时显式剔除 ".engram." 权重防 KeyError。
        语法点: 生成器表达式做惰性过滤，避免物化整个权重列表。
        """
        if not engram_enabled(self.model.config):
            return self._load_model_weights((name, tensor) for name, tensor in weights if ".engram." not in name)
        loaded = self._load_model_weights((name, tensor) for name, tensor in weights if self._is_milestone_weight(name))
        return loaded

    def set_moe_parameters(self):
        """扫描层收集 MoE 元数据（专家权重表/层数/样例层），逻辑同
        DeepseekV41MixtureOfExperts.extract_moe_parameters 的数据源侧。"""
        self.expert_weights = []

        self.num_expert_groups = getattr(self.config, "n_group", 1)

        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            if isinstance(layer.mlp, DeepseekV41MoE):
                # Pick last one layer since the first ones may be dense layers.
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)

        self.extract_moe_parameters(example_moe)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """词元 ID → 嵌入，委托主干。"""
        return self.model.embed_input_ids(input_ids)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """隐状态 → logits（PP 末级; LogitsProcessor 负责采样相关处理）。"""
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """专家权重加载映射 (参数名, checkpoint 名, expert_id, shard_id)。

        原理: mix_placement（融合共享专家）时共享专家作为附加专家槽位，
        专家总数相应加 n_shared_experts。"""
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return fused_moe_make_expert_params_mapping(
            self.model,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts
            + (self.config.n_shared_experts if getattr(get_ascend_config(), "mix_placement", False) else 0),
            num_redundant_experts=0,
        )

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        """Pre-hc_head residual stream buffer (max_num_batched_tokens,
        hc_mult * hidden_size) for the MTP draft model. Populated by
        forward(); valid after each target step."""
        """【中文说明】MTP 草稿模型消费的"hc_head 之前"残差流缓冲:
        [max_num_batched_tokens, hc_mult * hidden_size]。由 forward 填充，
        每个目标步之后有效——投机解码提议器据此构建草稿条件输入。"""
        return getattr(self.model, "_mtp_hidden_buffer", None)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        """设置需要抽取辅助隐状态的层号（DSpark 目标层，1-based），
        委托主干（runner 在投机解码初始化时调用）。"""
        self.model._set_aux_hidden_state_layers(layers)

    def _load_model_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """常规权重加载（含 checkpoint→运行时的名字重映射）。

        步骤:
            1) 构建堆叠映射(gate/up→gate_up_proj)与专家映射；
            2) 逐权重: 跳过 mtp.*（草稿模型自己加载）→ 名字规范化
               （w1/w2/w3、head/lm_head、embed、attn、ffn、ffn_norm、
               attn_norm、.scale 等 checkpoint→运行时改名）→ 跳过
               RoPE inv_freq（运行时重算）；
            3) attn_sink 特殊处理（TP 头切片 / CP 全量）；
            4) 专家权重走 expert_params_mapping（expert_id 定位）；
            5) 堆叠权重（gate/up）走 stacked 映射（shard_id 定位列区间）；
            6) 其余走默认加载器（weight_loader 处理 TP 切片）。
        """
        fuse_shared_experts = getattr(get_ascend_config(), "mix_placement", False)
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        # 【中文】专家参数映射: 覆盖每个专家的权重/FP8 权重缩放/激活缩放。
        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self.model,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts + (self.config.n_shared_experts if fuse_shared_experts else 0),
            num_redundant_experts=self.num_redundant_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        # Attention heads per rank
        # 【中文】本 rank 的注意力头区间（attn_sink 切片用）。
        heads_per_rank = self.config.num_attention_heads // tp_size
        head_start = tp_rank * heads_per_rank

        for name, loaded_weight in weights:
            # 步骤2a: 跳过投机解码层（草稿模型独立加载 mtp.*）。
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model

            # TODO:
            # 【中文】缺 model. 前缀的顶层权重补前缀。
            if not name.startswith("model"):
                name = f"model.{name}"

            # ---- checkpoint → 运行时名字规范化（纯字符串替换） ----
            # w1/w2/w3 是 checkpoint 的专家 FFN 命名，对应 gate/down/up。
            if ".w1." in name:
                name = name.replace(".w1.", ".gate_proj.")
            if ".w2." in name:
                name = name.replace(".w2.", ".down_proj.")
            if ".w3." in name:
                name = name.replace(".w3.", ".up_proj.")

            if "model.head." in name and "model.lm_head." not in name:
                name = name.replace("model.head.", "lm_head.")
            if "model.lm_head." in name:
                name = name.replace("model.lm_head.", "lm_head.")
            # engram 表的缩放张量: .scale → .weight_scale_inv（Ascend 命名）。
            if name.endswith(".engram.embed.scale"):
                name = name.removesuffix(".scale") + ".weight_scale_inv"
            if "embed." in name and "embed_token." not in name:
                name = name.replace("embed.", "embed_tokens.")
            if "attn" in name and "self_attn" not in name:
                name = name.replace(".attn.", ".self_attn.")
            if ".ffn." in name:
                name = name.replace(".ffn.", ".mlp.")
            if ".ffn_norm." in name:
                name = name.replace(".ffn_norm.", ".post_attention_layernorm.")
            if ".attn_norm." in name:
                name = name.replace(".attn_norm.", ".input_layernorm.")
            # 通用量化缩放后缀改名。
            if name.endswith(".scale"):
                name = name.replace(".scale", ".weight_scale")

            # RoPE 频率表运行时按配置重算，checkpoint 里的跳过。
            if "rotary_emb.inv_freq" in name:
                continue
            if ".gate.bias_vl" in name:
                # The parameter keeps the checkpoint name on Ascend. It is
                # passed to the hash router as its vision-only correction
                # bias, while text rows continue to use tid2eid.
                # 【中文】视觉路由偏置保持 checkpoint 名（Ascend 约定），
                # 交给 hash 路由器使用; 文本 token 仍走 tid2eid。
                pass
            elif ".gate.bias" in name:
                name = name.replace(".gate.bias", ".gate.e_score_correction_bias")

            # Hash-router layers route text tokens through ``tid2eid`` and keep
            # ``e_score_correction_bias`` unset, but the checkpoint still ships
            # a router bias for them. Skip it instead of raising a KeyError.
            # 【中文】hash 路由层不设 e_score_correction_bias，但 checkpoint
            # 仍带了该偏置——跳过而非报 KeyError。
            if name.endswith(".gate.e_score_correction_bias") and name not in params_dict:
                continue

            # ---- attn_sink 特殊处理 ----
            if "sink" in name:
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                if enable_dsa_cp():
                    # CP 模式: 每 rank 需要全部头的 sink（序列切分而非头切分）。
                    param.data.copy_(loaded_weight)
                else:
                    # Handle attention sinks (distributed across ranks)
                    # 【中文】普通 TP: 按 rank 头区间切片加载。
                    narrow_weight = loaded_weight.narrow(0, head_start, heads_per_rank)
                    param.data.copy_(narrow_weight)
                loaded_params.add(name)
                continue

            # 融合共享专家层的判定（mix_placement 时 shared_experts 并入 experts）。
            is_fusion_moe_shared_experts_layer = fuse_shared_experts and ("mlp.shared_experts" in name)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                # 【中文】专家的 gate_proj 不能在这里改名（否则与下面的
                # 专家映射二次替换出错），先跳过交给专家分支。
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                if is_fusion_moe_shared_experts_layer:
                    continue
                name_mapped = name.replace(weight_name, param_name)

                # QKV fusion is optional, fall back to normal
                # weight loading if it's not enabled
                # if go with fusion option, then update name
                # 【中文】可选的 QKV 融合: 融合参数不存在则回退常规加载。
                if (param_name == "fused_qkv_a_proj") and name_mapped not in params_dict:
                    continue
                else:
                    name = name_mapped
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                # 堆叠权重: weight_loader 按 shard_id 写入融合参数的对应列段。
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # 语法点: for...else——循环未 break（无堆叠映射命中）时执行。
                is_expert_weight = False

                # Special handling: when AITER fusion_shared_experts is enabled,
                # checkpoints may provide a single widened shared_experts tensor
                # without explicit expert indices
                # (e.g. ...mlp.shared_experts.gate_proj.weight).
                # For models with multiple shared experts, split that tensor
                # evenly into per-shared-expert slices and load them into
                # appended expert slots mlp.experts.{n_routed_experts + j}.*
                # accordingly.
                # 【中文】融合共享专家: checkpoint 可能给一个加宽的共享专家
                # 张量（无专家下标）。这里按 n_shared_experts 均匀切分，
                # 逐片装入附加专家槽位 mlp.experts.{n_routed+j}。
                num_chunks = 1
                if is_fusion_moe_shared_experts_layer:
                    num_chunks = getattr(self.config, "n_shared_experts", 1) or 1
                    # Determine split axis based on op type
                    # gate/up: ColumnParallel → split along dim 0
                    # down: RowParallel → split along dim 1
                    # 【中文】切分轴: gate/up 是列并行→按 dim0；down 是行
                    # 并行→按 dim1。
                    split_dim = 1 if "down_proj.weight" in name else 0
                    total = loaded_weight.shape[split_dim]
                    assert total % num_chunks == 0, (
                        f"Shared expert weight dim {total} not divisible by num_chunks {num_chunks}"
                    )
                    chunk_size = total // num_chunks

                for j in range(num_chunks):
                    chunk_name = name
                    weight_to_load = loaded_weight

                    if is_fusion_moe_shared_experts_layer:
                        # 按切分轴取出第 j 片共享专家权重。
                        if split_dim == 0:
                            weight_to_load = loaded_weight[j * chunk_size : (j + 1) * chunk_size, :]
                        else:
                            weight_to_load = loaded_weight[:, j * chunk_size : (j + 1) * chunk_size]
                        # Synthesize an expert-style name so expert mapping
                        # can route it
                        # 【中文】合成专家式名字，让下面的专家映射路由该片。
                        chunk_name = name.replace(
                            "mlp.shared_experts",
                            f"mlp.experts.{self.config.n_routed_experts + j}",
                        )

                    # Use expert_params_mapping to locate the destination
                    # param and delegate to its expert-aware weight_loader
                    # with expert_id.
                    # 【中文】专家权重: 映射定位目标参数，用带 expert_id 的
                    # 加载器（EP 场景只装入本 rank 驻留的专家）。
                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in chunk_name:
                            continue

                        # Anyway, this is an expert weight and should not be
                        # attempted to load as other weights later
                        is_expert_weight = True

                        # Do not modify `name` since the loop may continue here
                        # Instead, create a new variable
                        # 【中文】不修改 name（循环可能继续），用新变量。
                        name_mapped = chunk_name.replace(weight_name, param_name)

                        if is_pp_missing_parameter(name_mapped, self):
                            continue

                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or
                        # not here since otherwise we may skip experts with
                        # other available replicas.
                        # 【中文】要求加载器返回成败: 否则带副本的专家可能
                        # 被误跳过（return_success=True 协议）。
                        weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
                        success = weight_loader(
                            param,
                            weight_to_load,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            if not is_fusion_moe_shared_experts_layer:
                                name = name_mapped
                            else:
                                loaded_params.add(name_mapped)
                            break
                    else:
                        # 非专家权重兜底（for...else）。
                        if is_expert_weight:
                            # We've checked that this is an expert weight
                            # However it's not mapped locally to this rank
                            # So we simply skip it
                            # 【中文】确认是专家权重但本 rank 无此专家（EP
                            # 驻留在别处），跳过。
                            continue

                        # Skip loading extra bias for GPTQ models.
                        if name.endswith(".bias") and name not in params_dict:
                            continue

                        # Remapping the name of FP8 kv-scale.
                        # 【中文】FP8 KV 缩放名重映射（可能返回 None=跳过）。
                        name = maybe_remap_kv_scale_name(name, params_dict)
                        if name is None:
                            continue

                        if is_pp_missing_parameter(name, self):
                            continue

                        # 默认加载: 参数自带 weight_loader（TP 切片逻辑）或
                        # 直接复制。
                        param = params_dict[name]
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight)
            if not is_fusion_moe_shared_experts_layer:
                loaded_params.add(name)

        return loaded_params

    @property
    def engram_cache_layer_name(self) -> str | None:
        """engram n-gram 历史绑定的 SWA cache 层名。

        原理: 哈希状态的 slot cache 读取该 SWA cache 的 block_size 与
        kv_cache（见 engram/hash_state.py 的 AscendEngramSlotCache）。"""
        if not engram_enabled(self.model.config):
            return None
        return self.model.layers[0].self_attn.dsa_attn.swa_cache_layer.prefix
