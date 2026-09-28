# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""DeepSeek V4 语言主干的昇腾 NPU 适配实现。

本文件实现 DeepSeek V4 文本模型（对应上游 vLLM 的 DeepSeek V4 模型结构）
的核心组件，是整个 deepseek_v4 包的基础:

  - DeepseekV2MLP           : dense FFN（SwiGLU），供共享专家复用;
  - DeepseekV4MoE           : DeepSeek V4 MoE 层——细粒度路由专家 + 共享
    专家 + 无辅助损失路由（e_score_correction_bias）+ 可选 hash 路由层
    （tid2eid: token id 直接查表路由）+ EPLB 冗余专家支持;
  - DeepseekV4Attention     : DSA 稀疏 MLA 注意力。MLA（多头潜在注意力）
    把 q/kv 低秩压缩到潜在空间（wq_a/wq_b、wkv），RoPE 部分解耦
    （nope/rope 两段）; 输出侧用分组低秩投影（wo_a/wo_b, o_lora_rank）;
    KV 经 Compressor 压缩存储，indexer 提供 TopK 稀疏索引，attn_sink
    提供每头可学习偏置;
  - DeepseekV4DecoderLayer  : 解码层 = Hyper-Connections 残差结构
    （hc_pre/hc_post，取代普通 add 残差）+ 注意力 + MoE;
  - DeepseekV4Model         : 堆叠 decoder 层的主干（支持 PP 流水线、
    sequence parallel MoE、aux hidden states 导出）;
  - DeepseekV2MixtureOfExperts: MoE 元数据混入（EPLB 接口）;
  - AscendDeepseekV4ForCausalLM: CausalLM 顶层（lm_head + logits +
    权重加载）。

Hyper-Connections 原理: V4 的残差流是 hc_mult 路并行隐状态
[N, hc_mult, H]，每个子层前后经 hc_pre/hc_post（Sinkhorn 归一化的
可学习混合矩阵）做层间混合——相比传统残差有更强的表达能力，
输出的多路隐状态由 hc_head 加权合并回单路。

关键 NPU 适配点:
  - 权重保持 ND 布局（skip_weight_nz_conversion）供转置 matmul 直接消费;
  - npu_rms_norm_cast / npu_hc_pre_v2 / npu_hc_post 等自定义融合算子;
  - attention sink 在 DSA-CP（上下文并行）下全量复制、普通 TP 下按头切分;
  - sequence parallel MoE（sp_shard/sp_all_gather/sp_reduce_scatter）。
"""
import math
import typing
from collections.abc import Callable, Iterable
from itertools import islice

import torch
import torch.nn.functional as F
import vllm.envs as envs
from torch import nn
from transformers import DeepseekV2Config, DeepseekV3Config
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ParallelConfig, VllmConfig
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
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache as VllmDeepseekV4SWACache
from vllm.v1.kv_cache_interface import KVCacheSpec

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec
from vllm_ascend.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_reduce_scatter,
    sp_shard,
)
from vllm_ascend.models.deepseek_v4.compressor import Compressor
from vllm_ascend.models.deepseek_v4.indexer import DeepseekV4Indexer
from vllm_ascend.ops.dsa import AscendDeepseekSparseAttention, DSAModules
from vllm_ascend.ops.rope_dsv4 import ComplexExpRotaryEmbedding
from vllm_ascend.ops.triton.mul_add import muls_add_triton
from vllm_ascend.utils import (
    enable_custom_op,
    enable_dsa_cp,
    extract_dsv4_layer_index,
    get_dsv4_compress_ratio,
)
from vllm_ascend.worker.v2.pp_utils import (
    PPTransportDataType,
    add_pp_transport_tensors,
    get_pp_transport_tensors,
)
from vllm_ascend.worker.v2.pp_utils import (
    make_empty_intermediate_tensors as make_pp_empty_intermediate_tensors,
)

# 别名: 上游 vLLM 用 sequence_parallel_chunk 这个名字调用序列切分，
# 昇腾实现统一叫 sp_shard（函数别名赋值，两个名字指向同一函数对象）。
sequence_parallel_chunk = sp_shard


class AscendDeepseekV4SWACache(VllmDeepseekV4SWACache):
    """DSA 注意力的滑动窗口 KV cache 伪层（继承上游 VllmDeepseekV4SWACache）。

    原理: DSA 注意力在每个 token 的“近邻滑窗”内做精确注意力（窗口外的
    长程上下文交给压缩状态 + indexer TopK）。本伪层让 vLLM 为滑窗 KV
    分配 paged cache，并选择 NPU 专用的 AscendDSASWABackend。

    NPU 适配: block_size 查表自 DSV4_BLOCK_SIZES; FP8 cache 时每条
    KV 额外多存 128 维（量化元数据随行存储）。
    """

    def __init__(
        self,
        head_dim: int,
        window_size: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
    ):
        """初始化。

        Args:
            head_dim: 单 KV 头维度（MLA 潜在 KV 维度）。
            window_size: 滑窗大小（token 数）。
            dtype: cache dtype（如 bfloat16 / float8_e4m3fn）。
            prefix: 模块名前缀。
            cache_config: cache 配置。
        """
        # 父类以 uint8 裸存储初始化，再改回真实 dtype（父类构造约定）。
        super().__init__(head_dim, window_size, torch.uint8, prefix, cache_config)
        from vllm_ascend.models.layer.attention.layer import DSV4_BLOCK_SIZES

        self.dtype = dtype

        # SWA cache 的 NPU 块大小查表（[mla, swa, c4_state, c128_state][1]）。
        self.block_size = DSV4_BLOCK_SIZES[cache_config.block_size][0][1]

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        """返回滑窗 cache 的 KVCacheSpec（AscendSlidingWindowMLASpec）。"""
        # FP8 cache: 每行多存 128 字节的量化元数据（scale 等）。
        cached_head_size = self.head_dim + 128 if self.dtype == torch.float8_e4m3fn else self.head_dim
        return AscendSlidingWindowMLASpec(
            block_size=self.block_size,
            # MLA 潜在 KV 是单头。
            num_kv_heads=1,
            head_size=cached_head_size,
            dtype=self.dtype,
            sliding_window=self.window_size,
            cache_dtype_str=self.cache_config.cache_dtype,
            model_version="deepseek_v4",
            alignment=None,
        )

    # 伪层占位 stub（cache 读写由 DSA 注意力后端的自定义算子完成）。
    def forward(self): ...

    def get_attn_backend(self):
        """返回滑窗 cache 配套的 NPU 注意力后端（AscendDSASWABackend）。"""
        from vllm_ascend.attention.dsa_v1 import AscendDSASWABackend

        return AscendDSASWABackend


def precompute_freqs_cis_cpu(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    # 【中文】在 CPU 上预计算 YaRN 式 RoPE 的复指数表（复数形式 e^{iθ}）。
    # YaRN 原理: 对低频维度做插值缩放（factor）+ 平滑过渡（ramp），
    # 使模型在扩展上下文长度时外推更平稳。

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """求“频率插值与外推的边界维度”（辅助闭包函数）。"""
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """把旋转周期数换算成 [low, high] 维度区间。"""
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        """构造 [0,1] 线性渐变掩码（平滑过渡因子）。"""
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # 步骤1: 标准逆频率 1/base^(2i/dim)。
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        # 步骤2: YaRN 修正——低频段（smooth~0）保持原频率（外推），
        # 高频段（smooth~1）按 1/factor 插值; smooth 是维度上的线性渐变。
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    # 步骤3: 位置 x 频率的外积 -> 极坐标合成复数 e^{iθ}。
    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    # 【中文】复数域 RoPE: 把最后一维按 (实,虚) 成对解释为复数，
    # 乘以单位复数 e^{iθ}（freqs_cis）完成旋转; inverse 时乘共轭（反向旋转）。
    # y = x 保存原张量引用——最终用 y.copy_ 原地写回（省一次分配，
    # 与 NPU 上的原地算子约定一致）。
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        # 3D 输入 [T, H, D]: 表按 [1, T, D] 广播。
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        # 4D 输入 [B, T, H, D]: 表按 [1, T, 1, D] 广播。
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    # 复数乘法 -> 转回实数对并 flatten。
    x = torch.view_as_real(x * freqs_cis.to(x.device)).flatten(-2)
    # 原地写回输入存储并返回（in-place，NPU 友好）。
    y.copy_(x)
    return y


def get_spec_layer_idx_from_weight_name(config: DeepseekV2Config | DeepseekV3Config, weight_name: str) -> int | None:
    """判断权重名是否属于投机采样草稿层（mtp.*）。

    Returns:
        属于 mtp.* 返回 0（草稿层起始索引），否则 None（目标模型权重）。
    """
    if weight_name.startswith("mtp."):
        return 0
    return None


class DeepseekV2MLP(nn.Module):
    """dense FFN 模块（SwiGLU 结构），供共享专家（shared_experts）使用。

    结构: gate_up_proj（gate/up 两半合并的 ColumnParallel 投影）->
    SiluAndMul 激活 -> down_proj（RowParallel 投影）。
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
        """初始化。

        Args:
            hidden_size: 输入/输出维度。
            intermediate_size: 中间维度（gate_up 输出为其 2 倍）。
            hidden_act: 激活名（仅支持 "silu"）。
            swiglu_limit: 可选的激活截断上限（SiluAndMulWithClamp）。
            quant_config: 量化配置。
            reduce_results: down_proj 后是否做 TP all-reduce。
            is_sequence_parallel: 序列并行模式——输入/输出在 TP 组内按
                token 切分，权重复制、无需集合通信（disable_tp=True）。
            prefix: 模块名前缀。
        """
        super().__init__()

        # If is_sequence_parallel, the input and output tensors are sharded
        # across the ranks within the tp_group. In this case the weights are
        # replicated and no collective ops are needed.
        # Otherwise we use standard TP with an allreduce at the end.
        # 【中文】见 Args 说明: 序列并行时 disable_tp=True（权重不切分）。
        # MergedColumnParallelLinear: 两个同尺寸 Column 投影（gate/up）
        # 合并为一个参数矩阵，一次 matmul 完成。
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
        # 激活函数: 带 clamp 上限的 SwiGLU（可选）或标准 SwiGLU。
        if swiglu_limit is not None:
            self.act_fn = SiluAndMulWithClamp(swiglu_limit)
        else:
            self.act_fn = SiluAndMul()

    def forward(self, x):
        """前向: x [N, hidden] -> gate_up -> SwiGLU -> down -> [N, hidden]。"""
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class DeepseekV4MoE(nn.Module):
    """DeepSeek V4 MoE 层: 细粒度路由专家 + 共享专家 + 无辅助损失路由。

    MoE 原理（DeepSeek-V3 式）:
      - 路由: gate 对每个 token 打分，经 softmax + e_score_correction_bias
        偏置后选 top-k 个专家（无辅助损失负载均衡: 用偏置代替辅助损失
        引导均衡，训练/推理行为一致）;
      - 细粒度专家: 专家数量多、单个专家较小（moe_intermediate_size 小），
        组合表达力更强;
      - 共享专家: n_shared_experts 个专家对所有 token 恒激活，承载
        公共知识; 最终输出 = routed 输出 x routed_scaling_factor + 共享输出。
    V4 新增:
      - hash 路由层（num_hash_layers 前几层）: 不打分，直接用
        token id 查表（tid2eid）确定专家——省去路由计算且天然负载均衡;
      - 多模态路由偏置 bias_vl: 视觉 token 使用独立的路由偏置;
      - EPLB: 支持冗余物理专家（n_redundant_experts），逻辑专家可映射
        到多个物理副本以均衡负载。

    NPU 适配点: FusedMoEFactory 走昇腾 fused MoE 内核; mix_placement
    （混部）模式下共享专家并入专家列表统一调度; muls_add_triton 融合
    “缩放+相加”。
    """

    def __init__(
        self,
        config: DeepseekV2Config | DeepseekV3Config | DeepseekV4Config,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        is_draft_layer: bool = False,
    ):
        """初始化 MoE 层。

        Args:
            config: HF 模型配置。
            parallel_config: 并行配置（EPLB/序列并行开关在其中）。
            quant_config: 量化配置。
            prefix: 模块名前缀（形如 "model.layers.{i}.mlp"，用于解析层号）。
            is_draft_layer: 是否属于草稿模型层（hash 路由仅目标层启用）。
        """
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        # 从 prefix 解析层号（prefix 倒数第二段即层号字符串）。
        layer_idx = int(prefix.split(sep=".")[-2])
        self.layer_idx = layer_idx
        # 路由输出缩放系数（routed 与 shared 的相对权重）。
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.5)
        self.swiglu_limit = getattr(config, "swiglu_limit", None)

        # EP（专家并行）分组信息。
        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = config.n_shared_experts

        # 序列并行 MoE: token 维切分（权重复制），见 DeepseekV2MLP 说明。
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {config.hidden_act}. Only silu is supported for now.")

        # 路由门: hidden -> n_routed_experts 的打分线性层（复制式，
        # 每个 EP rank 都要完整打分才能路由）。
        self.gate = ReplicatedLinear(
            config.hidden_size, config.n_routed_experts, bias=False, quant_config=None, prefix=f"{prefix}.gate"
        )
        # NPU 适配: 路由权重预转 FP32（打分精度要求; 动态量化场景避免
        # 反复 cast）。
        self.gate.precast_fp32_weight = True

        # Load balancing settings.
        # 【中文】EPLB（专家并行负载均衡）配置。
        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        # 逻辑专家数 = 配置的 n_routed_experts; 物理专家数 = 逻辑 + 冗余
        # 副本; 本 rank 持有的物理专家数按 EP 均分。
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        # 本 rank 的物理专家区间 [start, end)。
        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = self.physical_expert_start + self.n_local_physical_experts

        # ROCm AITER 开关（在 Ascend 上立即被下一行覆盖为混部配置）。
        self.is_rocm_aiter_moe_enabled = rocm_aiter_ops.is_fused_moe_enabled()
        self.is_fusion_moe_shared_experts_enabled = rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        # Ascend 分支: 混部（mix_placement）时共享专家并入 fused MoE。
        self.is_fusion_moe_shared_experts_enabled = getattr(get_ascend_config(), "mix_placement", False)
        if config.n_shared_experts is None or self.is_fusion_moe_shared_experts_enabled:
            # 混部/无共享专家: 不单独建 shared_experts 模块。
            self.shared_experts = None
        else:
            # 共享专家: n_shared_experts 个专家并联成一个大 MLP
            #（intermediate_size 乘上专家数）。reduce_results=False:
            # 与 routed 输出的合并推迟到 forward 里融合做。
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts

            self.shared_experts = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                swiglu_limit=self.swiglu_limit,
                quant_config=quant_config,
                is_sequence_parallel=self.is_sequence_parallel,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        # hash 路由层判定: 层号 < num_hash_layers 且非草稿层。
        self.hash = layer_idx < config.num_hash_layers and not is_draft_layer
        # 多模态路由偏置 bias_vl: 视觉模型才创建。
        self.gate.bias_vl = None
        if getattr(config, "vision_n_layers", 0) > 0:
            self.gate.bias_vl = nn.Parameter(
                torch.empty(
                    config.n_routed_experts,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
        if self.hash:
            # MC2 dispatch/combine fails
            # in the dummy profile run if a token repeats an expert id
            # 【中文】hash 路由表 tid2eid: [vocab_size, num_experts_per_tok]
            # ——token id 加列偏移取模，保证同一 token 的 top-k 个专家
            # 互不相同（MC2 dispatch 在哑/profile 运行中不允许重复专家 id）。
            token_ids = torch.arange(config.vocab_size, dtype=torch.int32).unsqueeze(1)
            expert_offsets = torch.arange(config.num_experts_per_tok, dtype=torch.int32).unsqueeze(0)
            token_to_expert = (token_ids + expert_offsets) % config.n_routed_experts

            self.gate.tid2eid = nn.Parameter(
                token_to_expert,
                requires_grad=False,
            )
            # hash 路由不用 e_score_correction_bias。
            self.gate.e_score_correction_bias = None
        else:
            # 常规路由: tid2eid 为 None，使用偏置修正的 softmax 打分。
            self.gate.tid2eid = None
            self.gate.e_score_correction_bias = nn.Parameter(torch.empty(config.n_routed_experts, dtype=torch.float32))

        # FusedMoEFactory: 构造融合 MoE 执行器（内部含 MoERunner/
        # dispatch/combine）。关键参数:
        # - renormalize=norm_topk_prob: top-k 权重归一化;
        # - scoring_func: softmax 打分;
        # - routed_scaling_factor 在路由外施加（与 V4 的
        #   “先归一化再缩放”顺序一致; AITER 内部缩放故除外）;
        # - image_sentinel_lo=129257: 多模态哨兵 token 的路由特殊处理;
        # - hash_indices_table: hash 路由表;
        # - enable_eplb/num_redundant_experts: EPLB 支持。
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
            # 【中文】缩放放在路由路径之外，与 V4 的顺序一致（见上）。
            routed_scaling_factor=self.routed_scaling_factor,
            swiglu_limit=self.swiglu_limit,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            bias_vl=self.gate.bias_vl,
            image_sentinel_lo=129257,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            n_shared_experts=config.n_shared_experts if self.is_fusion_moe_shared_experts_enabled else 0,
            hash_indices_table=self.gate.tid2eid,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        hidden_states_fp32: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """MoE 前向。

        Args:
            hidden_states: [num_tokens, hidden_size] 输入。
            input_ids: [num_tokens] token id（hash 路由层必需）。
            hidden_states_fp32: 预升 FP32 的输入（路由打分用，避免重复
                cast; 由上层的 rms_norm_cast 一并产出）。
        Returns:
            [num_tokens, hidden_size] routed(+shared) 输出。
        """
        # hash 路由必须有 token id。
        if self.gate.tid2eid is not None and input_ids is None:
            raise ValueError("DeepSeek V4 hash MoE routing requires input_ids.")

        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if hidden_states_fp32 is not None:
            hidden_states_fp32 = hidden_states_fp32.view(-1, hidden_dim)

        if self.experts.is_internal_router:
            # In this case, the gate/router runs inside the FusedMoEFactory class
            # 【中文】分支1: 融合内部路由——router 打分在 FusedMoE 内部
            # 完成，直接把（FP32）输入当 router 输入传入。
            router_input = hidden_states if hidden_states_fp32 is None else hidden_states_fp32
            fused_moe_out = self.experts(
                hidden_states=hidden_states,
                router_logits=router_input,
                input_ids=input_ids,
            )
        else:
            # 分支2: 外部路由——这里显式打分。
            # router_logits: (num_tokens, n_experts)
            # 【中文】路由打分在 FP32 下做（.float() 或用预升好的
            # hidden_states_fp32），保证 top-k 选择数值稳定。
            router_input = hidden_states.float() if hidden_states_fp32 is None else hidden_states_fp32
            router_logits = F.linear(router_input, self.gate.weight)
            fused_moe_out = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                input_ids=input_ids,
            )

        # 输出可能是 (shared_output, final) 元组（共享专家单独返回）或
        # 已合并的张量。
        fused_moe_out_is_tuple = isinstance(fused_moe_out, tuple)
        if fused_moe_out_is_tuple:
            shared_output, final_hidden_states = fused_moe_out
            if self.shared_experts is None:
                assert shared_output is None

            if hidden_states.dtype != torch.float16:
                # 非 FP16: 需要在这里补 routed_scaling_factor
                #（内部未缩放）。muls_add_triton: 融合的
                # “y = x*scale + y”算子（NPU Triton 移植）。
                if not self.is_rocm_aiter_moe_enabled:
                    if self.shared_experts is not None:
                        assert shared_output is not None
                        final_hidden_states = muls_add_triton(
                            final_hidden_states, shared_output, self.routed_scaling_factor
                        )
                    else:
                        final_hidden_states *= self.routed_scaling_factor
            elif self.shared_experts is not None:
                # FP16 + AITER: 缩放反向作用在共享输出上（AITER 内部已
                # 乘过 routed_scaling_factor）。
                assert shared_output is not None
                final_hidden_states = muls_add_triton(
                    shared_output, final_hidden_states, 1.0 / self.routed_scaling_factor
                )
        else:
            final_hidden_states = fused_moe_out

        if not self.is_sequence_parallel and self.tp_size > 1 and fused_moe_out_is_tuple:
            # Legacy tuple outputs are reduced here. Tensor outputs from the
            # upstream MoERunner have already gone through its final reduction.
            # 【中文】旧式元组输出在此做 TP all-reduce; 新式张量输出已在
            # MoERunner 内部完成归约。
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    """计算 YaRN 的注意力缩放补偿因子 mscale。

    原理: YaRN 插值改变了注意力 logits 的幅度分布，需按
    0.1*mscale*ln(scale)+1 放大 softmax 缩放以保持熵稳定。
    局部 import math: 遮蔽（shadowing）模块级 math——原实现如此，效果
    等价（同一对象）。
    """
    import math

    if scale <= 1:
        # 未扩展（scale<=1）: 无需补偿。
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _get_llama_4_scaling(
    original_max_position_embeddings: int, scaling_beta: float, positions: torch.Tensor
) -> torch.Tensor:
    """LLaMA-4 式的按位置对数缩放因子（可选特性，当前未启用）。

    原理: scaling = 1 + beta*log(1 + pos/max_pos)，对远距离 token 放大
    注意力 logits（Llama4 的 NoPE+scaling 长上下文方案）。
    """
    scaling = 1 + scaling_beta * torch.log(1 + torch.floor(positions / original_max_position_embeddings))
    # Broadcast over num_heads and head_dim
    # 【中文】在尾部补两个维度，广播到 [T, H, D] 的注意力 logits。
    return scaling[..., None, None]


class DeepseekV4Attention(nn.Module):
    """DSA 稀疏 MLA 注意力层（参数与子模块的容器，计算委托给 dsa_attn）。

    MLA（多头潜在注意力）原理:
      - q 侧: hidden --wq_a--> q_lora_rank 低秩空间 --RMSNorm--> qr
        --wq_b--> n_heads*head_dim 的多头 q（q 的低秩压缩降参且压缩 KV）;
      - kv 侧: hidden --wkv--> head_dim 的单条潜在 KV 向量（所有头共享，
        即 num_kv_heads=1）——KV Cache 只需存这条压缩向量，显存占用
        大幅下降;
      - RoPE 解耦: q/k 的 head_dim 分为 nope 部分（不旋转，随 KV 压缩
        存储）与 rope 部分（旋转，单独存储/计算），避免旋转破坏压缩;
      - 输出侧: 分组低秩投影 wo_a（H -> n_groups*o_lora_rank）+ wo_b，
        进一步压缩输出参数量。

    DSA 稀疏注意力（V4）: 上述 MLA 的精确注意力只在“滑窗 + indexer
    TopK 选中的 token”上计算; 压缩 KV 由 compressor 维护，TopK 索引由
    indexer 产出（compress_ratio=4 时）。attn_sink 是每头可学习偏置
    （注意力“汇聚”先验）。

    本类主要做参数定义与子模块组装，真正的 forward 在
    AscendDeepseekSparseAttention（vllm_ascend/ops/dsa.py）。
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config | DeepseekV4Config,
        max_position_embeddings: int = 0,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        reduce_results: bool = True,
        need_gather_q_kv: bool = False,
    ) -> None:
        """初始化注意力层。

        Args:
            vllm_config: vLLM 全局配置。
            config: HF 模型配置。
            max_position_embeddings: 最大位置数（RoPE 预计算用）。
            cache_config: cache 配置。
            quant_config: 量化配置。
            prefix: 模块名前缀（含层号）。
            topk_indices_buffer: 跨层共享的 TopK 缓冲（IndexCache）。
            reduce_results: wo_b 后是否做 TP all-reduce（SP MoE 时否）。
            need_gather_q_kv: SP+DSA-CP 时是否需要 gather 完整 q/kv。
        """
        super().__init__()
        # 从 prefix 解析层号。
        layer_idx = int(prefix.split(sep=".")[-2])
        self.layer_idx = layer_idx
        # 提取本层的“全局层号”（hash/压缩率等按层配置时用）。
        config_layer_idx = extract_dsv4_layer_index(config, prefix)
        tp_size = get_tensor_model_parallel_world_size()
        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_local_heads = config.num_attention_heads // tp_size
        # q 低秩维度。
        self.q_lora_rank = config.q_lora_rank
        # 输出低秩维度（o 侧分组压缩）。
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        # RoPE 维度与 nope 维度（解耦）。
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = config.head_dim - config.qk_rope_head_dim
        # 输出分组数（wo_a 的分组低秩）。
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // tp_size
        # DSA 滑窗大小。
        self.window_size = config.sliding_window
        self.eps = config.rms_norm_eps
        self.norm_eps = config.rms_norm_eps
        self.scale = self.head_dim**-0.5
        # DSA 上下文并行开关（环境变量控制）。
        self.enable_dsa_cp = enable_dsa_cp()

        # attention sink: 每头可学习偏置。DSA-CP 时所有 rank 持全部头，
        # 否则只持本 rank 的头（与 load_weights 的切分逻辑对应）。
        attn_sink_heads = self.n_heads if self.enable_dsa_cp else self.n_local_heads
        self.attn_sink = nn.Parameter(torch.empty(attn_sink_heads, dtype=torch.float32))
        # wq_a: q 低秩压缩投影 hidden -> q_lora_rank（复制式，每卡全量）。
        self.wq_a = ReplicatedLinear(
            self.dim,
            self.q_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_a",
            return_bias=False,
        )
        # qr 的 RMSNorm（低秩空间的归一化）。
        self.q_norm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        # q 无权重 RMSNorm（head 维度的归一化，DSA kernel 内部约定）。
        self.q_norm_without_weight = RMSNorm(self.head_dim, eps=config.rms_norm_eps, has_weight=False)
        # wq_b: qr -> n_heads*head_dim 升维投影。DSA-CP 时复制式
        #（上下文并行下每卡需完整 q），否则按头做 Column TP 切分。
        wq_b_cls = ReplicatedLinear if self.enable_dsa_cp else ColumnParallelLinear
        self.wq_b = wq_b_cls(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
            return_bias=False,
        )

        # wkv: hidden -> head_dim 的潜在 KV 投影（MLA 单 KV 头核心）。
        self.wkv = ReplicatedLinear(
            self.dim,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wkv",
            return_bias=False,
        )
        # 潜在 KV 的 RMSNorm。
        self.kv_norm = RMSNorm(self.head_dim, self.norm_eps)
        # wo_a: 分组低秩输出投影 [n_heads*head_dim/n_groups] ->
        # [n_groups*o_lora_rank]（Column TP 按组切分）。
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
        # 【中文】NPU 适配: DSA 的 o_proj 路径经
        # npu_transpose(_quant)_batchmatmul 直接消费 wo_a.weight
        #（转置 matmul），故权重必须保持 ND 布局、不做 NZ 转换。
        self.wo_a.skip_weight_nz_conversion = True
        # wo_b: 低秩空间 -> hidden（Row TP; SP MoE 时推迟归约）。
        self.wo_b = RowParallelLinear(
            self.n_groups * config.o_lora_rank,
            self.dim,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.wo_b",
            return_bias=False,
        )
        # 本层压缩率（按层配置 compress_ratios 查询）。
        self.compress_ratio = get_dsv4_compress_ratio(config, config_layer_idx)

        if self.compress_ratio > 1:
            # 压缩层: RoPE 基底与旋转组不同——用压缩专用 rope_theta，
            # 并注册 "c{ratio}" 旋转组（与 "default" 组并存，按张量选择）。
            rope_theta = config.compress_rope_theta
            rope_groups = ["default", f"c{self.compress_ratio}"]
        else:
            # 非压缩层: 标准 rope_theta，仅默认旋转组。
            rope_theta = config.rope_theta
            rope_groups = ["default"]
        # ComplexExpRotaryEmbedding: 复指数形式的 RoPE 模块（昇腾实现），
        # 支持 YaRN 缩放参数与多旋转组。
        self.rotary_emb = ComplexExpRotaryEmbedding(
            vllm_config=vllm_config,
            layername=f"{prefix}.attn",
            head_size=self.rope_head_dim,
            rotary_dim=self.rope_head_dim,
            max_position_embeddings=max_position_embeddings,
            is_neox_style=False,
            scaling_factor=config.rope_parameters["factor"],
            base=rope_theta,
            beta_fast=config.rope_parameters["beta_fast"],
            beta_slow=config.rope_parameters["beta_slow"],
            rope_groups=rope_groups,
        )

        self.compressor: Compressor | None = None
        self.indexer: DeepseekV4Indexer | None = None

        use_index_cache = getattr(config, "use_index_cache", False)

        # IndexCache: decide whether this layer reuses topk from a previous
        # indexer-bearing layer. Refer: https://arxiv.org/abs/2603.12201
        # Only meaningful when this layer actually owns an Indexer (c4) and
        # IndexCache is enabled via hf-overrides. MTP layers are excluded
        # because spec_decode shares topk_indices_buffer at the model level
        # only, leaving impl-level references stale.
        # 【中文】IndexCache: 相邻 indexer 层可复用同一 TopK 结果（论文
        # arXiv:2603.12201）。仅当本层确实持有 indexer（c4 压缩）且通过
        # hf-overrides 启用时生效; MTP 草稿层除外——投机解码只在模型级
        # 共享 topk_indices_buffer，实现层的引用会过期。
        skip_topk = False
        if self.compress_ratio == 4 and use_index_cache and ".mtp." not in prefix:
            # 本层之前（含本层）的 c4 层数，即本层是第几个 indexer 层。
            compress_ratios = getattr(config, "compress_ratios", None) or []
            indexer_seq_idx = sum(1 for r in compress_ratios[:config_layer_idx] if r == 4)
            # 两种复用模式: 按频率（index_topk_freq: 每 freq 层算一次）
            # 或按显式模式串（"FSS..." F=算 S=跳过）。
            pattern = getattr(config, "index_topk_pattern", None)
            freq = getattr(config, "index_topk_freq", 1)
            if pattern is None:
                # 频率模式: 不是第 freq 的倍数层就跳过。
                skip_topk = max(indexer_seq_idx - 1, 0) % freq != 0
            else:
                # 模式串必须以 F 开头（第一层必须算）。
                assert pattern[0] == "F", "index_topk_pattern must start with 'F'"
                if 0 <= indexer_seq_idx < len(pattern):
                    skip_topk = pattern[indexer_seq_idx] == "S"

        if self.compress_ratio > 1:
            # 压缩层: 创建 Compressor（attention 自己的压缩 KV 来源）。
            self.compressor = Compressor(
                vllm_config,
                config,
                self.compress_ratio,
                head_dim=self.head_dim,
                quant_config=quant_config,
                cache_config=cache_config,
                prefix=f"{prefix}.compressor",
            )  # Compressor(4, 128)

            if self.compress_ratio == 4:
                # c4 压缩层: 创建 Indexer（TopK 选择）。
                self.indexer = DeepseekV4Indexer(
                    vllm_config,
                    config,
                    self.compress_ratio,
                    skip_topk=skip_topk,
                    use_index_cache=use_index_cache,
                    quant_config=quant_config,
                    cache_config=cache_config,
                    prefix=f"{prefix}.indexer",
                    topk_indices_buffer=topk_indices_buffer,
                )

        # 滑窗 cache 伪层（近邻精确注意力的 KV）。
        kv_cache_dtype = kv_cache_dtype_str_to_dtype(vllm_config.cache_config.cache_dtype, vllm_config.model_config)
        swa_cache_layer = AscendDeepseekV4SWACache(
            head_dim=self.head_dim,
            window_size=self.window_size,
            dtype=kv_cache_dtype,
            prefix=f"{prefix}.swa_cache",
            cache_config=cache_config,
        )

        # DSAModules: 把本层全部子模块打包传给注意力实现（数据类，
        # dsa.py 中定义）——实现层据此直接调用这些模块。
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

        # AscendDeepseekSparseAttention: 真正的 DSA 前向实现
        #（vllm_ascend/ops/dsa.py，包装 MLA 注意力接口）。
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

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None,
    ) -> torch.Tensor:
        """注意力前向（纯委托）。

        Args:
            positions: [num_tokens] 位置 id。
            hidden_states: [num_tokens, hidden_size] 输入。
            llama_4_scaling: 可选的按位置缩放因子（未启用时 None）。
        Returns:
            [num_tokens, hidden_size] 注意力输出。
        """
        return self.dsa_attn(positions, hidden_states, llama_4_scaling)


class DeepseekV4DecoderLayer(nn.Module):
    """DeepSeek V4 解码层: Hyper-Connections 残差 + DSA 注意力 + MoE。

    前向结构（与普通 pre-norm 层的差异在于残差路径）:
      x1 = hc_post(self_attn(input_layernorm(hc_pre(x))), x, post, comb)
      x2 = hc_post(mlp(norm(hc_pre(x1))), x1, post', comb')
    hc_pre/hc_post 是 Hyper-Connections 的混合算子（npu_hc_pre_v2 /
    npu_hc_post 自定义融合算子），把 [N, hc_mult, H] 的多路隐状态做
    Sinkhorn 归一化的加权混合; post/comb 是 hc_pre 返回的辅助系数，
    供 hc_post 恢复/合并残差。
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config: DeepseekV2Config | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        is_draft_layer: bool = False,
    ) -> None:
        """初始化解码层。

        Args:
            vllm_config: vLLM 全局配置。
            prefix: 模块名前缀（末段为层号）。
            config: 层级配置（None 时取全局 hf_config; 草稿模型可传
                自己的配置）。
            topk_indices_buffer: IndexCache 共享缓冲。
            is_draft_layer: 是否草稿层（hash 路由等仅目标层启用）。
        """
        super().__init__()

        if config is None:
            config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = config.rope_parameters["original_max_position_embeddings"]
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        # 【中文】make_layers 传入的 prefix 末段就是层号。
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx
        self.norm_eps = config.rms_norm_eps
        self.use_sequence_parallel_moe = parallel_config.use_sequence_parallel_moe
        self.enable_dsa_cp = enable_dsa_cp()  # TODO: delete this when enable_dsa_cp is sunset.

        attn_cls = DeepseekV4Attention

        # DSA 注意力子层。SP MoE 且非 DSA-CP 时: 注意力不归约
        #（reduce_results=False）且需要 gather 完整 q/kv。
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

        # MoE 子层（V4 所有层均为 MoE; 草稿层标志透传）。
        self.mlp = DeepseekV4MoE(
            config=config,
            parallel_config=parallel_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            is_draft_layer=is_draft_layer,
        )
        # 两个 pre-norm（注意: 实际作用在 hc_pre 的输出上）。
        self.input_layernorm = RMSNorm(config.hidden_size, eps=self.norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=self.norm_eps)
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        # Hyper-Connections 参数: 注意力侧与 FFN 侧各一套
        #（fn 混合矩阵 [2+hc_mult)*hc_mult, hc_mult*H]、base 偏置、
        # scale 三段缩放）。
        self.hc_mult = hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def rms_norm_cast(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize once and provide the exact FP32 routing input."""
        # 【中文】“归一化一次、顺带产出精确 FP32 路由输入”: MoE 路由打分
        # 需要 FP32 输入，若单独 .float() 会多一次归一化或拷贝; NPU 适配:
        # 自定义融合算子 npu_rms_norm_cast 一次返回（归一化结果, FP32 结果），
        # 环境开关 enable_custom_op() 关闭时回退到两次计算。
        if enable_custom_op():
            return torch.ops._C_ascend.npu_rms_norm_cast(
                hidden_states,
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.variance_epsilon,
            )
        hidden_states = self.post_attention_layernorm(hidden_states)
        return hidden_states, hidden_states.float()

    def hc_pre(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        """Hyper-Connections 前混合（NPU 融合算子 npu_hc_pre_v2）。

        原理: 对多路隐状态做“门控混合”——Sinkhorn 归一化的混合矩阵
        （hc_sinkhorn_iters 次迭代保证双随机性）+ sigmoid 门控，输出
        混合后的隐状态与两个辅助系数 (post, comb)（供 hc_post 使用）。

        Args:
            x: [N, hc_mult, H] 多路隐状态。
            hc_fn/hc_scale/hc_base: 注意力侧或 FFN 侧的混合参数。
        Returns:
            (mixed, post, comb): 混合结果 + hc_post 所需系数。
        """
        y = torch.ops._C_ascend.npu_hc_pre_v2(
            x, hc_fn, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.norm_eps, self.hc_eps
        )
        return y

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        """Hyper-Connections 后混合（NPU 融合算子 npu_hc_post）。

        原理: 把子层输出与残差按 hc_pre 给出的系数 (post, comb) 合并，
        恢复多路残差流（替代普通的 x + residual）。
        unsqueeze(0)/squeeze(0): 算子要求带 batch 维。
        """
        y = torch.ops._C_ascend.npu_hc_post(
            x.unsqueeze(dim=0), residual.unsqueeze(dim=0), post.unsqueeze(dim=0), comb.unsqueeze(dim=0)
        )
        return y.squeeze(dim=0)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """解码层前向。

        Args:
            positions: [num_tokens] 位置 id。
            hidden_states: [num_tokens, hc_mult, hidden_size] 多路隐状态。
            residual: 上一层的残差（首层为 None 时内部 clone 输入）。
            llama_4_scaling: 可选按位置缩放（未启用恒 None）。
            input_ids: [num_tokens] token id（hash MoE 路由用）。
        Returns:
            (hidden_states, residual): 本层输出与残差（V4 中 residual
            与输出同值，为兼容上游接口保留）。
        """
        # 保存残差（hc_post 需要原始输入流）。clone(): 不改原张量。
        residual = hidden_states.clone()
        full_num_tokens = positions.shape[0]
        # 子块1: 注意力——hc_pre 混合 -> input_layernorm -> self_attn ->
        # hc_post 合并残差。
        hidden_states, post, comb = self.hc_pre(hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        hidden_states = self.input_layernorm(hidden_states)

        if self.use_sequence_parallel_moe and not self.enable_dsa_cp:
            # SP MoE（非 DSA-CP）: 注意力需要完整 token 维——先 all-gather
            # 裁剪到真实长度，注意力后再 reduce-scatter 回切分状态。
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]

        # 注意力前向（kwargs 解包调用）。
        attn_kwargs = {"positions": positions, "hidden_states": hidden_states, "llama_4_scaling": llama_4_scaling}
        hidden_states = self.self_attn(**attn_kwargs)

        if self.use_sequence_parallel_moe and not self.enable_dsa_cp:
            # SP: 注意力输出 reduce-scatter 回各 rank。
            hidden_states = sp_reduce_scatter(hidden_states)

        # hc_post 合并注意力残差。
        hidden_states = self.hc_post(hidden_states, residual, post, comb)

        # 子块2: MoE——同样 hc_pre -> norm(带 FP32 路由输入) -> mlp ->
        # hc_post。
        residual = hidden_states.clone()
        hidden_states, post, comb = self.hc_pre(hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        # rms_norm_cast: 一次归一化同时得到 bf16 输入与 FP32 路由输入。
        hidden_states, hidden_states_fp32 = self.rms_norm_cast(hidden_states)
        hidden_states = self.mlp(
            hidden_states,
            input_ids=input_ids,
            hidden_states_fp32=hidden_states_fp32,
        )
        # hc_post 合并 MoE 残差。
        hidden_states = self.hc_post(hidden_states, residual, post, comb)

        return hidden_states, residual


@support_torch_compile
class DeepseekV4Model(nn.Module, EagleModelMixin):
    """DeepSeek V4 语言主干模型（堆叠 decoder 层）。

    继承/混入:
      - EagleModelMixin: EAGLE3 式投机采样的目标模型接口（aux hidden
        states 导出、make_empty_intermediate_tensors 等）。
    装饰器 @support_torch_compile: 参与 vLLM 的 piecewise 编译
    （以注意力/MoE 为边界切分编译区域，NPU 上消除 Python 开销）。

    支持特性:
      - PP 流水线并行: 首级 embedding、末级 norm; 中间级收发
        IntermediateTensors（supports_aux_hidden_states_over_pp=True:
        aux 隐状态随 PP 中间张量一起传输）;
      - sequence parallel MoE: token 维切分;
      - MTP 缓冲: 末级缓存 hc_head 之前的多路隐状态供草稿模型。
    """

    fall_back_to_pt_during_load = False
    # vLLM #50514 validates and relays the model's existing PP aux payload.
    # 【中文】类属性: 加载时不回退 PyTorch 原生路径; PP 间传输时直接
    # 复用模型自带的 aux 隐状态负载（不做重打包）。
    supports_aux_hidden_states_over_pp = True
    # PP 中间张量里 aux 隐状态的键名前缀。
    AUX_HIDDEN_STATE_KEY = "pp_transport_aux_hidden_states_"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        """初始化主干。

        Args:
            vllm_config: vLLM 全局配置。
            prefix: 模块名前缀。
        """
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = current_platform.device_type
        self.use_sequence_parallel_moe = vllm_config.parallel_config.use_sequence_parallel_moe

        self.vocab_size = config.vocab_size
        # is_v32: 带 index_topk 即 V3.2+ 结构（稀疏注意力 indexer）。
        self.is_v32 = hasattr(config, "index_topk")
        if self.is_v32:
            # IndexCache 共享缓冲: [max_num_batched_tokens, index_topk]，
            # 供多个 indexer 层复用 TopK 结果（也供 MTP 草稿共享）。
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
        # 【中文】暴露在模型级——spec_decode/llm_base_proposer 通过属性
        # 替换把该缓冲共享给 MTP 草稿模型。
        self.topk_indices_buffer = topk_indices_buffer

        if get_pp_group().is_first_rank:
            # PP 首级: 词嵌入（VocabParallel: 词表按 TP 切片）。
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            # 非首级: embedding 占位（PPMissingLayer 不会真正建参数）。
            self.embed_tokens = PPMissingLayer()
        # make_layers: 按 PP 级别裁剪层范围; lambda 闭包构造每层
        #（start_layer/end_layer 是本 rank 负责的层区间）。
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV4DecoderLayer(vllm_config, prefix, topk_indices_buffer=topk_indices_buffer),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            # PP 末级: 最终 RMSNorm。
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        def make_empty_intermediate_tensors(
            batch_size: int,
            dtype: torch.dtype,
            device: torch.device,
        ) -> IntermediateTensors:
            """构造 PP 通信用的空中间张量（闭包捕获 config）。"""
            return IntermediateTensors(
                {
                    # 多路隐状态: [batch, hc_mult, hidden]。
                    "hidden_states": torch.zeros(
                        (batch_size, self.hc_mult, config.hidden_size),
                        dtype=dtype,
                        device=device,
                    ),
                }
            )

        # 用 PP 工具包装: 自动附加 aux 隐状态等传输字段（vllm_ascend 的
        # v2 PP 传输协议）。
        self.make_empty_intermediate_tensors = make_pp_empty_intermediate_tensors(
            self,
            make_empty_intermediate_tensors,
        )

        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.hc_mult = hc_mult = config.hc_mult
        hc_dim = hc_mult * config.hidden_size

        # hc_head 参数: 多路隐状态 -> 单路的混合头（见 hc_head）。
        self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.hc_norm = RMSNorm(hc_dim, eps=config.rms_norm_eps, has_weight=False, dtype=torch.float32)

        # Pre-hc_head residual stream buffer for the speculative draft
        # (MTP / DSpark / DFlash). Only needed when the decoder consumes
        # target-model hidden states; allocating it unconditionally would
        # permanently cost max_num_batched_tokens * hc_dim per rank.
        # Aligned with upstream DeepSeekV4 (see vllm PR #50312).
        # 【中文】MTP 草稿需要“hc_head 之前”的目标隐状态。仅当处于 PP
        # 末级且启用投机采样（EAGLE 或独立草稿模型）才分配缓冲——
        # 无条件分配会永久占用 max_num_batched_tokens*hc_dim 的显存。
        spec_config = vllm_config.speculative_config
        self._needs_mtp_hidden_states = bool(
            get_pp_group().is_last_rank
            and spec_config is not None
            and (spec_config.use_eagle() or spec_config.uses_draft_model())
        )
        self._mtp_buffer_shape = (
            vllm_config.scheduler_config.max_num_batched_tokens,
            hc_dim,
        )
        self._mtp_buffer_dtype = vllm_config.model_config.dtype
        self._mtp_hidden_buffer: torch.Tensor | None = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """token id -> 嵌入。Args: input_ids: [num_tokens]。
        Returns: [num_tokens, hidden_size]。"""
        return self.embed_tokens(input_ids)

    def hc_head(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        """Hyper-Connections 输出混合头: hc_mult 路 -> 单路。

        算法: 与 mtp.py/dspark.py 的 hc_head 一致——RMSNorm 后经 fn 得
        门控系数，sigmoid(mixes*scale+base)+eps 加权求和; 全程 float32。

        Args:
            x: [N, hc_mult, H] 多路隐状态。
        Returns:
            [N, H] 单路隐状态（原 dtype）。
        """
        shape, dtype = x.size(), x.dtype
        x = x.flatten(1).float()
        x_norm = self.hc_norm(x)
        mixes = torch.nn.functional.linear(x_norm, hc_fn)
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        # 加权求和: [N, hc_mult, 1] 广播乘 [N, hc_mult, H] 后沿路维求和。
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
        return y.to(dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        """主干前向。

        Args:
            input_ids: [num_tokens] token id（PP 非首级为 None）。
            positions: [num_tokens] 位置 id。
            intermediate_tensors: PP 上游传来的中间张量（首级为 None）。
            inputs_embeds: 预合并的输入嵌入（多模态路径优先于 input_ids）。
        Returns:
            PP 非末级: IntermediateTensors（hidden_states + aux）;
            末级: hidden_states（或有 aux 时返回二元组）。
        """
        pp_group = get_pp_group()
        if pp_group.is_first_rank:
            # 首级: 从 input_ids 或 inputs_embeds 得到初始隐状态。
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            # 非首级: 从 PP 中间张量恢复隐状态。
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = None
        # 从中间张量取出上游累积的 aux 隐状态列表（DSpark 草稿需要）。
        aux_hidden_states = get_pp_transport_tensors(
            intermediate_tensors,
            PPTransportDataType.AUX_HIDDEN_STATES,
        )

        if self.use_sequence_parallel_moe:
            # SP: 更新 is_padding 掩码（跳过纯 padding token 的 MoE 计算）
            # 并把 token 维切分到各 rank。
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                forward_context.is_padding = sp_padding_mask(forward_context.is_padding, hidden_states)
            hidden_states = sp_shard(hidden_states)
            # Non-first PP ranks receive None input_ids (the embedding was
            # done upstream); only shard on the rank that owns the tokens.
            # 【中文】非首级 PP rank 的 input_ids 为 None（embedding 在
            # 上游完成），只在持有 token 的 rank 上切分。
            if input_ids is not None:
                input_ids = sp_shard(input_ids)

        # Compute llama 4 scaling once per forward pass if enabled
        # 【中文】llama4 缩放当前未启用（config 为 None -> scaling=None），
        # 保留代码路径供后续开启。
        llama_4_scaling_config = None
        llama_4_scaling: torch.Tensor | None
        if llama_4_scaling_config is not None:
            llama_4_scaling = _get_llama_4_scaling(
                original_max_position_embeddings=llama_4_scaling_config["original_max_position_embeddings"],
                scaling_beta=llama_4_scaling_config["beta"],
                positions=positions,
            )
        else:
            llama_4_scaling = None

        if pp_group.is_first_rank:
            # 单路嵌入 -> 复制为 hc_mult 路 (b, s, h) -> (b, s, c, h)
            #（Hyper-Connections 的多路残差流起点）。
            hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)  # (b, s, h) -> (b, s, c, h)
        # islice(layers, start, end): 只遍历本 PP rank 负责的层。
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                llama_4_scaling,
                input_ids=input_ids,
            )
            # 该层被选中导出 aux 隐状态（DSpark 的 target_layer_ids）
            # 时: 对多路取均值得到单路并收集。SP 下先 all-gather。
            if layer.layer_idx + 1 in self.aux_hidden_state_layers:
                aux_hidden_state = hidden_states.mean(dim=1)
                if self.use_sequence_parallel_moe:
                    aux_hidden_state = sp_all_gather(aux_hidden_state)[: positions.shape[0]]
                aux_hidden_states.append(aux_hidden_state)

        if not pp_group.is_last_rank:
            # The next PP rank expects full-sequence hidden states; undo the
            # sequence sharding applied above before crossing the PP boundary.
            # 【中文】非末级: 下一 PP 级期望完整序列——先撤销 SP 切分
            #（all-gather 并裁剪回真实 token 数），再打包发送。
            if self.use_sequence_parallel_moe:
                hidden_states = sp_all_gather(hidden_states)[: positions.shape[0]]
            intermediate_tensors = IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                }
            )
            # 附加 aux 隐状态负载到中间张量（v2 PP 传输协议）。
            return add_pp_transport_tensors(
                intermediate_tensors,
                PPTransportDataType.AUX_HIDDEN_STATES,
                aux_hidden_states,
            )

        if self.use_sequence_parallel_moe:
            # 末级: 撤销 SP 切分。
            hidden_states = sp_all_gather(hidden_states)[: positions.shape[0]]

        # Stash pre-hc_head residual for the MTP draft (captured copy_).
        # 【中文】缓存“hc_head 之前”的多路隐状态给 MTP/DSpark 草稿
        #（惰性分配缓冲; copy_ 拷贝前 num_tokens 行）。
        if self._needs_mtp_hidden_states:
            if self._mtp_hidden_buffer is None:
                self._mtp_hidden_buffer = torch.empty(
                    self._mtp_buffer_shape,
                    dtype=self._mtp_buffer_dtype,
                    device=self.device,
                )
            num_tokens = hidden_states.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))

        # 多路隐状态 -> hc_head 混合为单路。
        hidden_states = self.hc_head(hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)

        # 最终 RMSNorm。
        hidden_states = self.norm(hidden_states)
        if len(aux_hidden_states) > 0:
            # 有 aux 隐状态时一并返回（DSpark 投机采样路径）。
            return hidden_states, aux_hidden_states
        return hidden_states


class DeepseekV2MixtureOfExperts(MixtureOfExperts):
    """MoE 元数据混入类（继承上游 vLLM 的 MixtureOfExperts 接口）。

    作用: 为 EPLB（专家并行负载均衡）与调度器提供统一的专家统计信息。
    被 AscendDeepseekV4ForCausalLM / DeepSeekV4MTP / DSparkDeepseekV4ForCausalLM
    三处多重继承复用。
    """

    moe_mlp_layers: list[DeepseekV4MoE]
    """
    List of MoE MLP layers in the model.
    """
    # 【中文】类型注解的类属性声明: 模型中全部 MoE 层的列表（由
    # set_moe_parameters 收集填充）。

    def extract_moe_parameters(self, example_moe: DeepseekV4MoE | None):
        """从样例 MoE 层提取专家统计; 无 MoE 层时全部置零。

        Args:
            example_moe: 任一 MoE 层实例（通常取最后一个）; None 表示
                本模型（或本 PP rank）没有 MoE 层。
        """
        if example_moe is None:
            # 无 MoE 层: 全部计数清零。
            self.num_moe_layers = 0
            self.num_expert_groups = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_shared_experts = 0
            self.num_redundant_experts = 0
        else:
            # 从样例层读取专家统计。
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
        """EPLB 运行时更新物理专家元数据（冗余专家重平衡后调用）。

        原理: EPLB 会动态重排逻辑专家 -> 物理专家的映射（增加/迁移冗余
        副本以均衡负载）; 映射更新后需同步每个 MoE 层的专家数与专家表
        （experts.update_expert_map）。
        """
        # 一致性断言: 调用方与本模型认知的本地专家数必须一致。
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        # 冗余数 = 物理 - 逻辑。
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        # 逐层同步统计并刷新专家映射表。
        for moe in self.moe_mlp_layers:
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()


class AscendDeepseekV4ForCausalLM(nn.Module, SupportsPP, DeepseekV2MixtureOfExperts, SupportsLoRA, SupportsEagle3):
    """DeepSeek V4 CausalLM 顶层模型（昇腾实现，vl_model 的语言主干也用它）。

    多重继承:
      - SupportsPP   : 流水线并行接口;
      - DeepseekV2MixtureOfExperts: MoE/EPLB 元数据混入;
      - SupportsLoRA : LoRA 微调接口（packed_modules_mapping 声明融合模块）;
      - SupportsEagle3: EAGLE3 投机采样目标模型接口。
    职责: 组装 model（主干）+ lm_head（输出头）+ logits_processor，
    并实现权重加载 load_weights。
    """

    # 融合模块映射: 告诉 LoRA/加载器 gate_up_proj 由 gate_proj/up_proj
    # 两个 checkpoint 权重堆叠而成。
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    # 主干类（vl_model 不改动它; dspark 用自己的模型类）。
    model_cls = DeepseekV4Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        """初始化。

        Args:
            vllm_config: vLLM 全局配置（keyword-only）。
            prefix: 模块名前缀。
        """
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        # 语言主干。
        self.model = self.model_cls(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            # PP 末级: lm_head（词表并行输出投影）。
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        # 把 PP 中间张量构造器提升到本类。
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        # Set MoE hyperparameters
        # 【中文】收集 MoE 元数据（EPLB 用）。
        self.num_moe_layers = self.config.num_hidden_layers
        self.set_moe_parameters()

    def set_moe_parameters(self):
        """遍历主干层收集 MoE 模块并提取专家统计（同 mtp.py 逻辑）。"""
        self.expert_weights = []

        self.num_expert_groups = getattr(self.config, "n_group", 1)

        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, DeepseekV4DecoderLayer)
            if isinstance(layer.mlp, DeepseekV4MoE):
                # Pick last one layer since the first ones may be dense layers.
                # 【中文】持续覆盖，留下最后一个 MoE 层作样例。
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)

        self.extract_moe_parameters(example_moe)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """token id -> 嵌入（委托主干; 多模态包装类经此共享词嵌入）。"""
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        """前向（委托主干）。

        Args:
            input_ids/positions: token 与位置 id。
            intermediate_tensors: PP 上游中间张量。
            inputs_embeds: 预合并嵌入（多模态）。
        Returns:
            隐状态或 PP 中间张量。
        """
        hidden_states = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """由隐状态计算 logits。

        Args:
            hidden_states: [num_tokens, hidden_size]。
        Returns:
            [num_tokens, vocab_size]（logits_processor 内部按需只算最后
            一个 token 等）。
        """
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """返回专家参数映射（EPLB 权重重排用）。

        Returns:
            (param_name, weight_name, expert_id, shard_id) 四元组列表。
        """
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        # 【中文】为权重/FP8 权重 scale/FP8 激活 scale 生成映射;
        # mix_placement 时共享专家计入专家总数。
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
        # 【中文】返回 hc_head 之前的目标隐状态缓冲（SupportsEagle3 接口）:
        # [max_num_batched_tokens, hc_mult*hidden_size]，每次目标前向后有效，
        # 供 MTP/DSpark 草稿作为 previous_hidden_states。
        return getattr(self.model, "_mtp_hidden_buffer", None)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        """设置需导出 aux 隐状态的层号（DSpark 的 target 层，投机采样
        框架在启动时调用）。"""
        self.model._set_aux_hidden_state_layers(layers)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """加载目标模型权重（含 DeepSeek 原始命名 -> vLLM 命名的重写）。

        流程: 逐条权重 -> 跳过 mtp.*（草稿）-> 名称重写 -> 三类分派:
        1) attention sink: 按头切分（或 DSA-CP 全量）;
        2) 堆叠权重（gate/up -> gate_up_proj）;
        3) 专家权重（expert_params_mapping + expert_id）/ 普通权重。

        Args:
            weights: (名字, 张量) 流（safetensors）。
        Returns:
            成功加载的参数名集合。
        """
        # ROCm AITER 开关（随即被 Ascend mix_placement 覆盖，后者生效）。
        rocm_aiter_moe_shared_expert_enabled = rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        rocm_aiter_moe_shared_expert_enabled = getattr(get_ascend_config(), "mix_placement", False)
        # 堆叠权重映射: gate_proj/up_proj 合并进 gate_up_proj（shard 0/1）。
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        # 【中文】专家参数映射（权重/FP8 scale 各一条）; mix_placement 时
        # 共享专家计入专家总数。
        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self.model,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts
            + (self.config.n_shared_experts if rocm_aiter_moe_shared_expert_enabled else 0),
            num_redundant_experts=self.num_redundant_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        # Attention heads per rank
        # 【中文】attention sink 按头切分: 每卡的头区间 [start, end)。
        heads_per_rank = self.config.num_attention_heads // tp_size
        head_start = tp_rank * heads_per_rank

        for name, loaded_weight in weights:
            # 跳过 mtp.*（草稿模型权重由各自的 loader 负责）。
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model

            # TODO:
            # 【中文】顶层权重统一补 "model." 前缀。
            if not name.startswith("model"):
                name = f"model.{name}"

            # DeepSeek 原始命名（w1/w2/w3）-> vLLM 命名。
            if ".w1." in name:
                name = name.replace(".w1.", ".gate_proj.")
            if ".w2." in name:
                name = name.replace(".w2.", ".down_proj.")
            if ".w3." in name:
                name = name.replace(".w3.", ".up_proj.")

            # head/lm_head/embed 的命名统一。
            if "model.head." in name and "model.lm_head." not in name:
                name = name.replace("model.head.", "lm_head.")
            if "model.lm_head." in name:
                name = name.replace("model.lm_head.", "lm_head.")
            if "embed." in name and "embed_token." not in name:
                name = name.replace("embed.", "embed_tokens.")
            # attn/ffn/ffn_norm/attn_norm -> vLLM 命名。
            if "attn" in name and "self_attn" not in name:
                name = name.replace(".attn.", ".self_attn.")
            if ".ffn." in name:
                name = name.replace(".ffn.", ".mlp.")
            if ".ffn_norm." in name:
                name = name.replace(".ffn_norm.", ".post_attention_layernorm.")
            if ".attn_norm." in name:
                name = name.replace(".attn_norm.", ".input_layernorm.")
            # Ascend 约定: 量化 scale -> weight_scale。
            if name.endswith(".scale"):
                name = name.replace(".scale", ".weight_scale")

            # RoPE 频率表不加载（运行时预计算）。
            if "rotary_emb.inv_freq" in name:
                continue
            if ".gate.bias_vl" in name:
                # The parameter keeps the checkpoint name on Ascend. It is
                # passed to the hash router as its vision-only correction
                # bias, while text rows continue to use tid2eid.
                # 【中文】bias_vl 保持 checkpoint 名（hash 路由器的
                # 视觉专用纠偏; 文本 token 走 tid2eid 表）。
                pass
            elif ".gate.bias" in name:
                # 普通路由偏置 -> e_score_correction_bias。
                name = name.replace(".gate.bias", ".gate.e_score_correction_bias")

            # Hash-router layers route text tokens through ``tid2eid`` and keep
            # ``e_score_correction_bias`` unset, but the checkpoint still ships
            # a router bias for them. Skip it instead of raising a KeyError.
            # 【中文】hash 路由层没有 e_score_correction_bias 参数，但
            # checkpoint 仍带偏置——跳过而非 KeyError。
            if name.endswith(".gate.e_score_correction_bias") and name not in params_dict:
                continue

            # ---- attention sink 特殊加载 ----
            if "sink" in name:
                # PP: 不属于本流水级的参数跳过。
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                if enable_dsa_cp():
                    # DSA-CP: 全量 sink（每卡全部头）。
                    param.data.copy_(loaded_weight)
                else:
                    # Handle attention sinks (distributed across ranks)
                    # 【中文】普通 TP: 切出本 rank 的头区间。
                    narrow_weight = loaded_weight.narrow(0, head_start, heads_per_rank)
                    param.data.copy_(narrow_weight)
                loaded_params.add(name)
                continue

            # ---- 堆叠 / 专家 / 普通权重 ----
            is_fusion_moe_shared_experts_layer = rocm_aiter_moe_shared_expert_enabled and ("mlp.shared_experts" in name)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                # 【中文】跳过非堆叠层; 专家权重在下方单独处理（必须在
                # 改名前判断，避免 gate_gate_up_proj 双重替换）。
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                if is_fusion_moe_shared_experts_layer:
                    continue
                name_mapped = name.replace(weight_name, param_name)

                # QKV fusion is optional, fall back to normal
                # weight loading if it's not enabled
                # if go with fusion option, then update name
                # 【中文】可选 QKV 融合: 融合参数不存在则回退普通加载。
                if (param_name == "fused_qkv_a_proj") and name_mapped not in params_dict:
                    continue
                else:
                    name = name_mapped
                # Skip loading extra bias for GPTQ models.
                # 【中文】GPTQ 额外 bias 跳过。
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                # 堆叠权重: shard_id 指明装入融合参数的哪一半。
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # for...else: 未匹配堆叠规则 -> 专家/普通权重路径。
                is_expert_weight = False

                # Special handling: when AITER fusion_shared_experts is enabled,
                # checkpoints may provide a single widened shared_experts tensor
                # without explicit expert indices
                # (e.g. ...mlp.shared_experts.gate_proj.weight).
                # For models with multiple shared experts, split that tensor
                # evenly into per-shared-expert slices and load them into
                # appended expert slots mlp.experts.{n_routed_experts + j}.*
                # accordingly.
                # 【中文】mix_placement: 单个加宽的 shared_experts 张量均分
                # 为 n_shared_experts 份，装入追加的专家槽位。
                num_chunks = 1
                if is_fusion_moe_shared_experts_layer:
                    num_chunks = getattr(self.config, "n_shared_experts", 1) or 1
                    # Determine split axis based on op type
                    # gate/up: ColumnParallel → split along dim 0
                    # down: RowParallel → split along dim 1
                    # 【中文】按并行方式选切分轴: Column 沿 dim0，Row 沿 dim1。
                    split_dim = 1 if "down_proj.weight" in name else 0
                    total = loaded_weight.shape[split_dim]
                    # 整除断言: 不整除说明模型结构与 checkpoint 不符。
                    assert total % num_chunks == 0, (
                        f"Shared expert weight dim {total} not divisible by num_chunks {num_chunks}"
                    )
                    chunk_size = total // num_chunks

                for j in range(num_chunks):
                    chunk_name = name
                    weight_to_load = loaded_weight

                    if is_fusion_moe_shared_experts_layer:
                        # 切出第 j 个共享专家的权重片。
                        if split_dim == 0:
                            weight_to_load = loaded_weight[j * chunk_size : (j + 1) * chunk_size, :]
                        else:
                            weight_to_load = loaded_weight[:, j * chunk_size : (j + 1) * chunk_size]
                        # Synthesize an expert-style name so expert mapping
                        # can route it
                        # 【中文】合成专家式名字让映射路由。
                        chunk_name = name.replace(
                            "mlp.shared_experts",
                            f"mlp.experts.{self.config.n_routed_experts + j}",
                        )

                    # Use expert_params_mapping to locate the destination
                    # param and delegate to its expert-aware weight_loader
                    # with expert_id.
                    # 【中文】专家权重: 命中映射后用带 expert_id 的 loader。
                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in chunk_name:
                            continue

                        # Anyway, this is an expert weight and should not be
                        # attempted to load as other weights later
                        # 【中文】匹配即认定为专家权重。
                        is_expert_weight = True

                        # Do not modify `name` since the loop may continue here
                        # Instead, create a new variable
                        # 【中文】不改 name（循环可能继续），用新变量。
                        name_mapped = chunk_name.replace(weight_name, param_name)

                        # PP: 不属于本流水级跳过。
                        if is_pp_missing_parameter(name_mapped, self):
                            continue

                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or
                        # not here since otherwise we may skip experts with
                        # other available replicas.
                        # 【中文】要求 loader 返回成败（EPLB 冗余专家副本
                        # 场景）; typing.cast 为类型断言。
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
                        if is_expert_weight:
                            # We've checked that this is an expert weight
                            # However it's not mapped locally to this rank
                            # So we simply skip it
                            # 【中文】专家权重不在本 rank: 跳过。
                            continue

                        # Skip loading extra bias for GPTQ models.
                        # 【中文】GPTQ 额外 bias 跳过。
                        if name.endswith(".bias") and name not in params_dict:
                            continue

                        # Remapping the name of FP8 kv-scale.
                        # 【中文】FP8 kv-scale 名字重映射; 无对应参数则跳过。
                        name = maybe_remap_kv_scale_name(name, params_dict)
                        if name is None:
                            continue

                        if is_pp_missing_parameter(name, self):
                            continue

                        # 普通参数: 有专用 loader（量化）用之，否则默认拷贝。
                        param = params_dict[name]
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight)
            if not is_fusion_moe_shared_experts_layer:
                loaded_params.add(name)

        return loaded_params
