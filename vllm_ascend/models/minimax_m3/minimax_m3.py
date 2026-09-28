# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The MiniMax AI team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""Inference-only MiniMaxM3 model."""
"""（MiniMax-M3 推理实现 —— 昇腾 NPU 适配版。）

【模型结构概览】
MiniMax M3 = 标准 GQA 注意力/MSA 块稀疏注意力混合 + SwiGLU-OAI MoE：
- 注意力按层配置两种形态（sparse_attention_config.sparse_attention_freq）：
  * 普通层：GQA 全注意力（MiniMaxM3Attention，Q/K 各过 GemmaRMSNorm）；
  * 稀疏层：块稀疏注意力 + 闪电索引器（MiniMaxM3SparseAttention）——
    indexer 用低维 index_q/index_k 给历史 block 打分，选出 top-k 个
    block 只做精确注意力（DeepSeek-V3.2 式 lightning indexer 思路），
    主注意力只在被选 block + 局部 block + 初始 block 上计算；
- MLP：MoE 层（无辅助损失路由可选 bias + 共享专家 + EPLB 冗余专家）
  与 dense 层（SwiGLU-OAI：带 alpha/beta/limit 参数的 SwiGLU 变体）混合；
- 投机解码：支持 EAGLE3（aux hidden states 捕获 + MTP 层权重过滤）。

【NPU 适配点】
- 融合算子 qkv_rmsnorm_rope（QKV 切分 + GemmaRMSNorm + RoPE 一次完成）；
- torch.ops.vllm.minimax_m3_sparse_forward 自定义算子封装稀疏前向
 （torch.compile 图内可调用，按 layer_name 从 static_forward_context 寻址）；
- A5 硬件分支：npu_scatter_pa_cache 分页 cache 写入、SwiGLU 手写路径、
  MXFP8 动态量化。
"""

from collections.abc import Iterable, MutableSequence, Sequence
from itertools import islice
from typing import Any

import torch
import torch_npu
from torch import nn
from transformers import PretrainedConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    GateLinear,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)

from vllm_ascend.device.device_op import DeviceOperator
# MSA 稀疏注意力的核心组件（indexer/后端/融合 QKV 投影）来自 msa_m3.py。
from vllm_ascend.models.minimax_m3.msa_m3 import (
    AscendMiniMaxM3Indexer,
    AscendMiniMaxM3IndexerLinear,
    AscendMiniMaxM3IndexerMetadata,
    AscendMiniMaxM3QKVParallelLinearWithIndexer,
    AscendMiniMaxM3SparseBackend,
    AscendMiniMaxM3SparseImpl,
    AscendMiniMaxM3SparseMetadata,
    _register_m3_sparse_packed_modules,
    _use_fused_qkv_indexer,
)
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type
# PP 流水线并行的中间张量传输工具（aux 隐状态随流水线透传给 EAGLE3 草稿）。
from vllm_ascend.worker.v2.pp_utils import (
    PPTransportDataType,
    add_pp_transport_tensors,
    get_pp_transport_tensors,
)
from vllm_ascend.worker.v2.pp_utils import (
    make_empty_intermediate_tensors as make_pp_empty_intermediate_tensors,
)

# FP8 E4M3 的最大可表示值（448）：转 FP8 前必须 clamp 防溢出为 inf/nan。
_FP8_E4M3_MAX = 448.0


def _resolve_layer_kv_cache_dtype(
    cache_config: CacheConfig | None,
    prefix: str,
) -> str:
    """Resolve the per-layer KV dtype using vLLM's skip-layer semantics."""
    # （按 vLLM 的"跳过层"语义解析逐层 KV dtype。）
    # kv_cache_dtype_skip_layers：指定的层不量化（保持 auto），
    # MiniMax-M3 用它让"稀疏注意力层 + 索引器层"绕过全局 FP8 设置。
    if cache_config is None:
        return "auto"

    kv_cache_dtype = cache_config.cache_dtype
    skip_layers = getattr(cache_config, "kv_cache_dtype_skip_layers", None)
    if skip_layers and str(extract_layer_index(prefix)) in skip_layers:
        kv_cache_dtype = "auto"
    return kv_cache_dtype


def _resolve_layer_kv_cache_dtypes(
    cache_config: CacheConfig | None,
    prefix: str,
    model_config: ModelConfig,
) -> tuple[str, torch.dtype]:
    """Resolve MiniMax-M3's semantic and physical per-layer KV dtypes."""
    # （解析 M3 的"语义 dtype 字符串"与"物理 torch dtype"二元组。）
    kv_cache_dtype = _resolve_layer_kv_cache_dtype(cache_config, prefix)
    if kv_cache_dtype in ("fp8", "fp8_e4m3"):
        # vLLM represents generic FP8 KV storage as uint8. MiniMax-M3's
        # fixed-scale sparse-attention and indexer paths consume native E4M3
        # tensors, so preserve that physical dtype at the cache-spec boundary.
        # （vLLM 通用 FP8 KV 用 uint8 存储；M3 的固定 scale 稀疏注意力与
        #   索引器直接消费原生 E4M3，因此在 cache-spec 边界保留该物理 dtype。）
        kv_cache_torch_dtype = torch.float8_e4m3fn
    else:
        kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(kv_cache_dtype, model_config)
    return kv_cache_dtype, kv_cache_torch_dtype


def _cast_for_cache(tensor: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
    """Cast fixed-scale MiniMax-M3 KV values to the physical cache dtype."""
    # （把固定 scale 的 KV 值转到物理 cache dtype。）
    # M3 的 FP8 KV 用固定 scale（不做 per-token 动态量化），
    # 因此转 FP8 只需 clamp 到表示范围后直接 cast。
    if tensor.dtype == cache.dtype:
        return tensor
    if cache.dtype == torch.float8_e4m3fn:
        tensor = tensor.clamp(min=-_FP8_E4M3_MAX, max=_FP8_E4M3_MAX)
    return tensor.to(cache.dtype)


def _scatter_index_cache(
    cache: torch.Tensor,
    updates: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Write MiniMax-M3 index keys into the paged cache."""
    # （把索引器的 index key 写入分页 cache。）
    # 索引器 cache 与主 KV cache 分开存储（只存低维 index key），
    # 写入方式按硬件分两条路径。
    slots = slot_mapping.reshape(-1)
    if slots.numel() == 0:
        return

    updates = updates.reshape(slots.shape[0], cache.shape[-1])
    if updates.dtype != cache.dtype:
        updates = updates.to(cache.dtype)

    if get_ascend_device_type() == AscendDeviceType.A5:
        # A5 路径：npu_scatter_pa_cache 是 CANN 的"分页注意力 cache 散射"
        # 专用算子（要求 3D key 布局 [num, 1, dim]，cache 增加一个头维）。
        if cache.ndim != 3:
            raise ValueError(f"Unexpected MiniMax-M3 index cache ndim on A5: {cache.ndim}")
        key = updates.reshape(slots.shape[0], 1, cache.shape[-1]).contiguous()
        key_cache = cache.unsqueeze(2)
        torch_npu.npu_scatter_pa_cache(
            key,
            slots.contiguous(),
            key_cache=key_cache,
        )
        return

    # 非 A5 路径：展平 cache 成 [num_slots, dim] 后用自定义散射更新算子。
    flat_cache = cache.view(-1, cache.shape[-1])
    torch.ops._C_ascend.npu_scatter_nd_update_sk(
        flat_cache,
        slots.view(-1, 1),
        updates,
    )


class MiniMaxM3SparseAttention(nn.Module, AttentionLayerBase):
    """Block-sparse attention with lightning indexer on Ascend."""
    """（块稀疏注意力 + 闪电索引器 —— 昇腾实现。）

    【原理】把上下文切成 sparse_block_size 大小的 block；
    indexer 用低维 index_q（每 KV 头一个 index 头）对全部历史 block 打分，
    选出 topk_blocks 个 block；主注意力（GQA）只对
    [初始 block + 局部滑窗 block + 被选 top-k block] 做精确计算。
    长上下文下注意力复杂度从 O(L²) 降到 O(L·(topk+local))。

    【类设计】多继承 nn.Module + AttentionLayerBase（vLLM 注意力层协议）：
    实现协议方法（get_kv_cache_spec 申报缓存布局 / get_attn_backend 申报后端 /
    注册 static_forward_context），使该层能被 vLLM v1 的统一 KV cache
    规划器和 torch.compile 图正确接管。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rotary_dim: int,
        rope_parameters: dict[str, Any] | None = None,
        attn_window_size: int | None = None,
        max_position_embeddings: int = 8192,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        sparse_cfg: dict[str, Any] | None = None,
        disable_index_value: bool = False,
        reduce_results: bool = True,
    ) -> None:
        """sparse_cfg 关键项：sparse_num_index_heads（index 头数=KV 头数）、
        sparse_index_dim（index 向量维）、sparse_topk_blocks（选块数）、
        sparse_block_size（块大小）、sparse_init_block/sparse_local_block
        （初始/局部保底块数）。disable_index_value：关闭 index_v 分支。
        """
        super().__init__()
        assert sparse_cfg is not None
        self.hidden_size = hidden_size
        self.disable_index_value = disable_index_value

        # ---- TP 切分计算：Q/KV 头按 TP world size 均分 ----
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # KV 头少于 TP size 时复制 KV 头（GQA 常规处理）。
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        # ---- 索引器维度：index 头数必须等于 KV 头数（每个 KV 头独立打分）----
        self.total_idx_heads = sparse_cfg["sparse_num_index_heads"]
        self.idx_head_dim = sparse_cfg["sparse_index_dim"]
        assert self.total_idx_heads == self.total_num_kv_heads, (
            "MiniMax M3 sparse attention requires sparse_num_index_heads == num_key_value_heads"
        )
        self.num_idx_heads = self.num_kv_heads
        self.index_q_size = self.num_idx_heads * self.idx_head_dim
        # 融合开关：把 indexer 的 index_q/index_k 投影并入主 QKV 投影
        #（一个 GEMM 出全部 5 段，省 kernel 启动）。
        self._use_fused_qkv_indexer = _use_fused_qkv_indexer(quant_config, prefix)
        _register_m3_sparse_packed_modules(quant_config, self._use_fused_qkv_indexer)

        if self._use_fused_qkv_indexer:
            # 融合路径：Q|K|V|index_q|index_k 五段一体的并行投影。
            self.qkv_proj = AscendMiniMaxM3QKVParallelLinearWithIndexer(
                hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                self.total_idx_heads,
                self.idx_head_dim,
                bias=qkv_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
            self.indexer_proj = None
        else:
            # 分离路径：主 QKV 投影 + 独立的 indexer 双投影。
            self.qkv_proj = QKVParallelLinear(
                hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=qkv_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
            self.indexer_proj = AscendMiniMaxM3IndexerLinear(
                hidden_size,
                self.total_idx_heads,
                self.idx_head_dim,
                bias=qkv_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.indexer_proj",
            )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # partial RoPE：只旋转 head_dim 的前 rotary_dim 维（partial_rotary_factor）。
        if rope_parameters is not None and "partial_rotary_factor" not in rope_parameters:
            rope_parameters["partial_rotary_factor"] = rotary_dim / self.head_dim
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
        )

        # GemmaRMSNorm 与 RMSNorm 的区别：输出乘 (1 + weight) 而非 weight，
        # 使零初始化权重对应恒等变换（Gemma 训练稳定性技巧）。
        # 主注意力与索引器各有 q/k 两套 norm。
        self.index_q_norm = GemmaRMSNorm(self.idx_head_dim, eps=rms_norm_eps)
        self.index_k_norm = GemmaRMSNorm(self.idx_head_dim, eps=rms_norm_eps)

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=rms_norm_eps)

        # ---- 注册进 vLLM 编译期前向上下文（torch.compile 图内按层名寻址）----
        vllm_config = get_current_vllm_config()
        self.layer_name = f"{prefix}.attn"
        self.kv_cache_dtype, self.kv_cache_torch_dtype = _resolve_layer_kv_cache_dtypes(
            cache_config,
            prefix,
            vllm_config.model_config,
        )
        self.attn_backend = AscendMiniMaxM3SparseBackend
        # impl：真正执行稀疏注意力的后端实现（含 topk 元数据）。
        self.impl = AscendMiniMaxM3SparseImpl(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            kv_cache_dtype=self.kv_cache_dtype,
            topk_blocks=sparse_cfg["sparse_topk_blocks"],
            sparse_block_size=sparse_cfg["sparse_block_size"],
        )
        self.topk_blocks = sparse_cfg["sparse_topk_blocks"]
        self.sparse_block_size = sparse_cfg["sparse_block_size"]
        # indexer：闪电索引器（持有独立的 index cache 并产出 topk 块索引）。
        self.indexer = AscendMiniMaxM3Indexer(
            num_kv_heads=self.num_kv_heads,
            scale=self.scaling,
            topk_blocks=self.topk_blocks,
            sparse_block_size=self.sparse_block_size,
            num_index_heads=self.num_idx_heads,
            index_head_dim=self.idx_head_dim,
            prefix=self.layer_name,
            init_blocks=sparse_cfg.get("sparse_init_block", 0),
            local_blocks=sparse_cfg.get("sparse_local_block", 0),
            cache_config=cache_config,
            kv_cache_dtype=self.kv_cache_dtype,
            kv_cache_torch_dtype=self.kv_cache_torch_dtype,
        )

        # 层名唯一性校验 + 注册（编译图中的自定义算子凭层名找到本层实例）。
        compilation_config = vllm_config.compilation_config
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self
        # kv_cache 占位：运行时由 KV cache 分配器填充真实张量。
        self.kv_cache = torch.tensor([])

    def get_attn_backend(self) -> type[AscendMiniMaxM3SparseBackend]:
        # AttentionLayerBase 协议：向 vLLM 申报本层使用的注意力后端类。
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        # AttentionLayerBase 协议：申报主注意力的缓存布局
        #（全注意力 paged KV，头数/头维/dtype，供显存规划器计算预算）。
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    def _insert_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        index_key: torch.Tensor,
    ) -> None:
        """把本批 K/V 与索引 key 写入各自的 paged cache。

        key/value: [num_tokens, kv_size]；index_key: [num_tokens, idx 维]。
        主 KV 走 reshape_and_cache 融合算子；索引 cache 走 _scatter_index_cache。
        """
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return
        # 从前向上下文取本层与索引器的元数据（槽位映射等）。
        main_meta = attn_metadata[self.layer_name]
        index_meta = attn_metadata[self.indexer.index_cache.prefix]
        assert isinstance(main_meta, AscendMiniMaxM3SparseMetadata)
        assert isinstance(index_meta, AscendMiniMaxM3IndexerMetadata)

        from vllm_ascend.device.device_op import DeviceOperator

        # kv_cache 是 (key_cache, value_cache) 二元组（分配器注入）。
        key_cache, value_cache = self.kv_cache
        num_tokens = main_meta.num_actual_tokens
        # 步骤1：主 K/V 变形 + dtype 转换后写入分页 cache。
        k_insert = key[:num_tokens].view(-1, self.num_kv_heads, self.head_dim)
        v_insert = value[:num_tokens].view(-1, self.num_kv_heads, self.head_dim)
        k_insert = _cast_for_cache(k_insert, key_cache)
        v_insert = _cast_for_cache(v_insert, value_cache)
        DeviceOperator.reshape_and_cache(
            k_insert,
            v_insert,
            key_cache,
            value_cache,
            main_meta.slot_mapping[:num_tokens],
        )

        # 步骤2：索引 key 写入独立的索引 cache（tuple 时取第一段）。
        idx_cache = self.indexer.index_cache.kv_cache
        if isinstance(idx_cache, (tuple, list)):
            idx_cache = idx_cache[0]
        _scatter_index_cache(
            idx_cache,
            index_key[:num_tokens],
            index_meta.slot_mapping[:num_tokens],
        )

    def _sparse_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """投影 + 归一化 + RoPE，产出主注意力的 q/k/v 与索引器的 index_q/index_k。"""
        # 步骤1：QKV（可能含索引）投影。
        qkv, _ = self.qkv_proj(hidden_states)
        main_qkv_size = self.q_size + 2 * self.kv_size
        if self.indexer_proj is None:
            # 融合路径：从拼接结果里 narrow 切出主 QKV 与 index_q/index_k 三段
            #（narrow 是零拷贝视图）。
            main_qkv = qkv.narrow(-1, 0, main_qkv_size)
            index_q = qkv.narrow(-1, main_qkv_size, self.index_q_size)
            index_k = qkv.narrow(
                -1,
                main_qkv_size + self.index_q_size,
                self.idx_head_dim,
            )
        else:
            # 分离路径：主投影 + 独立索引投影。
            main_qkv = qkv
            index_qk, _ = self.indexer_proj(hidden_states)
            index_q, index_k = index_qk.split(
                [self.index_q_size, self.idx_head_dim],
                dim=-1,
            )

        # 步骤2：主 QKV 的 norm + RoPE。
        # 融合分支条件：非 A5 && NPU && bf16 && 1D positions && NeoX 风格 ——
        # 满足时走 CANN 融合算子（一次完成切分+norm+RoPE）；
        # 否则走逐算子参考路径（兼容性兜底）。
        if (
            get_ascend_device_type() == AscendDeviceType.A5
            or main_qkv.device.type != "npu"
            or main_qkv.dtype != torch.bfloat16
            or positions.ndim != 1
            or not getattr(self.rotary_emb, "is_neox_style", True)
        ):
            q, k, v = main_qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            v = v.contiguous()
            q, k = self._qk_norm(q, k)
            q, k = self.rotary_emb(positions, q, k)
        else:
            # 融合算子：注意 GemmaRMSNorm 的 (1+w) 在这里体现为传 1.0+weight。
            q, k, v = torch.ops.vllm.qkv_rmsnorm_rope(
                input=main_qkv.contiguous(),
                q_weight=1.0 + self.q_norm.weight,
                k_weight=1.0 + self.k_norm.weight,
                q_hidden_size=self.q_size,
                kv_hidden_size=self.kv_size,
                head_dim=self.head_dim,
                eps=self.q_norm.variance_epsilon,
                q_bias=None,
                k_bias=None,
                cos_sin_cache=self.rotary_emb.cos_sin_cache,
                positions=positions,
            )

        # 步骤3：索引 q/k 的 norm + RoPE。
        # FP8 索引 cache 时用 out_dtype 直接产出 E4M3（量化在 RoPE 内完成）。
        index_q, index_k = self._index_qk_norm(index_q, index_k)
        if self.indexer.index_cache.dtype == torch.float8_e4m3fn:
            index_q, index_k = self.rotary_emb(
                positions,
                index_q,
                index_k,
                out_dtype=torch.float8_e4m3fn,
            )
        else:
            index_q, index_k = self.rotary_emb(positions, index_q, index_k)

        return q, k, v, index_q, index_k

    def _run_sparse_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        index_query: torch.Tensor,
        index_key: torch.Tensor,
        attn_output: torch.Tensor,
    ) -> None:
        """Insert KV, build sparse top-k indices, then run sparse attention."""
        # （写 KV → 索引器选 top-k 块 → 稀疏注意力内核。）
        # 三步：缓存写入、topk 块选择、稀疏注意力计算（结果写入 attn_output）。
        self._insert_kv(key, value, index_key)
        topk_idx = self.indexer(index_query)
        self.impl.forward(self, query, self.kv_cache, topk_idx, attn_output)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """稀疏注意力前向：prepare → 自定义算子 → o_proj。"""
        q, k, v, index_q, index_k = self._sparse_prepare(positions, hidden_states)
        attn_out = torch.empty_like(q)
        # 自定义算子封装：torch.compile 把整段稀疏前向当作一个 opaque 算子，
        # 运行时按 layer_name 从 static_forward_context 找回本层实例，
        # 调用 _run_sparse_attention 完成实际计算（图编译友好）。
        torch.ops.vllm.minimax_m3_sparse_forward(
            q,
            k,
            v,
            index_q,
            index_k,
            attn_out,
            self.layer_name,
        )
        projected, _ = self.o_proj(attn_out)
        return projected

    def _qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """主注意力的 Q/K GemmaRMSNorm（按 head 维 reshape 后归一化再还原）。"""
        q_shape = q.shape
        k_shape = k.shape
        # reshape 到 [*, head_dim] 让 norm 独立作用于每个头。
        q = q.reshape(-1, self.head_dim).contiguous()
        k = k.reshape(-1, self.head_dim).contiguous()
        q = self.q_norm(q).reshape(q_shape)
        k = self.k_norm(k).reshape(k_shape)
        return q, k

    def _index_qk_norm(self, idx_q: torch.Tensor, idx_k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """索引器 index_q/index_k 的 GemmaRMSNorm（同上，按 idx_head_dim）。"""
        idx_q_shape = idx_q.shape
        idx_k_shape = idx_k.shape
        idx_q = idx_q.reshape(-1, self.idx_head_dim)
        idx_k = idx_k.reshape(-1, self.idx_head_dim)
        idx_q = self.index_q_norm(idx_q).reshape(idx_q_shape)
        idx_k = self.index_k_norm(idx_k).reshape(idx_k_shape)
        return idx_q, idx_k


def _sparse_attention_layer_ids(config: PretrainedConfig) -> set[int]:
    """从配置解析"哪些层用稀疏注意力"的层号集合。

    sparse_attention_freq 是与层数等长的 0/1 列表：非 0 即稀疏层。
    """
    cfg = getattr(config, "sparse_attention_config", None)
    if not cfg:
        return set()
    freq = cfg.get("sparse_attention_freq")
    if freq is None:
        return set()
    return {i for i, f in enumerate(freq) if f != 0}


def _get_text_config(vllm_config: VllmConfig) -> PretrainedConfig:
    # 取文本配置（多模态 checkpoint 也有统一的 hf_text_config 入口）。
    return vllm_config.model_config.hf_text_config


def _get_max_position_embeddings(config: PretrainedConfig) -> int:
    # 位置编码上限 = max(max_position_embeddings, max_model_len)：
    # 部分部署配置的 max_model_len 大于训练上限，RoPE 表需覆盖到服务长度。
    max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
    max_model_len = getattr(config, "max_model_len", None)
    if isinstance(max_model_len, int):
        max_position_embeddings = max(max_position_embeddings, max_model_len)
    return max_position_embeddings


def _get_rope_parameters(config: PretrainedConfig) -> dict[str, Any] | None:
    # RoPE 参数解析：优先 checkpoint 显式提供的 rope_parameters 字典，
    # 否则从扁平字段（rope_theta / partial_rotary_factor）组装。
    rope_parameters = getattr(config, "rope_parameters", None)
    if rope_parameters is not None:
        rope_parameters = dict(rope_parameters)
    else:
        rope_parameters = {
            "rope_theta": getattr(config, "rope_theta", 10000),
            "partial_rotary_factor": getattr(config, "partial_rotary_factor", 1.0),
        }
    return rope_parameters


def _is_w8a8_mxfp8_linear(layer: nn.Module) -> bool:
    """判断线性层是否使用昇腾 W8A8 MXFP8 动态量化方案。

    通过 quant_method 的类名探测（避免硬导入可能不存在的量化类）。
    """
    quant_method = getattr(layer, "quant_method", None)
    quant_scheme = getattr(quant_method, "quant_method", quant_method)
    return quant_scheme is not None and quant_scheme.__class__.__name__ == "AscendW8A8MXFP8DynamicLinearMethod"


class MiniMaxM3SwiGLUOAI(nn.Module):
    """MiniMax-M3 SwiGLU-OAI activation for packed gate/up outputs."""
    """（SwiGLU-OAI 激活 —— MiniMax 定制的带限幅 SwiGLU 变体。）

    公式：out = clamp(gate) · sigmoid(alpha·gate) · (clamp(up) + beta)
    其中 gate/up 是 gate_up_proj 输出的两半（非交错拼接）。
    alpha/beta/limit 是可训练超参（OAI 风格 SwiGLU 的推广）；
    limit 限幅防止激活值爆炸（训练/量化稳定性）。
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        limit: float,
        use_mx_quant: bool = False,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.limit = float(limit)
        # MXFP8 路径：激活后立即做动态量化（供下游 W8A8 down_proj 消费，
        # 省一次单独的量化 kernel 与中间显存）。
        self.use_mx_quant = use_mx_quant

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """x: [N, 2*intermediate]（gate|up 拼接）。返回激活值（或量化对）。"""
        if self.use_mx_quant or get_ascend_device_type() == AscendDeviceType.A5:
            # 手写路径（A5 或需在线量化）：逐算子组合，FP32 语义精确可控。
            d = x.shape[-1] // 2
            gate = torch.clamp(x[..., :d], max=self.limit)
            up = torch.clamp(x[..., d:], min=-self.limit, max=self.limit)
            activated = gate * torch.sigmoid(self.alpha * gate) * (up + self.beta)
            if self.use_mx_quant:
                # MXFP8 动态量化：输出 (量化值, 缩放因子) 二元组。
                quantized_x, scale = DeviceOperator.npu_dynamic_quant(
                    activated,
                    act_quant_type=torch.float8_e4m3fn,
                    use_mxfp_quant=True,
                )
                assert scale is not None
                return quantized_x, scale
            return activated
        # 融合路径：CANN 的 npu_clipped_swiglu 算子一次完成（非 A5 硬件）。
        return torch.ops.npu.npu_clipped_swiglu(
            x,
            dim=-1,
            alpha=self.alpha,
            limit=self.limit,
            bias=self.beta,
            interleaved=False,
        )


class MiniMaxM3MLP(nn.Module):
    """dense MLP：gate_up 融合投影 → SwiGLU-OAI → down 投影。"""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        intermediate_size: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        hidden_act = config.hidden_act
        if intermediate_size is None:
            intermediate_size = config.intermediate_size

        # MergedColumnParallelLinear：gate/up 两个同形投影合并成一个 GEMM
        #（输出 [2*intermediate]，forward 时再切开），减少 kernel 启动。
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act == "swigluoai":
            # A5 + W8A8 MXFP8 量化组合时启用激活内联量化。
            use_mx_quant = (
                get_ascend_device_type() == AscendDeviceType.A5
                and _is_w8a8_mxfp8_linear(self.gate_up_proj)
                and _is_w8a8_mxfp8_linear(self.down_proj)
            )
            self.act_fn = MiniMaxM3SwiGLUOAI(
                alpha=config.swiglu_alpha,
                beta=getattr(config, "swiglu_beta", 1.0),
                limit=config.swiglu_limit,
                use_mx_quant=use_mx_quant,
            )
        else:
            raise ValueError(f"Unsupported activation: {hidden_act}. Only swigluoai is supported.")

    def forward(
        self,
        x,
    ):
        """x: [N, hidden] → [N, hidden]。"""
        gate_up, _ = self.gate_up_proj(x)
        # act_fn 可能返回 (激活值, 量化 scale) 二元组（MXFP8 路径），
        # 量化感知的 down_proj 会自行解包。
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MiniMaxM3MoE(nn.Module):
    """MiniMax-M3 MoE 层（EPLB 感知的专家并行组装）。"""

    def __init__(
        self,
        config: PretrainedConfig,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.n_shared_experts = getattr(config, "n_shared_experts", 0) or 0

        # ---- 专家并行（EP）元数据 ----
        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_local_experts

        # ---- EPLB（专家负载均衡）元数据 ----
        # 逻辑专家 = checkpoint 中的专家；物理专家 = 逻辑 + 冗余副本
        #（EPLB 热更新时冗余副本承载被复制的热门专家）。
        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        # 本 rank 负责的物理专家区间（物理专家在 EP 组内均分）。
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size
        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = self.physical_expert_start + self.n_local_physical_experts

        if self.tp_size > config.num_local_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than the number of experts {config.num_local_experts}."
            )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        # 可选路由偏置（无辅助损失路由）：加载器用静态方法强制 FP32。
        self.use_routing_bias = getattr(config, "use_routing_bias", False)
        if self.use_routing_bias:
            self.e_score_correction_bias = nn.Parameter(torch.empty(config.num_local_experts, dtype=torch.float32))
            self.e_score_correction_bias.weight_loader = MiniMaxM3MoE.ebias_weight_loader
        else:
            self.e_score_correction_bias = None

        self.shared_experts: MiniMaxM3MLP | None
        if self.n_shared_experts:
            # 共享专家：中间层加宽的单 MLP（× n_shared_experts），恒激活。
            # reduce_results=False：输出加法融合进 FusedMoE 内核。
            intermediate_size = config.intermediate_size * self.n_shared_experts
            self.shared_experts = MiniMaxM3MLP(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_experts",
                reduce_results=False,
                intermediate_size=intermediate_size,
            )
        else:
            self.shared_experts = None

        # 路由器：FP32 参数 + FP32 输出（路由打分对精度敏感）。
        self.gate = GateLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            params_dtype=torch.float32,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )

        # 融合 MoE 内核：注意 activation="swigluoai_uninterleave"
        #（融合内核内的 SwiGLU-OAI 变体）与 routed scale 应用方式。
        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            num_experts=config.num_local_experts,
            gate=self.gate,
            top_k=config.num_experts_per_tok,
            scoring_func=config.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            renormalize=True,
            activation="swigluoai_uninterleave",
            swiglu_limit=config.swiglu_limit,
            swiglu_alpha=config.swiglu_alpha,
            swiglu_beta=getattr(config, "swiglu_beta", 1.0),
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            router_logits_dtype=self.gate.out_dtype,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scale_to_output=True,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
        )

        # Ascend dispatch uses this metadata to size the global physical
        # expert space. The upstream V2 EPLB factory only updates moe_config.
        # （昇腾分发内核用该元数据确定全局物理专家空间大小；
        #   上游 V2 EPLB 工厂只更新 moe_config，这里两处都同步。）
        self.experts.global_redundant_expert_num = self.n_redundant_experts
        self.experts.moe_config.global_redundant_expert_num = self.n_redundant_experts

    @staticmethod
    def ebias_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        """路由偏置的专用加载器：强制转 FP32（checkpoint 可能是 bf16）。"""
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight.to(torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """MoE 前向。[N, hidden] → [N, hidden]。"""
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.experts.is_internal_router:
            # 内部路由模式：hidden 直接塞进 router_logits 参数位，
            # 融合内核在内部自行调用 gate（少一次 host 端算子调用）。
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=hidden_states,
            )
        else:
            # router_logits: (num_tokens, n_experts)
            router_logits, _ = self.gate(hidden_states)
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )

        return final_hidden_states.view(num_tokens, hidden_dim)


class MiniMaxM3Attention(nn.Module):
    """普通 GQA 全注意力层（非稀疏层使用）。

    结构：QKV 融合投影 → Q/K GemmaRMSNorm → RoPE → vLLM Attention
    （平台后端，NPU 上走 ASCEND flash attention）→ o_proj。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rotary_dim: int,
        rope_parameters: dict[str, Any] | None = None,
        attn_window_size: int | None = None,
        max_position_embeddings: int = 8192,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        # TP 切分（与稀疏注意力层同构：Q 头均分、KV 头均分或复制）。
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            # （KV 头数 ≥ TP：KV 头跨卡切分。）
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            # （KV 头数 < TP：KV 头跨卡复制。）
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if rope_parameters is not None and "partial_rotary_factor" not in rope_parameters:
            rope_parameters["partial_rotary_factor"] = rotary_dim / self.head_dim
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
        )

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=rms_norm_eps)

        # vLLM 标准 Attention 层：按平台注册的后端执行
        #（attn_window_size 非空时为滑窗注意力）。
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            per_layer_sliding_window=attn_window_size,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def _qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Q/K 的 GemmaRMSNorm（按头 reshape 归一化后还原形状）。"""
        q_shape = q.shape
        k_shape = k.shape
        q = q.reshape(-1, self.head_dim).contiguous()
        k = k.reshape(-1, self.head_dim).contiguous()
        q = self.q_norm(q).reshape(q_shape)
        k = self.k_norm(k).reshape(k_shape)
        return q, k

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """全注意力前向。positions [N]、hidden_states [N, H] → [N, H]。"""
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        v = v.contiguous()

        q, k = self._qk_norm(q, k)
        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class MiniMaxM3DecoderLayer(nn.Module):
    """MiniMax-M3 解码层：按层配置选择注意力形态与 MLP 形态。

    - 注意力：稀疏层 → MiniMaxM3SparseAttention；普通层 → MiniMaxM3Attention；
    - MLP：MoE 层 → MiniMaxM3MoE；dense 层 → MiniMaxM3MLP；
    - 残差：标准 pre-norm（GemmaRMSNorm 支持融合的"残差相加+归一化"）。
    """

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        max_position_embeddings = _get_max_position_embeddings(config)
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        # （make_layers 创建层时传入带层号的 prefix，从末段解析层号。）
        layer_idx = int(prefix.split(sep=".")[-1])

        self.layer_idx = layer_idx

        sparse_attention_config = getattr(config, "sparse_attention_config", None)

        if sparse_attention_config is not None:
            # 该层是否稀疏注意力层 + 是否禁用 index_v 分支（逐层配置）。
            is_sparse_attention_layer = layer_idx in _sparse_attention_layer_ids(config)
            disable_index_value = sparse_attention_config["sparse_disable_index_value"][layer_idx] == 1
        else:
            is_sparse_attention_layer = False
            disable_index_value = False

        # 注意力公共参数（两种形态共享的构造 kwargs）。
        attn_kwargs = dict(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rotary_dim=config.rotary_dim,
            rope_parameters=_get_rope_parameters(config),
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        if is_sparse_attention_layer:
            self.self_attn = MiniMaxM3SparseAttention(
                **attn_kwargs,
                sparse_cfg=sparse_attention_config,
                disable_index_value=disable_index_value,
            )
        else:
            self.self_attn = MiniMaxM3Attention(**attn_kwargs)

        moe_layer_freq = getattr(config, "moe_layer_freq", None)
        # ``is_layer_sparse`` here means "this layer's MLP is a sparse MoE",
        # not anything about attention sparsity. The name is kept (instead of
        # the clearer ``is_layer_moe``) to match the convention used by the
        # rest of sglang -- ``OperationsStrategy``, ``LayerScatterModes``,
        # ``LayerCommunicator``, ``gpt_oss``, ``falcon_h1`` etc all access
        # ``layer.is_layer_sparse``.
        # （is_layer_sparse 指"MLP 是稀疏 MoE"而非注意力稀疏；命名沿用
        #   sglang 系生态的约定。）
        self.is_layer_sparse = moe_layer_freq[layer_idx] != 0 if moe_layer_freq is not None else True

        if self.is_layer_sparse:
            self.block_sparse_moe = MiniMaxM3MoE(
                config=config,
                quant_config=quant_config,
                parallel_config=parallel_config,
                prefix=f"{prefix}.block_sparse_moe",
            )
        else:
            self.mlp = MiniMaxM3MLP(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                intermediate_size=config.dense_intermediate_size,
            )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        """标准 pre-norm 解码层前向。返回 (子层输出, 更新后的残差)。"""
        # Self Attention
        if residual is None:
            # 首层：残差 = 输入本身。
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            # 融合式：一次完成残差相加 + 归一化。
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        if self.is_layer_sparse:
            hidden_states = self.block_sparse_moe(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


# @support_torch_compile：标记模型主体可被 torch.compile 分段编译。
@support_torch_compile
class MiniMaxM3Model(nn.Module, EagleModelMixin):
    """MiniMax-M3 文本模型主体（解码层堆叠 + EAGLE3 aux 隐状态捕获）。

    多继承 EagleModelMixin 获得 aux hidden states 收集能力：
    EAGLE3 投机解码时按 aux_hidden_state_layers 记录指定层的隐状态，
    供草稿模型消费（跨 PP 段时通过 pp_utils 的传输张量透传）。
    """
    # 权重加载阶段不回退到 PyTorch 原生加载（完全走 vLLM 加载器）。
    fall_back_to_pt_during_load = False
    # vLLM #50514 validates and relays the model's existing PP aux payload.
    # （vLLM #50514：校验并转发模型已有的 PP aux 负载。）
    supports_aux_hidden_states_over_pp = True
    # aux 隐状态在 IntermediateTensors 中的键前缀（PP 传输用）。
    AUX_HIDDEN_STATE_KEY = "pp_transport_aux_hidden_states_"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        # config = vllm_config.model_config.hf_config
        config = _get_text_config(vllm_config)
        text_config = config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        # EAGLE3 投机解码开启时才启用 aux 隐状态捕获。
        self._enable_eagle3_aux_hidden_states = (
            vllm_config.speculative_config is not None and vllm_config.speculative_config.method == "eagle3"
        )

        self.vocab_size = text_config.vocab_size
        self.num_hidden_layers = text_config.num_hidden_layers
        # PP 首卡建词嵌入，其余卡占位。
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                text_config.vocab_size,
                text_config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # 层堆叠（make_layers 按 PP rank 裁剪出本地层区间）。
        self.start_layer, self.end_layer, self.layers = make_layers(
            text_config.num_hidden_layers,
            lambda prefix: MiniMaxM3DecoderLayer(
                config,
                prefix,
                model_config=model_config,
                parallel_config=vllm_config.parallel_config,
                cache_config=cache_config,
                quant_config=quant_config,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        # 构造 PP 中间张量工厂（hidden_states + residual 两项基础负载，
        # aux 隐状态由 pp_utils 动态附加）。
        self.make_empty_intermediate_tensors = make_pp_empty_intermediate_tensors(
            self,
            make_empty_intermediate_tensors_factory(["hidden_states", "residual"], config.hidden_size),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        """文本主干前向。

        返回：最终隐状态；或（PP 中间卡）IntermediateTensors；
        或（EAGLE3 开启）(隐状态, aux 隐状态列表)。
        """
        pp_group = get_pp_group()
        if pp_group.is_first_rank:
            # 首卡：嵌入起步，记录第 0 层的 aux 隐状态（若配置）。
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
            aux_hidden_states: list[torch.Tensor] = []
            self._maybe_add_hidden_state(aux_hidden_states, 0, hidden_states, residual)
        else:
            # 中间卡：从上一段恢复 hidden/residual 与透传的 aux 隐状态。
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
            aux_hidden_states = get_pp_transport_tensors(
                intermediate_tensors,
                PPTransportDataType.AUX_HIDDEN_STATES,
            )

        # 逐层前向；islice 只迭代本地层区间（跳过非本卡的层）。
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual = layer(positions, hidden_states, residual)
            # 每层后尝试记录 aux 隐状态（未配置时是空操作）。
            self._maybe_add_hidden_state(aux_hidden_states, idx + 1, hidden_states, residual)

        if not pp_group.is_last_rank:
            # 中间卡：打包 hidden/residual + aux 传给下一段。
            intermediate_tensors = IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
            return add_pp_transport_tensors(
                intermediate_tensors,
                PPTransportDataType.AUX_HIDDEN_STATES,
                aux_hidden_states,
            )
        # 末卡：最终 norm（融合残差相加）。
        hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states

        return hidden_states

    def _set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        # 内部设置 aux 捕获层：EAGLE3 开启时用请求的层，否则清空。
        if self._enable_eagle3_aux_hidden_states:
            EagleModelMixin._set_aux_hidden_state_layers(self, layers)
        else:
            EagleModelMixin._set_aux_hidden_state_layers(self, ())

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        # 公开接口：由投机解码调度器调用，声明草稿需要的层。
        self._set_aux_hidden_state_layers(layers)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # 专家权重名映射（checkpoint 的 w1/w2/w3 → 融合 MoE 的 gate/down/up）。
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_local_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """手写权重加载器（跳过/重映射/统计一体的流式处理）。

        处理顺序（每条权重依次尝试）：
        1. 前缀清理与跳过：去 "model." 前缀；跳过 mtp.*（投机层单独加载）、
           rotary_emb.inv_freq（RoPE 运行时重建）、spec 层；
        2. scale 名重映射：weight_scale_inv → weight_scale（昇腾量化命名）；
        3. index_* 特殊权重（非 q/k 投影的，如 index_v/index_o）直接装填；
        4. 堆叠参数映射：q/k/v → qkv_proj、index_q/k → indexer_proj 或
           融合 qkv_proj、gate/up → gate_up_proj；
        5. 专家权重映射：experts.{i}.w1/w2/w3（含 FP8 scale）；
        6. 兜底：直接按名装填（含 KV scale 重映射）。
        全程统计 loaded/skipped 与原因，加载完输出 warning 日志便于排查。
        """
        # (目标参数名片段, checkpoint 名片段, 分片 id)：堆叠/融合参数装配表。
        stacked_params_mapping: list[tuple[str, str, int | str]] = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".indexer_proj", ".index_q_proj", "index_q"),
            (".indexer_proj", ".index_k_proj", "index_k"),
            # 融合 qkv_indexer 路径：index 投影并入 qkv_proj。
            (".qkv_proj", ".index_q_proj", "index_q"),
            (".qkv_proj", ".index_k_proj", "index_k"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        # （专家参数映射：含 FP8 权重 scale 与激活 scale 的四元组。）
        expert_params_mapping = self.get_expert_mapping()

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        # 统计计数器（闭包内用 nonlocal 修改外层变量）。
        loaded_tensors = 0
        skipped_tensors = 0
        skipped_reasons: dict[str, int] = {}

        def mark_loaded(param_name: str) -> None:
            nonlocal loaded_tensors
            loaded_params.add(param_name)
            loaded_tensors += 1

        def mark_skipped(reason: str) -> None:
            nonlocal skipped_tensors
            skipped_tensors += 1
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1

        for name, loaded_weight in weights:
            # ---- 阶段1：前缀清理与全局跳过规则 ----
            if name.startswith("model."):
                name = name[len("model.") :]
            if "mtp." in name:
                mark_skipped("mtp")
                continue
            if "weight_scale_inv" in name:
                name = name.replace("weight_scale_inv", "weight_scale")
            elif "scale_inv" in name:
                name = name.replace("scale_inv", "scale")
            if "rotary_emb.inv_freq" in name:
                mark_skipped("rotary_emb")
                continue

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                mark_skipped("spec_decode")
                continue  # skip spec decode layers for main model
                # （主模型跳过投机解码层，草稿模型单独加载。）

            # ---- 阶段2：index_* 特殊权重直接装填 ----
            # Sparse layers fold index_q/index_k into fused qkv_proj (handled below).
            # Other index_* weights (e.g. index_v/index_o) load explicitly here.
            # （稀疏层的 index_q/k 并入融合 qkv_proj（走下方映射）；
            #   其他 index_* 权重（如 index_v/index_o）在这里直接装填。）
            if ".index_" in name and ".index_q_proj" not in name and ".index_k_proj" not in name:
                if name.endswith(".bias") and name not in params_dict:
                    mark_skipped("missing_bias")
                    continue
                if is_pp_missing_parameter(name, self):
                    mark_skipped("pp_missing")
                    continue
                if name not in params_dict:
                    mark_skipped("missing_index_param")
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                mark_loaded(name)
                continue

            # ---- 阶段3：堆叠参数映射（for...else：全不匹配才进入 else）----
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Routed experts (w1/w2/w3) are handled below; don't let
                # stacked dense/shared-expert mappings rewrite them.
                # （路由专家的 w1/w2/w3 走下方专家映射，防止这里的
                #   dense/共享专家映射误改写专家权重名。）
                if ("block_sparse_moe.experts." in name) and name not in params_dict:
                    continue
                param_name_full = name.replace(weight_name, param_name)
                if param_name_full.endswith(".bias") and param_name_full not in params_dict:
                    mark_skipped("missing_bias")
                    continue
                if is_pp_missing_parameter(param_name_full, self):
                    mark_skipped("pp_missing")
                    continue
                # KV 量化 scale 的名字重映射（K/V cache 缩放因子）。
                if param_name_full.endswith((".k_scale", ".v_scale")):
                    remapped_name = maybe_remap_kv_scale_name(param_name_full, params_dict)
                    if remapped_name is not None and remapped_name in params_dict:
                        param = params_dict[remapped_name]
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight)
                        mark_loaded(remapped_name)
                        break
                if param_name_full not in params_dict:
                    continue

                param = params_dict[param_name_full]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                mark_loaded(param_name_full)
                break
            else:
                # ---- 阶段4：专家权重映射 ----
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name_mapped, self):
                        mark_skipped("pp_missing")
                        break
                    if name_mapped not in params_dict:
                        continue

                    param = params_dict[name_mapped]
                    weight_loader = param.weight_loader
                    # return_success=True：专家加载器回报是否本 rank 负责
                    #（EP 模式下非本地专家返回 False，继续尝试其他映射）。
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        mark_loaded(name_mapped)
                        break
                else:
                    # ---- 阶段5：兜底直接装填 ----
                    if is_expert_weight:
                        mark_skipped("nonlocal_or_unmapped_expert")
                        continue

                    if name.endswith(".bias") and name not in params_dict:
                        mark_skipped("missing_bias")
                        continue

                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        mark_skipped("kv_scale_remap_missing")
                        continue

                    if is_pp_missing_parameter(name, self):
                        mark_skipped("pp_missing")
                        continue
                    if name not in params_dict:
                        mark_skipped("missing_param")
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    # MoE 参数落到兜底路径属于异常：形状对不上时必须报错
                    #（提示开发者补专家映射），而不是静默错装。
                    if getattr(weight_loader, "supports_moe_loading", False):
                        if loaded_weight.shape == param.shape:
                            default_weight_loader(param, loaded_weight)
                            mark_loaded(name)
                            continue
                        raise ValueError(
                            f"FusedMoE parameter {name!r} reached the "
                            "fallback loader with an incompatible shape: "
                            f"checkpoint={tuple(loaded_weight.shape)}, "
                            f"parameter={tuple(param.shape)}. Add an expert "
                            "mapping for this checkpoint weight instead."
                        )
                    weight_loader(param, loaded_weight)
                    mark_loaded(name)
        # 加载统计日志（warning 级别：方便对照 checkpoint 内容排查缺漏）。
        logger.warning(
            "MiniMax M3 text load_weights loaded %d checkpoint tensors into "
            "%d parameter names; skipped %d tensors by reason: %s",
            loaded_tensors,
            len(loaded_params),
            skipped_tensors,
            skipped_reasons,
        )
        return loaded_params


class MiniMaxM3SparseForCausalLM(
    nn.Module,
    SupportsLoRA,
    SupportsPP,
    SupportsEagle3,
    MixtureOfExperts,
):
    """MiniMax-M3 顶层 CausalLM（注册名 MiniMaxM3SparseForCausalLM）。

    实现的接口：LoRA / 流水线并行 / EAGLE3 投机解码 / MoE 元数据
    （MixtureOfExperts 接口供 EPLB 与调度器查询专家拓扑）。
    """
    # 融合参数 ↔ checkpoint 源参数映射（供 vLLM 通用加载器使用）。
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "indexer_proj": ["index_q_proj", "index_k_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "experts": ["experts.0.w1", "experts.0.w2", "experts.0.w3"],
    }

    # 权重名前缀映射：多模态 checkpoint 的文本部分挂在 language_model.* 下。
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.model.": "model.",
            "language_model.lm_head.": "lm_head.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = _get_text_config(vllm_config)
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        # 服务长度写回配置（RoPE 表覆盖到 max_model_len）。
        if hasattr(vllm_config.model_config, "max_model_len"):
            self.config.max_model_len = vllm_config.model_config.max_model_len
        self.model = MiniMaxM3Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            self.logits_processor = LogitsProcessor(config.vocab_size)
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        # 收集 MoE 元数据（供 EPLB/调度器查询）。
        self._set_moe_parameters()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        # EAGLE3 接口：透传 aux 捕获层配置。
        self.model.set_aux_hidden_state_layers(layers)

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        # EAGLE3 默认 aux 层：浅/中/深三层（2, N/2, N-3），
        # 与 EAGLE3 论文"多层特征融合"的设计对应。
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        return hidden_states

    def _set_moe_parameters(self) -> None:
        """汇总全部 MoE 层的专家拓扑元数据（MixtureOfExperts 接口）。"""
        self.expert_weights: MutableSequence[Sequence[torch.Tensor]] = []
        self.num_expert_groups = 1
        self.moe_layers = []
        self.moe_mlp_layers: list[MiniMaxM3MoE] = []

        # 遍历层收集 MoE 实例（PP 下非本卡的层是占位符，跳过）。
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            assert isinstance(layer, MiniMaxM3DecoderLayer)
            if layer.is_layer_sparse:
                example_moe = layer.block_sparse_moe
                self.moe_mlp_layers.append(example_moe)
                self.moe_layers.append(example_moe.experts)

        self.num_moe_layers = len(self.moe_layers)
        if example_moe is None:
            # 无 MoE 层的纯 dense 配置：全部清零。
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_shared_experts = 0
            self.num_redundant_experts = 0
            return

        # 从任一 MoE 层读取专家计数（各层拓扑一致）。
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
        """EPLB 热更新接口：冗余专家数量变化时同步全部 MoE 层的拓扑。"""
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        # 逐层同步并重建专家映射表（逻辑 → 物理专家的路由表）。
        for moe in self.moe_mlp_layers:
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()
            moe.experts.global_redundant_expert_num = self.num_redundant_experts
            moe.experts.moe_config.global_redundant_expert_num = self.num_redundant_experts

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """顶层加载：过滤多模态权重后交给 AutoWeightsLoader。"""
        loader = AutoWeightsLoader(self)
        raw_tensors = 0
        text_tensors = 0
        skipped_multimodal_tensors = 0

        # 生成器过滤：视觉塔/投影器/patch 合并权重跳过（文本模型不消费）。
        def text_weights() -> Iterable[tuple[str, torch.Tensor]]:
            nonlocal raw_tensors, text_tensors, skipped_multimodal_tensors
            for name, weight in weights:
                raw_tensors += 1
                if "vision_tower" in name or "multi_modal_projector" in name or "patch_merge_mlp" in name:
                    skipped_multimodal_tensors += 1
                    continue
                text_tensors += 1
                yield name, weight

        loaded_params = loader.load_weights(text_weights(), mapper=self.hf_to_vllm_mapper)
        logger.warning(
            "MiniMax M3 top-level load_weights saw %d checkpoint tensors, "
            "passed %d text tensors, skipped %d multimodal tensors, "
            "returned %d loaded parameter names",
            raw_tensors,
            text_tensors,
            skipped_multimodal_tensors,
            len(loaded_params),
        )
        return loaded_params

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


def get_spec_layer_idx_from_weight_name(config: PretrainedConfig, weight_name: str) -> int | None:
    """判断权重是否属于投机解码（MTP）层，是则返回其层号。

    MTP 层编号紧随主模型之后（num_hidden_layers + i），
    主模型加载时据此跳过，草稿模型加载时据此认领。
    """
    if hasattr(config, "num_mtp_modules") and (config.num_mtp_modules > 0):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_mtp_modules):
            if weight_name.startswith(f"model.layers.{layer_idx + i}."):
                return layer_idx + i
    return None
