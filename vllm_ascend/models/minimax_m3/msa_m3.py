# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 sparse attention and indexer backends for Ascend."""
"""（MiniMax-M3 稀疏注意力与索引器的昇腾后端实现。）

【MSA 工作原理（MiniMax Sparse Attention）】
本文件实现 M3 的"闪电索引器 + 块稀疏注意力"两段式管线：

1. 索引器（AscendMiniMaxM3Indexer）：
   - 每个稀疏层维护一个独立的 index cache（只存每 token 一个低维
     index_k 向量，idx_head_dim 通常 128，远小于主 KV）；
   - forward(index_q) 时对全部历史 block 打分（index_q · index_k 的
     block 级聚合），选出 topk_blocks 个 block id；
   - 初始 block（init_blocks）与局部滑窗 block（local_blocks）保底入选。

2. 块稀疏注意力（AscendMiniMaxM3SparseImpl）：
   - 主 GQA 注意力只在 [初始块 + 局部块 + top-k 块] 上计算；
   - prefill 用变长批量核（cu_seqlens），decode 用批量解码核。

【vLLM v1 注意力后端协议】
- *Backend：静态能力申报（支持 dtype/头维/块大小/cache 布局）；
- *MetadataBuilder：每步从 CommonAttentionMetadata 构建本后端专属元数据
 （prefill/decode 分拆、上下文长度、block table 切片）；
- *Impl：真正的注意力计算（消费元数据 + topk 索引）。

【NPU 多路径】
- AscendC（CANN 原生算子）路径：prefill 打分恒定启用；
  decode 打分在非 A5 设备启用；
- A5 Triton 路径：decode 走更低延迟的 Triton-Ascend kernel；
- TP 分块并行：非 A5 设备上 decode 打分把 block 维切给各 rank
  并行打分后 all_gather 归并（_decode_topk_tp_sharded）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from torch import nn
from torch.nn.parameter import Parameter
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.distributed import divide, get_tensor_model_parallel_world_size, get_tp_group
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    adjust_block_scale_shard,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.parameter import BasevLLMParameter, BlockQuantScaleParameter
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImplBase,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
)

from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec
# NPU 原生算子（AscendC 封装）：打分/解码/稀疏注意力内核。
from vllm_ascend.models.minimax_m3.ops.msa_m3_npu import (
    MiniMaxM3TPDecodeScoreMetadata,
    minimax_m3_index_tp_block_parallel_decode,
    minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode,
)
from vllm_ascend.models.minimax_m3.ops.msa_m3_npu import (
    minimax_m3_index_decode as minimax_m3_index_decode_ascendc,
)
from vllm_ascend.models.minimax_m3.ops.msa_m3_npu import (
    minimax_m3_index_prefill as minimax_m3_index_prefill_ascendc,
)
from vllm_ascend.ops.linear import AscendColumnParallelLinear
from vllm_ascend.ops.linear_op import get_parallel_op
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

# The bundled MsaIndexScore includes the Ascend 950 arch35 FP8 kernel. Keep it
# enabled for A5 prefill, while A5 decode uses its lower-latency Triton path.
# （AscendC 打分内核含 950 arch35 的 FP8 实现：A5 的 prefill 用它，
#   A5 的 decode 用更低延迟的 Triton 路径。）
_USE_ASCENDC_INDEX_SCORE_PREFILL = True
_USE_ASCENDC_INDEX_SCORE_DECODE = get_ascend_device_type() != AscendDeviceType.A5

if get_ascend_device_type() == AscendDeviceType.A5:
    # A5 专属 Triton kernel：decode 打分 / 通用打分 / topk 选择。
    from vllm_ascend.models.minimax_m3.ops.msa_m3_triton_a5 import (
        minimax_m3_index_decode,
        minimax_m3_index_score,
        minimax_m3_index_topk,
    )


def _should_use_tp_sharded_index_decode(tp_size: int, num_prefills: int) -> bool:
    # The A5 Triton decode kernel operates on the complete, replicated index-K
    # cache on every TP rank. Keep the mainline block-sharded optimization for
    # the other device families only.
    # （A5 的 Triton decode 核在每个 TP rank 上操作完整复制的 index-K cache；
    #   块分片优化只保留给其他设备族。）
    # 混合批次（含 prefill）时不用分片路径，保持与 prefill 路径的一致性。
    return get_ascend_device_type() != AscendDeviceType.A5 and tp_size > 1 and num_prefills == 0


def _active_decode_num_reqs(
    num_decodes: int,
    num_decode_tokens: int,
    decode_query_len: int,
) -> int:
    """Return the number of real decode requests, ignoring FIA/graph padding."""
    # （返回真实 decode 请求数，忽略 FIA/图捕获的填充段。）
    # 图捕获/FIA（fully-inference-attention）会补齐 batch 到固定大小，
    # 真实请求数 = 总 decode token 数 ÷ 每请求 query 数。
    if decode_query_len <= 0:
        return 0
    return min(num_decodes, num_decode_tokens // decode_query_len)


def _active_prefill_num_reqs(
    num_prefills: int,
    num_prefill_tokens: int,
    query_start_loc_cpu: torch.Tensor,
    num_decodes: int,
) -> int:
    """Return real prefill requests, ignoring FIA/SP tail padding segments."""
    # （返回真实 prefill 请求数，忽略 FIA/SP 尾部填充段。）
    # 从 CPU 侧的 query_start_loc 逐请求累加，直到耗尽 prefill token 预算；
    # 一个有效请求都没有时至少保底 1（避免 0 引发的退化行为）。
    if num_prefills <= 0 or num_prefill_tokens <= 0:
        return 0
    qsl_cpu = query_start_loc_cpu.detach().cpu()
    num_reqs_fia = int(qsl_cpu.shape[0] - 1)
    active = 0
    tokens_accounted = 0
    # 只扫描 prefill 区间（decode 之后到 num_prefills 个请求）。
    for i in range(num_decodes, min(num_reqs_fia, num_decodes + num_prefills)):
        query_len = int(qsl_cpu[i + 1] - qsl_cpu[i])
        if query_len <= 0:
            continue
        if tokens_accounted + query_len > num_prefill_tokens:
            break
        tokens_accounted += query_len
        active += 1
    if active > 0:
        return active
    return min(1, num_prefills)


class AscendMiniMaxM3IndexerBackend(AttentionBackend):
    """索引器后端：申报 index cache 的能力与布局（vLLM v1 后端协议）。

    注意 index cache 是"单头"的（num_kv_heads 维被压掉）：
    get_kv_cache_shape 返回 (num_blocks, block_size, head_size)，
    与主 KV cache 的 5D 布局不同。
    """
    # ClassVar：类级常量注解（不属于实例属性）。
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16, torch.float16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_name() -> str:
        return "ASCEND_MINIMAX_M3_SPARSE_INDEXER"

    @staticmethod
    def get_impl_cls() -> type[AscendMiniMaxM3IndexerImpl]:
        return AscendMiniMaxM3IndexerImpl

    @staticmethod
    def get_builder_cls() -> type[AscendMiniMaxM3IndexerMetadataBuilder]:
        return AscendMiniMaxM3IndexerMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [128]

    @classmethod
    def is_sparse(cls) -> bool:
        # 声明为稀疏后端：调度器据此关闭部分与前缀共享不兼容的优化。
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # del 显式忽略未用参数（签名由协议规定）。
        del num_kv_heads, cache_dtype_str
        # index cache 布局：[块数, 块大小, 头维]（单头，无 KV 头维）。
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # 步长顺序：按维度自然顺序（C 连续）。
        if include_num_layers_dimension:
            raise NotImplementedError
        return (0, 1, 2)


class AscendMiniMaxM3IndexerCache(nn.Module, AttentionLayerBase):
    """索引器缓存伪层：只持有 index cache 张量并向框架申报布局。

    以 nn.Module + AttentionLayerBase 的形式存在，使 vLLM 的 KV cache
    规划器把它当作一个"注意力层"统一分配显存；
    forward 是空操作（缓存写入由 MiniMaxM3SparseAttention._insert_kv 完成）。
    """

    def __init__(
        self,
        head_dim: int,
        prefix: str,
        cache_config: CacheConfig | None = None,
        kv_cache_dtype: str = "auto",
        kv_cache_torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        # 占位空张量：运行时由 KV cache 分配器替换为真实缓存。
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.dtype = kv_cache_torch_dtype
        self.kv_cache_dtype = kv_cache_dtype
        self.prefix = prefix
        self.cache_config = cache_config
        # 注册进编译期前向上下文（唯一性校验防重复注册）。
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # 申报 SFA 索引缓存规格（num_kv_heads=1 的特殊 spec）。
        return AscendSFAIndexerCacheSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            cache_dtype_str=self.kv_cache_dtype,
        )

    # 空实现（... 是函数体的占位写法；缓存写入由 _insert_kv 完成）。
    def forward(self) -> None: ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return AscendMiniMaxM3IndexerBackend


@dataclass
class AscendMiniMaxM3IndexerPrefillMetadata:
    """索引器的 prefill 元数据。

    cu_seqlens_q：各 prefill 请求 query 的累计偏移（变长批处理）；
    context_lens：各请求的 KV 上下文长度（不含当前 query）；
    start_loc：上下文起始块号（AscendC 打分核用，块对齐）。
    """
    cu_seqlens_q: torch.Tensor
    seq_lens: torch.Tensor
    context_lens: torch.Tensor
    block_table: torch.Tensor
    max_query_len: int
    max_seq_len: int
    start_loc: torch.Tensor | None = None


@dataclass
class AscendMiniMaxM3IndexerDecodeMetadata:
    """索引器的 decode 元数据。decode_query_len：每请求 query 数
    （投机解码时 >1）；tp_score：TP 分块并行打分所需的图稳定元数据。
    """
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    max_seq_len: int
    decode_query_len: int
    cu_seqlens_q: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    start_loc: torch.Tensor | None = None
    tp_score: MiniMaxM3TPDecodeScoreMetadata | None = None


@dataclass
class AscendMiniMaxM3IndexerMetadata(AttentionMetadata):
    """索引器的整步元数据（slot_mapping 供 index cache 写入用）。"""
    seq_lens: torch.Tensor
    max_seq_len: int
    slot_mapping: torch.Tensor
    num_actual_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    causal_mask: torch.Tensor | None = None
    prefill: AscendMiniMaxM3IndexerPrefillMetadata | None = None
    decode: AscendMiniMaxM3IndexerDecodeMetadata | None = None


class AscendMiniMaxM3IndexerMetadataBuilder(AttentionMetadataBuilder[AscendMiniMaxM3IndexerMetadata]):
    """索引器元数据构建器：每步构建 prefill/decode 两套元数据。

    _cudagraph_support = UNIFORM_BATCH：支持图捕获（要求批内形状一致）；
    context_len_buffer：预分配的上下文长度缓冲（避免每步重建，
    图捕获时地址稳定）。
    """
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    # decode 阈值 1：query 数 ≤1 视为 decode（投机验证的 spec token 也算 decode）。
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # supports_spec_as_decode=True：投机解码的多 query 验证也按 decode 处理。
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        self.context_len_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=device,
        )
        self.block_size = kv_cache_spec.block_size
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        # 掩码构建器：AscendC prefill 打分核需要 splitfuse 因果掩码。
        self.attn_mask_builder = AttentionMaskBuilder(device) if _USE_ASCENDC_INDEX_SCORE_PREFILL else None

    def _build_tp_score_metadata(
        self,
        block_table: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        context_lens: torch.Tensor,
        *,
        max_seq_len: int,
        decode_query_len: int,
    ) -> MiniMaxM3TPDecodeScoreMetadata:
        """Package graph-stable inputs; derive TP tensors in model forward."""
        # （打包图稳定输入；TP 张量留到模型前向再推导。）
        # 计算本 rank 负责的 block 区间：全部 block 均分给 TP 各 rank。
        tp_rank = get_tp_group().rank_in_group
        max_block_count = (max_seq_len + self.block_size - 1) // self.block_size
        blocks_per_tp = (max_block_count + self.tp_size - 1) // self.tp_size
        block_offset = tp_rank * blocks_per_tp
        # 边界裁剪：最后一个 rank 可能分不到满额的块。
        block_count = max(0, min(blocks_per_tp, max_block_count - block_offset))
        return MiniMaxM3TPDecodeScoreMetadata(
            block_table=block_table,
            cu_seqlens_q=cu_seqlens_q,
            context_lens=context_lens,
            max_block_count=max_block_count,
            block_size=self.block_size,
            block_offset=block_offset,
            block_count=block_count,
            decode_query_len=decode_query_len,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendMiniMaxM3IndexerMetadata:
        """从公共注意力元数据构建索引器专属元数据。

        步骤：decode/prefill 分拆 → 统计真实请求数（剔除填充）→
        分别构建 prefill/decode 子元数据（切出各自的 seq_lens/block_table
        等切片）→ 汇总。cu_seqlens_q 减去 num_decode_tokens 是为了把
        prefill 段的 query 偏移归零（变长核从 0 起算）。
        """
        num_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table = common_attn_metadata.block_table_tensor
        qsl_cpu = common_attn_metadata.query_start_loc_cpu

        # split_decodes_and_prefills：按 query 长度把请求分成 decode 与 prefill
        # 两组（require_uniform：decode 组要求 query 长度一致，图捕获友好）。
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.reorder_batch_threshold,
            require_uniform=True,
        )

        prefill_metadata: AscendMiniMaxM3IndexerPrefillMetadata | None = None
        active_prefills = 0
        if num_prefills > 0:
            # 剔除 FIA/SP 填充后的真实 prefill 请求数。
            active_prefills = _active_prefill_num_reqs(num_prefills, num_prefill_tokens, qsl_cpu, num_decodes)
            prefill_end = num_decodes + active_prefills
            # prefill 段各请求的 query 长度（CPU 侧差分）。
            prefill_query_lens_cpu = qsl_cpu[num_decodes + 1 : prefill_end + 1] - qsl_cpu[num_decodes:prefill_end]
            # 上下文长度 = 序列长 - query 长（写入预分配缓冲，non_blocking 异步拷贝）。
            prefill_context_lens = self.context_len_buffer[num_decodes:prefill_end]
            prefill_context_lens.copy_(
                (seq_lens[num_decodes:prefill_end].detach().cpu() - prefill_query_lens_cpu).to(
                    device=self.context_len_buffer.device,
                    dtype=torch.int32,
                    non_blocking=True,
                ),
                non_blocking=True,
            )
            # query 累计偏移归零化（减去 decode 段 token 数）。
            cu_seqlens_q = (query_start_loc[num_decodes : prefill_end + 1] - num_decode_tokens).to(torch.int32)
            prefill_metadata = AscendMiniMaxM3IndexerPrefillMetadata(
                cu_seqlens_q=cu_seqlens_q,
                seq_lens=seq_lens[num_decodes:prefill_end],
                context_lens=prefill_context_lens,
                block_table=block_table[num_decodes:prefill_end],
                max_query_len=common_attn_metadata.max_query_len,
                max_seq_len=common_attn_metadata.max_seq_len,
                # start_loc：上下文起始块号（AscendC 打分核的块对齐入口）。
                start_loc=(
                    torch.div(
                        prefill_context_lens,
                        self.block_size,
                        rounding_mode="floor",
                    ).to(dtype=torch.int32)
                    if _USE_ASCENDC_INDEX_SCORE_PREFILL
                    else None
                ),
            )

        decode_metadata: AscendMiniMaxM3IndexerDecodeMetadata | None = None
        active_decodes = 0
        if num_decodes > 0:
            qsl_cpu = common_attn_metadata.query_start_loc_cpu
            # decode 组 query 长度一致（uniform），取第一个即可。
            query_lens_cpu = qsl_cpu[1 : num_decodes + 1] - qsl_cpu[:num_decodes]
            decode_query_len = int(query_lens_cpu[0].item())
            active_decodes = _active_decode_num_reqs(num_decodes, num_decode_tokens, decode_query_len)
            decode_context_lens = None
            decode_cu_seqlens_q = None
            if _USE_ASCENDC_INDEX_SCORE_DECODE:
                # AscendC 路径需要额外的上下文长度与 query 偏移。
                decode_context_lens = self.context_len_buffer[:active_decodes]
                decode_context_lens.copy_(
                    seq_lens[:active_decodes] - decode_query_len,
                    non_blocking=True,
                )
                decode_cu_seqlens_q = query_start_loc[: active_decodes + 1].to(torch.int32)
            decode_metadata = AscendMiniMaxM3IndexerDecodeMetadata(
                seq_lens=seq_lens[:active_decodes],
                block_table=block_table[:active_decodes],
                max_seq_len=common_attn_metadata.max_seq_len,
                decode_query_len=decode_query_len,
                cu_seqlens_q=decode_cu_seqlens_q,
                context_lens=decode_context_lens,
            )
            # 纯 decode 步 + TP>1 时挂上 TP 分块打分元数据。
            if _USE_ASCENDC_INDEX_SCORE_DECODE and self.tp_size > 1 and active_prefills == 0:
                decode_metadata.tp_score = self._build_tp_score_metadata(
                    decode_metadata.block_table,
                    decode_cu_seqlens_q,
                    decode_context_lens,
                    max_seq_len=decode_metadata.max_seq_len,
                    decode_query_len=decode_query_len,
                )

        return AscendMiniMaxM3IndexerMetadata(
            seq_lens=seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_actual_tokens=num_tokens,
            num_decodes=active_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=active_prefills,
            num_prefill_tokens=num_prefill_tokens,
            # splitfuse 掩码：混合 batch（decode+prefill）的因果掩码。
            causal_mask=(
                self.attn_mask_builder.get_splitfuse_attn_mask() if self.attn_mask_builder is not None else None
            ),
            prefill=prefill_metadata,
            decode=decode_metadata,
        )


class AscendMiniMaxM3IndexerImpl(nn.Module):
    """索引器实现：持 index cache 并执行"打分 + top-k 选块"。

    forward(index_query) 返回三元组 (decode_topk, prefill_topk,
    decode_select_num_idx)：decode/prefill 各自的 top-k 块索引，
    及 decode 的选择数（TP 分片路径的辅助量）。
    """

    def __init__(
        self,
        *,
        num_kv_heads: int,
        scale: float,
        topk_blocks: int,
        sparse_block_size: int,
        num_index_heads: int,
        index_head_dim: int,
        prefix: str,
        init_blocks: int = 0,
        local_blocks: int = 0,
        cache_config: CacheConfig | None = None,
        kv_cache_dtype: str = "auto",
        kv_cache_torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.scale = scale
        self.topk_blocks = topk_blocks
        self.block_size = sparse_block_size
        # 初始/局部保底块数（每个请求前 init_blocks 个块与最近 local_blocks
        # 个块无条件入选，保证基础局部性）。
        self.init_blocks = init_blocks
        self.local_blocks = local_blocks
        self.num_index_heads = num_index_heads
        self.index_head_dim = index_head_dim
        self.index_cache = AscendMiniMaxM3IndexerCache(
            head_dim=index_head_dim,
            prefix=f"{prefix}.index_cache",
            cache_config=cache_config,
            kv_cache_dtype=kv_cache_dtype,
            kv_cache_torch_dtype=kv_cache_torch_dtype,
        )

    def _decode_topk_tp_sharded(
        self,
        idx_q: torch.Tensor,
        index_kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        decode_query_len: int,
        tp_group: Any,
        tp_size: int,
        tp_rank: int,
    ) -> torch.Tensor:
        """TP 分块并行的 decode 打分：block 维切给各 rank，结果归并。

        原理：每个 rank 只对属于自己的 block 区间打分（top-k 局部），
        分数与索引 all_gather 汇总后做全局 top-k ——
        通信量只有 top-k 结果（小），而非全部 block 分数（大）。
        """
        # 步骤1：all_gather 拼齐所有 rank 的 index_q（头维），
        # 每个分片内核需要完整头集才能给本 rank 的块打全头分数。
        full_idx_q = tp_group.all_gather(idx_q.contiguous(), dim=1)

        # 步骤2：计算本 rank 的 block 区间（与 builder 的切分一致）。
        max_block_count = (max_seq_len + self.block_size - 1) // self.block_size
        blocks_per_tp = (max_block_count + tp_size - 1) // tp_size
        block_offset = tp_rank * blocks_per_tp
        block_count = max(0, min(blocks_per_tp, max_block_count - block_offset))
        # 本地区间的 block table 切片。
        local_block_table = block_table[:, block_offset : block_offset + block_count].contiguous()
        # 本地区间内的"有效序列长度"：把全局序列长度平移到本地区间坐标系
        # 并 clamp 到区间宽度（区间外的部分不归本 rank 管）。
        local_seq_lens = torch.clamp(
            seq_lens - block_offset * self.block_size,
            min=0,
            max=block_count * self.block_size,
        )

        # 步骤3：本地区间打分 + 局部 top-k（返回分数用于全局归并）。
        local_topk, local_scores = minimax_m3_index_decode(
            full_idx_q,
            index_kv_cache,
            local_block_table,
            local_seq_lens,
            max_seq_len,
            self.topk_blocks,
            self.init_blocks,
            self.local_blocks,
            full_idx_q.shape[1],
            decode_query_len,
            sm_scale=self.scale,
            block_offset=block_offset,
            block_count=block_count,
            global_seq_lens=seq_lens,
            return_scores=True,
        )
        # 步骤4：all_gather 各 rank 的局部分数/索引（沿 block 维拼接）。
        gathered_scores = tp_group.all_gather(local_scores.contiguous(), dim=-1)
        gathered_topk = tp_group.all_gather(local_topk.contiguous(), dim=-1)

        # 步骤5：每个头只需要全局 top-k：从拼接结果里切出本 rank 负责的头段，
        # 对全部候选块分数做 topk，再按位置 gather 出对应的块 id。
        local_head_count = idx_q.shape[1]
        local_head_start = tp_rank * local_head_count
        local_gathered_scores = gathered_scores.narrow(0, local_head_start, local_head_count)
        _, merged_pos = torch.topk(
            local_gathered_scores,
            k=self.topk_blocks,
            dim=-1,
        )
        local_gathered_topk = gathered_topk.narrow(0, local_head_start, local_head_count)
        return torch.gather(local_gathered_topk, dim=-1, index=merged_pos)

    def forward(
        self,
        index_query: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """索引器前向：按元数据分发到 decode/prefill 的各条计算路径。

        路径选择矩阵：
        - decode：AscendC（含 TP 分块并行/单机）或 Triton（含 TP 分片/单机）；
        - prefill：AscendC 融合打分+选块，或 Triton 分离式（打分 → topk）。
        """
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None, None, None
        index_md = attn_metadata[self.index_cache.prefix]
        assert isinstance(index_md, AscendMiniMaxM3IndexerMetadata)
        num_tokens = index_md.num_actual_tokens
        num_decode_tokens = index_md.num_decode_tokens
        # index_q: [num_tokens, num_index_heads, idx_head_dim]。
        iq = index_query[:num_tokens].view(-1, self.num_index_heads, self.index_head_dim)
        kv = self.index_cache.kv_cache

        decode_topk: torch.Tensor | None = None
        prefill_topk: torch.Tensor | None = None
        decode_select_num_idx: torch.Tensor | None = None
        if index_md.num_decodes > 0:
            d = index_md.decode
            assert d is not None
            tp_group = get_tp_group()
            # decode 段的 index_q（批内前 num_decode_tokens 个 token）。
            decode_iq = iq[:num_decode_tokens]
            if _USE_ASCENDC_INDEX_SCORE_DECODE:
                if tp_group.world_size > 1 and index_md.num_prefills == 0:
                    # AscendC + 纯 decode + TP>1：专用 TP 分块并行核。
                    decode_topk = minimax_m3_index_tp_block_parallel_decode(
                        decode_iq,
                        kv,
                        d.tp_score,
                        index_md.causal_mask,
                        topk=self.topk_blocks,
                        init_blocks=self.init_blocks,
                        local_blocks=self.local_blocks,
                        tp_group=tp_group,
                    )
                else:
                    # AscendC 单机（或混合批次）路径：先算 decode 起始块号。
                    decode_start_loc = torch.div(
                        d.context_lens,
                        self.block_size,
                        rounding_mode="floor",
                    ).to(dtype=torch.int32)
                    decode_topk = minimax_m3_index_decode_ascendc(
                        decode_iq,
                        kv,
                        d.block_table,
                        d.cu_seqlens_q,
                        d.seq_lens,
                        d.context_lens,
                        decode_start_loc,
                        index_md.causal_mask,
                        topk=self.topk_blocks,
                        init_blocks=self.init_blocks,
                        local_blocks=self.local_blocks,
                        decode_query_len=d.decode_query_len,
                    )
            else:
                # Triton 路径（A5）。
                tp_size = tp_group.world_size
                tp_rank = tp_group.rank_in_group
                if _should_use_tp_sharded_index_decode(tp_size, index_md.num_prefills):
                    # 非 A5 的 Triton TP 分片路径（block 维并行）。
                    decode_topk = self._decode_topk_tp_sharded(
                        decode_iq,
                        kv,
                        d.block_table,
                        d.seq_lens,
                        d.max_seq_len,
                        d.decode_query_len,
                        tp_group,
                        tp_size,
                        tp_rank,
                    )
                else:
                    # Triton 单机路径（额外返回 select_num_idx）。
                    decode_topk, decode_select_num_idx = minimax_m3_index_decode(
                        decode_iq,
                        kv,
                        d.block_table,
                        d.seq_lens,
                        d.max_seq_len,
                        self.topk_blocks,
                        self.init_blocks,
                        self.local_blocks,
                        self.num_kv_heads,
                        d.decode_query_len,
                        sm_scale=self.scale,
                    )
        if index_md.num_prefills > 0:
            p = index_md.prefill
            assert p is not None
            if _USE_ASCENDC_INDEX_SCORE_PREFILL:
                # AscendC 融合 prefill：打分 + top-k 一次完成。
                prefill_topk = minimax_m3_index_prefill_ascendc(
                    iq[num_decode_tokens:],
                    kv,
                    p.block_table,
                    p.cu_seqlens_q,
                    p.seq_lens,
                    p.context_lens,
                    p.start_loc,
                    index_md.causal_mask,
                    max_query_len=p.max_query_len,
                    max_seq_len=p.max_seq_len,
                    topk=self.topk_blocks,
                    init_blocks=self.init_blocks,
                    local_blocks=self.local_blocks,
                )
            else:
                # Triton 分离式 prefill：先全量打分，再 top-k 选块。
                score = minimax_m3_index_score(
                    iq[num_decode_tokens:],
                    kv,
                    p.block_table,
                    p.cu_seqlens_q,
                    p.seq_lens,
                    p.context_lens,
                    p.max_query_len,
                    p.max_seq_len,
                    self.num_kv_heads,
                    self.scale,
                )
                prefill_topk = minimax_m3_index_topk(
                    score,
                    p.cu_seqlens_q,
                    p.context_lens,
                    p.max_query_len,
                    self.topk_blocks,
                    self.init_blocks,
                    self.local_blocks,
                )
        return decode_topk, prefill_topk, decode_select_num_idx

    @staticmethod
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config=None,
        draft_attn_metadatas=None,
    ):
        """No-op: indexer replay parameters are owned by its metadata builder,
        not by the full-graph parameter-update channel.

        Selection note: _get_graph_update_backend picks the first executable
        backend, so correct updates for mixed-attention models rely on the
        full-attention (GQA) group being scanned first — M3's full-attention
        layers are the first layers registered, which holds today.
        """
        # （空操作：索引器的图重放参数由其元数据构建器管理，
        #   不走全图参数更新通道。选择说明：_get_graph_update_backend 取
        #   第一个可执行后端，混合注意力模型的正确更新依赖全注意力组
        #   先被扫描 —— M3 的全注意力层注册在前，当前成立。）


class AscendMiniMaxM3Indexer(nn.Module):
    """索引器对外包装：持有 impl 并提供 index_cache 属性代理。"""

    def __init__(
        self,
        *,
        num_kv_heads: int,
        scale: float,
        topk_blocks: int,
        sparse_block_size: int,
        num_index_heads: int,
        index_head_dim: int,
        prefix: str,
        init_blocks: int = 0,
        local_blocks: int = 0,
        cache_config: CacheConfig | None = None,
        kv_cache_dtype: str = "auto",
        kv_cache_torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.impl = AscendMiniMaxM3IndexerImpl(
            num_kv_heads=num_kv_heads,
            scale=scale,
            topk_blocks=topk_blocks,
            sparse_block_size=sparse_block_size,
            num_index_heads=num_index_heads,
            index_head_dim=index_head_dim,
            prefix=prefix,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            cache_config=cache_config,
            kv_cache_dtype=kv_cache_dtype,
            kv_cache_torch_dtype=kv_cache_torch_dtype,
        )

    @property
    def index_cache(self) -> AscendMiniMaxM3IndexerCache:
        # 代理到 impl 的缓存（外部通过 indexer.index_cache 访问）。
        return self.impl.index_cache

    def forward(
        self,
        index_query: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        return self.impl(index_query)


class AscendMiniMaxM3SparseBackend(AttentionBackend):
    """主稀疏注意力后端：申报主 KV cache 的能力与 5D 布局。"""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16, torch.float16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_name() -> str:
        return "ASCEND_MINIMAX_M3_SPARSE"

    @staticmethod
    def get_impl_cls() -> type[AscendMiniMaxM3SparseImpl]:
        return AscendMiniMaxM3SparseImpl

    @staticmethod
    def get_builder_cls() -> type[AscendMiniMaxM3SparseMetadataBuilder]:
        return AscendMiniMaxM3SparseMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [128]

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # 主 KV 布局：[2(K/V), 块数, 块大小, KV 头数, 头维]。
        # K/V 拼在第一维（与常见布局把 K/V 放最后维不同，
        # 便于内核按连续内存分别访问 K 段与 V 段）。
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            raise NotImplementedError
        return (0, 1, 2, 3, 4)


@dataclass
class AscendMiniMaxM3SparsePrefillMetadata:
    """主稀疏注意力的 prefill 元数据。

    cu_seqlens_k：各请求 KV 长度的累计偏移（变长核用）；
    total_kv_blocks / max_kv_blocks：块数统计（核内显存规划）。
    """
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    seq_lens: torch.Tensor
    context_lens: torch.Tensor
    block_table: torch.Tensor
    max_query_len: int
    max_seq_len: int
    total_kv_blocks: int
    max_kv_blocks: int


@dataclass
class AscendMiniMaxM3SparseDecodeMetadata:
    """主稀疏注意力的 decode 元数据。"""
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    max_seq_len: int
    decode_query_len: int


@dataclass
class AscendMiniMaxM3SparseMetadata(AttentionMetadata):
    """主稀疏注意力的整步元数据。"""
    seq_lens: torch.Tensor
    max_seq_len: int
    slot_mapping: torch.Tensor
    num_actual_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    prefill: AscendMiniMaxM3SparsePrefillMetadata | None = None
    decode: AscendMiniMaxM3SparseDecodeMetadata | None = None


class AscendMiniMaxM3SparseMetadataBuilder(AttentionMetadataBuilder[AscendMiniMaxM3SparseMetadata]):
    """主稀疏注意力的元数据构建器（与索引器构建器同构，多算块统计）。"""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        self.context_len_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendMiniMaxM3SparseMetadata:
        """构建主稀疏注意力元数据：decode/prefill 分拆 + KV 块统计。

        与索引器构建器的差异：prefill 段额外计算 cu_seqlens_k（KV 长度
        累计）与块数统计（total/max kv blocks），供变长稀疏核规划迭代。
        """
        num_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table = common_attn_metadata.block_table_tensor
        qsl_cpu = common_attn_metadata.query_start_loc_cpu

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.reorder_batch_threshold,
            require_uniform=True,
        )

        prefill_metadata: AscendMiniMaxM3SparsePrefillMetadata | None = None
        active_prefills = 0
        if num_prefills > 0:
            active_prefills = _active_prefill_num_reqs(num_prefills, num_prefill_tokens, qsl_cpu, num_decodes)
            prefill_end = num_decodes + active_prefills
            prefill_kv_lens = seq_lens[num_decodes:prefill_end]
            # KV 块数 = ceil(KV 长 / 块大小)，逐请求计算。
            prefill_kv_lens_cpu = prefill_kv_lens.detach().cpu()
            prefill_block_lens_cpu = torch.div(
                prefill_kv_lens_cpu + self.block_size - 1,
                self.block_size,
                rounding_mode="floor",
            )
            # 块数统计：总数（核内显存规划）与最大值（单请求上限）。
            total_kv_blocks = int(prefill_block_lens_cpu.sum().item())
            max_kv_blocks = int(prefill_block_lens_cpu.max().item()) if prefill_block_lens_cpu.numel() else 0
            # cu_seqlens_k：KV 长度累计偏移（cumsum 写入预分配张量，省一次分配）。
            prefill_cu_seqlens_k = torch.empty(active_prefills + 1, dtype=torch.int32, device=seq_lens.device)
            prefill_cu_seqlens_k[0] = 0
            torch.cumsum(prefill_kv_lens, dim=0, out=prefill_cu_seqlens_k[1:])
            prefill_query_lens_cpu = qsl_cpu[num_decodes + 1 : prefill_end + 1] - qsl_cpu[num_decodes:prefill_end]
            prefill_context_lens = self.context_len_buffer[num_decodes:prefill_end]
            prefill_context_lens.copy_(
                (prefill_kv_lens_cpu - prefill_query_lens_cpu).to(
                    device=self.context_len_buffer.device,
                    dtype=torch.int32,
                    non_blocking=True,
                ),
                non_blocking=True,
            )
            cu_seqlens_q = (query_start_loc[num_decodes : prefill_end + 1] - num_decode_tokens).to(torch.int32)
            prefill_metadata = AscendMiniMaxM3SparsePrefillMetadata(
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=prefill_cu_seqlens_k,
                seq_lens=prefill_kv_lens,
                context_lens=prefill_context_lens,
                block_table=block_table[num_decodes:prefill_end],
                max_query_len=common_attn_metadata.max_query_len,
                max_seq_len=common_attn_metadata.max_seq_len,
                total_kv_blocks=total_kv_blocks,
                max_kv_blocks=max_kv_blocks,
            )

        decode_metadata: AscendMiniMaxM3SparseDecodeMetadata | None = None
        active_decodes = 0
        if num_decodes > 0:
            qsl_cpu = common_attn_metadata.query_start_loc_cpu
            query_lens_cpu = qsl_cpu[1 : num_decodes + 1] - qsl_cpu[:num_decodes]
            decode_query_len = int(query_lens_cpu[0].item())
            active_decodes = _active_decode_num_reqs(num_decodes, num_decode_tokens, decode_query_len)
            decode_metadata = AscendMiniMaxM3SparseDecodeMetadata(
                seq_lens=seq_lens[:active_decodes],
                block_table=block_table[:active_decodes],
                max_seq_len=common_attn_metadata.max_seq_len,
                decode_query_len=decode_query_len,
            )

        return AscendMiniMaxM3SparseMetadata(
            seq_lens=seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_actual_tokens=num_tokens,
            num_decodes=active_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=active_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )


class AscendMiniMaxM3SparseImpl(AttentionImplBase[AscendMiniMaxM3SparseMetadata]):
    """主稀疏注意力实现：消费索引器的 top-k 块索引执行精确注意力。"""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        kv_cache_dtype: str = "auto",
        *,
        topk_blocks: int,
        sparse_block_size: int,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.topk_blocks = topk_blocks
        self.block_size = sparse_block_size
        # FP8 cache 的反量化 scale（固定 scale=1 的占位，惰性创建）。
        self._dequant_scale: torch.Tensor | None = None

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        topk_idx: tuple[
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
        ],
        output: torch.Tensor,
    ) -> torch.Tensor:
        """稀疏注意力前向：decode 段与 prefill 段分别调用 NPU 内核。

        query: [num_tokens, heads*head_dim]；kv_cache 是 (K, V) 5D 张量；
        topk_idx 是索引器返回的三元组；output 预分配的输出缓冲。
        """
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return output
        main_md = attn_metadata[layer.layer_name]
        assert isinstance(main_md, AscendMiniMaxM3SparseMetadata)
        decode_topk, prefill_topk, decode_select_num_idx = topk_idx

        num_decode_tokens = main_md.num_decode_tokens
        num_tokens = main_md.num_actual_tokens
        hd = self.head_size
        # 展平为 [num_tokens, heads, head_dim] 视图。
        q = query[:num_tokens].view(-1, self.num_heads, hd)
        out = output[:num_tokens].view(-1, self.num_heads, hd)

        if main_md.num_decodes > 0:
            # decode 段：批量解码核（只 gather top-k 块的 KV）。
            d = main_md.decode
            assert d is not None and decode_topk is not None
            # FP8 缓存首次使用时创建单位 scale（M3 固定 scale 方案）。
            if self.kv_cache_dtype.startswith("fp8") and self._dequant_scale is None:
                self._dequant_scale = torch.ones(
                    (1, 1, 1, 1),
                    dtype=torch.float32,
                    device=query.device,
                )
            minimax_m3_sparse_attn_decode(
                q[:num_decode_tokens],
                kv_cache,
                decode_topk,
                d.block_table,
                d.seq_lens,
                self.num_kv_heads,
                self.scale,
                out[:num_decode_tokens],
                d.decode_query_len,
                block_size=self.block_size,
                select_num_idx=decode_select_num_idx,
                dequant_scale=self._dequant_scale,
            )

        if main_md.num_prefills > 0:
            # prefill 段：变长批量核（cu_seqlens + 块统计）。
            p = main_md.prefill
            assert p is not None and prefill_topk is not None
            minimax_m3_sparse_attn(
                q[num_decode_tokens:],
                kv_cache,
                prefill_topk,
                p.block_table,
                p.cu_seqlens_q,
                p.seq_lens,
                p.context_lens,
                p.max_query_len,
                self.num_kv_heads,
                self.scale,
                out[num_decode_tokens:],
                block_size=self.block_size,
                total_kv_blocks=p.total_kv_blocks,
                max_kv_blocks=p.max_kv_blocks,
            )
        return output

    @staticmethod
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config=None,
        draft_attn_metadatas=None,
    ):
        """No-op: sparse-attention replay parameters are owned by its metadata
        builder, not by the full-graph parameter-update channel.

        Selection note: _get_graph_update_backend picks the first executable
        backend, so correct updates for mixed-attention models rely on the
        full-attention (GQA) group being scanned first — M3's full-attention
        layers are the first layers registered, which holds today.
        """
        # （同索引器：图重放参数由元数据构建器管理，此处空操作。）


class AscendMiniMaxM3QKVParallelLinearWithIndexer(QKVParallelLinear):
    """Fused [q | k | v | index_q | index_k] column-parallel GEMM for M3 sparse layers."""
    """（五段融合 QKV+索引投影：一个 GEMM 产出主 Q/K/V 与索引 q/k。）

    TP 切分策略：
    - q/k/v 按 QKVParallelLinear 标准切分（头维切分/复制）；
    - index_q 跟随 KV 头切分（每 KV 头一个 index 头）；
    - index_k 跨 rank 复制（所有 rank 都要完整 index_k 给本地块打分）。
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        total_num_index_heads: int,
        index_head_size: int,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        # index 头数必须等于 KV 头数（每个 KV 头配一个 index 头）。
        assert total_num_index_heads == total_num_kv_heads, (
            "AscendMiniMaxM3QKVParallelLinearWithIndexer requires total_num_index_heads == total_num_kv_heads"
        )
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.v_head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.total_num_index_heads = total_num_index_heads
        self.index_head_size = index_head_size

        # TP 切分：divide 是带断言的整除（不整除立即报错）。
        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            # KV 头少于 TP：每 rank 1 个头，多份复制。
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        self.num_index_heads = self.num_kv_heads

        # 五段输出宽度（× tp_size 还原成"全局"总宽度，供权重加载对齐）。
        # 注意 index_k 段只有 1 份（不随 TP 放大），加载时特殊处理。
        q = self.num_heads * self.head_size
        kv = self.num_kv_heads * self.head_size
        iq = self.num_index_heads * self.index_head_size
        ik = self.index_head_size
        self.output_sizes = [
            q * tp_size,
            kv * tp_size,
            kv * tp_size,
            iq * tp_size,
            ik * tp_size,
        ]

        # 昇腾并行线性算子（编译图内的自定义算子形式）。
        self.custom_op, _, _ = get_parallel_op(False, prefix, self, "column")
        # 借用 AscendColumnParallelLinear 的初始化（跳过 QKVParallelLinear
        # 的 __init__，因为分片逻辑已在上面自行完成）。
        AscendColumnParallelLinear.__init__(
            self,
            input_size=self.hidden_size,
            output_size=sum(self.output_sizes),
            bias=bias,
            gather_output=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def forward(self, input_):
        # 优先走自定义算子（图编译友好），否则走父类 forward。
        if self.custom_op is not None:
            return self.custom_op.apply(input_)
        return super().forward(input_)

    def validate_shard_id(self, loaded_shard_id: str | None) -> None:
        # 校验权重加载的分片 id 合法性（五段之一）。
        if loaded_shard_id is None:
            return
        if loaded_shard_id not in ("q", "k", "v", "index_q", "index_k"):
            raise ValueError(
                "Shard id for AscendMiniMaxM3QKVParallelLinearWithIndexer must be "
                "one of 'q', 'k', 'v', 'index_q', 'index_k'; got "
                f"{loaded_shard_id}."
            )

    def _get_shard_offset_mapping(self, loaded_shard_id: str) -> int | None:
        # 各分片在融合输出中的起始偏移：[q | k | v | index_q | index_k]。
        h = self.head_size
        nq, nkv, nidx = self.num_heads, self.num_kv_heads, self.num_index_heads
        return {
            "q": 0,
            "k": nq * h,
            "v": (nq + nkv) * h,
            "index_q": (nq + 2 * nkv) * h,
            "index_k": (nq + 2 * nkv + nidx) * h,
        }.get(loaded_shard_id)

    def _get_shard_size_mapping(self, loaded_shard_id: str) -> int | None:
        # 各分片的宽度（注意 index_k 是 index_head_size 而非 头数×头维）。
        h = self.head_size
        return {
            "q": self.num_heads * h,
            "k": self.num_kv_heads * h,
            "v": self.num_kv_heads * h,
            "index_q": self.num_index_heads * h,
            "index_k": self.index_head_size,
        }.get(loaded_shard_id)

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight,
        loaded_shard_id: str | None = None,
    ) -> None:
        """v2 加载协议：基于分片元数据的参数装填。"""
        self.validate_shard_id(loaded_shard_id)
        assert loaded_shard_id in ("q", "k", "v", "index_q", "index_k")

        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)
        assert shard_offset is not None and shard_size is not None
        # 块量化 scale 的分片对齐（scale 的块结构与权重不同）。
        if isinstance(param, BlockQuantScaleParameter):
            weight_block_size = getattr(self, "weight_block_size", None)
            shard_size, shard_offset = adjust_block_scale_shard(weight_block_size, shard_size, shard_offset)

        # index_k 复制到全部 tp_size 个槽位；其余按 KV 头复制数装填。
        num_heads = self.tp_size if loaded_shard_id == "index_k" else self.num_kv_head_replicas
        param.load_qkv_weight(
            loaded_weight=loaded_weight,
            num_heads=num_heads,
            shard_id=loaded_shard_id,
            shard_offset=shard_offset,
            shard_size=shard_size,
            tp_rank=self.tp_rank,
        )

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight,
        loaded_shard_id: str | None = None,
    ) -> None:
        """v1 加载协议：按偏移切窄条直接拷贝。"""
        self.validate_shard_id(loaded_shard_id)
        assert loaded_shard_id in ("q", "k", "v", "index_q", "index_k")
        output_dim = getattr(param, "output_dim", None)
        assert output_dim is not None

        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)
        assert shard_offset is not None and shard_size is not None
        if isinstance(param, BlockQuantScaleParameter):
            weight_block_size = getattr(self, "weight_block_size", None)
            shard_size, shard_offset = adjust_block_scale_shard(weight_block_size, shard_size, shard_offset)

        # narrow 零拷贝切出参数的目标窄条。
        param_data = param.data.narrow(output_dim, shard_offset, shard_size)
        if loaded_shard_id == "q":
            # q 段：每 rank 取自己的头段。
            shard_rank = self.tp_rank
        elif loaded_shard_id == "index_k":
            # index_k 段：全部 rank 取第 0 份（复制语义）。
            shard_rank = 0
        else:
            # k/v/index_q 段：按 KV 头复制组取段。
            shard_rank = self.tp_rank // self.num_kv_head_replicas
        loaded_weight = loaded_weight.narrow(output_dim, shard_rank * shard_size, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


class AscendMiniMaxM3IndexerLinear(AscendColumnParallelLinear):
    """Merged [index_q | index_k] projection for M3 sparse layers.

    index_q follows KV-head tensor parallel sharding. index_k is always
    replicated, so every TP rank loads the first index_k shard.
    """
    """（独立索引投影 [index_q | index_k]：q 段按 KV 头切分、k 段复制。）"""

    def __init__(
        self,
        hidden_size: int,
        total_num_index_heads: int,
        index_head_size: int,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        self.hidden_size = hidden_size
        self.total_num_index_heads = total_num_index_heads
        self.index_head_size = index_head_size

        tp_size = get_tensor_model_parallel_world_size()
        if tp_size >= self.total_num_index_heads:
            self.num_index_heads = 1
            self.num_index_head_replicas = divide(tp_size, self.total_num_index_heads)
        else:
            self.num_index_heads = divide(self.total_num_index_heads, tp_size)
            self.num_index_head_replicas = 1

        self.index_q_size = self.num_index_heads * self.index_head_size
        self.index_k_size = self.index_head_size
        # 两段全局宽度（k 段名义上 ×tp_size，实际每 rank 相同副本）。
        self.output_sizes = [
            self.index_q_size * tp_size,
            self.index_k_size * tp_size,
        ]

        self.custom_op, _, _ = get_parallel_op(False, prefix, self, "column")
        AscendColumnParallelLinear.__init__(
            self,
            input_size=self.hidden_size,
            output_size=sum(self.output_sizes),
            bias=bias,
            gather_output=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def forward(self, input_):
        if self.custom_op is not None:
            return self.custom_op.apply(input_)
        return super().forward(input_)

    def validate_shard_id(self, loaded_shard_id: str | None) -> None:
        if loaded_shard_id is None:
            return
        if loaded_shard_id not in ("index_q", "index_k"):
            raise ValueError(
                f"Shard id for AscendMinimaxM3Indexer must be one of 'index_q', 'index_k'; got {loaded_shard_id}."
            )

    def _get_shard_offset_mapping(self, loaded_shard_id: str) -> int | None:
        # [index_q | index_k] 两段布局。
        return {
            "index_q": 0,
            "index_k": self.index_q_size,
        }.get(loaded_shard_id)

    def _get_shard_size_mapping(self, loaded_shard_id: str) -> int | None:
        return {
            "index_q": self.index_q_size,
            "index_k": self.index_k_size,
        }.get(loaded_shard_id)

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight,
        loaded_shard_id: str | None = None,
    ) -> None:
        """v2 加载协议（同融合版的逻辑，仅两段）。"""
        self.validate_shard_id(loaded_shard_id)
        assert loaded_shard_id in ("index_q", "index_k")

        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)
        assert shard_offset is not None and shard_size is not None
        if isinstance(param, BlockQuantScaleParameter):
            weight_block_size = getattr(self, "weight_block_size", None)
            shard_size, shard_offset = adjust_block_scale_shard(weight_block_size, shard_size, shard_offset)

        num_heads = self.tp_size if loaded_shard_id == "index_k" else self.num_index_head_replicas
        param.load_qkv_weight(
            loaded_weight=loaded_weight,
            num_heads=num_heads,
            shard_id=loaded_shard_id,
            shard_offset=shard_offset,
            shard_size=shard_size,
            tp_rank=self.tp_rank,
        )

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight,
        loaded_shard_id: str | None = None,
    ) -> None:
        """v1 加载协议（同融合版的逻辑，仅两段）。"""
        self.validate_shard_id(loaded_shard_id)
        assert loaded_shard_id in ("index_q", "index_k")
        output_dim = getattr(param, "output_dim", None)
        assert output_dim is not None

        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)
        assert shard_offset is not None and shard_size is not None
        if isinstance(param, BlockQuantScaleParameter):
            weight_block_size = getattr(self, "weight_block_size", None)
            shard_size, shard_offset = adjust_block_scale_shard(weight_block_size, shard_size, shard_offset)
        assert shard_size is not None

        param_data = param.data.narrow(output_dim, shard_offset, shard_size)
        if loaded_shard_id == "index_k":
            # index_k 复制：所有 rank 取第 0 份。
            shard_rank = 0
        else:
            shard_rank = self.tp_rank // self.num_index_head_replicas
        loaded_weight = loaded_weight.narrow(output_dim, shard_rank * shard_size, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


def _quant_description_value(
    quant_config: QuantizationConfig | None,
    key: str,
) -> Any | None:
    """从量化配置的描述字典中按键取值（无描述时返回 None）。"""
    quant_description = getattr(quant_config, "quant_description", None)
    if not isinstance(quant_description, dict):
        return None
    return quant_description.get(key)


def _sparse_proj_quant_type(
    quant_config: QuantizationConfig | None,
    prefix: str,
    proj_name: str,
) -> Any | None:
    """查询某个投影的量化类型，兼容多种前缀拼写。

    候选前缀：原样 / language_model. 前缀（多模态 checkpoint 命名差异）。
    dict.fromkeys(candidates)：保序去重（避免重复查询）。
    """
    candidates = [prefix]
    if not prefix.startswith("language_model."):
        candidates.append(f"language_model.{prefix}")
    if prefix.startswith("model."):
        candidates.append(f"language_model.{prefix}")

    for candidate in dict.fromkeys(candidates):
        value = _quant_description_value(
            quant_config,
            f"{candidate}.{proj_name}.weight",
        )
        if value is not None:
            return value
    return None


def _use_fused_qkv_indexer(
    quant_config: QuantizationConfig | None,
    prefix: str,
) -> bool:
    """判定能否把 index 投影融合进主 QKV 投影。

    融合的前提：q/k/v 与 index_q/index_k 的量化类型完全一致
    （一个 GEMM 只能用一种量化方案）。类型不一致时：
    - q/k/v 之间不一致 → 报错（配置矛盾）；
    - index 与主 QKV 不一致 → 不融合（走分离投影路径）。
    """
    if quant_config is None:
        return True

    qkv_types = [
        _sparse_proj_quant_type(quant_config, prefix, proj_name) for proj_name in ("q_proj", "k_proj", "v_proj")
    ]
    index_q_type = _sparse_proj_quant_type(quant_config, prefix, "index_q_proj")
    index_k_type = _sparse_proj_quant_type(quant_config, prefix, "index_k_proj")

    # 有未知类型（不在量化描述中）时保守地融合（按未量化处理）。
    if any(value is None for value in (*qkv_types, index_q_type, index_k_type)):
        return True
    if len(set(qkv_types)) != 1:
        raise ValueError(f"MiniMax M3 q/k/v quantization types differ for {prefix}: {qkv_types}")
    if index_q_type != index_k_type:
        raise ValueError(
            f"MiniMax M3 index_q/index_k quantization types differ for {prefix}: {index_q_type} vs {index_k_type}"
        )
    return qkv_types[0] == index_q_type


def _register_m3_sparse_packed_modules(
    quant_config: QuantizationConfig | None,
    fused_qkv_indexer: bool,
) -> None:
    """向量化配置登记融合参数映射（量化器据此处理融合参数的 scale）。"""
    if quant_config is None:
        return
    packed_modules_mapping = getattr(quant_config, "packed_modules_mapping", None)
    if not isinstance(packed_modules_mapping, dict):
        return
    packed_modules_mapping["qkv_proj"] = ["q_proj", "k_proj", "v_proj"]
    if not fused_qkv_indexer:
        packed_modules_mapping["indexer_proj"] = ["index_q_proj", "index_k_proj"]


def minimax_m3_sparse_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    index_query: torch.Tensor,
    index_key: torch.Tensor,
    attn_output: torch.Tensor,
    layer_name: str,
) -> None:
    """稀疏注意力自定义算子的真身（torch.compile 图内调用入口）。

    从 no_compile_layers 按层名找回真实层实例（图编译时模块被副本替换），
    执行三步：写 KV → 索引器 top-k → 稀疏注意力。
    """
    forward_context: ForwardContext = get_forward_context()

    attn_metadata = forward_context.attn_metadata
    if not isinstance(attn_metadata, dict):
        # 无注意力元数据（如 profile 阶段）：清零输出安全返回。
        attn_output.zero_()
        return

    layer = forward_context.no_compile_layers[layer_name]
    layer._run_sparse_attention(query, key, value, index_query, index_key, attn_output)


def minimax_m3_sparse_forward_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    index_query: torch.Tensor,
    index_key: torch.Tensor,
    attn_output: torch.Tensor,
    layer_name: str,
) -> None:
    # fake 实现：Dynamo 追踪时的元内核（无实际计算，只保证形状传播）。
    return


# direct_register_custom_op：向 torch 注册自定义算子（torch.ops.vllm.* 命名空间）。
# mutates_args 声明被原地修改的参数（attn_output），保证编译器正确处理副作用；
# dispatch_key="PrivateUse1"：注册到 NPU 的私有后端分发键。
direct_register_custom_op(
    op_name="minimax_m3_sparse_forward",
    op_func=minimax_m3_sparse_forward,
    mutates_args=["attn_output"],
    fake_impl=minimax_m3_sparse_forward_fake,
    dispatch_key="PrivateUse1",
)
