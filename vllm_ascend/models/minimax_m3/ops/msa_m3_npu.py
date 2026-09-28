# SPDX-License-Identifier: Apache-2.0
# =====================================================================================
# 中文注释（教学向）：MiniMax-M3 MSA 稀疏注意力的昇腾 NPU 原生算子封装层。
#
# 文件职责：
#   将华为 CANN 提供的 AscendC 自定义算子（通过 torch.ops._C_ascend.* 与
#   torch_npu.* 暴露）包装成 Python 友好的接口，供 msa_m3.py 中的注意力后端调用。
#   包含三部分功能：
#     1. 索引打分（index score）：闪电索引器 Q_idx 与索引 K cache 做点积，
#        为每个 128-token 稀疏块算一个分数 —— npu_msa_index_score；
#     2. TopK 块选择：从分数中选出每 query 的 top-k 个块 id（含强制 init/local 块）；
#     3. 块稀疏注意力主计算：只在被选中的块上做 softmax(QK^T)V ——
#        npu_sparse_attention_score（Q-gather-KV / A3 路径）与
#        npu_sparse_attention_score_prefill（KV-gather-Q / K2Q CSR 路径）。
#
# MSA 稀疏注意力原理（MiniMax Sparse Attention）：
#   MiniMax M3 把 KV 序列切成 128-token 的"块"（block）。每个 query token 先用
#   低成本的"索引头"（index head，比主注意力头维度更低）对每个块打分：
#       score(block) = max_{k in block} (q_idx · k_idx)
#   然后取分数最高的 topk_blocks 个块，主注意力头只在这些块上做标准 softmax
#   注意力，从而把注意力复杂度从 O(n^2) 降到 O(n * topk * block_size)。
#   此外强制保留两类块：init_blocks（序列头部块）与 local_blocks（紧邻当前
#   query 的最近块），保证局部信息不丢失。
#
# 与 Triton 参考实现的对应关系：
#   - _minimax_m3_index_score       ≈ msa_m3_triton*.py: _prefill_index_score_kernel /
#                                    _decode_qk_score_kernel（索引打分）
#   - _index_score_topk_candidates  ≈ minimax_m3_index_topk / _index_topk_postprocess（TopK）
#   - _minimax_m3_sparse_attn_a3    ≈ _gqa_sparse_fwd_kernel（Q 逐块 gather KV 注意力）
#   - _minimax_m3_sparse_attn_kv_gather_q ≈ 同上，但反过来按 KV 块 gather 所有
#                                    相关 Q（prefill 更高效，用 K2Q CSR 稀疏格式）
#
# 关键 NPU 适配点：
#   - FP8 KV cache：E4M3 定标存储（scale=1），算子原生支持，需传 dequant_scale=1；
#   - A3/A5 设备差异：inner_precise 精度模式不同（A3 用 FP32 累加保精度）；
#   - 图捕获（ACL Graph）安全：派生张量在 forward 内创建而非 metadata builder。
# =====================================================================================

"""NPU sparse attention ops for MiniMax-M3 on Ascend."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.utils import enable_custom_op

# Existing mode for legacy prefill, decode, and A5 FP8 prefill.
# 中文注释：AscendC 稀疏注意力算子的 inner_precise 模式常量（4 = 默认精度模式），
# 用于传统 prefill、decode 以及 A5 的 FP8 prefill 路径。
_SPARSE_ATTN_INNER_PRECISE = 4

# Existing mode for A5 KV-gather-Q prefill with BF16 inputs.
# 中文注释：A5 设备上 KV-gather-Q prefill（BF16 输入）使用的 inner_precise 模式（1）。
_PREFILL_KV_GATHER_Q_INNER_PRECISE = 1

# A3 KV-gather-Q prefill with BF16 inputs: use FP32 scores and partial
# outputs to preserve accuracy.
# 中文注释：A3 设备用 0 号模式（FP32 分数与部分和累加），因为 A3 的 BF16 矩阵
# 指令累加精度不足，需要更高精度中间结果保住精度指标。
_A3_PREFILL_KV_GATHER_Q_INNER_PRECISE = 0
# MSA 稀疏块大小：128 token 一个块，也是 KV cache 的 page 大小（一页 = 一块）
_MSA_INDEX_BLOCK_SIZE = 128
# 分数张量最后一维（块维）的对齐要求：按 16 对齐避免算子内部越界/性能惩罚
_MSA_SCORE_BLOCK_ALIGNMENT = 16
# FP8 E4M3 可表示的最大有限值（448），量化前截断防止溢出到 inf/nan
_FP8_E4M3_MAX = 448.0

# 中文注释：按硬件能力选择 prefill TopK 的实现：
#   - A5（Atlas 950，支持 FP8 注意力）：用 A5 专用 Triton topk kernel（更快）；
#   - A3（Atlas 910 系列，支持运行时自定义算子）：用通用 Triton topk kernel。
# 两者接口一致，都是 minimax_m3_index_topk。
if get_current_hardware_profile().supports(HardwareCapability.FP8_ATTENTION):
    from vllm_ascend.models.minimax_m3.ops.msa_m3_triton_a5 import (
        minimax_m3_index_topk as _minimax_m3_index_prefill_topk,
    )
elif get_current_hardware_profile().supports(HardwareCapability.RUNTIME_CUSTOM_OPS):
    from vllm_ascend.models.minimax_m3.ops.msa_m3_triton import (
        minimax_m3_index_topk as _minimax_m3_index_prefill_topk,
    )


def _k2q_csr_block_stats(cu_block_lens: torch.Tensor) -> tuple[int, int]:
    """Derive ``(total_rows, max_kv)`` from ``cu_block_lens`` on host.

    中文注释：在主机端（CPU）从"每请求 KV 块数的累积和"推导 K2Q CSR 所需的
    两个规模参数：
      - total_rows：所有 prefill 请求的 KV 块总数（CSR 矩阵的总行数）
      - max_kv：单个请求最多的 KV 块数（用于确定 Q 索引位宽）
    原理：cu_block_lens 形如 [0, b0, b0+b1, ...]，相邻差分即每请求块数。
    """
    cu = cu_block_lens.reshape(-1)
    if cu.numel() <= 1:
        # 只有一个元素（0）说明没有请求，返回 (0, 0)
        return 0, 0
    # 差分得到每请求的块数列表
    block_lens = cu[1:] - cu[:-1]
    total_rows = int(cu[-1].item())
    max_kv = int(block_lens.max().item()) if block_lens.numel() else 0
    return total_rows, max_kv


@torch.no_grad()
def _npu_k2q_csr(
    q2k: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_block_lens: torch.Tensor,
    order_method: int = 0,
    total_rows: int = -1,
    max_kv: int = -1,
    use_simt: int | bool = 0,
    q_global_offset: int | bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert MiniMax-M3 q2k indices to k2q CSR on NPU.

    中文注释：把 Q2K（每个 query 选了哪些块）的 topk 结果转置成 K2Q（每个 KV 块
    被哪些 query 选中）的 CSR 稀疏格式。这是 prefill KV-gather-Q 路径的核心
    预处理：后续算子按 KV 块遍历，每块 gather 所有相关 query 一次算完。
    参数：
      q2k: [num_heads, total_q, topk] 每 query 选中的块 id（-1 为无效）
      cu_seqlens: [batch+1] query 累积偏移
      cu_block_lens: [batch+1] KV 块数累积偏移
      order_method: CSR 内部排序方式（1 = 按 query 位置排序，利于稳定访存）
      total_rows/max_kv: 可预先推导好的规模参数，-1 表示由本函数推导
      use_simt: 是否用 SIMT（标量）模式实现，某些形状下更优
      q_global_offset: Q 索引是否带全局（跨请求）偏移
    返回：(row_ptr, q_indices, slot_indices) 三个 CSR 组件。
    """
    # A5 disables the generic custom-op loader, so register the in-tree
    # MiniMax M3 operators lazily after the NPU runtime has been initialized.
    # 中文注释：import 语句本身触发算子库的 lazy 注册（A5 关闭了通用自定义算子
    # 加载器，必须在 NPU runtime 初始化后再注册）。
    import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: F401, PLC0415

    enable_custom_op()
    # 未提供规模参数时在主机端推导
    if total_rows < 0 or max_kv < 0:
        derived_total_rows, derived_max_kv = _k2q_csr_block_stats(cu_block_lens)
        if total_rows < 0:
            total_rows = derived_total_rows
        if max_kv < 0:
            max_kv = derived_max_kv
    return torch.ops._C_ascend.npu_k2q_csr(
        q2k,
        cu_seqlens,
        cu_block_lens,
        int(order_method),
        int(total_rows),
        int(max_kv),
        int(use_simt),
        int(q_global_offset),
    )


@dataclass
class MiniMaxM3TPDecodeScoreMetadata:
    """Graph-stable inputs for packed TP decode scoring.

    中文注释：TP（张量并行）分块 decode 打分所需的"图稳定"输入打包。
    所谓图稳定：这些张量的形状在 ACL Graph 捕获/回放期间保持不变（只有内容
    变化），因此可以在 metadata builder 中预分配，回放时原地更新。
    字段：
      block_table: [num_reqs, max_blocks] KV 块表（逻辑块 -> 物理 page）
      cu_seqlens_q: [num_reqs+1] query 累积偏移
      context_lens: [num_reqs] 每请求已生成的上下文长度（不含当前 query）
      max_block_count: 全局最大块数
      block_size: 块大小（128）
      block_offset/block_count: 当前 TP rank 负责的块区间 [offset, offset+count)
      decode_query_len: 每 request 的 query 数（投机解码时 >1）
    """

    block_table: torch.Tensor
    cu_seqlens_q: torch.Tensor
    context_lens: torch.Tensor
    max_block_count: int
    block_size: int
    block_offset: int
    block_count: int
    decode_query_len: int


def _as_ascendc_index_kv_cache(
    index_kv_cache: torch.Tensor | tuple[torch.Tensor],
) -> torch.Tensor:
    """Convert the runtime index K cache to the AscendC BBND layout.

    The model runner binds this K-only cache as a one-element tuple whose
    tensor is shaped ``[num_blocks, 128, head_dim]``. Direct calls may pass
    that tensor without the tuple. MsaIndexScore expects the BBND shape
    ``[num_blocks, 128, 1, head_dim]`` instead.
    """
    if isinstance(index_kv_cache, tuple):
        # 模型 runner 把 K-only 缓存绑定为单元素元组，这里解包
        index_kv_cache = index_kv_cache[0]
    # unsqueeze(2) 在头部维插入大小为 1 的维：[num_blocks, 128, head_dim] ->
    # [num_blocks, 128, 1, head_dim]，即 MsaIndexScore 期望的 BBND 四维布局
    return index_kv_cache.unsqueeze(2)


def _split_main_kv_cache(
    kv_cache: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    # 中文注释：把主 KV cache 拆成独立的 K、V 两个张量。
    # 支持三种输入布局：
    #   1. (k_cache, v_cache) 元组/列表 —— Ascend 绑定的分离布局；
    #   2. 单个 5 维张量且 dim0==2：[2, num_blocks, block, heads, dim]（K 在前）；
    #   3. 单个 5 维张量且 dim1==2：[num_blocks, 2, block, heads, dim]。
    # 拆分后统一校验为 4 维 [num_blocks, block_size, num_kv_heads, head_dim]。
    if isinstance(kv_cache, (tuple, list)):
        if len(kv_cache) < 2:
            raise ValueError("Main kv cache tuple must contain K and V tensors")
        k_cache, v_cache = kv_cache[0], kv_cache[1]
    else:
        if kv_cache.ndim != 5:
            raise ValueError(f"Unexpected main kv cache ndim: {kv_cache.ndim}")
        if kv_cache.shape[0] == 2:
            k_cache, v_cache = kv_cache[0], kv_cache[1]
        elif kv_cache.shape[1] == 2:
            k_cache, v_cache = kv_cache[:, 0], kv_cache[:, 1]
        else:
            raise ValueError(f"Unexpected main kv cache shape: {tuple(kv_cache.shape)}")
    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError(f"Unexpected split main kv cache shapes: {tuple(k_cache.shape)}, {tuple(v_cache.shape)}")
    return k_cache, v_cache


def _select_num_idx_from_topk(topk_idx: torch.Tensor) -> torch.Tensor:
    # 中文注释：统计每个 (head, query) 的有效块数：topk_idx 中 >=0 的元素个数。
    # 无效槽位用 -1 填充（块数不足 topk 时），AscendC 算子用 select_num_idx
    # 决定实际参与注意力的块数。输出 int32，形状 [num_heads, total_q]。
    return (topk_idx >= 0).sum(dim=-1).to(dtype=torch.int32)


def _to_fp8_e4m3(tensor: torch.Tensor) -> torch.Tensor:
    # 中文注释：转 FP8 E4M3。先 clamp 到 [-448, 448] 防止溢出（E4M3 最大有限值），
    # 再做 dtype 转换。MiniMax M3 的 KV cache 采用"固定 scale=1"的 E4M3 存储，
    # 即量化系数恒为 1，无需 per-tensor scale。
    return tensor.clamp(min=-_FP8_E4M3_MAX, max=_FP8_E4M3_MAX).to(torch.float8_e4m3fn)


def _build_cu_block_lens(
    seq_lens: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Build cumulative logical KV-block counts for each prefill request.

    中文注释：计算每个 prefill 请求的 KV 块数（向上取整）的累积和。
    例：seq_lens=[300, 100]，block_size=128 -> block_lens=[3, 1] ->
    cu_block_lens=[0, 3, 4]。这是 K2Q CSR 转换的输入之一。
    """
    # ceil 除法：每请求块数 = (seq_len + block_size - 1) // block_size
    block_lens = torch.div(
        seq_lens.to(torch.int32) + block_size - 1,
        block_size,
        rounding_mode="floor",
    )
    cu_block_lens = torch.empty(
        block_lens.numel() + 1,
        dtype=torch.int32,
        device=seq_lens.device,
    )
    cu_block_lens[0] = 0
    torch.cumsum(block_lens, dim=0, out=cu_block_lens[1:])
    return cu_block_lens


@torch.no_grad()
def _minimax_m3_index_score(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor | tuple[torch.Tensor],
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    start_loc: torch.Tensor,
    causal_mask: torch.Tensor | None,
    *,
    init_blocks: int = 0,
    local_blocks: int = 0,
) -> torch.Tensor:
    """Compute MSA index scores with the bundled AscendC operator.

    中文注释：调用 AscendC 自定义算子 npu_msa_index_score 计算索引分数。
    这是 MSA 闪电索引器的核心打分：对每个 (head, query, block) 输出一个
    分数 = 块内 index_q·index_k 的最大值（max 语义，捕获块内最强匹配）。

    参数：
      idx_q: [total_q, num_idx_heads, head_dim] 索引 query
      index_kv_cache: 索引 K cache（K-only，无 V）
      block_table / cu_seqlens_q / seq_lens: 分页块表与长度信息
      start_loc: 当前 query 在（可能 TP 分片的）块表中的起始块号
      causal_mask: 因果掩码（sparse_mode=3 使用）；None 则用 mode=0 全量模式

    NPU 适配点：BBND 布局转换、FP8 时把 idx_q 也转成 E4M3。

    ``start_loc`` is the current query block in the *passed block table*.  For
    a TP-sharded table this is a per-request local block index, not the scalar
    global ``block_offset`` used by the Triton decode kernel.

    A causal mask selects sparse mode 3. Passing no mask selects dense mode 0
    for a TP chunk that is entirely before the current query positions.
    ``init_blocks`` and ``local_blocks`` are kept for parity with the index
    scoring interface; candidate forcing is applied by the TopK stage.
    """
    index_kv_cache = _as_ascendc_index_kv_cache(index_kv_cache)
    # FP8 cache 时把 query 也转 E4M3，保证算子输入 dtype 一致
    if index_kv_cache.dtype == torch.float8_e4m3fn and idx_q.dtype != index_kv_cache.dtype:
        idx_q = _to_fp8_e4m3(idx_q)
    # sparse_mode: 3 = 带 causal mask 的稀疏模式；0 = 无 mask 的稠密模式
    # （TP 分片完全在 query 之前的块无需因果掩码）
    return torch.ops._C_ascend.npu_msa_index_score(
        idx_q,
        index_kv_cache,
        block_table,
        start_loc,
        atten_mask=causal_mask,
        actual_seq_qlen=cu_seqlens_q,
        actual_seq_klen=seq_lens,
        layout_key="BBND",
        sparse_mode=3 if causal_mask is not None else 0,
    )


@torch.no_grad()
def minimax_m3_index_prefill(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor | tuple[torch.Tensor],
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    context_lens: torch.Tensor,
    start_loc: torch.Tensor,
    causal_mask: torch.Tensor,
    *,
    max_query_len: int,
    max_seq_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    """Compute AscendC prefill scores and finalize their block TopK.

    中文注释：prefill 索引选择的两步流水：
      步骤1: _minimax_m3_index_score 用 AscendC 算子打分，输出
             [heads, total_q, score_width] 的块分数（score_width 可能大于
             逻辑块数，算子内部按对齐宽度填充）；
      步骤2: 截断到逻辑块宽度（按 16 对齐）后，调 Triton 的
             _minimax_m3_index_prefill_topk 完成 init/local 强制块注入与
             topk 选择，返回 [heads, total_q, topk] 的块 id。
    """
    score = _minimax_m3_index_score(
        idx_q,
        index_kv_cache,
        block_table,
        cu_seqlens_q,
        seq_lens,
        start_loc,
        causal_mask,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
    )
    # 逻辑块数 = ceil(max_seq_len / 128)
    logical_block_count = (max_seq_len + _MSA_INDEX_BLOCK_SIZE - 1) // _MSA_INDEX_BLOCK_SIZE
    # 把分数宽度向上取整到 16 的倍数（与算子内部填充宽度对齐），再截断
    logical_score_width = (
        (logical_block_count + _MSA_SCORE_BLOCK_ALIGNMENT - 1)
        // _MSA_SCORE_BLOCK_ALIGNMENT
        * _MSA_SCORE_BLOCK_ALIGNMENT
    )
    score = score[..., :logical_score_width]
    return _minimax_m3_index_prefill_topk(
        score,
        cu_seqlens_q,
        context_lens,
        max_query_len,
        topk,
        init_blocks,
        local_blocks,
    )


def _index_score_topk_candidates(
    score: torch.Tensor,
    context_lens: torch.Tensor,
    decode_query_len: int,
    topk: int,
    block_offset: int = 0,
    init_blocks: int = 0,
    local_blocks: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return global block IDs and scores after exactly one local TopK.

    中文注释：对（可能是 TP 分片的）分数张量做一次本地 TopK，返回全局块 id。
    这是 NPU decode 路径的 TopK 核心，等价于 Triton 版的 _mask_decode_topk_
    indices_kernel + 强制块注入的组合。

    参数：
      score: [heads, total_q, local_block_count] 本 TP rank 分片的块分数
      context_lens: [num_reqs] 上下文长度（query 之前的 token 数）
      decode_query_len: 每 request 的 query 数（>1 时为投机/MTP 多步解码）
      topk: 要选的块数
      block_offset: 本分片在全局块表中的起始块号（TP 分片用）
      init_blocks/local_blocks: 强制选择的头部队列块数 / 滑动局部块数

    算法步骤：
      1. 计算每个 (req, query_offset) 可见的全局块数（因果上界）；
      2. 未来块（不可见块）分数置 -inf，防止挤掉有效候选；
      3. init 块分数置 1e30、local 块置 1e29（local 优先级低于 init，重叠时
         local 后写覆盖 —— 保持与 Triton kernel 相同的优先级语义）；
      4. 一次 torch.topk 取前 topk；
      5. 越界/无效槽位置 -1，有效 id 加回 block_offset 变成全局 id。
    返回：(topk_indices [heads, total_q, topk] int32, topk_scores)。
    """
    block_count = score.shape[-1]
    # 构造每个 query token 的请求内偏移 [0, decode_query_len) 重复 num_reqs 次
    query_offsets = torch.arange(
        decode_query_len,
        dtype=context_lens.dtype,
        device=context_lens.device,
    ).repeat(context_lens.shape[0])
    # 可见 token 数 = context_len + query_offset + 1（+1 含当前 token 自身）
    visible_tokens = context_lens.repeat_interleave(decode_query_len) + query_offsets + 1
    # 换算成全局可见块数（ceil 除）
    global_valid_block_count = torch.div(
        visible_tokens + block_offset * _MSA_INDEX_BLOCK_SIZE + _MSA_INDEX_BLOCK_SIZE - 1,
        _MSA_INDEX_BLOCK_SIZE,
        rounding_mode="floor",
    ).clamp(min=0)
    # 本分片内有效块数 = 全局可见块数 - 分片偏移，再夹到 [0, 本分片宽度]
    valid_block_count = (global_valid_block_count - block_offset).clamp(min=0, max=block_count)
    local_block_ids = torch.arange(block_count, device=score.device)
    global_block_ids = local_block_ids + block_offset
    valid_blocks = global_block_ids[None, :] < global_valid_block_count[:, None]
    # A dense raw-score TP chunk can contain whole future speculative blocks
    # for a shorter request. Remove them before TopK so they cannot displace a
    # valid candidate and only then be discarded by output validation.
    score = torch.where(valid_blocks[None, :, :], score, float("-inf"))

    # Apply forced-block scores per token before the one and only TopK.
    if init_blocks > 0:
        init_mask = valid_blocks & (global_block_ids[None, :] < init_blocks)
        score = torch.where(init_mask[None, :, :], 1.0e30, score)
    if local_blocks > 0:
        local_start = (global_valid_block_count - local_blocks).clamp(min=0)
        local_mask = valid_blocks & (global_block_ids[None, :] >= local_start[:, None])
        score = torch.where(local_mask[None, :, :], 1.0e29, score)

    actual_topk_count = min(topk, block_count)
    # 唯一一次 TopK：init/local 的强制分数已在上一步注入，TopK 自然会选中它们
    raw_scores, raw_topk = torch.topk(
        score,
        k=actual_topk_count,
        dim=-1,
    )
    if actual_topk_count == topk:
        topk_indices = raw_topk.to(dtype=torch.int32)
        topk_scores = raw_scores
    else:
        topk_indices = torch.full(
            (*raw_topk.shape[:-1], topk),
            -1,
            dtype=torch.int32,
            device=raw_topk.device,
        )
        topk_scores = torch.full(
            (*raw_scores.shape[:-1], topk),
            float("-inf"),
            dtype=raw_scores.dtype,
            device=raw_scores.device,
        )
        topk_indices[..., :actual_topk_count].copy_(raw_topk)
        topk_scores[..., :actual_topk_count].copy_(raw_scores)

    # 有效候选判定：槽位非 -1 且未超本请求可见块数；有效则加 block_offset
    # 还原成全局块 id，无效置 -1（后续稀疏注意力 kernel 会跳过 -1）
    valid_candidate = (topk_indices >= 0) & (topk_indices < valid_block_count[None, :, None])
    topk_indices = torch.where(
        valid_candidate,
        topk_indices + block_offset,
        -1,
    )
    topk_scores = torch.where(
        valid_candidate,
        topk_scores,
        float("-inf"),
    )
    return topk_indices, topk_scores


@torch.no_grad()
def _minimax_m3_index_decode(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor | tuple[torch.Tensor],
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    context_lens: torch.Tensor,
    start_loc: torch.Tensor,
    causal_mask: torch.Tensor | None,
    *,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    decode_query_len: int,
    block_offset: int = 0,
    block_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run AscendC decode score followed by one non-fused local TopK."""
    score = _minimax_m3_index_score(
        idx_q,
        index_kv_cache,
        block_table,
        cu_seqlens_q,
        seq_lens,
        start_loc,
        causal_mask,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
    )
    if block_count is None:
        block_count = block_table.shape[-1]
    return _index_score_topk_candidates(
        score[..., :block_count],
        context_lens,
        decode_query_len,
        topk,
        block_offset,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
    )


@torch.no_grad()
def minimax_m3_index_decode(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor | tuple[torch.Tensor],
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    context_lens: torch.Tensor,
    start_loc: torch.Tensor,
    causal_mask: torch.Tensor,
    *,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    decode_query_len: int,
) -> torch.Tensor:
    """Compute AscendC decode scores and return their block TopK."""
    topk_indices, _ = _minimax_m3_index_decode(
        idx_q,
        index_kv_cache,
        block_table,
        cu_seqlens_q,
        seq_lens,
        context_lens,
        start_loc,
        causal_mask,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        decode_query_len=decode_query_len,
    )
    return topk_indices


@torch.no_grad()
def minimax_m3_index_tp_block_parallel_decode(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor | tuple[torch.Tensor],
    metadata: MiniMaxM3TPDecodeScoreMetadata,
    causal_mask: torch.Tensor,
    *,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    tp_group: Any,
) -> torch.Tensor:
    """Run packed-query scoring and TopK over TP-sharded KV blocks."""
    full_idx_q = tp_group.all_gather(idx_q.contiguous(), dim=1).contiguous()
    tp_rank = tp_group.rank_in_group
    block_offset = metadata.block_offset
    block_count = metadata.block_count
    if block_count == 0:
        # MsaIndexScore requires a non-empty block-table width. Ranks without
        # logical blocks contribute neutral candidates to the collectives.
        candidate_shape = (
            full_idx_q.shape[1],
            full_idx_q.shape[0],
            topk,
        )
        local_topk = torch.full(
            candidate_shape,
            -1,
            dtype=torch.int32,
            device=full_idx_q.device,
        )
        local_scores = torch.full(
            candidate_shape,
            float("-inf"),
            dtype=torch.float32,
            device=full_idx_q.device,
        )
    else:
        # Keep all derived tensors in the model forward. FULL_DECODE_ONLY
        # captures and replays this work, whereas tensors allocated by the
        # metadata builder would leave the graph holding stale device pointers.
        halo_blocks = (metadata.decode_query_len - 1 + metadata.block_size - 1) // metadata.block_size
        score_block_end = min(
            block_offset + block_count + halo_blocks,
            metadata.max_block_count,
        )
        score_block_table = metadata.block_table[:, block_offset:score_block_end].contiguous()
        local_context_lens = metadata.context_lens - block_offset * metadata.block_size
        chunk_capacity = block_count * metadata.block_size
        score_capacity = score_block_table.shape[-1] * metadata.block_size
        max_score_k_len = min(
            chunk_capacity + metadata.decode_query_len - 1,
            score_capacity,
        )
        score_k_lens = torch.clamp(
            metadata.context_lens + metadata.decode_query_len - block_offset * metadata.block_size,
            min=0,
            max=max_score_k_len,
        )
        score_start_loc = (
            torch.div(
                metadata.context_lens,
                metadata.block_size,
                rounding_mode="floor",
            )
            .sub(block_offset)
            .clamp(
                min=0,
                max=score_block_table.shape[-1] - 1,
            )
        )
        local_topk, local_scores = _minimax_m3_index_decode(
            full_idx_q,
            index_kv_cache,
            score_block_table,
            metadata.cu_seqlens_q,
            score_k_lens,
            local_context_lens,
            score_start_loc,
            None if metadata.decode_query_len == 1 else causal_mask,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            decode_query_len=metadata.decode_query_len,
            block_offset=block_offset,
            block_count=block_count,
        )

    gathered_scores = tp_group.all_gather(local_scores.contiguous(), dim=-1)
    gathered_topk = tp_group.all_gather(local_topk.contiguous(), dim=-1)

    local_head_count = idx_q.shape[1]
    local_head_start = tp_rank * local_head_count
    local_gathered_scores = gathered_scores.narrow(0, local_head_start, local_head_count)
    _, merged_pos = torch.topk(
        local_gathered_scores,
        k=topk,
        dim=-1,
    )
    local_gathered_topk = gathered_topk.narrow(0, local_head_start, local_head_count)
    return torch.gather(local_gathered_topk, dim=-1, index=merged_pos)


@torch.no_grad()
def _minimax_m3_sparse_attn_a3(
    q: torch.Tensor,
    kv_cache: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    block_size: int,
    *,
    supports_fp8: bool = False,
) -> None:
    key, value = _split_main_kv_cache(kv_cache)
    # Q-gather-KV is also the A5 fallback when experimental Split-KV is
    # unavailable. Match decode's unscaled E4M3 inputs and BF16 output.
    op_kwargs: dict[str, Any] = {}
    if key.dtype == torch.float8_e4m3fn:
        if not supports_fp8:
            raise TypeError("MiniMax-M3 FP8 sparse attention is not supported on this device")
        if value.dtype != torch.float8_e4m3fn:
            raise TypeError("MiniMax-M3 FP8 sparse attention requires both K and V caches in E4M3")
        q = _to_fp8_e4m3(q)
        dequant_scale = torch.ones((1, 1, 1, 1), dtype=torch.float32, device=q.device)
        op_kwargs = {
            "q_dequant_scale": dequant_scale,
            "k_dequant_scale": dequant_scale,
            "v_dequant_scale": dequant_scale,
            "attention_out_dtype": torch.bfloat16,
        }
    q_lens_t = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    out = torch.ops._C_ascend.npu_sparse_attention_score(
        q,
        key,
        value,
        topk_idx,
        block_table,
        select_num_idx=_select_num_idx_from_topk(topk_idx),
        actual_seq_lengths=q_lens_t,
        actual_seq_lengths_kv=seq_lens,
        num_key_value_heads=num_kv_heads,
        scale_value=sm_scale,
        block_size=block_size,
        top_k=topk_idx.shape[-1],
        inner_precise=_SPARSE_ATTN_INNER_PRECISE,
        **op_kwargs,
    )
    output.copy_(out)


def _minimax_m3_sparse_attn_kv_gather_q(
    q: torch.Tensor,
    kv_cache: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    block_size: int,
    total_kv_blocks: int,
    max_kv_blocks: int,
    *,
    supports_fp8: bool,
) -> None:
    key, value = _split_main_kv_cache(kv_cache)

    # A3 uses FP32 scores and partial outputs. Keep A5's existing BF16
    # precision mode; its FP8 cache path overrides this below as before.
    inner_precise = _PREFILL_KV_GATHER_Q_INNER_PRECISE if supports_fp8 else _A3_PREFILL_KV_GATHER_Q_INNER_PRECISE
    if key.dtype == torch.float8_e4m3fn:
        if not supports_fp8:
            raise TypeError("MiniMax-M3 FP8 sparse attention is not supported on this device")
        if value.dtype != torch.float8_e4m3fn:
            raise TypeError("MiniMax-M3 FP8 sparse attention requires both K and V caches in E4M3")
        q = _to_fp8_e4m3(q)
        inner_precise = _SPARSE_ATTN_INNER_PRECISE

    cu_block_lens = _build_cu_block_lens(seq_lens, block_size)
    # Keep the -1 sentinel: K2Q drops invalid entries, while replacing them
    # with zero would add duplicate attention edges for logical KV block 0.
    k2q_row_ptr, k2q_q_indices, k2q_slot_indices = _npu_k2q_csr(
        topk_idx,
        cu_seqlens_q,
        cu_block_lens,
        order_method=1,
        total_rows=total_kv_blocks,
        max_kv=max_kv_blocks,
        use_simt=0,
        q_global_offset=True,
    )

    k2q_row_ptr = k2q_row_ptr.to(dtype=torch.int32).contiguous()
    k2q_q_indices = k2q_q_indices.to(dtype=torch.int32).contiguous()
    k2q_slot_indices = k2q_slot_indices.to(dtype=torch.int32).contiguous()
    q_lens_t = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int32).contiguous()
    kv_lens_t = seq_lens.to(torch.int32).contiguous()
    out = torch.ops._C_ascend.npu_sparse_attention_score_prefill(
        q,
        key,
        value,
        block_table,
        k2q_row_ptr,
        k2q_q_indices,
        k2q_slot_indices,
        num_kv_heads,
        sm_scale,
        block_size,
        topk_idx.shape[-1],
        inner_precise,
        actual_seq_lengths=q_lens_t,
        actual_seq_lengths_kv=kv_lens_t,
    )
    output.copy_(out)


@lru_cache
def _is_minimax_sparse_attention_split_kv_available() -> bool:
    """Check compatible vendor entry points once per worker process.

    Restart workers after installing the experimental package and sourcing its
    environment; the ACLNN loader also caches its library search paths.
    """
    import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: F401, PLC0415

    enable_custom_op()
    return torch.ops._C_ascend.is_minimax_sparse_attention_split_kv_available()


@torch.no_grad()
def minimax_m3_sparse_attn(
    q: torch.Tensor,
    kv_cache: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    block_size: int = 128,
    total_kv_blocks: int = -1,
    max_kv_blocks: int = -1,
) -> None:
    del prefix_lens, max_query_len
    common_args = (
        q,
        kv_cache,
        topk_idx,
        block_table,
        cu_seqlens_q,
        seq_lens,
        num_kv_heads,
        sm_scale,
        output,
        block_size,
    )
    hardware_profile = get_current_hardware_profile()
    supports_fp8 = hardware_profile.supports(HardwareCapability.FP8_ATTENTION)
    # Select the optional optimization by ACLNN availability. The installed
    # experimental package must provide kernels for the current device.
    if not _is_minimax_sparse_attention_split_kv_available():
        _minimax_m3_sparse_attn_a3(*common_args, supports_fp8=supports_fp8)
        return

    _minimax_m3_sparse_attn_kv_gather_q(
        *common_args,
        total_kv_blocks,
        max_kv_blocks,
        supports_fp8=supports_fp8,
    )


@torch.no_grad()
def minimax_m3_sparse_attn_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    decode_query_len: int,
    block_size: int = 128,
    *,
    select_num_idx: torch.Tensor | None = None,
    dequant_scale: torch.Tensor | None = None,
) -> None:
    """Run sparse decode through the AscendC sparse-attention operator."""
    if q.shape[0] != seq_lens.shape[0] * decode_query_len:
        raise ValueError("Decode query tokens must equal request count times decode_query_len")

    key, value = _split_main_kv_cache(kv_cache)
    query_lens = torch.full_like(seq_lens, decode_query_len, dtype=torch.int32)
    if select_num_idx is None:
        select_num_idx = _select_num_idx_from_topk(topk_idx)

    op_kwargs: dict[str, Any] = {}
    if key.dtype == torch.float8_e4m3fn:
        if value.dtype != torch.float8_e4m3fn:
            raise TypeError("MiniMax-M3 FP8 sparse attention requires both K and V caches in E4M3")
        q = _to_fp8_e4m3(q)
        if dequant_scale is None:
            dequant_scale = torch.ones((1, 1, 1, 1), dtype=torch.float32, device=q.device)
        op_kwargs = {
            "q_dequant_scale": dequant_scale,
            "k_dequant_scale": dequant_scale,
            "v_dequant_scale": dequant_scale,
            "attention_out_dtype": torch.bfloat16,
        }
    out = torch.ops._C_ascend.npu_sparse_attention_score(
        q,
        key,
        value,
        topk_idx,
        block_table,
        select_num_idx=select_num_idx,
        actual_seq_lengths=query_lens,
        actual_seq_lengths_kv=seq_lens,
        num_key_value_heads=num_kv_heads,
        scale_value=sm_scale,
        block_size=block_size,
        top_k=topk_idx.shape[-1],
        inner_precise=_SPARSE_ATTN_INNER_PRECISE,
        **op_kwargs,
    )
    output.copy_(out)
