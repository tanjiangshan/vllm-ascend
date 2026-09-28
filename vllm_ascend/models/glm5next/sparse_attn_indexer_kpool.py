# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor orchestration for the GLM-Next Triton KPool indexer.

GLM-Next KPool 索引器的张量编排层。

职责（介于注意力后端与 Triton/AscendC 内核之间）：
  1. SparseAttnIndexerKpool.forward：
     - 调用 glm5_next_kpool_tail_compress_and_write_cache_triton 完成尾部
       环压缩写缓存（把 FP32 的 K/gate 环中凑满一池的数据经 APE+gate
       加权池化，写入 BF16 压缩缓存）；
     - 可选调用 glm5_next_lightning_indexer_triton 做 top-k 选择
       （查询与压缩 K 相关性打分，选池粒度的 top-k 上下文）；
     - append_causal_tail 把未满池的尾部 token 追加进索引，保证 CANN
       SFA（稀疏注意力）内核要求的"连续有效前缀"。
  2. 缓存绑定与前向上下文查找属于模型侧后端；本类只接收显式张量与
     元数据，便于独立于 vLLM 注意力包装器做单元测试。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm_ascend.ops.triton.glm5_next_kpool_tail_compress import (  # type: ignore[import-untyped]
    glm5_next_kpool_tail_compress_and_write_cache_triton,
)
from vllm_ascend.ops.triton.glm5_next_lightning_indexer import (  # type: ignore[import-untyped]
    glm5_next_lightning_indexer_triton,
)

# 语法点：if TYPE_CHECKING 块内的 import 仅类型检查期生效，运行时不导入
# （避免循环导入）；配合 from __future__ import annotations 的惰性注解。
if TYPE_CHECKING:
    from vllm_ascend.attention.indexer_kpool import (
        AscendIndexerKPoolMetadata,
        AscendIndexerKPoolTailMetadata,
    )


def append_causal_tail(
    indices: torch.Tensor,
    positions: torch.Tensor,
    topk_tokens: int,
    pool_size: int,
) -> None:
    """Append unpooled tokens to the valid prefix required by CANN SFA.

    把未池化的尾部 token 追加到索引的有效前缀（CANN SFA 要求）。

    参数：
        indices: [num_tokens, topk_width] top-k 索引（原地修改）。
        positions: [num_tokens] token 位置。
        topk_tokens: top-k token 数（池粒度展开前的 token 计数）。
        pool_size: KPool 池大小 kpool。

    原理：CANN 的稀疏注意力（SFA）要求索引行是"连续有效前缀 + 填充"。
    压缩池放在行首，但"当前未满池"的 token（0..kpool-1 个）不在压缩
    缓存里，需要追加在 top-k 列之后形成连续区。短序列历史不足
    topk_tokens 时，把尾部放在 min(tail_start, topk) 处，避免中间
    出现 SFA 会跳过的无效洞。
    """
    tail_width = pool_size - 1
    if tail_width == 0:
        # 池大小为 1 时无"尾部"概念，直接返回。
        return
    # 步骤1: 计算每个 token 所在池的起点（floor((pos+1)/pool)*pool）。
    positions = positions.to(torch.int64)
    tail_start = torch.div(positions + 1, pool_size, rounding_mode="floor") * pool_size
    # 步骤2: 候选尾部 token = 池起点 + 0..pool-1 偏移；
    # 超出当前位置的候选置 -1（尚未生成的 token）。
    tail_cols = torch.arange(tail_width, device=indices.device, dtype=torch.int64)
    tail_tokens = tail_start.unsqueeze(1) + tail_cols
    tail_values = torch.where(
        tail_cols < (positions + 1 - tail_start).unsqueeze(1),
        tail_tokens,
        -1,
    ).to(indices.dtype)
    # PKI packs complete pools at the front. Short requests have fewer than
    # topk_tokens history entries; placing the tail at that fixed column would
    # leave invalid holes, and SFA would skip the unpooled tokens.
    # 步骤3: PKI（lightning indexer）把完整池打包在行首。短序列历史不足
    # topk_tokens，若把尾部放在固定列会产生无效洞（SFA 会跳过未池化
    # token）——因此先把 topk 之后的所有列清为 -1，再把尾部 scatter 到
    # min(tail_start, topk) 起始的位置，保证连续。
    indices[:, topk_tokens:] = -1
    tail_offsets = tail_start.clamp(max=topk_tokens).unsqueeze(1) + tail_cols
    # scatter_(dim, index, src)：按索引原地散布尾部值。
    indices.scatter_(1, tail_offsets, tail_values)


class SparseAttnIndexerKpool(nn.Module):
    """Update KPool caches and optionally select sparse token indices.

    KPool 索引器核心：更新 KPool 缓存并（可选）选出稀疏 token 索引。

    Cache binding and forward-context lookup belong to the model-side backend.
    This helper receives explicit tensors and typed metadata so the cache update
    can be tested independently from the vLLM attention wrapper.

    张量流：
      k/gate_score (FP32) + tail_cache (FP32 环)
        -> tail_compress 内核：满池数据经 APE+gate 加权池化
        -> indexer_cache (BF16 压缩池)
      q_values + weights -> lightning_indexer 内核
        -> indices [num_tokens, 1, topk+kpool-1]（token 粒度 top-k）
        -> append_causal_tail 追加未满池尾部
        -> 输出给稀疏 MLA（SFA）作为注意力键值选择
    """

    def __init__(self, topk_tokens: int, head_dim: int) -> None:
        """初始化。

        参数：
            topk_tokens: top-k 的 token 数（必须被池大小整除）。
            head_dim: 索引器头维度。
        """
        super().__init__()
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim

    def forward(
        self,
        k: torch.Tensor,
        q_values: torch.Tensor | None,
        weights: torch.Tensor | None,
        positions: torch.Tensor,
        indexer_cache: torch.Tensor,
        tail_cache: torch.Tensor,
        indexer_metadata: AscendIndexerKPoolMetadata,
        tail_metadata: AscendIndexerKPoolTailMetadata,
        *,
        gate_score: torch.Tensor,
        compress_ape: torch.Tensor,
        index_kpool: int,
        max_pool_seq_len: int,
        compute_topk: bool,
    ) -> torch.Tensor | None:
        """缓存更新 + 可选 top-k 选择。

        参数（语法点：* 之后 keyword-only）：
            k: [num_tokens, head_dim] FP32 原始索引 K。
            q_values: [num_tokens, n_head*rope_dim] 查询（compute_topk 时必需）。
            weights: [n_head] 每头权重（compute_topk 时必需）。
            positions: [num_tokens] token 位置。
            indexer_cache: BF16 压缩池缓存（内核写入）。
            tail_cache: [blocks, 2, capacity, head_dim] FP32 K/gate 环。
            indexer_metadata: 压缩索引器元数据（cum_query_lens 等）。
            tail_metadata: 尾部环元数据（slot_mapping/block_table）。
            gate_score: [num_tokens, head_dim] FP32 门控分数。
            compress_ape: [pool_size, head_dim] FP32 APE 参数。
            index_kpool: 池大小。
            max_pool_seq_len: 池粒度最大序列长度（内核平铺用）。
            compute_topk: 是否执行 top-k 选择（False 只更新缓存）。

        返回：
            compute_topk=False: None。
            compute_topk=True: [num_tokens, 1, topk+kpool-1] int32 索引；
            填充行（padding）为 -1。
        """
        num_tokens = k.shape[0]
        # 步骤1: 参数与张量形状/类型校验。
        if index_kpool <= 0 or self.topk_tokens % index_kpool:
            raise ValueError("KPool top-k must be divisible by its positive pool size.")
        if num_tokens == 0:
            # 空批：无缓存更新；top-k 返回空索引张量。
            return (
                None
                if not compute_topk
                else torch.empty((0, 1, self.topk_tokens + index_kpool - 1), dtype=torch.int32, device=k.device)
            )
        if indexer_metadata.cum_query_lens is None or indexer_metadata.raw_seq_lens is None:
            raise ValueError("GLM KPool metadata requires cum_query_lens and raw_seq_lens.")
        if indexer_cache.dtype != torch.bfloat16:
            raise TypeError("GLM KPool compressed cache must be bfloat16.")
        if tail_cache.dtype != torch.float32 or k.dtype != torch.float32 or gate_score.dtype != torch.float32:
            raise TypeError("GLM KPool keys, gates and compressor tail must be float32.")
        if (
            tail_cache.ndim != 4
            or tail_cache.shape[1] != 2
            or tail_cache.shape[2] < index_kpool
            or tail_cache.shape[3] != self.head_dim
            or gate_score.shape != k.shape
        ):
            raise ValueError("GLM KPool tail requires [blocks, 2, capacity, head_dim] K/gate storage.")
        if tail_metadata.block_size != tail_cache.shape[2]:
            raise ValueError("GLM KPool tail metadata capacity must match the bound cache.")
        if compress_ape.shape != (index_kpool, self.head_dim) or compress_ape.dtype != torch.float32:
            raise ValueError("GLM KPool APE must be FP32 with shape [pool_size, head_dim].")

        # 步骤2: 尾部压缩写缓存——把环中凑满一池的 K/gate 经 APE+gate
        # 加权池化，写入 BF16 压缩缓存（原地更新两个缓存）。
        glm5_next_kpool_tail_compress_and_write_cache_triton(
            tail_cache,
            indexer_cache,
            k,
            gate_score,
            compress_ape,
            positions,
            indexer_metadata.cum_query_lens,
            indexer_metadata.raw_seq_lens,
            tail_metadata.slot_mapping[:num_tokens],
            tail_metadata.block_table,
            indexer_metadata.slot_mapping[:num_tokens],
            index_kpool,
        )
        # Sharing top-k still advances the compressed cache and raw tail.
        # 步骤3: top-k 复用模式（MTP 步 1+）仍需推进压缩缓存与原始尾部，
        # 因此到此即可返回。
        if not compute_topk:
            return None
        if q_values is None or weights is None:
            raise ValueError("GLM KPool top-k requires query and head weights.")
        # 步骤4: lightning indexer——查询与压缩 K 相关性打分 + 池粒度 top-k
        # 选择，输出 token 粒度索引 [num_tokens, 1, topk]。
        indices = glm5_next_lightning_indexer_triton(
            q_values,
            indexer_cache,
            weights.to(q_values.dtype),
            indexer_metadata.cum_query_lens,
            indexer_metadata.seq_lens,
            indexer_metadata.block_table,
            positions,
            index_topk=self.topk_tokens,
            index_kpool=index_kpool,
            max_pool_seq_len=max_pool_seq_len,
        )
        # A2/A3 SFA requires a contiguous valid prefix; the reference indexer
        # puts the running tail at the fixed top-k column for short requests.
        # 步骤5: 追加未满池尾部（保证 CANN SFA 的连续有效前缀要求）；
        # 短序列时参考实现把运行尾部放在固定 top-k 列处。
        append_causal_tail(indices[:, 0], positions, self.topk_tokens, index_kpool)
        # 步骤6: 填充行（超过 cum_query_lens[-1] 的 padding token）置 -1。
        valid = torch.arange(num_tokens, device=k.device) < indexer_metadata.cum_query_lens[-1]
        indices.masked_fill_(~valid[:, None, None], -1)
        return indices
