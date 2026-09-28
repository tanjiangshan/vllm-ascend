# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# 【文件职责】上游 Engram n-gram 哈希状态的 Ascend KV-slot 适配层。
#
# 【n-gram 哈希原理】对每个 token 位置 t、每个 engram 层、每个"头"计算
# 多阶 n-gram 哈希：以 t 为终点回看 shift=0..MAX_NGRAM-1 个 token 组成
# 滚动异或多项式（每阶一个乘法因子 multiplier），再对每头取
# hash % prime + offset 落桶。哈希值即嵌入表行号。历史 token 来源三级
# 回退：当前 chunk 内 → runner 的 lookback 窗口 → SWA slot cache。
#
# 【NPU 适配点】
# - 上游把"请求二分搜索"与"哈希主体"放在同一个 Triton kernel；Triton-
#   Ascend（bishengir 后端）编译该合并 kernel 会崩溃（LLVM buffer released），
#   故拆成 _engram_req_index_kernel + _hash_ids_kernel 两个 kernel；
# - SWA cache 的 kv_cache 是单元素列表且分配前为空，适配器 AscendEngram-
#   SlotCache 把它包装成上游 NgramHashState 期望的"单张量"视图。
# =============================================================================
"""Ascend KV-slot adapter for the upstream Engram hash state.

Upstream ``NgramHashState`` reads exactly two things from its
``swa_cache_module``: ``block_size`` once at construction, and ``kv_cache`` on
every ``ensure_cache()`` to size (and re-size) the slot-keyed history. The
Ascend SWA cache differs in ways the adapter absorbs:

* the storage block size comes from the Ascend block-size table
  (``DSV4_BLOCK_SIZES``) and is not the logical block size the scheduler uses;
* ``kv_cache`` is a one-element list and is absent until the KV allocator binds
  it, so ``numel()``/``shape[0]`` must not be called on the raw attribute.

The subclass retains upstream history allocation and lifecycle. Its hash
launch is split into request search and hashing for the Ascend compiler.
"""

import torch

# Upstream #56741 normalized the V4.1 model package name.
# 【中文】从上游 vLLM 的 V4.1 公共包导入: DEAD_ID（死哨兵 ID）、EngramLayout
# （桶布局）、NgramHashState（哈希状态基类）、哈希 cache 写入 kernel。
from vllm.models.deepseek_v41.common.engram import (
    DEAD_ID,
    EngramLayout,
    NgramHashState,
    _write_hash_cache_kernel,
)
from vllm.triton_utils import tl, triton


class AscendEngramSlotCache:
    """``swa_cache_module`` view for upstream ``NgramHashState``."""
    """【中文说明】把 Ascend SWA cache 层包装成上游 NgramHashState 期望的
    "swa_cache_module"视图：构造时读一次 block_size，之后每次 ensure_cache()
    读 kv_cache（单张量形态）。"""

    def __init__(self, swa_cache_layer) -> None:
        """参数: swa_cache_layer 为 Ascend SWA cache 层（dsa_v41.py）。"""
        self._layer = swa_cache_layer
        # 物理存储 block 大小（来自 DSV4_BLOCK_SIZES 表，非调度器逻辑值）。
        self.block_size = int(swa_cache_layer.block_size)

    @property
    def kv_cache(self) -> torch.Tensor:
        """The SWA KV cache as a single tensor, or an empty one when unbound."""
        """【中文说明】把（可能尚未绑定的）SWA cache 返回成单张量视图。
        Ascend 的 kv_cache 是单元素列表——逐层解包；未绑定（显存探测阶段）
        或多平面形态时返回空 INT32 张量，避免上游对原始属性调 numel() 崩溃。
        语法点: @property 把方法暴露为只读属性。"""
        cache = getattr(self._layer, "kv_cache", None)
        # while 循环解包所有单元素嵌套（list/tuple）层级。
        while isinstance(cache, (list, tuple)) and len(cache) == 1:
            cache = cache[0]
        if cache is None or isinstance(cache, (list, tuple)):
            # Unbound (profiling pass) or a multi-plane cache we do not index.
            # 【中文】未绑定（探测前向）或我们不索引的多平面 cache → 空张量。
            return torch.empty(0, dtype=torch.int32)
        return cache


def engram_dead_mask(
    token_ids: torch.Tensor,
    image_token_id: int,
    image_pad_token_id: int,
) -> torch.Tensor:
    """Positions that must not take part in an n-gram.

    Covers both image sentinels, for the current chunk and for the lookback
    window alike. A ``-1`` lookback padding entry is not an image: callers pass
    the window through the same sentinel check and rely on the negative value
    staying a non-match.
    """
    """【中文说明】不得参与 n-gram 的"死"位置掩码：覆盖两种图像哨兵（image
    token 与 pad token），当前 chunk 与回看窗口同样适用。-1 填充不是图像
    哨兵——调用方让窗口过同样的哨兵检查，负数值天然不匹配任何哨兵。"""
    return (token_ids == image_token_id) | (token_ids == image_pad_token_id)


def create_engram_hash_state(vllm_config, config, swa_cache_layer) -> NgramHashState:
    """Bind the upstream hash state to the Ascend SWA cache."""
    """【中文说明】构造绑定 Ascend SWA cache 的哈希状态。
    步骤: EngramLayout.from_config 解析桶布局（每层每头的素数/偏移表）→
    断言至少一个 engram 层 → 返回 AscendNgramHashState 实例。"""
    layout = EngramLayout.from_config(config)
    assert layout is not None, "Engram hash state needs at least one Engram layer"
    return AscendNgramHashState(vllm_config, layout, AscendEngramSlotCache(swa_cache_layer))


# 语法点: @triton.jit 把 Python 函数编译为 Triton kernel；do_not_specialize
# 列出的参数不按值特化（避免每换个 num_tokens 就重新编译一份 kernel）。
@triton.jit(do_not_specialize=["num_tokens", "num_query_rows"])
def _engram_req_index_kernel(
    query_start_loc,
    req_ids,
    num_tokens,
    num_query_rows,
    query_stride,
    BLOCK_T: tl.constexpr,
):
    """Map each token to the request row whose chunk contains it.

    Split out of ``_hash_ids_kernel``: the Ascend bishengir backend aborts on
    the per-token search when it shares a kernel with the hash body
    ("LLVM ERROR: The buffer memory has been released"), while both halves
    compile on their own. Hoisting also runs the search once per token instead
    of once per (token, layer) program.
    """
    # 【中文】token → 请求行号二分搜索 kernel。每个 program 处理 BLOCK_T 个
    # token，在 query_start_loc（前缀和）里二分查找其所属请求。
    # 拆分原因: bishengir 后端无法把"搜索+哈希"编进同一 kernel（见 docstring）；
    # 拆出后搜索也只需每 token 一次而非每 (token, layer) 一次。
    token = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    valid = token < num_tokens
    # 经典二分: lo/hi 向量对各 token 独立收敛到所属请求行。
    lo = tl.full((BLOCK_T,), 0, tl.int32)
    hi = tl.full((BLOCK_T,), num_query_rows, tl.int32)
    # 语法点: while + tl.sum——还有未收敛的 token 就继续（向量化二分）。
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        mid = (lo + hi) // 2
        end = tl.load(query_start_loc + (mid + 1) * query_stride, lo < hi, other=0)
        right = token >= end
        active = lo < hi
        lo = tl.where(active & right, mid + 1, lo)
        hi = tl.where(active & ~right, mid, hi)
    # 收敛后钳到合法区间写入 req_ids。
    tl.store(req_ids + token, tl.minimum(lo, num_query_rows - 1), mask=valid)


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "num_slots",
        "num_table_rows",
        "max_blocks",
    ]
)
def _hash_ids_kernel(
    input_ids,
    token_map,
    dead_mask,
    positions,
    block_table,
    query_start_loc,
    req_ids,
    multipliers,
    primes,
    offsets,
    cache,
    lookback_token_ids,
    lookback_dead_mask,
    output,
    num_tokens,
    num_slots,
    pad_id,
    input_stride,
    mask_stride,
    position_stride,
    table_stride,
    table_col_stride,
    query_stride,
    num_table_rows,
    max_blocks,
    cache_block_size,
    MAX_NGRAM: tl.constexpr,
    num_heads,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    dead_id,
    lookback_depth,
    lookback_row_stride,
    lookback_col_stride,
    lookback_mask_row_stride,
    lookback_mask_col_stride,
):
    # 【中文】n-gram 哈希主体 kernel。网格: (token 分块, 层数)。
    # 每个 program 处理 BLOCK_T 个 token × 一个 engram 层的全部头。
    # 滚动哈希: rolling ^= source * multiplier（每阶一个乘子），随后对每头
    # 取 rolling % prime + offset 落桶写入 output。
    token = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    layer = tl.program_id(1)
    num_layers = tl.num_programs(1)
    valid = token < num_tokens
    # The request search lives in _engram_req_index_kernel (see its docstring).
    # 【中文】请求行号由前置 kernel 算好，这里直接读。
    req = tl.load(req_ids + token, valid, other=0).to(tl.int64)
    # 该请求 chunk 的起始 token 下标（query_start_loc[req]），钳到合法范围。
    chunk_idx = tl.load(query_start_loc + req * query_stride)
    chunk_idx = tl.minimum(chunk_idx, num_tokens - 1).to(tl.int64)
    # chunk 起始的绝对位置（n-gram 不得跨越 chunk 边界向回看）。
    chunk_start = tl.load(positions + chunk_idx * position_stride)
    position = tl.load(positions + token * position_stride, valid, other=0).to(tl.int64)
    head = tl.arange(0, BLOCK_H)
    # blocked: 该 token 的 n-gram 已"断裂"（越界起点或死 token）。
    blocked = tl.full((BLOCK_T,), False, tl.int1)
    # rolling: 多项式滚动哈希累积值（int64 防溢出）。
    rolling = tl.full((BLOCK_T,), 0, tl.int64)
    # 语法点: tl.static_range 编译期展开循环（MAX_NGRAM 是 constexpr）。
    for shift in tl.static_range(MAX_NGRAM):
        lookback = position - shift
        # ---- 来源1: 当前 chunk 内（lookback >= chunk_start）----
        in_batch = lookback >= chunk_start
        batch_idx = tl.maximum(token - shift, 0)
        batch_token = tl.load(input_ids + batch_idx * input_stride, valid & in_batch, other=0)
        # token_map: 原始 token ID → 压缩后的哈希词表 ID（上游构建）。
        batch_source = tl.load(token_map + batch_token, valid & in_batch, other=0)
        batch_dead = tl.load(dead_mask + batch_idx * mask_stride, valid & in_batch, other=False)
        # 死 token（图像哨兵）替换为 dead_id，保证 n-gram 不跨图像区。
        batch_source = tl.where(batch_dead, dead_id, batch_source)

        # ---- 来源2: runner 回看窗口（chunk 之前的 prompt 历史）----
        # 列号 = chunk_start - 1 - lookback（窗口内从近到远编列）。
        col = chunk_start - 1 - lookback
        in_window = valid & ~in_batch & (col >= 0) & (col < lookback_depth)
        col = tl.minimum(tl.maximum(col, 0), lookback_depth - 1)
        window_token = tl.load(
            lookback_token_ids + req * lookback_row_stride + col * lookback_col_stride,
            in_window,
            other=-1,
        )
        # 已知 = 窗口命中有效 token（>=0，-1 是填充）。
        known = in_window & (window_token >= 0)
        window_source = tl.load(token_map + window_token, known, other=0)
        window_dead = tl.load(
            lookback_dead_mask + req * lookback_mask_row_stride + col * lookback_mask_col_stride,
            known,
            other=False,
        )
        window_source = tl.where(window_dead, dead_id, window_source)

        # ---- 来源3: SWA slot cache（更早的历史，从 cache 逐槽读取）----
        if cache is not None:
            # 位置钳到 cache 覆盖范围，经 block_table 换算物理槽位。
            clamped = tl.minimum(tl.maximum(lookback, 0), max_blocks * cache_block_size - 1)
            block_row = tl.minimum(req, num_table_rows - 1)
            # 只有"chunk 外且窗口未命中"的位置才需要 cache 兜底。
            needs_cache = valid & ~in_batch & ~known
            block = tl.load(
                block_table + block_row * table_stride + (clamped // cache_block_size) * table_col_stride,
                needs_cache,
                other=0,
            ).to(tl.int64)
            slot = tl.minimum(
                tl.maximum(block * cache_block_size + clamped % cache_block_size, 0),
                num_slots - 1,
            )
            fallback = tl.load(cache + slot, needs_cache, other=0)
        else:
            # 无 cache（如 PP 前几级）: 用 pad_id 占位。
            fallback = tl.full((BLOCK_T,), pad_id, tl.int32)
        # 三级来源按优先级选择: chunk 内 > 窗口已知 > cache/pad 兜底。
        source = tl.where(in_batch, batch_source, tl.where(known, window_source, fallback)).to(tl.int64)
        # 位置越界或死 token → n-gram 从此阶起永久断裂。
        blocked |= (lookback < 0) | (source == dead_id)
        value = tl.where(blocked, pad_id, source)
        # 本阶乘子混入滚动哈希（异或累积）。
        multiplier = tl.load(multipliers + layer * MAX_NGRAM + shift)
        rolling ^= value * multiplier
        if shift > 0:
            # shift>=1 起对每头输出落桶哈希: rolling % prime + offset。
            # 输出布局: [token, layer, (MAX_NGRAM-1)*num_heads]。
            col = (shift - 1) * num_heads + head
            param_offset = layer * (MAX_NGRAM - 1) * num_heads + col
            prime = tl.load(primes + param_offset, head < num_heads, other=1)
            offset = tl.load(offsets + param_offset, head < num_heads, other=0)
            hashed = rolling[:, None] % prime[None, :] + offset[None, :]
            out_offset = (token.to(tl.int64) * num_layers + layer)[:, None] * ((MAX_NGRAM - 1) * num_heads) + col[
                None, :
            ]
            tl.store(output + out_offset, hashed, valid[:, None] & (head < num_heads))


class AscendNgramHashState(NgramHashState):
    """Reuse upstream history lifecycle, splitting only the NPU hash launch.

    Triton-Ascend cannot compile the combined request-search/hash kernel.
    Keep that compiler workaround in this subclass, without changing vLLM.
    """
    """【中文说明】继承上游 NgramHashState（复用其历史分配与生命周期管理），
    仅重写哈希发射路径：Triton-Ascend 编译不了"搜索+哈希"合并 kernel，
    该编译器 workaround 收敛在本子类，不改动 vLLM 上游。"""

    def dummy_hashes(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Participate in DP lookups without valid rows or hash-cache updates."""
        """【中文说明】DP 集合查表的"空参与"路径: 返回全 DEAD_ID 的哈希与全
        False 的 keep 掩码——不产生有效行、不更新历史，但保证其他副本的
        all_gather 不会因本 rank 缺席而挂起。"""
        num_tokens = input_ids.shape[0]
        num_layers, max_ngram = self.multipliers.shape
        num_heads = self.primes.shape[-1]
        hashes = input_ids.new_full(
            (num_tokens, num_layers, (max_ngram - 1) * num_heads),
            DEAD_ID,
            dtype=torch.int32,
        )
        keep = torch.zeros(num_tokens, dtype=torch.bool, device=input_ids.device)
        return hashes, keep

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        dead_mask: torch.Tensor,
        lookback_token_ids: torch.Tensor,
        lookback_dead_mask: torch.Tensor,
        slot_mapping: torch.Tensor | None,
        block_table: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute [tokens, layers, hash columns] int32 n-gram hashes.

        History comes from the current chunk, then the runner's lookback
        window, then the optional V1 slot cache.
        """
        """【中文说明】计算 [tokens, layers, (max_ngram-1)*num_heads] 的 INT32
        n-gram 哈希。步骤:
            1) 先把本 chunk 的 token 写进哈希 slot cache（供未来 chunk 回看）；
            2) 启动请求二分搜索 kernel（每 token 找到所属请求行）；
            3) 启动哈希主体 kernel（网格 = token 分块 × 层数）。
        参数: input_ids/positions 为本 chunk 数据; dead_mask 标记图像哨兵;
            lookback_* 为 runner 提供的窗口与死掩码; slot_mapping/block_table
            用于读写 SWA slot cache。"""
        cache = self._cache if self.use_slot_cache else None
        num_tokens = input_ids.shape[0]
        num_layers, max_ngram = self.multipliers.shape
        num_heads = self.primes.shape[-1]
        output = input_ids.new_empty((num_tokens, num_layers, (max_ngram - 1) * num_heads), dtype=torch.int32)
        if num_tokens == 0:
            return output
        # Ascend needs the request search out of the hash body (E3); one launch
        # still covers every layer.
        # 【中文】步骤2: 请求搜索独立成 kernel（编译器限制），一次启动覆盖
        # 全部 token；哈希主体再按层并行。
        req_ids = input_ids.new_empty(num_tokens, dtype=torch.int32)
        _engram_req_index_kernel[(triton.cdiv(num_tokens, 32),)](
            query_start_loc,
            req_ids,
            num_tokens,
            query_start_loc.numel() - 1,
            query_start_loc.stride(0),
            BLOCK_T=32,
        )
        if self.use_slot_cache:
            assert cache is not None and slot_mapping is not None
            assert block_table is not None
            # Finish writes before other thread blocks read fallback history.
            # 【中文】步骤1: 先写后读——本 chunk 的 token 先落入 slot cache，
            # 哈希 kernel 读 fallback 历史时才能看到刚写入的行（上游 kernel）。
            _write_hash_cache_kernel[(triton.cdiv(num_tokens, 256),)](
                input_ids,
                self.token_map,
                dead_mask,
                slot_mapping,
                cache,
                num_tokens,
                input_ids.stride(0),
                dead_mask.stride(0),
                slot_mapping.stride(0),
                256,
                DEAD_ID,
            )
        # 步骤3: 哈希主体。do_not_specialize 的参数逐个按名传入。
        _hash_ids_kernel[(triton.cdiv(num_tokens, 32), num_layers)](
            input_ids,
            self.token_map,
            dead_mask,
            positions,
            block_table,
            query_start_loc,
            req_ids,
            self.multipliers,
            self.primes,
            self.offsets,
            cache,
            lookback_token_ids,
            lookback_dead_mask,
            output,
            num_tokens,
            cache.shape[0] if cache is not None else 0,
            self.pad_id,
            input_stride=input_ids.stride(0),
            mask_stride=dead_mask.stride(0),
            position_stride=positions.stride(0),
            table_stride=block_table.stride(0) if block_table is not None else 0,
            table_col_stride=block_table.stride(1) if block_table is not None else 0,
            query_stride=query_start_loc.stride(0),
            num_table_rows=block_table.shape[0] if block_table is not None else 0,
            max_blocks=block_table.shape[1] if block_table is not None else 0,
            cache_block_size=self.block_size,
            MAX_NGRAM=max_ngram,
            num_heads=num_heads,
            BLOCK_T=32,
            BLOCK_H=triton.next_power_of_2(num_heads),
            dead_id=DEAD_ID,
            lookback_depth=lookback_token_ids.shape[1],
            lookback_row_stride=lookback_token_ids.stride(0),
            lookback_col_stride=lookback_token_ids.stride(1),
            lookback_mask_row_stride=lookback_dead_mask.stride(0),
            lookback_mask_col_stride=lookback_dead_mask.stride(1),
            num_warps=4,
        )
        return output
