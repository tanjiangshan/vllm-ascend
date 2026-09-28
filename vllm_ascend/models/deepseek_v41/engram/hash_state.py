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
from vllm.triton_utils import triton


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


class AscendNgramHashState(NgramHashState):
    """Reuse upstream history lifecycle, splitting only the NPU hash launch.

    Triton-Ascend cannot compile the combined request-search/hash kernel.
    Keep that compiler workaround in this subclass, without changing vLLM.
    """
    """【中文说明】继承上游 NgramHashState（复用其历史分配与生命周期管理），
    仅重写哈希发射路径：Triton-Ascend 编译不了"搜索+哈希"合并 kernel，
    该编译器 workaround 收敛在本子类，不改动 vLLM 上游。"""

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
        from vllm_ascend.ops.triton.engram_hash import _engram_req_index_kernel

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
        from vllm_ascend.ops.triton.engram_hash import _hash_ids_kernel

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
