# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# 【文件职责】engram 的 EDP（Engram 数据并行）交换辅助。
#
# 【背景】engram 表按"哈希头"在 TP×EDP 组上分片（每 rank 拥有连续若干头
# 的完整行）。查表是按行的 gather，但各 DP 副本的 token 各不相同——因此:
#   - 查表前: gather_engram_hashes 把各副本的哈希 ID 拼成 rank-major 大批
#     （每副本填充到相同 token 槽位，保证 all_gather 形状静态、图捕获友好）；
#   - 查表后: _gather_engram_rows 把各 rank 查到的"别人 token 的头行"经
#     all_gather 汇聚，再各自切回自己 token 窗口并把各 rank 的头并排拼宽。
#
# 【单节点假设】PP=PCP=DCP=1 时，现有 DP 组的成员恰为上游"节点内 engram
# DP 组"的成员，直接复用——不新建通信子、不改 vLLM 并行状态。
# =============================================================================
"""Uniform Engram DP exchange, backported from vLLM f84b0c4bce.

For single-node PP=PCP=DCP=1, the existing DP group has exactly the
membership of upstream's node-local Engram DP group. Reuse it without
creating another communicator or modifying vLLM parallel state.
"""

import torch
from vllm.distributed import get_dp_group, get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context

# Upstream #56741 normalized the V4.1 model package name.
# 【中文】DEAD_ID: 无效哈希的哨兵 ID（填充行用它，gather 后仍可识别）。
from vllm.models.deepseek_v41.common.engram import DEAD_ID
from vllm.triton_utils import tl, triton


def get_engram_dp_group():
    """取 engram 使用的 DP 组；world_size<=1（无 DP）时返回 None。"""
    group = get_dp_group()
    return group if group.world_size > 1 else None


def get_engram_dp_size():
    """engram DP 组大小；无 DP 时为 1。"""
    group = get_engram_dp_group()
    return group.world_size if group is not None else 1


def engram_head_shard_rank() -> int:
    """This rank's slot among the hash-head shards of one engram table.

    TP-major, so the shards a DP gather brings in are contiguous heads and
    the following TP gather completes the head order.
    """
    """【中文说明】本 rank 在一张 engram 表的头分片序号（TP-major 编号:
    rank = tp_rank * dp_size + dp_rank）。这样一次 DP gather 汇入的分片恰好
    是连续头段，随后的 TP gather（按 dim=1 拼接）正好恢复 checkpoint 顺序。"""
    dp_group = get_engram_dp_group()
    dp_size = dp_group.world_size if dp_group is not None else 1
    dp_rank = dp_group.rank_in_group if dp_group is not None else 0
    return get_tensor_model_parallel_rank() * dp_size + dp_rank


def engram_gathered_num_tokens() -> int:
    """Per-replica token slot for the node-local Engram DP group."""
    """【中文说明】DP gather 时每副本的统一 token 槽位数（各副本的最大值）。
    原理: all_gather 要求各副本形状一致，先取组内最大 token 数作为公共槽；
    不足的副本用 DEAD_ID 填充。DP 元数据缺失则无法确定槽位，直接报错。"""
    dp_metadata = get_forward_context().dp_metadata
    if dp_metadata is None:
        raise RuntimeError("a DP-shared engram table needs DP token metadata")
    group = get_engram_dp_group()
    assert group is not None
    # Engram groups are contiguous slices of the full DP group.
    # 【中文】engram 组是完整 DP 组的连续切片——用组内起始偏移切出本组
    # 各成员的 token 计数再取最大。注意 *_cpu 后缀：CPU 张量，可安全取 int。
    start = get_dp_group().rank_in_group - group.rank_in_group
    return int(dp_metadata.num_tokens_across_dp_cpu[start : start + group.world_size].max())


def gather_engram_hashes(hash_ids: torch.Tensor, *, dp_shared_memory: bool = False) -> torch.Tensor:
    """Collect the n-gram ids of every DP replica sharing one table.

    Replicas are padded to a common token slot, so the gathered shape is
    static under CUDA graph capture (where DP already pads alike).
    """
    """【中文说明】收集共享同一张表的所有 DP 副本的 n-gram 哈希 ID：各副本
    填充到公共 token 槽位后 all_gather（gather 后形状静态、图捕获友好，
    DP 本身在图模式下也做同样填充）。共享内存模式下每副本直接查自己的
    token，无需 gather，原样返回。"""
    dp_group = get_engram_dp_group()
    if dp_group is None or dp_shared_memory:
        # 无 DP 或共享内存模式: 直接返回本副本哈希。
        return hash_ids
    slot = engram_gathered_num_tokens()
    if hash_ids.shape[0] > slot:
        raise ValueError("Engram token count exceeds the DP token slot")
    # 不足槽位用 DEAD_ID 填充（查表时会被清零）。
    if hash_ids.shape[0] < slot:
        pad = hash_ids.new_full((slot - hash_ids.shape[0], *hash_ids.shape[1:]), DEAD_ID)
        hash_ids = torch.cat((hash_ids, pad))
    # rank-major 拼接: [dp_size * slot, ...]。
    return dp_group.all_gather(hash_ids, dim=0)


@triton.jit(do_not_specialize=["num_tokens", "token_start", "num_elements"])
def _engram_select_rows_kernel(
    gathered,
    output,
    num_tokens,
    token_start,
    num_elements,
    LOCAL_WIDTH: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # 【中文】行选择 kernel: 从 rank-major 的 gathered 缓冲中切出本副本
    # token 窗口，并把各 rank 的本地宽度并排拼成完整宽度。
    # gathered 布局: [rank][token][local_width]，本 rank 保留的窗口是
    # [token_start, token_start+num_tokens)。
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # 输出扁平下标 → (输出 token, 输出列)。
    tokens = token_start + offsets // WIDTH
    cols = offsets % WIDTH
    # 输出列 → (源 rank, 本地列): 源 rank = 输出列 // LOCAL_WIDTH，
    # 再由 (rank 的 token 基址 + 本地 token) 算源扁平下标。
    source = (cols // LOCAL_WIDTH * num_tokens + tokens) * LOCAL_WIDTH
    source += cols % LOCAL_WIDTH
    values = tl.load(gathered + source, (offsets < num_elements) & (tokens < num_tokens), other=0)
    tl.store(output + offsets, values, offsets < num_elements)


def _engram_select_rows(
    gathered: torch.Tensor,
    output: torch.Tensor,
    source_tokens: int,
    token_start: int,
    local_width: int,
) -> None:
    """Copy one token window out of a rank-major gathered buffer.

    Both gathers land rank-major ([rank][token][local width]); this walks the
    window the rank keeps and lays its ranks out side by side as width.
    """
    """【中文说明】把本副本的 token 窗口从 rank-major gather 结果中拷出：
    遍历本 rank 保留的窗口，把各 rank 的段并排铺成完整宽度
    （[rank][token][本地宽] → [token][全宽]）。"""
    if output.numel() == 0:
        return
    _engram_select_rows_kernel[(triton.cdiv(output.numel(), 1024),)](
        gathered,
        output,
        source_tokens,
        token_start,
        output.numel(),
        local_width,
        output.shape[1] * output.shape[2],
        BLOCK_SIZE=1024,
    )


def _gather_engram_rows(staged: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Exchange DP tokens for heads, retaining only this replica's tokens."""
    """【中文说明】查表后的 DP 反向交换: 各 rank 持有"所有副本 token × 本地
    头"的结果，经 all_gather 后切出自己 token 的窗口，并把各 rank 的头
    段并排拼成 [num_tokens, dp_size*local_heads, dim]。
    参数: staged [dp_slot, local_heads, dim]（dp_slot = 公共槽位 token 数）。
    返回: [num_tokens, dp_size*local_heads, dim]。"""
    dp_group = get_engram_dp_group()
    assert dp_group is not None
    # divmod 分解出公共槽位大小（staged 一定是 world_size 的整数倍）。
    slot, remainder = divmod(staged.shape[0], dp_group.world_size)
    assert remainder == 0 and 0 <= num_tokens <= slot
    gathered = dp_group.all_gather(staged, dim=0)
    local_heads, dim = staged.shape[1:]
    rows = staged.new_empty((num_tokens, dp_group.world_size * local_heads, dim))
    # token_start = 本副本在 rank-major 布局中的窗口起点。
    _engram_select_rows(
        gathered,
        rows,
        staged.shape[0],
        dp_group.rank_in_group * slot,
        local_heads * dim,
    )
    return rows
