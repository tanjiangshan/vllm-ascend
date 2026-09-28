# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# 【文件职责】engram 的 EDP（Engram 数据并行）交换辅助。
#
# 【背景】engram 表按"哈希头"在 TP×EDP 组上分片（每 rank 拥有连续若干头
# 的完整行）。查表是按行的 gather，但各 DP 副本的 token 各不相同——因此:
#   - 查表前: gather_engram_hashes 把各副本的哈希 ID 拼成 rank-major 大批
#     （每副本填充到相同 token 槽位，保证 all_gather 形状静态、图捕获友好）；
#   - 查表后: exchange_engram_rows 用 all_to_all 把各 rank 查到的"别人
#     token 的头行"直接发回属主，各自切回自己 token 窗口并把各 rank 的
#     头并排拼宽。
#
# 【单节点假设】PP=PCP=DCP=1 时，现有 DP 组的成员恰为上游"节点内 engram
# DP 组"的成员，直接复用——不新建通信子、不改 vLLM 并行状态。
# =============================================================================
"""Node-local Engram exchange with A5 recompute-decode fixed token slots.

Reuse mainline EDP ownership and row exchange. The A5 scheduler can skip DP
metadata synchronization, so hash gathering must retain its fixed-slot path.
"""

import torch
import torch.distributed as dist
from vllm.config import get_current_vllm_config
from vllm.distributed import get_dp_group, get_engram_dp_group, get_engram_dp_size
from vllm.forward_context import get_forward_context

# Upstream #56741 normalized the V4.1 model package name.
# 【中文】DEAD_ID: 无效哈希的哨兵 ID（填充行用它，gather 后仍可识别）。
from vllm.models.deepseek_v41.common.engram import DEAD_ID
from vllm.models.deepseek_v41.nvidia.engram import (
    engram_head_shard_rank as engram_head_shard_rank,
)

from vllm_ascend.utils import get_potential_max_tokens, is_pd_decode_recompute_scheduler_enabled


def resolve_dp_shared_memory(requested: bool) -> bool:
    """Share a TP shard when local DP or PCP peers map the same host table."""
    return requested and get_engram_dp_size() > 1


def engram_gathered_num_tokens() -> int:
    """Per-rank token slot for the node-local Engram DP x PCP group."""
    """【中文说明】DP gather 时每 rank 的统一 token 槽位数（组内最大值）。
    原理: all_gather 要求各 rank 形状一致，先取组内最大 token 数作为公共槽，
    不足的副本用 DEAD_ID 填充。A5 重计算 decode 可能跳过 DP 元数据同步
    （本地 token 向量只有本 rank 计数），此时槽位改由共享配置推算。
    DP 元数据缺失则无法确定槽位，直接报错。"""
    context = get_forward_context()
    if (
        not getattr(context, "in_profile_run", False)
        and not getattr(context, "engram_uniform_dp_warmup", False)
        and is_pd_decode_recompute_scheduler_enabled()
    ):
        # Recompute decode can skip the DP metadata all-reduce. Its token
        # vector then contains only local counts, including on idle ranks or
        # ranks using different graph buckets. Size both Engram exchanges
        # from the shared configuration, never from that local vector.
        config = get_current_vllm_config()
        scheduler = config.scheduler_config
        query_len = 1 + config.speculative_config.num_speculative_tokens if config.speculative_config else 1
        return max(
            get_potential_max_tokens(),
            min(scheduler.max_num_batched_tokens, scheduler.max_num_seqs * query_len),
        )
    dp_metadata = context.dp_metadata
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
    """Collect the n-gram ids of every DP x PCP rank using one split table.

    Replicas are padded to a common token slot, including when recompute
    decode skips DP metadata synchronization and local graph sizes differ.
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


def exchange_engram_rows(staged: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Send each padded DP x PCP token block directly to its owning rank.

    Input is [EDP * slot, local_heads, dim], in destination-rank order.
    Each destination receives only its token block from every head owner,
    instead of materializing every destination's block as all-gather does.
    All ranks keep the same slot, including idle ranks with zero valid tokens.
    """
    """【中文说明】查表后的 DP 反向交换（all_to_all 版）: 各 rank 持有
    "所有副本 token × 本地头"的 staged 结果，all_to_all_single 把每个填充
    token 块直接发给其属主 rank（而非 all_gather 那样物化全量）；收到的
    块按源 rank 重排后，切回本副本 token 窗口并把各源的头段并排拼成
    [num_tokens, dp_size*local_heads, dim]。"""
    group = get_engram_dp_group()
    assert group is not None
    dp_size = group.world_size
    # divmod 分解出公共槽位大小（staged 一定是 world_size 的整数倍）。
    slot, remainder = divmod(staged.shape[0], dp_size)
    assert remainder == 0 and 0 <= num_tokens <= slot
    if dp_size == 1:
        return staged[:num_tokens]
    local_heads, dim = staged.shape[1:]
    recv = torch.empty_like(staged)
    dist.all_to_all_single(recv, staged.contiguous(), group=group.device_group)
    # Received chunks are source-rank major; heads are contiguous across EDP.
    # 【中文】收到的块是源 rank 主序: view 出 [dp_size, slot, ...] 后
    # permute 成 [slot, dp_size, ...]，头段沿 dim=2 天然连续并排。
    rows = recv.view(dp_size, slot, local_heads, dim).permute(1, 0, 2, 3)
    return rows[:num_tokens].reshape(num_tokens, dp_size * local_heads, dim)
