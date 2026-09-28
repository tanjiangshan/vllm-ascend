# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend recurrent-state gather/scatter for the GLM-5.3-Flash KDA layers.

GLM-5.3-Flash KDA 层的循环状态 gather/scatter（Ascend 版）。

The upstream helpers in ``vllm.model_executor.layers.mamba.ops`` assert
``state.is_cuda`` and launch Triton kernels that reference
``tl.extra.cuda.gdc_wait``, neither of which holds on Ascend. Both are plain
index ops, so they are expressed in torch here. Keeping the masking on device
(rather than branching on ``has_initial_state``) avoids a host sync and leaves
the sequence ACL-graph safe.

（上游实现断言 state.is_cuda 且引用 CUDA 专属的 Triton 原语，Ascend 上
均不成立。这里改用纯 torch 索引操作表达；掩码保留在设备端而非分支
判断 has_initial_state，避免 host 同步，保证序列 ACL 图安全。）
"""

import torch


def gather_initial_states(
    state: torch.Tensor,
    indices: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> torch.Tensor:
    """Read the cache rows at ``indices``, zeroing sequences that start fresh.

    读取 indices 指定的缓存行；无初始状态的序列置零。

    参数：
        state: [slots, heads, dk, dv] 持久状态缓存。
        indices: [n] 状态槽号。
        has_initial_state: [n] 0/1 掩码（chunked prefill 续写标记）。

    返回：
        [n, heads, dk, dv] 收集到的初始状态。

    技巧：idx * has_initial_state 把"无初始状态"的行重定向到第 0 行
    （避免越界），再用 torch.where 掩零——全程设备端，无 host 同步。
    """
    idx = indices.to(torch.int64) * has_initial_state.to(torch.int64)
    out = state.index_select(0, idx)
    # view([-1] + [1]*(dim-1))：把 [n] 掩码广播到状态的全部尾维。
    keep = has_initial_state.view([-1] + [1] * (state.dim() - 1)).to(torch.bool)
    # Fresh requests must ignore stale cache values, including NaN/Inf:
    # multiplying such values by zero would still produce NaN.
    # 新请求必须忽略缓存中的陈旧值（包括 NaN/Inf）：乘零对 NaN 仍得 NaN，
    # 所以必须用 where 选择而非乘法掩码。
    return torch.where(keep, out, 0)


def scatter_states(
    state: torch.Tensor,
    src: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    """Scatter ``src`` rows into ``state`` at ``indices`` (in place).

    把 src 的各行散布写回 state 的 indices 行（原地）。

    Equivalent to ``state[indices] = src``. Cache slots are unique per sequence,
    so the write needs no atomics. ``gather_initial_states`` is the read-side
    counterpart.

    参数：
        state: [slots, heads, dk, dv] 持久状态缓存（原地修改）。
        src: [n, heads, dk, dv] 新状态。
        indices: [n] 目标槽号（每序列唯一，无需原子操作）。

    返回：
        None（原地写）。gather_initial_states 是其读取侧对应。
    """
    state.index_copy_(0, indices.to(torch.int64), src)
