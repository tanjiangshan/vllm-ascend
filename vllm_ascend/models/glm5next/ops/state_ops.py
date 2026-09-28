# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend recurrent-state gather/scatter for the GLM-5.3-Flash KDA layers.

GLM-5.3-Flash KDA 层的循环状态 gather/scatter（Ascend 版）：
gather_initial_states 直接复用 vllm_ascend.ops.triton.mamba.state_ops
的实现；scatter_states 在 NPU 上走 torch_npu.npu_scatter_nd_update_，
把每个连续状态视为一整行缓存，保留页步长、零拷贝。
"""

import math

import torch
import torch_npu

from vllm_ascend.ops.triton.mamba.state_ops import (
    gather_initial_states as gather_initial_states,
)


def scatter_states(state: torch.Tensor, src: torch.Tensor, indices: torch.Tensor) -> None:
    """Write unique selected state rows without a contiguous copy of the pool."""
    """【中文说明】把 src 各行按 indices 散布写回 state（原地），
    不对状态池做连续拷贝：NPU 上走 npu_scatter_nd_update_ 的"整行"
    视图更新；非 NPU 设备退回 index_copy_。

    参数：
        state: [slots, ...] 持久状态缓存（原地修改）。
        src: [n, ...] 新状态行（须与 state 尾维一致）。
        indices: [n] 目标槽号（每序列唯一，无需原子操作）。
    """
    if state.device.type != "npu":
        state.index_copy_(0, indices.to(torch.int64), src)
        return
    if indices.numel() == 0 or math.prod(state.shape[1:]) == 0:
        return
    assert state.ndim >= 2 and src.ndim == state.ndim
    assert indices.ndim == 1 and indices.device == state.device
    assert src.shape == (indices.numel(), *state.shape[1:])
    assert indices.dtype in (torch.int32, torch.int64)
    assert state[0].is_contiguous() and src[0].is_contiguous()
    # Treat each contiguous state as one cache row. These views preserve the
    # page stride and storage offset without copying the padded state pool.
    # （中文补充）把每个连续状态视为一整行缓存：以下 view 保留页步长与
    # 存储偏移，不拷贝填充后的状态池。
    row_size = math.prod(state.shape[1:])
    torch_npu.npu_scatter_nd_update_(
        state.view(state.shape[0], row_size),
        indices.unsqueeze(-1),
        src.view(indices.numel(), row_size),
    )
