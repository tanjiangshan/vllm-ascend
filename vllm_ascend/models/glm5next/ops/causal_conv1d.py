# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AscendC short convolution for GLM prefill, decode and MTP verification.

GLM KDA 层的短因果卷积（AscendC 实现），覆盖 prefill、decode 与
MTP 验证三种模式。

因果卷积原理：每个输出位置只看当前及之前 kernel_size-1 个输入
（因果性）；KDA 中 q/k/v 在进线性注意力前各过一个短卷积（kernel=4），
为纯递归的状态更新引入局部时序感受野。卷积状态 conv_state 缓存了
最近 kernel_size-1 个输入，使 decode 时每步只需当前输入即可滑窗。

NPU 适配点：
  - 核心计算走 AscendC 自定义算子
    torch.ops._C_ascend.npu_causal_conv1d_custom（torch_npu 注册）；
  - aclnnCausalConv1d 对非连续状态不回写视图——非连续 conv_state
    （分页缓存视图）先 gather 出本批的连续副本再调用，算完 scatter 回。
"""

import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.ops.triton.kda.conv_state import copy_conv_state


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    *,
    run_mode: int,
    initial_state_mode: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Consume GDN metadata and update the caller's [cache, state_len, dim] state.

    消费 GDN 元数据并更新调用方的 [cache, state_len, dim] 卷积状态。

    参数（语法点：* 之后 keyword-only）：
        x: [num_tokens, channels] 输入（q|k|v 已沿通道拼接）。
        weight: [kernel_size, channels] 卷积权重。
        conv_state: [num_cache, state_len, dim] 卷积状态缓存视图。
        query_start_loc: [n+1] 每序列累积 query 长度（变长批）。
        cache_indices: [n] 每序列的状态缓存行号。
        run_mode: 0=prefill（首 token 需初始化状态），
                  1=decode/verify（读旧状态滑窗）。
        initial_state_mode: [n] 每序列是否有初始状态（prefill 续写）。
        num_accepted_tokens: [n] 每序列被接受的投机 token 数
            （MTP 验证后回滚窗口用）。

    返回：
        [num_tokens, channels] 卷积输出（padded 请求的行为 0）。
    """
    # Padded requests can be skipped by the kernel; their output must stay zero.
    # 步骤1: 输出初始化为零——padding 请求被内核跳过，其输出必须保持 0。
    output = torch.zeros_like(x)
    if cache_indices.shape[0] == 0:
        return output
    kernel_state = conv_state
    kernel_indices = cache_indices
    # aclnnCausalConv1d materializes a non-contiguous state without writing its
    # mutations back to the view. Stage only this batch's rows, retaining both
    # page strides and DS layouts; never copy the entire persistent cache.
    # 步骤2: 非连续状态中转——aclnnCausalConv1d 会把非连续状态物化成
    # 连续副本而不回写视图。只中转本批的行（保留页步长与 DS 布局），
    # 绝不拷贝整个持久缓存。
    if not conv_state.is_contiguous():
        requests = cache_indices.shape[0]
        state_len, dim = conv_state.shape[1:]
        kernel_state = torch.empty((requests, state_len, dim), dtype=conv_state.dtype, device=conv_state.device)
        kernel_indices = torch.empty(requests, dtype=torch.int32, device=cache_indices.device)
        # gather：持久缓存视图 -> 连续内核副本（write_back=False 只读出）。
        copy_conv_state(conv_state, kernel_state, cache_indices, query_start_loc, kernel_indices, write_back=False)
    # Return the declared result so graph functionalization retains the call.
    # 步骤3: 调用 AscendC 因果卷积内核。
    # 返回声明的结果使图函数化（graph functionalization）保留该调用
    # （自定义算子可能原地写状态，返回值保证它不被优化掉）。
    result = torch.ops._C_ascend.npu_causal_conv1d_custom(
        output,
        x,
        weight,
        conv_state=kernel_state,
        bias_opt=None,
        query_start_loc_opt=query_start_loc,
        cache_indices_opt=kernel_indices,
        initial_state_mode_opt=initial_state_mode,
        num_accepted_tokens_opt=num_accepted_tokens,
        activation_mode=1,
        pad_slot_id=PAD_SLOT_ID,
        run_mode=run_mode,
    )
    # 步骤4: 非连续状态 scatter 回原缓存视图（write_back=True）。
    if not conv_state.is_contiguous():
        copy_conv_state(conv_state, kernel_state, cache_indices, query_start_loc, kernel_indices, write_back=True)
    return result
