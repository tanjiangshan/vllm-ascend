# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM bounded-gate contracts for the AscendC KDA operators.

GLM 有界门控（bounded gate）KDA 算子的调用契约层。

KDA 状态更新公式（每头、FP32）：
    S_t = a_t * S_{t-1} + beta_t * k_t v_t^T     （状态）
    o_t = (q_t^T S_t) / ((q_t^T u_t) + eps)      （输出，u 累积分母）
  其中 a_t = lower_bound + (1-lower_bound)*sigmoid(A_log 衰减) 是 GLM 的
  "有界 sigmoid 遗忘门"（safe gate，替代 GDN 的无界 softplus）。

本模块封装 vllm_ascend.ops.kda 中的两个 AscendC 内核：
  - run_recurrent_kda：逐 token 递归（decode / MTP 验证，
    支持 num_accepted_tokens 拒绝采样回滚）；
  - run_chunk_kda：分块并行（prefill，CPU 端 chunk 描述符）。
外加状态 gather/scatter 与输出写回。
"""

import torch

from vllm_ascend.models.glm5next.ops.state_ops import gather_initial_states, scatter_states
from vllm_ascend.ops.kda import run_chunk_kda, run_recurrent_kda
from vllm_ascend.ops.triton.kda.output_writeback import write_recurrent_output

# 递归内核一次能处理的最大 token 列数（约束 num_spec+1 <= 8，
# 即至多 7 个投机 token）。
KDA_MAX_RECURRENT_TOKENS = 8


def recurrent_kda(
    q,
    k,
    v,
    raw_gate,
    raw_beta,
    state,
    cu_seqlens,
    state_indices,
    a_log,
    dt_bias,
    lower_bound,
    num_accepted_tokens=None,
    output_buffer=None,
):
    """Update the selected VK state slots, including MTP rejection rollback.

    递归 KDA：更新选中的状态槽位（含 MTP 拒绝回滚）。

    参数：
        q/k/v: [1, n, heads, head_dim] 查询/键/值（n 为批内 token 列数）。
        raw_gate: [1, n, heads, head_dim] 原始门控 g1（未过激活；
            内核按有界 sigmoid 处理）。
        raw_beta: [1, n, heads] 原始 beta（内核内 sigmoid）。
        state: [num_slots, heads, head_dim, head_dim] 持久递归状态缓存。
        cu_seqlens: [n_seq+1] 每序列累积列数。
        state_indices: [n_seq] 每序列的状态槽号。
        a_log/dt_bias: 衰减与时间步偏置参数。
        lower_bound: 有界 sigmoid 的下界。
        num_accepted_tokens: [n_seq] 每序列被接受的投机 token 数——
            状态只前进 accepted 步（拒绝的 draft 列不提交）。
        output_buffer: 可选输出缓冲（提供时内核直写并返回它）。

    返回：
        [1, n, heads, head_dim] 输出（或 output_buffer 本身）；
        超出 cu_seqlens[-1] 的填充列置 0。
    """
    num_seqs = cu_seqlens.numel() - 1
    # 裁剪索引到实际序列数（批可能带填充）。
    state_indices = state_indices[:num_seqs]
    if num_accepted_tokens is not None:
        num_accepted_tokens = num_accepted_tokens[:num_seqs]
    output = run_recurrent_kda(
        q,
        k,
        v,
        raw_gate,
        raw_beta,
        state,
        cu_seqlens,
        state_indices,
        a_log,
        dt_bias,
        lower_bound=lower_bound,
        # beta 未预过 sigmoid（内核以 SIGMOID_BETA 模式处理）。
        beta_is_preprocessed=False,
        num_accepted_tokens=num_accepted_tokens,
    )
    if output_buffer is not None:
        # 提供了缓冲：把输出写回捕获缓冲并原样返回（省一次拷贝）。
        write_recurrent_output(output, output_buffer, cu_seqlens)
        return output_buffer
    # 无缓冲：把超出实际 token 的填充列置 0（valid 掩码）。
    valid = torch.arange(q.shape[1], device=q.device) < cu_seqlens[-1]
    return output.masked_fill(~valid[None, :, None, None], 0)


def chunk_kda(
    q,
    k,
    v,
    raw_gate,
    raw_beta,
    state,
    state_indices,
    has_initial_state,
    metadata,
    a_log,
    dt_bias,
    lower_bound,
):
    """Run prefill with CPU chunk descriptors prepared by the GDN builder.

    分块 KDA（prefill）：使用 GDN 构建器准备的 CPU 端 chunk 描述符。

    原理：prefill 序列长，逐 token 递归太慢；分块算法把序列切成
    固定大小（64）的块，块内用矩阵乘并行计算（qK^T 菱形分解），
    块间串行传递状态——兼顾并行度与因果性。

    参数：
        q/k/v: [1, n, heads, head_dim]（n 为 prefill token 数）。
        raw_gate: [1, n, heads, head_dim] 原始门控。
        raw_beta: [1, n, heads] 原始 beta（此处先做 fp32 sigmoid 预处理）。
        state: [slots, heads, head_dim, head_dim] 持久状态缓存。
        state_indices: [n_seq] 状态槽号。
        has_initial_state: [n_seq] 是否有初始状态（chunked prefill 续写）。
        metadata: GDN 元数据（含 cu_seqlens_host/kern 与 chunk 索引）。
        a_log/dt_bias/lower_bound: 同 recurrent_kda。

    返回：
        [1, n, heads, head_dim] prefill 输出（最终状态已 scatter 回缓存）。
    """
    if metadata.keep_meta is not None:
        # keep_meta：只保留真正参与 prefill 的序列（过滤被剪枝的）。
        state_indices = state_indices[metadata.keep_meta]
        has_initial_state = has_initial_state[metadata.keep_meta]
    # 步骤1: gather 初始状态（无初始状态的序列置零，FP32 连续化）。
    initial_state = gather_initial_states(state, state_indices, has_initial_state).float().contiguous()
    # 步骤2: cu_seqlens 优先用设备端（kern），否则用 CPU 端（host）。
    cu_seqlens = metadata.cu_seqlens_host if metadata.cu_seqlens_kern is None else metadata.cu_seqlens_kern
    # 步骤3: 分块内核（beta 在此路径需要预过 fp32 sigmoid）。
    output, final_state = run_chunk_kda(
        q,
        k,
        v,
        raw_gate,
        raw_beta.float().sigmoid(),
        initial_state,
        cu_seqlens,
        metadata.chunk_indices_chunk64_host,
        a_log,
        dt_bias,
        lower_bound=lower_bound,
    )
    # 步骤4: 最终状态写回持久缓存（转回归存 dtype）。
    scatter_states(state, final_state.to(state.dtype), state_indices)
    return output
