# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# MTP 输入融合内核（Triton，NPU 上经 Triton-Ascend 后端运行）。
#
# 职责：把 MTP 层的输入预处理融合为一个内核——
#   1. 位置 0 的 embeds 置零（首 token 无前文嵌入）；
#   2. enorm(embeds) 与 hnorm(prev_hidden) 两次 RMSNorm；
#   3. 两者并排写入输出 [N, 2H]，直接供 eh_proj GEMM 消费。
# 替代原始的 where + 2 次 RMSNorm + cat 共 4 个算子。
#
# 语法点：@triton.jit 装饰器把 Python 函数编译为 Triton 内核；
# tl.constexpr 标注编译期常量（此处为隐藏维度 H 与块宽 BLOCK）。
# =============================================================================

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _rms_norm(x, w, eps, HIDDEN_SIZE: tl.constexpr):
    """Triton 内联 RMSNorm：x * rsqrt(mean(x^2)+eps) * w（FP32 计算）。

    参数（内核内为标量/向量语义）：
        x: 输入向量 [H]（FP32）。
        w: 归一化权重 [H]。
        eps: 防除零小量。
        HIDDEN_SIZE: 编译期常量，用于求均值。
    """
    x = x.to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / HIDDEN_SIZE
    # rsqrt：近似倒数平方根（比 1/sqrt 快）。
    rrms = tl.rsqrt(mean_sq + eps)
    w = w.to(tl.float32)
    return (x * rrms) * w


@triton.jit
def _fused_eh_norm_kernel(
    pos_ptr,
    embeds_ptr,
    embeds_stride,
    prev_ptr,
    prev_stride,
    enorm_w_ptr,
    hnorm_w_ptr,
    eps,
    out_ptr,
    out_stride,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """MTP input fusion: zero embeds at position 0, RMSNorm(embeds) with enorm
    and RMSNorm(prev_hidden) with hnorm, written side-by-side into ``out``
    ([N, 2H]) ready for the eh_proj GEMM. Replaces where + 2x RMSNorm + cat.

    融合内核主体：每个 token 一个 program（grid = (N,)）。

    参数：
        pos_ptr: [N] 位置数组（位置 0 置零 embeds）。
        embeds_ptr/prev_ptr: 嵌入与上一步隐状态的指针（带行步长）。
        enorm_w_ptr/hnorm_w_ptr: 两个 RMSNorm 权重。
        eps: RMSNorm epsilon。
        out_ptr/out_stride: 输出 [N, 2H] 及其行步长。
        H: 隐藏维度（编译期常量）。
        BLOCK: 处理 H 的块宽（2 的幂，编译期常量）。
    """
    # program_id(0)：本 program 负责的 token 下标。
    tok = tl.program_id(0)
    # arange(0, BLOCK)：块内偏移；mask 屏蔽超出 H 的部分。
    off = tl.arange(0, BLOCK)
    mask = off < H

    # 步骤1: embeds 分支——位置 0 置零后做 enorm RMSNorm，写前半 [0, H)。
    pos = tl.load(pos_ptr + tok)
    e = tl.load(embeds_ptr + tok * embeds_stride + off, mask=mask, other=0.0)
    # where(pos==0, 0, e)：首 token 的嵌入置零。
    e = tl.where(pos == 0, 0.0, e.to(tl.float32))
    ew = tl.load(enorm_w_ptr + off, mask=mask)
    e_normed = _rms_norm(e, ew, eps, H)
    tl.store(out_ptr + tok * out_stride + off, e_normed, mask=mask)

    # 步骤2: prev_hidden 分支——hnorm RMSNorm，写后半 [H, 2H)。
    p = tl.load(prev_ptr + tok * prev_stride + off, mask=mask, other=0.0)
    hw = tl.load(hnorm_w_ptr + off, mask=mask)
    p_normed = _rms_norm(p, hw, eps, H)
    tl.store(out_ptr + tok * out_stride + H + off, p_normed, mask=mask)


def fused_eh_norm(
    positions: torch.Tensor,
    inputs_embeds: torch.Tensor,
    previous_hidden: torch.Tensor,
    enorm_w: torch.Tensor,
    hnorm_w: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Returns cat([enorm(masked embeds), hnorm(prev_hidden)]) -> [N, 2H].

    融合入口：cat([enorm(掩码 embeds), hnorm(prev_hidden)]) -> [N, 2H]。

    参数：
        positions: [N] token 位置（位置 0 的 embeds 置零）。
        inputs_embeds: [N, H] 本步输入嵌入。
        previous_hidden: [N, H] 主模型上一步隐状态。
        enorm_w/hnorm_w: [H] RMSNorm 权重。
        eps: RMSNorm epsilon。

    返回：
        [N, 2H] 融合输出（供 eh_proj GEMM）。
    """
    n, h = inputs_embeds.shape
    out = torch.empty(n, 2 * h, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
    # 语法点：kernel[(grid,)](args) 是 Triton 的启动语法——
    # 每个启动一个 program；next_power_of_2 把 H 上取整到 2 的幂块宽。
    _fused_eh_norm_kernel[(n,)](
        positions,
        inputs_embeds,
        inputs_embeds.stride(0),
        previous_hidden,
        previous_hidden.stride(0),
        enorm_w,
        hnorm_w,
        eps,
        out,
        out.stride(0),
        h,
        triton.next_power_of_2(h),
    )
    return out
