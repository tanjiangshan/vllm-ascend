# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hyper-connection width helpers used by the GLM-5.3-Flash decoder layers.

mHC（multi-head hyper-Connection，超连接）宽度辅助函数。

Upstream keeps these next to the MHC ops in ``vllm.model_executor.layers.mhc``.
They are plain shape ops with no backend-specific behavior, so vLLM Ascend
carries its own copy while the GLM-5.3-Flash architecture lives downstream.

原理：mHC 把残差流从 1 条扩展为 n 条（mhc_num_residual_streams）。
进 mHC 段前用 hc_expand 复制扩展，出 mHC 段（最后一层）用
hc_contract 收缩回单流。
"""

import torch


def hc_expand(x: torch.Tensor, n: int) -> torch.Tensor:
    """[s, hidden_size] -> [s, n * hidden_size] by replication.

    宽度扩展：[s, H] -> [s, n, H]（按复制），后续 reshape 成 [s, n*H]。

    参数：
        x: [s, H] 输入。
        n: 残差流条数。

    返回：
        [s, n, H]（连续内存）。expand 本身是视图（不拷贝），
        contiguous() 物化成实际张量。
    """
    return x.unsqueeze(1).expand(-1, n, -1).contiguous()


def hc_contract(x: torch.Tensor, n: int) -> torch.Tensor:
    """[s, n * hidden_size] -> [s, hidden_size] by averaging.

    宽度收缩：[s, n, H] -> [s, H]（按平均）。

    参数：
        x: [s, n, H] 输入（n 条超连接残差流）。
        n: 残差流条数（用于形状校验，未直接参与计算）。

    返回：
        [s, H]（n 条流的逐元素平均）。
    """
    return x.mean(dim=1)
