# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# 【文件职责】engram 的平台无关辅助：开关判定、token 资格掩码、门控公式。
#
# 【engram_gate 门控原理】记忆检索结果不能无条件写入残差流——门控向量的
# 大小由"当前隐状态与记忆 key 的余弦相似度"决定（经各自 RMS 归一后内积），
# 再过 sigmoid 得到 0~1 门控；value 以门控强度加到残差上。隐状态与 key
# 都在"旋转基"（checkpoint 提供的 global_rotation，见 model.py 的
# engram_rotation）下工作，门控计算需先还原到原始基。
# =============================================================================
"""Gating and token-eligibility helpers for the Ascend Engram port.

The bucket layout and the hashing itself come from upstream, so a checkpoint
lands on the same rows here as it does on the accelerators upstream supports;
the Ascend token history lives in ``hash_state`` next to the SWA slot cache,
which is where upstream keeps it too.
"""

import torch


def engram_enabled(text_config) -> bool:
    """Whether the checkpoint declares Engram n-gram layers."""
    """【中文说明】checkpoint 是否声明了 engram 层（engram_layer_ids 非空）。
    语法点: getattr(..., None) 兜底 + bool() 归一化为布尔。"""

    return bool(getattr(text_config, "engram_layer_ids", None))


def valid_engram_token_mask(
    input_ids: torch.Tensor,
    image_token_id: int,
    image_pad_token_id: int,
) -> torch.Tensor:
    """Exclude the complete V4.1 image region from n-gram history."""
    """【中文说明】n-gram 历史的 token 资格掩码：排除图像哨兵与图像填充 ID。
    原理: 图像 token 的 n-gram 无文本语义，混入历史会污染哈希匹配。"""

    return (input_ids != image_token_id) & (input_ids != image_pad_token_id)


def engram_gate(
    hidden: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    channel_weight: torch.Tensor,
    rotation_block: torch.Tensor,
    token_mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply original-basis gating to a rotated residual and rotated value.

    ``hidden`` and ``key`` have shape [tokens, hc_mult, hidden_size].
    The saved rotation consists of identical diagonal blocks. Restore hidden
    in FP32; the value projection already includes the forward rotation.
    """
    """【中文说明】engram 门控: 用原始基下的隐状态-key 相似度调制 value 并写回。

    参数:
        hidden: [tokens, hc_mult, hidden] 旋转基下的多支路残差流。
        key: [tokens, hc_mult, hidden] 旋转基下的记忆 key（已含前向旋转）。
        value: [tokens, hidden] 记忆 value（投影时已含前向旋转，无需再转）。
        channel_weight: [hidden] 通道级融合权重（q_weight×k_weight 逐元素积）。
        rotation_block: [32, 32] 旋转块（checkpoint 的 global_rotation 左上块）。
        token_mask: [tokens] bool（True=参与 engram 的 token）。
        eps: RMS 归一的 epsilon。
    返回:
        [tokens, hc_mult, hidden] 门控更新后的残差流（hidden + gate·value）。
    算法步骤:
        1) 隐状态从旋转基还原到原始基（分块乘 rotation_block 的转置）；
        2) 分别算 hidden/key 的 RMS 归一因子，二者相乘；
        3) 加权内积得到逐支路相似度 dot，量级取 sqrt(|dot|) 保持符号；
        4) sigmoid(dot 量级) 作为门控；无效 token 的门控清零；
        5) 输出 = hidden + gate ⊗ value。
    """
    dim = hidden.shape[-1]
    # 步骤1: 旋转还原。unflatten 把 hidden 切成 [-1, 32, 32] 的块，每块
    # 右乘 rotation_block^T——"相同对角块"结构意味着只需一个 32×32 块。
    original = (hidden.float().unflatten(-1, (-1, rotation_block.shape[0])) @ rotation_block.float().T).flatten(-2)
    # 步骤2: key 转 FP32；hidden/key 各自的 RMS 因子相乘（相似度的归一）。
    key = key.float()
    rstd = torch.rsqrt(original.square().mean(-1) + eps)
    rstd *= torch.rsqrt(key.square().mean(-1) + eps)
    # 步骤3: 通道加权的内积相似度，再乘两个 RMS 因子与 1/sqrt(dim)。
    dot = (original * channel_weight.float() * key).sum(-1) * rstd * dim**-0.5
    # 步骤4: 量级 = sqrt(|dot|)（clamp 防 0），保留符号后过 sigmoid 门控。
    magnitude = dot.abs().clamp_min(1e-6).sqrt()
    gate = torch.sigmoid(torch.where(torch.signbit(dot), -magnitude, magnitude))
    # 无效 token（如图像区）门控置 0——记忆完全不写入。
    gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    # 步骤5: value 沿支路维广播相加（gate: [t, hc, 1] × value: [t, 1, hidden]）。
    return (hidden.float() + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(hidden.dtype)
