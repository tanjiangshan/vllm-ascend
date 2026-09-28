# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# 【文件职责】engram 记忆层的投影与门控（Ascend 旋转 checkpoint 版）。
#
# 【在模型中的位置】DeepseekV41DecoderLayer 里 self.engram = AscendEngram(config)；
# 主干 forward 中，engram 层先拿到 runner 预查好的记忆行（rows），本模块
# 用 wkv 投影出 key/value，再经 engram_gate（common.py）门控写回残差流。
#
# 【与 model.py 的张量流】rows [tokens, (ngram-1)*n_heads*head_dim]（嵌入表
# 检索结果的展平）→ wkv → (key[tokens, hc_mult, hidden], value[tokens, hidden])
# → engram_gate → 更新后的多支路残差流。
# =============================================================================
"""Engram projection and gate for the rotated Ascend checkpoint."""

import torch
from torch import nn

from .common import engram_gate


class AscendEngram(nn.Module):
    """Consume rows prepared by the v1 runner using the existing rotated-checkpoint gate."""
    """【中文说明】engram 记忆层：消费 runner 预检索的记忆行，经投影与门控
    写回残差流。继承 nn.Module，门控公式复用 common.engram_gate（旋转
    checkpoint 的原版实现）。"""

    def __init__(self, config) -> None:
        """参数: config 为 V4.1 文本配置（engram_*/hc_mult/hidden_size 等）。"""
        super().__init__()
        self.dim = config.hidden_size
        self.hc_mult = config.hc_mult
        self.eps = config.rms_norm_eps
        # wkv: 记忆行 → (hc_mult+1)*hidden。输出一次性切分成
        # key（hc_mult 份，对应每条残差支路一个记忆 key）与 value（1 份）。
        self.wkv = nn.Linear(
            (config.engram_max_ngram_size - 1) * config.engram_n_heads * config.engram_head_dim,
            (self.hc_mult + 1) * self.dim,
            bias=False,
            dtype=torch.bfloat16,
        )
        # q_weight/k_weight: [hc_mult, hidden] 通道级融合权重——逐元素积后
        # 作为 engram_gate 的 channel_weight（放大相关通道的相似度判别）。
        self.q_weight = nn.Parameter(torch.empty(self.hc_mult, self.dim, dtype=torch.bfloat16))
        self.k_weight = nn.Parameter(torch.empty(self.hc_mult, self.dim, dtype=torch.bfloat16))

    def forward(
        self,
        hidden_states: torch.Tensor,
        rows: torch.Tensor,
        token_mask: torch.Tensor,
        rotation: torch.Tensor,
    ) -> torch.Tensor:
        """engram 前向：记忆行 → key/value → 门控写回。

        参数:
            hidden_states: [tokens, hc_mult, hidden] 多支路残差流（旋转基）。
            rows: [tokens, (ngram-1)*n_heads*head_dim] 嵌入表检索行（展平）。
            token_mask: [tokens] bool（True=参与 engram）。
            rotation: [32, 32] 全局旋转块（model.py 的 engram_rotation）。
        返回:
            [tokens, hc_mult, hidden] 门控更新后的残差流。
        步骤:
            1) wkv 一次性投影记忆行；
            2) 按 [hc_mult*dim, dim] 切分成 key 与 value；
            3) q_weight×k_weight 得通道权重，连同旋转块交给 engram_gate。
        """
        # 步骤1+2: 投影后切分——前 hc_mult*dim 是 key，最后 dim 是 value。
        kv = self.wkv(rows)
        key, value = kv.split([self.hc_mult * self.dim, self.dim], -1)
        # 步骤3: 门控融合。key reshape 出支路维；通道权重转 FP32 精确计算。
        return engram_gate(
            hidden_states,
            key.view(hidden_states.shape[0], self.hc_mult, self.dim),
            value,
            self.q_weight.float() * self.k_weight.float(),
            rotation,
            token_mask,
            self.eps,
        )
