# SPDX-License-Identifier: Apache-2.0
# =====================================================================================
# MiniMax M3 模型包入口（vllm-ascend 华为昇腾 NPU 插件）。
#
# 模块职责：
#   本文件是 vllm_ascend.models.minimax_m3 包的 __init__.py，负责把包内各实现
#   文件中的核心类汇总导出，供 vLLM 的模型注册机制（通过 Registry 扫描
#   vllm_ascend.models 下所有模块）发现并实例化 MiniMax M3 模型。
#
# 包结构：
#   - minimax_m3.py     : 文本模型主体（DecoderLayer / MoE / 稀疏注意力层 / CausalLM）
#   - minimax_m3_vl.py  : 多模态（视觉-语言）包装器 MiniMaxM3SparseForConditionalGeneration
#   - msa_m3.py         : MSA（MiniMax Sparse Attention，MiniMax 稀疏注意力）后端，
#                         包含 indexer（闪电索引器）与 block-sparse 注意力实现
#   - ops/              : 底层算子（NPU AscendC 算子封装 + Triton 参考实现）
#
# 导出符号说明：
#   - MiniMaxM3Attention            : 全注意力（GQA）层，用于非稀疏层
#   - MiniMaxM3MoE                  : 混合专家（Mixture of Experts）MLP 模块
#   - MiniMaxM3SparseAttention      : MSA 块稀疏注意力层（含闪电索引器）
#   - MiniMaxM3SparseForCausalLM    : 文本生成模型顶层封装（LM head + logits）
#   - MiniMaxM3SparseForConditionalGeneration : 多模态 VL 模型顶层封装
#   - _get_rope_parameters          : 从 HF config 提取 RoPE 参数的辅助函数
#   - _sparse_attention_layer_ids   : 计算哪些层启用稀疏注意力的辅助函数
# =====================================================================================

from vllm_ascend.models.minimax_m3.minimax_m3 import (
    MiniMaxM3Attention,
    MiniMaxM3MoE,
    MiniMaxM3SparseAttention,
    MiniMaxM3SparseForCausalLM,
    _get_rope_parameters,
    _sparse_attention_layer_ids,
)
# 导入多模态 VL 顶层模型（视觉塔 + 语言模型组合）
from vllm_ascend.models.minimax_m3.minimax_m3_vl import MiniMaxM3SparseForConditionalGeneration

# __all__ 声明本包对外公开的符号清单：使用 `from ... import *` 时只会导入这些名字。
# 这是 Python 包的标准做法，同时也向读者标明包的公共 API 边界。
__all__ = [
    "MiniMaxM3Attention",
    "MiniMaxM3MoE",
    "MiniMaxM3SparseAttention",
    "MiniMaxM3SparseForCausalLM",
    "MiniMaxM3SparseForConditionalGeneration",
    "_get_rope_parameters",
    "_sparse_attention_layer_ids",
]
