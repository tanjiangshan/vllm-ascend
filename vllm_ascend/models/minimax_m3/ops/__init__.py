# SPDX-License-Identifier: Apache-2.0
# =====================================================================================
# MiniMax M3 MSA 稀疏注意力算子子包（ops）入口。
#
# 模块职责：
#   存放 MiniMax M3 稀疏注意力（MSA, MiniMax Sparse Attention）的底层计算算子，
#   供上层 msa_m3.py 中的注意力后端调用。本包不导出任何符号（无 __all__），
#   上层代码按需直接 `from ...ops.msa_m3_xxx import ...` 具体模块。
#
# 子模块结构：
#   - msa_m3_npu.py       : NPU 原生算子封装层。调用华为 CANN 的 AscendC 自定义算子
#                           （torch.ops._C_ascend.* / torch_npu.*），是生产环境在
#                           昇腾 NPU 上的高性能实现路径。
#   - msa_m3_triton.py    : Triton 参考实现（面向 A3 等 Atlas 910 系列）。包含索引
#                           打分、topk 选择、块稀疏 GQA 注意力（prefill + split-K decode）
#                           的完整 Triton kernel。
#   - msa_m3_triton_a5.py : 面向 A5（Atlas 950 系列）的 Triton 变体实现，适配 A5 的
#                           KV cache 布局与 AI Core 特性，decode 路径延迟更低。
#
# 两套实现的关系：NPU 算子与 Triton 实现功能等价（算子接口一一对应），运行时由
# msa_m3.py 依据设备类型（A3/A5）与硬件能力（如 FP8注意力支持）动态选择。
# =====================================================================================
