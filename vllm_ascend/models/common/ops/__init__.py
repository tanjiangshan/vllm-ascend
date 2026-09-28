# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ops shared across model implementations."""
# ============================================================================
# 中文说明（补充注释，原英文 docstring 保留在上）：
#
# 本子包存放"跨模型共享"的 NPU 算子/辅助函数。与某个具体模型强绑定的
# 算子应放在各自模型的子包中（如 models/minimax_m3/ops、models/glm5next/ops）。
#
# 当前包含：
# - sequence_parallel.py: 序列并行（Sequence Parallelism, SP）工具函数，
#   提供 sp_all_gather / sp_reduce_scatter / sp_shard / sp_padding_mask，
#   在 TP（张量并行）组内沿 token（序列）维度做切分与聚合，供
#   deepseek_v4 / deepseek_v41 / kimi_k3 / glm5next 等模型在
#   "TP + SP"混合并行模式下复用。
#
# 架构背景：vllm-ascend 是 vLLM 的硬件插件，模型文件通过继承/组合上游
# vLLM 层并替换 NPU 特定实现来适配昇腾硬件；本包抽取的公共算子正是
# 多个模型适配过程中沉淀下来的共用代码。
# ============================================================================
