# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""模型公共组件包（vllm_ascend.models.common）。

本包存放"跨模型共享"的公共组件，供 vllm_ascend/models 下各个模型实现复用，
避免同一逻辑在多个模型文件中重复出现。

在 vllm-ascend 插件架构中的位置：
- vllm_ascend/models/ 下的模型文件（deepseek_v4/、deepseek_v41/、kimi_k3.py、
  glm5next/ 等）是 vLLM 硬件插件对上游模型的 NPU 定制实现，通过
  vllm_ascend/models/__init__.py 中的 register_model() 注册进 vLLM 的
  ModelRegistry。
- 本包（common）聚合这些模型实现中"与具体模型无关"的公共部分。当前包含：
  - ops/sequence_parallel.py: 序列并行（Sequence Parallelism）辅助算子
    （sp_all_gather / sp_reduce_scatter / sp_shard / sp_padding_mask），
    被 DeepSeek-V4/V4.1、Kimi-K3、GLM5Next 等模型的 TP+SP 混合并行路径，
    以及 DSA 上下文并行（attention/context_parallel/dsa_cp.py）使用。

设计约定：本包只放通用组件；某个模型专属的算子应放在该模型自己的子包中
（如 minimax_m3/ops、glm5next/ops），以保持 common 包的通用性与低耦合。
"""
