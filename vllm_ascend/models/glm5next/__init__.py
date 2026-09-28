# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# GLM-5.Next（智谱 GLM 新一代模型）昇腾 NPU 适配包的入口文件。
#
# 包结构说明（本 __init__.py 仅保留 SPDX 版权声明，不做任何显式导出，
# 各子模块通过相对导入/绝对导入被 model.py 等上层文件引用）：
#   - config.py                  : 模型配置（文本/视觉/顶层三段式 PretrainedConfig）
#   - model.py                   : 主模型（DecoderLayer / MoE / MHC 超连接 / 前向与权重加载）
#   - attention.py               : MLA（多头潜在注意力）+ 稀疏索引器（KPool top-k）
#   - kda.py                     : KDA 线性注意力层（Gated DeltaNet 变体，含因果卷积）
#   - mtp.py                     : MTP 多 token 预测（投机解码草稿模型）
#   - multimodal.py              : 视觉塔（ViT）+ 多模态处理器注册
#   - processor.py               : vLLM 原生图像/视频处理器（token 预算、帧采样）
#   - kv_cache.py / cache_config.py / cache_views.py
#                                : KV cache 的层注册、分组调度与物理视图（混合层缓存组织）
#   - sparse_attn_indexer_kpool.py: KPool 稀疏注意力索引器的张量编排
#   - ops/                       : NPU 自定义算子封装（因果卷积、KDA、MHC、状态读写等）
#
# 在 vllm-ascend 插件架构中的位置：
#   vllm_ascend/models/glm5next 是 vLLM 硬件插件（vllm-ascend）为 GLM-5.Next
#   提供的模型实现目录；上游 vLLM 通过 Registry 机制发现
#   Glm5NextForCausalLM / Glm5NextForConditionalGeneration 两个架构。
# =============================================================================
