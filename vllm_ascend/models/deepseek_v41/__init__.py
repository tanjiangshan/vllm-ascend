# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# DeepSeek V4.1 昇腾 NPU 适配包（vllm-ascend 插件的模型实现子包）。
#
# 【包结构导览】（本包在 vllm-ascend 插件架构中的位置：
#   vllm_ascend/ → models/ → deepseek_v41/）
#   ├── model.py        文本主干：MLA 风格注意力、MoE 路由、mHC 矩阵超连接、
#   │                   engram 记忆装配、源共享(source-shared)混合 KV cache 图。
#   ├── vl_model.py     多模态包装器：注册 MM processor，组合视觉塔与语言模型。
#   ├── vision.py       ViT 视觉塔 + Aligner 对齐器（2D RoPE、空间合并下采样）。
#   ├── dspark.py       DSpark 投机解码草稿模型（EAGLE3 接口，3 层 mtp 草稿块）。
#   ├── indexer.py      稀疏注意力 indexer：小侧注意力 + INT8 QLI TopK 选 token。
#   ├── compressor.py   C2 环形压缩器：把源层 KV 压成长上下文环形状态(ratio 1/2)。
#   ├── cache_config.py V4.1 混合 KV cache 布局：层元组分组 + 全局 block 池分配。
#   └── engram/         engram n-gram 记忆子包：
#       ├── common.py       平台无关的门控/词元资格判定
#       ├── hash_state.py   n-gram 哈希状态（Triton-Ascend kernel）
#       ├── embedding.py    INT8+组32缩放的 engram 嵌入表（TP/EDP 头分片）
#       ├── layer.py        engram 投影与门控层（写回主网络残差流）
#       ├── npu.py          NPU 侧存储：设备表 gather、主机 UVA 卸载、共享内存
#       └── parallel.py     EDP(专家数据并行)哈希/行交换辅助
#
# 【与上游 vLLM 的关系】本包不直接修改上游模型文件，而是自带一套独立的
# Ascend 实现（插件架构推荐做法之一），复用上游 vllm.models.deepseek_v41.common
# 中的公共组件（engram 布局、多模态预处理等）。
# =============================================================================
"""DeepSeek V4.1 construction components (not yet a runnable model)."""
