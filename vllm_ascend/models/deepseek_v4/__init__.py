# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""DeepSeek V4 系列模型的昇腾 NPU 适配实现包。

本包是 vllm-ascend 硬件插件中 DeepSeek V4 家族（文本模型 + V4-VL 多模态 +
MTP/DSpark 投机采样草稿）的模型实现集合。与上游 vLLM 的关系：
vllm-ascend 不直接修改上游模型文件，而是通过插件式的独立模型目录提供
NPU 适配版本，注册到 vLLM 模型注册表后由平台层按硬件选择加载。

模块导览（详见各模块 docstring）：
- ``model.py``       : 语言主干 DeepseekV4Model / AscendDeepseekV4ForCausalLM，
                       含 DSA 稀疏 MLA 注意力（q/kv 低秩压缩 + RoPE 解耦）、
                       DeepSeek V4 MoE（细粒度路由专家 + 共享专家 + hash 路由层）、
                       Hyper-Connections（hc_* 参数）与 EPLB 专家负载均衡支持。
- ``indexer.py``     : DSA 稀疏注意力索引器（DeepSeek V3.2 式闪电 indexer），
                       量化 q/k 后用闪电注意力为每个 query 选出 TopK 历史 token，
                       供注意力层只对这些 token 计算精确注意力。
- ``compressor.py``  : KV 压缩器（compress_ratio=4/128），把历史 KV 递归压缩为
                       低维 state（kv_state + score_state），是 indexer 与
                       长程注意力的数据来源。
- ``mtp.py``         : MTP 多 Token 预测草稿模型（串联草稿层），
                       配合 verify 阶段实现投机采样加速。
- ``dspark.py``      : DSpark 块级草稿模型（block drafter），一次草拟整块
                       token，配合 Markov head / confidence head 估计接受率。
- ``vision.py``      : V4-VL 视觉塔（ViT + 2D RoPE + Aligner 空间合并投影）。
- ``mm_preprocess.py``: 多模态预处理：图像 patch 化、动态分辨率、哨兵 token
                       （sentinel）块布局构造与 prompt 替换。
- ``vl_model.py``    : V4-VL 顶层包装 AscendDeepseekV4ForConditionalGeneration，
                       组合视觉塔与 AscendDeepseekV4ForCausalLM 语言主干。

典型张量流（多模态推理）：
图像 PIL -> mm_preprocess（patch/网格/排列） -> vision.py（ViT+Aligner 得到
图像嵌入） -> vl_model.embed_input_ids（哨兵位替换与合并） -> model.py 解码器
（compressor 压缩 KV -> indexer 产 TopK 索引 -> 稀疏注意力 -> MoE） -> logits。
"""
