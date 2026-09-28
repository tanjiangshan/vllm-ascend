"""NPU 定制网络层包（vllm_ascend.models.layer）。

本包存放"可作为独立网络层复用"的 NPU 定制模块，位于 vllm_ascend/models
之下，与各具体模型实现（deepseek_v4/、deepseek_v41/、kimi_k3.py 等）并列。

定位说明：
- 上游 vLLM 的通用网络层位于 vllm.model_executor.layers，与硬件无关；
- vllm-ascend 作为硬件插件，在本包中提供需要深度适配昇腾 NPU 的层实现，
  供多个模型共享（避免在每个模型文件里复制一份）；
- 当前包含 attention/ 子包：DeepSeek-V4 稀疏注意力（DSA）在昇腾上的
  注意力层实现（DSAAttention，多头潜在注意力 MLA + 稀疏注意力）及其
  KV Cache 分块规格定义，被 vllm_ascend/ops/dsa.py 中的
  AscendDeepseekSparseAttention 包装器使用，其分块表常量
  （DSV4_BLOCK_SIZES 等）也被 deepseek_v4 / deepseek_v41 模型引用。
"""
