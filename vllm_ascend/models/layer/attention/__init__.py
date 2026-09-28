"""昇腾注意力层子包（vllm_ascend.models.layer.attention）。

提供 vLLM Attention 层抽象在昇腾 NPU 上的定制实现。

为何需要自定义 Attention 层（vLLM Attention 抽象原理）：
- vLLM v1 通过 AttentionBackend 插件机制隔离硬件差异：ModelRunner 与
  调度器统一面向"注意力层 + 注意力后端"抽象编程——GPU 上对接
  FlashAttention / FlashInfer 等，NPU 上则必须提供调用昇腾算子
  （CANN 的 flash attention / 稀疏注意力算子）的后端与层实现。
- 注意力层负责"申报"：通过 get_kv_cache_spec() 告知 KV Cache 管理器
  本层需要的缓存布局（块大小、头数、head_size、dtype），管理器据此
  规划 NPU 显存（HBM）池；通过 get_attn_backend() 告知运行时选用哪个
  后端算子；并把自身注册进 vLLM 的 static_forward_context，使
  torch.compile 计算图中的自定义算子能按层名找到本层与 KV cache。

本子包内容（layer.py）：
- DSAAttention: DeepSeek-V4 的 MLA（多头潜在注意力）+ DSA（稀疏注意力）
  层，按 KV 压缩比选择昇腾后端（C4 / C128 / SWA，实现见
  vllm_ascend/attention/dsa_v1.py）。
- get_dsv4_block_sizes / dsv4_block_sizes / DSV4_BLOCK_SIZES*:
  不同硬件能力（是否支持压缩缓存、A5 BF16 KV）下的 KV Cache 分块
  大小查找表。
"""
