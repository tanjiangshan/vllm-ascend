# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# 【文件职责】C2 环形压缩器（ring compressor）：V4.1 长上下文机制的一部分。
#
# 【原理】V4.1 的"源层"（kv_source_layer）用 MLA 把 KV 压成潜向量后，还需要
# 进一步压缩成长上下文状态。compressor 把源层输出投影为固定宽度(width=head_dim)
# 的状态向量：
#   - ratio=1 路径：无压缩，wkv(bf16) 逐 token 投影 + RMSNorm（forward()）；
#   - ratio=2 路径：每 2 个 token 压成 1 条状态。wkv/wgate 用 FP32 计算门控，
#     输出写入持久化环形缓冲 _ring_pooled，同时维护一个 FP32 环形状态 cache
#     (CircularBufferSpec, head_size=2*width) 供 pool_projected() 读取。
#
# 【张量流】decoder 层的隐状态 x → wkv 投影 → (ratio=2) compressor_from_projected
# Triton kernel 做 ring 池化 → RMSNorm → 得到压缩状态，供 DSA 稀疏注意力
# 的长上下文分支消费。
#
# 【NPU 适配点】
#   - 池化走 Triton-Ascend kernel（compressor_from_projected），按 cube 核数
#     (AI Core 的 Cube 单元) 切分并行度；
#   - _ring_pooled 在显存探测(memory profiling)之前注册为持久 buffer，
#     使其占用计入 cache 预算而不是探测后再补分配。
# =============================================================================
"""FP32 C2 ring compressor, ratio-1 path, and fused RMS normalization."""

import torch
from torch import nn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.v1.kv_cache_interface import CircularBufferSpec

from vllm_ascend.attention.dsa_v41 import DeepseekV41CacheLayer
from vllm_ascend.models.deepseek_v41.cache_config import STATE_RING_ROWS


class DeepseekV41Compressor(nn.Module):
    """C2 环形压缩器模块：把源层隐状态压成 ring 状态供长上下文注意力使用。

    继承自 torch.nn.Module（纯本地实现，无上游 vLLM 对应类）。
    两条路径:
        ratio=1: 仅 wkv(BF16) 投影 + RMSNorm，逐 token 输出（不压缩）；
        ratio=2: wkv/wgate 均 FP32，配合环形池化 kernel 把 2 token 压成 1 状态。
    """

    def __init__(self, config, ratio, vllm_config=None, prefix="compressor"):
        """初始化压缩器。

        参数:
            config: 归一化后的 DeepSeek V4.1 文本配置。
            ratio: 压缩比（每 ratio 个 token 产出 1 条状态）。
            vllm_config: 引擎配置，用于确定池化容量与状态 cache 规格；
                独立(非引擎)测试可不传。
            prefix: 层参数名前缀（权重加载与 cache 注册用）。
        """
        super().__init__()
        self.ratio = ratio
        # width 即压缩后的状态宽度，等于 MLA 的 head_dim（潜向量维度）。
        self.width = config.head_dim
        dim = config.hidden_size
        # ratio=2 时用 FP32 计算保证门控数值精度；ratio=1 用 BF16 走常规路径。
        self.wkv = nn.Linear(dim, self.width, bias=False, dtype=torch.float32 if ratio == 2 else torch.bfloat16)
        self.norm = RMSNorm(self.width, eps=config.rms_norm_eps, dtype=torch.bfloat16)
        if ratio == 2:
            # ratio=2 专属：门控投影 wgate（FP32），输出与 wkv 结果做逐元素门控。
            self.wgate = nn.Linear(dim, self.width, bias=False, dtype=torch.float32)
            # Allocate persistent output before memory profiling, so its footprint
            # is included in the cache budget rather than added after allocation.
            # 【中文】持久化输出缓冲：必须在显存探测前注册，占用才会被计入
            # cache 预算（探测后再分配会导致可用 block 数虚高、OOM）。
            if vllm_config is not None:
                capacity = getattr(vllm_config.scheduler_config, "max_num_batched_tokens", 4096)
                # 语法点: register_buffer(persistent=False)——注册为模块缓冲但
                # 不进 state_dict（权重保存/加载时跳过）。
                self.register_buffer(
                    "_ring_pooled",
                    torch.empty(capacity, self.width, dtype=torch.bfloat16, device=self.wkv.weight.device),
                    persistent=False,
                )
            # Standalone unfused-reference tests may supply pages explicitly.
            # 【中文】FP32 环形状态 cache：每 block 32 行、head_size=2*width
            # （一条状态存 2 份投影结果），由 Ascend KV allocator 绑定实际张量。
            if vllm_config is not None:
                self.state_cache = DeepseekV41CacheLayer(
                    vllm_config,
                    f"{prefix}.state_cache",
                    CircularBufferSpec(
                        block_size=STATE_RING_ROWS,
                        num_kv_heads=1,
                        head_size=2 * self.width,
                        dtype=torch.float32,
                        head_size_v=0,
                    ),
                )

    def prepare_ring_compressor(self, max_tokens, device):
        """Resolve ring-compressor hardware before capture."""
        """【中文说明】在 ACL Graph 捕获前解析硬件参数。原理: 查询当前 NPU 的
        cube 核数量，供 Triton 池化 kernel 决定并行切分粒度（每核一段 ring）。"""
        from vllm_ascend.ops.triton.compressor.compressor_triton import _cube_core_num

        self._ring_num_cores = _cube_core_num()

    def pool_projected(self, kv, scores, metadata):
        """执行 ratio=2 的环形池化。

        参数:
            kv: [tokens, width] wkv 投影结果（BF16 输出缓冲复用）。
            scores: [tokens, width] wgate 门控分数。
            metadata: 注意力元数据，需含 c2_ring_metadata（ring 读写位置）与
                max_query_len。
        返回:
            [tokens, width] RMSNorm 后的池化状态（写入 self._ring_pooled）。
        原理: compressor_from_projected 在一个 Triton kernel 里完成——
            按核分段读取状态 cache、用 scores 门控融合新旧状态、写回
            ring 缓冲并输出池化结果，避免多次 HBM 往返。
        """
        from vllm_ascend.ops.triton.compressor.compressor_triton import compressor_from_projected

        pooled = compressor_from_projected(
            kv,
            scores,
            # 状态 cache 的 kv_cache 是单元素列表，squeeze(-2) 去掉多余的
            # 头维得到 [blocks, block_size, 2*width] 的环形历史。
            self.state_cache.kv_cache[0].squeeze(-2),
            metadata.c2_ring_metadata,
            # 复用持久缓冲的前 kv.shape[0] 行，保证 ACL Graph 地址固定。
            self._ring_pooled[: kv.shape[0]],
            max_query_len=metadata.max_query_len,
            num_cores=self._ring_num_cores,
        )
        return self.norm(pooled)

    def forward(self, x):
        """Project an uncompressed source; ratio-2 uses ``pool_projected``."""
        """【中文说明】ratio=1 前向：x [tokens, hidden_size] → wkv 投影 → RMSNorm，
        直接返回（不做池化）。ratio=2 的压缩路径不走 forward，由注意力实现
        直接调用 pool_projected()。"""
        return self.norm(self.wkv(x))
