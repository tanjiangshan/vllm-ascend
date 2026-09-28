# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# 【文件职责】V4.1 稀疏注意力 indexer：索引投影、量化 QLI 打分与跨层候选选择。
#
# 【稀疏注意力原理】V4.1 长上下文层不全量计算注意力，而是两阶段筛选：
#   1) indexer 用一个"小侧注意力"（n_heads 个 index 头，每头 width 维）对
#      历史 KV 打分：query 由 q_lora 潜向量经 wq_b 投影得到，key 是源层
#      潜向量经 wk 投影 + 部分旋转位置编码(RoPE) 后动态量化成 INT8 的
#      "index K"（每 compress_ratio 个 token 一条，存于专用 k_cache）；
#   2) 量化闪电索引算子 npu_quant_lightning_indexer_v3（QLI）在分页 INT8 K 上
#      计算 query·key 打分并选出 TopK 个 token 位置（index_topk），
#      主注意力（DSA）只在这些位置上做完整 MLA 注意力。
#
# 【跨层候选(candidate)机制】candidate_source 层先粗选 topk_blocks 个 block
# （候选块 ID，不是位置），其后的消费层用这些块过滤自己的 TopK 搜索空间，
# 逐层缩小范围（mode 1=产出候选 / 2=消费候选过滤 / 3=不过滤）。
#
# 【NPU 适配点】
#   - QLI 是 CANN 原生算子（torch.ops._C_ascend），支持分页布局 PA_BBND；
#   - index K 用 torch_npu.npu_dynamic_quant 做 INT8 动态量化（FP16 scale）；
#   - 部分旋转用 torch.ops._C_ascend.inplace_partial_rotary_mul（interleave 模式）；
#   - QLI 的 op 元数据经 DeviceMetadata 机制异步下发，需 wait_for_device_metadata。
# =============================================================================
"""V4.1 index projections, quantized QLI and cross-layer candidate selection."""

import torch
import torch_npu
from torch import nn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear

from vllm_ascend.attention.dsa_v41 import (
    DeepseekV41CacheLayer,
    scatter_cache_sk,
)
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.ops.triton.prepare_indexer_indices import prepare_indexer_indices
from vllm_ascend.ops.triton.quantize_indexer_query import quantize_indexer_query
from vllm_ascend.worker.device_metadata import (
    DeviceMetadataStage,
    wait_for_device_metadata,
)


class DeepseekV41Indexer(nn.Module):
    """Small side attention that selects compressed KV positions.

    All index heads are replicated on each TP rank for the correctness path,
    so every rank produces identical sparse indices without an all-reduce.

    【中文说明】稀疏注意力的"小侧注意力"：为长上下文层挑选参与完整注意力
    计算的 token 位置（TopK 选择器）。继承 nn.Module，无上游 vLLM 对应类。
    为了正确性路径不做 TP 切分——所有 index 头在每个 TP rank 上复制，
    每个 rank 独立算出相同的稀疏索引，省掉一次 all-reduce 通信。
    """

    def __init__(
        self,
        config,
        owns_k,
        vllm_config,
        prefix,
        compress_ratio,
        quant_config=None,
    ):
        """初始化 indexer。

        参数:
            config: 归一化后的 V4.1 文本配置（用 index_n_heads/index_head_dim/
                index_topk/qk_rope_head_dim 等）。
            owns_k: 本层是否为 index K 的"拥有层"（即 kv source 层）。
                只有拥有层才建 wk/k_norm/k_cache；消费层直接读源层 cache。
            vllm_config: 引擎配置（读 block_size 建 cache 规格）。
            prefix: 参数名前缀。
            compress_ratio: 压缩比，index K 每 ratio 个 token 存一条。
            quant_config: 量化配置（wq_b/weights_proj 可被量化）。
        """
        super().__init__()
        self.owns_k = owns_k
        self.compress_ratio = compress_ratio
        self.n_heads = int(config.index_n_heads)
        self.width = int(config.index_head_dim)
        self.rope_width = int(config.qk_rope_head_dim)
        self.index_topk = int(config.index_topk)
        # softmax_scale: index 注意力的 1/sqrt(width)；weights_scale 额外除以
        # sqrt(n_heads)，因为多头的分数还要对 head 维求平均（乘性融合权重）。
        self.softmax_scale = self.width**-0.5
        self.weights_scale = self.softmax_scale * self.n_heads**-0.5
        # wq_b: q_lora_rank -> n_heads*width，从 query 潜向量升维出各 index 头。
        self.wq_b = ReplicatedLinear(
            config.q_lora_rank,
            self.n_heads * self.width,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
            return_bias=False,
        )
        # weights_proj: hidden_size -> n_heads，从原始隐状态产生每头的融合权重
        # （决定各 index 头分数在聚合时的权重）。
        self.weights_proj = ReplicatedLinear(
            config.hidden_size,
            self.n_heads,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
            return_bias=False,
        )
        if owns_k:
            # 只有 index K 拥有层才创建以下模块：
            # wk: head_dim(MLA 潜向量) -> width(index K)；k_norm 做尺度归一。
            self.wk = nn.Linear(
                config.head_dim,
                self.width,
                bias=False,
                dtype=torch.bfloat16,
            )
            self.k_norm = RMSNorm(self.width, eps=config.rms_norm_eps, dtype=torch.bfloat16)
            # 专用 INT8 k_cache：storage_block_size = block_size // compress_ratio
            # （每 ratio 个逻辑 token 的 index K 占一个存储 token 位）；
            # scale_dim=1 表示 scale 与 K 同布局存储（按行一组 FP16 缩放）。
            self.k_cache = DeepseekV41CacheLayer(
                vllm_config,
                f"{prefix}.k_cache",
                AscendMLAAttentionSpec(
                    block_size=vllm_config.cache_config.block_size,
                    num_kv_heads=1,
                    head_size=self.width,
                    dtype=torch.int8,
                    tokens_per_state=compress_ratio,
                    model_version="deepseek_v41",
                    storage_block_size=(vllm_config.cache_config.block_size // compress_ratio),
                    scale_dim=1,
                    scale_dtype=torch.float16,
                ),
            )

    @staticmethod
    def _output(linear, value):
        """取线性层输出张量。原理: vLLM 的 Linear 层返回 (output, bias) 元组，
        ReplicatedLinear 也遵循该约定，这里统一剥出 output。"""
        output = linear(value)
        return output[0] if isinstance(output, tuple) else output

    def update_keys(self, latent, slots, cos, sin):
        """Publish source-owned index K before latent is RoPE'd as long KV."""
        """【中文说明】源层把新 token 的 index K 写入 cache。注意时机：必须在
        潜向量被 RoPE 变换成长 KV 之前调用（index K 与长 KV 用不同的旋转配置）。

        参数:
            latent: [tokens, head_dim] 源层 MLA 潜向量（未 RoPE）。
            slots: [tokens, 2] 或 [tokens] 槽位映射（block_idx, block 内偏移）。
            cos/sin: index K 专用的 RoPE 余弦/正弦表。
        算法步骤:
            1) wk 投影 + k_norm 归一得到 [tokens, 1, width] 的 key；
            2) 对最后 rope_width 维做部分旋转（interleave 交织模式，原地操作）；
            3) npu_dynamic_quant 动态量化成 INT8 + FP16 scale；
            4) scatter_cache_sk 把量化和 scale 分别写入 k_cache 两个平面。
        """
        if not self.owns_k or latent.shape[0] == 0:
            return
        # 步骤1: 投影 + 归一，reshape 成 [tokens, 1, width]（单"头"布局）。
        key = self.k_norm(self.wk(latent)).view(-1, 1, self.width)
        # 步骤2: 部分旋转——只旋转每个 key 的最后 rope_width 维（partial_slice
        # 指明旋转区间），rotary_mode="interleave" 表示偶奇交织式旋转。
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            key.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.width - self.rope_width, self.width],
        )
        key = key.squeeze(1)
        # 步骤3: NPU 动态量化：INT8 码 + 每行一个 FP16 缩放因子。
        quantized, scale = torch_npu.npu_dynamic_quant(key, dst_type=torch.int8)
        # 步骤4: kv_cache[0] 是 (k 平面, scale 平面) 二元组，分别 scatter。
        k_cache, scale_cache = self.k_cache.kv_cache[0]
        scatter_cache_sk(k_cache, slots, quantized)
        scatter_cache_sk(
            scale_cache,
            slots,
            scale.unsqueeze(-1).to(torch.float16),
        )

    def select(
        self,
        hidden_states,
        qr,
        positions,
        cos,
        sin,
        source_cache,
        source_metadata,
        *,
        is_candidate_source,
        uses_candidate_filter,
        candidate_topk_blocks,
        candidate_block_size,
        candidates,
        output_indices=None,
    ):
        """Score index K, optionally filter blocks, then return position TopK."""
        """【中文说明】完整选择流程：投影 query → 旋转 → 融合权重 → 委托
        select_projected 执行 QLI。

        参数:
            hidden_states: [tokens, hidden_size] 当前层输入（算融合权重用）。
            qr: [tokens, q_lora_rank] query 潜向量。
            positions: [tokens] token 位置（把块内偏移换算成绝对位置）。
            cos/sin: index query 的 RoPE 表。
            source_cache: (k 平面, scale 平面) 源层 index K cache。
            source_metadata: 注意力元数据（block_table/seq_lens 等）。
            is_candidate_source: 本层是否产出候选块（candidate source）。
            uses_candidate_filter: 本层是否用上游候选块过滤搜索空间。
            candidate_topk_blocks: 候选块数量上限。
            candidate_block_size: 候选块大小（token 数）。
            candidates: 输入候选块 ID [tokens, 1, candidate_topk_blocks] INT32。
            output_indices: 可选输出缓冲（ACL Graph 固定地址复用）。
        返回:
            (selected, candidates): selected 是 [tokens, topk] 的绝对位置索引；
            candidates 是本层产出的候选块 ID（仅 candidate source 有效）。
        算法步骤:
            1) wq_b 把 qr 升维并 unflatten 成 [tokens, n_heads, width]；
            2) 对最后 rope_width 维部分旋转；
            3) weights_proj 算各头融合权重并乘 weights_scale（FP32 精度）；
            4) 委托 select_projected 做量化 QLI TopK。
        """
        # 步骤1: 潜向量 -> n_heads 个 index 头的 query。
        query = self._output(self.wq_b, qr).unflatten(-1, (self.n_heads, self.width))
        # 步骤2: query 部分旋转（与 update_keys 中 key 的旋转方式配对）。
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            query.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.width - self.rope_width, self.width],
        )
        # 步骤3: 融合权重转 FP32 并乘缩放（含 1/sqrt(width) 与 1/sqrt(heads)）。
        weights = self._output(self.weights_proj, hidden_states)
        weights = weights.float() * self.weights_scale

        return self.select_projected(
            query,
            weights,
            positions,
            source_cache,
            source_metadata,
            is_candidate_source=is_candidate_source,
            uses_candidate_filter=uses_candidate_filter,
            candidate_topk_blocks=candidate_topk_blocks,
            candidate_block_size=candidate_block_size,
            candidates=candidates,
            output_indices=output_indices,
        )

    def select_projected(
        self,
        query,
        weights,
        positions,
        source_cache,
        source_metadata,
        *,
        is_candidate_source,
        uses_candidate_filter,
        candidate_topk_blocks,
        candidate_block_size,
        candidates,
        output_indices=None,
    ):
        """Run QLI V2 on paged INT8 K; candidates are block IDs, not positions.

        Source and consumer share [tokens, 1, candidate_topk_blocks] INT32
        block IDs only within this forward. Query quantization and position
        ordering stay outside the native QLI/candidate operator.

        【中文说明】在分页 INT8 index K 上运行原生 QLI（量化闪电索引）算子。
        候选机制中的"候选"是 block ID（块号）而非 token 位置；源层与消费层
        只在同一次前向内共享候选块 ID。query 量化和位置换算都留在算子外
        （Python/Triton 侧），算子只负责核心打分与 TopK。

        算法步骤:
            1) 空请求/空 cache 的边界检查，返回全 -1 的占位结果；
            2) 量化 query（INT8 + scale），转 FP16 融合权重；
            3) 组装 QLI 公共参数（变长序列元数据、分页布局、压缩比）；
            4) 等待 DeviceMetadata 就绪（QLI op 元数据异步下发机制）；
            5) 按 candidate 模式调用 npu_quant_lightning_indexer_v3；
            6) prepare_indexer_indices 把选中位置换算成绝对 token 位置。
        """
        candidate_shape = (query.shape[0], 1, candidate_topk_blocks)
        topk = self.index_topk
        # 步骤1a: 无 query token（如纯 prefill 前的空批）——返回空 TopK；
        # 若本层是候选源，还要产出全 -1 的候选块占位。
        if query.shape[0] == 0:
            selected = torch.full((0, topk), -1, dtype=torch.int32, device=query.device)
            if is_candidate_source:
                candidates = torch.full(candidate_shape, -1, dtype=torch.int32, device=query.device)
            return selected, candidates
        # 步骤1b: cache 里还没有任何历史 token——同样返回空结果。
        if source_metadata.max_cache_seq_len == 0:
            selected = torch.full((query.shape[0], 0), -1, dtype=torch.int32, device=query.device)
            if is_candidate_source:
                candidates = torch.full(candidate_shape, -1, dtype=torch.int32, device=query.device)
            return selected, candidates

        # 步骤2: 量化 index query；融合权重转 FP16（与 QLI 算子的输入约定一致）。
        quantized_query, query_scale = quantize_indexer_query(query)
        weights = weights.to(torch.float16)
        # 步骤2b: key 与 key_scale 分页取自源层 cache；squeeze(-1) 去掉 scale
        # 尾维，保持 Hybrid cache 的页步长（page stride）一致。
        key, key_scale = source_cache
        key_scale = key_scale.squeeze(-1)  # Preserve the Hybrid cache page stride.
        cu_seqlens_q = source_metadata.query_start_loc
        seqused_k = source_metadata.cache_seq_lens
        residual = source_metadata.cmp_residual
        # 步骤3: QLI 公共参数——TND(query 布局)/PA_BBND(key 分页布局)、
        # mask_mode=3（因果+变长）、cmp_ratio=压缩比等。
        common = dict(
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            cmp_residual_k=residual,
            max_seqlen_q=source_metadata.max_query_len,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=self.compress_ratio,
        )
        # 步骤4: QLI 的 op 元数据由设备侧异步准备，此处等待其就绪（按元数据
        # 对象 id 索引），避免宿主-设备同步阻塞。
        op_metadata = source_metadata.qli_metadata
        wait_for_device_metadata(DeviceMetadataStage.INDEXER, id(op_metadata))
        # 步骤5: candidate 模式选择——1=本层产出候选块；2=用上游候选块过滤；
        # 3=不参与候选机制（纯 TopK）。
        mode = 1 if is_candidate_source else 2 if uses_candidate_filter else 3
        selected, _, candidate_out = torch.ops._C_ascend.npu_quant_lightning_indexer_v3(
            quantized_query,
            key,
            weights,
            query_scale,
            key_scale,
            topk,
            2,
            block_table=source_metadata.block_table,
            metadata=op_metadata,
            candidate_topk_index=candidates if uses_candidate_filter else None,
            candidate_mode=mode,
            candidate_topk_blocks=candidate_topk_blocks,
            candidate_block_size=candidate_block_size,
            **common,
        )
        # 步骤6: QLI 返回的是块内相对位置，prepare_indexer_indices 依据
        # positions 与压缩比换算成绝对 token 位置；output 缓冲可选复用。
        selected = prepare_indexer_indices(
            selected.squeeze(1),
            positions,
            self.compress_ratio,
            output=output_indices,
        )
        # 候选块只有候选源层会更新输出；消费层原样透传输入候选。
        return selected, candidate_out if is_candidate_source else candidates
