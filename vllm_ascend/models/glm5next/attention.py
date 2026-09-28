# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# GLM-5.Next 的注意力模块（MLA 多头潜在注意力 + KPool 稀疏索引器）。
#
# 核心内容：
#   1. Indexer           —— KPool 稀疏注意力索引器：为每个 token 从历史中选出
#                           top-k 个"关键"上下文 token（以 k 池为粒度压缩存储），
#                           供稀疏 MLA（DSA，DeepSeek 风格稀疏注意力）只对被选
#                           token 做完整注意力计算。
#   2. Glm5NextMLAAttention —— 完整的 MLA 注意力层：借鉴 DeepSeek-V2/V3 的
#                           低秩 KV 压缩（kv_lora_rank）+ 解耦 RoPE 分量，
#                           并把上游 vLLM 的 MultiHeadLatentAttentionWrapper
#                           组合进来，实现"索引器选 token + MLA 注意力"的融合。
#
# NPU 适配点：
#   - 索引器的前向计算不在 Python 端实现，而是委托给
#     vllm_ascend.attention.indexer_kpool 中的 Ascend 后端（自定义算子）。
#   - k_cache / tail_cache 分别用 Glm5NextIndexerCache / Glm5NextTailCache
#     注册到 vLLM 的静态前向上下文，由模型运行时统一分配 NPU 显存。
# =============================================================================

import torch
from torch import nn
from vllm.config import (
    CacheConfig,
    VllmConfig,
)
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mla import (
    MLAModules,
    MultiHeadLatentAttentionWrapper,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding, get_rope
from vllm.model_executor.models.deepseek_v2 import (
    DeepSeekV2FusedQkvAProjLinear,
    yarn_get_mscale,
)

from vllm_ascend.models.glm5next.config import Glm5NextConfig
from vllm_ascend.models.glm5next.kv_cache import (
    Glm5NextIndexerCache,
    Glm5NextTailCache,
    get_kpool_tail_ring_capacity,
)


class Indexer(nn.Module):
    """KPool 稀疏注意力索引器（GLM-5.Next v32 配置专用）。

    作用：在 MLA 注意力之前，用一个轻量的"索引头"对历史 KV 做相关性打分，
    选出 top-k 个候选 token 位置，输出到共享的 topk_indices_buffer，
    供后续稀疏注意力算子（SFA，Sparse Flash Attention）只计算被选中的键值对。

    原理（KPool 压缩）：
    - 每 ``index_kpool``（默认 4）个连续 token 的索引 K 向量被"池化"为 1 个
      存储条目（带可学习的 APE 绝对位置编码 + gate 门控），因此缓存体积缩小
      kpool 倍；未凑满一池的"尾巴" token 存放在 per-request 的 TailCache 环中。
    - topk 以"池"为粒度执行（select_k = index_topk // index_kpool）。

    注意：本类的 forward() 刻意抛异常——实际计算走 IndexerWrapper 的
    Ascend 后端（自定义 NPU 算子），本类只承载权重与缓存注册。
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Glm5NextConfig,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        prefix: str = "",
    ):
        """初始化索引器。

        参数：
            vllm_config: vLLM 全局配置。
            config: GLM-5.Next 模型配置。
            hidden_size: 模型隐藏维度（如 4096）。
            q_lora_rank: MLA 的 Q 低秩维度（如 1536），索引器 Q 从它投影。
            quant_config: 量化配置（FP8 等），可为 None。
            cache_config: KV cache 配置（含 block_size）。
            topk_indices_buffer: 共享的 top-k 结果缓冲区，
                形状 [max_num_batched_tokens, buffer_width]，int32。
            prefix: 层名前缀（用于权重命名与静态上下文注册）。
        """
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        # self.indexer_cfg = config.attn_module_list_cfg[0]["attn_index"]
        # Indexer is only constructed for v32 configs, where these sparse-indexer
        # fields are guaranteed populated; narrow away the `int | None` declared
        # on Glm5NextConfig for the optional-indexer case.
        assert config.index_topk is not None
        assert config.index_n_heads is not None
        assert config.index_head_dim is not None
        assert config.index_kpool is not None
        assert cache_config is not None
        assert topk_indices_buffer is not None
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.index_kpool = config.index_kpool
        self.q_lora_rank = q_lora_rank  # 1536

        # kpool
        # 步骤1: KPool 压缩的可学习参数。
        # compress_ape 形状 [kpool, head_dim]，FP32：绝对位置编码（APE），
        #   池内第 j 个 token 的 K 会被加上 ape[j] 再做门控加权，保留池内顺序信息。
        # compress_gate 形状 [head_dim, hidden_size]，BF16：门控投影，
        #   由 hidden_states 生成与各池内条目做内积的"查询式"门控分数。
        # 注意：参数名刻意不带 ".weight" 后缀以匹配 checkpoint 命名，
        # torch.mm 直接消费其 [head_dim, hidden_size] 形状。
        self.index_kpool_compress_ape = nn.Parameter(torch.zeros(self.index_kpool, self.head_dim, dtype=torch.float32))
        # Keep the checkpoint name ``index_kpool_compress_gate`` without a
        # ``.weight`` suffix. torch.mm consumes its [head_dim, hidden_size] shape.
        self.index_kpool_compress_gate = nn.Parameter(torch.empty(self.head_dim, hidden_size, dtype=torch.bfloat16))

        # no tensor parallel, just replicated
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        # Fused wk + weights_proj: single GEMM producing [head_dim + n_head].
        # FP8 wk weights are upcasted to BF16 during loading to maintain fusion.
        self.wk_weights_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.head_dim, self.n_head],
            bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.wk_weights_proj",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.softmax_scale = self.head_dim**-0.5
        self.scale_fmt = None
        self.quant_block_size = self.head_dim
        self.topk_indices_buffer = topk_indices_buffer
        # Completed pools store BF16 vectors without quantization scales.
        # 步骤2: 注册两类缓存层。
        # k_cache：已"池化完成"的压缩 K 缓存（BF16，无量化 scale），
        # 以 AscendMLAAttentionSpec 描述，tokens_per_state=kpool 表示压缩比。
        self.k_cache = Glm5NextIndexerCache(
            head_dim=self.head_dim,
            dtype=torch.bfloat16,
            cache_role="indexer",
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
            compress_ratio=self.index_kpool,
        )
        # Request-owned FP32 K/gate ring retains the incomplete pool.
        # tail_cache：请求私有的 FP32 环形缓存，保存"未凑满一池"的原始 K
        # 与门控分数；环容量 = kpool + 投机解码 lookahead（见 kv_cache.py）。
        self.tail_cache = Glm5NextTailCache(
            head_dim=self.head_dim,
            dtype=torch.float32,
            prefix=f"{prefix}.tail_cache",
            compress_ratio=self.index_kpool,
            ring_capacity=get_kpool_tail_ring_capacity(vllm_config, self.index_kpool),
        )
        self.prefix = prefix

    def get_ascend_indexer_backend_cls(self):
        """返回 Ascend 侧索引器后端类（Glm5NextKPoolIndexerBackend）。

        延迟导入（lazy import）使本模型模块在进程启动阶段不依赖共享
        算子注册表，避免循环导入并缩短导入时间。
        """
        # Lazy import keeps the model module independent from the shared ops
        # registry during process startup.
        from vllm_ascend.attention.indexer_kpool import (
            Glm5NextKPoolIndexerBackend,
        )

        return Glm5NextKPoolIndexerBackend

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions,
        rotary_emb,
    ) -> torch.Tensor:
        """占位 forward：实际稀疏索引计算由 Ascend 后端执行。

        参数（仅作接口示意）：
            hidden_states: [num_tokens, hidden_size] 隐藏状态。
            qr: RoPE 后的索引查询。
            positions: [num_tokens] token 位置。
            rotary_emb: 旋转位置编码模块。

        原理：真正的计算路径是 IndexerWrapper 在其 Ascend 后端中调用
        SparseAttnIndexerKpool（见 sparse_attn_indexer_kpool.py）完成
        "tail 压缩写缓存 + top-k 选择"，因此这里直接抛错以防误用。
        """
        del hidden_states, qr, positions, rotary_emb
        raise RuntimeError("GLM-Next Indexer must run through IndexerWrapper's Ascend backend.")


class Glm5NextMLAAttention(nn.Module):
    """GLM-5.Next 的 MLA（Multi-head Latent Attention，多头潜在注意力）层。

    结构（DeepSeek-V2/V3 风格 + GLM-5.Next 定制）：
    - KV 低秩压缩：kv_a_proj 把 hidden 压到 kv_lora_rank 维的"潜在 KV"，
      推理时 cache 只存压缩向量（+RoPE 分量），大幅节省 KV cache 显存。
    - Q 侧可选低秩（q_lora_rank）：q_a_proj -> q_a_layernorm -> q_b_proj。
    - qk_nope_head_dim 与 qk_rope_head_dim 分离：RoPE 只作用于 rope 分量，
      保证压缩后的 KV 仍能施加位置相关的注意力。
    - mla_nope（skip_rope）配置：GLM 部分层完全不加 RoPE。
    - v32 配置额外挂载 Indexer（KPool 稀疏索引器）实现稀疏注意力。

    实现方式：本类只负责"组装权重 + 配置"，前向委托给上游 vLLM 的
    MultiHeadLatentAttentionWrapper（其中融合了稀疏索引器调度）。
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Glm5NextConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        input_size: int | None = None,
        skip_rope: bool | None = False,
    ) -> None:
        """初始化 MLA 注意力层。

        参数：
            vllm_config: vLLM 全局配置。
            config: Glm5NextConfig 模型配置。
            hidden_size: 隐藏维度 H。
            num_heads: 注意力头总数（跨 TP 切分）。
            qk_nope_head_dim: 每头非 RoPE 维度（潜在 KV 解压出的部分）。
            qk_rope_head_dim: 每头 RoPE 维度。
            v_head_dim: 每头 V 维度。
            q_lora_rank: Q 低秩秩数；None 表示不做 Q 低秩压缩。
            kv_lora_rank: KV 低秩秩数（压缩后的潜在 KV 维度）。
            max_position_embeddings: 最大位置数（RoPE 用）。
            cache_config: KV cache 配置。
            quant_config: 量化配置。
            prefix: 层名前缀。
            topk_indices_buffer: 稀疏索引共享缓冲区（v32 配置必需）。
            input_size: 投影输入维度；None 时取 hidden_size
                （Eagle3 + MLA 草稿模型会传入不同的输入维度）。
            skip_rope: 是否完全跳过 RoPE（GLM mla_nope 配置）。
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size

        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        # Use input_size for projection input dimensions if provided,
        # otherwise default to hidden_size (used in Eagle3 Deepseek with MLA)
        # 步骤1: 确定投影输入维度。普通层 = hidden_size；
        # MTP/Eagle3 草稿层可能拼接了上一层的隐状态，输入维度不同。
        proj_input_size = input_size if input_size is not None else self.hidden_size

        # 步骤2: 构建 a 侧投影（低秩压缩入口）。
        # 有 Q 低秩时融合 q_a_proj + kv_a_proj 为单个 GEMM（省一次矩阵乘）；
        # 无 Q 低秩时仅做 kv_a_proj_with_mqa（MQA 式共享 KV）。
        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
                proj_input_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
            )
        else:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                proj_input_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )

        # 步骤3: Q 侧投影。q_lora_rank 非空：q_a_layernorm + q_b_proj（低秩两段式）；
        # 否则直接 q_proj 一步到全部头。ColumnParallelLinear 按注意力头做 TP 切分。
        if self.q_lora_rank is not None:
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            self.q_proj = ColumnParallelLinear(
                proj_input_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
        # 步骤4: KV 解压投影与输出投影。
        # kv_a_layernorm 归一化压缩的潜在 KV；kv_b_proj 把潜在 KV 解压为
        # 各头的 nope-K 与 V。FP8 checkpoint 中 kv_b_proj 保持 BF16（无量化 scale），
        # 因此 quant_config=None。
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=None,  # kv_b_proj stays BF16 in the FP8 checkpoint
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # 步骤5: RoPE 旋转位置编码。
        # 非 skip_rope 时根据 rope_parameters 构建编码器；Yarn 类型会把
        # rope_type 改写为 deepseek_yarn / deepseek_llama_scaling，
        # 并按 yarn_get_mscale 调整注意力缩放因子（长外推补偿）。
        if not skip_rope:
            assert config.rope_parameters is not None
            if config.rope_parameters["rope_type"] != "default":
                config.rope_parameters["rope_type"] = (
                    "deepseek_yarn"
                    if config.rope_parameters.get("apply_yarn_scaling", True)
                    else "deepseek_llama_scaling"
                )

            self.rotary_emb: RotaryEmbedding | None = get_rope(
                qk_rope_head_dim,
                max_position=max_position_embeddings,
                rope_parameters=config.rope_parameters,
                is_neox_style=False,
            )

            if (
                config.rope_parameters["rope_type"] != "default"
                and config.rope_parameters["rope_type"] == "deepseek_yarn"
            ):
                mscale_all_dim = config.rope_parameters.get("mscale_all_dim", False)
                scaling_factor = config.rope_parameters["factor"]
                mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
                self.scaling = self.scaling * mscale * mscale
        else:
            self.rotary_emb = None

        # 步骤6: v32 配置判定——index_topk 非空即为稀疏（v32）配置，
        # 需要构建索引器 RoPE 与 Indexer 实例；否则为纯稠密 MLA。
        self.is_v32 = config.index_topk is not None

        if self.is_v32:
            # 索引器有独立的 RoPE；其排列方式由 indexer_rope_interleave 决定
            # （是否交替取维度，对应 is_neox_style 取反）。
            self.indexer_rope_emb: RotaryEmbedding | None = get_rope(
                qk_rope_head_dim,
                max_position=max_position_embeddings,
                rope_parameters=config.rope_parameters,
                is_neox_style=not config.indexer_rope_interleave,
            )
            # The sparse indexer projects from the MLA q-lora rank, which is
            # always set for v32 MLA configs; narrow away the `int | None`.
            assert q_lora_rank is not None
            self.indexer: Indexer | None = Indexer(
                vllm_config,
                config,
                hidden_size,
                q_lora_rank,
                quant_config,
                cache_config,
                topk_indices_buffer,
                f"{prefix}.indexer",
            )

        else:
            self.indexer_rope_emb = None
            self.indexer = None

        # 步骤7: 把上述子模块打包成 MLAModules（上游 vLLM 的可插拔 MLA
        # 组件容器），再交给 MultiHeadLatentAttentionWrapper 统一执行前向。
        # 语法点：条件表达式 `x if cond else None` 按是否有 Q 低秩二选一填充。
        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj if self.q_lora_rank is not None else None,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa if self.q_lora_rank is None else None,
            q_a_layernorm=self.q_a_layernorm if self.q_lora_rank is not None else None,
            q_b_proj=self.q_b_proj if self.q_lora_rank is not None else None,
            q_proj=self.q_proj if self.q_lora_rank is None else None,
            indexer=self.indexer,
            indexer_rotary_emb=self.indexer_rope_emb,
            is_sparse=self.is_v32,
            topk_indices_buffer=topk_indices_buffer,
        )

        self.mla_attn = MultiHeadLatentAttentionWrapper(
            self.hidden_size,
            self.num_local_heads,
            self.scaling,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.q_lora_rank,
            self.kv_lora_rank,
            mla_modules,
            cache_config,
            quant_config,
            prefix,
            skip_topk=False,
            fuse_qkv_rmsnorm=True,
        )
        # The pluggable MLA wrapper owns the AttentionLayerBase registered in
        # the static forward context. Publish GLM-Next's cache contract on that
        # layer, matching the dedicated cache layer used by the source branch,
        # so the model runner can remain model agnostic.
        # 步骤8: 在 MLA wrapper 持有的 AttentionLayerBase 上发布 GLM-Next 的
        # 缓存契约标记：model_version="glm5_next" 让 cache_config.py /
        # cache_views.py 能识别该层；indexes_kv_by_block_stride=True 表示
        # 该层的 KV 槽位映射按"块步长"重排（KPool 压缩后的索引规则）。
        # 这样模型运行时（model runner）可以保持模型无关。
        mla_cache_layer = self.mla_attn.mla_attn
        mla_cache_layer.model_version = "glm5_next"
        mla_cache_layer.indexes_kv_by_block_stride = True

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """MLA 前向：委托给 MultiHeadLatentAttentionWrapper。

        参数：
            hidden_states: [num_tokens, hidden_size] 输入隐藏状态。
            positions: [num_tokens] 每个 token 的位置（RoPE 用）。

        返回：
            [num_tokens, hidden_size] 注意力输出。

        说明：wrapper 内部会先运行稀疏索引器（v32 配置）选出 top-k 上下文，
        再执行 MLA 注意力与 KV cache 读写（含 NPU 自定义算子）。
        """
        # The wrapper also runs the sparse indexer before MLA attention.
        return self.mla_attn(positions, hidden_states)
