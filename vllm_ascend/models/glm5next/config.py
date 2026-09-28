# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# GLM-5.Next 模型配置定义（三段式结构，继承 HuggingFace PretrainedConfig）。
#
#   1. Glm5NextTextConfig  —— 文本塔配置：混合层布局（layer_types 指定每层是
#      "linear_attention"(KDA) 还是 "deepseek_sparse_attention"(稀疏 MLA)）、
#      MLA 低秩参数、MoE 专家参数、KDA 线性注意力头参数、KPool 稀疏索引器
#      参数、MHC 超连接（multi-Head hyper-Connection）参数等。
#   2. Glm5NextVisionConfig —— 视觉塔配置：ViT patch 大小、层数、融合参数等。
#   3. Glm5NextConfig      —— 顶层多模态配置：组合 text_config 与 vision_config，
#      并通过 __setattr__/__getattribute__ 魔术方法把文本配置的属性"镜像"到
#      顶层，使纯文本调用方可以像访问 Glm5NextTextConfig 一样访问顶层配置。
#
# 设计要点：GLM-5.Next 是混合架构模型（类似 Kimi K3 / DeepSeek V3.2 的思路）：
#   - 部分层用 KDA（Kimi Delta Attention，Gated DeltaNet 类线性注意力），
#     无 KV cache、常数复杂度、用循环状态（state）代替；
#   - 其余层用稀疏 MLA（DeepSeek Sparse Attention 风格 + KPool 索引器），
#     只对 top-k 相关上下文做完整注意力；
#   - MLP 侧为 MoE（默认 288 路由专家 + 1 共享专家，每 token 选 7 个）。
# =============================================================================

from transformers.configuration_utils import PretrainedConfig


class Glm5NextTextConfig(PretrainedConfig):
    """GLM-5.Next 文本模型配置。

    继承 transformers.PretrainedConfig（语法点：HF 配置基类，提供序列化、
    保存/加载、from_pretrained 等能力）。model_type / base_config_key 用于
    HF AutoConfig 按嵌套子配置解析多模态 checkpoint。

    配置分组：
    - 基础：vocab_size / hidden_size / num_hidden_layers / 头数等；
    - MLA：mla, q_lora_rank, kv_lora_rank, qk_nope/rope_head_dim, v_head_dim；
    - MoE：n_routed_experts, num_experts_per_token, n_shared_experts 等；
    - 混合布局：layer_types（每层注意力类型）、mlp_layer_types（每层 MLP 类型）；
    - KDA 线性注意力：linear_head_dim / linear_num_heads / linear_conv_kernel_dim
      / linear_lower_bound（有界门控下界）；
    - KPool 稀疏索引：index_topk / index_n_heads / index_head_dim / index_kpool 等；
    - MHC 超连接：mhc, mhc_num_residual_streams, mhc_tau, sinkhorn 迭代数等。
    """

    model_type = "glm5_next_text"
    # base_config_key: HF 多模态配置中嵌套文本配置的键名
    base_config_key = "text_config"
    # 推理时需要忽略的 checkpoint 字段（历史 KV，仅训练用）
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        model_type="glm5_next_text",
        vocab_size: int = 154880,
        hidden_size: int = 4096,
        head_dim: int | None = None,
        intermediate_size: int = 12288,
        num_hidden_layers: int = 45,
        num_attention_heads: int = 64,
        num_key_value_heads: int | None = None,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-5,
        pad_token_id: int | None = 151329,
        bos_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        rope_parameters: dict | None = None,
        max_position_embeddings: int = 1048576,
        tie_word_embeddings: bool = False,
        moe_intermediate_size: int = 2048,
        moe_renormalize: bool = True,
        scoring_func: str = "sigmoid",
        n_routed_experts: int | None = 288,
        num_experts_per_token: int = 7,
        n_shared_experts: int = 1,
        routed_scaling_factor: float = 2.5,
        topk_method: str | None = None,
        first_k_dense_replace: int = 0,
        moe_layer_freq: int = 1,
        use_grouped_topk: bool = True,
        n_group: int = 1,
        topk_group: int = 1,
        mla: bool = True,
        q_lora_rank: int | None = 1536,
        kv_lora_rank: int | None = 512,
        qk_nope_head_dim: int = 256,
        qk_rope_head_dim: int = 0,
        v_head_dim: int | None = 256,
        mla_nope: bool | None = True,
        num_nextn_predict_layers: int = 1,
        # Per-layer layout: "linear_attention" | "deepseek_sparse_attention"
        # layer_types：逐层注意力类型（线性注意力 / DeepSeek 式稀疏注意力）
        layer_types: list[str] | None = None,
        # Per-layer MLP: "dense" | "sparse"
        # mlp_layer_types：逐层 MLP 类型（dense 稠密 / sparse 即 MoE）
        mlp_layer_types: list[str] | None = None,
        # Linear-attention (KDA) head config (flattened from the old
        # linear_attn_config dict).
        # KDA 线性注意力头配置（从旧的 linear_attn_config 字典扁平化而来）
        linear_head_dim: int = 128,
        linear_num_heads: int = 64,
        linear_conv_kernel_dim: int = 4,
        linear_lower_bound: float = -5.0,
        index_head_dim: int | None = None,
        index_topk: int | None = None,
        index_n_heads: int | None = None,
        index_dsa_use_layernorm: bool = True,
        index_kpool_compress: bool = True,
        # Every ``index_kpool`` indexer K entries pool into one stored entry
        # (compress_ratio). The 300B checkpoint ships 4; topk runs at pool
        # granularity (select_k = index_topk // index_kpool).
        # KPool 压缩比：每 index_kpool 个索引器 K 条目池化为 1 个存储条目。
        # 300B checkpoint 值为 4；topk 以池粒度执行（select_k = topk // kpool）。
        index_kpool: int | None = 4,
        index_kpool_always_select_tail: bool = True,
        indexer_rope_interleave: bool = False,
        mhc: bool | None = True,
        mhc_num_residual_streams: int = 4,
        hc_eps: float | None = 1e-06,
        mhc_tau: float = 0.05,
        hres_vwnstyle: bool | None = True,
        mhc_no_norm_weight: bool | None = False,
        mhc_sinkhorn_iterations: int | None = 20,
        mhc_post_mult_value: float | None = 2.0,
        swiglu_limit: float | None = None,
        logit_scale: float = 1.0,
        **kwargs,
    ):
        """初始化文本配置；兼容多种 checkpoint 字段拼写并做合法性校验。

        语法点：**kwargs 收集未知字段；`int | None` 是 PEP 604 联合类型注解。
        """
        # Preserve checkpoint field names and local aliases because their
        # consumers use different spellings.
        # 步骤1: 字段别名兼容——checkpoint 可能用旧拼写，这里统一映射：
        #   num_experts_per_tok  -> num_experts_per_token
        #   norm_topk_prob       -> moe_renormalize
        #   hc_mult              -> mhc_num_residual_streams
        #   hc_sinkhorn_iters    -> mhc_sinkhorn_iterations
        num_experts_per_token = kwargs.get("num_experts_per_tok", num_experts_per_token)
        moe_renormalize = kwargs.get("norm_topk_prob", moe_renormalize)
        mhc_num_residual_streams = kwargs.get("hc_mult", mhc_num_residual_streams)
        mhc_sinkhorn_iterations = kwargs.get("hc_sinkhorn_iters", mhc_sinkhorn_iterations)
        # Checkpoint ships ``mla_use_nope`` (not ``mla_nope``); without this
        # alias self.mla_nope silently stays at the param default.
        mla_nope = kwargs.get("mla_use_nope", mla_nope)
        # Checkpoints ship the KDA head config as the ``linear_attn_config``
        # dict (head_dim / num_heads / short_conv_kernel_size /
        # gate_lower_bound) rather than the flattened top-level fields; fold it
        # in so the trained values are read instead of the param defaults
        # (which only match this checkpoint by coincidence).
        # 步骤2: 折叠嵌套的 linear_attn_config 字典——checkpoint 把 KDA 头配置
        # 存为字典（head_dim / num_heads / short_conv_kernel_size /
        # gate_lower_bound），这里展开到顶层字段，保证读到训练值而非默认值。
        linear_cfg = kwargs.get("linear_attn_config") or {}
        if linear_cfg:
            linear_head_dim = linear_cfg.get("head_dim", linear_head_dim)
            linear_num_heads = linear_cfg.get("num_heads", linear_num_heads)
            linear_conv_kernel_dim = linear_cfg.get("short_conv_kernel_size", linear_conv_kernel_dim)
            linear_lower_bound = linear_cfg.get("gate_lower_bound", linear_lower_bound)

        # 步骤3: v32 稀疏索引器与 MHC 的硬性约束校验——当前实现只支持
        # 特定的开关组合（其余组合直接 NotImplementedError，防止静默降级出错）。
        if index_topk is not None:
            if index_dsa_use_layernorm is not True:
                raise NotImplementedError("GLM-5.3 sparse indexer requires index_dsa_use_layernorm=True")
            if index_kpool_compress is not True:
                raise NotImplementedError("GLM-5.3 sparse indexer requires index_kpool_compress=True")
            if index_kpool_always_select_tail is not True:
                raise NotImplementedError("GLM-5.3 sparse indexer requires index_kpool_always_select_tail=True")

        if mhc:
            if hres_vwnstyle is not True:
                raise NotImplementedError("GLM-5.3 mHC requires hres_vwnstyle=True")
            if mhc_no_norm_weight not in (False, None):
                raise NotImplementedError("GLM-5.3 mHC requires mhc_no_norm_weight=False")

        self.model_type = model_type
        # 步骤4: 逐项落盘到实例属性（HF 配置对象的常规做法）。
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_parameters = rope_parameters

        # mla config
        self.mla = mla
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mla_nope = mla_nope
        # moe config
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_token = num_experts_per_token
        self.moe_renormalize = moe_renormalize
        self.n_shared_experts = n_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.topk_method = topk_method
        self.scoring_func = scoring_func
        assert self.scoring_func in ("softmax", "sigmoid")
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.use_grouped_topk = use_grouped_topk
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_nextn_predict_layers = num_nextn_predict_layers

        # Per-layer attention / MLP layout. Normalize mlp_layer_types from
        # first_k_dense_replace when the new-schema field is absent so layer
        # construction sees a consistent layout (mirrors cohere2_moe).
        # 步骤5: 层布局归一化——若 checkpoint 未提供 mlp_layer_types，则由
        # first_k_dense_replace（前 k 层用 dense，其余用 MoE）推导，
        # 使层构建逻辑总能看到一致的布局（与 cohere2_moe 相同的做法）。
        self.layer_types = layer_types
        if mlp_layer_types is None:
            n = self.num_hidden_layers
            if first_k_dense_replace is not None:
                mlp_layer_types = ["dense"] * first_k_dense_replace + ["sparse"] * (n - first_k_dense_replace)
            else:
                mlp_layer_types = ["sparse"] * n
        self.mlp_layer_types = mlp_layer_types

        # Linear-attention (KDA) head config.
        self.linear_head_dim = linear_head_dim
        self.linear_num_heads = linear_num_heads
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_lower_bound = linear_lower_bound

        # dsa index config
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.index_n_heads = index_n_heads
        self.index_dsa_use_layernorm = index_dsa_use_layernorm
        self.index_kpool_compress = index_kpool_compress
        self.index_kpool = index_kpool
        self.index_kpool_always_select_tail = index_kpool_always_select_tail
        self.indexer_rope_interleave = indexer_rope_interleave

        # mhc config
        self.mhc = mhc
        self.mhc_num_residual_streams = mhc_num_residual_streams
        self.mhc_tau = mhc_tau
        self.hres_vwnstyle = hres_vwnstyle
        self.hc_eps = hc_eps
        self.mhc_no_norm_weight = mhc_no_norm_weight
        self.mhc_sinkhorn_iterations = mhc_sinkhorn_iterations
        self.mhc_post_mult_value = mhc_post_mult_value

        self.swiglu_limit = swiglu_limit
        self.logit_scale = logit_scale

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def is_mla(self):
        """是否为 MLA 架构：任一 MLA 相关字段被设置即视为 MLA。

        语法点：@property 装饰器把方法变成只读属性，访问形如 config.is_mla。
        """
        return (
            self.q_lora_rank is not None
            or self.kv_lora_rank is not None
            or self.qk_nope_head_dim is not None
            or self.qk_rope_head_dim is not None
            or self.v_head_dim is not None
            or self.mla_nope is True
        )

    @property
    def is_moe(self):
        """是否为 MoE 模型：设置了路由专家数即视为 MoE。"""
        return self.n_routed_experts is not None

    @property
    def is_linear_attn(self) -> bool:
        """是否存在 KDA 线性注意力层（layer_types 中任一层为 linear_attention）。"""
        return self.layer_types is not None and any(t == "linear_attention" for t in self.layer_types)

    def is_kda_layer(self, layer_idx: int):
        """判断第 layer_idx 层是否为 KDA 线性注意力层。

        参数：
            layer_idx: 解码层下标（从 0 开始）。

        返回：
            bool：该层在 layer_types 中标记为 "linear_attention" 时为 True。
        """
        return (
            self.layer_types is not None
            and layer_idx < len(self.layer_types)
            and self.layer_types[layer_idx] == "linear_attention"
        )

    @property
    def layers_block_type(self):
        """把逐层类型映射为 vLLM 混合显存统计能识别的块类型字符串。

        原理：vLLM 的 hybrid accounting（get_num_layers_by_block_type）用
        "linear_attention" / "attention" 两类块来推导 KDA 状态缓存与
        KV cache 的配额；这里把所有非线性注意力变体（稀疏 MLA 等）都
        归并为 "attention"，仅保留 KDA 层的 "linear_attention" 标记。
        """
        # Map the schema's per-layer types onto the block strings vLLM's hybrid
        # accounting (get_num_layers_by_block_type) recognizes: linear-attention
        # layers stay "linear_attention"; every other attention variant collapses
        # to "attention".
        if self.layer_types is None:
            return ["attention"] * self.num_hidden_layers
        return ["linear_attention" if t == "linear_attention" else "attention" for t in self.layer_types]


class Glm5NextVisionConfig(PretrainedConfig):
    """GLM-5.Next 视觉塔（ViT）配置。

    继承 PretrainedConfig。描述视觉编码器的 patch 化参数（patch_size /
    temporal_patch_size / spatial_merge_size）、ViT 深度/宽度、以及
    PatchMerger 的投影瓶颈宽度（projection_intermediate_size）。
    """

    model_type = "glm5_next_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth: int = 24,
        hidden_size: int = 1024,
        hidden_act: str = "silu",
        image_size: int = 448,
        intermediate_size: int = 4096,
        num_heads: int = 16,
        out_hidden_size: int = 4096,
        projection_intermediate_size: int = 10240,
        in_channels: int = 3,
        initializer_range: float = 0.02,
        patch_size: int = 14,
        rms_norm_eps: float = 1e-5,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        attention_dropout: float = 0.0,
        attention_bias: bool = True,
        swiglu_limit: float | None = None,
        **kwargs,
    ):
        """初始化视觉配置；参数含义见各类属性赋值处注释。"""
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.image_size = image_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.out_hidden_size = out_hidden_size
        # GLM-5.3-Flash merger bottleneck width (absent from the generic
        # GLM-OCR vision config); the tower uses it as the PatchMerger
        # context_dim instead of text_config.intermediate_size.
        # PatchMerger 的瓶颈宽度（通用 GLM-OCR 视觉配置没有此字段）；
        # 视觉塔用它作为 PatchMerger 的 context_dim，
        # 而不是 text_config.intermediate_size。
        self.projection_intermediate_size = projection_intermediate_size
        self.in_channels = in_channels
        self.initializer_range = initializer_range
        self.patch_size = patch_size
        # GLM-5.3-Flash checkpoints ship vision_config.rms_norm_eps = 1e-5,
        # but the vision tower was trained with 1e-6. Serving with 1e-5 drifts
        # the RMSNorm and produces repetitive/degraded image descriptions, so
        # force the trained value regardless of the checkpoint field.
        # 关键修正：checkpoint 里的 rms_norm_eps=1e-5 是错的，视觉塔实际用
        # 1e-6 训练；若按 1e-5 服务会导致 RMSNorm 漂移、图像描述重复退化，
        # 因此这里无视 checkpoint 字段强制写死训练值 1e-6。
        self.rms_norm_eps = 1e-6
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.swiglu_limit = swiglu_limit


class Glm5NextConfig(PretrainedConfig):
    """GLM-5.Next 顶层（多模态）配置：组合视觉与文本两个子配置。

    继承 PretrainedConfig。sub_configs 声明子配置类，供 HF AutoConfig 把
    checkpoint 中嵌套的 "vision_config" / "text_config" 字典实例化为
    Glm5NextVisionConfig / Glm5NextTextConfig。

    核心机制（属性镜像）：
    - __setattr__：对不属于 _UNMIRRORED_KEYS 的字段，若 text_config 中
      已存在同名属性，则同步写入 text_config（双向一致）。
    - __getattribute__：读取时若 text_config（实例或类）有该属性则转发，
      使顶层对象"看起来就是"文本配置（纯文本调用方无感）。
    """

    model_type = "glm5_next"
    sub_configs = {
        "vision_config": Glm5NextVisionConfig,
        "text_config": Glm5NextTextConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id: int = 154854,
        video_token_id: int = 154855,
        image_start_token_id: int = 154830,
        image_end_token_id: int = 154831,
        video_start_token_id: int = 154832,
        video_end_token_id: int = 154833,
        **kwargs,
    ):
        """初始化顶层配置。

        参数：
            text_config: 文本子配置（dict / Glm5NextTextConfig / None）。
            vision_config: 视觉子配置（dict / Glm5NextVisionConfig / None）。
            image_token_id 等: 多模态特殊 token 的词表 id。
        """
        # Init super() first so base-class defaults don't clobber text-config
        # values set below (PretrainedConfig has many text-related defaults
        # that differ from Glm5NextTextConfig).
        # 步骤1: 先调用父类构造——否则基类的文本相关默认值会覆盖下方
        # 设置的 text_config 字段。
        super().__init__(**kwargs)

        # 步骤2: 实例化视觉子配置。dict -> 用 sub_configs 中的类构造；
        # None -> 全默认；否则直接使用传入对象。
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        # 步骤3: 实例化文本子配置。向后兼容：扁平化 checkpoint（没有嵌套
        # text_config）时把顶层 kwargs 折入 Glm5NextTextConfig。
        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            # Backward compatibility: a flat top-level checkpoint (no nested
            # text_config) folds its text fields into Glm5NextTextConfig.
            self.text_config = self.sub_configs["text_config"](**kwargs)
        else:
            self.text_config = text_config

        # 步骤4: 记录多模态特殊 token id。
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id

        # Mirror attention implementation recursively onto sub-configs.
        # 步骤5: 把注意力实现方式（如 FA/FLASHINFER）递归镜像到子配置。
        # 语法点：kwargs.pop(key, None) 取出并删除该键，不存在时返回 None。
        self._attn_implementation = kwargs.pop("attn_implementation", None)

    # Config-metadata fields that belong to the top-level (multimodal) config
    # and must NOT be mirrored onto text_config: ``architectures`` /
    # ``torch_dtype`` differ between the top-level config and the text
    # sub-config, and mirroring them makes the top-level ``architectures``
    # silently read back as None (PretrainedConfig initializes both to None),
    # which then fails model-class resolution ("No model architectures are
    # specified").
    # 不参与镜像的字段清单：这些元数据字段只属于顶层（多模态）配置。
    # 若把 architectures / torch_dtype 镜像到 text_config，顶层的
    # architectures 会被静默读回 None，导致模型类解析失败。
    _UNMIRRORED_KEYS = [
        "_name_or_path",
        "model_type",
        "dtype",
        "torch_dtype",
        "architectures",
        "_attn_implementation_internal",
    ]

    def __setattr__(self, key, value):
        """写属性时的镜像逻辑：text_config 已有的字段同步写入 text_config。

        语法点：
        - super().__getattribute__("__dict__") 绕过本类重写的
          __getattribute__ 直接读实例字典，避免无限递归；
        - 海象运算符 `:=` 在表达式内赋值并同时参与判断。
        """
        unmirrored = type(self)._UNMIRRORED_KEYS
        if (
            (text_config := super().__getattribute__("__dict__").get("text_config")) is not None
            and key not in unmirrored
            and key in text_config.__dict__
        ):
            setattr(text_config, key, value)
        else:
            super().__setattr__(key, value)

    def __getattribute__(self, key):
        """读属性时的转发逻辑：text_config 的实例属性与类属性/方法都转发。

        原理：同时检查 text_config.__dict__（实例属性）与
        type(text_config).__dict__（类定义的 property/方法），使扁平化
        纯文本 checkpoint（model_type "glm5_next"、无嵌套 text_config）
        也能透明访问 is_moe / is_kda_layer / layers_block_type 等。
        """
        unmirrored = type(self)._UNMIRRORED_KEYS
        if "text_config" in super().__getattribute__("__dict__") and key not in unmirrored:
            text_config = super().__getattribute__("text_config")
            # Forward both instance attributes AND class-defined properties/
            # methods of the text config, so a flat text-only checkpoint
            # (model_type "glm5_next", no nested text_config) sees is_moe /
            # is_kda_layer / layers_block_type like a Glm5NextTextConfig.
            if key in text_config.__dict__ or key in type(text_config).__dict__:
                return getattr(text_config, key)

        return super().__getattribute__(key)
