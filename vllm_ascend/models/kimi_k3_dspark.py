# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 MLA DSpark draft model for Ascend.

Kimi K3 的 DSpark 草稿模型（MLA 注意力版）—— 昇腾 NPU 适配实现。

【DSpark + MLA 组合原理】
DSpark 是块级投机解码草稿器：目标模型把若干层的隐状态（aux hidden states）
交给草稿，草稿用 ``context_proj`` 投影成上下文，随后对一整块候选 token
并行前向打分。K3 草稿的注意力是 MLA（多头潜在注意力）：
- Q 侧低秩两段式压缩（q_lora_rank）；
- KV 压缩成单个潜向量（kv_lora_rank），KV Cache 只存压缩向量，显存开销极小；
- 本实现的亮点 ``precompute_and_store_context_kv``：把目标隐状态直接投影成
  草稿层的共享 KV 并**预写入草稿 KV cache**，草稿块前向时无需重复计算上下文。

【适配思路】
上游参考实现位于 ``vllm.models.kimi_k3.nvidia.dspark_mla``（NVIDIA 版）。
与本项目其它模型一致：显式调用 ``nn.Module.__init__`` 搭建同构容器，
把注意力换成 ``AscendKimiMLAAttention``（昇腾算子路径），MLP 复用
``KimiMLP``（硬件无关），forward 逻辑尽量继承上游。
"""

from collections.abc import Iterable

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
# DSparkMarkovHead：马尔可夫头 —— 用上一个草稿 token 的嵌入给 logits 加
# 低秩先验偏置（建模 bigram 统计），提升草稿命中率。与 Qwen3 DSpark 共享实现。
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    _merge_multimodal_embeddings,
    get_draft_quant_config,
    maybe_prefix,
)
from vllm.models.kimi_k3.amd.linear import KimiMLP
from vllm.models.kimi_k3.nvidia.dspark_mla import (
    K3DSparkDecoderLayer as UpstreamK3DSparkDecoderLayer,
)
from vllm.models.kimi_k3.nvidia.dspark_mla import (
    K3DSparkForCausalLM as UpstreamK3DSparkForCausalLM,
)
from vllm.models.kimi_k3.nvidia.dspark_mla import (
    K3DSparkModel as UpstreamK3DSparkModel,
)

from vllm_ascend.models.kimi_k3 import (
    AscendKimiMLAAttention,
)
from vllm_ascend.models.qwen3_dspark import align_draft_weights
# 昇腾 RoPE 工具：按位置批量计算 MLA 用的 cos/sin 表。
from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_mla


def _uses_causal_draft_attention(config) -> bool:
    """探测草稿注意力是否使用因果（causal）掩码。

    DFlash/DSpark 草稿可以选双向（非因果）注意力 —— 因为草稿块内 token
    的"未来"本来就是草稿要猜的对象，双向信息不构成作弊，反而提高接受率。
    配置来源两级探测：优先 dflash_config["causal"]（字典），
    否则回退到 full_attention_causal（扁平字段）。
    """
    dflash_config = getattr(config, "dflash_config", None)
    if isinstance(dflash_config, dict) and "causal" in dflash_config:
        return bool(dflash_config["causal"])
    return bool(getattr(config, "full_attention_causal", False))


class AscendK3DSparkDecoderLayer(UpstreamK3DSparkDecoderLayer):
    """DSpark 草稿解码层（昇腾版）。

    继承上游 ``K3DSparkDecoderLayer`` 复用类契约；构造时把硬编码的
    NVIDIA 注意力/MLP 换成 Ascend 注意力 + 硬件无关 MLP。
    结构：input_layernorm → self_attn(MLA) → post_attention_layernorm → mlp，
    标准 pre-norm 残差解码层。
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config,
        layer_idx: int,
        start_layer_id: int,
        prefix: str,
    ) -> None:
        # The upstream constructor hard-codes NVIDIA attention and MLP
        # components. Keep its class contract while constructing the Ascend
        # equivalents below.
        # （上游构造函数硬编码 NVIDIA 注意力与 MLP；这里保持类契约不变，
        #   换用下方构建的 Ascend 等价组件。）
        nn.Module.__init__(self)
        # 草稿模型可独立配置量化（与目标模型解耦），例如草稿用更低精度。
        quant_config = get_draft_quant_config(vllm_config)
        # 层命名：草稿层编号 = 目标层数 + 草稿层下标，与 checkpoint 命名空间对齐。
        layer_prefix = maybe_prefix(
            prefix,
            f"layers.{start_layer_id + layer_idx}",
        )
        self.self_attn = AscendKimiMLAAttention(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            use_output_gate=False,
            use_rope=True,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=f"{layer_prefix}.self_attn",
            # 双向注意力开关：非因果模式允许草稿块内互相看见（见 _uses_causal_draft_attention）。
            non_causal_multi_token_decode=not _uses_causal_draft_attention(config),
            disable_mlapo=True,
        )
        self.mlp = KimiMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{layer_prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """草稿层前向：norm → attention → norm → mlp，返回 (输出, 残差)。

        hidden_states: [num_tokens, hidden_size]（已展平的 token 流）。
        residual: 上层传来的残差分支；None 表示本层是第一层（残差即输入本身）。
        vLLM 的 RMSNorm 支持双参调用：一次完成"残差相加 + 归一化"融合计算，
        返回 (归一化结果, 更新后的残差)，等价于 x = x + residual; y = norm(x)。
        """
        if residual is None:
            # 第一层：残差 = 输入，归一化输入作为子层输入。
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states,
                residual,
            )
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states,
            residual,
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class AscendK3DSparkModel(UpstreamK3DSparkModel):
    """DSpark 草稿模型主体（昇腾版）。

    组件：context_proj（目标多层隐状态 → 草稿上下文投影）、若干草稿层、
    final_norm、markov_head（bigram 先验）。embed_tokens 通常为 None
    （与目标模型共享词嵌入，由顶层动态注入）。
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int,
        prefix: str,
    ) -> None:
        # The upstream constructor hard-codes CUDA/NVIDIA attention and decoder
        # classes.  Initialize only the module base, then keep the upstream
        # model methods while constructing the Ascend-specific components.
        # （上游硬编码 NVIDIA 组件；只初始化模块基类，保留上游模型方法，
        #   自行构建 Ascend 专属组件。）
        nn.Module.__init__(self)
        # 投机解码配置必须存在（草稿模型只在投机解码场景下构建）。
        assert vllm_config.speculative_config is not None
        draft_model_config = vllm_config.speculative_config.draft_model_config
        assert draft_model_config is not None
        self.config = draft_model_config.hf_config
        self.quant_config = get_draft_quant_config(vllm_config)
        # 词嵌入默认为空：草稿与目标共享 embed_tokens，避免复制大矩阵。
        self.embed_tokens: nn.Module | None = None

        # context_proj：输入宽度 = 目标隐状态宽度 × 目标层数（拼接多层 aux 隐状态），
        # 输出 = 草稿 hidden_size。ColumnParallel 按输出维切分 + gather_output
        # 拼回完整结果（草稿上下文每层都需要完整向量）。
        self.context_proj = ColumnParallelLinear(
            self.config.target_hidden_size * self.config.num_target_layers,
            self.config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "context_proj"),
            gather_output=True,
        )
        self.context_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
        self.layers = nn.ModuleList(
            [
                AscendK3DSparkDecoderLayer(
                    vllm_config=vllm_config,
                    config=self.config,
                    layer_idx=layer_idx,
                    start_layer_id=start_layer_id,
                    prefix=prefix,
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        self.final_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
        # 马尔可夫头：以草稿词表（draft_vocab_size，通常是目标词表的频繁子集）
        # 和低秩 markov_rank 做 bigram 先验偏置。
        self.markov_head = DSparkMarkovHead(
            self.config.vocab_size,
            self.config.draft_vocab_size,
            self.config.markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: (
            torch.Tensor | list[torch.Tensor | None] | tuple[torch.Tensor | None, ...] | None
        ) = None,
    ) -> None:
        """把目标模型隐状态预投影成草稿各层的共享 KV 并写入 KV cache。

        【原理】DSpark 的草稿块需要"attend 到目标模型已经算过的上下文"。
        常规做法是每层前向时重新对上下文做注意力；这里改为**一次性预计算**：
        上下文隐状态经各层 MLA 的 fused_qkv_a_proj 投影出 KV 潜向量
        （截掉 Q 分量，只留 [q_lora_rank:] 段的 KV），再用 RoPE 旋转后
        直接 scatter 进该层 KV cache 的指定槽位。草稿块前向时上下文
        直接从 cache 读取，省去逐层重复投影。

        参数：
        - context_states: [num_context_tokens, target_hidden_size] 目标隐状态；
        - context_positions: [num_context_tokens] 各 token 的位置（RoPE 用）；
        - context_slot_mapping: 每层（或全局共享）的 cache 槽位映射；
          None 或空张量时跳过（无上下文可写）。
        """
        if context_slot_mapping is None or context_states.numel() == 0:
            return
        # 槽位映射可能是"逐层一个张量"的列表（各层槽位不同）或全局共享单张量。
        per_layer_slot_mapping = isinstance(context_slot_mapping, (list, tuple))
        # 一次性算好本批位置的 cos/sin（所有层共用，省重复计算）。
        cos, sin = get_cos_and_sin_mla(context_positions)
        for layer_idx, layer in enumerate(self.layers):
            attn = layer.self_attn
            assert attn.fused_qkv_a_proj is not None
            assert attn.q_lora_rank is not None
            # 步骤1：融合 QKV-a 投影 —— 输出 = [Q潜向量 | KV潜向量] 拼接。
            qkv_lora = attn.fused_qkv_a_proj(context_states)[0]
            # 步骤2：切片取 KV 段（跳过前 q_lora_rank 列的 Q 分量），
            # contiguous() 保证后续算子拿到连续内存（切片结果是视图）。
            kv_no_split = qkv_lora[..., attn.q_lora_rank :].contiguous()
            # 步骤3：取本层槽位映射（None 表示该层不写，跳过）。
            slots = context_slot_mapping[layer_idx] if per_layer_slot_mapping else context_slot_mapping
            if slots is None:
                continue
            # 步骤4：调用昇腾 MLA 实现的 KV 预填接口：应用 RoPE + scatter 进 cache。
            attn.impl.exec_kv_prefill(
                kv_no_split,
                cos,
                sin,
                attn.kv_cache,
                slots,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """草稿主体前向：嵌入（可选）→ 逐层 → final_norm。

        input_ids: [num_tokens]；inputs_embeds 优先于 input_ids（外部已嵌入时）。
        返回最终隐状态 [num_tokens, hidden_size]。
        """
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = inputs_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        # 收尾 norm：融合"残差相加 + 归一化"，残差分支此后不再需要（用 _ 丢弃）。
        hidden_states, _ = self.final_norm(hidden_states, residual)
        return hidden_states


class AscendK3DSparkForCausalLM(UpstreamK3DSparkForCausalLM):
    """K3 DSpark 草稿模型顶层（注册名 "K3DSparkModel"）。

    持有 AscendK3DSparkModel 与 logits_processor；lm_head 为 None 表示
    与目标模型共享输出头（投机解码常见做法：草稿只出 logits，采样共享）。
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        assert self.draft_model_config is not None
        self.config = self.draft_model_config.hf_config
        # 草稿层编号从目标层数开始（避免与目标层命名冲突）。
        target_layer_num = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        self.model = AscendK3DSparkModel(
            vllm_config=vllm_config,
            start_layer_id=target_layer_num,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head: nn.Module | None = None
        # logits 处理基于 draft_vocab_size（草稿缩减词表），
        # logit_scale：可选的输出缩放系数（部分 checkpoint 使用）。
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size,
            scale=getattr(self.config, "logit_scale", 1.0),
        )

    def post_process(self, vllm_config: VllmConfig) -> None:
        # 加载后钩子：对 context_proj 做 QuaRot 旋转对齐（复用 Qwen3 DSpark 的实现）。
        align_draft_weights(self, self.model.context_proj, vllm_config)

    def configure_target_aux_hidden_capture(self, target_model: nn.Module) -> None:
        """Select the raw-prefix-sum inputs required by this MLA checkpoint."""
        # 配置目标模型的辅助隐状态捕获：K3 MLA 草稿要求目标输出"原始前缀和"
        # 形式的 aux 状态（materialized=False，即不做融合压缩的原始隐状态）。
        # 多模态目标需先取语言主干；不支持该接口的目标直接报错（快速失败）。
        target = target_model.get_language_model() if hasattr(target_model, "get_language_model") else target_model
        setter = getattr(target, "set_dspark_aux_capture_materialized", None)
        if setter is None:
            raise ValueError("K3 MLA DSpark requires a target supporting raw-prefix-sum auxiliary capture.")
        config = self.config
        # 草稿声明的目标层列表（两种历史命名字段都探测）。
        target_layers = getattr(config, "dspark_target_layer_ids", None) or getattr(config, "target_layer_ids", None)
        # 边界 = 每个目标层号 + 1（捕获"该层输出之后"的位置）。
        boundaries = tuple(int(layer) + 1 for layer in (target_layers or ()))
        aux_layers = getattr(target.model, "aux_hidden_state_layers", None)
        # 一致性校验：目标实际捕获的层与草稿期望必须逐项相等，
        # 且目标 hidden_size 与草稿的 target_hidden_size 一致，否则属于配置错误。
        if (
            aux_layers is None
            or tuple(aux_layers) != boundaries
            or target.model.config.hidden_size != config.target_hidden_size
        ):
            raise ValueError("K3 MLA draft and target auxiliary states are incompatible.")
        # False：要求目标输出原始（未物化压缩）的 aux 隐状态。
        setter(False)

    def get_draft_attn_causal(self) -> list[bool]:
        """向调度器申报各草稿层的因果性（用于 attention metadata 构建）。"""
        causal = _uses_causal_draft_attention(self.config)
        return [causal] * len(self.model.layers)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load the per-layer KV projections used by the Ascend draft model.

        Upstream additionally duplicates these weights into a CUDA-specific
        cross-layer ``context_kv_proj``.  Ascend deliberately retains the
        quantization-aware per-layer projections, so use vLLM's public loader
        interface without creating that extra packed parameter.
        （上游会把逐层 KV 投影额外复制打包成 CUDA 专用的跨层 context_kv_proj；
        昇腾实现刻意保留逐层、量化友好的投影，因此直接用公共加载器即可。）
        """
        # AutoWeightsLoader：按模块树自动匹配 (名字, 张量) 并装填，
        # mapper 负责把 checkpoint 命名映射到本实现的模块命名（继承自上游）。
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed draft tokens and replace multimodal placeholder positions.

        vLLM 0.27 passes the target model's precomputed multimodal embeddings
        through the speculative proposer.  K3 DSpark shares the target token
        embedding but upstream still exposes the older text-only method
        signature, so adapt that interface without duplicating the vision
        tower in the draft model.
        （vLLM 0.27 会把目标模型预计算好的多模态嵌入透传给草稿器；
        草稿与目标共享词嵌入，但上游还是旧版纯文本签名 —— 这里做接口
        适配，而不在草稿里复制一份视觉塔。）
        """
        # 纯文本路径：无多模态嵌入或掩码时直接走上游的嵌入方法。
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0 or is_multimodal is None:
            return self.model.embed_input_ids(input_ids)

        # Placeholder ids are overwritten below.  Mask them before the shared
        # vocabulary lookup so out-of-vocabulary multimodal ids are safe too.
        # （先把多模态占位 id 置 0 再查共享词表，防止越界 id 引发索引错误。）
        text_input_ids = input_ids.masked_fill(
            is_multimodal.to(device=input_ids.device, non_blocking=True),
            0,
        )
        inputs_embeds = self.model.embed_input_ids(text_input_ids)
        # 合并：占位位置用目标模型算好的多模态嵌入覆盖文本嵌入。
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )
