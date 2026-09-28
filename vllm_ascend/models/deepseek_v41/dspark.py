# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# 【文件职责】DeepSeek V4.1 的 DSpark 投机解码(speculative decoding)草稿模型。
#
# 【DSpark 是什么】DSpark 是 DeepSeek 自研的多 token 预测(MTP)草稿架构，
# 本文件实现其 Ascend 版本，实现 vLLM 的 SupportsEagle3 接口（EAGLE3 是
# "以目标模型隐状态为条件"的草稿范式）。结构特点：
#   - 3 个串行草稿块（对应 checkpoint 的 mtp.{0,1,2}.* 权重树），每个块是
#     目标模型解码层的轻量版：SWA 注意力（滑窗）+ MoE + mHC 超连接；
#   - main_proj/main_norm 把目标模型若干层(dspark_target_layer_ids)的残差
#     流拼接投影，作为草稿的条件输入；
#   - Markov 头（低秩马尔可夫偏置）：markov_w1/markov_w2 两个低秩矩阵，
#     由上一草稿 token 嵌入产生对 logits 的加性偏置（n-gram 统计先验）；
#   - Confidence 头：预测每个草稿 token 的接受概率，供验证阶段使用。
#
# 【与目标模型的关系】草稿注意力复用 model.py 的投影结构
# (DeepseekV41SWAAttention)与解码层(DeepseekV41DecoderLayer)，仅替换
# 注意力后端为草稿专用 SWA cache；词嵌入/lm_head 可与目标模型共享。
#
# 【NPU 适配点】权重名重映射(_remap_*)兼容 Ascend 保留 wq_a/wkv 分离参数
# 的命名约定；MoE 通信方法按 MoECommType 枚举构造。
# =============================================================================
"""DeepSeek V4.1 dSPark draft model for Ascend."""

import typing
from collections.abc import Iterable

import regex as re
import torch
import torch.nn as nn
import vllm.envs as envs
from transformers import PretrainedConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsEagle3
from vllm.model_executor.models.qwen3_dspark import DSparkConfidenceHead, DSparkMarkovHead
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix, process_eagle_weight

from vllm_ascend.attention.context_parallel.dsa_v41_cp import get_v41_cp_classes
from vllm_ascend.attention.dsa_v41 import DeepseekV41CacheBackend, scatter_cache_sk
from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec
from vllm_ascend.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_shard,
)
from vllm_ascend.models.deepseek_v41.model import (
    AscendDeepseekV41SWACache,
    DeepseekV41Attention,
    DeepseekV41DecoderLayer,
    DeepseekV41LayerRole,
    DeepseekV41SWAAttention,
)
from vllm_ascend.ops.rope_dsv4 import get_cos_and_sin_dsa
from vllm_ascend.utils import enable_dsa_cp, normalize_deepseek_v41_config

from .model import DeepseekV41MixtureOfExperts, DeepseekV41MoE


def _apply_dsv4_rope(
    rotary_emb: nn.Module,
    positions: torch.Tensor,
    x: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """对张量施加（或逆向施加）V4.1 复指数 RoPE。

    参数:
        rotary_emb: ComplexExpRotaryEmbedding 模块（其 layername 属性决定
            用哪一层的 cos/sin 表）。
        positions: [tokens] 位置。
        x: [tokens, heads, rope_head_dim] 待旋转张量。
        inverse: True 时取 -sin（旋转逆运算，用于把已旋转数据转回原域）。
    返回:
        旋转后的 x（同形状）。
    语法点: * 之后的关键字默认参数为 keyword-only，调用方必须写 inverse=True。
    """
    cos, sin = get_cos_and_sin_dsa(positions)
    # 按层名从缓存表取出本层的 cos/sin（V4.1 每层可有不同 RoPE 参数）。
    layer_name = rotary_emb.layername
    cos_t = cos[layer_name]
    sin_t = sin[layer_name]
    if inverse:
        # 逆旋转 = 用 -sin（复指数共轭），可撤销此前施加的 RoPE。
        sin_t = -sin_t
    return rotary_emb(x, cos_t, sin_t)


def _get_dspark_num_mtp_layers(config: PretrainedConfig) -> int:
    """读取 DSpark 草稿层数：优先 n_mtp_layers，回退 dspark_num_mtp_layers，
    默认 3。语法点: getattr(config, x, None) 在属性缺失时返回 None。"""
    num_layers = getattr(config, "n_mtp_layers", None)
    if num_layers is None:
        num_layers = getattr(config, "dspark_num_mtp_layers", 3)
    return int(num_layers or 3)


class DeepseekV41DSparkSWACache(AscendDeepseekV41SWACache):
    """DeepSeek V4.1 DSpark draft SWA cache layer.

    ``DeepseekV41DraftSWASpec`` exists only to identify the draft cache.

    【中文说明】DSpark 草稿模型的滑窗 KV cache。继承目标模型的
    AscendDeepseekV41SWACache，重写 get_kv_cache_spec/get_attn_backend。
    重新构造一份 AscendSlidingWindowMLASpec 的目的只是"身份标记"——让
    cache_config.py 能把草稿 cache（".mtp." 层）与目标 SWA cache 区分开，
    归入草稿专用分组。
    """

    # TODO: Extract DeepseekV41DraftSWASpec construction from this cache subclass.
    def get_kv_cache_spec(self, vllm_config):
        """构造草稿专用的滑窗 MLA 规格（复制父类 spec 字段，保持类型标记）。"""
        spec = super().get_kv_cache_spec(vllm_config)
        return AscendSlidingWindowMLASpec(
            block_size=spec.block_size,
            num_kv_heads=spec.num_kv_heads,
            head_size=spec.head_size,
            dtype=spec.dtype,
            sliding_window=spec.sliding_window,
            cache_dtype_str=spec.cache_dtype_str,
            model_version=spec.model_version,
        )

    def get_attn_backend(self):
        """返回草稿注意力后端：与目标模型共用 DeepseekV41CacheBackend。"""
        return DeepseekV41CacheBackend


class DeepseekV41DSparkAttention(DeepseekV41SWAAttention):
    """DeepSeek V4.1 DSpark draft attention layer."""
    """【中文说明】DSpark 草稿注意力层。继承目标模型的 DeepseekV41SWAAttention
    （共享 MLA 投影结构），差异点：
    1) swa_cache_cls 换成草稿专用 cache 类（见上）；
    2) 额外构造 CP(上下文并行)感知的 v41_impl 实现对象并注册到
       static_forward_context，使 torch.ops.vllm.dsa_v41_forward 能按层名
       找到本层（与目标层同一调度机制）；
    3) forward 直接复用目标模型 DeepseekV41Attention.forward。
    """

    # 草稿专属 SWA cache 类（身份标记用）。
    swa_cache_cls = DeepseekV41DSparkSWACache

    def __init__(self, *args, **kwargs):
        """参数与父类一致（vllm_config/config/prefix 等经 kwargs 透传）。"""
        super().__init__(*args, **kwargs)
        self.softmax_scale = self.scale
        self.shared_state = None
        prefix = kwargs["prefix"]
        # Returns (metadata_builder_cls, impl_cls); select the CP-aware implementation.
        # 【中文】get_v41_cp_classes 返回 (元数据构建器类, 实现类) 二元组；
        # 这里取实现类并按"草稿角色"（compress_ratio=0、非任何源层）实例化。
        self.v41_impl = get_v41_cp_classes()[1](
            prefix=prefix,
            # 语法点: DeepseekV41LayerRole 是 frozen dataclass，必须按
            # 关键字传全部字段；layer_idx 从 prefix 倒数第二段解析。
            role=DeepseekV41LayerRole(
                layer_idx=int(prefix.split(".")[-2]),
                compress_ratio=0,
                kv_source_layer=None,
                index_source_layer=None,
                is_kv_source=False,
                is_index_source=False,
                is_candidate_source=False,
                uses_candidate_filter=False,
                engram_slot=None,
            ),
            topology=None,
            long_kv_source_prefix=None,
            index_k_source_prefix=None,
        )
        # 注册到静态前向上下文：torch.ops.vllm.dsa_v41_forward 按此名寻层，
        # 同时保证 ACL Graph 捕获时能通过名字访问模块。
        self.v41_layer_name = f"{prefix}.v41_attn"
        context = kwargs["vllm_config"].compilation_config.static_forward_context
        context[self.v41_layer_name] = self

    # 语法点: 类体内赋值 forward = 父类方法——直接复用 DeepseekV41Attention
    # 的 forward（经自定义算子调度），避免重复实现。
    forward = DeepseekV41Attention.forward


class DeepseekV41DSparkDecoderLayer(DeepseekV41DecoderLayer):
    """V4.1 delayed-mHC block with a draft-only SWA attention backend."""
    """【中文说明】DSpark 草稿解码层：结构与目标解码层完全一致（延迟 mHC
    系数交接 + 注意力 + MoE），只把注意力类换成草稿专用实现。"""

    attention_cls = DeepseekV41DSparkAttention


class DeepseekV41DSparkModel(torch.nn.Module):
    """Three serial draft blocks matching the checkpoint's ``mtp.*`` tree."""
    """【中文说明】DSpark 草稿模型主体：3 个串行草稿块（对应 checkpoint 的
    mtp.0/mtp.1/mtp.2 权重），外加 main_proj/main_norm 条件投影、最终
    RMSNorm、Markov 头与 Confidence 头。"""

    def __init__(self, *, vllm_config, prefix="") -> None:
        """构建草稿模型。

        步骤:
            1) 从 speculative_config.draft_model_config 读草稿文本配置并归一化；
            2) 构建 VocabParallelEmbedding 词嵌入（可与目标模型共享权重）；
            3) 构建 3 个草稿解码层，层号从 num_hidden_layers 起编号（与
               checkpoint 的 mtp.{i} 一一对应）；
            4) main_proj/main_norm 挂到第一个草稿层（条件投影入口）；
            5) norm/markov_head 挂到最后一个草稿层（输出端）；
            6) 判断是否需要把 token IDs 传给 MoE（hash 路由/视觉偏置）。
        """
        super().__init__()
        self.vllm_config = vllm_config
        # 草稿模型配置来自 speculative_config（--speculative-config 指定）。
        draft_model_config = vllm_config.speculative_config.draft_model_config
        config = normalize_deepseek_v41_config(draft_model_config.hf_text_config)
        self.config = config
        # hc_mult: mHC 并行残差流份数（超连接倍率）。
        self.hc_mult = config.hc_mult
        self.hidden_size = config.hidden_size
        self.block_size = int(config.dspark_block_size)
        # DSpark 消费目标模型哪些层的残差流（checkpoint 以 1-based 记录）。
        self.target_layer_ids = list(config.dspark_target_layer_ids)
        self.num_dspark_layers = _get_dspark_num_mtp_layers(config)
        # 草稿层全局层号起点 = 目标模型层数（mtp 层排在主干层之后编号）。
        self.mtp_start_layer_idx = config.num_hidden_layers

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        # 语法点: nn.ModuleDict 需要字符串键；这里用全局层号作键，
        # 使权重名 model.layers.{n}.* 与 checkpoint 映射规则吻合。
        self.layers = torch.nn.ModuleDict(
            {
                str(self.mtp_start_layer_idx + idx): DeepseekV41DSparkDecoderLayer(
                    vllm_config,
                    prefix=f"mtp.{idx}",
                    config=config,
                    is_draft_layer=True,
                )
                for idx in range(self.num_dspark_layers)
            }
        )

        first_layer = self.layers[str(self.mtp_start_layer_idx)]
        self.use_sequence_parallel_moe = vllm_config.parallel_config.use_sequence_parallel_moe
        # main_proj: 把 len(target_layer_ids) 份目标隐状态拼接后投影回
        # hidden_size（DSpark 的条件输入融合）；gather_output=True 在 TP 下
        # 先聚集完整输出。V4.1 该投影固定存 BF16（不量化），故 quant_config=None。
        self.main_proj = ColumnParallelLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=None,  # DeepSeek V4.1 stores this projection in BF16.
            prefix=maybe_prefix(prefix, f"layers.{self.mtp_start_layer_idx}.main_proj"),
            gather_output=True,
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 把 main_proj/main_norm 挂到第一个草稿层对象上：权重名会落在
        # layers.{mtp_start}.main_proj 命名空间，与重映射规则匹配。
        first_layer.main_proj = self.main_proj
        first_layer.main_norm = self.main_norm

        # 最终 RMSNorm 与 Markov 头挂到最后一个草稿层。
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        last_layer_idx = self.mtp_start_layer_idx + self.num_dspark_layers - 1
        # Markov 头: 低秩马尔可夫偏置——draft_vocab（可小于词表）嵌入经
        # 两个低秩矩阵产生 logits 偏置（上一 token 的 n-gram 统计先验）。
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            getattr(config, "draft_vocab_size", None) or config.vocab_size,
            config.dspark_markov_rank,
            prefix=maybe_prefix(prefix, f"layers.{last_layer_idx}.markov_head"),
        )
        # Confidence 头: 输入 = 隐状态 ⊕ Markov 嵌入，输出每位置接受概率。
        self.confidence_head = DSparkConfidenceHead(
            input_dim=config.hidden_size + config.dspark_markov_rank,
            prefix=maybe_prefix(prefix, "confidence_head"),
            bias=False,
            with_markov=True,
        )
        last_layer = self.layers[str(last_layer_idx)]
        last_layer.norm = self.norm
        last_layer.markov_head = self.markov_head

        # 任一层 MoE gate 存在 hash 路由表(tid2eid)或视觉路由偏置(bias_vl)，
        # 则前向需要把原始 token IDs 传进 MoE。
        self.needs_moe_input_ids = any(
            layer.mlp.gate.tid2eid is not None or layer.mlp.gate.bias_vl is not None for layer in self.layers.values()
        )

    def _store_standard_swa_kv(self, shared_kv, slot_mapping, attn=None):
        """把共享 KV 写入草稿层的标准 SWA cache。

        参数:
            shared_kv: [tokens, 1, head_dim] 投影后的 KV 潜向量。
            slot_mapping: [tokens, 2] (block_idx, 偏移) 或 [tokens] 展平槽位；
                负值表示无效槽（不写入）。
            attn: 提供缓存层引用的注意力模块。
        步骤: 一维槽位先拆成 (block, 偏移) 二维格式，再用 scatter_cache_sk
            按槽位散写。
        """
        if slot_mapping is None or slot_mapping.numel() == 0:
            return
        cache = attn.dsa_attn.swa_cache_layer
        if slot_mapping.ndim == 1:
            # 一维展开槽位 → 二维 (block_idx, block 内偏移)，无效位填 -1。
            valid = slot_mapping >= 0
            physical = slot_mapping.clamp_min(0)
            slot_mapping = torch.stack((physical // cache.block_size, physical % cache.block_size), dim=-1).to(
                torch.int32
            )
            slot_mapping.masked_fill_(~valid.unsqueeze(-1), -1)
        # squeeze(1) 去掉单头维后散写进 cache 平面。
        scatter_cache_sk(cache.kv_cache[0], slot_mapping, shared_kv.squeeze(1))

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """草稿模型前向（单步）。

        参数:
            input_ids: [tokens] 草稿输入词元（可为 -1 占位，SP 填充产生）。
            positions: [tokens] 位置。
        返回:
            [tokens, hidden_size] mHC 折叠后的最终隐状态。
        步骤:
            1) 词嵌入 → 复制 hc_mult 份并行残差流；
            2) 序列并行(SP)分片（若启用）：隐状态与 token IDs 按 TP 切分；
            3) pre_mix 初始化为 one-hot（首流权重 1，其余 0）；
            4) 逐层执行（attention+MoE），层间传递 (hidden_states, pre_mix)；
            5) 最后一层 hc_collapse 把多流折叠成单流；
            6) SP 模式 all_gather 恢复完整 token 维。
        """
        # 步骤1: 嵌入并复制成 [tokens, hc_mult, hidden]（mHC 多流展开）。
        hidden_states = self.embed_tokens(input_ids).unsqueeze(-2).repeat(1, self.hc_mult, 1)
        full_num_tokens = positions.shape[0]
        if self.use_sequence_parallel_moe:
            # 步骤2: 序列并行——token 维按 TP rank 切分，省去 MoE 前的
            # all-gather；VLLM_MOE_SKIP_PADDING 开启时维护填充掩码。
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                forward_context.is_padding = sp_padding_mask(
                    forward_context.is_padding,
                    hidden_states,
                )
            hidden_states = sp_shard(hidden_states)
            input_ids = sp_shard(input_ids)
        # 步骤3: pre_mix 是 mHC 的混合系数状态，初始只信任第一条流。
        pre_mix = hidden_states.new_zeros(hidden_states.shape[0], self.hc_mult, dtype=torch.float32)
        pre_mix[:, 0] = 1.0
        last_layer = None
        moe_input_ids = input_ids
        if self.needs_moe_input_ids:
            # -1 是 SP 填充占位 ID，MoE hash 路由前替换成 0 防越界。
            moe_input_ids = torch.where(input_ids == -1, 0, input_ids)
        # 步骤4: 串行过 3 个草稿块；每层返回新的 (hidden_states, pre_mix)。
        for layer in self.layers.values():
            last_layer = layer
            hidden_states, pre_mix = layer(
                positions,
                hidden_states,
                pre_mix,
                llama_4_scaling=None,
                input_ids=moe_input_ids,
            )
        assert last_layer is not None, "Hyper-connection collapse requires at least one decoder layer"
        # 步骤5: mHC 折叠——用最终 pre_mix 加权求和 hc_mult 条流。
        hidden_states = last_layer.hc_collapse(hidden_states, pre_mix)
        if self.use_sequence_parallel_moe:
            # 步骤6: SP 恢复：all_gather 后截回真实 token 数（去掉 SP 填充）。
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
        return hidden_states

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        """列出草稿各层的 SWA cache 层名（供 runner 注册/绑定 cache）。"""
        return [layer.self_attn.dsa_attn.swa_cache_layer.prefix for layer in self.layers.values()]

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        """把目标模型多层辅助隐状态融合成草稿条件输入。

        参数: aux_hidden_states: [tokens, len(target_layers)*hidden]。
        返回: [tokens, hidden] main_norm(main_proj(·))。
        """
        return self.main_norm(self.main_proj(aux_hidden_states))

    def _project_shared_kv(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attn: type[nn.Module],
    ) -> torch.Tensor:
        """把上下文隐状态投影成共享 KV 潜向量（复用目标层的 wkv/kv_norm）。

        步骤: wkv 投影 → kv_norm → 按 nope/rope 维切分 → 对 rope 段施加
        RoPE → 拼回 [tokens, 1, head_dim]。
        """
        kv = attn.kv_norm(attn.wkv(hidden_states))
        k_nope, k_pe = kv.split([attn.nope_head_dim, attn.rope_head_dim], dim=-1)
        k_pe = _apply_dsv4_rope(attn.rotary_emb, positions, k_pe.unsqueeze(1)).squeeze(1)
        return torch.cat([k_nope, k_pe], dim=-1).view(-1, 1, attn.head_dim).contiguous()

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: list[torch.Tensor | None] | None = None,
    ) -> None:
        """为草稿层预计算并写入 prompt 上下文的共享 KV。

        原理: 投机解码的"上下文预填"阶段——目标模型跑完 prompt 后，草稿层
        需要同一份上下文的 KV 才能在后续草稿步里做注意力。这里逐层投影
        隐状态并散写进各草稿层的 SWA cache。

        参数:
            context_states: [ctx_tokens, hidden] 上下文隐状态。
            context_positions: [ctx_tokens] 上下文位置。
            context_slot_mapping: 每层一个槽位映射（None 表示跳过）。
        """
        if context_states.numel() == 0 or context_slot_mapping is None:
            return
        for layer_idx, layer in enumerate(self.layers.values()):
            layer_context_slot_mapping = None if context_slot_mapping is None else context_slot_mapping[layer_idx]
            if context_positions.numel() == 0:
                return
            attn = layer.self_attn
            # 复用该层注意力模块的投影权重生成共享 KV。
            shared_kv = self._project_shared_kv(context_states, context_positions, attn)
            self._store_standard_swa_kv(shared_kv, layer_context_slot_mapping, attn)

    def hc_head(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        """独立的 mHC 头计算（等价 npu_hc_pre_v3 的 PyTorch 参考实现）。

        步骤: 展平多流 → RMS 归一因子 → 线性混合 → sigmoid 门控 → 加权求和。
        供测试/非融合路径使用，验证自定义算子正确性。
        """
        shape, dtype = x.size(), x.dtype
        x = x.flatten(1).float()
        # RMS 归一化因子 1/sqrt(mean(x²)+eps)。
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = torch.nn.functional.linear(x, hc_fn) * rsqrt
        # sigmoid 缩放门控 + hc_eps 防零。
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        # 按 hc_mult 流加权求和折叠回单流。
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
        return y.to(dtype)

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Markov 头的 token 嵌入: [tokens] → [tokens, markov_rank]。"""
        return self.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor, logits_processor: LogitsProcessor) -> torch.Tensor:
        """由 Markov 嵌入产生 logits 加性偏置（低秩两步投影）。"""
        return self.markov_head.bias(markov_embed, logits_processor)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: ParallelLMHead,
        logits_processor: LogitsProcessor,
    ) -> torch.Tensor:
        """草稿 logits: norm → lm_head → logits_processor 采样处理。"""
        return logits_processor(lm_head, self.norm(hidden_states))

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """生成专家权重加载映射 (参数名, checkpoint 名, expert_id, shard_id)。"""
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=0,
        )


@support_torch_compile
class DSparkDeepseekV41ForCausalLM(torch.nn.Module, DeepseekV41MixtureOfExperts, SupportsEagle3):
    """DSpark 草稿模型顶层类（vLLM 投机解码的 draft model 入口）。

    多继承:
        torch.nn.Module            —— PyTorch 模块基类；
        DeepseekV41MixtureOfExperts—— MoE 元数据混入（EPLB 统计需要）；
        SupportsEagle3             —— vLLM EAGLE3 草稿接口（compute_logits/
                                     markov_embed/markov_bias/compute_confidence 等）。
    语法点: @support_torch_compile 装饰器标记该类支持 torch.compile 编译。
    """

    # 声明 checkpoint 的 gate/up 两个权重合并加载进 gate_up_proj（融合算子）。
    packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}

    def __init__(self, *, vllm_config, prefix="") -> None:
        """构建草稿顶层：配置、旋转路径、草稿主体、lm_head 与 MoE 通信方法。"""
        super().__init__()
        self.config = vllm_config.speculative_config.draft_model_config.hf_text_config

        from vllm_ascend.utils import get_rotation_path

        # 量化场景下的权重旋转路径（如 GPTQ/W8A8 的旋转重打包）；非量化为 None。
        self.rotation_path = get_rotation_path(vllm_config) if vllm_config.quant_config is not None else None
        self.model = DeepseekV41DSparkModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        # 局部导入避免模块级循环依赖。
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        # 收集 MoE 元数据（专家数/层数等，EPLB 需要）。
        self.set_moe_parameters()
        from vllm_ascend.ascend_forward_context import MoECommType
        from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method

        # 为每种 MoE 通信类型预构建通信方法实例（EP all-to-all 等的 Ascend 实现）。
        self.moe_comm_methods = {kind: get_moe_comm_method(kind) for kind in MoECommType}

    def _remap_dspark_name(self, name: str) -> str | None:
        """在通用重映射之上叠加 DSpark 专属名字修正。

        原理: DeepSeek V4.1 checkpoint 用"操作名"命名低秩 Markov 矩阵，
        而运行时用明确的 embedding/projection 参数名；这里做三处字符串
        替换对齐（markov_w1/w2、confidence_head.proj）。
        """
        mapped = self._remap_checkpoint_name(name)
        if mapped is None:
            return None
        # DeepSeek V4.1 names the low-rank Markov matrices after their operations,
        # while the runtime uses explicit embedding/projection parameter names.
        mapped = mapped.replace(".markov_head.embed.weight", ".markov_head.markov_w1.weight")
        mapped = mapped.replace(".markov_head.head.weight", ".markov_head.markov_w2.weight")
        mapped = mapped.replace("model.confidence_head.weight", "model.confidence_head.proj.weight")
        return mapped

    def set_moe_parameters(self) -> None:
        """扫描草稿层收集 MoE 元数据（供 EPLB/专家管理使用）。

        原理: 遍历所有非 PPMissingLayer 的层，取最后一个 DeepseekV41MoE
        作为样例（前面的层可能是稠密层），提取专家数量等参数。
        语法点: typing.MutableSequence/Sequence 是可变/不可变序列的抽象类型。
        """
        self.expert_weights: typing.MutableSequence[typing.Sequence[torch.Tensor]] = []
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.moe_layers: list[nn.Module] = []
        self.moe_mlp_layers: list[DeepseekV41MoE] = []
        example_moe = None
        for layer in self.model.layers.values():
            if isinstance(layer, PPMissingLayer):
                continue

            if isinstance(layer.mlp, DeepseekV41MoE):
                # Pick last one layer since the first ones may be dense layers.
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)

        self.extract_moe_parameters(example_moe)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """草稿前向入口（EAGLE3 proposer 每个草稿步调用一次）。"""
        return self.model(
            input_ids=input_ids,
            positions=positions,
        )

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """草稿 logits（EAGLE3 接口）。

        # Full-vocab draft: base logits, no d2t scatter.
        【中文】DSpark 用全词表草稿：直接算 base logits，无需 draft-to-target
        词表散射（d2t scatter 是小词表草稿模型的操作）。
        """
        return self.compute_logits(hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        """草稿 token ID → 目标 token ID 映射。全词表草稿时二者相同，恒等返回。"""
        return draft_ids  # full-vocab: draft ids are target ids

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        """计算草稿 logits；spec_step_idx（第几个投机步）在本模型中无用，删除。"""
        del spec_step_idx
        return self.model.compute_logits(
            hidden_states,
            self.lm_head,
            self.logits_processor,
        )

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Markov 头嵌入接口，转发到草稿主体。"""
        return self.model.markov_embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        """Markov 头偏置接口，转发到草稿主体。"""
        return self.model.markov_bias(markov_embed, self.logits_processor)

    def compute_confidence(self, head_hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        """Per-position acceptance probability for each drafted token."""
        """【中文说明】每个草稿 token 的逐位置接受概率（sigmoid 输出，0~1）。
        参数: head_hidden 是最后一个草稿块折叠后的隐状态（经 norm/lm_head
        之前的表示），markov_embed 是 Markov 嵌入。验证器可用该概率做
        概率性接受或统计期望接受长度。"""
        return torch.sigmoid(self.model.confidence_head(head_hidden, markov_embed))

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        """草稿 KV cache 层名列表（runner 据此绑定 KV 分配）。"""
        return self.model.get_draft_kv_cache_layer_names()

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        """融合目标模型多层辅助隐状态 → 草稿条件输入，转发到草稿主体。"""
        return self.model.combine_hidden_states(aux_hidden_states)

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: list[torch.Tensor | None] | None = None,
    ) -> None:
        """prompt 上下文 KV 预计算入口，转发到草稿主体（见其 docstring）。"""
        self.model.precompute_and_store_context_kv(
            context_states,
            context_positions,
            context_slot_mapping,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the ``mtp.{i}.*`` draft weights from the target checkpoint.

        Non-MTP weights belong to the target model and are skipped, except for
        standalone embedding/head weights used by the Ascend draft loader.

        【中文说明】从目标模型 checkpoint 中挑出 mtp.{i}.* 草稿权重并加载。
        流程:
            1) 名字规范化: embed/head/hc_head_* 等顶层名先处理；
            2) _remap_dspark_name 做 mtp.{stage}.* → model.layers.{n}.* 映射，
               非 mtp 权重返回 None 直接跳过（属于目标模型）；
            3) 专家权重走 expert_mapping（带 expert_id 的加载器）；
            4) 其余走 stacked（gate/up 合并）或默认加载器；
            5) attn_sink 按 TP 切片（CP 模式下整份加载）。
        """
        expert_mapping = self.model.get_expert_mapping()

        # (param_name, checkpoint shard name, shard_id) for non-expert
        # stacked parameters. Ascend keeps wq_a and wkv as separate parameters.
        # 【中文】非专家的堆叠参数映射：checkpoint 的 gate_proj/up_proj 两个
        # 分片加载进融合的 gate_up_proj；shard_id 标明在融合权重中的列区间。
        stacked_params_mapping = [
            ("mlp.gate_up_proj", "mlp.gate_proj", 0),
            ("mlp.gate_up_proj", "mlp.up_proj", 1),
            ("shared_experts.gate_up_proj", "shared_experts.gate_proj", 0),
            ("shared_experts.gate_up_proj", "shared_experts.up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # TP 切片信息: 注意力头按 TP rank 均分，attn_sink 需取本 rank 分片。
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_local_head = self.config.num_attention_heads // tp_size
        head_start = n_local_head * tp_rank
        head_end = n_local_head * (tp_rank + 1)

        for name, loaded_weight in weights:
            # 步骤1: 顶层权重名规范化（无旋转路径时套 model. 前缀）。
            if name == "embed.weight" and not self.rotation_path:
                name = "model.embed_tokens.weight"
            elif name == "head.weight" and not self.rotation_path:
                name = "lm_head.weight"
            elif name in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
                name = f"model.{name}"
            else:
                # 步骤2: mtp.* 名字映射；None = 非草稿权重，跳过。
                mapped_name = self._remap_dspark_name(name)
                if mapped_name is None:
                    continue
                name = mapped_name

            # Detect whether the checkpoint ships its own embed_tokens / lm_head
            # for the draft model.
            # 【中文】检测 checkpoint 是否为草稿模型单独提供了 embed/lm_head。
            process_eagle_weight(self, name)

            # Expert scale parameters use Ascend's ``weight_scale`` convention.
            # 【中文】专家 scale 参数改用 Ascend 的 weight_scale 命名。
            if name.endswith(".scale"):
                name = name.replace(".scale", ".weight_scale")

            # The multimodal checkpoint also contains one vision-router bias
            # for each MTP/DSpark layer.  DSpark runs only during text decode,
            # so draft MoE gates intentionally do not expose ``bias_vl``.
            # Do not alias it to the text correction bias: that would change
            # text routing whenever speculative decoding is enabled.
            # 【中文】多模态 checkpoint 里每个 MTP 层带一个视觉路由偏置；
            # 草稿只在纯文本解码期运行，不设 bias_vl 参数——直接跳过，
            # 绝不能错误映射到文本路由偏置（否则投机解码会改变文本路由）。
            if name.endswith(".e_score_correction_bias_vl") and name not in params_dict:
                logger.info_once("Ignoring vision-only router bias while loading the text-only DSpark drafter")
                continue

            # 步骤3: 专家权重——遍历 expert_mapping 找到 (参数名, expert_id)，
            # 调用带 expert_id 的加载器；return_success=True 使无本地副本的
            # 专家（在其他 EP rank 上）能正确返回失败而非误报成功。
            if ".experts." in name:
                for param_name, weight_name, expert_id, shard_id in expert_mapping:
                    if weight_name not in name:
                        continue
                    name_mapped = name.replace(weight_name, param_name)
                    param = params_dict[name_mapped]
                    # 语法点: typing.cast 仅做静态类型标注，运行时无开销。
                    weight_loader = typing.cast(typing.Callable[..., bool], param.weight_loader)
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        loaded_params.add(name_mapped)
                        break
                continue

            # Stacked rules only apply to decoder-layer weights. Head-stack
            # parameters load directly through the fallback below.
            # 【中文】步骤4: 堆叠合并规则只适用于解码层权重；头部的堆叠参数
            # 直接走下面的默认加载分支。
            is_layer_param = name.startswith("model.layers.")
            # 语法点: for...else——else 在循环未被 break 时执行（即没有命中
            # 任何堆叠映射，走默认加载）。
            for param_name, weight_name, stacked_shard_id in stacked_params_mapping:
                if not is_layer_param or f".{weight_name}." not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, stacked_shard_id)
                loaded_params.add(name)
                break
            else:
                # 步骤5: attn_sink 特殊处理——按 TP 头切片加载；
                # enable_dsa_cp()（DSA 上下文并行）开启时每个 rank 需要全部头。
                if "attn_sink" in name:
                    if enable_dsa_cp():
                        narrow = loaded_weight
                    else:
                        narrow = loaded_weight[head_start:head_end]
                    # 语法点: torch.no_grad() 上下文——加载权重不需要梯度。
                    with torch.no_grad():
                        params_dict[name].copy_(narrow)
                    loaded_params.add(name)
                    continue
                # 默认路径: 取参数自带的 weight_loader（处理 TP 切片等）或
                # default_weight_loader（直接复制）。
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        logger.info_once("DSpark draft model loaded: %d params", len(loaded_params))
        return loaded_params

    def _remap_checkpoint_name(self, name: str) -> str | None:
        """把 checkpoint 的 mtp.{stage}.{rest} 名字映射到运行时参数名。

        规则:
            - 非 mtp.* 名字返回 None（属于目标模型，由调用方跳过）；
            - 最后一级的 confidence_head.* → model.confidence_head.*；
            - 第 0 级的 embed.weight → model.embed_tokens.weight；
            - 最后一级的 head.weight → lm_head.weight；
            - hc_head_fn/base/scale → model.hc_head_*；
            - main_proj/main_norm 归第一草稿层，norm/markov_head 归最后
              草稿层，其余按 stage 顺序归各层；
            - 层内再做字段重命名（attn→self_attn、w1→gate_proj 等）。
        语法点: re.match + 分组捕获——(\\d+) 取级号，(.*) 取剩余名。
        """
        m = re.match(r"mtp\.(\d+)\.(.*)", name)
        if m is None:
            return None
        stage = int(m.group(1))
        rest = m.group(2)

        # 最后一级承载 confidence_head。
        if stage == self.model.num_dspark_layers - 1 and rest.startswith("confidence_head."):
            return f"model.{rest}"

        if stage == 0 and rest == "embed.weight":
            return "model.embed_tokens.weight"
        if stage == self.model.num_dspark_layers - 1 and rest == "head.weight":
            return "lm_head.weight"
        if rest.startswith(("hc_head_fn", "hc_head_base", "hc_head_scale")):
            return f"model.{rest}"

        # 计算各级对应的运行时层号区间。
        first_layer_idx = self.config.num_hidden_layers
        last_layer_idx = first_layer_idx + self.model.num_dspark_layers - 1
        # main_proj/main_norm 挂第一层；norm/markov_head 挂最后层；
        # 其余按 stage 偏移。
        if rest.startswith(("main_proj.", "main_norm.")):
            layer_idx = first_layer_idx
        elif rest.startswith(("norm.", "markov_head.")):
            layer_idx = last_layer_idx
        else:
            layer_idx = first_layer_idx + stage
        name = f"model.layers.{layer_idx}.{rest}"

        # 层内字段重命名表: checkpoint 命名 → 运行时命名。
        replacements = (
            (".attn.", ".self_attn."),
            (".ffn_norm.", ".post_attention_layernorm."),
            (".attn_norm.", ".input_layernorm."),
            (".ffn.", ".mlp."),
            (".w1.", ".gate_proj."),
            (".w2.", ".down_proj."),
            (".w3.", ".up_proj."),
            (".mlp.gate.bias", ".mlp.gate.e_score_correction_bias"),
        )
        for checkpoint_name, param_name in replacements:
            name = name.replace(checkpoint_name, param_name)
        return name
