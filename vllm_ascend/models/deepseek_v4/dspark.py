# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ============================================================================
# 【模块职责】DeepSeek V4 DSpark 草稿模型（块级草稿器 block drafter）。
# 与 mtp.py 的“串行单 token 草稿”不同，DSpark 一次草拟一整块 token:
#   - 目标模型提供若干选中层的隐状态（aux hidden states）;
#   - main_proj 把它们拼接投影成草稿注意力上下文，并预先把上下文 KV
#     写入草稿层自己的 SWA cache（precompute_and_store_context_kv）;
#   - 草稿层（复用 DeepseekV4DecoderLayer）对整个 draft block 前向，
#     markov_head 给出 draft logits 的马尔可夫偏置，confidence_head
#     估计每个草稿 token 的接受概率（供 verify 阶段做自适应接受）。
# 权重存放在目标 checkpoint 的 mtp.* 命名空间（与 MTP 共用命名空间，
# 但结构与用途完全不同）。
# ============================================================================
"""DeepSeek V4 DSpark draft model for Ascend.

DSpark weights are stored under the target checkpoint's ``mtp.*`` namespace,
but the draft path is a block drafter rather than the ordinary serial MTP
module. The target model provides selected layer hidden states; this model
projects them into the draft attention context and emits a full draft block.
"""

import typing
from collections.abc import Iterable

import regex as re
import torch
import torch.nn as nn
import vllm.envs as envs
from transformers import PretrainedConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
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

from vllm_ascend.models.common.ops.sequence_parallel import sp_padding_mask, sp_shard
from vllm_ascend.models.deepseek_v4.model import (
    DeepseekV2MixtureOfExperts,
    DeepseekV4DecoderLayer,
    DeepseekV4MoE,
)
from vllm_ascend.ops.rope_dsv4 import get_cos_and_sin_dsa
from vllm_ascend.utils import enable_dsa_cp


def _apply_dsv4_rope(
    rotary_emb: nn.Module,
    positions: torch.Tensor,
    x: torch.Tensor,
    *,
    inverse: bool = False,
    rope=None,
) -> torch.Tensor:
    """对输入施加（或逆向解除）DSV4 的 RoPE 旋转。

    原理: DSV4 的 RoPE 表按“层名分组”预计算（get_cos_and_sin_dsa 返回
    {layername: cos/sin} 字典）。本函数按层名取出对应表并调用旋转模块；
    inverse=True 时把 sin 取反即可实现逆向旋转（数学上等价乘共轭）。

    Args:
        rotary_emb: ComplexExpRotaryEmbedding 旋转模块（带 layername 属性）。
        positions: [num_tokens] 位置 id（rope 为 None 时用于现算表）。
        x: [num_tokens, 1, rope_head_dim] 待旋转张量。
        inverse: 是否逆向旋转（keyword-only 参数）。
        rope: 预计算的 (cos_dict, sin_dict)，避免重复计算。
    Returns:
        旋转后的张量（原地写回 x 的存储）。
    """
    # 三元表达式: 已有预计算表则直接用，否则按 positions 现算。
    cos, sin = rope if rope is not None else get_cos_and_sin_dsa(positions)
    # 按层名索引该层专属的 RoPE 表。
    layer_name = rotary_emb.layername
    cos_t = cos[layer_name]
    sin_t = sin[layer_name]
    if inverse:
        # 逆向旋转: sin 取负等价于角度取负（共轭）。
        sin_t = -sin_t
    return rotary_emb(x, cos_t, sin_t)


def _get_dspark_num_mtp_layers(config: PretrainedConfig) -> int:
    """读取 DSpark 草稿层数（配置字段兼容多种命名，缺省 3）。

    getattr 链式兜底: 依次尝试 n_mtp_layers -> dspark_num_mtp_layers -> 3;
    int(...) 兜底处理 None（int(None or 3)）。
    """
    num_layers = getattr(config, "n_mtp_layers", None)
    if num_layers is None:
        num_layers = getattr(config, "dspark_num_mtp_layers", 3)
    return int(num_layers or 3)


class DeepseekV4DSparkModel(nn.Module):
    """DSpark 草稿模型主体（块级草稿器）。

    结构:
      - embed_tokens: 草稿自己的词嵌入（可共享目标模型权重）;
      - layers: num_dspark_layers 个 DeepseekV4DecoderLayer（复用目标层
        的 DSA 注意力 + MoE 结构，编号从 num_hidden_layers 起）;
      - main_proj/main_norm: 把目标模型多个选中层的隐状态拼接并投影为
        草稿注意力上下文（写回第一层）;
      - markov_head: draft logits 的马尔可夫先验偏置（低秩分解，见
        qwen3_dspark 上游实现）;
      - confidence_head: 每个草稿 token 的接受概率估计头;
      - hc_head_*: Hyper-Connections 多路隐状态混合参数（挂在最后一层）。
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        """初始化。

        Args:
            vllm_config: 全局配置（speculative_config 必须存在）。
            prefix: 模块名前缀。
        """
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.vllm_config = vllm_config
        # 草稿模型配置（来自 speculative_config.draft_model_config）。
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        # Hyper-Connections: 每个隐状态展开成 hc_mult 路。
        self.hc_mult = config.hc_mult
        self.hidden_size = config.hidden_size
        # 草稿块大小（一次草拟的 token 数）。
        self.block_size = int(config.dspark_block_size)
        # 目标模型中被选中的层号（其隐状态作为草稿上下文来源）。
        self.target_layer_ids = list(config.dspark_target_layer_ids)
        self.num_dspark_layers = _get_dspark_num_mtp_layers(config)
        # 草稿层编号从 num_hidden_layers 起（与 checkpoint 命名对齐）。
        self.mtp_start_layer_idx = config.num_hidden_layers

        # 草稿词嵌入（VocabParallelEmbedding: 词表按 TP 切片）。
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        # 字典推导式构造 ModuleDict: 键为全局层号字符串，值为草稿层;
        # is_draft_layer=True 使 MoE 跳过 hash 路由等目标层专属逻辑。
        self.layers = nn.ModuleDict(
            {
                str(self.mtp_start_layer_idx + idx): DeepseekV4DecoderLayer(
                    vllm_config,
                    prefix=f"mtp.{idx}",
                    is_draft_layer=True,
                )
                for idx in range(self.num_dspark_layers)
            }
        )

        # 第一层决定序列并行 MoE 是否启用（所有草稿层一致）。
        first_layer = self.layers[str(self.mtp_start_layer_idx)]
        self.use_sequence_parallel_moe = first_layer.use_sequence_parallel_moe

        # main_proj: [hidden*(len(target_layers)), hidden] —— 把目标模型
        # 多个选中层的隐状态拼接后投影为草稿上下文。仅当草稿为 fp8 量化
        # 时才传 quant_config（三元表达式 + 短路 and）。
        _model_quant_cfg = getattr(config, "quantization_config", None)
        _main_proj_qconfig = (
            vllm_config.quant_config
            if _model_quant_cfg is not None and _model_quant_cfg.get("quant_method") == "fp8"
            else None
        )
        # gather_output=True: ColumnParallel 输出 all-gather 成完整张量
        # （上下文需要完整 hidden 维）。
        self.main_proj = ColumnParallelLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=_main_proj_qconfig,
            prefix=maybe_prefix(prefix, f"layers.{self.mtp_start_layer_idx}.main_proj"),
            gather_output=True,
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 把 main_proj/main_norm 挂到第一层（forward 时由第一层调用，
        # 保持与 checkpoint 命名一致的模块归属）。
        first_layer.main_proj = self.main_proj
        first_layer.main_norm = self.main_norm

        # 末端 RMSNorm（挂在最后一层）。
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        last_layer_idx = self.mtp_start_layer_idx + self.num_dspark_layers - 1
        # 草稿词表可裁剪（draft_vocab_size），markov_head 做低秩分解。
        draft_vocab_size = getattr(config, "draft_vocab_size", None) or config.vocab_size
        # markov_head: 马尔可夫头——按上一个 token 查表给出 logits 偏置
        # （复用上游 Qwen3-DSpark 的实现）。
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            draft_vocab_size,
            config.dspark_markov_rank,
            prefix=maybe_prefix(
                prefix,
                f"layers.{last_layer_idx}.markov_head",
            ),
        )

        # confidence_head: 接受概率估计头，输入 = hidden + markov 嵌入
        # 拼接（with_markov=True），输出经 sigmoid 即接受概率。
        self.confidence_head = DSparkConfidenceHead(
            input_dim=config.hidden_size + config.dspark_markov_rank,
            prefix=maybe_prefix(prefix, "confidence_head"),
            bias=False,
            with_markov=True,
        )
        # norm 与 markov_head 挂到最后一层（模块归属与 checkpoint 对齐）。
        last_layer = self.layers[str(last_layer_idx)]
        last_layer.norm = self.norm
        last_layer.markov_head = self.markov_head

        # hc_head 三件套: 多路隐状态混合的 fn/base/scale。
        # requires_grad=False: 推理框架参数（不训练）。
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )
        # hc_head 参数挂到最后一层。
        last_layer.hc_head_fn = self.hc_head_fn
        last_layer.hc_head_base = self.hc_head_base
        last_layer.hc_head_scale = self.hc_head_scale

        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        """返回草稿层 KV cache 伪层的名字列表（供 vLLM 显存规划器
        识别哪些 cache 属于草稿模型）。"""
        return [layer.self_attn.dsa_attn.swa_cache_layer.prefix for layer in self.layers.values()]

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        """把目标模型选中层的隐状态拼接并投影为草稿上下文。

        Args:
            aux_hidden_states: [num_tokens, len(target_layers)*hidden] 拼接的
                目标层隐状态。
        Returns:
            [num_tokens, hidden] 归一化后的草稿上下文。
        """
        return self.main_norm(self.main_proj(aux_hidden_states))

    def _project_shared_kv(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attn: type[nn.Module] | None = None,
        rope=None,
    ) -> torch.Tensor:
        """由隐状态投影草稿层的共享 KV（MLA 式压缩 kv + RoPE 解耦）。

        原理: 与目标层注意力相同——wkv 把 hidden 压到 head_dim 的潜在
        KV 向量（单 KV 头），尾部 rope_head_dim 维做 RoPE。所有草稿层
        共用同一条 kv（shared_kv），写入各自的 SWA cache。

        Args:
            hidden_states: [num_tokens, hidden_size] 草稿上下文。
            positions: [num_tokens] 位置 id。
            attn: 目标注意力模块（借用其 wkv/kv_norm/rotary_emb）。
            rope: 预计算 RoPE 表。
        Returns:
            [num_tokens, 1, head_dim] 共享 KV。
        """
        assert attn is not None
        # 步骤1: wkv 投影 + kv_norm 归一化。
        kv = attn.kv_norm(attn.wkv(hidden_states))
        # npu_rotary_mul writes its result back to the input storage
        # (ComplexExpRotaryEmbedding.forward ends with y.copy_(...)), so rope
        # can run in-place on the rope-segment view of kv; the previous
        # split -> rope -> cat -> contiguous round-trip was a redundant copy.
        # 【中文】NPU 优化: npu_rotary_mul 会把结果原地写回输入存储，
        # 因此直接在 kv 的 RoPE 段视图上原地旋转即可——原先的
        # split->rope->cat->contiguous 往返是多余的拷贝（NPU 适配点:
        # 消除冗余显存拷贝）。
        k_pe = kv[:, attn.nope_head_dim :]
        _apply_dsv4_rope(
            attn.rotary_emb,
            positions,
            k_pe.unsqueeze(1),
            rope=rope,
        )
        # reshape 为单 KV 头形状。
        return kv.view(-1, 1, attn.head_dim)

    def _store_standard_swa_kv(
        self,
        shared_kv: torch.Tensor,
        slot_mapping: torch.Tensor | None,
        attn: type[nn.Module] | None = None,
    ) -> None:
        """把共享 KV 写入草稿层的标准 SWA cache。

        原理（KV Cache 写入与索引）: 通过 get_dsa_attn_kv_plan 获得
        slot 映射计划——把逻辑 slot 变换成 NPU cache 的物理块内偏移
        （format_dsa_slot_mapping），再用 dsa_kv_compress_scatter 把
        shared_kv 分散写入 kv_cache 张量的对应槽位。

        Args:
            shared_kv: [num_tokens, 1, head_dim] 共享 KV。
            slot_mapping: 各 token 的 cache 槽位（None 或空则跳过）。
            attn: 提供目标 cache 层（swa_cache_layer）的注意力模块。
        """
        if slot_mapping is None or slot_mapping.numel() == 0:
            return

        assert attn is not None
        # 目标层的 SWA cache 伪层。
        swa_cache_layer = attn.dsa_attn.swa_cache_layer
        swa_kv_cache = getattr(swa_cache_layer, "kv_cache", None)
        if swa_kv_cache is None:
            return
        # 兼容 vLLM 新版把 kv_cache 包在单元素 list/tuple 里的情况:
        # isinstance 连续解包直到拿到真正的张量。
        while isinstance(swa_kv_cache, (list, tuple)) and len(swa_kv_cache) == 1:
            swa_kv_cache = swa_kv_cache[0]

        # 懒导入避免循环依赖。
        from vllm_ascend.attention.dsa_attn_kv_plan import get_dsa_attn_kv_plan

        if slot_mapping.ndim == 1:
            # 一维 slot_mapping 需格式化为 (block_idx, in_block_offset)
            # 的物理坐标，与 NPU cache 布局匹配。
            slot_mapping = get_dsa_attn_kv_plan(self.vllm_config).format_dsa_slot_mapping(
                slot_mapping, swa_cache_layer.block_size
            )
        # 分散写入: 按 slot_mapping 把 shared_kv 写入 cache。
        get_dsa_attn_kv_plan(self.vllm_config).dsa_kv_compress_scatter(swa_kv_cache, shared_kv, slot_mapping)

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: list[torch.Tensor | None] | None = None,
    ) -> None:
        """预计算并写入草稿的上下文 KV（propose 前调用一次）。

        原理: DSpark 属于 EAGLE3 风格——目标模型的上下文信息以“隐状态”
        形式给出。这里把草稿上下文（目标选中层隐状态的投影）经各草稿层
        的 wkv 投影为共享 KV 并写入各自 SWA cache，使草稿块内的注意力
        能“看到”目标上下文。

        Args:
            context_states: [num_context_tokens, hidden] 草稿上下文。
            context_positions: [num_context_tokens] 上下文位置 id。
            context_slot_mapping: 每层各自的 slot 映射列表（None 跳过）。
        """
        if context_states.numel() == 0 or context_slot_mapping is None:
            return
        # 收集所有草稿层的 RoPE 层名，一次性预计算各层的 RoPE 表。
        rope_layers = [layer.self_attn.rotary_emb.layername for layer in self.layers.values()]
        rope = get_cos_and_sin_dsa(
            context_positions,
            layer_names=rope_layers,
        )
        # 逐层投影共享 KV 并写入该层 cache。
        for layer_idx, layer in enumerate(self.layers.values()):
            layer_context_slot_mapping = None if context_slot_mapping is None else context_slot_mapping[layer_idx]
            if context_positions.numel() == 0:
                return
            attn = layer.self_attn
            shared_kv = self._project_shared_kv(
                context_states,
                context_positions,
                attn,
                rope=rope,
            )
            self._store_standard_swa_kv(shared_kv, layer_context_slot_mapping, attn)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """草稿模型前向: 输出整个 draft block 的多路隐状态。

        Args:
            input_ids: [num_tokens] 草稿块 token id（含多候选并行展开）。
            positions: [num_tokens] 位置 id。
        Returns:
            [num_tokens, hc_mult, hidden] 多路隐状态（hc_head 尚未混合）。
        """
        # 步骤1: 词嵌入 -> 复制为 hc_mult 路 [N, hc_mult, H]。
        hidden_states = self.embed_tokens(input_ids).unsqueeze(-2).repeat(1, self.hc_mult, 1)
        full_num_tokens = positions.shape[0]
        use_sp = self.use_sequence_parallel_moe
        # SP 分支: 与 mtp.py 相同——临时切分 is_padding 掩码与输入张量
        # （VLLM_MOE_SKIP_PADDING 打开时跳过纯 padding token 的 MoE 计算）。
        orig_is_padding = None
        forward_context = None
        if use_sp:
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                orig_is_padding = forward_context.is_padding
                forward_context.is_padding = sp_padding_mask(orig_is_padding, hidden_states)
            # token 维切分到各 TP rank。
            hidden_states = sp_shard(hidden_states)
            input_ids = sp_shard(input_ids)

        # 步骤2: 逐层过草稿 decoder 层; input_ids 透传供 hash MoE 路由。
        residual = None
        for layer in self.layers.values():
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                llama_4_scaling=None,
                input_ids=input_ids,
            )
        if use_sp:
            # SP 结束: all-gather 恢复完整 token 维并裁掉对齐 pad。
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            hidden_states = hidden_states[:full_num_tokens]

        # 恢复原 is_padding（防止泄漏到下一次 forward）。
        if forward_context is not None:
            forward_context.is_padding = orig_is_padding
        # 步骤3: hc_head 混合多路隐状态为单路（confidence/采样用）。
        head_hidden = self.hc_head(hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
        return head_hidden

    def hc_head(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        """Hyper-Connections 混合头（与 mtp.py 中同名方法一致的算法）。

        原理: sigmoid(fn(norm(x))*scale + base) + eps 作为各路门控权重，
        对原始多路输入加权求和。全程 float32。

        Args:
            x: [N, hc_mult, H] 多路隐状态。
        Returns:
            [N, H] 混合后单路隐状态（原 dtype）。
        """
        shape, dtype = x.size(), x.dtype
        # 展平 + float32。
        x = x.flatten(1).float()
        # 就地 RMSNorm（rsqrt 公式，含 norm_eps）。
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        # 门控系数（与 mtp.py 不同: 用未归一化 x 直接过 fn，再乘 rsqrt）。
        mixes = torch.nn.functional.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        # 加权求和: [N, hc_mult, 1] * [N, hc_mult, H] -> 沿路维求和。
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
        return y.to(dtype)

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """马尔可夫嵌入: 由 token id 查 markov_head 的低秩嵌入表。

        Args:
            token_ids: [num_tokens] 上一个 token 的 id。
        Returns:
            [num_tokens, dspark_markov_rank] 马尔可夫嵌入。
        """
        return self.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor, logits_processor: LogitsProcessor) -> torch.Tensor:
        """马尔可夫偏置: 由马尔可夫嵌入计算 logits 的加性先验偏置。

        Args:
            markov_embed: [num_tokens, markov_rank]。
            logits_processor: 用于按 TP 切分方式融合偏置。
        Returns:
            [num_tokens, draft_vocab] 偏置张量。
        """
        return self.markov_head.bias(markov_embed, logits_processor)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: ParallelLMHead,
        logits_processor: LogitsProcessor,
    ) -> torch.Tensor:
        """由（已 hc_head 混合的）隐状态计算 draft logits。

        Args:
            hidden_states: [N, hidden]。
            lm_head: 输出投影。
            logits_processor: logits 后处理（TP gather/采样温度等）。
        Returns:
            [N, vocab] logits。
        """
        return logits_processor(lm_head, self.norm(hidden_states))

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """专家参数映射（供权重加载），四元组 (param_name, weight_name,
        expert_id, shard_id) 列表。"""
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=0,
        )


@support_torch_compile
class DSparkDeepseekV4ForCausalLM(nn.Module, DeepseekV2MixtureOfExperts, SupportsEagle3):
    """DSpark 草稿模型顶层（投机采样 drafter 入口）。

    多重继承:
      - DeepseekV2MixtureOfExperts: MoE 元数据混入（定义在 model.py，
        提供专家数量等属性供 EPLB/调度器查询）;
      - SupportsEagle3: EAGLE3 式投机采样接口（compute_draft_logits /
        compute_confidence / precompute_and_store_context_kv 等）。
    装饰器 @support_torch_compile: 语法点——标记该类参与 vLLM 的
    torch.compile / piecewise 编译（按注意力等边界切分编译区域）。
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        """初始化。

        Args:
            vllm_config: 全局配置（speculative_config 必须存在）。
            prefix: 前缀。
        """
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.config = vllm_config.speculative_config.draft_model_config.hf_config

        # check if quant config exist
        # 【中文】量化模型的权重“旋转”路径（如 fp8 per-tensor scale 的
        # 在线重标定），非量化模型为 None。
        from vllm_ascend.utils import get_rotation_path

        self.rotation_path = get_rotation_path(vllm_config) if vllm_config.quant_config is not None else None

        # 草稿主体 + 输出头。
        self.model = DeepseekV4DSparkModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        # 汇总 MoE 元数据（EPLB 用）。
        self.set_moe_parameters()

    def set_moe_parameters(self) -> None:
        """遍历草稿层收集 MoE 模块并提取专家元数据（同 mtp.py 逻辑）。"""
        self.expert_weights: typing.MutableSequence[typing.Sequence[torch.Tensor]] = []
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.moe_layers: list[nn.Module] = []
        self.moe_mlp_layers: list[DeepseekV4MoE] = []
        example_moe = None
        for layer in self.model.layers.values():
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, DeepseekV4DecoderLayer)
            if isinstance(layer.mlp, DeepseekV4MoE):
                # Pick last one layer since the first ones may be dense layers.
                # 【中文】持续覆盖，留下最后一个 MoE 层作为样例。
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
        """草稿前向（委托内部 model）。Args: 见 DeepseekV4DSparkModel.forward。
        inputs_embeds 未使用（DSpark 用自己的 embed_tokens）。"""
        return self.model(
            input_ids=input_ids,
            positions=positions,
        )

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """由草稿隐状态计算 draft logits（EAGLE3 接口）。
        全词表草稿: 无 draft->target 词表散射（d2t scatter）。

        Args:
            hidden_states: [N, hidden]（已 hc_head 混合）。
        Returns:
            [N, vocab] draft logits。
        """
        # Full-vocab draft: base logits, no d2t scatter.
        # 【中文】全词表: 直接用基础 logits（无需 d2t 散射）。
        return self.compute_logits(hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        """把草稿 token id 映射到目标词表 id。全词表草稿: 恒等映射。"""
        return draft_ids  # full-vocab: draft ids are target ids

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        """由隐状态计算 logits（委托内部 model）。

        Args:
            hidden_states: [N, hidden]。
            spec_step_idx: 投机步号（未用，del 丢弃以保持接口一致）。
        """
        del spec_step_idx
        return self.model.compute_logits(
            hidden_states,
            self.lm_head,
            self.logits_processor,
        )

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """马尔可夫嵌入（委托内部 model）。"""
        return self.model.markov_embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        """马尔可夫偏置（委托内部 model）。"""
        return self.model.markov_bias(markov_embed, self.logits_processor)

    def compute_confidence(self, head_hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        """Per-position acceptance probability for each drafted token."""
        # 【中文】接受概率: confidence_head(hidden 与 markov 嵌入拼接) 过
        # sigmoid 得到 [0,1] 的每位置概率，verify 阶段据此决定接受长度
        # （自适应投机采样: 草稿可信处多接受，不可信处提前截断）。
        assert self.model.confidence_head is not None
        return torch.sigmoid(self.model.confidence_head(head_hidden, markov_embed))

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        """草稿 KV cache 伪层名列表（委托内部 model）。"""
        return self.model.get_draft_kv_cache_layer_names()

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        """目标选中层隐状态 -> 草稿上下文（委托内部 model）。"""
        return self.model.combine_hidden_states(aux_hidden_states)

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: list[torch.Tensor | None] | None = None,
    ) -> None:
        """预计算并写入草稿上下文 KV（委托内部 model）。

        Args: 见 DeepseekV4DSparkModel.precompute_and_store_context_kv。
        """
        self.model.precompute_and_store_context_kv(
            context_states,
            context_positions,
            context_slot_mapping,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the ``mtp.{i}.*`` draft weights from the target checkpoint.

        Non-MTP weights belong to the target model and are skipped, except for
        standalone embedding/head weights used by the Ascend draft loader.
        """
        # 【中文】加载 DSpark 草稿权重: 只认 checkpoint 的 mtp.{i}.* 命名
        # 空间（经 _remap_dspark_name 重写到本模型模块树），其余权重属于
        # 目标模型一律跳过。专家权重经 expert_mapping 分派; 堆叠权重
        # （gate/up 合并）经 stacked_params_mapping 分派; attention sink
        # 按头切分（或 DSA-CP 全量）。
        expert_mapping = self.model.get_expert_mapping()

        # (param_name, checkpoint shard name, shard_id) for non-expert
        # stacked parameters. Ascend keeps wq_a and wkv as separate parameters.
        # 【中文】非专家堆叠参数映射: gate/up 两片合并进 gate_up_proj。
        # （Ascend 上 wq_a 与 wkv 保持独立参数，不做 QKV 融合。）
        stacked_params_mapping = [
            ("mlp.gate_up_proj", "mlp.gate_proj", 0),
            ("mlp.gate_up_proj", "mlp.up_proj", 1),
            ("shared_experts.gate_up_proj", "shared_experts.gate_proj", 0),
            ("shared_experts.gate_up_proj", "shared_experts.up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        # attention sink 的按头切分参数。
        n_local_head = self.config.num_attention_heads // tp_size
        head_start = n_local_head * tp_rank
        head_end = n_local_head * (tp_rank + 1)

        for name, loaded_weight in weights:
            # ---- 顶层简名与 hc_head 参数的重命名 ----
            if name == "embed.weight" and not self.rotation_path:
                name = "model.embed_tokens.weight"
            elif name == "head.weight" and not self.rotation_path:
                name = "lm_head.weight"
            elif name in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
                name = f"model.{name}"
            else:
                # mtp.{stage}.* 命名空间重写; 非 mtp 权重（目标模型）跳过。
                mapped_name = self._remap_dspark_name(name)
                if mapped_name is None:
                    continue
                name = mapped_name

            # Detect whether the checkpoint ships its own embed_tokens / lm_head
            # for the draft model.
            # 【中文】探测 checkpoint 是否为草稿单独提供 embed/lm_head
            #（EAGLE3 风格: 无则共享目标模型权重）。
            process_eagle_weight(self, name)

            # Expert scale parameters use Ascend's ``weight_scale`` convention.
            # 【中文】Ascend 约定: 专家 scale 参数名 -> weight_scale。
            if name.endswith(".scale"):
                name = name.replace(".scale", ".weight_scale")

            # The multimodal checkpoint also contains one vision-router bias
            # for each MTP/DSpark layer.  DSpark runs only during text decode,
            # so draft MoE gates intentionally do not expose ``bias_vl``.
            # Do not alias it to the text correction bias: that would change
            # text routing whenever speculative decoding is enabled.
            # 【中文】多模态 checkpoint 里每个草稿层还带一个 vision-router
            # bias; DSpark 只在纯文本解码时运行，草稿路由门不暴露 bias_vl，
            # 也不能把它别名为文本纠偏（否则投机采样会改变文本路由结果）
            # ——直接跳过。logger.info_once: 同一消息只打一次日志。
            if name.endswith(".e_score_correction_bias_vl") and name not in params_dict:
                logger.info_once("Ignoring vision-only router bias while loading the text-only DSpark drafter")
                continue

            # ---- 专家权重分派 ----
            if ".experts." in name:
                for param_name, weight_name, expert_id, shard_id in expert_mapping:
                    if weight_name not in name:
                        continue
                    name_mapped = name.replace(weight_name, param_name)
                    param = params_dict[name_mapped]
                    # 带 expert_id 的专家加载器; return_success 用于
                    # EPLB 冗余专家副本场景。
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

            # ---- 堆叠权重 / 普通权重分派 ----
            # Stacked rules only apply to decoder-layer weights. Head-stack
            # parameters load directly through the fallback below.
            # 【中文】堆叠规则只作用于 decoder 层权重; 头部堆叠参数走
            # 下方 fallback 直接加载。
            is_layer_param = name.startswith("model.layers.")
            for param_name, weight_name, stacked_shard_id in stacked_params_mapping:
                if not is_layer_param or f".{weight_name}." not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, stacked_shard_id)
                loaded_params.add(name)
                break
            else:
                # for...else: 不匹配堆叠规则时的普通加载路径。
                if "attn_sink" in name:
                    # attention sink: DSA-CP 时全量，否则按本 rank 的
                    # 头区间切片; with torch.no_grad(): 禁止梯度追踪的
                    # 纯拷贝（参数本就不训练）。
                    if enable_dsa_cp():
                        narrow = loaded_weight
                    else:
                        narrow = loaded_weight[head_start:head_end]
                    with torch.no_grad():
                        params_dict[name].copy_(narrow)
                    loaded_params.add(name)
                    continue
                # 普通参数: 有专用 loader（量化）用之，否则默认拷贝。
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        logger.info_once("DSpark draft model loaded: %d params", len(loaded_params))
        return loaded_params

    def _remap_dspark_name(self, name: str) -> str | None:
        """把 checkpoint 的 ``mtp.{stage}.{rest}`` 名字重写到本模型模块树。

        规则:
          - 非法（非 mtp.*）返回 None（跳过）;
          - 末层的 confidence_head 与 head、首层的 embed、hc_head_* ->
            提升到顶层（model.*/lm_head*）;
          - main_proj/main_norm 归第一草稿层; norm/markov_head 归末层;
          - 其余按 stage 偏移到对应草稿层;
          - 最后做 DeepSeek 原始命名 -> vLLM 命名的批量替换。

        Args:
            name: checkpoint 权重名。
        Returns:
            重写后的参数名; 非草稿权重返回 None。
        """
        # 正则匹配 mtp.{数字}.{其余}，re.match 从头匹配。
        m = re.match(r"mtp\.(\d+)\.(.*)", name)
        if m is None:
            return None
        stage = int(m.group(1))
        rest = m.group(2)

        # 末层的 confidence_head -> 顶层 model.confidence_head.*。
        if stage == self.model.num_dspark_layers - 1 and rest.startswith("confidence_head."):
            return f"model.{rest}"

        # 首层 embed / 末层 head -> 顶层 embed_tokens / lm_head。
        if stage == 0 and rest == "embed.weight":
            return "model.embed_tokens.weight"
        if stage == self.model.num_dspark_layers - 1 and rest == "head.weight":
            return "lm_head.weight"
        # hc_head 参数 -> 顶层 model.*。
        if rest.startswith(("hc_head_fn", "hc_head_base", "hc_head_scale")):
            return f"model.{rest}"

        # 其余权重按“归属层”重写:
        # main_proj/main_norm -> 第一草稿层; norm/markov_head -> 末层;
        # 其它 -> 第 stage 个草稿层。
        first_layer_idx = self.config.num_hidden_layers
        last_layer_idx = first_layer_idx + self.model.num_dspark_layers - 1
        if rest.startswith(("main_proj.", "main_norm.")):
            layer_idx = first_layer_idx
        elif rest.startswith(("norm.", "markov_head.")):
            layer_idx = last_layer_idx
        else:
            layer_idx = first_layer_idx + stage
        name = f"model.layers.{layer_idx}.{rest}"

        # DeepSeek 原始命名 -> vLLM 命名的批量替换表。
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
