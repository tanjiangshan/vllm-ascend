# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# GLM-5.Next 的 MTP（Multi-Token Prediction，多 token 预测）投机解码草稿模型。
#
# MTP 原理（DeepSeek-V3 MTP 风格 + GLM 定制）：
#   - 目标模型每步产出 1 个 token；MTP 层利用目标模型的隐状态"额外"
#     预测未来的 num_nextn_predict_layers 个 token（草稿 draft）；
#   - 解码流程：草稿模型快速生成 k 个候选 token -> 目标模型一次前向
#     验证（verify）全部候选 -> 拒绝采样决定接受多少个 -> 接受的 token
#     一次提交。接受长度 >1 时吞吐显著提升；
#   - 输入融合：eh_proj(cat(enorm(embeds), hnorm(prev_hidden)))，
#     其中位置 0 的 embeds 置零（首 token 无前文）；该融合在 NPU 上由
#     fused_eh_norm Triton 内核一次完成；
#   - 草稿层复用 Glm5NextDecoderLayer（MLA/DSA 形态，非 KDA），
#     共享缓冲 topk_indices_buffer 与主模型一致（宽度对齐 BLOCK_N=128）；
#   - set_skip_topk/compact_topk_indices 支持多步草稿间的 top-k 复用
#     与槽位压缩（投机验证后只保留被接受 token 的索引行）。
#
# 类结构：
#   Glm5NextMultiTokenPredictorLayer —— 单个 MTP 层（enorm/hnorm/eh_proj
#       + mtp_block(DecoderLayer) + shared_head）；
#   Glm5NextMultiTokenPredictor     —— MTP 层容器（多步轮转）+ 词嵌入
#       + logits_processor；
#   Glm5NextMTP                     —— 对外包装（继承 DeepseekV2MixtureOfExperts
#       的 MoE 参数管理接口），含权重名重写与加载。
# =============================================================================
import typing
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.deepseek_mtp import SharedHead
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MixtureOfExperts
from vllm.model_executor.models.utils import maybe_prefix
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from vllm_ascend.utils import is_rot_weight_used

from .model import (
    Glm5NextDecoderLayer,
    Glm5NextMLAAttention,
    Glm5NextMoE,
    _try_load_fp8_indexer_wk,
    get_spec_layer_idx_from_weight_name,
)
from .ops.fused_eh_norm import fused_eh_norm


class Glm5NextMultiTokenPredictorLayer(nn.Module):
    """单个 MTP 预测层：输入融合 + 一个解码层 + 共享输出头。

    结构：
      embeds(本步输入) ┐
                       ├ fused_eh_norm -> eh_proj -> mtp_block(DecoderLayer)
      prev_hidden ─────┘                                 │
                                     shared_head.norm（融合残差加+RMSNorm）
                                     shared_head.head（LM 头，输出 logits）
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        """初始化 MTP 层。

        参数：
            vllm_config: vLLM 全局配置（speculative_config 必须存在）。
            prefix: 层名前缀（形如 "model.layers.45"，含层号）。
        """
        super().__init__()
        # 语法点：断言草稿配置存在；draft_model_config.hf_config 是
        # 与主模型同结构的配置（MTP 层与主模型共享超参）。
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config

        # enorm/hnorm：对 embeds 与 prev_hidden 分别做 RMSNorm，
        # eh_proj 把拼接结果 [N, 2H] 投回 [N, H]。
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)

        # Reserve room for the incomplete pool tail and align the sparse MLA
        # buffer width to BLOCK_N=128.
        # topk 缓冲宽度 = topk + 未满池余量(kpool-1)，向上对齐到 128
        # （稀疏 MLA 内核按 BLOCK_N=128 分块）。
        topk_tokens = config.index_topk
        assert topk_tokens is not None
        kpool = config.index_kpool
        assert kpool is not None
        buffer_width = topk_tokens + (kpool - 1 if kpool > 1 else 0)
        sparse_topk_block_n = 128
        buffer_width = ((buffer_width + sparse_topk_block_n - 1) // sparse_topk_block_n) * sparse_topk_block_n
        # 共享 top-k 索引缓冲：[max_num_batched_tokens, buffer_width] int32。
        topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            buffer_width,
            dtype=torch.int32,
            device=current_platform.device_type,
        )
        # 共享输出头（含最终 norm 与 LM head，结构与 DeepSeek MTP 相同）。
        self.shared_head = SharedHead(config=config, prefix=prefix, quant_config=quant_config)
        # MTP layers sit past the base model's hidden layers; parse the index
        # from the prefix (e.g. "...layers.32") so the decoder builds an MLA
        # (DSA) layer rather than KDA for the MTP path.
        # 从前缀解析层号（如 "...layers.45"），使解码层构建为 MLA(DSA)
        # 而非 KDA——MTP 层位于主模型隐藏层之后。
        layer_idx = int(prefix.rsplit(".", 1)[-1])
        self.mtp_block = Glm5NextDecoderLayer(
            vllm_config=vllm_config,
            config=config,
            layer_idx=layer_idx,
            prefix=prefix,
            topk_indices_buffer=topk_indices_buffer,
            is_mtp_layer=True,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        """MTP 层前向。

        参数：
            input_ids: [N] token id（当前未直接使用，嵌入由调用方提供）。
            positions: [N] token 位置（位置 0 的 embeds 会被置零）。
            previous_hidden_states: [N, H] 主模型上一步隐状态。
            inputs_embeds: [N, H] 本步输入嵌入（必须提供）。
            spec_step_index: 投机步序号（本层未直接使用，接口保留）。

        返回：
            (hidden_states, hidden_states)：元组两项相同——既是草稿 logits
            的输入（LM 头），也作为下一步 MTP 的 recycled 隐状态。
        """
        assert inputs_embeds is not None
        # Fused: zero pos-0 embeds + enorm(embeds) + hnorm(prev) + cat -> [N, 2H].
        # 步骤1: Triton 融合内核一次完成：位置 0 的 embeds 置零 +
        # enorm(embeds) + hnorm(prev_hidden) + 拼接 -> [N, 2H]。
        eh_input = fused_eh_norm(
            positions,
            inputs_embeds,
            previous_hidden_states,
            self.enorm.weight,
            self.hnorm.weight,
            self.enorm.variance_epsilon,
        )
        # 步骤2: eh_proj 投回 [N, H]。
        hidden_states = self.eh_proj(eh_input)
        # Fuse the residual add and final RMSNorm. Glm5NextMoE already performs
        # its all-reduce, so no collective is needed here. The post-norm result
        # feeds both draft logits and the next recycled hidden state.
        # 步骤3: 解码层前向。MTP 层返回"未求和"的 (mlp_out, residual)，
        # 由 shared_head 的融合内核完成残差加+最终 RMSNorm（Glm5NextMoE
        # 内部已做 all-reduce，无需再集合通信）。归一化结果既供草稿
        # logits 也作为下一步的 recycled 隐状态。
        hidden_states, residual, _, _ = self.mtp_block(positions=positions, hidden_states=hidden_states, residual=None)
        hidden_states, _ = self.shared_head.norm(hidden_states, residual=residual)
        return hidden_states, hidden_states


class Glm5NextMultiTokenPredictor(nn.Module):
    """MTP 预测器：num_nextn_predict_layers 个 MTP 层的容器 + 词嵌入 + logits。

    多步轮转：第 s 步（spec_step_idx）使用第 s % num_mtp_layers 层——
    多个 MTP 层各自负责一步草稿，循环复用。
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        """初始化 MTP 预测器。

        参数（语法点：* 为 keyword-only）：
            vllm_config: vLLM 全局配置。
            prefix: 模块名前缀。
        """
        super().__init__()
        config = vllm_config.model_config.hf_config
        # MTP 层从主模型隐藏层之后开始编号。
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        # 语法点：字典推导 + ModuleDict——按字符串键注册子模块，
        # 键名与 checkpoint 的 layers.<idx> 对应。
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): Glm5NextMultiTokenPredictorLayer(vllm_config, f"{prefix}.layers.{idx}")
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        # 草稿词嵌入（与主模型词表一致）。
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        # Plain list for the per-propose lookup: ModuleDict[str(...)] builds a
        # string and hashes it on every draft step.
        # 普通 list 做逐步查找：ModuleDict[str(...)] 每个草稿步都要构造
        # 字符串并哈希，list 索引更快。
        self._mtp_layers = list(self.layers.values())
        # 预提取各层的 MLA wrapper 引用（set_skip_topk 等热路径用）。
        self._mtp_mla_attns = []
        for layer in self._mtp_layers:
            self_attn = layer.mtp_block.self_attn
            assert isinstance(self_attn, Glm5NextMLAAttention)
            self._mtp_mla_attns.append(self_attn.mla_attn)
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def set_skip_topk(self, skip: bool):
        """控制稀疏索引 top-k 的复用（index_share_for_mtp_iteration 机制）。

        原理：投机步 0 计算 top-k，步 1+ 复用——给所有 MLA wrapper 设置
        skip_topk 标志，避免每步重复做索引器计算。
        """
        # index_share_for_mtp_iteration: step 0 computes top-k, steps 1+ reuse.
        for mla_attn in self._mtp_mla_attns:
            mla_attn.skip_topk = skip

    def compact_topk_indices(self, slot_ids: torch.Tensor):
        """Gather the top-k index rows at ``slot_ids`` to the front of the buffer.

        验证后压缩 top-k 索引缓冲：把被接受 token 对应的行 gather 到缓冲
        前部（拒绝的 token 行丢弃，后续步骤复用前部行）。

        参数：
            slot_ids: [num_accepted] 被接受 token 的行号。
        """
        num_slots = slot_ids.numel()
        for mla_attn in self._mtp_mla_attns:
            topk_indices_buffer = mla_attn.topk_indices_buffer
            assert topk_indices_buffer is not None
            # 原地 gather：前 num_slots 行 <- slot_ids 指定的行。
            topk_indices_buffer[:num_slots] = topk_indices_buffer[slot_ids]

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """草稿词嵌入查表：[N] -> [N, H]。"""
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        """MTP 预测器前向：按投机步选择对应 MTP 层执行。

        参数：
            input_ids: [N] token id（inputs_embeds 为 None 时现场嵌入）。
            positions: [N] 位置。
            previous_hidden_states: [N, H] 主模型（或上一步草稿）隐状态。
            inputs_embeds: 可选预计算嵌入。
            spec_step_idx: 投机步序号 s，用第 s % num_mtp_layers 层。

        返回：
            (hidden_states, hidden_states)，见 PredictorLayer.forward。
        """
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # 取模轮转选择当前步的 MTP 层。
        current_step_idx = spec_step_idx % self.num_mtp_layers
        return self._mtp_layers[current_step_idx](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        """计算当前步草稿 logits。

        参数：
            hidden_states: [N, H] 已最终归一化的隐状态。
            spec_step_idx: 投机步序号。

        返回：
            [N, vocab_size] logits。

        注意：输入已是 post-final-norm（层 forward 产出并原样回收），
        只做 LM 头投影，不再做第二次 RMSNorm。
        """
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self._mtp_layers[current_step_idx]
        # hidden_states is already post-final-norm (produced in the layer
        # forward and recycled as-is); apply the LM head only, without a
        # second RMSNorm.
        return self.logits_processor(mtp_layer.shared_head.head, hidden_states)

    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        """贪心草稿 token：vocab-并行 argmax，不物化全词表 logits。

        原理：每 rank 对本地词表分片投影 + argmax，再做 [batch, 2*tp]
        的 (value, index) 归约。平票裁决与全局 argmax 一致（分片连续且
        按 rank 有序，低 rank 的胜者是更小的全局下标），贪心草稿不变。

        参数：
            hidden_states: [N, H] 隐状态。
            spec_step_idx: 投机步序号。

        返回：
            [N] 草稿 token id。
        """
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self._mtp_layers[current_step_idx]
        # Vocab-parallel argmax for the greedy draft: per-rank head projection
        # + local argmax + a [batch, 2*tp] (value, index) reduce, instead of
        # materializing and all-gathering full [N, vocab] logits per draft
        # step. Tie-breaking matches the full argmax (shards are contiguous
        # and rank-ordered, so the lowest-rank winner is the lowest global
        # index), so greedy draft tokens are unchanged.
        return self.logits_processor.get_top_tokens(mtp_layer.shared_head.head, hidden_states)


class Glm5NextMTP(nn.Module, DeepseekV2MixtureOfExperts):
    """GLM-5.Next MTP 草稿模型对外包装（vLLM 投机解码接口）。

    继承 DeepseekV2MixtureOfExperts（语法点：混入其 MoE 专家参数管理
    方法 extract_moe_parameters 等，供 EPLB/专家并行使用），组合
    Glm5NextMultiTokenPredictor 为 self.model。
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        """初始化 MTP 包装。

        参数：
            vllm_config: vLLM 全局配置。
            prefix: 模块名前缀。
        """
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = Glm5NextMultiTokenPredictor(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        # QuaRot 旋转修正（量化 checkpoint 导出的输入变换）。
        self.is_rot_weight_used = is_rot_weight_used(vllm_config)
        if self.is_rot_weight_used:
            self.rot = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)
        self.set_moe_parameters()

    def set_moe_parameters(self):
        """从 MTP 层中提取 MoE 专家参数（供专家并行/EPLB 管理）。

        遍历各 MTP 层的 mlp，收集 Glm5NextMoE 实例与专家模块列表，
        然后交给（继承自 DeepseekV2MixtureOfExperts 的）
        extract_moe_parameters 统一登记。
        """
        self.num_moe_layers = self.config.num_nextn_predict_layers
        self.num_expert_groups = self.config.n_group
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers.values():
            mlp = layer.mtp_block.mlp
            if isinstance(mlp, Glm5NextMoE):
                example_moe = mlp
                self.moe_mlp_layers.append(mlp)
                self.moe_layers.append(mlp.experts)
        self.extract_moe_parameters(example_moe)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """词嵌入查表（透传）。"""
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        """MTP 前向：可选 rot 修正后委托 Glm5NextMultiTokenPredictor。

        参数：
            input_ids: [N] token id（可 None，用 inputs_embeds）。
            positions: [N] 位置。
            hidden_states: [N, H] 主模型隐状态。
            intermediate_tensors: PP 中间张量（MTP 单层一般不用）。
            inputs_embeds: 可选预计算嵌入。
            spec_step_idx: 投机步序号。
        """
        # Apply the checkpoint-exported correction before MTP's hnorm,
        # matching the quantized AscendDeepSeekMTP input path.
        # 在 MTP 的 hnorm 之前应用 checkpoint 导出的 rot 修正
        # （与量化的 AscendDeepSeekMTP 输入路径一致）。
        if self.is_rot_weight_used:
            hidden_states = self.rot(hidden_states)
        return self.model(input_ids, positions, hidden_states, inputs_embeds, spec_step_idx)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        """计算草稿 logits（透传）。"""
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        """贪心草稿 token（透传；use_local_argmax_reduction 启用时的路径）。"""
        # Greedy-draft path used when use_local_argmax_reduction is enabled:
        # vocab-parallel argmax, no full-vocab logits.
        return self.model.get_top_tokens(hidden_states, spec_step_idx)

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        """重写 checkpoint 的 MTP 权重名到运行时模块结构。

        规则：
          - 属于层私有的权重（enorm/hnorm/eh_proj/shared_head）：
            "model.layers.N." -> "model.layers.N.mtp_block."；
          - 共享权重（embed_tokens）："model.layers.N." -> "model."；
          - 其余（解码层内部权重）不改写（mtp_block 内部结构同名）。

        参数：
            spec_layer: MTP 层下标。
            name: 原始权重名。

        返回：
            str: 重写后的权重名。
        """
        # 层私有权重名列表与共享权重名列表。
        spec_layer_weight_names = [
            "embed_tokens",
            "enorm",
            "hnorm",
            "eh_proj",
            "shared_head",
        ]
        shared_weight_names = ["embed_tokens"]
        spec_layer_weight = False
        shared_weight = False
        for weight_name in spec_layer_weight_names:
            if weight_name in name:
                spec_layer_weight = True
                if weight_name in shared_weight_names:
                    shared_weight = True
                break
        if not spec_layer_weight:
            name = name.replace(f"model.layers.{spec_layer}.", f"model.layers.{spec_layer}.mtp_block.")
        elif shared_weight:
            name = name.replace(f"model.layers.{spec_layer}.", "model.")
        return name

    def _maybe_set_own_lm_head(self, loaded_weights: set[str]) -> None:
        """Record whether the checkpoint shipped an MTP head.

        Glm5NextMultiTokenPredictorLayer always constructs ``shared_head``, so
        module existence does not prove head ownership. GLM-5.3-Flash ships no
        MTP head, leaving ``shared_head.head`` at its allocation-time contents;
        recording that lets the proposer share the target ``lm_head`` instead of
        deciding from those values.

        记录 checkpoint 是否携带 MTP 专属 LM 头。

        原理：Glm5NextMultiTokenPredictorLayer 总是构造 shared_head，
        模块存在不代表权重属于它。GLM-5.3-Flash 不带 MTP 头，
        shared_head.head 保持分配时的初始值；记录该事实让提议器改为
        共享目标模型的 lm_head，而不是根据（未加载的）权重值判断。

        参数：
            loaded_weights: 已加载的权重名集合。
        """
        own_head_weight = f"model.layers.{self.model.mtp_start_layer_idx}.shared_head.head.weight"
        self.has_own_lm_head = own_head_weight in loaded_weights

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """MTP 权重加载（含名字重写、堆叠/专家映射、FP8 索引器反量化）。

        参数：
            weights: (name, tensor) 迭代器。

        返回：
            set[str]: 成功加载的权重名集合。

        异常：
            - MTP 层权重缺失时 ValueError；
            - is_rot_weight_used 但缺 rot.weight 时 ValueError。
        """
        # 堆叠映射：融合 gate/up、q_a/kv_a、wk/weights_proj。
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("fused_qkv_a_proj", "q_a_proj", 0),
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
            ("wk_weights_proj", "wk", 0),
            ("wk_weights_proj", "weights_proj", 1),
        ]
        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        # FP8 索引器 WK 待配对缓冲（同主模型 load_weights）。
        _pending_wk_fp8: dict = {}
        # 步骤1: 逐权重循环。
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            # This is an MTP input transform, not a decoder-layer weight.
            # Handle it before filtering out weights outside the MTP layers.
            # rot 是 MTP 输入变换（非解码层权重），须在过滤非 MTP 层
            # 权重之前处理。
            if name == "rot.weight":
                if self.is_rot_weight_used:
                    default_weight_loader(self.rot.weight, loaded_weight)
                    loaded_params.add(name)
                continue
            # Multimodal (Glm5NextForConditionalGeneration) checkpoints prefix
            # the text-tower weights with "model.language_model."; the MTP head
            # is built as a text-only model (model.layers.*), so strip the
            # prefix to match.
            # 多模态 checkpoint 的文本塔权重带 "model.language_model." 前缀；
            # MTP 头按纯文本模型构建（model.layers.*），需剥掉前缀匹配。
            if name.startswith("model.language_model."):
                name = name.replace("model.language_model.", "model.", 1)
            # 非 MTP 层权重：跳过。
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is None:
                continue
            # MTP 权重名重写（enorm/shared_head 等改挂 mtp_block 或 model）。
            name = self._rewrite_spec_layer_name(spec_layer, name)

            # FP8 索引器 WK：配对后反量化融合（同主模型逻辑）。
            if _try_load_fp8_indexer_wk(
                name,
                loaded_weight,
                _pending_wk_fp8,
                params_dict,
                loaded_params,
            ):
                continue

            # 步骤2: 堆叠权重匹配（for-else：未匹配则进入专家/普通路径）。
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # 专家权重由 expert_params_mapping 处理，先跳过。
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                # 融合模块不存在（如无 q 低秩的 fused_qkv_a_proj）则跳过。
                if (param_name == "fused_qkv_a_proj") and name_mapped not in params_dict:
                    continue
                else:
                    name = name_mapped
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # 步骤3: 专家权重匹配（带 return_success 的探测式加载）。
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping  # type: ignore[assignment]
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    param = params_dict[name_mapped]
                    # typing.cast：把 weight_loader 断言为可调用对象。
                    weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
                    # return_success=True：加载器返回是否成功（名字可能
                    # 属于其它 MoE 配置），失败则继续尝试下一个映射。
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        name = name_mapped
                        break
                else:
                    # 步骤4: 普通权重加载（KV scale 重映射 + 首层之外的
                    # 顶层共享权重要求带 .layers 名）。
                    if is_expert_weight:
                        continue
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    name = maybe_remap_kv_scale_name(name, params_dict)  # type: ignore[assignment]
                    if name is None:
                        continue
                    # 非首个 MTP 层的权重必须带 ".layers"（防止误加载
                    # 顶层共享模块的重复名字）。
                    if spec_layer != self.model.mtp_start_layer_idx and ".layers" not in name:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)

        # 步骤5: 完整性校验——每个 MTP 层都必须有权重加载，缺一报错；
        # rot 启用时 rot.weight 必须存在。
        loaded_layers: set[int] = set()
        for param_name in loaded_params:
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, param_name)
            if spec_layer is not None:
                loaded_layers.add(spec_layer)
        for layer_idx in range(
            self.model.mtp_start_layer_idx,
            self.model.mtp_start_layer_idx + self.model.num_mtp_layers,
        ):
            if layer_idx not in loaded_layers:
                raise ValueError(f"MTP speculative decoding layer {layer_idx} weights missing from checkpoint.")
        if self.is_rot_weight_used and "rot.weight" not in loaded_params:
            raise ValueError("GLM5-Next MTP requires rot.weight when the quantization config sets is_rot_used.")
        self._maybe_set_own_lm_head(loaded_params)
        return loaded_params
