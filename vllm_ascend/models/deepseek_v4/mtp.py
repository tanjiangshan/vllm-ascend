# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 MTP（Multi-Token Prediction，多 Token 预测）草稿模型的昇腾实现。

MTP 原理（DeepSeek-V3 技术报告提出）: 在目标模型之后串联若干“草稿层”，
每层以上一层的目标隐状态 + 当前 token 嵌入为输入，预测下一个 token。
推理时用作投机采样（speculative decoding）的草稿模型:
  1) propose 阶段: 草稿层快速串行地“吐出” k 个候选 token;
  2) verify 阶段: 目标模型一次前向并行打分，接受与目标分布一致的
     前缀 token，从而一步完成多个 token 的生成。

本文件的 DeepSeek V4 适配要点:
  - 草稿层复用 DeepseekV4DecoderLayer（model.py），共享目标模型的
    DSA 稀疏注意力结构（含 indexer/compressor）与 MoE;
  - Hyper-Connections（hc_*）: V4 的草稿输入是 [N, hc_mult, H] 的
    多路隐状态，hc_head 把多路混合为单路;
  - 支持 sequence parallel MoE（sp_shard/all_gather 切分 token 维）;
  - 支持 PP 流水线并行（SupportsPP）与 torch.compile
    （@support_torch_compile 装饰器）。

类结构:
  - SharedHead                  : 草稿共享输出头（RMSNorm + LM head）;
  - DeepSeekMultiTokenPredictorLayer: 单个 MTP 草稿层（e_proj/h_proj 投影
    + enorm/hnorm 归一化 + 一个 DecoderLayer + hc_head 参数）;
  - DeepSeekMultiTokenPredictor : 多层 MTP 的容器（ModuleDict 按层索引）;
  - DeepSeekV4MTP               : 顶层包装，实现 SupportsPP 与
    DeepseekV2MixtureOfExperts（MoE 元数据/EPLB 支持）。
"""
import typing
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn
import vllm.envs as envs
from transformers import PretrainedConfig
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader, maybe_remap_kv_scale_name
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.models.common.ops.sequence_parallel import sp_padding_mask, sp_shard
from vllm_ascend.models.deepseek_v4.model import (
    DeepseekV2MixtureOfExperts,
    DeepseekV4DecoderLayer,
    DeepseekV4MoE,
    get_spec_layer_idx_from_weight_name,
)
from vllm_ascend.utils import enable_dsa_cp


class SharedHead(nn.Module):
    """草稿模型共享输出头: RMSNorm + ParallelLMHead（词表并行投影）。

    DeepSeek-V3 报告指出 MTP 模块与目标模型共享 embedding/lm_head 权重，
    这里以独立模块形式承载（权重加载时由 checkpoint 的 head.weight 填充）。
    """

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        """初始化。

        Args:
            config: HF 模型配置。
            prefix: 模块名前缀。
            quant_config: 量化配置（可选）。
        """
        super().__init__()
        # 输出前的最终 RMSNorm。
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # ParallelLMHead: 词表按 TP 切片的输出投影（hidden -> vocab）。
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "head"),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """前向只做 RMSNorm（logits 计算由 LogitsProcessor 单独驱动，
        见 DeepSeekMultiTokenPredictor.compute_logits）。

        Args:
            hidden_states: [num_tokens, hidden_size]。
        Returns:
            归一化后的隐状态，同形状。
        """
        return self.norm(hidden_states)


class DeepSeekMultiTokenPredictorLayer(nn.Module):
    """单个 MTP 草稿层。

    结构（DeepSeek-V3 MTP 变体 + V4 Hyper-Connections）:
      input = e_proj(norm(embed(t))) + h_proj(norm(h_prev))
      其中 h_prev 是目标模型（或上一层草稿）的多路隐状态 [N, hc_mult, H]，
      两路投影相加后进入一个完整的 DeepseekV4DecoderLayer（复用目标层的
      DSA 注意力 + MoE 结构），输出同样为多路隐状态。
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        """初始化草稿层。

        Args:
            vllm_config: vLLM 全局配置（speculative_config 指向草稿模型配置）。
            prefix: 模块名前缀。
        """
        super().__init__()

        # 草稿层自己的 hf_config（speculative_config.draft_model_config）。
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        # v2 model runner 标志: 影响 forward 返回形状（是否 flatten 多路维）。
        self.use_v2_model_runner = vllm_config.use_v2_model_runner
        quant_config = vllm_config.quant_config

        # e_proj: token 嵌入路径投影 [N, H] -> [N, hc_mult*H]。
        self.e_proj = ReplicatedLinear(
            config.hidden_size, config.hidden_size, bias=False, quant_config=quant_config, return_bias=False
        )
        # h_proj: 上一层隐状态路径投影（同样升维到 hc_mult 路）。
        self.h_proj = ReplicatedLinear(
            config.hidden_size, config.hidden_size, bias=False, quant_config=quant_config, return_bias=False
        )

        # 两路输入各自的 RMSNorm（对多路隐状态逐元素归一化）。
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.device = current_platform.device_type

        # is_v32: 配置含 index_topk 即为 V3.2+ 式结构（带稀疏注意力 indexer）。
        self.is_v32 = hasattr(config, "index_topk")
        if self.is_v32:
            # 分配 TopK 索引缓冲: [max_num_batched_tokens, index_topk]，
            # int32。作用: 目标模型与草稿模型之间共享 indexer 的选择结果
            # （draft verify 时 indexer 复用 target 的 topk）。
            topk_tokens = config.index_topk
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                topk_tokens,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            topk_indices_buffer = None

        # 共享输出头 + 草稿 decoder 层（is_draft_layer=True 跳过 hash 路由）。
        self.shared_head = SharedHead(config=config, prefix=prefix, quant_config=quant_config)
        self.mtp_block = DeepseekV4DecoderLayer(
            vllm_config,
            prefix,
            config=self.config,
            topk_indices_buffer=topk_indices_buffer,
            is_draft_layer=True,
        )
        # Hyper-Connections（hc_*）参数: 多路隐状态的混合系数。
        self.hc_eps = config.hc_eps
        self.hc_mult = hc_mult = config.hc_mult
        hc_dim = hc_mult * config.hidden_size

        # hc_head 三件套: fn（混合矩阵 [hc_mult, hc_mult*H]）、base（偏置
        # [hc_mult]）、scale（缩放 [1]）。sigmoid(mixes*scale+base)+eps 作为
        # 各路权重，见 hc_head。
        self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

        self.norm_eps = config.rms_norm_eps
        # 无权重 RMSNorm（has_weight=False），仅用于 hc_head 内部的归一化。
        self.hc_norm = RMSNorm(hc_dim, eps=config.rms_norm_eps, has_weight=False, dtype=torch.float32)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        """草稿层前向。

        Args:
            input_ids: [num_tokens] 当前步 token id。
            positions: [num_tokens] 位置 id。
            previous_hidden_states: [num_tokens, hc_mult*hidden_size] 目标
                （或上一层草稿）的多路隐状态。
            inputs_embeds: [num_tokens, hidden_size] 预查表嵌入（可选）。
            spec_step_index: 投机采样步号（多草稿层时选层用）。
        Returns:
            [num_tokens, hc_mult, hidden_size]（v1 runner）或 flatten 后的
            [num_tokens, hc_mult*hidden_size]（v2 runner）。
        """
        assert inputs_embeds is not None
        # masking inputs at position 0, as not needed by MTP
        # 【中文】MTP 不需要序列首 token 的嵌入（它没有“前一个 token”的
        # 语义），把 position==0 处嵌入置零。torch.where 按条件逐元素选择。
        inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
        inputs_embeds = self.enorm(inputs_embeds)
        # 目标隐状态 [N, hc_mult*H] -> [N, hc_mult, H]，逐路 hnorm。
        previous_hidden_states = previous_hidden_states.view(-1, self.hc_mult, self.config.hidden_size)
        previous_hidden_states = self.hnorm(previous_hidden_states)

        full_num_tokens = positions.shape[0]
        use_sp = self.mtp_block.use_sequence_parallel_moe
        # Shard the mask only for the duration of this forward: the same
        # forward_context is reused across draft steps with full-length
        # inputs, so a sharded mask must not leak to the next step.
        # 【中文】SP（sequence parallel）分支: 仅在本 forward 期间切分
        # is_padding 掩码——forward_context 会被后续 draft step 复用，
        # 切分后的掩码不能泄漏到下一步，因此先保存原值、结尾恢复。
        orig_is_padding = None
        forward_context = None
        if use_sp:
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                orig_is_padding = forward_context.is_padding
                forward_context.is_padding = sp_padding_mask(orig_is_padding, inputs_embeds)
            # sp_shard supports arbitrary leading dims; upstream
            # sequence_parallel_chunk only pads 2D correctly and would pad
            # the hc_mult dim of [N, hc_mult, H] instead of the token dim.
            # 【中文】sp_shard 支持任意前导维度; 上游的
            # sequence_parallel_chunk 只能正确 pad 2D 张量，会把
            # [N, hc_mult, H] 的 hc_mult 维当 token 维 pad 错——NPU 适配点。
            inputs_embeds = sp_shard(inputs_embeds)
            previous_hidden_states = sp_shard(previous_hidden_states)

        # 核心公式: 两路投影相加得到草稿层输入 [N, hc_mult, H]。
        # unsqueeze(-2) 把 e_proj 输出 [N, H] 变 [N, 1, H] 广播到 hc_mult 路。
        hidden_states = self.e_proj(inputs_embeds).unsqueeze(-2) + self.h_proj(previous_hidden_states)

        # 过草稿 decoder 层（DSA 注意力 + MoE），input_ids=None——hash 路由
        # 层在草稿中不可用（草稿可能没有完整 token 上下文）。
        hidden_states, residual = self.mtp_block(
            positions=positions,
            hidden_states=hidden_states,
            residual=None,
            input_ids=None,
        )

        if use_sp:
            # SP 结束: all_gather 恢复完整 token 维，并裁掉 pad 到的
            # full_num_tokens（sp_shard 为整除可能 pad 了 token 数）。
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            hidden_states = hidden_states[:full_num_tokens]

        # 恢复原 is_padding 掩码（防止泄漏到下一个 draft step）。
        if forward_context is not None:
            forward_context.is_padding = orig_is_padding

        # hidden_states = self.hc_head(hidden_states, self.hc_head_fn,
        #                              self.hc_head_scale, self.hc_head_base)
        # 【中文】注意: hc_head 的混合被推迟到 compute_logits 中执行
        # （见 DeepSeekMultiTokenPredictor.compute_logits），此处保留多路输出。

        if self.use_v2_model_runner:
            # v2 runner 期望扁平的 [N, hc_mult*H]。
            return hidden_states.flatten(1)
        return hidden_states

    def hc_head(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        """Hyper-Connections 混合头: 把 hc_mult 路隐状态加权合并为单路。

        原理: 对展平输入先 RMSNorm，再经 fn 线性层得到每路的门控系数，
        sigmoid(mixes*scale+base)+eps 后作为权重，对原始（未归一化）的
        多路输入加权求和。全程 float32 保证数值稳定。

        Args:
            x: [N, hc_mult, H] 多路隐状态。
            hc_fn/hc_scale/hc_base: 混合矩阵/缩放/偏置。
        Returns:
            [N, H] 混合后的单路隐状态（原 dtype）。
        """
        shape, dtype = x.size(), x.dtype
        # 展平并升 float32。
        x = x.flatten(1).float()
        # 归一化后的输入用于计算门控系数（与混合输入解耦，稳定训练）。
        x_norm = self.hc_norm(x)
        mixes = torch.nn.functional.linear(x_norm, hc_fn)
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        # 加权求和: pre.unsqueeze(-1) 广播到 [N, hc_mult, H]，沿路维求和。
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
        return y.to(dtype)


class DeepSeekMultiTokenPredictor(nn.Module):
    """MTP 容器: 持有 num_nextn_predict_layers 个草稿层 + 共享 embedding。

    投机采样的每一步（spec_step_idx）轮转使用一个草稿层:
    step i 使用 layers[i % num_mtp_layers]，形成链式多步草稿。
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        """初始化。Args: vllm_config: 全局配置; prefix: 前缀（keyword-only）。"""
        super().__init__()
        config = vllm_config.model_config.hf_config
        # MTP 层编号从 num_hidden_layers 起（与 checkpoint 命名对应）。
        self.mtp_start_layer_idx = config.num_hidden_layers
        # 草稿层数量（DeepSeek-V3 为 1，可配置多个形成链）。
        self.num_mtp_layers = getattr(config, "num_nextn_predict_layers", 1)
        # to map the exact layer index from weights
        # 【中文】此索引用于把 checkpoint 中的层号映射到本容器的键。

        # 字典推导式构造 ModuleDict: 键为层号字符串（ModuleDict 只接受
        # str 键），值为 DeepSeekMultiTokenPredictorLayer。
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): DeepSeekMultiTokenPredictorLayer(vllm_config, f"{prefix}.{idx}")
                for idx in range(
                    0,
                    self.num_mtp_layers,
                )
            }
        )
        # 与目标模型共享的词嵌入（DeepSeek-V3: MTP 复用 embedding）。
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """token id -> 嵌入。Args: input_ids: [num_tokens]。Returns: [num_tokens, hidden_size]。"""
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        """MTP 前向: 按投机步号选择对应草稿层并执行。

        Args:
            input_ids: [num_tokens] 本步 token id。
            positions: [num_tokens] 位置 id。
            previous_hidden_states: [num_tokens, hc_mult*H] 上一步隐状态。
            inputs_embeds: 预查表嵌入（None 时内部查表）。
            spec_step_idx: 投机步号（0 起）。
        Returns:
            本步草稿隐状态。
        """
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # 取模轮转: 步号超过草稿层数时从头复用（链式循环草稿）。
        current_step_idx = spec_step_idx % self.num_mtp_layers
        return self.layers[str(current_step_idx)](
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
        """由草稿隐状态计算 logits（含 hc_head 混合 + 共享头）。

        Args:
            hidden_states: [num_tokens, hc_mult*H] 草稿隐状态。
            spec_step_idx: 投机步号（选择对应层的 hc_head 参数）。
        Returns:
            [num_tokens, vocab_size] logits。
        """
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(current_step_idx)]
        # 步骤1: 还原多路形状并做 hc_head 混合 -> [N, H]。
        hidden_states = hidden_states.view(-1, mtp_layer.hc_mult, mtp_layer.config.hidden_size)
        hidden_states = mtp_layer.hc_head(
            hidden_states, mtp_layer.hc_head_fn, mtp_layer.hc_head_scale, mtp_layer.hc_head_base
        )
        # 步骤2: 共享头（norm + lm_head）+ LogitsProcessor 得 logits。
        logits = self.logits_processor(mtp_layer.shared_head.head, mtp_layer.shared_head(hidden_states))
        return logits


@support_torch_compile
class DeepSeekV4MTP(nn.Module, SupportsPP, DeepseekV2MixtureOfExperts):
    """MTP 顶层模型（用于 EAGLE 式投机采样的草稿模型入口）。

    多重继承:
      - SupportsPP: 支持 PP 流水线并行（forward 接受 intermediate_tensors）;
      - DeepseekV2MixtureOfExperts: MoE 元数据混入（提供专家映射/EPLB
        所需的 num_experts 等属性，方法定义在 model.py）。
    装饰器 @support_torch_compile: 语法点——标记该类参与 vLLM 的
    torch.compile / piecewise 编译，forward 会被拆分为可编译的子图，
    在 NPU 上消除 Python 开销。
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        """初始化。Args: vllm_config: 全局配置; prefix: 前缀（keyword-only）。"""
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = DeepSeekMultiTokenPredictor(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "mtp"))
        # Set MoE hyperparameters
        # 【中文】汇总 MoE 元数据（供 EPLB 专家负载均衡与调度器查询）。
        self.set_moe_parameters()

    def set_moe_parameters(self):
        """遍历草稿层收集 MoE 模块，提取专家数量等元数据。

        原理: EPLB（Expert Parallelism Load Balancer）需要知道每层专家
        的逻辑/物理数量; 这里从“最后一个 MoE 层”（example_moe）读取配置
        （前面的层可能是 dense 层，取最后一个保证是 MoE）。
        """
        self.expert_weights = []
        # n_group: 无辅助损失路由的分组数（DeepSeek-V3 式分组路由）。
        self.num_expert_groups = getattr(self.config, "n_group", 1)

        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers.values():
            # PP 缺失层（不属于本流水级的占位）直接跳过。
            if isinstance(layer, PPMissingLayer):
                continue
            assert isinstance(layer, DeepSeekMultiTokenPredictorLayer)
            # 取内层真正的 decoder 层再判断 MoE。
            layer = layer.mtp_block
            assert isinstance(layer, DeepseekV4DecoderLayer)
            if isinstance(layer.mlp, DeepseekV4MoE):
                # Pick last one layer since the first ones may be dense layers.
                # 【中文】持续覆盖 example_moe，最终留下最后一个 MoE 层。
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)
        self.extract_moe_parameters(example_moe)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """token id -> 嵌入（委托内部 predictor）。"""
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        """草稿前向（一步）。

        Args:
            input_ids/positions: 当前步 token 与位置。
            hidden_states: 目标模型（或上一步草稿）的多路隐状态。
            intermediate_tensors: PP 中间张量（草稿模型通常单级，未用）。
            inputs_embeds: 预查表嵌入（可选）。
            spec_step_idx: 投机步号。
        Returns:
            本步草稿隐状态。
        """
        hidden_states = self.model(input_ids, positions, hidden_states, inputs_embeds, spec_step_idx)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        """由草稿隐状态计算 logits（委托内部 predictor）。"""
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """加载 MTP 草稿权重（checkpoint 的 mtp.0.* 命名空间）。

        原理: DeepSeek-V3 的 MTP 权重在 checkpoint 中以 ``mtp.{i}.*`` 存储
        （官方实现把 MTP 当作第 num_hidden_layers+i 层）。本方法把该命名
        空间的名字重写到本模型的模块树（model.layers.0.* / model.embed_tokens.*
        / lm_head 等），再按“专家权重 / 堆叠权重 / 普通权重”三类分派加载。

        Args:
            weights: (名字, 张量) 流。
        Returns:
            成功加载的参数名集合。
        """
        # 注意: 下一行查询 ROCm AITER 融合开关，随后即被 Ascend 的
        # mix_placement（混部）配置覆盖（同名赋值，后者生效）。
        rocm_aiter_moe_shared_expert_enabled = rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        rocm_aiter_moe_shared_expert_enabled = getattr(get_ascend_config(), "mix_placement", False)
        # 堆叠权重映射: gate_proj/up_proj 两个 checkpoint 权重合并进
        # 一个 gate_up_proj 参数（shard_id 0/1 区分两半）。
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # 专家参数映射: (param_name, weight_name, expert_id, shard_id)。
        # fused_moe_make_expert_params_mapping 自动为每个专家 x 每种投影
        # 生成一条映射。mix_placement 时把共享专家也并入专家列表。
        expert_params_mapping = fused_moe_make_expert_params_mapping(
            model=self.model,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts
            + (self.config.n_shared_experts if rocm_aiter_moe_shared_expert_enabled else 0),
            num_redundant_experts=self.num_redundant_experts,
        )

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        # Attention heads per rank
        # 【中文】注意力 sink 按头切分: 每卡只保存自己负责的头。
        heads_per_rank = self.config.num_attention_heads // tp_size
        head_start = tp_rank * heads_per_rank

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            # 跳过 RoPE 频率表（由 ComplexExpRotaryEmbedding 自行预计算）。
            if "rotary_emb.inv_freq" in name:
                continue

            # deepseek_v4_fp8 量化的 checkpoint 用简化名 embed/head。
            if self.quant_config is not None and self.quant_config.get_name() == "deepseek_v4_fp8":
                if name == "embed.weight":
                    name = "mtp.0.emb.tok_emb.weight"

                if name == "head.weight":
                    name = "mtp.0.head.weight"

            # 只处理 mtp.* 命名空间的权重，其余属于目标模型 -> 跳过。
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is None:
                continue

            # ---- 名称重写: mtp.0.* -> 本模型模块树 ----
            assert "mtp.0." in name
            if ".emb.tok_emb." in name:
                # embedding 提升到顶层 model.embed_tokens。
                name = name.replace("mtp.0.", "model.")
            elif self.no_mtp_block_in_name(name):
                # 草稿层直属模块（e_proj/h_proj/enorm/...）映射到 model.layers.0.*。
                name = name.replace("mtp.0.", "model.layers.0.")
            else:
                # transformer 块内模块加 mtp_block 前缀。
                name = name.replace("mtp.0.", "model.layers.0.mtp_block.")

            # DeepSeek 原始命名（w1/w2/w3）-> vLLM 命名的逐条替换。
            if ".w1." in name:
                name = name.replace(".w1.", ".gate_proj.")
            if ".w2." in name:
                name = name.replace(".w2.", ".down_proj.")
            if ".w3." in name:
                name = name.replace(".w3.", ".up_proj.")

            # Ascend 约定: 量化 scale 参数名重写为 weight_scale。
            if name.endswith(".scale"):
                name = name.replace(".scale", ".weight_scale")

            if ".head." in name:
                name = name.replace(".head.", ".shared_head.head.")

            if ".norm." in name:
                name = name.replace(".norm.", ".shared_head.norm.")

            if ".emb.tok_emb." in name:
                name = name.replace(".emb.tok_emb.", ".embed_tokens.")

            if "attn" in name and "self_attn" not in name:
                name = name.replace(".attn.", ".self_attn.")
            if ".ffn." in name:
                name = name.replace(".ffn.", ".mlp.")
            if ".ffn_norm." in name:
                name = name.replace(".ffn_norm.", ".post_attention_layernorm.")
            if ".attn_norm." in name:
                name = name.replace(".attn_norm.", ".input_layernorm.")

            # 路由偏置 -> e_score_correction_bias（无辅助损失路由偏置）。
            if ".gate.bias" in name:
                name = name.replace(".gate.bias", ".gate.e_score_correction_bias")

            # ---- attention sink 特殊加载（按头切分或 DSA-CP 全量）----
            if "sink" in name:
                param = params_dict[name]
                if enable_dsa_cp():
                    # DSA 上下文并行: 所有 rank 持有全部头的 sink。
                    param.data.copy_(loaded_weight)
                else:
                    # Handle attention sinks (distributed across ranks)
                    # 【中文】普通 TP: 每卡只拷贝自己负责的那段头的 sink。
                    narrow_weight = loaded_weight.narrow(0, head_start, heads_per_rank)
                    param.data.copy_(narrow_weight)
                loaded_params.add(name)
                continue

            # ---- 主加载循环: 堆叠权重 / 专家权重 / 普通权重 ----
            is_fusion_moe_shared_experts_layer = rocm_aiter_moe_shared_expert_enabled and ("mlp.shared_experts" in name)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                # 【中文】跳过非堆叠层与专家权重（专家在下方单独处理）。
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                # 【中文】必须在改名前判断是否专家权重，避免双重替换破坏加载。
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                if is_fusion_moe_shared_experts_layer:
                    continue
                name_mapped = name.replace(weight_name, param_name)

                # QKV fusion is optional, fall back to normal
                # weight loading if it's not enabled
                # 【中文】可选的 QKV 融合: 融合参数不存在则回退普通加载。
                if (param_name == "fused_qkv_a_proj") and name_mapped not in params_dict:
                    continue
                else:
                    name = name_mapped

                # Skip loading extra bias for GPTQ models.
                # 【中文】GPTQ 额外 bias 的参数不存在时跳过。
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                # shard_id 告诉 loader 本次装的是融合参数的哪一半。
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # for...else 语法点: 循环未被 break（不匹配任何堆叠规则）时
                # 才执行本分支。
                # Special handling: when AITER fusion_shared_experts is enabled,
                # checkpoints may provide a single widened shared_experts tensor
                # without explicit expert indices
                # (e.g. ...mlp.shared_experts.gate_proj.weight).
                # For models with multiple shared experts, split that tensor
                # evenly into per-shared-expert slices and load them into
                # appended expert slots mlp.experts.{n_routed_experts + j}.*
                # accordingly.
                # 【中文】mix_placement 模式: checkpoint 提供单个“加宽”的
                # shared_experts 张量（无专家下标），需均分为 n_shared_experts
                # 份，分别装入追加的专家槽位。
                num_chunks = 1
                if is_fusion_moe_shared_experts_layer:
                    num_chunks = getattr(self.config, "n_shared_experts", 1) or 1
                    # Determine split axis based on op type
                    # gate/up: ColumnParallel → split along dim 0
                    # down: RowParallel → split along dim 1
                    # 【中文】按并行方式选切分轴: Column 沿 dim0，Row 沿 dim1。
                    split_dim = 1 if "down_proj.weight" in name else 0
                    total = loaded_weight.shape[split_dim]
                    # 整除断言: 切分必须无余数，否则模型结构与 checkpoint 不符。
                    assert total % num_chunks == 0, (
                        f"Shared expert weight dim {total} not divisible by num_chunks {num_chunks}"
                    )
                    chunk_size = total // num_chunks

                for j in range(num_chunks):
                    chunk_name = name
                    weight_to_load = loaded_weight

                    if is_fusion_moe_shared_experts_layer:
                        # 按轴切出第 j 个共享专家的权重片。
                        if split_dim == 0:
                            weight_to_load = loaded_weight[j * chunk_size : (j + 1) * chunk_size, :]
                        else:
                            weight_to_load = loaded_weight[:, j * chunk_size : (j + 1) * chunk_size]
                        # Synthesize an expert-style name so expert mapping
                        # can route it
                        # 【中文】合成专家式名字，让下方映射能够路由。
                        chunk_name = name.replace(
                            "mlp.shared_experts",
                            f"mlp.experts.{self.config.n_routed_experts + j}",
                        )

                    # Use expert_params_mapping to locate the destination
                    # param and delegate to its expert-aware weight_loader
                    # with expert_id.
                    # 【中文】专家权重: 遍历映射，命中后用带 expert_id 的
                    # loader 加载。
                    is_expert_weight = False
                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in chunk_name:
                            continue

                        # Anyway, this is an expert weight and should not be
                        # attempted to load as other weights later
                        # 【中文】只要匹配了映射即认定为专家权重。
                        is_expert_weight = True

                        # Do not modify `name` since the loop may continue here
                        # Instead, create a new variable
                        # 【中文】循环可能继续，不能原地改 name，用新变量。
                        name_mapped = chunk_name.replace(weight_name, param_name)

                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or
                        # not here since otherwise we may skip experts with
                        # other available replicas.
                        # 【中文】要求 loader 返回成败——EPLB 冗余专家下同一
                        # 专家可能在多卡有副本，本卡失败不代表全局失败。
                        # typing.cast: 类型断言（运行时无操作）。
                        weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
                        success = weight_loader(
                            param,
                            weight_to_load,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            if not is_fusion_moe_shared_experts_layer:
                                name = name_mapped
                            else:
                                loaded_params.add(name_mapped)
                            break
                    else:
                        if is_expert_weight:
                            # We've checked that this is an expert weight
                            # However it's not mapped locally to this rank
                            # So we simply skip it
                            # 【中文】专家权重但不在本 rank: 跳过。
                            continue

                        # Skip loading extra bias for GPTQ models.
                        # 【中文】GPTQ 额外 bias 跳过。
                        if name.endswith(".bias") and name not in params_dict:
                            continue

                        # FP8 kv-scale 名字重映射; 无对应参数则跳过。
                        name = maybe_remap_kv_scale_name(name, params_dict)
                        if name is None:
                            continue

                        # # According to DeepSeek-V3 Technical Report, MTP modules
                        # # shares embedding layer. We only load the first weights.
                        # # 【中文】（历史注释保留）按 V3 报告 MTP 共享 embedding，
                        # # # 只加载首份权重——该逻辑已被注释停用。
                        # if (
                        #     spec_layer != self.model.mtp_start_layer_idx
                        #     and ".layers" not in name
                        # ):
                        #     continue

                        param = params_dict[name]
                        # 普通参数: 有专用 loader 用之（量化参数），
                        # 否则 default_weight_loader（直接拷贝/按 TP 切分）。
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight)
            if not is_fusion_moe_shared_experts_layer:
                loaded_params.add(name)
        return loaded_params

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        """
        Rewrite the weight name to match the format of the original model.
        Add .mtp_block for modules in transformer layer block for spec layer
        and rename shared layer weights to be top level.
        """
        # 【中文】把草稿层权重名重写为本模型格式:
        # - transformer 块内模块 -> 加 .mtp_block 前缀;
        # - 草稿层直属/共享模块（embedding 等）-> 提升为顶层 model.*。
        # 草稿层直属模块名单（这些名字直接挂在层下，不进 mtp_block）。
        spec_layer_weight_names = [
            "embed_tokens",
            "enorm",
            "hnorm",
            "eh_proj",
            "shared_head",
        ]
        # 跨层共享的模块名单（提升到 model 顶层）。
        shared_weight_names = ["embed_tokens"]
        spec_layer_weight = False
        shared_weight = False
        # 两个标志位的判定循环。
        for weight_name in spec_layer_weight_names:
            if weight_name in name:
                spec_layer_weight = True
                if weight_name in shared_weight_names:
                    shared_weight = True
                break
        if not spec_layer_weight:
            # treat rest weights as weights for transformer layer block
            # 【中文】非直属模块 -> 视作 transformer 块内权重，加 mtp_block。
            name = name.replace(f"model.layers.{spec_layer}.", f"model.layers.{spec_layer}.mtp_block.")
        elif shared_weight:
            # treat shared weights as top level weights
            # 【中文】共享权重 -> 提升到 model 顶层。
            name = name.replace(f"model.layers.{spec_layer}.", "model.")
        return name

    def no_mtp_block_in_name(self, layer_name: str) -> bool:
        """判断权重名是否属于“草稿层直属模块”（即不需要 mtp_block 前缀）。

        原理: 这些模块（e_proj/h_proj/enorm/hnorm/norm/head/embed 及
        hc_head 参数）在模型树中直接挂在 DeepSeekMultiTokenPredictorLayer
        下，而注意力/MoE 等在 mtp_block（DeepseekV4DecoderLayer）内。
        any(...): 生成器逐项短路判断，任一名字出现即 True。
        """
        names = [
            ".hc_head_fn",
            ".hc_head_base",
            ".hc_head_scale",
            ".e_proj.",
            ".h_proj.",
            ".enorm.",
            ".hnorm.",
            ".norm.",
            ".head.",
            ".emb.tok_emb.",
        ]
        return any(name in layer_name for name in names)
