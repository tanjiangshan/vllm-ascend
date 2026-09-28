# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
# =============================================================================
# 【文件职责】DeepSeek V4.1 多模态(VL)模型的 Ascend 包装器。
#
# 【架构角色】本类是注册到 vLLM 多模态注册表的顶层模型：组合视觉塔
# （vision.py 的 DeepseekV41ViT + DeepseekV41Aligner）与语言模型
# （model.py 的 AscendDeepseekV41LLMForCausalLM）。图像处理流水线：
#   MM processor（上游 deepseek_v41.common.mm_preprocess，负责图像切片成
#   patch、生成 types 序列与网格信息）
#   → embed_multimodal: 逐图 ViT 编码 + Aligner 空间合并投影
#   → _build_image_span: 按 types 模板把 [IMAGE_START, 图像 token, 换行符,
#     IMAGE_END, ...] 拼成完整图像跨度（含哨兵可学习向量）
#   → embed_input_ids: 与文本词嵌入合并（_merge_multimodal_embeddings）。
#
# 【与文本模型的关系】forward/compute_logits/engram 准备/MTP 接口全部
# 委托给 language_model；仅视觉编码、图像跨度构建与权重分发在本类完成。
# 图像哨兵（image_start/end/newline）是三个可学习的 hidden_size 维 FP32
# 向量，标记图像内容在文本序列中的边界与换行。
# =============================================================================
"""Ascend multimodal wrapper for DeepSeek V4.1."""

from collections.abc import Iterable, Iterator

import torch
from torch import nn
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import MultiModalEmbeddings, SupportsEagle3, SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import maybe_prefix

# Upstream #56741 normalized the V4.1 model package name from deepseek_v4_1
# to deepseek_v41.
from vllm.models.deepseek_v41.common.mm_preprocess import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_PLACEHOLDER,
    IMAGE_START,
    DeepseekV4VLDummyInputsBuilder,
    DeepseekV4VLMultiModalProcessor,
    DeepseekV4VLProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from .model import AscendDeepseekV41LLMForCausalLM
from .vision import DeepseekV41Aligner, DeepseekV41ViT


def _vision_parameter_name(name: str) -> str | None:
    """Map a checkpoint vision tensor to the wrapper parameter namespace."""
    """【中文说明】判断某条 checkpoint 权重是否属于视觉塔，并返回包装器内的
    参数名；不属于视觉则返回 None（交由语言模型加载）。

    规则: 先去掉 "model." 前缀（HF checkpoint 中视觉权重位于 model.vision.*
    下），再要求剩余名字以 vision./aligner./image_ 开头。"""
    if name.startswith("model."):
        name = name.removeprefix("model.")
    if name.startswith(("vision.", "aligner.", "image_")):
        return name
    return None


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekV4VLMultiModalProcessor,
    info=DeepseekV4VLProcessingInfo,
    dummy_inputs=DeepseekV4VLDummyInputsBuilder,
)
class AscendDeepseekV41ForCausalLM(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsEagle3,
):
    """V4.1 image-span semantics with the shared Ascend vision tower."""
    """【中文说明】DeepSeek V4.1 多模态顶层模型。继承 vLLM 的三个能力接口：
    SupportsMultiModal（多模态输入）、SupportsPP（流水线并行）、
    SupportsEagle3（EAGLE3 投机解码，供 DSpark 草稿模型使用）。
    视觉塔与语言模型共享实现（"shared Ascend vision tower"）。"""

    # Engram history and vision MoE routing also consume the original token IDs.
    # 【中文】engram 的 n-gram 历史与视觉 MoE 路由都需要原始 token ID，
    # 因此要求调度器把原始输入词元传入模型（而非只有嵌入）。
    requires_raw_input_tokens = True
    # packed_modules_mapping: 声明 checkpoint 的 gate_proj/up_proj 两个权重
    # 在运行时合并加载进一个 gate_up_proj 融合参数（见 model.py 的 MoE）。
    packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    # 语言模型实现类，供 speculative/EAGLE3 proposer 按类属性发现。
    language_model_cls = AscendDeepseekV41LLMForCausalLM

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        """返回多模态占位符文本（prompt 模板中图像出现的位置标记）。

        语法点: @classmethod——类方法，无需实例即可调用；i（第几张图）被
        del 显式丢弃，因为 V4.1 所有图像共用同一占位符。
        """
        del i
        if modality == "image":
            return IMAGE_PLACEHOLDER
        return None

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        """构建多模态模型。

        参数:
            vllm_config: 引擎全局配置。
            prefix: 模型名前缀（PP/嵌套场景）。
        语法点: * 表示其后只接受关键字参数（keyword-only）。
        步骤:
            1) 若配置了视觉层数，标记 is_mm_prefix_lm（前缀 LM 语义）并把
               滑窗限制到 mm 前缀；
            2) 在"塔模型"标记上下文中构建视觉塔/对齐器/哨兵向量（仅当
               图像输入配额 > 0 时才真正创建，节省纯文本部署的显存）；
            3) 在"语言模型"标记上下文中构建语言模型；
            4) 语言模型的 MoE 通信方法与中间张量构造器提升为本类属性。
        """
        super().__init__()
        model_config = vllm_config.model_config
        config = model_config.hf_config
        if getattr(config, "vision_n_layers", 0) > 0:
            # is_mm_prefix_lm: 多模态前缀语言模型——图像永远在 prompt 前缀；
            # mm_prefix_clamp_sliding_window: 滑窗注意力对前缀部分做钳制处理。
            config.is_mm_prefix_lm = True
            config.mm_prefix_clamp_sliding_window = True
        self.config = config
        self.multimodal_config = model_config.multimodal_config

        # 图像功能开启 = 配置有视觉层 且 用户未把每 prompt 图像配额设为 0。
        image_enabled = config.vision_n_layers > 0 and self.multimodal_config.get_limit_per_prompt("image") > 0
        # _mark_tower_model/_mark_language_model: vLLM 的上下文管理器，用于
        # 区分"视觉塔权重"与"语言模型权重"的加载范围（量化/精度配置可不同）。
        # 语法点: with ... as 无返回值时仅起标记作用。
        with self._mark_tower_model(vllm_config, {"image"}):
            self.vision: DeepseekV41ViT | None = None
            self.aligner: DeepseekV41Aligner | None = None
            self.image_start: nn.Parameter | None = None
            self.image_end: nn.Parameter | None = None
            self.image_newline: nn.Parameter | None = None
            if image_enabled:
                # 真正创建视觉塔、对齐器与三个图像哨兵向量（FP32 参数，
                # 与文本嵌入合并时再转到模型 dtype）。
                self.vision = DeepseekV41ViT(config)
                self.aligner = DeepseekV41Aligner(config)
                for name in ("image_start", "image_end", "image_newline"):
                    setattr(
                        self,
                        name,
                        nn.Parameter(torch.empty(config.hidden_size, dtype=torch.float32)),
                    )
                # 视觉塔与对齐器整体转到模型计算 dtype（如 BF16）。
                self.vision.to(dtype=model_config.dtype)
                self.aligner.to(dtype=model_config.dtype)

        with self._mark_language_model(vllm_config):
            self.language_model = self.language_model_cls(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )
        # 把语言模型的两个方法/属性提升到包装器上，供 runner 直接访问。
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors
        self.moe_comm_methods = self.language_model.moe_comm_methods

    def _parse_and_validate_image_input(self, **kwargs: object) -> dict | None:
        """从 forward 的关键字参数中提取并校验图像输入。

        参数(**kwargs):
            patches: [total_patches, 3*patch_size²] 所有图 patch 拼接。
            vit_grid: [n_images, 2] 每张图的 ViT 网格 (n_h, n_w)。
            llm_grid: [n_images, 2] 每张图合并后的 LLM 网格 (n_h, n_w)。
            types: 图像跨度模板类型序列（IMAGE/IMAGE_START/IMAGE_END/NEW_LINE）。
        返回:
            dict 或 None（无图像输入时）。
        语法点: kwargs.pop 把键从字典中取出并删除，防止其透传到语言模型。
        """
        patches = kwargs.pop("patches", None)
        if patches is None:
            return None
        vit_grid = kwargs.pop("vit_grid", None)
        llm_grid = kwargs.pop("llm_grid", None)
        types = kwargs.pop("types", None)
        return {
            "patches": patches,
            "vit_grid": vit_grid,
            "llm_grid": llm_grid,
            "types": types,
        }

    def _encode_image(
        self,
        patches: torch.Tensor,
        n_vit_h: int,
        n_vit_w: int,
    ) -> torch.Tensor:
        """编码单张图像: ViT 提特征 → Aligner 合并投影。

        参数:
            patches: [n_vit_h*n_vit_w, 3*patch_size²] 单图 patch。
        返回:
            [n_llm_h*n_llm_w, hidden_size] 该图的视觉 token 嵌入。
        """
        assert self.vision is not None and self.aligner is not None, "Image encoding requires an enabled vision tower"
        return self.aligner(
            self.vision(patches, n_vit_h, n_vit_w),
            n_vit_h,
            n_vit_w,
        )

    def _build_image_span(
        self,
        image_embeds: torch.Tensor,
        types: torch.Tensor,
    ) -> torch.Tensor:
        """按模板类型序列构建完整图像跨度张量。

        参数:
            image_embeds: [n_image_tokens, hidden_size] 对齐后的视觉嵌入。
            types: [span_len] 每个位置的类型标记（IMAGE_START/IMAGE/
                IMAGE_NEW_LINE/IMAGE_END）。
        返回:
            [span_len, hidden_size] 图像跨度（哨兵向量与视觉 token 按模板
            混合排布），后续按位置替换进文本嵌入序列。
        原理: types 由 MM processor 依据图像布局生成（如逐行排布、行间插
            换行符），这里把每个类型翻译成对应向量。
        """
        assert self.image_start is not None and self.image_end is not None and self.image_newline is not None
        types = types.to(image_embeds.device)
        span = image_embeds.new_empty(types.numel(), image_embeds.shape[-1])
        dtype = image_embeds.dtype
        # 三个哨兵位置放可学习边界/换行向量（FP32 参数转到嵌入 dtype）。
        span[types == IMAGE_START] = self.image_start.to(dtype)
        span[types == IMAGE_END] = self.image_end.to(dtype)
        span[types == IMAGE_NEW_LINE] = self.image_newline.to(dtype)
        # IMAGE 类型位置放真正的视觉 token 嵌入。
        span[types == IMAGE] = image_embeds
        return span

    def _process_image_input(
        self,
        patches: torch.Tensor,
        vit_grid: torch.Tensor,
        llm_grid: torch.Tensor,
        types: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """处理一批图像输入：逐图编码并构建图像跨度。

        参数:
            patches: 所有图 patch 的拼接张量。
            vit_grid/llm_grid: 每图 [n_h, n_w] 网格（ViT 侧/LLM 侧）。
            types: 所有图跨度模板类型的拼接序列。
        返回:
            tuple[image_span, ...] 每张图一个跨度张量。
        步骤:
            1) patches 转到对齐器权重 dtype；
            2) 逐图: 按 ViT 网格切出该图 patch → _encode_image → 按跨度
               长度切出该图 types → _build_image_span；
            3) 推进 vit_offset/span_offset 游标。
        语法点: zip(..., strict=True) 在两序列长度不一致时抛错（防错位）。
        """
        assert self.aligner is not None, "Image processing requires an enabled vision tower"
        patches = patches.to(self.aligner.w1.weight.dtype)
        embeds: list[torch.Tensor] = []
        vit_offset = 0
        span_offset = 0
        for (n_vit_h, n_vit_w), (n_llm_h, n_llm_w) in zip(
            vit_grid.tolist(),
            llm_grid.tolist(),
            strict=True,
        ):
            n_vit = n_vit_h * n_vit_w
            # 跨度长度 = LLM 网格 token 数 + 每行末一个换行符 + 首尾哨兵。
            span_len = n_llm_h * (n_llm_w + 1) + 2
            image_embeds = self._encode_image(
                patches[vit_offset : vit_offset + n_vit],
                n_vit_h,
                n_vit_w,
            )
            embeds.append(
                self._build_image_span(
                    image_embeds,
                    types[span_offset : span_offset + span_len],
                )
            )
            vit_offset += n_vit
            span_offset += span_len
        return tuple(embeds)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """多模态嵌入入口（vLLM SupportsMultiModal 接口）。

        返回: 每张图一个跨度的列表；无图像输入或视觉塔未启用时返回空列表。
        """
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None or self.vision is None:
            return []
        return self._process_image_input(
            image_input["patches"],
            image_input["vit_grid"],
            image_input["llm_grid"],
            image_input["types"],
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """把文本词嵌入与多模态嵌入合并成最终输入嵌入。

        参数:
            input_ids: [num_tokens] 词元 ID。
            multimodal_embeddings: 各图跨度嵌入（None 表示纯文本）。
            is_multimodal: [num_tokens] 布尔掩码，标记哪些位置是多模态占位。
        返回:
            [num_tokens, hidden_size] 合并后的输入嵌入。
        原理: 词嵌入先查表，再由上游 _merge_multimodal_embeddings 把
            is_multimodal 为 True 的位置替换成对应的多模态嵌入。
        """
        from vllm.model_executor.models.utils import (
            _merge_multimodal_embeddings,
        )

        embedding_ids = input_ids
        inputs_embeds = self.language_model.embed_input_ids(embedding_ids)
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def prepare_engram_graph_inputs(self, padded_tokens=None):
        """委托语言模型准备 engram 的固定地址图捕获缓冲（见 model.py）。"""
        return self.language_model.prepare_engram_graph_inputs(padded_tokens)

    def prepare_engram_inputs(
        self,
        input_ids,
        positions,
        padded_tokens=None,
        lookback_token_ids=None,
        query_start_loc=None,
        slot_mapping=None,
        block_table=None,
    ):
        """委托语言模型同步刷新 engram 行（ACL Graph 重放前调用）。"""
        return self.language_model.prepare_engram_inputs(
            input_ids,
            positions,
            padded_tokens,
            lookback_token_ids,
            query_start_loc,
            slot_mapping,
            block_table,
        )

    @property
    def token_lookback_depth(self) -> int:
        """What the runner sizes the prompt lookback buffer from."""
        """【中文说明】engram 哈希需要回看(prompt 前的)多少个历史 token；
        runner 据此分配 lookback_token_ids 缓冲。纯转发到语言模型。"""
        return self.language_model.token_lookback_depth

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """前向入口：多模态嵌入已在 runner 侧生成，这里直接进语言模型。

        参数:
            input_ids: [num_tokens] 词元 ID（多模态位置是占位 ID）。
            positions: [num_tokens] 位置。
            intermediate_tensors: PP 并行的上游中间张量。
            inputs_embeds: 已合并的多模态输入嵌入（优先使用）。
        返回:
            [num_tokens, hidden_size]（或 PP 场景的中间张量）。
        """
        return self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **kwargs,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """计算输出 logits（PP 最后一级），委托语言模型。"""
        return self.language_model.compute_logits(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """MoE 专家权重加载映射表，委托语言模型。"""
        return self.language_model.get_expert_mapping()

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        """取 MTP 目标隐状态缓冲（DSpark 草稿消费），委托语言模型。"""
        return self.language_model.get_mtp_target_hidden_states()

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        """设置 DSpark 需要抽取辅助隐状态的层（一键 checkpoint 层号），委托。"""
        self.language_model.set_aux_hidden_state_layers(layers)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """两级权重分发：视觉塔权重本地加载，其余转发给语言模型。

        步骤:
            1) language_weights() 是一个生成器（语法点: yield 惰性过滤）：
               视觉权重（_vision_parameter_name 命中）就地写入参数并记录，
               其余 yield 给语言模型的 load_weights；
            2) 语言模型返回其已加载集合，加 "language_model." 前缀后与视觉
               集合并集返回（上游据此校验权重覆盖完整性）。
        """
        params = dict(self.named_parameters())
        loaded_vision: set[str] = set()

        def language_weights() -> Iterator[tuple[str, torch.Tensor]]:
            for name, loaded_weight in weights:
                vision_name = _vision_parameter_name(name)
                if vision_name is None:
                    # 非视觉权重：透传给语言模型加载器。
                    yield name, loaded_weight
                    continue
                # 视觉权重：取参数的 weight_loader（或默认加载器）就地写入。
                param = params[vision_name]
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, loaded_weight)
                loaded_vision.add(vision_name)

        loaded_language = self.language_model.load_weights(language_weights())
        return loaded_vision | {f"language_model.{name}" for name in loaded_language}

    def process_weights_after_loading(self) -> None:
        """权重加载完成后的后处理钩子（如量化重打包），转发给语言模型。"""
        hook = getattr(
            self.language_model,
            "process_weights_after_loading",
            None,
        )
        if hook is not None:
            hook()

    @property
    def engram_cache_layer_name(self) -> str | None:
        """engram 哈希历史绑定的 SWA cache 层名（engram/hash_state.py 使用）。"""
        return self.language_model.engram_cache_layer_name
