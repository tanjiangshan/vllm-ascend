# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The MiniMax AI team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.

# =====================================================================================
# 中文注释（教学向）：MiniMax M3 多模态（视觉-语言）模型 —— 昇腾 NPU 适配版。
#
# 文件职责：
#   实现 MiniMax-M3-VL 多模态大模型在昇腾 NPU 上的推理顶层封装：
#     1. 复用 vLLM 官方（GPU 版）的 ViT 视觉塔实现 MiniMaxVLVisionModel —— 视觉
#        部分与硬件无关，直接从 vLLM 安装目录动态加载公共源码，避免重复维护；
#     2. 语言模型路径使用本插件（vllm-ascend）原生的
#        MiniMaxM3SparseForCausalLM（含 MSA 稀疏注意力 + MoE 的 NPU 适配）；
#     3. 处理图像/视频两种模态输入的解析、视觉编码与文本 embedding 的拼接。
#
# 在 vllm-ascend 插件架构中的位置：
#   vLLM 通过模型注册机制（MULTIMODAL_REGISTRY）发现多模态模型。本文件注册了
#   MiniMaxM3SparseForConditionalGeneration，vLLM 加载 checkpoint 后：
#     pixel_values/video --> vision_tower(ViT) --> 多模态 embedding
#     多模态 embedding + input_ids embedding --> language_model(稀疏注意力+MoE)
#     --> lm_head --> logits
#
# 关键 NPU 适配点：
#   - _install_fused_allreduce_norm_fallback(): 用纯通信+GemmaRMSNorm 组合替换
#     vLLM 中仅支持 CUDA 的"融合 allreduce+RMSNorm"算子（NPU 上无此 CUDA kernel）；
#   - 视觉塔 DP（数据并行）模式：use_data_parallel 时按 DP 切分视觉输入。
# =====================================================================================

"""MiniMax M3 multimodal wrapper for Ascend.

The ViT implementation is reused from vLLM's common MiniMax M3 vision tower,
while the language model path remains the Ascend-native implementation in
``vllm_ascend.models.minimax_m3``.
"""

import sys
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import torch
import vllm
from torch import nn
from transformers import PretrainedConfig
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.logger import logger
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.models.interfaces import (
    MixtureOfExperts,
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper, maybe_prefix
from vllm.model_executor.models.vision import run_dp_sharded_mrope_vision_model
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.utils.import_utils import import_from_path

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.models.minimax_m3.minimax_m3 import MiniMaxM3SparseForCausalLM
from vllm_ascend.utils import is_vl_model


def _load_vllm_minimax_m3_common_module(module_name: str):
    # 中文注释：从已安装的 vLLM 包目录中动态加载 MiniMax M3 的"硬件无关公共模块"。
    # 背景：vLLM 官方的 minimax_m3 模型目录下有 common/ 子目录存放 GPU/NPU 共用代码
    # （如视觉塔、多模态预处理）。直接 import vllm.model_executor.models.minimax_m3
    # 会触发其内部的 NVIDIA/AMD 平台分发逻辑，因此这里绕过包导入，直接按文件路径加载。
    if vllm.__file__ is None:
        # vllm.__file__ 为 None 说明 vLLM 以非常规方式安装（如嵌入解释器），无法定位源码目录
        raise ImportError("Unable to locate the installed vLLM package.")

    # 拼接公共模块的绝对路径：<vllm包目录>/models/minimax_m3/common/<module_name>.py
    module_path = Path(vllm.__file__).resolve().parent / "models" / "minimax_m3" / "common" / f"{module_name}.py"
    if not module_path.is_file():
        raise ImportError(
            "The vLLM MiniMax M3 common source was not found at "
            f"{module_path}. This vllm-ascend adapter requires a vLLM version "
            "with MiniMax M3 common modules."
        )

    # Import the hardware-neutral common module directly. Importing
    # vllm.models.minimax_m3 first would execute its NVIDIA/AMD platform
    # dispatcher, while Ascend only needs the shared VL helpers here.
    return import_from_path(
        f"vllm_ascend.models._vllm_minimax_m3_common_{module_name}",
        module_path,
    )


_mm_preprocess = _load_vllm_minimax_m3_common_module("mm_preprocess")
# 中文注释：复用 vLLM 公共多模态预处理模块的三个类：
#   - MiniMaxM3VLDummyInputsBuilder : 生成 dummy 多模态输入（用于 CUDA Graph 捕获/预热）
#   - MiniMaxM3VLMultiModalProcessor: 多模态输入处理器（把图像/视频转成模型输入格式）
#   - MiniMaxM3VLProcessingInfo     : 多模态处理元信息（支持的模态、最大尺寸等）
MiniMaxM3VLDummyInputsBuilder = _mm_preprocess.MiniMaxM3VLDummyInputsBuilder
MiniMaxM3VLMultiModalProcessor = _mm_preprocess.MiniMaxM3VLMultiModalProcessor
MiniMaxM3VLProcessingInfo = _mm_preprocess.MiniMaxM3VLProcessingInfo


def _install_fused_allreduce_norm_fallback() -> None:
    """Avoid importing vLLM's CUDA-only fusion module on Ascend."""
    module_name = "vllm.model_executor.layers.fused_allreduce_gemma_rms_norm"
    if module_name in sys.modules:
        return

    fallback_module = ModuleType(module_name)

    def fused_allreduce_gemma_rms_norm(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        norm: GemmaRMSNorm,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 中文注释：CUDA 融合算子的 NPU 等价实现 —— 分两步：先做 TP 通信，再做 GemmaRMSNorm。
        # 原理：Gemma 风格的 RMSNorm 计算为 x * (1 + weight) / rms(x)，与残差流更新
        # 一起做。NPU 上没有融合的 allreduce+norm 单算子，故拆开执行，数值结果一致。
        if get_tensor_model_parallel_world_size() > 1:
            # 仅在张量并行（TP>1）时需要跨卡通信。第一个 rank 之后的层输入是
            # RowParallelLinear 的部分和，必须 all-reduce 才是完整结果。
            # The config context may be unset (e.g. profiling/dummy runs);
            # fall back to the plain all-reduce path in that case.
            try:
                # 读取是否启用"序列并行 MoE"（SP）：SP 模式下视觉 token 均匀切分到各 TP rank
                sp_enabled = get_current_vllm_config().parallel_config.use_sequence_parallel_moe
            except AssertionError:
                sp_enabled = False
            if sp_enabled and not (_EXTRA_CTX.is_draft_model and is_vl_model()):
                # 序列并行路径：把 token 维补齐到 TP 大小的整数倍后 reduce-scatter，
                # 每张卡只保留 1/tp 的 token，后续计算量随之降低。
                # 语法点：(-n) % tp 计算 n 前还需补齐的 token 数（Python 取模结果非负）。
                padding = (-hidden_states.shape[0]) % get_tensor_model_parallel_world_size()
                if padding:
                    # F.pad 参数 (0,0)*(ndim-1) + (0,padding) 表示只在最后一维（token 展平维）尾部补零
                    hidden_states = torch.nn.functional.pad(
                        hidden_states, (0, 0) * (hidden_states.ndim - 1) + (0, padding)
                    )
                hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)
            else:
                # 常规 TP 路径：全量 all-reduce，每张卡得到完整结果
                hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        # 执行 GemmaRMSNorm（带残差），返回 (归一化输出, 新残差)
        return norm(hidden_states, residual)

    cast(Any, fallback_module).fused_allreduce_gemma_rms_norm = fused_allreduce_gemma_rms_norm
    # 中文注释：把 fallback 模块注入 sys.modules —— 之后任何代码
    # `import vllm.model_executor.layers.fused_allreduce_gemma_rms_norm` 都会命中这个
    # 假模块，从而拿到 NPU 版实现。这是 monkey-patch 依赖模块的标准手法。
    sys.modules[module_name] = fallback_module


# 模块加载时立即安装 fallback，必须早于视觉塔的导入（视觉塔内部会 import 该 CUDA 模块）
_install_fused_allreduce_norm_fallback()

# 加载 vLLM 公共视觉塔实现 MiniMaxVLVisionModel（ViT + 多模态投影 + patch 合并 MLP）
MiniMaxVLVisionModel = _load_vllm_minimax_m3_common_module("vision_tower").MiniMaxVLVisionModel


class MiniMaxM3VLModel(nn.Module):
    """中文注释：MiniMax M3 VL 的内部模型容器（视觉塔 + 语言模型）。

    作用：
        组合 MiniMaxVLVisionModel（视觉编码器，复用 vLLM 公共实现）与
        MiniMaxM3SparseForCausalLM（昇腾原生语言模型，含 MSA 稀疏注意力与 MoE）。
        forward 时输入已是拼接好的 inputs_embeds（多模态 embedding 已替换到
        对应 placeholder token 位置），故直接透传给语言模型。

    张量流：
        input_ids/inputs_embeds --> language_model --> hidden_states
        （视觉塔由外层 forward 之前通过 embed_multimodal 单独调用）
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # hf_config: VL 顶层配置（含 vision_config）；hf_text_config: 语言模型子配置
        config = vllm_config.model_config.hf_config
        text_config = vllm_config.model_config.hf_text_config
        vision_config = getattr(config, "vision_config", None)
        if vision_config is None:
            raise ValueError("MiniMax-M3 VL requires config.vision_config.")

        # 语法点：getattr 返回的可能是 dict，PretrainedConfig.from_dict 把它转成
        # 可用属性访问的对象，方便视觉塔读取超参。
        if isinstance(vision_config, dict):
            vision_config = PretrainedConfig.from_dict(vision_config)

        projector_hidden_size = getattr(config, "projector_hidden_size", None)
        # 视觉塔：输入 pixel_values，输出已投影到语言模型 hidden_size 的 embedding。
        # text_hidden_size 用于把视觉特征投影到语言模型的隐空间维度。
        self.vision_tower = MiniMaxVLVisionModel(
            config=vision_config,
            text_hidden_size=text_config.hidden_size,
            projector_hidden_size=projector_hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "vision_tower"),
        )
        # 语言模型：昇腾原生实现（MSA 稀疏注意力 + MoE），即 minimax_m3.py 中的主体
        self.language_model = MiniMaxM3SparseForCausalLM(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        # positions: [num_tokens] 每个 token 的绝对位置 id（RoPE 用）
        positions: torch.Tensor,
        # intermediate_tensors: 流水线并行（PP）中间张量，非首 stage 时接收上游输出
        intermediate_tensors: IntermediateTensors | None = None,
        # inputs_embeds: [num_tokens, hidden_size] 已含多模态 embedding 的输入嵌入
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        """中文注释：语言模型前向。多模态拼接已完成，直接透传。"""
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )


@MULTIMODAL_REGISTRY.register_processor(
    MiniMaxM3VLMultiModalProcessor,
    info=MiniMaxM3VLProcessingInfo,
    dummy_inputs=MiniMaxM3VLDummyInputsBuilder,
)
class MiniMaxM3SparseForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsLoRA,
    SupportsPP,
    SupportsEagle3,
    MixtureOfExperts,
):
    """中文注释：MiniMax M3 VL 多模态生成模型顶层封装（昇腾版）。

    继承关系（多继承 mixin）：
        - nn.Module            : PyTorch 模块基类
        - SupportsMultiModal   : 声明支持多模态输入（图像/视频），vLLM 由此调用
                                  embed_multimodal 等接口
        - SupportsLoRA         : 支持 LoRA 微调适配器
        - SupportsPP           : 支持流水线并行（Pipeline Parallelism）
        - SupportsEagle3       : 支持 EAGLE3 投机解码（提供 aux hidden states）
        - MixtureOfExperts     : 声明 MoE 模型接口（EPLB 专家负载均衡等需要）

    类装饰器 @MULTIMODAL_REGISTRY.register_processor(...)：
        向 vLLM 多模态注册表登记输入处理器/信息/dummy 输入构造器三件套，
        使调度器知道如何预处理图像/视频数据并按需注册 placeholder。

    NPU 适配点：
        - 视觉塔直接复用 vLLM 公共实现（硬件无关）；
        - 语言模型走本插件的 MSA 稀疏注意力 + MoE NPU 实现；
        - fused allreduce+norm 已替换为 NPU fallback（见文件头部）。
    """

    # 编码器（视觉塔）支持 TP 切分数据：vLLM 会按此标志分发视觉输入
    supports_encoder_tp_data = True

    packed_modules_mapping = {
        **MiniMaxM3SparseForCausalLM.packed_modules_mapping,
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "model.language_model.model.",
            "language_model.model.": "model.language_model.model.",
            "language_model.lm_head.": "model.language_model.lm_head.",
            "model.vision_tower.": "model.vision_tower.",
            "vision_tower.": "model.vision_tower.",
            "multi_modal_projector.": ("model.vision_tower.multi_modal_projector."),
            "patch_merge_mlp.": "model.vision_tower.patch_merge_mlp.",
            "lm_head.": "model.language_model.lm_head.",
        },
        orig_to_new_substr={
            ".mlp.fc1.": ".fc1.",
            ".mlp.fc2.": ".fc2.",
        },
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        """中文注释：返回第 i 个指定模态输入在文本中的占位符字符串。

        MiniMax M3 使用自定义占位符语法 "]<]image[>[" / "]<]video[>["（区别于
        常见的 <image>），多模态处理器会把它们替换为对应模态的 embedding。
        参数：modality 为 "image"/"video"；i 为该模态第几个输入的序号。
        """
        if modality == "image":
            return "]<]image[>["
        if modality == "video":
            return "]<]video[>["
        raise ValueError(f"Unsupported modality: {modality!r}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.model_config = vllm_config.model_config
        self.multimodal_config = vllm_config.model_config.multimodal_config
        # mm_encoder_tp_mode == "data" 表示视觉编码器用"数据并行"模式：
        # 各 TP rank 各处理一部分图像 patch，而非按权重切分（模型并行）
        self.use_data_parallel = (
            self.multimodal_config is not None and self.multimodal_config.mm_encoder_tp_mode == "data"
        )

        # 语法点：with (a, b): 同时进入两个上下文管理器（Python 3.10+ 组合写法）。
        # _mark_language_model / _mark_tower_model 是 SupportsMultiModal 提供的
        # 上下文管理器，用于在构建子模型期间登记"这部分属于语言模型/视觉塔"，
        # 使 vLLM 正确处理两类子模型不同的并行与数据分发策略。
        with (
            self._mark_language_model(
                vllm_config,
                targets=MiniMaxM3SparseForCausalLM,
            ),
            self._mark_tower_model(
                vllm_config,
                {"image", "video"},
                targets=MiniMaxVLVisionModel,
            ),
        ):
            self.model = MiniMaxM3VLModel(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "model"),
            )

        # 暴露子模块为顶层属性，方便 vLLM 的通用逻辑（如 load_weights）按
        # self.vision_tower / self.language_model 访问
        self.vision_tower = self.model.vision_tower
        self.language_model = self.model.language_model
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors
        # 同步 MoE 元信息（专家数、MoE 层列表等）到本层，供 EPLB 等使用
        self._sync_moe_parameters()

    def _sync_moe_parameters(self) -> None:
        """中文注释：把语言模型的 MoE 元信息同步到 VL 顶层。

        EPLB（Expert Parallelism Load Balancing，专家负载均衡）与专家并行调度
        需要在模型顶层读取专家数量/层级等信息；VL 顶层只是转发语言模型的同名属性。
        """
        language_model = self.language_model
        self.expert_weights = language_model.expert_weights
        self.num_expert_groups = language_model.num_expert_groups
        self.moe_layers = language_model.moe_layers
        self.moe_mlp_layers = language_model.moe_mlp_layers
        self.num_moe_layers = language_model.num_moe_layers
        self.num_logical_experts = language_model.num_logical_experts
        self.num_physical_experts = language_model.num_physical_experts
        self.num_local_physical_experts = language_model.num_local_physical_experts
        self.num_routed_experts = language_model.num_routed_experts
        self.num_shared_experts = language_model.num_shared_experts
        self.num_redundant_experts = language_model.num_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        self.language_model.update_physical_experts_metadata(
            num_physical_experts,
            num_local_physical_experts,
        )
        self._sync_moe_parameters()

    @property
    def lm_head(self) -> nn.Module:
        return self.language_model.lm_head

    def _parse_and_validate_image_input(self, **kwargs: object) -> dict | None:
        """中文注释：从 forward 的 kwargs 中提取并校验图像输入。

        支持两种形式（互斥）：
          - pixel_values + image_grid_thw : 原始像素（需过视觉塔），grid_thw 记录
            每张图的 (t, h, w) 网格尺寸
          - image_embeds + image_grid_thw  : 已预计算好的视觉 embedding（跳过视觉塔）
        无图像输入时返回 None。
        """
        # 语法点：kwargs.pop(name, None) 从字典取值并删除该键，不存在时返回默认值
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        image_embeds = kwargs.pop("image_embeds", None)
        if pixel_values is None and image_embeds is None:
            return None
        if pixel_values is not None:
            return {
                "type": "pixel_values",
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            }
        return {
            "type": "image_embeds",
            "image_embeds": image_embeds,
            "image_grid_thw": image_grid_thw,
        }

    def _parse_and_validate_video_input(self, **kwargs: object) -> dict | None:
        """中文注释：提取并校验视频输入，逻辑与图像输入相同。

        视频以 pixel_values_videos（多帧像素）或预计算的 video_embeds 提供，
        video_grid_thw 中 t 维为帧数。
        """
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        video_embeds = kwargs.pop("video_embeds", None)
        if pixel_values_videos is None and video_embeds is None:
            return None
        if pixel_values_videos is not None:
            return {
                "type": "pixel_values_videos",
                "pixel_values_videos": pixel_values_videos,
                "video_grid_thw": video_grid_thw,
            }
        return {
            "type": "video_embeds",
            "video_embeds": video_embeds,
            "video_grid_thw": video_grid_thw,
        }

    def _process_image_input(self, image_input: dict) -> tuple[torch.Tensor, ...]:
        """中文注释：运行视觉塔处理图像，返回按每张图切开的 embedding 元组。

        算法步骤：
          1. 若输入是 image_embeds（预计算），直接使用（转换 dtype）；
          2. 否则把 pixel_values 送入视觉塔（ViT + 投影），得到
             [total_visual_tokens, text_hidden_size] 的 embedding；
             DP 模式下调用 run_dp_sharded_mrope_vision_model 按数据并行切分；
          3. 按 spatial_merge_size 计算每张图合并后的 token 数，用 split 切开。
        返回：tuple[Tensor, ...]，每个 Tensor 形如 [tokens_i, hidden_size]。
        """
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw is not None and grid_thw.ndim == 2
        # 转成 Python list，视觉塔接口接收 list[tuple[int,int,int]]
        grid_thw_list = grid_thw.tolist()

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.vision_tower.dtype)
        else:
            pixel_values = image_input["pixel_values"].type(self.vision_tower.dtype)
            if self.use_data_parallel:
                return run_dp_sharded_mrope_vision_model(
                    self.vision_tower,
                    pixel_values,
                    grid_thw_list,
                    rope_type="rope_3d",
                )
            image_embeds = self.vision_tower(
                pixel_values=pixel_values,
                grid_thw=grid_thw_list,
            )

        # 视觉 patch 合并尺寸：spatial_merge_size x spatial_merge_size 个相邻
        # patch 合并为 1 个视觉 token（如 2x2 合并），显著减少视觉 token 数
        merge_size = self.vision_tower.spatial_merge_size
        # 每张图 token 数 = t*h*w / (merge_size^2)；split 按每张图的大小切开
        sizes = (grid_thw.prod(-1) // (merge_size * merge_size)).tolist()
        return image_embeds.split(sizes)

    def _process_video_input(self, video_input: dict) -> tuple[torch.Tensor, ...]:
        """中文注释：运行视觉塔处理视频，逻辑与 _process_image_input 完全一致。"""
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw is not None and grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.vision_tower.dtype)
        else:
            pixel_values = video_input["pixel_values_videos"].type(self.vision_tower.dtype)
            if self.use_data_parallel:
                return run_dp_sharded_mrope_vision_model(
                    self.vision_tower,
                    pixel_values,
                    grid_thw_list,
                    rope_type="rope_3d",
                )
            video_embeds = self.vision_tower(
                pixel_values=pixel_values,
                grid_thw=grid_thw_list,
            )

        merge_size = self.vision_tower.spatial_merge_size
        sizes = (grid_thw.prod(-1) // (merge_size * merge_size)).tolist()
        return video_embeds.split(sizes)

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict[str, dict]:
        """中文注释：统一收集所有模态的输入，返回 {模态名: 输入dict} 映射。

        只要有任一图像相关键（pixel_values/image_embeds）就解析图像输入；
        同理视频。每个模态只解析一次（in 判断防重复）。
        """
        mm_input_by_modality: dict[str, dict] = {}
        for input_key in kwargs:
            if input_key in ("pixel_values", "image_embeds") and "image" not in mm_input_by_modality:
                image_input = self._parse_and_validate_image_input(**kwargs)
                if image_input is not None:
                    mm_input_by_modality["image"] = image_input
            if input_key in ("pixel_values_videos", "video_embeds") and "video" not in mm_input_by_modality:
                video_input = self._parse_and_validate_video_input(**kwargs)
                if video_input is not None:
                    mm_input_by_modality["video"] = video_input
        return mm_input_by_modality

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """中文注释：vLLM 多模态接口 —— 计算所有模态输入的 embedding。

        调用时机：vLLM 在构造 inputs_embeds 之前调用本方法，把返回的 embedding
        元组按顺序替换到 placeholder token 位置上。
        返回：tuple[Tensor, ...]，与多模态输入项一一对应；无输入时返回空 tuple。
        """
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return []

        multimodal_embeddings: list[torch.Tensor] = []
        # 逐模态处理并汇集成扁平列表（每个元素对应一个多模态输入项）
        for modality, multimodal_input in mm_input_by_modality.items():
            if modality == "image":
                multimodal_embeddings.extend(self._process_image_input(multimodal_input))
            elif modality == "video":
                multimodal_embeddings.extend(self._process_video_input(multimodal_input))
        return tuple(multimodal_embeddings)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """中文注释：把 input_ids 映射为 embedding，并替换多模态占位符。

        委托 SupportsMultiModal.embed_input_ids 实现：先查词嵌入表，再把
        multimodal_embeddings 逐个填到 placeholder 位置。is_multimodal 是
        bool 掩码，标记哪些 token 是多模态占位符。
        """
        return SupportsMultiModal.embed_input_ids(
            self,
            input_ids,
            multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        """中文注释：VL 模型前向。视觉编码在 vLLM 引擎层已通过 embed_multimodal
        完成并写入 inputs_embeds，这里直接调用语言模型。"""
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.language_model.set_aux_hidden_state_layers(layers)

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return self.language_model.get_eagle3_default_aux_hidden_state_layers()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.language_model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """中文注释：加载 VL checkpoint 权重（视觉塔 + 语言模型）。

        使用 AutoWeightsLoader 递归加载，配合 hf_to_vllm_mapper 做 checkpoint
        命名到模块命名的映射。本方法额外统计各前缀（language_model/
        vision_tower/multi_modal_projector/patch_merge_mlp）的张量数量并打印
        日志，便于排查漏载/错载。
        返回：成功加载的参数名集合。
        """
        loader = AutoWeightsLoader(self)
        raw_tensors = 0
        prefix_counts: dict[str, int] = {}

        def counted_weights() -> Iterable[tuple[str, torch.Tensor]]:
            nonlocal raw_tensors
            for name, weight in weights:
                raw_tensors += 1
                if name.startswith("language_model."):
                    bucket = "language_model"
                elif name.startswith("vision_tower."):
                    bucket = "vision_tower"
                elif name.startswith("multi_modal_projector."):
                    bucket = "multi_modal_projector"
                elif name.startswith("patch_merge_mlp."):
                    bucket = "patch_merge_mlp"
                else:
                    bucket = name.split(".", 1)[0]
                prefix_counts[bucket] = prefix_counts.get(bucket, 0) + 1
                yield name, weight

        logger.warning("MiniMax M3 VL load_weights entered")
        loaded_params = loader.load_weights(counted_weights(), mapper=self.hf_to_vllm_mapper)
        logger.warning(
            "MiniMax M3 VL load_weights saw %d checkpoint tensors by prefix %s; returned %d loaded parameter names",
            raw_tensors,
            prefix_counts,
            len(loaded_params),
        )
        return loaded_params
