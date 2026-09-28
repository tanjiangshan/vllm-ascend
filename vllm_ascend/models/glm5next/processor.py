# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM-native multimodal processor for GLM-5.3-Flash.

GLM-5.3-Flash 的 vLLM 原生多模态处理器（图像/视频 -> 模型输入）。

处理流水线：
  1. 帧采样（视频）：glm_sample_frame_indices 按 fps-interval 策略选帧；
  2. 几何：smart_resize 在 token 预算内计算"向上对齐画布"
     （h/w 向上取整到 patch*merge*patch_expand 的倍数，只补边不裁剪）；
  3. 像素变换：resize/pad + rescale(1/255) + CLIP mean/std 归一化；
  4. patchify：重排为 [B, grid_t*grid_h*grid_w, C*T*P*P] 展平 patch；
  5. 分词：文本原样过 tokenizer（prompt 展开由 vLLM 的
     prompt-update 机制完成）。

关键概念：
  - token 预算（min/max_image_tokens）：一个视觉 token 覆盖
    temporal_patch_size * (patch*merge)^2 像素，processor_config.json
    以 token 数而非像素数给预算；
  - patch_expand_factor：GLM 特有的对齐因子倍乘（checkpoint 值为 1）；
  - resize_mode="pad"：保持纵横比缩放后右侧/底部补零。

本文件不再依赖 transformers 的 GLM processor 类（checkpoint 的
processor_config.json 声明了自定义 processor_class，AutoProcessor
无法解析），是训练侧流水线的移植。
"""

import json
import math
import os

import numpy as np
import torch
from torchvision.transforms.v2 import functional as tvF
from transformers.image_processing_utils import BatchFeature
from transformers.image_processing_utils_fast import (
    BaseImageProcessorFast,
    group_images_by_shape,
    reorder_images,
)
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size,
)
from transformers.models.auto.image_processing_auto import get_image_processor_config
from transformers.processing_utils import (
    ImagesKwargs,
    MultiModalData,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
    VideosKwargs,
)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import TensorType, logging
from transformers.video_processing_utils import BaseVideoProcessor
from transformers.video_utils import (
    VideoInput,
    VideoMetadata,
    group_videos_by_shape,
    reorder_videos,
)

logger = logging.get_logger(__name__)

# Cap video inputs at 30,000 vision tokens to keep encoder profiling from
# starving the KV cache. Image inputs retain their checkpoint-defined budget.
# 视频输入上限 30000 视觉 token：防止编码器 profiling 挤占 KV cache
# 显存。图像输入保持 checkpoint 定义的预算。
_MAX_VIDEO_TOKENS = 30000

# Frame-sampler fallbacks (mirror the checkpoint's fps_interval=2 /
# max_frame_count_dynamic=2048); used when neither the request nor the
# processor config overrides them.
# 帧采样兜底默认值（镜像 checkpoint 的 fps_interval=2 /
# max_frame_count_dynamic=2048）；请求与处理器配置都未覆盖时使用。
GLM_VIDEO_DEFAULT_FPS = 2.0
GLM_VIDEO_DEFAULT_MAX_FRAMES = 2048


def glm_sample_frame_indices(
    total_frames: int,
    fps: float,
    duration: float,
    *,
    target_fps: float | None = None,
    max_frame_count: int | None = None,
    temporal_patch_size: int = 2,
) -> list[int]:
    """GLM video frame sampling (training-reference parity).

    GLM 视频帧采样（与训练参考实现对齐）。

    ``target_fps`` is the ``fps_interval`` request knob. The greedy walk
    advances at ``1 / (temporal_patch_size * target_fps)`` seconds, so on
    frame-dense sources it collects more candidates than ``extract_t`` and
    the ``> extract_t`` fixup re-spreads the picks uniformly with
    ``np.linspace`` -- that fallback is the intended reference behavior, not
    an accident. Short clips (fewer frames than ``extract_t``) are spread at
    evenly spaced timestamps (``floor`` sampling; the linspace variant
    samples frames unevenly and cost 4 points on video grounding evals).
    Request overrides: ``target_fps`` -> fps interval, ``max_frame_count``
    -> frame cap.

    参数（语法点：* 之后 keyword-only）：
        total_frames: 视频总帧数。
        fps: 源视频帧率。
        duration: 时长（秒）；0 时由帧数/帧率推算。
        target_fps: 目标采样率（fps_interval 请求旋钮）。
        max_frame_count: 帧数上限。
        temporal_patch_size: 时间 patch 大小（采样步长含此因子）。

    返回：
        list[int]: 选中的帧下标（去重、偶数个——不足时复制最后一帧
        补齐成 temporal_patch_size 的整数倍）。

    算法：
      1. extract_t = min(duration*target_fps, max_frame_count)：目标帧数；
      2. 帧多时：贪心游走，每 1/(T*fps) 秒取一帧；
      3. 帧少时：floor 等间隔铺满 extract_t；
      4. 兜底：候选数不等于 extract_t 时用 np.linspace 重铺。
    """
    max_frame_idx = total_frames - 1
    if not duration:
        # 时长缺失：由最后一帧时间戳推算（round(idx/fps)+1）。
        duration = (round(max_frame_idx / fps) + 1) if fps else 0
    if max_frame_count is None:
        max_frame_count = GLM_VIDEO_DEFAULT_MAX_FRAMES
    if target_fps is None:
        target_fps = GLM_VIDEO_DEFAULT_FPS

    # 步骤1: 目标帧数。
    extract_t = int(duration * target_fps)
    extract_t = min(extract_t, int(max_frame_count))

    # 步骤2: 每帧的时间戳与最大秒数。
    duration_per_frame = 1 / fps
    timestamps = [i * duration_per_frame for i in range(total_frames)]
    max_second = int(duration)

    if total_frames < extract_t:
        # 步骤3a: 短片——floor 等间隔铺满（linspace 版本采样不均，
        # 在视频 grounding 评测上掉 4 分，故用 floor）。
        frame_indices = [math.floor(_i * total_frames / extract_t) for _i in range(extract_t)]
    else:
        # 步骤3b: 贪心游走——时间戳到达 current_second 即取帧，
        # 步进 1/(temporal_patch*target_fps) 秒（含时间 patch 因子）。
        frame_indices = []
        current_second = 0.0
        inv_fps = 1 / (temporal_patch_size * target_fps)
        for frame_index in range(total_frames):
            if timestamps[frame_index] >= current_second:
                current_second += inv_fps
                frame_indices.append(frame_index)
                if current_second >= max_second:
                    break

    # 步骤4: 候选数不等于目标时的 linspace 重铺（多/少皆兜底）。
    if len(frame_indices) < extract_t:
        if len(frame_indices) == 0:
            start, end = 0, max(total_frames - 1, 0)
        else:
            start, end = frame_indices[0], frame_indices[-1]
        frame_indices = np.linspace(start, end, extract_t, dtype=int).tolist()
    elif len(frame_indices) > extract_t:
        frame_indices = np.linspace(0, total_frames - 1, extract_t, dtype=int).tolist()

    # 步骤5: 去重保序。
    seen, uniq = set(), []
    for idx in frame_indices:
        if idx not in seen:
            seen.add(idx)
            uniq.append(int(idx))

    # 步骤6: 奇数帧补最后一帧（凑成 temporal_patch_size 的偶数倍）。
    if len(uniq) & 1:
        uniq.append(uniq[-1])

    return uniq


def _ceil_to_factor(value: int, factor: int) -> int:
    """Round a positive integer upward to the nearest multiple of factor.

    把正整数向上取整到 factor 的倍数（对齐函数）。
    """
    return math.ceil(value / factor) * factor


def _fit_aligned_size_within_budget(
    t: int,
    h: int,
    w: int,
    h_factor: int,
    w_factor: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Largest proportional size whose upward-aligned canvas fits the budget.

    在像素预算内找"等比缩放 + 向上对齐"后最大的画布尺寸。

    Binary search on the unaligned content height; each candidate is rounded
    upward to h_factor/w_factor, so the returned canvas always satisfies
    ``t * aligned_h * aligned_w <= max_pixels``.

    参数：
        t: 时间 patch 数（帧维）。
        h/w: 原始高宽。
        h_factor/w_factor: 高/宽对齐因子。
        max_pixels: t*h*w 总像素预算。

    返回：
        (aligned_h, aligned_w)：对齐后且满足预算的最大画布。
    """
    # 预算连一个对齐 patch 都放不下时报错。
    minimum_pixels = t * h_factor * w_factor
    if max_pixels < minimum_pixels:
        raise ValueError(
            f"max_pixels={max_pixels} is too small. At least "
            f"{minimum_pixels} pixels are required for one aligned patch."
        )

    # 二分搜索未对齐的内容高度 low..high；候选按比例求宽后向上对齐，
    # 满足预算则记录并尝试更大（low 右移），否则缩小（high 左移）。
    low, high = 1, h
    best_h, best_w = h_factor, w_factor
    while low <= high:
        content_h = (low + high) // 2
        # 等比宽度（向下取整防超预算）。
        content_w = max(1, math.floor(w * content_h / h))
        aligned_h = _ceil_to_factor(content_h, h_factor)
        aligned_w = _ceil_to_factor(content_w, w_factor)
        if t * aligned_h * aligned_w <= max_pixels:
            best_h, best_w = aligned_h, aligned_w
            low = content_h + 1
        else:
            high = content_h - 1
    return best_h, best_w


def smart_resize(
    t: int,
    h: int,
    w: int,
    t_factor: int = 1,
    h_factor: int = 28,
    w_factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> tuple[int, int]:
    """GLM-5.3-Flash ``smart_resize``: upward-aligned canvas under a
    ``t_bar * h_bar * w_bar`` pixel budget.

    GLM-5.3-Flash 的智能缩放：在 t*h*w 像素预算内的向上对齐画布。

    Height/width always round UP to their factors (content is then padded,
    never cropped or distorted); an over-budget canvas is refit by binary
    search instead of one-shot square-root scaling. ``h_factor`` /
    ``w_factor`` carry ``patch_expand_factor`` on top of
    ``patch_size * merge_size``; ``t_factor`` is ``temporal_patch_size``. For
    a still image ``t = t_factor = temporal_patch_size`` so ``t_bar =
    temporal_patch_size``.

    参数：
        t/h/w: 时间（帧）/高/宽。
        t_factor: 时间对齐因子（temporal_patch_size）。
        h_factor/w_factor: 高/宽对齐因子（patch*merge*patch_expand）。
        min_pixels/max_pixels: 总像素下/上限。

    返回：
        (h_bar, w_bar)：对齐后的画布高宽（内容再按需缩放+补边）。
    """
    if min(t, h, w, t_factor, h_factor, w_factor) <= 0:
        raise ValueError("Image dimensions and alignment factors must be positive.")
    if min_pixels <= 0 or max_pixels <= 0:
        raise ValueError("min_pixels and max_pixels must be positive.")
    if min_pixels > max_pixels:
        raise ValueError("min_pixels must be less than or equal to max_pixels.")

    # 步骤1: 时间向上对齐（t_bar 至少 t_factor）；高宽向上取整到因子倍。
    t_bar = max(t_factor, round(t / t_factor) * t_factor)
    h_bar = _ceil_to_factor(h, h_factor)
    w_bar = _ceil_to_factor(w, w_factor)

    if t_bar * h_bar * w_bar > max_pixels:
        # 步骤2a: 超预算——二分搜索找预算内最大的对齐画布。
        h_bar, w_bar = _fit_aligned_size_within_budget(
            t=t_bar,
            h=h,
            w=w,
            h_factor=h_factor,
            w_factor=w_factor,
            max_pixels=max_pixels,
        )
    elif t_bar * h_bar * w_bar < min_pixels:
        # 步骤2b: 低于下限——按 sqrt 比例放大内容再对齐。
        beta = math.sqrt(min_pixels / (t * h * w))
        h_bar = _ceil_to_factor(max(1, math.ceil(h * beta)), h_factor)
        w_bar = _ceil_to_factor(max(1, math.ceil(w * beta)), w_factor)

        # Alignment can push a candidate slightly over a tight max_pixels
        # budget. Refit it when that happens.
        # 对齐可能把候选轻微推出紧凑的 max_pixels 预算，超出则重拟合。
        if t_bar * h_bar * w_bar > max_pixels:
            h_bar, w_bar = _fit_aligned_size_within_budget(
                t=t_bar,
                h=h,
                w=w,
                h_factor=h_factor,
                w_factor=w_factor,
                max_pixels=max_pixels,
            )

    return h_bar, w_bar


def _get_pad_content_size(
    image_height: int,
    image_width: int,
    canvas_height: int,
    canvas_width: int,
    allow_upscale: bool = False,
) -> tuple[int, int]:
    """Aspect-ratio-preserving content size that fits the canvas.

    保持纵横比、能放进画布的内容尺寸。

    Oversized images are shrunk proportionally. Small images are enlarged
    only when ``allow_upscale``. Padding is applied after the resize.

    参数：
        image_height/width: 原始尺寸。
        canvas_height/width: 目标画布尺寸。
        allow_upscale: 允许放大小图（小于 min_pixels 时启用）。

    返回：
        (content_height, content_width)：缩放后的内容尺寸（补边前）。
    """
    # 缩放比取两维的较小者（保证都放得下）。
    scale = min(canvas_height / image_height, canvas_width / image_width)
    if not allow_upscale:
        scale = min(1.0, scale)
    content_height = max(1, min(canvas_height, math.floor(image_height * scale)))
    content_width = max(1, min(canvas_width, math.floor(image_width * scale)))
    return content_height, content_width


def _resize_or_pad(
    stacked_images: torch.Tensor,
    target_height: int,
    target_width: int,
    resize_mode: str,
    resample: "PILImageResampling | tvF.InterpolationMode | int | None",
    resize,
    allow_upscale: bool = False,
) -> torch.Tensor:
    """Resize onto the aligned canvas, or keep the aspect ratio and
    zero-pad the right/bottom sides (``resize_mode="pad"``).

    缩放到对齐画布，或保持纵横比并在右侧/底部补零（pad 模式）。

    参数：
        stacked_images: [B, C, H, W] 待处理图像栈。
        target_height/width: 画布尺寸。
        resize_mode: "resize"（直接拉伸）或 "pad"（等比+补零）。
        resample: 重采样插值方式。
        resize: 缩放函数（self.resize，torchvision fast 实现）。
        allow_upscale: 允许放大小图。

    返回：
        [B, C, target_h, target_w] 处理后的图像栈。
    """
    height, width = stacked_images.shape[-2:]

    if resize_mode == "resize":
        # 模式1: 直接缩放到画布（可能变形）。
        return resize(
            stacked_images,
            size=SizeDict(height=target_height, width=target_width),
            resample=resample,
        )

    if resize_mode != "pad":
        raise ValueError("resize_mode must be either 'resize' or 'pad'.")

    # 模式2: pad——先算画布内等比内容尺寸。
    content_height, content_width = _get_pad_content_size(
        image_height=height,
        image_width=width,
        canvas_height=target_height,
        canvas_width=target_width,
        allow_upscale=allow_upscale,
    )

    if (content_height, content_width) != (height, width):
        # 内容与原尺寸不同才缩放（省一次插值）。
        stacked_images = resize(
            stacked_images,
            size=SizeDict(height=content_height, width=content_width),
            resample=resample,
        )

    # torchvision padding order: [left, top, right, bottom] -> pad only the
    # right and bottom sides.
    # torchvision 的 padding 顺序是 [left, top, right, bottom]——
    # 只补右边和底边（fill=0 零填充）。
    return tvF.pad(
        stacked_images,
        padding=[0, 0, target_width - content_width, target_height - content_height],
        fill=0,
    )


def _pixel_budget(
    min_image_tokens: int | None,
    max_image_tokens: int | None,
    patch_size: int,
    merge_size: int,
    temporal_patch_size: int,
) -> tuple[int, int]:
    """(min_pixels, max_pixels) from the token bounds of
    ``processor_config.json``; one vision token covers
    ``temporal_patch_size * (patch_size * merge_size) ** 2`` pixels.

    由 processor_config.json 的 token 预算推导 (min_pixels, max_pixels)。
    一个视觉 token 覆盖 temporal_patch_size * (patch*merge)^2 像素。

    参数：
        min_image_tokens/max_image_tokens: 视觉 token 下/上限。
        patch_size: patch 边长。
        merge_size: 空间合并边长。
        temporal_patch_size: 时间 patch 大小。

    返回：
        (min_pixels, max_pixels)。
    """
    if min_image_tokens is None or max_image_tokens is None:
        raise ValueError(
            "min_image_tokens and max_image_tokens must be provided by processor_config.json (or per-call kwargs)."
        )
    factor = temporal_patch_size * (patch_size * merge_size) ** 2
    return min_image_tokens * factor, max_image_tokens * factor


class Glm5NextImageProcessorKwargs(ImagesKwargs, total=False):  # type: ignore[call-arg]
    """图像处理器的合法 kwargs 类型声明（TypedDict）。

    语法点：TypedDict + total=False——所有键可选，用于处理器的
    参数类型检查与 IDE 补全；# type: ignore 抑制 mypy 对动态生成的
    父类属性的误报。
    """

    patch_size: int | None
    temporal_patch_size: int | None
    merge_size: int | None
    patch_expand_factor: int | None
    resize_mode: str | None
    min_image_tokens: int | None
    max_image_tokens: int | None


class Glm5NextImageProcessor(BaseImageProcessorFast):
    """Fast torchvision image processor for GLM-5.3-Flash.

    GLM-5.3-Flash 的快速图像处理器（torchvision 后端）。

    ``patch_expand_factor`` multiplies into the ``smart_resize`` spatial
    factor ``patch_size * merge_size``. ``resize_mode`` picks the geometry:
    ``"pad"`` (default) preserves the aspect ratio and zero-pads the
    right/bottom of the upward-aligned canvas, ``"resize"`` stretches onto
    it. Defaults mirror the checkpoint's ``image_processor`` config.

    继承 transformers 的 BaseImageProcessorFast（快速图像处理器基类，
    提供 preprocess 模板/分组/重排等骨架；子类实现 _preprocess）。
    类属性即默认配置（可被 processor_config.json 覆盖）。
    """

    do_resize = True
    resample = PILImageResampling.BICUBIC
    size = {"longest_edge": 1}  # unused: budgets come from the token bounds
    # size 字段未用：预算来自 token 上下限（GLM 特有）。
    do_rescale = True
    do_normalize = True
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    do_convert_rgb = True
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    patch_expand_factor = 1
    resize_mode = "pad"
    min_image_tokens = 16
    max_image_tokens = 8000
    valid_kwargs = Glm5NextImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    def _preprocess(
        self,
        images: list[torch.Tensor],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        patch_expand_factor: int,
        resize_mode: str | None,
        min_image_tokens: int | None,
        max_image_tokens: int | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        """图像预处理主体（由基类 preprocess 模板调用）。

        参数：
            images: 图像张量列表。
            do_resize/do_rescale/do_normalize: 各级开关。
            size/resample: 缩放尺寸与插值（size 未用，预算制）。
            patch_size/temporal_patch_size/merge_size: patch 几何。
            patch_expand_factor: 对齐因子倍乘。
            resize_mode: "pad" / "resize"。
            min/max_image_tokens: token 预算。
            disable_grouping: 禁用同形状分组（图捕获确定性）。
            return_tensors: 返回张量类型。

        返回：
            BatchFeature：pixel_values [L, C*T*P*P] 与
            image_grid_thw [n, 3]（每图的 (t, h, w) 网格）。
        """
        # 步骤1: 参数兜底 + token 预算 -> 像素预算。
        resize_mode = resize_mode if resize_mode is not None else self.resize_mode
        min_pixels, max_pixels = _pixel_budget(
            min_image_tokens if min_image_tokens is not None else self.min_image_tokens,
            max_image_tokens if max_image_tokens is not None else self.max_image_tokens,
            patch_size,
            merge_size,
            temporal_patch_size,
        )
        # 步骤2: 按相同形状分组处理（批量化缩放更快），最后按原顺序重排。
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            height, width = stacked_images.shape[-2:]
            if do_resize:
                # 步骤2a: smart_resize 求对齐画布，再 resize/pad。
                resized_height, resized_width = smart_resize(
                    t=temporal_patch_size,
                    h=height,
                    w=width,
                    t_factor=temporal_patch_size,
                    h_factor=patch_size * merge_size * patch_expand_factor,
                    w_factor=patch_size * merge_size * patch_expand_factor,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
                stacked_images = _resize_or_pad(
                    stacked_images,
                    target_height=resized_height,
                    target_width=resized_width,
                    resize_mode=resize_mode,
                    resample=resample,
                    resize=self.resize,
                    # 小于 min_pixels 时允许放大（否则补零后信息量不足）。
                    allow_upscale=(temporal_patch_size * height * width < min_pixels),
                )
            resized_images_grouped[shape] = stacked_images

        # 按原始输入顺序重排（分组打乱了顺序）。
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # 步骤3: 归一化 + patchify（再次分组处理）。
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        processed_grids = {}

        for shape, stacked_images in grouped_images.items():
            resized_height, resized_width = stacked_images.shape[-2:]

            # 步骤3a: rescale(1/255) + CLIP mean/std 归一化。
            patches = self.rescale_and_normalize(
                stacked_images,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
            )
            if patches.ndim == 4:  # (B, C, H, W)
                patches = patches.unsqueeze(1)  # (B, T=1, C, H, W)
                # 补时间维 T=1（图像只有一"帧"）。

            # 步骤3b: 帧数补齐到 temporal_patch_size 的倍数（复制最后一帧）。
            if patches.shape[1] % temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(
                    1,
                    temporal_patch_size - (patches.shape[1] % temporal_patch_size),
                    1,
                    1,
                    1,
                )
                patches = torch.cat([patches, repeats], dim=1)

            # 步骤3c: 计算网格维度。
            batch_size, t_len, channel = patches.shape[:3]
            grid_t = t_len // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            # 步骤3d: patchify 重排——把 [B, T, C, H, W] 切成
            # [B, gt, tp, C, gh//m, m, P, gw//m, m, P] 十维视图，
            # 再 permute 成 (B, gt, gh, gw, mh, mw, C, tp, ph, pw)
            # 使每个视觉 token 对应的 (mh=m, mw=m) 空间块与时间块相邻。
            patches = patches.view(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            # (B, grid_t, gh, gw, mh, mw, C, tp, ph, pw)
            patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)

            # 步骤3e: 展平成 [B, gt*gh*gw, C*tp*P*P]——
            # 每行是一个视觉 token 对应的全部像素（conv3d 输入格式）。
            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )

            processed_images_grouped[shape] = flatten_patches
            # 网格 (t, h, w)：每图一份。
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        # 步骤4: 重排回原顺序并拼接成批。
        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_grids = reorder_images(processed_grids, grouped_images_index)

        pixel_values = torch.cat(processed_images, dim=0)
        image_grid_thw = torch.tensor(processed_grids)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
            tensor_type=return_tensors,
        )

    def preprocess(self, images: ImageInput, **kwargs: Unpack[Glm5NextImageProcessorKwargs]) -> BatchFeature:
        """预处理入口（透传给基类模板，参数类型由 Kwargs TypedDict 约束）。

        语法点：Unpack[TypedDict]——把 TypedDict 展开成 **kwargs 的
        键值类型约束（PEP 692）。
        """
        return super().preprocess(images, **kwargs)

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs: dict | None = None) -> int:
        """Number of image patches (pre-merge) for a given (height, width).

        给定 (height, width) 的 patch 数（合并前）——供 vLLM 做 token
        预算估算（无需真正跑一遍预处理）。

        参数：
            height/width: 原始图像尺寸。
            images_kwargs: 可覆盖处理器参数。

        返回：
            int: grid_h * grid_w（未除 merge^2）。
        """
        images_kwargs = images_kwargs or {}
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        patch_expand_factor = images_kwargs.get("patch_expand_factor", self.patch_expand_factor)
        min_pixels, max_pixels = _pixel_budget(
            images_kwargs.get("min_image_tokens", self.min_image_tokens),
            images_kwargs.get("max_image_tokens", self.max_image_tokens),
            patch_size,
            merge_size,
            self.temporal_patch_size,
        )
        # 与 _preprocess 相同的几何推导：smart_resize -> 网格。
        resized_height, resized_width = smart_resize(
            t=self.temporal_patch_size,
            h=height,
            w=width,
            t_factor=self.temporal_patch_size,
            h_factor=patch_size * merge_size * patch_expand_factor,
            w_factor=patch_size * merge_size * patch_expand_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        return grid_h * grid_w


class Glm5NextVideoProcessorKwargs(VideosKwargs, total=False):  # type: ignore[call-arg]
    """视频处理器的合法 kwargs 类型声明（TypedDict，含帧采样旋钮）。"""

    fps: list[float] | float
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    patch_expand_factor: int
    resize_mode: str | None
    target_fps: float | None
    max_frames: int | None
    fps_interval: int | None
    max_frame_count_dynamic: int | None
    min_image_tokens: int | None
    max_image_tokens: int | None


class Glm5NextVideoProcessor(BaseVideoProcessor):
    """Fast video processor for GLM-5.3-Flash.

    GLM-5.3-Flash 的视频处理器。

    Shares ``smart_resize`` / the pad-mode geometry / the patchify with the
    image processor, and adds GLM-5.3-Flash frame sampling
    (``glm_sample_frame_indices``: ``fps_interval`` semantics with a
    temporal-patch-scaled greedy walk). Defaults mirror the checkpoint's
    ``video_processor`` config.

    继承 transformers 的 BaseVideoProcessor（视频处理器基类，提供
    sample_frames 钩子与 preprocess 骨架）。与图像处理器共享
    smart_resize/pad 几何与 patchify，额外做 GLM 帧采样。
    """

    resample = PILImageResampling.BICUBIC
    size = {"longest_edge": 1}  # unused: budgets come from the token bounds
    # size 未用：预算来自 token 上下限。
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    do_sample_frames = True
    patch_size = 14
    temporal_patch_size = 2
    patch_expand_factor = 1
    merge_size = 2
    valid_kwargs = Glm5NextVideoProcessorKwargs
    num_frames = 16
    fps = 2
    fps_interval = 2.0
    max_frame_count_dynamic = 2048
    resize_mode = "pad"
    min_image_tokens = 16
    max_image_tokens = 240000
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def sample_frames(
        self,
        metadata: VideoMetadata,
        fps: int | float | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Sample frame indices with GLM's fps-interval policy.

        用 GLM 的 fps-interval 策略采样帧下标。

        ``fps`` / ``target_fps``, ``max_frames`` and ``fps_interval`` /
        ``max_frame_count_dynamic`` are the overrides described in
        :func:`glm_sample_frame_indices`.

        参数：
            metadata: 视频元数据（总帧数/帧率/时长，必须提供）。
            fps: 目标采样率覆盖（优先于 target_fps）。
            **kwargs: target_fps / max_frames 等覆盖项。

        返回：
            np.ndarray: 选中的帧下标数组。
        """
        if metadata is None or getattr(metadata, "fps", None) is None:
            raise ValueError(
                "Asked to sample frames per second but no video metadata was "
                "provided which is required when sampling in GLM-5.3-Flash. Please "
                "pass in `VideoMetadata` object or set `do_sample_frames=False`."
            )

        # 目标采样率优先级：fps 参数 > target_fps kwarg > 处理器默认。
        target_fps = fps if fps is not None else kwargs.get("target_fps")
        if target_fps is None:
            target_fps = self.fps_interval
        indices = glm_sample_frame_indices(
            metadata.total_num_frames,
            metadata.fps,
            metadata.duration or 0,
            target_fps=target_fps,
            max_frame_count=kwargs.get("max_frames") or self.max_frame_count_dynamic,
            temporal_patch_size=self.temporal_patch_size,
        )
        return np.array(indices)

    def _preprocess(
        self,
        videos: list[torch.Tensor],
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        size: SizeDict | None = None,
        resample: "PILImageResampling | int | None" = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        patch_size: int | None = None,
        temporal_patch_size: int | None = None,
        patch_expand_factor: int | None = None,
        merge_size: int | None = None,
        resize_mode: str | None = None,
        min_image_tokens: int | None = None,
        max_image_tokens: int | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        """视频预处理主体（帧已由 sample_frames 选好）。

        参数：同图像处理器（几何/归一化/patch 参数），外加
        videos: [B, T, C, H, W] 已采样帧栈。

        返回：
            BatchFeature：pixel_values_videos [L, C*T*P*P] 与
            video_grid_thw [n, 3]。
        """
        # 步骤1: 参数兜底 + token -> 像素预算。
        patch_expand_factor = self.patch_expand_factor
        patch_size = patch_size if patch_size is not None else self.patch_size
        temporal_patch_size = temporal_patch_size if temporal_patch_size is not None else self.temporal_patch_size
        merge_size = merge_size if merge_size is not None else self.merge_size
        resize_mode = resize_mode if resize_mode is not None else self.resize_mode
        min_pixels, max_pixels = _pixel_budget(
            min_image_tokens if min_image_tokens is not None else self.min_image_tokens,
            max_image_tokens if max_image_tokens is not None else self.max_image_tokens,
            patch_size,
            merge_size,
            temporal_patch_size,
        )
        # 步骤2: 按形状分组 -> RGB 转换 -> smart_resize + resize/pad。
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            if do_convert_rgb:
                stacked_videos = self.convert_to_rgb(stacked_videos)
            b, t_len, c, h, w = stacked_videos.shape
            num_frames, height, width = t_len, h, w
            if do_resize:
                # 视频的时间维 t = 实际帧数（不是 temporal_patch_size），
                # t_bar 会向上对齐到 temporal_patch_size 的倍数。
                resized_height, resized_width = smart_resize(
                    t=num_frames,
                    h=height,
                    w=width,
                    t_factor=temporal_patch_size,
                    h_factor=patch_size * merge_size * patch_expand_factor,
                    w_factor=patch_size * merge_size * patch_expand_factor,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
                # 把帧维折进批维做统一的 2D resize 再展开。
                stacked_videos = stacked_videos.view(b * t_len, c, h, w)
                stacked_videos = _resize_or_pad(
                    stacked_videos,
                    target_height=resized_height,
                    target_width=resized_width,
                    resize_mode=resize_mode,
                    resample=resample,
                    resize=self.resize,
                    allow_upscale=(num_frames * height * width < min_pixels),
                )
                stacked_videos = stacked_videos.view(b, t_len, c, resized_height, resized_width)
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        # 步骤3: 归一化 + 帧补齐 + patchify（与图像路径同构）。
        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}
        for shape, stacked_videos in grouped_videos.items():
            resized_height, resized_width = get_image_size(stacked_videos[0], channel_dim=ChannelDimension.FIRST)
            stacked_videos = self.rescale_and_normalize(
                stacked_videos,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
            )
            patches = stacked_videos

            # 步骤3a: 帧数补齐到 temporal_patch_size 的倍数
            # （语法点：海象运算符 := 在 if 中赋值 pad 量）。
            if pad := -patches.shape[1] % temporal_patch_size:
                repeats = patches[:, -1:].expand(-1, pad, -1, -1, -1)
                patches = torch.cat((patches, repeats), dim=1)
            batch_size, grid_t, channel = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            # 步骤3b: 十维 patchify 视图 + permute + 展平（同图像路径）。
            patches = patches.view(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )

            processed_videos_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        # 步骤4: 重排 + 拼批，输出 pixel_values_videos 与 video_grid_thw。
        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        pixel_values_videos = torch.cat(processed_videos, dim=0)
        video_grid_thw = torch.tensor(processed_grids)
        return BatchFeature(
            data={
                "pixel_values_videos": pixel_values_videos,
                "video_grid_thw": video_grid_thw,
            },
            tensor_type=return_tensors,
        )


class Glm5NextProcessorKwargs(ProcessingKwargs, total=False):  # type: ignore[call-arg]
    """组合处理器的 kwargs 类型声明（text/images/videos 三段式）。

    _defaults：各子处理器的默认参数——文本不开 padding/token_type，
    视频要求返回元数据（帧采样需要 fps/时长）。
    """

    images_kwargs: Glm5NextImageProcessorKwargs
    videos_kwargs: Glm5NextVideoProcessorKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_token_type_ids": False,
            "return_mm_token_type_ids": False,
        },
        "videos_kwargs": {"return_metadata": True},
    }


class Glm5NextProcessor(ProcessorMixin):
    """Wraps a GLM-5.3-Flash image processor, video processor and tokenizer.

    组合处理器：图像处理器 + 视频处理器 + 分词器。

    Token expansion per image = ``prod(image_grid_thw) // merge_size**2``; video
    frames are expanded with ``<|begin_of_image|>...<|end_of_image|>{ts} seconds``
    structure (mrope timestamps).

    （每张图的 token 展开 = prod(grid_thw) // merge^2；视频帧按
    <|begin_of_image|>...<|end_of_image|>{ts} seconds 结构展开——
    mrope 时间戳。实际的 prompt 展开由 vLLM 的 prompt-update 机制
    负责，本处理器只产出特征与原文 token。）
    """

    # attributes：序列化时保存的子模块名；*_class 声明可加载的类名。
    attributes = ["image_processor", "tokenizer", "video_processor"]
    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = ("PreTrainedTokenizer", "PreTrainedTokenizerFast")

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        **kwargs,
    ) -> None:
        """初始化：记录图像/视频特殊 token 及其 id。

        参数：
            image_processor: Glm5NextImageProcessor。
            tokenizer: 分词器。
            video_processor: Glm5NextVideoProcessor。
            chat_template: 聊天模板。
        """
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)
        # 特殊 token：<|image|> / <|video|>；优先取分词器自带定义。
        self.image_token = "<|image|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = "<|video|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        # 对应 token id（分词器未带时查表转换）。
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Build the processor directly from the checkpoint config.

        直接从 checkpoint 配置构建处理器。

        GLM-5.3-Flash stores nested image/video configs in
        ``processor_config.json`` and declares a custom processor class. This
        method reads those configs directly and caps only the video token budget.

        参数：
            pretrained_model_name_or_path: 模型目录。

        返回：
            Glm5NextProcessor 实例（image/video 处理器 + 分词器）。
        """
        from transformers import AutoTokenizer

        model_path = pretrained_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)

        def _cap_cfg(cfg: dict, *, is_video: bool) -> dict:
            # Video keeps a serving token cap (the checkpoint's 240k-token
            # budget would starve the KV cache at startup profiling); images
            # follow the checkpoint budget verbatim so preprocessing matches
            # the HF reference exactly.
            # 视频保留服务端 token 上限（checkpoint 的 24 万 token 预算
            # 会让启动 profiling 榨干 KV cache）；图像严格按 checkpoint
            # 预算，保证预处理与 HF 参考实现完全一致。
            # 语法点：*, is_video 为 keyword-only 参数的内部函数。
            if is_video and cfg.get("max_image_tokens") is not None:
                cfg["max_image_tokens"] = min(cfg["max_image_tokens"], _MAX_VIDEO_TOKENS)
            return cfg

        # 图像配置来自 preprocessor_config.json（get_image_processor_config）。
        ip_cfg = _cap_cfg(dict(get_image_processor_config(model_path)), is_video=False)
        image_processor = Glm5NextImageProcessor(**{k: v for k, v in ip_cfg.items() if k != "image_processor_type"})

        # 视频配置嵌在 processor_config.json 的 video_processor 键下。
        with open(os.path.join(model_path, "processor_config.json")) as f:
            vp_cfg = _cap_cfg(dict(json.load(f)["video_processor"]), is_video=True)
        video_processor = Glm5NextVideoProcessor(**{k: v for k, v in vp_cfg.items() if k != "video_processor_type"})

        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
        )

    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        videos: VideoInput | None = None,
        **kwargs: Unpack[Glm5NextProcessorKwargs],
    ) -> BatchFeature:
        """组合调用：图像/视频特征提取 + 文本分词。

        参数：
            images: 图像输入（可选）。
            text: prompt 文本（单条或列表）。
            videos: 视频输入（可选）。
            **kwargs: 分处理器参数（Unpack 类型约束）。

        返回：
            BatchFeature：input_ids + pixel_values(+image_grid_thw)
            + pixel_values_videos(+video_grid_thw)。
        """
        # _merge_kwargs：按 Kwargs TypedDict 把调用参数分发到
        # text/images/videos 三个子段。
        output_kwargs = self._merge_kwargs(
            Glm5NextProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        # 步骤1: 图像特征。
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
        else:
            image_inputs = {}

        # 步骤2: 视频特征（弹出 video_metadata，调用方未显式要求时不返回）。
        if videos is not None:
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            if "return_metadata" not in kwargs:
                videos_inputs.pop("video_metadata")
        else:
            videos_inputs = {}

        # 步骤3: 文本统一为列表。
        if not isinstance(text, list):
            text = [text]

        # Prompt updates expand the unchanged image/video markers after this call.
        # prompt-update 机制会在本次调用之后展开未变化的图像/视频占位符。
        # 弹出 return_tensors/return_mm_token_type_ids（不属于分词参数）。
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        # 文本原样分词（不做占位符展开）。
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        # 可选：生成 mm_token_type_ids（图像位置标 1），供下游区分多模态
        # token（语法点：np.array(input_ids) 向量化比较 + 布尔赋值）。
        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()
        return BatchFeature(
            data={**text_inputs, **image_inputs, **videos_inputs},
            tensor_type=return_tensors,
        )

    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        """估算多模态 token 数（调度器配额用，不真正跑预处理）。

        参数：
            image_sizes: [(h, w), ...] 图像尺寸列表。
            video_sizes: [(t, h, w), ...] 视频尺寸列表。
            **kwargs: 处理器参数覆盖。

        返回：
            MultiModalData：num_image_tokens/num_image_patches/
            num_video_tokens 列表。
        """
        vision_data = {}
        if image_sizes is not None:
            # 图像：get_number_of_image_patches 逐个估算，token 数 = patch/merge^2。
            images_kwargs = Glm5NextProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)
            merge_size = images_kwargs.get("merge_size", None) or self.image_processor.merge_size

            num_image_patches = [
                self.image_processor.get_number_of_image_patches(*image_size, images_kwargs)
                for image_size in image_sizes
            ]
            num_image_tokens = [(n // merge_size**2) for n in num_image_patches]
            vision_data.update(
                {
                    "num_image_tokens": num_image_tokens,
                    "num_image_patches": num_image_patches,
                }
            )

        if video_sizes is not None:
            # 视频：get_number_of_video_patches 逐个估算（帧数参与预算）。
            videos_kwargs = Glm5NextProcessorKwargs._defaults.get("videos_kwargs", {})
            videos_kwargs.update(kwargs)
            num_video_patches = [
                self.video_processor.get_number_of_video_patches(*video_size, videos_kwargs)
                for video_size in video_sizes
            ]
            num_video_tokens = [(n // merge_size**2) for n in num_video_patches]
            vision_data["num_video_tokens"] = num_video_tokens

        return MultiModalData(**vision_data)

    def post_process_image_text_to_text(
        self,
        generated_outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
        **kwargs,
    ):
        """图像-文本到文本任务的输出后处理：批量解码为文本。"""
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )


__all__ = [
    # 模块公开导出清单（from processor import * 时只导出这些）。
    "Glm5NextImageProcessor",
    "Glm5NextVideoProcessor",
    "Glm5NextProcessor",
    "smart_resize",
]
