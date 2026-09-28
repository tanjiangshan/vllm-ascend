# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ============================================================================
# 【模块职责】DeepSeek-V4 视觉变体（DeepSeek-V4-Flash-Vision-Exp）的多模态
# 预处理管线，在 prompt 进入模型前完成:
#   1) 图像变换: PIL 图 -> 动态分辨率 resize（受 vision_max_n_token 预算
#      约束）-> patch 切块 -> bf16 归一化像素;
#   2) 哨兵块（sentinel block）构造: 每个 <｜deepseek_image｜> 占位符展开为
#      一段定长布局的哨兵 token（IMAGE_START/PAD/IMAGE/NEWLINE/IMAGE_END），
#      其中只有 type==IMAGE 的位置会注入视觉嵌入，其余哨兵用可学习嵌入;
#   3) prompt 替换与对齐: 哨兵块插入位置需满足压缩对齐（COMPRESS_PAD_TO=4）
#      及 N 布局（两行交错）要求。
# 与官方参考实现的差异: 官方用词表外 id（vocab_size+type），这里借用 5 个
# 连续的保留特殊 token（<|place_holder_mm_span_0431|>.._0435|>），使 id
# 保持在词表内，从而兼容 stock 校验/logprob/解码。
# ============================================================================
"""Multimodal preprocessing for the DeepSeek-V4 vision variants
(DeepSeek-V4-Flash-Vision-Exp).

The image transform and sentinel-block construction are ported from the
official repository's ``image_processor.py`` so that token counts bit-match
the reference. Each ``<｜deepseek_image｜>`` placeholder in the prompt expands
to a variable-length block of sentinel tokens; only positions with
``type == IMAGE`` receive vision embeddings, the other sentinels are looked
up from learned embedding vectors in the model.

Unlike the reference (which uses out-of-vocab ids ``vocab_size + type``),
the sentinel block borrows five consecutive reserved tokenizer tokens
(``<|place_holder_mm_span_0431|>`` .. ``_0435|>``): they are special tokens
the tokenizer never emits from plain text, so the ids stay in-vocabulary and
work with stock token validation, logprobs and detokenization. The ids are
pure markers — every sentinel position's embedding is overwritten with
vision/sentinel vectors before the decoder layers see it, exactly like the
reference's out-of-vocab scheme.
"""

import copy
import math
import threading
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import BatchFeature
from vllm.config.multimodal import BaseDummyOptions, ImageDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.processing.processor import (
    MultiModalPromptUpdates,
    PlaceholderFeaturesInfo,
)
from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config

# 五种哨兵角色: 块首 / 对齐填充 / 图像内容 / 换行 / 块尾。
# range(5) 依次产出 0..4，即 IMAGE_START=0, IMAGE_PAD=1, IMAGE=2,
# IMAGE_NEW_LINE=3, IMAGE_END=4 —— 顺序即哨兵 id 相对基址的偏移。
IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)
# 压缩对齐模数: 图像块的起始位置需对齐到 4 的倍数（压缩组不跨界）。
COMPRESS_PAD_TO = 4

# prompt 中的图像占位符字符串（一个占位符对应一张图像）。
IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"

# Sentinel roles borrow five consecutive ``<|place_holder_mm_span_XXXX|>``
# tokens (reserved special tokens, never emitted from plain text). The order
# must match IMAGE_START..IMAGE_END above so that
# ``id == IMAGE_SENTINEL_BASE_ID + type``.
# 【中文】哨兵基址 id: 五个保留 token 在词表中连续，且顺序与
# IMAGE_START..IMAGE_END 枚举一致，从而哨兵 id = 基址 + 角色编号。
IMAGE_SENTINEL_BASE_ID = 129257
IMAGE_SENTINEL_TOKEN_NAMES = (
    "<|place_holder_mm_span_0431|>",  # IMAGE_START
    "<|place_holder_mm_span_0432|>",  # IMAGE_PAD
    "<|place_holder_mm_span_0433|>",  # IMAGE
    "<|place_holder_mm_span_0434|>",  # IMAGE_NEW_LINE
    "<|place_holder_mm_span_0435|>",  # IMAGE_END
)

# Fast tokenizers temporarily mutate truncation and padding state during a
# call. Multimodal preprocessing runs in a thread pool while the renderer can
# use the same tokenizer concurrently, so keep an independent tokenizer
# backend per preprocessing thread.
# 【中文】线程安全: 快速分词器在调用期间会临时修改截断/填充状态，而多模态
# 预处理在线程池中运行、渲染线程可能并发使用同一分词器，因此用
# threading.local() 为每个预处理线程保存一份独立的分词器副本
# （thread-local 存储: 同一变量在不同线程中互不可见）。
_TOKENIZER_THREAD_LOCAL = threading.local()


def _get_thread_local_tokenizer(tokenizer):
    """返回当前线程专属的分词器副本（按源分词器身份缓存）。

    原理: 首次调用时 deepcopy 一份分词器存入线程本地存储；之后若源
    分词器没变（用 id() 判断对象身份）则直接复用副本，避免每条请求
    都深拷贝。

    Args:
        tokenizer: 源分词器（可能被其他线程共享）。
    Returns:
        当前线程私有的分词器副本。
    """
    cached = getattr(_TOKENIZER_THREAD_LOCAL, "tokenizer", None)
    source_id = getattr(_TOKENIZER_THREAD_LOCAL, "source_id", None)
    # 副本不存在或源分词器已更换（id 变化）时重新深拷贝。
    if cached is None or source_id != id(tokenizer):
        cached = copy.deepcopy(tokenizer)
        _TOKENIZER_THREAD_LOCAL.tokenizer = cached
        _TOKENIZER_THREAD_LOCAL.source_id = id(tokenizer)
    return cached


def image_sentinel_mask(token_ids: torch.Tensor) -> torch.Tensor:
    """Boolean mask for image-block sentinel positions (in-vocab ids)."""
    # 【中文】哨兵位掩码: id 落在 [BASE, BASE+5) 区间内的位置为 True。
    # 两次比较 + & 按位与，纯张量运算可在 NPU 上无同步执行。
    return (token_ids >= IMAGE_SENTINEL_BASE_ID) & (
        token_ids < IMAGE_SENTINEL_BASE_ID + len(IMAGE_SENTINEL_TOKEN_NAMES)
    )


def validate_image_sentinel_ids(tokenizer) -> None:
    """Check the borrowed sentinel ids against the tokenizer."""
    # 【中文】启动期校验: 逐一确认 5 个保留 token 在分词器中的实际 id
    # 等于 基址+序号。若用户换了 tokenizer（id 不连续），在此快速失败。
    for i, name in enumerate(IMAGE_SENTINEL_TOKEN_NAMES):
        token_id = tokenizer.convert_tokens_to_ids(name)
        if token_id != IMAGE_SENTINEL_BASE_ID + i:
            raise ValueError(
                f"Image sentinel token {name!r} has id {token_id}, expected "
                f"{IMAGE_SENTINEL_BASE_ID + i}; the DeepSeek-V4 vision "
                "sentinel block requires these consecutive reserved ids."
            )


def grid_tokens(best_height, best_width, patch_size, downsample_ratio):
    """Number of LLM tokens the aligner grid occupies (N-layout, including
    row/align padding)."""
    # 【中文】哨兵块 token 计数（N 布局，含对齐填充）。
    # 步骤1: LLM 网格 = ViT 网格向上取整除以 downsample_ratio。
    n_llm_h = math.ceil((best_height // patch_size) / downsample_ratio)
    n_llm_w = math.ceil((best_width // patch_size) / downsample_ratio)
    # 步骤2: 每行 n_llm_w 个 IMAGE + 1 个 NEWLINE；外加首尾 START/END 2 个。
    num_tokens = n_llm_h * (n_llm_w + 1) + 2
    # 步骤3: 行数为奇数时补一整行 PAD（N 布局两行交错要求偶数行）。
    if n_llm_h % 2 == 1:
        num_tokens += n_llm_w + 1
    # 步骤4: 交错排列的尾部对齐填充（每对行需要偶数长度，不足补 2）。
    # 运算优先级: // 与 * 与 % 从左到右结合，即 ((rows//2)*(row_len))%2*2。
    num_tokens += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
    return n_llm_h, n_llm_w, num_tokens


def solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    """在 token 预算内求最大可用分辨率（整数格点解）。

    原理: 以宽为变量解一元二次（由 num_tokens = h*(w+1)+2 且 h/w=r 推出），
    得到宽/高的浮数上限 max_w_float/max_h_float，再按宽/高过小的退化分支
    分别处理，最后按缩放系数 beta 反推 patch 对齐的像素尺寸。

    Returns:
        (n_llm_h, n_llm_w, best_height, best_width, num_tokens)。
    """
    r = height / width
    # 步骤1: 由 token 预算反解最大宽度（浮点解）。
    # 推导: num_tokens ≈ h*w + h + 2, h = w*r => r*w^2 + r*w + 2 <= N，
    # 解二次方程得 w <= sqrt((N-2)/r + 0.25) - 0.5。
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    if max_w_float < 1.0:
        # 分支A: 极窄图（宽不足 1 格）——宽钉在 1，高按预算直接除出来，
        # 若高为奇数再减 1（保证偶数行）。
        max_w = 1
        max_h = (max_n_token - 2) // (max_w + 1)
        if max_h % 2 == 1:
            max_h -= 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    elif max_h_float < 2.0:
        # 分支B: 极扁图（高不足 2 格）——高钉在 2，宽按预算反推。
        max_h = 2
        max_w = ((max_n_token - 2) // max_h) - 1
        assert max_w > 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    else:
        # 分支C: 常规情形——向下取整得到整数格点；高为奇数时减 1。
        max_w = math.floor(max_w_float)
        max_h = math.floor(max_h_float)
        if max_h % 2 == 1:
            max_h -= 1
        # 缩放系数 beta 取宽/高两个约束中更紧的（不放大原图）。
        beta = min(
            max_w * patch_size * downsample_ratio / width,
            max_h * patch_size * downsample_ratio / height,
        )
        # 像素尺寸对齐到 patch_size 的整数倍（避免切割不完整 patch）。
        best_width = math.floor(width * beta / patch_size) * patch_size
        best_height = math.floor(height * beta / patch_size) * patch_size
    # 最后按新尺寸重新精确计数 token。
    n_llm_h, n_llm_w, num_tokens = grid_tokens(best_height, best_width, patch_size, downsample_ratio)
    return n_llm_h, n_llm_w, best_height, best_width, num_tokens


def safe_resize(height, width, best_height, best_width, patch_size, downsample_ratio, max_n_token):
    """迭代收缩分辨率直到哨兵块 token 数落入预算。

    原理: solve_resize_ratio 的近似解可能仍超预算（取整误差/退化分支），
    这里逐次把预算减 1 重解，直到 num_tokens <= max_n_token，保证结果
    一定满足约束（迭代上界很小，代价可忽略）。

    Returns:
        (n_llm_h, n_llm_w, best_height, best_width)。
    """
    # 预留 COMPRESS_PAD_TO-1 个 token 给压缩对齐填充。
    max_n_token -= COMPRESS_PAD_TO - 1
    n_llm_h, n_llm_w, num_tokens = grid_tokens(best_height, best_width, patch_size, downsample_ratio)
    budget = max_n_token
    # 循环收缩预算直到满足约束。
    while num_tokens > max_n_token:
        n_llm_h, n_llm_w, best_height, best_width, num_tokens = solve_resize_ratio(
            height, width, patch_size, downsample_ratio, budget
        )
        budget -= 1
    return n_llm_h, n_llm_w, best_height, best_width


def load_image(
    image: Image.Image,
    *,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
    min_pixels: int,
    max_wh_ratio: float | None,
):
    """Transform one PIL image into ViT patches.

    Same math as the reference ``load_image``, except the image is already
    decoded (vLLM supplies PIL images instead of a record dict).
    """
    # 【中文】单图变换: PIL -> 分辨率约束 -> resize/pad -> 归一化 -> patch 化。
    p = patch_size
    # 步骤1: 统一转 RGB（丢弃 alpha 通道）。
    image = image.convert("RGB")
    width, height = image.size
    # 步骤2: 宽高比上限——超宽图直接裁剪宽度（防止极端宽的截图爆 token）。
    if max_wh_ratio is not None and width > height * max_wh_ratio:
        width = height * max_wh_ratio
    # 步骤3: 最小像素下限——过小的图等比放大到 min_pixels（保信息量）。
    if 0 < width * height < min_pixels:
        ratio = (min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    # 步骤4: 像素尺寸向上对齐到 patch 的整数倍（初值）。
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    # 步骤5: 在 token 预算内迭代收缩到合法尺寸。
    n_llm_h, n_llm_w, best_height, best_width = safe_resize(
        height, width, best_height, best_width, p, downsample_ratio, max_n_token
    )
    # ViT 网格 = 最终像素尺寸 / patch_size。
    n_vit_h, n_vit_w = best_height // p, best_width // p
    # 步骤6: 超宽图直接 resize（丢失比例）；普通图保持纵横比、用灰色
    # (127,127,127) pad 到目标尺寸。
    if max_wh_ratio is not None and image.width >= max_wh_ratio * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    # 步骤7: PIL -> CHW 张量，/255 到 [0,1]，再标准化到 [-1,1]，转 bf16。
    x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
    x = ((x - 0.5) / 0.5).to(torch.bfloat16)
    # 步骤8: 像素 unshuffling——把 [3, H, W] 切成 n_vit_h*n_vit_w 个
    # [3, p, p] patch。reshape/permute/reshape 三步完成维度重排。
    patches = x.reshape(3, n_vit_h, p, n_vit_w, p).permute(1, 3, 0, 2, 4).reshape(n_vit_h * n_vit_w, 3, p, p)
    return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


def build_image_block(n_llm_h: int, n_llm_w: int, start_pos: int):
    """Builds the N-layout token types (final order) and the aligner-row order
    for IMAGE slots."""
    # 【中文】构造哨兵块:
    # 1) types——最终顺序（N 布局）的哨兵角色序列;
    # 2) perm——把 Aligner 行主序输出重排到 N 布局 IMAGE 槽位的索引。
    # N 布局原理: 哨兵块内部把“两行”交错排列（第 0/2/4... 位置的 token
    # 组成一行，第 1/3/5... 组成相邻行），使空间相邻 token 在序列中也
    # 相邻，利于局部性敏感的注意力/压缩。
    # 压缩对齐填充: 保证块首（去掉 pad 后）的绝对位置是 COMPRESS_PAD_TO
    # 的倍数，使压缩组不跨越图像边界。
    compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
    # 行数补到偶数（N 布局两行一组交错的要求）。
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    # 每行长度 = n_llm_w 个 IMAGE + 1 个 NEWLINE。
    row_len = n_llm_w + 1
    # 尾部对齐填充: 每两行一组的长度需为偶数，否则补 2 个 PAD。
    pad_last = rows // 2 * row_len % 2 * 2
    # 行主序角色序列: 每行 [IMAGE]*n_llm_w + [NEWLINE]，再加 pad_h 行 PAD。
    types = torch.tensor(
        ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_PAD] * (row_len * pad_h),
        dtype=torch.int64,
    )
    # 构造交错排列 order: [rows*row_len] -> [rows//2, 2, row_len] ->
    # transpose(1,2) -> 展平。即把“两行一组”改为“按列交错”。
    order = torch.arange(rows * row_len).view(rows // 2, 2, row_len)
    order = order.transpose(1, 2).reshape(-1)
    # image_idx: 行主序位置 -> 图内 IMAGE 序号（非 IMAGE 槽位为 -1）。
    image_idx = torch.full((rows * row_len,), -1, dtype=torch.int64)
    image_idx.view(rows, row_len)[:n_llm_h, :n_llm_w] = torch.arange(n_llm_h * n_llm_w).view(n_llm_h, n_llm_w)
    # perm = 交错序下的 IMAGE 序号（滤除 -1）——供 vl_model 把 Aligner
    # 的行主序输出重排到哨兵块 IMAGE 槽位。
    perm = image_idx[order]
    perm = perm[perm >= 0]
    # 最终 types = [压缩对齐 pad] + [START] + 交错后的主体 + [尾部 pad] + [END]。
    types = torch.cat(
        [
            torch.full((compress_pad,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_START]),
            types[order],
            torch.full((pad_last,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_END]),
        ]
    )
    return types, perm


def build_image_block_pad_free(n_llm_h: int, n_llm_w: int):
    """``build_image_block`` without the position-dependent compressor pad.

    ``start_pos`` is chosen so that ``compress_pad == 0``; the pad is instead
    prepended when the block is spliced into the final prompt, where its
    position is known.
    """
    # 【中文】“无 pad”版本: 预处理阶段图像块的最终位置未知，故先取
    # start_pos=COMPRESS_PAD_TO-1 使 compress_pad==0；真正的对齐 pad 在
    # _apply_prompt_updates 拼接 prompt 时（位置已知）再前插。
    return build_image_block(n_llm_h, n_llm_w, COMPRESS_PAD_TO - 1)


class DeepseekV4VLImageProcessor:
    """Per-image transform (the PIL-input equivalent of the reference
    ``load_image``)."""

    # 【中文】单图处理器: 仅是 load_image 的配置化封装（保存 5 个超参），
    # 供 DeepseekV4VLProcessor 逐图调用。

    def __init__(self, config: DeepseekV4Config) -> None:
        """初始化。Args: config: 从中读取 patch 大小/降采样比/token 预算等。"""
        super().__init__()
        self.patch_size = config.vision_patch_size
        self.downsample_ratio = config.vision_downsample_ratio
        self.max_n_token = config.vision_max_n_token
        self.min_pixels = config.vision_min_pixels
        self.max_wh_ratio = config.vision_max_wh_ratio

    def __call__(self, image: Image.Image):
        """可调用对象（实现了 __call__ 的实例可像函数一样使用）。
        Returns: load_image 的输出 (patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w)。"""
        return load_image(
            image,
            patch_size=self.patch_size,
            downsample_ratio=self.downsample_ratio,
            max_n_token=self.max_n_token,
            min_pixels=self.min_pixels,
            max_wh_ratio=self.max_wh_ratio,
        )


class DeepseekV4VLProcessor:
    """Minimal stand-in for the HF processor of DeepSeek-V4 vision models.

    The official repository ships image preprocessing as plain functions in
    ``image_processor.py`` (no ``auto_map`` processor), so this class wraps
    their ports directly and the model loads without ``--trust-remote-code``.

    ``__call__`` returns a ``BatchFeature`` with one entry per image
    (flattened across images):

    - ``patches``: ``(sum(n_vit_h * n_vit_w), 3, p, p)`` bf16 ViT patches.
    - ``vit_grid``: ``(num_images, 2)`` int64 ``[n_vit_h, n_vit_w]``.
    - ``llm_grid``: ``(num_images, 2)`` int64 ``[n_llm_h, n_llm_w]``.
    - ``perm``: concatenated per-image ``(n_llm_h * n_llm_w,)`` int64 index
      selecting aligner outputs into the final N-layout order.
    - ``types``: concatenated per-image pad-free sentinel block types;
      ``block ids = IMAGE_SENTINEL_BASE_ID + types``.
    """

    def __init__(self, config: DeepseekV4Config) -> None:
        super().__init__()
        self.config = config
        self.image_processor = DeepseekV4VLImageProcessor(config)

    def __call__(
        self,
        text: str | None = None,
        images: Sequence[Image.Image] | None = None,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> BatchFeature:
        """处理一批图像: 逐图变换 + 构造 pad-free 哨兵块，拼接为 BatchFeature。

        Args:
            text: prompt 文本（本处理器不使用，仅保持 HF 接口兼容）。
            images: PIL 图像序列。
            return_tensors: 返回张量格式（兼容参数）。
        Returns:
            BatchFeature，字段见类 docstring（patches/vit_grid/llm_grid/perm/types）。
        """
        # 五个结果列表分别累积每张图的输出。
        patches_list = []
        vit_grid = []
        llm_grid = []
        perm_list = []
        types_list = []
        # 逐图: 变换得 patch 与两个网格，再构造 pad-free 哨兵块。
        for image in images or []:
            patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = self.image_processor(image)
            types, perm = build_image_block_pad_free(n_llm_h, n_llm_w)
            patches_list.append(patches)
            vit_grid.append((n_vit_h, n_vit_w))
            llm_grid.append((n_llm_h, n_llm_w))
            perm_list.append(perm)
            types_list.append(types)

        # 无图像: 返回空 BatchFeature。
        if not patches_list:
            return BatchFeature({})

        # 沿第 0 维拼接所有图像的结果; 网格转成 [num_images, 2] int64 张量。
        return BatchFeature(
            {
                "patches": torch.cat(patches_list),
                "vit_grid": torch.tensor(vit_grid, dtype=torch.int64),
                "llm_grid": torch.tensor(llm_grid, dtype=torch.int64),
                "perm": torch.cat(perm_list),
                "types": torch.cat(types_list),
            }
        )


class DeepseekV4VLProcessingInfo(BaseProcessingInfo):
    """处理信息类: 向 vLLM 多模态框架描述本模型的处理器与输入约束。

    继承上游 vLLM 的 BaseProcessingInfo，框架据此查询 HF 配置/分词器、
    计算每项多模态数据的最大 token 数（用于调度器预算估计）等。
    """

    def get_hf_config(self) -> DeepseekV4Config:
        """返回强类型化的 HF 配置（框架按 DeepseekV4Config 解析）。"""
        return self.ctx.get_hf_config(DeepseekV4Config)

    def get_hf_processor(self, **kwargs: object) -> DeepseekV4VLProcessor:
        """构造本模型的处理器实例。

        Raises:
            ValueError: 传入了不支持的处理器参数。
        """
        if kwargs:
            raise ValueError(f"Unexpected processor kwargs: {sorted(kwargs)}")
        return DeepseekV4VLProcessor(self.get_hf_config())

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        """返回各模态每条 prompt 的数量上限。None 表示不限制（图像数
        仅受总 token 预算约束）。"""
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        """单张图像占用的最大 token 数（调度器估算 batch 预算用）。

        Args:
            seq_len: 序列长度（本模型不依赖）。
            mm_counts: 各模态数量。
        Returns:
            {"image": 最大哨兵块长度}。
        """
        # ``safe_resize`` reserves COMPRESS_PAD_TO - 1 tokens of
        # vision_max_n_token for the compressor-alignment pad, so the full
        # sentinel block is bounded by vision_max_n_token; the margin is kept
        # in case that reservation changes.
        # 【中文】safe_resize 已为压缩对齐 pad 预留 COMPRESS_PAD_TO-1 个
        # token，因此完整哨兵块被 vision_max_n_token 约束；这里再保留裕量
        # 以防该预留策略变化。
        return {
            "image": self.get_hf_config().vision_max_n_token + COMPRESS_PAD_TO - 1,
        }

    def get_image_placeholder_token_id(self) -> int:
        """查询 <｜deepseek_image｜> 占位符的 token id（找不到则报错）。"""
        token_id = self.get_tokenizer().convert_tokens_to_ids(IMAGE_PLACEHOLDER)
        if token_id is None:
            raise ValueError(f"Token not found in tokenizer: {IMAGE_PLACEHOLDER}")
        return token_id

    def get_image_size_with_most_features(self) -> ImageSize:
        """返回“特征最多”的哑图像尺寸（用于 profile run / graph capture）。

        原理: 相同 token 预算下正方形图像的 patch 数（面积）最大; 直接
        从预算反解该正方形边长，避免构造巨大的哑图浪费内存。
        """
        hf_config = self.get_hf_config()
        patch_size = hf_config.vision_patch_size
        downsample_ratio = hf_config.vision_downsample_ratio
        # A square maximizes the ViT patch count (area) within the token
        # budget; solve the budget-derived size directly to keep the dummy
        # image small.
        # 【中文】见上——正方形最大化 patch 数，直接解预算约束。
        budget = hf_config.vision_max_n_token - (COMPRESS_PAD_TO - 1)
        side = budget * patch_size * downsample_ratio
        _, _, best_h, best_w, _ = solve_resize_ratio(side, side, patch_size, downsample_ratio, budget)
        return ImageSize(width=best_w, height=best_h)


class DeepseekV4VLDummyInputsBuilder(BaseDummyInputsBuilder[DeepseekV4VLProcessingInfo]):
    """哑输入构造器: 为 profile run / ACL Graph Capture 生成代表性输入。

    泛型语法点: BaseDummyInputsBuilder[Info] 表示带类型参数的泛型基类，
    指定本构造器服务的处理信息类，便于类型检查器推导 self.info 的类型。
    """

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        """返回哑 prompt 文本: 按图像数量重复占位符。
        mm_counts.get("image", 0): 字典取值，键不存在时返回默认 0。"""
        return IMAGE_PLACEHOLDER * mm_counts.get("image", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        """返回哑多模态数据: 尺寸取“特征最多”的图像，按数量复制。

        Args:
            seq_len: 目标序列长度。
            mm_counts: 各模态数量。
            mm_options: 覆盖选项（如指定宽高）。
        Returns:
            {"image": [PIL 图像列表]}。
        """
        size = self.info.get_image_size_with_most_features()
        return {
            "image": self._get_dummy_images(
                width=size.width,
                height=size.height,
                num_images=mm_counts.get("image", 0),
                # cast(目标类型, 值): 类型断言（告诉类型检查器“视为该类型”），
                # 运行时不做任何转换。
                overrides=cast(ImageDummyOptions | None, mm_options.get("image")),
            ),
        }


class DeepseekV4VLMultiModalProcessor(BaseMultiModalProcessor[DeepseekV4VLProcessingInfo]):
    """DeepSeek-V4 视觉模型的多模态处理器（继承 vLLM BaseMultiModalProcessor）。

    职责: 调用本地图像变换 + 分词、声明多模态字段拆分规则、生成占位符
    替换计划（占位符 -> 哨兵块），并在 v0.27 框架上补充“按位置注入压缩
    对齐 pad”的逻辑。
    """

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """Combine the local image transform with v0.27 tokenization."""
        # 【中文】组合“本地图像变换”与“v0.27 分词”:
        # 上游基类期望这里调用 HF processor；本模型没有 HF processor，
        # 因此直接调用 DeepseekV4VLProcessor（图像变换）并用线程本地的
        # 分词器副本对 prompt 分词，把 input_ids 合并进 BatchFeature。
        processor = self.info.get_hf_processor(**mm_kwargs)
        processed = processor(
            text=prompt,
            images=cast(Sequence[Image.Image] | None, mm_data.get("images")),
            return_tensors="pt",
        )
        # 使用线程本地副本分词（见 _get_thread_local_tokenizer 的线程安全说明）。
        tokenizer = _get_thread_local_tokenizer(self.info.get_tokenizer())
        tokenizer_outputs = tokenizer(
            prompt,
            return_tensors="pt",
            **tok_kwargs,
        )
        processed["input_ids"] = tokenizer_outputs["input_ids"]
        return processed

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        """是否由 HF processor 自行完成占位符替换。返回 False——本地
        processor 只做图像变换，占位符替换由 vLLM 在分词后统一执行。"""
        # del 显式丢弃未用参数（本方法的判定与参数无关，恒为 False）。
        del prompt_text, mm_items, hf_processor_mm_kwargs, tokenization_kwargs
        # The local processor transforms images only; vLLM performs the
        # placeholder replacement after tokenization.
        # 【中文】本地 processor 仅变换图像; 占位符替换在分词后由 vLLM 执行。
        return False

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """声明 BatchFeature 中各多模态字段的“拆分/分批”规则。

        原理: vLLM 需要知道每个字段如何按图像分组（flat_from_sizes 按
        sizes 序列切段; batched 每图一行）。types/网格类小张量可留在 CPU
        （keep_on_cpu=True），避免无谓的 H2D 拷贝。
        """
        vit_grid = hf_inputs.get("vit_grid")
        llm_grid = hf_inputs.get("llm_grid")

        if vit_grid is None or llm_grid is None:
            # 无图像: 各 sizes 用空张量占位（长度 0）。
            empty = torch.empty(0, dtype=torch.long)
            patch_sizes = perm_sizes = types_sizes = empty
        else:
            # 每张图的 patch 数 = n_vit_h*n_vit_w（prod(-1) 沿最后一维求积）。
            patch_sizes = vit_grid.prod(-1)
            # 每张图的 perm 长度 = n_llm_h*n_llm_w。
            perm_sizes = llm_grid.prod(-1)
            n_llm_h, n_llm_w = llm_grid[:, 0], llm_grid[:, 1]
            # Pad-free block length; same formula as ``grid_tokens`` given
            # the LLM grid.
            # 【中文】pad-free 哨兵块长度（与 grid_tokens 公式一致，
            # 只是 compress_pad 恒为 0）。
            types_sizes = (
                n_llm_h * (n_llm_w + 1) + 2 + (n_llm_h % 2) * (n_llm_w + 1) + (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
            )

        # 字段规则:
        # - patches: 按 patch_sizes 切段（flat），随 batch 到设备;
        # - vit_grid/llm_grid: 每图一行（batched），保留在 CPU;
        # - perm: 按 perm_sizes 切段;
        # - types: 按 types_sizes 切段，保留在 CPU（拼接 prompt 时才用）。
        return {
            "patches": MultiModalFieldConfig.flat_from_sizes("image", patch_sizes),
            "vit_grid": MultiModalFieldConfig.batched("image", keep_on_cpu=True),
            "llm_grid": MultiModalFieldConfig.batched("image", keep_on_cpu=True),
            "perm": MultiModalFieldConfig.flat_from_sizes("image", perm_sizes),
            "types": MultiModalFieldConfig.flat_from_sizes("image", types_sizes, keep_on_cpu=True),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """生成占位符替换计划: 占位符 token -> 该图的哨兵块 token 序列。

        原理: 每个占位符出现处将替换为 (BASE+types) 的 id 序列，其中
        type==IMAGE 的位置（select_token_id 指定的锚点 id）后续由视觉
        嵌入覆盖。
        """
        # 占位符 token id（<｜deepseek_image｜>）。
        image_token_id = self.info.get_image_placeholder_token_id()
        # 校验哨兵 id 连续性（快速失败）。
        validate_image_sentinel_ids(self.info.get_tokenizer())
        # 嵌入锚点 id: IMAGE 型哨兵（会被视觉嵌入覆盖的位置标识）。
        image_embed_id = IMAGE_SENTINEL_BASE_ID + IMAGE

        def get_image_replacement(item_idx: int) -> PromptUpdateDetails:
            """闭包: 返回第 item_idx 张图的哨兵块替换详情。

            语法点: 闭包捕获外层的 out_mm_kwargs/image_embed_id;
            PromptUpdateDetails.select_token_id 指定“该 id 的位置将被
            多模态嵌入替换”。
            """
            # 从预处理输出中取出该图的 types 张量，映射为实际 token id。
            types: torch.Tensor = out_mm_kwargs["image"][item_idx]["types"].data
            full = (IMAGE_SENTINEL_BASE_ID + types).tolist()
            return PromptUpdateDetails.select_token_id(full, image_embed_id)

        # PromptReplacement: 声明“target 序列 -> replacement”的替换规则。
        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_image_replacement,
            ),
        ]

    def _apply_prompt_updates(
        self,
        token_ids: list[int],
        mm_prompt_updates: MultiModalPromptUpdates,
    ) -> tuple[
        list[int],
        Mapping[str, list[PlaceholderFeaturesInfo]],
    ]:
        """Apply v0.27 prompt updates and inject position-dependent padding.

        vLLM main plans prompt replacements before rendering and exposes that
        plan to model processors. v0.27 does not expose that private API, so
        first use its supported base implementation, then prepend each image
        block's compressor-alignment padding and rebuild placeholder offsets.
        """
        # 【中文】两步走:
        # 1) 调用 v0.27 基类完成占位符替换（pad-free 哨兵块拼接进 token 序列）;
        # 2) 因为此时各图像块的位置已知，计算并前插“压缩对齐 pad”，
        #    同时修正所有后续占位符的偏移（start_idx）与嵌入掩码（is_embed）。
        new_token_ids, base_placeholders = super()._apply_prompt_updates(
            token_ids,
            mm_prompt_updates,
        )
        # 字典推导式: 为每个模态初始化空的结果列表。
        placeholders: dict[str, list[PlaceholderFeaturesInfo]] = {modality: [] for modality in base_placeholders}
        # 对齐 pad 用的哨兵 id（IMAGE_PAD 型）。
        pad_id = IMAGE_SENTINEL_BASE_ID + IMAGE_PAD
        # 生成器 + sorted: 把所有占位符按 start_idx 排序，保证从左到右
        # 插入时偏移量可以累加修正（元组比较: 先比 start_idx，再比 modality）。
        ordered = sorted(
            (
                placeholder.start_idx,
                modality,
                placeholder,
            )
            for modality, items in base_placeholders.items()
            for placeholder in items
        )
        # 已累计插入的 pad 数（其后所有占位符的起始位置都要加此偏移）。
        inserted = 0
        for _, modality, placeholder in ordered:
            # 修正后的起始位置 = 原位置 + 之前累计插入量。
            start_idx = placeholder.start_idx + inserted
            tokens = list(placeholder.tokens)
            is_embed = placeholder.is_embed
            if modality == "image":
                # 步骤1: 计算对齐 pad 数——使哨兵块主体（去掉 pad 后）
                # 的起始绝对位置为 COMPRESS_PAD_TO 的倍数。
                compress_pad = COMPRESS_PAD_TO - 1 - start_idx % COMPRESS_PAD_TO
                # 步骤2: 列表切片插入 [start:start] = ... 在 start 处
                # 原地前插 compress_pad 个 pad id。
                new_token_ids[start_idx:start_idx] = [pad_id] * compress_pad
                # 步骤3: 占位符 token 序列同步前插 pad（记录信息保持一致）。
                tokens = [pad_id] * compress_pad + tokens
                # 步骤4: 嵌入掩码: pad 位补 False（不注入视觉嵌入）;
                # 若原掩码缺失则默认全 True。三元条件表达式 + torch.cat。
                original_mask = (
                    is_embed
                    if is_embed is not None
                    else torch.ones(
                        len(placeholder.tokens),
                        dtype=torch.bool,
                    )
                )
                is_embed = torch.cat(
                    [
                        torch.zeros(compress_pad, dtype=torch.bool),
                        original_mask,
                    ]
                )
                # 步骤5: 累计插入量。
                inserted += compress_pad

            # 重建修正后的占位符信息记录。
            placeholders[modality].append(
                PlaceholderFeaturesInfo(
                    modality=modality,
                    item_idx=placeholder.item_idx,
                    start_idx=start_idx,
                    tokens=tokens,
                    is_embed=is_embed,
                )
            )

        return new_token_ids, placeholders
