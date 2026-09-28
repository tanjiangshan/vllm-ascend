# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pooled-cache physical views for GLM-Next on Model Runner V1.

GLM-Next 池化缓存的物理视图构建（v1 模型运行时）。

原理：GLM-Next 把"压缩索引器缓存"与"KPool 尾部缓存"打包进同一个物理
小页槽（small slot）——两者共用一块显存但各占一半（见 cache_config.py 的
张量分配）。模型运行时拿到原始（raw）缓存张量后，需要按各 spec 的形状
把它"切"成注意力后端期望的视图（view）。本模块就是这层"物理布局 ->
后端视图"的适配器：全部用 torch.as_strided / view 构造零拷贝视图。
"""

from collections.abc import Callable

import torch
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec

from vllm_ascend.core.kv_cache_interface import (
    AscendIndexerKPoolTailSpec,
    AscendMLAAttentionSpec,
    get_kv_cache_compression_ratio,
    get_storage_block_size,
)


def _row_major_strides(shape: tuple[int, ...]) -> list[int]:
    """计算给定形状按行主序（C 连续）存放时的各维步长（元素数）。

    参数：
        shape: 张量形状，如 [num_blocks, block_size, head, head_dim]。

    返回：
        list[int]：每维的步长。最后一维为 1，其余为右侧维度大小之积。

    原理：行主序即 C 语言的内存布局，stride[i] = prod(shape[i+1:])。
    """
    strides = [1] * len(shape)
    for dim_idx in range(len(shape) - 2, -1, -1):
        strides[dim_idx] = strides[dim_idx + 1] * shape[dim_idx + 1]
    return strides


def _view_kpool_tail_cache(
    layer_name: str,
    kv_cache_spec: AscendIndexerKPoolTailSpec,
    raw_cache: torch.Tensor,
    num_blocks: int,
) -> list[torch.Tensor]:
    """把 KPool 尾部缓存的原始张量切成 [blocks, 2, capacity, head_dim] 视图。

    参数：
        layer_name: 层名（用于报错信息）。
        kv_cache_spec: 尾部缓存规格（含 block_size/head_size/dtype 等）。
        raw_cache: 模型运行时分配的原始字节张量（uint8 视角）。
        num_blocks: 物理块数。

    返回：
        list[torch.Tensor]：单元素列表，视图形状
        [num_tail_blocks, 2, block_size, head_size]——第 1 维的 2 表示
        K 与 gate 两份存储（原始键 + 门控分数）。

    原理：尾部缓存与压缩索引器缓存共享一个"小页槽"（small slot），
    本函数取该槽的后半段（typed_slot.numel() - num_tail_blocks*tail_block_el
    之后的部分），前半段留给压缩索引器缓存（见 _view_compressed_indexer_cache）。
    """
    if not isinstance(raw_cache, torch.Tensor):
        raise ValueError(f"KPool tail cache for {layer_name} must use one raw tensor.")
    # 步骤1: 字节张量重解释为 spec 声明的 dtype（FP32）。
    typed_slot = raw_cache.view(kv_cache_spec.dtype)
    tail_block_el = kv_cache_spec.unpadded_page_size_bytes // get_dtype_size(kv_cache_spec.dtype)
    num_tail_blocks = num_blocks
    # 步骤2: 容量校验——尾部缓存最多只能占小页槽的一半。
    if num_tail_blocks * tail_block_el * 2 > typed_slot.numel():
        raise ValueError(
            f"KPool tail cache for {layer_name} exceeds half the small slot: "
            f"packed={num_tail_blocks * tail_block_el} elements, "
            f"slot={typed_slot.numel()}."
        )
    # 步骤3: 取槽的后半段并重排为 [块数, K/gate=2, 环容量, head_dim]。
    return [
        typed_slot[typed_slot.numel() - num_tail_blocks * tail_block_el :].view(
            num_tail_blocks,
            2,
            kv_cache_spec.block_size,
            kv_cache_spec.head_size,
        )
    ]


def _view_compressed_indexer_cache(
    layer_name: str,
    kv_cache_spec: AscendMLAAttentionSpec,
    raw_cache: torch.Tensor | tuple[torch.Tensor, ...],
    attn_backend: AttentionBackend,
    kernel_block_size: int,
) -> tuple[torch.Tensor]:
    """把压缩索引器缓存的原始张量切成注意力后端期望形状的视图。

    参数：
        layer_name: 层名。
        kv_cache_spec: MLA 规格但携带 tokens_per_state>1（KPool 压缩比）。
        raw_cache: 原始张量（或单元素元组）。
        attn_backend: 注意力后端（提供 get_kv_cache_shape）。
        kernel_block_size: 主 MLA 内核块大小（token 粒度）。

    返回：
        tuple[torch.Tensor]：单元素元组，内含按行主序步长构造的
        as_strided 视图，形状由 attn_backend.get_kv_cache_shape 决定。

    原理：KPool 把每 compress_ratio 个 token 的 K 压成 1 条，因此索引器
    缓存的"内核块大小"= kernel_block_size // compress_ratio（以池为单位）。
    视图占用小页槽的前半段（与尾部缓存共享同一槽）。
    """
    if isinstance(raw_cache, tuple):
        if len(raw_cache) != 1:
            raise ValueError(f"Compressed indexer cache for {layer_name} must be a single tensor.")
        raw_single = raw_cache[0]
    else:
        raw_single = raw_cache
    # 步骤1: 推导压缩后的内核块大小与块数。
    compression_ratio = get_kv_cache_compression_ratio(kv_cache_spec)
    indexer_kernel_block_size = kernel_block_size // compression_ratio
    num_blocks = raw_single.numel() // kv_cache_spec.page_size_bytes
    num_blocks_per_kv_block = get_storage_block_size(kv_cache_spec) // indexer_kernel_block_size
    # 步骤2: 由后端给出目标形状，并计算行主序步长。
    shape = tuple(
        attn_backend.get_kv_cache_shape(
            num_blocks * num_blocks_per_kv_block,
            indexer_kernel_block_size,
            kv_cache_spec.num_kv_heads,
            kv_cache_spec.head_size,
        )
    )
    strides = _row_major_strides(shape)
    typed_slot = raw_single.view(kv_cache_spec.dtype)
    # 步骤3: 容量校验（最多占小页槽的一半）。
    if strides[0] * shape[0] * 2 > typed_slot.numel():
        raise ValueError(
            f"Compressed indexer cache for {layer_name} exceeds half the small slot: "
            f"packed={strides[0] * shape[0]} elements, slot={typed_slot.numel()}."
        )
    # 步骤4: 零拷贝构造视图（torch.as_strided 按 size/stride 重排同一存储）。
    return (torch.as_strided(typed_slot, size=shape, stride=tuple(strides)),)


def _view_nope_main_mla_cache(
    kv_cache_spec: AscendMLAAttentionSpec,
    raw_cache: torch.Tensor,
    attn_backend: AttentionBackend,
    kernel_block_size: int,
) -> list[torch.Tensor]:
    """构建主 MLA 缓存（无 V 分量 / 无 RoPE 的 nope 变体）的视图。

    参数：
        kv_cache_spec: 主 MLA 规格（compression_ratio == 1）。
        raw_cache: 原始字节张量。
        attn_backend: 注意力后端。
        kernel_block_size: 内核块大小（token 数）。

    返回：
        list[torch.Tensor]：[k_cache, rope_cache]。
        - k_cache: 按 MLA 注意力后端期望形状的 as_strided 视图；
        - rope_cache: 最后一维大小为 0 的"空"视图——nope 配置没有 RoPE
          分量，返回空张量占位以保持接口一致。

    原理：GLM-5.Next 的 mla_nope 配置（qk_rope_head_dim=0）下 MLA 不使用
    RoPE 分量，因此 rope 缓存宽度为 0；vLLM 上游接口要求返回 K/V 两个
    视图，这里用 size 最后一维为 0 的 as_strided 满足之。
    """
    num_blocks = raw_cache.numel() // kv_cache_spec.page_size_bytes
    num_blocks_per_kv_block = get_storage_block_size(kv_cache_spec) // kernel_block_size
    shape = tuple(
        attn_backend.get_kv_cache_shape(
            num_blocks * num_blocks_per_kv_block,
            kernel_block_size,
            kv_cache_spec.num_kv_heads,
            kv_cache_spec.head_size,
        )
    )
    strides = _row_major_strides(shape)
    typed_slot = raw_cache.view(kv_cache_spec.dtype)
    k_cache = torch.as_strided(typed_slot, size=shape, stride=tuple(strides))
    # 空的 rope 视图：宽度 0，仅作占位。
    rope_cache = torch.as_strided(
        typed_slot,
        size=(*shape[:-1], 0),
        stride=tuple(strides[:-1]) + (1,),
    )
    return [k_cache, rope_cache]


def view_glm5_next_cache(
    layer_name: str,
    kv_cache_spec: KVCacheSpec,
    raw_cache: torch.Tensor | tuple[torch.Tensor, ...],
    *,
    attn_backend: AttentionBackend,
    kernel_block_size: int,
    num_blocks: int,
    get_kv_cache_dims: Callable[[str, AttentionSpec], tuple[int, int]],
) -> list[torch.Tensor] | tuple[torch.Tensor, ...] | None:
    """Build the pooled-cache views for one GLM-Next layer.

    Returns the view container the runner registers for the layer, or None
    when the spec is not owned by the GLM-Next pooled layout, so the caller
    falls through to its generic reshape paths.

    为一个 GLM-Next 层构建池化缓存视图的总入口（分发器）。

    参数（语法点：* 之后为 keyword-only 参数，调用时必须写关键字）：
        layer_name: 层名。
        kv_cache_spec: 该层的缓存规格。
        raw_cache: 原始缓存张量（或元组）。
        attn_backend: 注意力后端。
        kernel_block_size: 内核块大小。
        num_blocks: 物理块数。
        get_kv_cache_dims: 回调，返回 (k_dim, v_dim)。

    返回：
        该层注册到运行时的视图容器；若规格不属于 GLM-Next 池化布局，
        返回 None，调用方回退到通用 reshape 路径。

    分发逻辑：
        1. AscendIndexerKPoolTailSpec         -> 尾部缓存视图；
        2. MLA + block_stride 索引 + 压缩比>1 -> 压缩索引器缓存视图；
        3. MLA + block_stride 索引 + v_dim==0 -> nope 主 MLA 缓存视图；
        4. 其余                                -> None（走通用路径）。
    """

    if isinstance(kv_cache_spec, AscendIndexerKPoolTailSpec):
        return _view_kpool_tail_cache(layer_name, kv_cache_spec, raw_cache, num_blocks)
    if isinstance(kv_cache_spec, AscendMLAAttentionSpec) and getattr(
        kv_cache_spec, "indexes_kv_by_block_stride", False
    ):
        if get_kv_cache_compression_ratio(kv_cache_spec) > 1:
            return _view_compressed_indexer_cache(layer_name, kv_cache_spec, raw_cache, attn_backend, kernel_block_size)
        _k_dim, v_dim = get_kv_cache_dims(layer_name, kv_cache_spec)
        if v_dim == 0:
            return _view_nope_main_mla_cache(kv_cache_spec, raw_cache, attn_backend, kernel_block_size)
    return None
