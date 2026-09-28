# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache layers and metadata helpers for the GLM-Next pooled indexer.

GLM-Next 池化索引器的 KV cache 层注册与元数据辅助函数。

本模块包含：
  - is_glm5_next_cache_spec / get_kpool_tail_ring_capacity /
    format_indexer_kpool_slot_mapping：工具函数；
  - Glm5NextIndexerCache：压缩索引器 K 缓存层（BF16，kpool:1 压缩）；
  - Glm5NextTailCache：请求私有 FP32 环形缓存（未满池的 K + gate）；
  - KpoolTailManager：尾部缓存的调度器管理器（继承 FullAttentionManager，
    但禁用前缀共享——尾部环是请求私有的瞬态数据）。

NPU 适配点：这两个缓存层实现 vLLM 的 AttentionLayerBase 协议
（get_kv_cache_spec / get_attn_backend），把自定义规格交给
vllm_ascend.core.kv_cache_interface 中的 Ascend spec 类，再由
Ascend 自定义注意力后端（AscendIndexerKPoolBackend 等）消费。
"""

from collections.abc import Sequence
from typing import Any, ClassVar

import torch
from torch import nn
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashList, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.request import Request

from vllm_ascend.core.kv_cache_interface import (
    AscendIndexerKPoolTailSpec,
    AscendMLAAttentionSpec,
)


def is_glm5_next_cache_spec(spec: KVCacheSpec) -> bool:
    """判断一个 KVCacheSpec 是否属于 GLM-Next（model_version 标记）。

    参数：
        spec: 任意缓存规格。

    返回：
        bool：spec.model_version == "glm5_next" 时为 True。
        语法点：getattr(obj, name, default) 安全取属性，缺失时返回默认值。
    """
    return getattr(spec, "model_version", None) == "glm5_next"


def get_kpool_tail_ring_capacity(vllm_config: VllmConfig, compress_ratio: int) -> int:
    """Keep the open pool plus every not-yet-accepted speculative token.

    Target verification writes all draft rows before rejection sampling is
    known.  The first row is the already-sampled token and is committed; the
    remaining ``num_speculative_tokens`` rows are lookahead.  Retaining that
    lookahead in addition to one full pool prevents rejected rows from wrapping
    around and destroying committed history that a replayed boundary token
    still needs.

    计算尾部环形缓存所需的容量：开放池 + 所有尚未验收的投机 token。

    原理：
    - 常规解码：环容量 = compress_ratio（正好容纳一个未满池）。
    - 投机解码（MTP）：目标模型验证阶段会先把所有 draft 行写入缓存，
      拒绝采样（rejection sampling）结果出来前不知道哪些行会被接受。
      第一行是已采样提交的 token，其余 num_speculative_tokens 行是
      前瞻（lookahead）。必须额外保留这些前瞻行，否则被拒绝的行会
      "绕回"环形缓冲区头部，破坏重放边界 token 仍需要的已提交历史。

    参数：
        vllm_config: vLLM 全局配置（读取 speculative_config）。
        compress_ratio: KPool 压缩比。

    返回：
        int: compress_ratio + num_speculative_tokens。
    """
    speculative_config = getattr(vllm_config, "speculative_config", None)
    lookahead = int(getattr(speculative_config, "num_speculative_tokens", 0) or 0)
    if lookahead < 0:
        raise ValueError(f"num_speculative_tokens must be nonnegative, got {lookahead}.")
    return compress_ratio + lookahead


def format_indexer_kpool_slot_mapping(
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    logical_block_size: int,
    compress_ratio: int,
) -> torch.Tensor:
    """Map completed token pools onto the compressed indexer cache.

    把常规 token 槽位映射转换为"池粒度"的压缩索引器槽位映射。

    参数：
        slot_mapping: [num_tokens] 常规 KV 槽位（token 粒度，-1 表示无效）。
        positions: [num_tokens] token 位置。
        logical_block_size: 逻辑块大小（token 数）。
        compress_ratio: KPool 压缩比（池大小）。

    返回：
        torch.Tensor: [num_tokens] 压缩后槽位。仅当该 token 是某个
        "已完成池"的最后一个 token（(pos+1) % kpool == 0 且槽位有效）时
        给出有效压缩槽位，否则为 -1（不写缓存）。

    算法步骤：
        1. valid = 槽位有效 且 (position+1) 整除 compress_ratio
           （即该 token 恰好凑满一池，需要做一次池化写入）；
        2. block_id = slot // block_size，offset = slot % block_size；
        3. 压缩槽位 = block_id * (block_size/kpool) + offset // kpool
           ——块内 token 数除以压缩比、池内偏移除以压缩比。
    """
    if compress_ratio <= 1 or logical_block_size <= 0 or logical_block_size % compress_ratio:
        raise ValueError(
            f"logical_block_size={logical_block_size} must be divisible by compress_ratio={compress_ratio}."
        )
    # 步骤1: 找出"完成池"的行——槽位有效且 (pos+1) 是 kpool 的整数倍。
    valid = (slot_mapping >= 0) & (torch.remainder(positions + 1, compress_ratio) == 0)
    # clamp_min(0) 保证无效槽位（-1）参与除法时不产生越界/NaN。
    safe_slots = slot_mapping.clamp_min(0)
    # 步骤2: 拆出块 id 与块内偏移。
    block_ids = torch.div(safe_slots, logical_block_size, rounding_mode="floor")
    offsets = torch.remainder(safe_slots, logical_block_size)
    # 步骤3: 依压缩比换算到池粒度槽位；无效行统一置 -1。
    compressed_slots = block_ids * (logical_block_size // compress_ratio) + torch.div(
        offsets,
        compress_ratio,
        rounding_mode="floor",
    )
    return torch.where(valid, compressed_slots, torch.full_like(compressed_slots, -1))


class Glm5NextIndexerCache(nn.Module, AttentionLayerBase):
    """Independently allocated compressed-K cache for the GLM-Next indexer.

    GLM-Next 索引器的独立压缩 K 缓存层。

    继承（语法点：多继承 nn.Module + AttentionLayerBase——前者提供 PyTorch
    模块能力，后者是 vLLM 的缓存层协议，要求实现 get_kv_cache_spec /
    get_attn_backend，模型运行时会据此分配显存并绑定后端）。

    原理：存"已完成池"的压缩 K 向量（BF16）。每 compress_ratio 个 token
    池化成 1 条存储，page/block 语义为"池"而非 token。
    """

    # Auxiliary caches use the GLM-specific small-page class instead of the
    # generic attention/Mamba page-size class.
    # 辅助缓存使用 GLM 专用小页类，而非通用 attention/Mamba 页大小类。
    align_kv_cache_with_mamba = False

    def __init__(
        self,
        *,
        head_dim: int,
        dtype: torch.dtype,
        cache_role: str,
        cache_config: CacheConfig,
        prefix: str,
        compress_ratio: int,
    ) -> None:
        """初始化压缩索引器缓存层。

        参数（语法点：* 之后全部为 keyword-only 参数）：
            head_dim: 索引器头维度（128）。
            dtype: 存储 dtype（BF16）。
            cache_role: 缓存角色标记（"indexer"）。
            cache_config: 缓存配置（提供 block_size）。
            prefix: 层名（全局唯一，注册进静态前向上下文）。
            compress_ratio: KPool 压缩比，必须 >1 且整除 block_size。
        """
        super().__init__()
        # 校验：block_size 必须能被压缩比整除（保证池与块边界对齐）。
        if compress_ratio <= 1 or cache_config.block_size % compress_ratio:
            raise ValueError(
                "GLM-Next indexer cache requires block_size divisible by a "
                f"compress_ratio greater than one, got {cache_config.block_size} "
                f"and {compress_ratio}."
            )
        self.head_dim = head_dim
        self.dtype = dtype
        self.cache_role = cache_role
        self.cache_config = cache_config
        self.compress_ratio = compress_ratio
        self.prefix = prefix
        # 步骤1: 为每个流水线（PP）stage 预留一个空张量占位；
        # 真实缓存由模型运行时在 NPU 上分配后回填到 self.kv_cache。
        current_config = get_current_vllm_config()
        self.kv_cache = [torch.tensor([]) for _ in range(current_config.parallel_config.pipeline_parallel_size)]
        # 步骤2: 注册到静态前向上下文（forward context）——
        # 注意力后端在前向时按 prefix 查找本层以绑定缓存。
        static_context = current_config.compilation_config.static_forward_context
        if prefix in static_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        static_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        """返回本层的缓存规格（供调度器与显存规划器使用）。

        返回：
            AscendMLAAttentionSpec：携带 model_version="glm5_next" 与
            indexes_kv_by_block_stride=True（KPool 块步长索引标记），
            tokens_per_state=compress_ratio 声明压缩比。
        语法点：del vllm_config 显式声明不使用该参数。
        """
        del vllm_config
        ratio_kwargs: dict[str, Any] = {"tokens_per_state": self.compress_ratio}
        return AscendMLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            cache_dtype_str=None,
            model_version="glm5_next",
            indexes_kv_by_block_stride=True,
            **ratio_kwargs,
        )

    def get_attn_backend(self):
        """返回配套的 Ascend 注意力后端类（延迟导入避免启动期依赖）。"""
        from vllm_ascend.attention.indexer_kpool import (
            AscendIndexerKPoolBackend,
        )

        return AscendIndexerKPoolBackend

    def forward(self): ...
    # 前向为空操作：本层只承载缓存注册，不参与计算图。


class Glm5NextTailCache(nn.Module, AttentionLayerBase):
    """Fixed per-request FP32 ring containing raw keys and gate scores.

    请求私有的 FP32 固定环形缓存：保存"未凑满一池"的原始 K 与门控分数。

    原理：KPool 压缩只在凑满 kpool 个 token 后发生；正在累积中的
    0..kpool-1 个 token 的原始 K/gate 暂存在本环里。环容量 =
    kpool + 投机前瞻（见 get_kpool_tail_ring_capacity）。环绑定到请求
    独占的块，不做前缀共享（见 KpoolTailManager）。
    """

    align_kv_cache_with_mamba = False

    def __init__(
        self,
        *,
        head_dim: int,
        dtype: torch.dtype,
        compress_ratio: int,
        prefix: str,
        ring_capacity: int | None = None,
    ) -> None:
        """初始化尾部缓存层。

        参数：
            head_dim: 索引器头维度。
            dtype: 必须 torch.float32（原始 K/gate 需要高精度累积）。
            compress_ratio: KPool 压缩比（即最小环容量）。
            prefix: 层名（注册进静态前向上下文）。
            ring_capacity: 环容量；None 时取 compress_ratio。
        """
        super().__init__()
        if dtype != torch.float32:
            raise ValueError(f"GLM-Next tail must use torch.float32, got {dtype}.")
        if compress_ratio <= 1:
            raise ValueError(f"GLM-Next tail requires compress_ratio greater than one, got {compress_ratio}.")
        self.block_size = compress_ratio if ring_capacity is None else ring_capacity
        if self.block_size < compress_ratio or head_dim <= 0:
            raise ValueError("Tail requires a positive head_dim and ring_capacity >= compress_ratio.")
        self.head_dim = head_dim
        self.dtype = dtype
        self.prefix = prefix
        self.compress_ratio = compress_ratio
        # 同 Glm5NextIndexerCache：PP 占位张量 + 静态上下文注册。
        current_config = get_current_vllm_config()
        self.kv_cache = [torch.tensor([]) for _ in range(current_config.parallel_config.pipeline_parallel_size)]
        static_context = current_config.compilation_config.static_forward_context
        if prefix in static_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        static_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        """返回尾部缓存规格：AscendIndexerKPoolTailSpec（滑窗=压缩比）。"""
        del vllm_config
        return AscendIndexerKPoolTailSpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            sliding_window=self.compress_ratio,
            compress_ratio=self.compress_ratio,
        )

    def get_attn_backend(self):
        """返回尾部缓存的 Ascend 注意力后端类。"""
        from vllm_ascend.attention.indexer_kpool import (
            AscendIndexerKPoolTailBackend,
        )

        return AscendIndexerKPoolTailBackend

    def forward(self): ...
    # 前向为空操作：仅承载缓存注册。


class KpoolTailManager(FullAttentionManager):
    """Own one unshared ring block until request completion or preemption.

    Prefix hits never initialize this transient cache. Logical pool-aligned
    prefix lookup lets the next forward seed it without historical tail reads.

    尾部缓存的调度管理器：每个请求独占一个不共享的环块，直到请求完成
    或被抢占（preemption）。

    继承关系：继承上游 vLLM 的 FullAttentionManager（全注意力块管理器），
    但覆写为"禁用一切前缀缓存/共享"——尾部环是请求私有的瞬态数据，
    逻辑上按池对齐的前缀查找已保证下一次前向无需读取历史尾部。

    覆写方法：find_longest_cache_hit（返回空命中）、cache_blocks（no-op）、
    get_num_common_prefix_blocks（返回 0）、get_num_blocks_to_allocate /
    allocate_new_blocks（每请求至多 1 块）等。
    """

    supports_fine_grained_hash_lookup: ClassVar[bool] = False
    # 语法点：ClassVar[bool] 类型注解表示类级常量（非实例字段）。

    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        scheduler_block_size: int,
        **kwargs: Any,
    ) -> None:
        """初始化管理器；强制 enable_caching=False。

        参数：
            kv_cache_spec: 本组缓存规格。
            block_pool: 全局块池（所有调度组共享）。
            enable_caching: 调用方传入的缓存开关（本类忽略并强制关闭）。
            kv_cache_group_id: 所属缓存组 id。
            scheduler_block_size: 调度器块大小。
        """
        # The global pool can cache other groups; this manager never does.
        # Disable it before entering the base constructor, independently of
        # coordinator-side filtering and the caller's global caching setting.
        # 全局块池可以为其他组做缓存；本管理器绝不缓存。在进入基类构造器
        # 之前强制关闭缓存开关，独立于协调器侧过滤与调用方的全局设置。
        super().__init__(
            kv_cache_spec=kv_cache_spec,
            block_pool=block_pool,
            enable_caching=False,
            kv_cache_group_id=kv_cache_group_id,
            scheduler_block_size=scheduler_block_size,
            **kwargs,
        )

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """前缀命中查找：永远返回空（尾部环不做前缀共享）。

        语法点：@classmethod 装饰器——方法接收类（cls）而非实例。
        """
        return tuple([] for _ in kv_cache_group_ids), 0

    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
        *,
        replay_boundary: int | None = None,
        replay_boundaries: Sequence[int] | None = None,
    ) -> None:
        """把块发布进共享缓存：no-op（请求私有环不发布）。

        上游有两种 replay boundary 拼写；两条路径都不把本请求私有的
        环发布到共享块缓存。
        """
        # Upstream uses either replay boundary spelling; neither path publishes
        # this request-private ring to the shared block cache.
        return

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """统计与运行请求的公共前缀块数：恒为 0。"""
        return 0

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        """需要分配的块数：无块则 1（首次出现），有块则 0。

        语法点：`0 if cond else 1` 条件表达式；req_to_blocks 是
        基类维护的 请求id -> 块列表 映射，空列表为假值。
        """
        return 0 if self.req_to_blocks.get(request_id) else 1

    def allocate_new_blocks(self, request_id: str, num_tokens: int, num_tokens_main_model: int) -> list[KVCacheBlock]:
        """为请求分配环块：至多一块，重复调用为 no-op。"""
        req_blocks = self.req_to_blocks[request_id]
        if req_blocks:
            return []
        new_blocks = self.block_pool.get_new_blocks(1)
        req_blocks.extend(new_blocks)
        if self._record_new_block_ids:
            self.new_block_ids.extend(block.block_id for block in new_blocks)
        return new_blocks

    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """登记本地已计算块：断言无（尾部块不能通过前缀缓存共享）。"""
        assert not new_computed_blocks, "Tail blocks cannot be shared through prefix caching."

    def allocate_external_computed_blocks(
        self, request_id: str, num_local_computed_tokens: int, num_external_computed_tokens: int
    ) -> None:
        """登记外部（如 PD 分离）已计算块：退化为普通分配（环仍不共享）。"""
        self.allocate_new_blocks(request_id, num_local_computed_tokens + num_external_computed_tokens, 0)
