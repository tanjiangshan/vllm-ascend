# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-Next cache groups and source-compatible physical pool layout.

GLM-Next 的缓存组划分与"源码兼容"的物理池布局。

The main MLA cache and compressed indexer cache share scheduler block IDs,
while compressor tail and every KDA/Mamba group allocate IDs independently.
Physical storage uses standard unpacked KV cache descriptors with two page-size
classes: main MLA/KDA pages and compressed-indexer/tail pages.
（主 MLA 缓存与压缩索引器缓存共享调度器块 id；压缩器尾部环与每个
KDA/Mamba 组各自独立分配 id。物理存储使用标准非打包 KV cache 描述符，
分两种页大小类：主 MLA/KDA 大页与 压缩索引器/尾部 小页。）

整体数据流（GLM-5.Next 混合层的 KV cache 组织）：
  1. 每个 MLA 层登记 3 份缓存：主 MLA cache + 压缩索引器 cache + 尾部环；
  2. 主 MLA 与压缩索引器合并成一个调度组（共享块表/块 id，按池对齐），
     尾部环独立成组（每请求 1 块），KDA 层的 Mamba 状态再按"连续段"
     分成若干组（每段一个组，交错使用）；
  3. 物理层把全局块 id 摊到多个张量槽：主槽（MLA/KDA 共用）+ 小槽
     （索引器 + 尾部各占一半），从而让通用模型运行时分配器无需
     模型特判即可工作。
"""

from dataclasses import dataclass

from vllm.config import VllmConfig
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.core.kv_cache_utils import (
    create_kv_cache_group_specs,
    may_override_num_blocks,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolTailSpec, get_kv_cache_compression_ratio


@dataclass(frozen=True)
class _Glm5NextCacheLayout:
    """GLM-Next 缓存布局描述（不可变数据类）。

    语法点：@dataclass(frozen=True) 自动生成 __init__/__repr__ 等，
    frozen=True 使实例不可变（防止布局被意外篡改）。

    字段：
        full_group: 主 MLA + 压缩索引器合并成的"全历史"调度组。
        tail_group: 尾部环调度组（独立块 id）。
        mamba_groups: 各 KDA/Mamba 组（连续层段分组）。
        mla_names / indexer_names / tail_names: 各类层名（按层号排序）。
        main_page_size / small_page_size: 大页/小页字节数。
        main_slot_count / small_slot_count: 大槽/小槽的张量槽数。
    """

    full_group: KVCacheGroupSpec
    tail_group: KVCacheGroupSpec
    mamba_groups: tuple[KVCacheGroupSpec, ...]
    mla_names: tuple[str, ...]
    indexer_names: tuple[str, ...]
    tail_names: tuple[str, ...]
    main_page_size: int
    small_page_size: int
    main_slot_count: int
    small_slot_count: int


def _is_glm5_next_spec(spec: KVCacheSpec) -> bool:
    """判断 spec 是否带 GLM-Next 标记（model_version == "glm5_next"）。"""
    return getattr(spec, "model_version", None) == "glm5_next"


def _unpadded_page_size(spec: KVCacheSpec) -> int:
    """取 spec 的"未对齐填充"页大小（字节），兼容多种属性名。

    优先 unpadded_page_size_bytes，其次 real_page_size_bytes，
    最后退回 page_size_bytes。
    """
    if hasattr(spec, "unpadded_page_size_bytes"):
        return spec.unpadded_page_size_bytes
    if hasattr(spec, "real_page_size_bytes"):
        return spec.real_page_size_bytes
    return spec.page_size_bytes


def _sorted_layer_names(layer_names: list[str]) -> tuple[str, ...]:
    """按层名中的数字下标排序层名。

    解析失败（层名不含数字，如测试里的合成层名）时保持注册顺序。
    """
    try:
        return tuple(sorted(layer_names, key=extract_layer_index))
    except ValueError:
        # Synthetic layer names used by callers or tests need not contain an
        # integer model-layer index. Preserve their registration order.
        return tuple(layer_names)


def _layer_indices(layer_names: tuple[str, ...]) -> tuple[int, ...] | None:
    """提取各层名中的层号；任一失败返回 None（用于一致性校验）。"""
    try:
        return tuple(extract_layer_index(name) for name in layer_names)
    except ValueError:
        return None


def _is_glm5_next_main_spec(spec: KVCacheSpec) -> bool:
    """是否为主 MLA 规格：MLAAttentionSpec + GLM 标记 + 压缩比 1。"""
    return isinstance(spec, MLAAttentionSpec) and _is_glm5_next_spec(spec) and get_kv_cache_compression_ratio(spec) == 1


def _is_glm5_next_indexer_spec(spec: KVCacheSpec) -> bool:
    """是否为压缩索引器规格：MLAAttentionSpec + GLM 标记 + 压缩比 > 1。"""
    return isinstance(spec, MLAAttentionSpec) and _is_glm5_next_spec(spec) and get_kv_cache_compression_ratio(spec) > 1


def _is_glm5_next_tail_spec(spec: KVCacheSpec) -> bool:
    """是否为尾部环规格：AscendIndexerKPoolTailSpec + GLM 标记 + indexer_tail 角色。"""
    return (
        isinstance(spec, AscendIndexerKPoolTailSpec)
        and _is_glm5_next_spec(spec)
        and getattr(spec, "cache_role", None) == "indexer_tail"
    )


def _align_glm5_next_cache_specs(kv_cache_spec: dict[str, KVCacheSpec]) -> None:
    """Align GLM-Next specs into two physical page-size classes in-place.

    把 GLM-Next 各 spec 就地对齐到两个物理页大小类。

    原理：vLLM 显存规划器要求同组 spec 的页大小一致。这里把
      - 主 MLA + Mamba -> 大页类（main_page_size = 其最大页大小）；
      - 压缩索引器 + 尾部 -> 小页类（small_page_size），
    且小页 >= 大页（保证小槽能容纳大页？实际上小页取 max(大页, 小类最大)，
    使两类页统一，简化池字节计算）。
    通过 object.__setattr__ 绕过 spec 的 frozen 限制直接改 page_size_padded。
    """

    # 步骤1: 按角色把 spec 分成四筐。
    main_specs = [spec for spec in kv_cache_spec.values() if _is_glm5_next_main_spec(spec)]
    indexer_specs = [spec for spec in kv_cache_spec.values() if _is_glm5_next_indexer_spec(spec)]
    tail_specs = [spec for spec in kv_cache_spec.values() if _is_glm5_next_tail_spec(spec)]
    mamba_specs = [spec for spec in kv_cache_spec.values() if isinstance(spec, MambaSpec)]

    # 步骤2: 非 GLM-Next 模型直接返回；三类 GLM spec 必须同时齐备。
    if not main_specs and not indexer_specs and not tail_specs:
        return
    if not main_specs or not indexer_specs or not tail_specs:
        raise ValueError("GLM-Next cache layout requires main MLA, compressed indexer, and compressor-tail specs.")

    # 步骤3: 计算大页（主 MLA + Mamba 的最大页）与小页（不得小于大页）。
    main_candidates = (*main_specs, *mamba_specs)
    main_page_size = max(
        max(spec.page_size_bytes for spec in main_candidates),
        max(_unpadded_page_size(spec) for spec in main_candidates),
    )
    small_candidates = (*indexer_specs, *tail_specs)
    small_page_size = max(
        main_page_size,
        max(spec.page_size_bytes for spec in small_candidates),
        max(_unpadded_page_size(spec) for spec in small_candidates),
    )

    # 步骤4: 就地写入对齐后的填充页大小。
    for spec in main_candidates:
        object.__setattr__(spec, "page_size_padded", main_page_size)
    for spec in small_candidates:
        object.__setattr__(spec, "page_size_padded", small_page_size)


def _create_glm5_next_attention_groups(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """Create the shared full-history group and separate tail group.

    创建注意力侧的两个调度组：共享全历史组（主 MLA + 压缩索引器）与
    独立的尾部环组。

    校验内容：
      - 三类缓存数量必须相等（每个 MLA 层 3 份）；
      - 三类层号必须一一对应；
      - 主 MLA 与索引器必须同 block_size 且整除压缩比；
      - 尾部环压缩比与索引器一致、容量 >= 压缩比。
    """

    # 步骤1: 按角色收集并排序层名。
    main_names = _sorted_layer_names([name for name, spec in kv_cache_spec.items() if _is_glm5_next_main_spec(spec)])
    indexer_names = _sorted_layer_names(
        [name for name, spec in kv_cache_spec.items() if _is_glm5_next_indexer_spec(spec)]
    )
    tail_names = _sorted_layer_names([name for name, spec in kv_cache_spec.items() if _is_glm5_next_tail_spec(spec)])

    # 步骤2: 校验完备性——不允许出现三类之外的缓存角色。
    classified_names = {*main_names, *indexer_names, *tail_names}
    if classified_names != set(kv_cache_spec):
        raise ValueError("GLM-Next KV cache specs contain an unsupported cache role.")
    if not (len(main_names) == len(indexer_names) == len(tail_names) > 0):
        raise ValueError(
            "Every GLM-Next MLA layer requires one main MLA, compressed indexer, and compressor-tail cache."
        )

    # 步骤3: 校验三层名集合的层号一致。
    main_indices = _layer_indices(main_names)
    if main_indices is not None and (
        main_indices != _layer_indices(indexer_names) or main_indices != _layer_indices(tail_names)
    ):
        raise ValueError("GLM-Next MLA, indexer, and tail cache layer indices do not match.")

    # 步骤4: 校验主 MLA 与索引器共用一个逻辑块大小。
    full_block_sizes = {kv_cache_spec[name].block_size for name in (*main_names, *indexer_names)}
    if len(full_block_sizes) != 1:
        raise ValueError("GLM-Next main MLA and compressed indexer caches must use one logical block size.")

    # 步骤5: 逐层校验压缩比约束（块大小整除压缩比；尾部环参数匹配）。
    for main_name, indexer_name, tail_name in zip(main_names, indexer_names, tail_names):
        main_spec = kv_cache_spec[main_name]
        indexer_spec = kv_cache_spec[indexer_name]
        tail_spec = kv_cache_spec[tail_name]
        assert isinstance(main_spec, MLAAttentionSpec)
        assert isinstance(indexer_spec, MLAAttentionSpec)
        assert isinstance(tail_spec, AscendIndexerKPoolTailSpec)

        compress_ratio = get_kv_cache_compression_ratio(indexer_spec)
        if main_spec.block_size % compress_ratio:
            raise ValueError(
                "GLM-Next logical block size must be divisible by the indexer "
                f"compression ratio: block_size={main_spec.block_size}, "
                f"compress_ratio={compress_ratio}."
            )
        if tail_spec.compress_ratio != compress_ratio or tail_spec.block_size < compress_ratio:
            raise ValueError(
                f"GLM-Next tail must use compression ratio {compress_ratio} and have at least that ring capacity."
            )

    # Main and indexer caches deliberately share scheduler block IDs. Keep a
    # main spec first because the scheduler unwraps the first nested spec and
    # must select FullAttentionManager for this combined group.
    # 步骤6: 组装"全历史"组——主 MLA 与压缩索引器故意共享调度器块 id
    # （按池对齐后两者块表语义一致）。主 spec 必须排第一：调度器解包
    # 第一个嵌套 spec 来选择管理器，必须是 FullAttentionManager。
    full_names = [
        name for main_name, indexer_name in zip(main_names, indexer_names) for name in (main_name, indexer_name)
    ]
    full_specs = {name: kv_cache_spec[name] for name in full_names}
    full_uniform_spec = UniformTypeKVCacheSpecs.from_specs(full_specs)
    if full_uniform_spec is None:
        raise ValueError(
            "GLM-Next main MLA and compressed indexer caches must have uniform full-attention block-table semantics."
        )

    # 步骤7: 组装尾部环组（要求各尾部 spec 的固定块环语义一致）。
    tail_specs = {name: kv_cache_spec[name] for name in tail_names}
    tail_uniform_spec = UniformTypeKVCacheSpecs.from_specs(tail_specs)
    if tail_uniform_spec is None:
        raise ValueError("GLM-Next compressor-tail caches must have uniform fixed-block circular semantics.")

    return [
        KVCacheGroupSpec(full_names, full_uniform_spec),
        KVCacheGroupSpec(list(tail_names), tail_uniform_spec),
    ]


def _get_glm5_next_cache_layout(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> _Glm5NextCacheLayout | None:
    """Recognize validated GLM-Next groups and derive physical slot counts.

    从（已构建好的）缓存组列表中识别 GLM-Next 布局并推导物理槽位数。

    返回：
        _Glm5NextCacheLayout；若列表中没有 GLM-Next 组则返回 None。

    识别逻辑：
      - MambaSpec 组 -> mamba_groups；
      - UniformTypeKVCacheSpecs 组，含主 MLA + 压缩索引器 -> full_groups；
      - 全部为尾部环 spec -> tail_groups。
      校验必须恰好 1 个 full 组 + 1 个 tail 组，不允许其它类型混入。
    """

    if not kv_cache_groups:
        return None

    # 步骤1: 分类收集各组。
    full_groups: list[KVCacheGroupSpec] = []
    tail_groups: list[KVCacheGroupSpec] = []
    mamba_groups: list[KVCacheGroupSpec] = []
    for group in kv_cache_groups:
        group_spec = group.kv_cache_spec
        if isinstance(group_spec, MambaSpec):
            mamba_groups.append(group)
            continue
        if not isinstance(group_spec, UniformTypeKVCacheSpecs):
            continue

        values = list(group_spec.kv_cache_specs.values())
        if (
            values
            and all(isinstance(spec, MLAAttentionSpec) and _is_glm5_next_spec(spec) for spec in values)
            and any(_is_glm5_next_main_spec(spec) for spec in values)
            and any(_is_glm5_next_indexer_spec(spec) for spec in values)
        ):
            full_groups.append(group)
        elif values and all(_is_glm5_next_tail_spec(spec) for spec in values):
            tail_groups.append(group)

    # 步骤2: 校验组数量与类型完备性。
    has_glm5_next_group = bool(full_groups or tail_groups)
    if not has_glm5_next_group:
        return None
    if len(full_groups) != 1 or len(tail_groups) != 1:
        raise ValueError(
            "GLM-Next requires exactly one combined main/indexer group and one compressor-tail KV cache group."
        )
    if len(full_groups) + len(tail_groups) + len(mamba_groups) != len(kv_cache_groups):
        raise ValueError("GLM-Next KV cache groups contain an unsupported cache spec.")

    # 步骤3: 从 full 组中按角色拆出层名并排序。
    full_group = full_groups[0]
    tail_group = tail_groups[0]
    assert isinstance(full_group.kv_cache_spec, UniformTypeKVCacheSpecs)
    assert isinstance(tail_group.kv_cache_spec, UniformTypeKVCacheSpecs)
    full_specs = full_group.kv_cache_spec.kv_cache_specs
    tail_specs = tail_group.kv_cache_spec.kv_cache_specs
    mla_names = _sorted_layer_names(
        [name for name in full_group.layer_names if _is_glm5_next_main_spec(full_specs[name])]
    )
    indexer_names = _sorted_layer_names(
        [name for name in full_group.layer_names if _is_glm5_next_indexer_spec(full_specs[name])]
    )
    tail_names = _sorted_layer_names(tail_group.layer_names)
    # 步骤4: 校验三类层名数量与层号一一对应。
    if not (len(mla_names) == len(indexer_names) == len(tail_names)):
        raise ValueError("Every GLM-Next MLA layer must own one compressed indexer and one compressor-tail cache.")
    mla_indices = _layer_indices(mla_names)
    if mla_indices is not None and (
        mla_indices != _layer_indices(indexer_names) or mla_indices != _layer_indices(tail_names)
    ):
        raise ValueError("GLM-Next MLA, indexer, and tail cache layer indices do not match.")

    # Pipeline-parallel projection keeps empty groups with their global spec.
    # Derive the two canonical page classes from the retained specs, while the
    # slot counts below use only this worker's projected layer names.
    # 步骤5: 推导两个规范页大小类。注意：流水线并行的投影会保留空组及其
    # 全局 spec，因此页大小从保留的 spec 推导；而槽位数只用本 worker
    # 投影后的层名计算。
    main_page_sizes = {spec.page_size_bytes for spec in full_specs.values() if _is_glm5_next_main_spec(spec)} | {
        group.kv_cache_spec.page_size_bytes for group in mamba_groups
    }
    small_page_sizes = {spec.page_size_bytes for spec in full_specs.values() if _is_glm5_next_indexer_spec(spec)} | {
        spec.page_size_bytes for spec in tail_specs.values()
    }
    if len(main_page_sizes) != 1 or len(small_page_sizes) != 1:
        raise ValueError("GLM-Next cache specs were not aligned to two physical page sizes.")

    # 步骤6: 计算大槽数量 = max(MLA 层数, 各 Mamba 组层数)——
    # 主槽被 MLA 层与 Mamba 组按"第 i 层 <-> 第 i 槽"复用（不同组块 id
    # 独立，可共享物理槽）。小槽数量 = 索引器层数。
    main_slot_count = max(
        (
            len(mla_names),
            *(len(group.layer_names) for group in mamba_groups),
        )
    )
    return _Glm5NextCacheLayout(
        full_group=full_group,
        tail_group=tail_group,
        mamba_groups=tuple(mamba_groups),
        mla_names=mla_names,
        indexer_names=indexer_names,
        tail_names=tail_names,
        main_page_size=next(iter(main_page_sizes)),
        small_page_size=next(iter(small_page_sizes)),
        main_slot_count=main_slot_count,
        small_slot_count=len(indexer_names),
    )


def _group_glm5_next_mamba_layer_names(
    kv_cache_spec: dict[str, KVCacheSpec],
    mamba_specs: dict[str, MambaSpec],
) -> list[list[str]]:
    """Recover recurrent runs without depending on spec insertion order.

    恢复 KDA/Mamba 层的"连续段"分组，不依赖 spec 的插入顺序。

    原理：GLM-5.Next 的 KDA 层通常是若干段连续层（如 [0..6], [15..21]...）。
    连续层段中的 KDA 层可安全地交错共享物理槽（第 i 段与第 j 段之间
    用 round-robin 交错），本函数先求最大连续段长度 max_run_length，
    再把全部 KDA 层名按 `names[offset::max_run_length]` 切成
    max_run_length 个交错组。

    返回：
        list[list[str]]：每组的层名列表（组内按层号排序）。
    """

    # 步骤1: 扫描所有层名，校验"一个模型层不能同时含 Mamba 与 MLA spec"
    # 且"每层至多一个 Mamba spec"，并建立 层号->mamba层名 映射。
    layer_is_mamba: dict[int, bool] = {}
    mamba_name_by_index: dict[int, str] = {}
    for name in kv_cache_spec:
        try:
            layer_idx = extract_layer_index(name)
        except ValueError as exc:
            raise ValueError(
                f"GLM-Next Mamba grouping requires layer names with numeric indices, got {name!r}."
            ) from exc

        is_mamba = name in mamba_specs
        # setdefault 返回已存在的值（若已设置），用于检测同层类型冲突。
        previous_kind = layer_is_mamba.setdefault(layer_idx, is_mamba)
        if previous_kind != is_mamba:
            raise ValueError(
                f"A GLM-Next model layer cannot contain both Mamba and MLA cache specs: layer index {layer_idx}."
            )
        if is_mamba:
            if layer_idx in mamba_name_by_index:
                raise ValueError(
                    f"A GLM-Next model layer must own exactly one Mamba cache spec: layer index {layer_idx}."
                )
            mamba_name_by_index[layer_idx] = name

    # 步骤2: 扫描排序后的层号，求最大连续 KDA 段长度。
    max_run_length = 0
    run_length = 0
    previous_layer_idx: int | None = None
    for layer_idx in sorted(layer_is_mamba):
        is_consecutive = previous_layer_idx is not None and layer_idx == previous_layer_idx + 1
        if layer_is_mamba[layer_idx]:
            run_length = run_length + 1 if is_consecutive else 1
            max_run_length = max(max_run_length, run_length)
        else:
            run_length = 0
        previous_layer_idx = layer_idx

    if max_run_length == 0:
        raise ValueError("GLM-Next Mamba specs were provided but no Mamba layers were found.")

    # 步骤3: 按最大段长交错切组：组 k 收集第 k, k+max_run, ... 个 KDA 层。
    # （`list[offset::step]` 切片语法：从 offset 起每 step 取一个。）
    sorted_mamba_names = [mamba_name_by_index[index] for index in sorted(mamba_name_by_index)]
    return [sorted_mamba_names[offset::max_run_length] for offset in range(max_run_length)]


def _create_mamba_groups(
    mamba_specs: dict[str, MambaSpec],
    grouped_layer_names: list[list[str]],
) -> list[KVCacheGroupSpec]:
    """按分组层名创建 Mamba 缓存组（组内层名先排序再交给上游工具）。"""
    sorted_groups = [list(_sorted_layer_names(layer_names)) for layer_names in grouped_layer_names]
    return create_kv_cache_group_specs(mamba_specs, sorted_groups)


def get_glm5_next_kv_cache_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """Build GLM-Next scheduler groups and align their physical page sizes.

    GLM-Next 缓存组构建总入口：对齐页大小 -> 建注意力组 -> 建 Mamba 组。

    参数：
        vllm_config: vLLM 全局配置。
        kv_cache_spec: 层名 -> KVCacheSpec 映射（模型 get_kv_cache_spec 的汇总）。

    返回：
        list[KVCacheGroupSpec]：调度组列表。

    异常：
        - 无 GLM-Next spec 时报错；
        - 禁用 hybrid KV cache manager 时报错（本布局强依赖混合管理器）。
    """

    if not any(_is_glm5_next_spec(spec) for spec in kv_cache_spec.values()):
        raise ValueError("Expected GLM-Next cache specs.")

    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    if getattr(scheduler_config, "disable_hybrid_kv_cache_manager", False):
        raise ValueError("GLM-Next's paired MLA/indexer and fixed tail layout requires the hybrid KV cache manager.")

    # 步骤1: 页大小对齐（大/小两类）。
    _align_glm5_next_cache_specs(kv_cache_spec)
    # 步骤2: 拆出 Mamba spec，构建注意力组（full + tail）。
    mamba_specs = {name: spec for name, spec in kv_cache_spec.items() if isinstance(spec, MambaSpec)}
    attention_specs = {name: spec for name, spec in kv_cache_spec.items() if not isinstance(spec, MambaSpec)}
    groups = _create_glm5_next_attention_groups(attention_specs)
    if not mamba_specs:
        # The standalone MTP runner has the same attention/tail pairing but
        # no recurrent groups.
        # 独立 MTP 运行器只有注意力/尾部配对，没有循环状态组。
        return groups

    # 步骤3: KDA 层连续段分组并追加 Mamba 组。
    grouped_names = _group_glm5_next_mamba_layer_names(kv_cache_spec, mamba_specs)
    groups.extend(_create_mamba_groups(mamba_specs, grouped_names))
    return groups


def get_glm5_next_pool_bytes_per_block(groups: list[KVCacheGroupSpec]) -> int:
    """Return physical bytes represented by one global block ID.

    一个全局块 id 对应的物理字节数。

    原理：一个全局块 id 会同时在大槽（main_slot_count 个）与小槽
    （small_slot_count 个）各占一页，因此
    bytes_per_block = main_slot_count * main_page_size
                   + small_slot_count * small_page_size。
    """
    layout = _get_glm5_next_cache_layout(groups)
    if layout is None:
        raise ValueError("Expected GLM-Next cache groups.")
    return layout.main_slot_count * layout.main_page_size + layout.small_slot_count * layout.small_page_size


def get_glm5_next_kv_cache_config(
    vllm_config: VllmConfig,
    groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """Describe one standard unpacked tensor per physical cache slot.

    生成 GLM-Next 的 KVCacheConfig：每个物理缓存槽描述一个标准
    "非打包"张量（unpacked：每个张量独立、不共享存储）。

    参数：
        vllm_config: 全局配置（可能带 num_blocks 覆盖）。
        groups: 调度组列表。
        available_memory: 可用显存字节数。

    返回：
        KVCacheConfig：块数 + 张量描述列表 + 组列表。
    """

    layout = _get_glm5_next_cache_layout(groups)
    if layout is None:
        raise ValueError("Expected GLM-Next cache groups.")

    # 步骤1: 依可用显存与每块字节数求块数（可被用户覆盖）。
    bytes_per_block = get_glm5_next_pool_bytes_per_block(groups)
    num_blocks = may_override_num_blocks(vllm_config, max(available_memory // bytes_per_block, 0))
    tensors: list[KVCacheTensor] = []

    def make_tensor(size: int, layer_names: list[str], page_size: int) -> KVCacheTensor:
        """构造一个张量描述：大小、所属层、零偏移、层步长 0、块步长=页大小。"""
        return KVCacheTensor(
            size=size,
            layers=layer_names,
            offset=0,
            layer_stride=0,
            block_stride=page_size,
        )

    # Layers in independent scheduler groups can reuse the same physical slot
    # because their block IDs are allocated independently. A standard unpacked
    # descriptor lets the existing model-runner allocator create one backing
    # tensor per slot without a model-specific allocation path.
    # 步骤2: 生成主槽张量描述。独立调度组的层可以复用同一物理槽——因为
    # 它们的块 id 各自独立分配（同一 id 值在不同组代表不同块）。
    # 用标准"非打包"描述符让现有模型运行时分配器为每个槽创建一个后端
    # 张量，无需模型专用分配路径。
    # 主槽 i 的共享者：第 i 个 MLA 层 + 各 Mamba 组的第 i 层。
    for slot in range(layout.main_slot_count):
        shared_by: list[str] = []
        if slot < len(layout.mla_names):
            shared_by.append(layout.mla_names[slot])
        for group in layout.mamba_groups:
            if slot < len(group.layer_names):
                shared_by.append(group.layer_names[slot])
        tensors.append(
            make_tensor(
                layout.main_page_size * num_blocks,
                shared_by,
                layout.main_page_size,
            )
        )

    # 步骤3: 生成小槽张量描述：每个 MLA 层一个槽，由该层的压缩索引器
    # 缓存与尾部环共享（cache_views.py 会把槽的前/后半段切成两份视图）。
    for indexer_name, tail_name in zip(layout.indexer_names, layout.tail_names):
        tensors.append(
            make_tensor(
                layout.small_page_size * num_blocks,
                [indexer_name, tail_name],
                layout.small_page_size,
            )
        )

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
    )


def get_glm5_next_max_memory_usage(
    vllm_config: VllmConfig,
    groups: list[KVCacheGroupSpec],
) -> int:
    """Return capacity for GLM-Next's shared global block-id pool.

    GLM-Next 共享全局块 id 池的最大显存占用。

    原理：调度组从共享全局 BlockPool 分配互不相交的 id，因此一个请求
    需要为每个组都准备足额的 id（即使不同组的层复用同一物理张量槽）。
    计算方式：各组 max_memory_usage_bytes 上取整到页，求和，再乘以
    每块字节数。
    """

    layout = _get_glm5_next_cache_layout(groups)
    if layout is None:
        raise ValueError("Expected GLM-Next cache groups.")
    # Scheduler groups allocate disjoint IDs from the shared global BlockPool.
    # One request therefore needs enough IDs for every group even though one
    # physical tensor slot can be reused by layers from different groups.
    # 各组的块数（上取整）求和后乘以每全局块的物理字节数。
    blocks = sum(
        (group.kv_cache_spec.max_memory_usage_bytes(vllm_config) + group.kv_cache_spec.page_size_bytes - 1)
        // group.kv_cache_spec.page_size_bytes
        for group in groups
    )
    return blocks * get_glm5_next_pool_bytes_per_block(groups)
