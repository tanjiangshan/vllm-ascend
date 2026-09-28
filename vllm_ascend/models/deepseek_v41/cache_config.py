# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# 【文件职责】DeepSeek V4.1 混合 KV cache 的"框架侧"布局与分配策略。
#
# 【为什么需要自定义布局】V4.1 的 cache 资源种类比普通模型多得多：
#   1. 全上下文 MLA 层的压缩 KV（kv latent，每 token 一个 head_dim 潜向量）；
#   2. indexer 的 INT8 index-K cache（每 compress_ratio 个 token 一条）；
#   3. compressor 的环形状态缓冲（CircularBufferSpec，FP32，供 ratio-2 门控）；
#   4. 目标模型 SWA（滑窗注意力）层的 KV；
#   5. DSpark 草稿模型的 SWA KV。
# 上游 vLLM 的通用策略（同类型 spec 自动分组合并）无法表达"跨层共享同一份
# cache + 单 block 池"的需求，因此本文件在框架侧（v1 core kv_cache_utils
# 调用路径）接管布局计算：
#   - get_layer_tuples(): 把每一组共享物理页的资源拼成"层元组"
#     (kv_name, index_name, 状态别名, SWA 别名, 草稿 SWA 别名)；
#   - group_cache_specs(): 补齐(padded)每页字节数后按 spec 类型分组；
#   - get_deepseek_v41_kv_cache_config(): 用一个全局 block-ID 池为每个层元组
#     分配独立 tensor，容量 = 可用显存 // Σ(每页字节数)。
#
# 【张量流】调度器给出各层 spec → 本文件算出层元组与页大小 → KVCacheConfig
# 描述每个 tensor 的 size/block_stride → Ascend KV allocator 据此绑定
# 各 DeepseekV41CacheLayer 的 kv_cache 张量。
# =============================================================================
"""Framework-side V4.1 layer-outermost cache placement and allocation."""

from dataclasses import replace

from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_utils import may_override_num_blocks
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
)

# 环形状态缓冲（compressor 的 CircularBufferSpec）每个逻辑 block 的行数。
# engram/compressor 的环形历史按 32 行一个 block 分配，与调度器 block 粒度对齐。
STATE_RING_ROWS = 32


def is_deepseek_v41_cache(specs_or_groups):
    """判断传入的 cache 规格/分组是否属于 DeepSeek V4.1 模型。

    参数:
        specs_or_groups: 两种输入形态——
            1) dict: {层名: KVCacheSpec}，由各层 get_kv_cache_spec() 汇总而来；
            2) list: KVCacheGroupSpec 列表（调度器已分好组的结果）。
    返回:
        bool: 只要任意一层的 spec 带 model_version == "deepseek_v41" 即为 True。
    原理: V4.1 的 Ascend 规格（AscendMLAAttentionSpec 等）构造时打上了
        model_version="deepseek_v41" 标记；框架据此识别 V4.1 并切换到本文件的
        自定义布局，而非上游通用的"同 spec 自动分组合并"策略。
    """
    if isinstance(specs_or_groups, dict):
        specs = list(specs_or_groups.values())
    else:
        specs = []
        for item in specs_or_groups:
            # 兼容两种元素：直接的 spec 对象，或包装了 kv_cache_spec 属性的
            # KVCacheGroupSpec；后者内部可能是 UniformTypeKVCacheSpecs
            # （同类型多层的统一容器），需要展开其内部所有 spec。
            spec = getattr(item, "kv_cache_spec", item)
            if isinstance(spec, UniformTypeKVCacheSpecs):
                specs.extend(spec.kv_cache_specs.values())
            else:
                specs.append(spec)
    return any(getattr(spec, "model_version", None) == "deepseek_v41" for spec in specs)


def _layer_number(name):
    """从层参数名中解析层号。原理: 名形如 "model.layers.24.self_attn.xxx"，
    以 ".layers." 为界取最后一段的首个数字。"""
    return int(name.rsplit(".layers.", 1)[1].split(".", 1)[0])


def _draft_layer_number(name):
    """解析 DSpark 草稿层号。原理: 草稿层名形如 "model.mtp.0.self_attn..."，
    以 ".mtp." 为界取段首数字；前缀 "." 保证名字开头就是 mtp 时也能切分。"""
    return int(("." + name).rsplit(".mtp.", 1)[1].split(".", 1)[0])


def get_layer_tuples(specs):
    """Return DSV4-style ordered layer tuples and their physical page sizes."""
    """【中文说明】把 V4.1 的各类 cache 资源组织成"层元组"，每个元组内的所有
    资源共享同一个物理页槽位（同一个 block ID 空间）。

    参数:
        specs: {层名: KVCacheSpec}。
    返回:
        (page_sizes, layer_tuples):
        - layer_tuples: 每项为 (kv_name, index_name, *aliases)——
          kv_name 是全上下文 MLA KV，index_name 是 indexer K cache，
          aliases 依次为：该槽位的环形状态缓冲（若有）、按层错开切片的
          目标 SWA 层、草稿(draft) SWA 层；
        - page_sizes: 每个元组的物理页字节数 = 元组内最大资源页大小
          （保证任何一个别名资源都放得下）。
    原理: V4.1 只有部分层（kv_source_layer_ids 指定的"源层"）真正拥有
        全上下文 KV；其余层是 SWA 层或复用源层 cache 的消费层。把 SWA 层
        "错开铺"到各源层槽位上（第 i 个 SWA 层放到 slot i % len(full)），
        可让多个 SWA 层与源层共享 block 池，从而压缩总显存。
    """
    # 步骤1: 按类型收集所有资源名。
    # mla = 全上下文 MLA KV 层；state = 环形状态缓冲；swa = 滑窗层（含草稿）。
    mla = {name for name, spec in specs.items() if isinstance(spec, AscendMLAAttentionSpec)}
    state = sorted((name for name, spec in specs.items() if isinstance(spec, CircularBufferSpec)), key=_layer_number)
    swa = {name for name, spec in specs.items() if isinstance(spec, AscendSlidingWindowMLASpec)}

    # 步骤2: 细分。full = scale_dim 为空(未压缩)的 MLA 层（真正的源层）；
    # target_swa = 主模型滑窗层（名字不含 ".mtp."）；draft_swa = 草稿模型滑窗层。
    full = sorted((name for name in mla if not specs[name].scale_dim), key=_layer_number)
    target_swa = sorted((name for name in swa if ".mtp." not in f".{name}"), key=_layer_number)
    draft_swa = sorted((name for name in swa if ".mtp." in f".{name}"), key=_draft_layer_number)

    layer_tuples: list[tuple[str, ...]] = []
    page_sizes: list[int] = []
    # 步骤3: 为每个源层槽位组装层元组。slot_idx 即槽位序号。
    for slot_idx, kv_name in enumerate(full):
        prefix = kv_name.rsplit(".", 1)[0]
        # indexer K cache 与 KV 同层：把 "...self_attn.long_kv_cache" 换成
        # "...self_attn.indexer.k_cache"。
        index_name = prefix + ".indexer.k_cache"
        index_spec = specs[index_name]
        kv_spec = specs[kv_name]
        # 别名 = 环形状态(仅当该槽位有对应 state) + 错开切片的目标 SWA 层。
        # 语法点: swa 列表按 len(full) 取步长切片，使 SWA 层均匀分布到槽位。
        aliases = ([state[slot_idx]] if slot_idx < len(state) else []) + target_swa[slot_idx :: len(full)]
        kv_bytes = kv_spec.unpadded_page_size_bytes
        index_bytes = index_spec.unpadded_page_size_bytes
        # 草稿 SWA 层也按槽位一一对应地挂进来。
        if slot_idx < len(draft_swa):
            aliases.append(draft_swa[slot_idx])
        # 页大小取"KV+indexer 合计"与"最大别名资源"中的较大者——
        # 同一物理页要能容纳元组内任何一种资源布局。
        capacity = max(
            kv_bytes + index_bytes,
            *(specs[name].unpadded_page_size_bytes for name in aliases),
        )
        layer_tuples.append((kv_name, index_name, *aliases))
        page_sizes.append(capacity)
    return page_sizes, layer_tuples


def group_cache_specs(specs):
    """Merge full-context resources and pad layer tuples without mutating inputs."""
    """【中文说明】为每个层元组内的资源补齐(padded)页大小并按 spec 类型重新分组。

    参数:
        specs: {层名: KVCacheSpec}。
    返回:
        list[UniformTypeKVCacheSpecs]: 统一类型分组列表——
        [0] 所有 MLA（KV + indexer K）合并组；
        [1] 环形状态缓冲组；
        [2:] 每个"错位 SWA 组"一组（见下方转置逻辑）+ 草稿 SWA 组（若有）。
    原理: 一个层元组内不同资源的页大小不同，但物理上共用一页。这里用
        dataclasses.replace 生成"页大小补齐到 capacity"的新 spec（不改输入），
        使同组资源页大小一致，才能落进同一个 KVCacheTensor。
    """
    page_sizes, layer_tuples = get_layer_tuples(specs)
    padded = {}
    # 步骤1: 逐元组补齐页大小。KV 保持自身字节数，indexer 占用剩余
    # (page_size - kv_bytes)，各别名资源直接占满整页 page_size。
    for page_size, layer_tuple in zip(page_sizes, layer_tuples):
        kv_name, index_name, *aliases = layer_tuple
        kv_bytes = specs[kv_name].unpadded_page_size_bytes
        padded[kv_name] = replace(specs[kv_name], page_size_padded=kv_bytes)
        padded[index_name] = replace(specs[index_name], page_size_padded=page_size - kv_bytes)
        padded.update((name, replace(specs[name], page_size_padded=page_size)) for name in aliases)

    # 步骤2: 收集 MLA 类与状态类的全部名字，各建一个统一类型组。
    mla_names = [name for name, spec in padded.items() if isinstance(spec, AscendMLAAttentionSpec)]
    state_names = [name for name, spec in padded.items() if isinstance(spec, CircularBufferSpec)]
    groups = [
        UniformTypeKVCacheSpecs.from_specs({name: padded[name] for name in mla_names}),
        UniformTypeKVCacheSpecs.from_specs({name: padded[name] for name in state_names}),
    ]

    # Transpose the physical tuples. Each scheduler SWA group takes one layer
    # from every tuple, so its members use distinct slots at the same block ID.
    # 【中文】步骤3: 物理转置。从每个层元组中抽出目标 SWA 层按列组成新组——
    # 这样每个 SWA 组的成员分别来自不同槽位，使用同一 block ID 时互不冲突
    # （即"错位铺放"：同一 block ID 下，槽位 0 给 SWA 层 A、槽位 1 给 SWA 层 B…）。
    swa_columns = [
        [
            name
            for name in layer_tuple
            if isinstance(padded[name], AscendSlidingWindowMLASpec) and ".mtp." not in f".{name}"
        ]
        for layer_tuple in layer_tuples
    ]
    # 语法点: zip(*swa_columns) 是矩阵转置——第 i 个元组取所有列的第 i 个 SWA 层。
    for names in zip(*swa_columns):
        groups.append(UniformTypeKVCacheSpecs.from_specs({name: padded[name] for name in names}))

    # 步骤4: 所有草稿(".mtp.")SWA 层合成一组。
    draft_names = [
        name
        for layer_tuple in layer_tuples
        for name in layer_tuple
        if isinstance(padded[name], AscendSlidingWindowMLASpec) and ".mtp." in f".{name}"
    ]
    if draft_names:
        groups.append(UniformTypeKVCacheSpecs.from_specs({name: padded[name] for name in draft_names}))
    return groups


def make_cache_groups(grouped_specs):
    """把统一类型分组包装成调度器需要的 KVCacheGroupSpec 列表。

    原理: KVCacheGroupSpec(layer_names, kv_cache_spec) 表示"这些层共用这种
    cache 规格"；层名顺序直接取自组内 spec 字典的键序。
    """
    return [KVCacheGroupSpec(layer_names=list(s.kv_cache_specs), kv_cache_spec=s) for s in grouped_specs]


def _specs_from_groups(groups):
    """把分组结果还原回 {层名: spec} 平铺字典（供 get_layer_tuples 复用）。"""
    specs = {}
    for group in groups:
        for name in group.layer_names:
            specs[name] = group.kv_cache_spec.kv_cache_specs[name]
    return specs


def get_deepseek_v41_pool_bytes_per_block(groups):
    """计算一个 block ID 槽位（跨所有层元组）的总字节数。

    原理: 全局 block 池里一个 block ID 对应每个层元组各一页，因此单 block
    总开销 = Σ(各元组页大小)。调度器用它估算可用 block 数。
    """
    page_sizes, _ = get_layer_tuples(_specs_from_groups(groups))
    return sum(page_sizes)


def get_deepseek_v41_kv_cache_config(
    vllm_config: VllmConfig,
    groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """Allocate four independent layer slots backed by one global block-ID pool."""
    """【中文说明】最终分配入口：用同一个全局 block-ID 数为每个层元组建一块
    独立 tensor，四类槽位（MLA/状态/SWA/草稿）共用一个 block 计数器。

    参数:
        vllm_config: 引擎全局配置。
        groups: group_cache_specs + make_cache_groups 的产物。
        available_memory: 探测得到的可用显存字节数。
    返回:
        KVCacheConfig: num_blocks（全局 block 数）+ 每个 tensor 的
        size/block_stride 布局 + 分组信息。
    算法步骤:
        1) 由层元组得到各页大小，capacity = available_memory // Σ页大小；
        2) may_override_num_blocks 允许用配置覆盖 block 数（测试/复现用）；
        3) 每个层元组生成一个 KVCacheTensor：size = num_blocks * page_size，
           block_stride = page_size（block 间步长），offset/layer_stride 为 0。
    """
    page_sizes, layer_tuples = get_layer_tuples(_specs_from_groups(groups))
    # 步骤1: 全局 block 容量 = 可用显存 // 单 block 总字节数（向下取整）。
    capacity = max(available_memory // sum(page_sizes), 0)
    # 步骤2: 允许用户配置覆盖自动探测的 block 数。
    num_blocks = may_override_num_blocks(vllm_config, capacity)
    tensors: list[KVCacheTensor] = []
    # 步骤3: 每个层元组一块独立 tensor，层名列表即元组内全部资源名。
    for page_size, layer_names in zip(page_sizes, layer_tuples):
        size = num_blocks * page_size
        tensors.append(
            KVCacheTensor(
                size=size,
                layers=list(layer_names),
                offset=0,
                layer_stride=0,
                block_stride=page_size,
            )
        )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
        prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,
    )
