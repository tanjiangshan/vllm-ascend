# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ============================================================================
# 中文模块说明（补充注释，原英文模块 docstring 保留在下方）：
#
# 本文件实现 DeepSeek-V4 系列 DSA（DeepSeek Sparse Attention，稀疏注意力）
# 在昇腾 NPU 上的"注意力层"封装，核心内容：
#   1. DSV4 KV Cache 分块大小表（get_dsv4_block_sizes / dsv4_block_sizes /
#      DSV4_BLOCK_SIZES* 常量）：不同硬件能力（是否支持压缩缓存、A5 BF16
#      KV）下，MLA / SWA / C4 状态 / C128 状态各类缓存的物理分块与页大小。
#   2. DSAAttention：多头潜在注意力（MLA）层，持有注意力超参、按压缩比
#      选择昇腾注意力后端（AscendDSAC4Backend / AscendDSAC128Backend /
#      AscendDSASWABackend），并向 vLLM 申报 KV Cache 规格。
#
# 在 vLLM Attention 抽象中的位置（为何要自定义注意力层）：
# - vLLM v1 的 Attention 层是"声明 + 派发"的薄封装：真正的注意力计算由
#   ModelRunner 持有的 AttentionBackend（封装硬件算子）执行；层对象通过
#   get_kv_cache_spec() 告知调度器"本层需要什么样的 KV Cache 布局"，
#   通过 get_attn_backend() 告知运行时"用哪个后端算子"。
# - 昇腾 NPU 上的 flash attention / 稀疏注意力走 CANN 算子（后端实现见
#   vllm_ascend/attention/dsa_v1.py），与 GPU 的 FA/FlashInfer 接口不同，
#   因此必须提供这套 NPU 定制层与后端。
#
# 使用方：vllm_ascend/ops/dsa.py（AscendDeepseekSparseAttention 包装器，
# 在其 __init__ 中创建 DSAAttention 实例）；deepseek_v4 / deepseek_v41
# 模型与 indexer/compressor 引用本文件的 DSV4_BLOCK_SIZES 等常量。
# ============================================================================
"""Attention layer."""

from typing import Any, cast

import torch
import torch.nn as nn
from vllm.config import CacheConfig, get_current_vllm_config
from vllm.config.vllm import VllmConfig
from vllm.model_executor.layers.attention.attention import _init_kv_cache_quant
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

# from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.platforms import current_platform
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
from vllm.v1.kv_cache_interface import KVCacheSpec

from vllm_ascend.attention.dsa_attn_kv_plan import is_a5_bf16_kv_enabled
# 昇腾 DSA 注意力后端（v1 架构）。原理: vLLM 通过 AttentionBackend 抽象
# 屏蔽硬件差异，下面三个后端分别对应 DeepSeek-V4 的三种 KV 压缩比：
# C4（4 个 token 压缩为 1 个状态）、C128（128:1）、SWA（滑动窗口注意力）。
from vllm_ascend.attention.dsa_v1 import (
    AscendDSAC4Backend,
    AscendDSAC128Backend,
    AscendDSASWABackend,
)
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile


def get_dsv4_block_sizes(use_a5_bf16_kv: bool = False):
    """构建 DeepSeek-V4 KV Cache 分块大小查找表（按硬件能力选择）。

    表结构: {逻辑块大小: [[mla块, swa块, c4状态块, c128状态块],
    [对齐页大小t1, 对齐页大小t2]]}；调用方以 cache_config.block_size
    （调度器侧的逻辑块大小）为键，查出各类缓存组件的物理分块规格。

    Args:
        use_a5_bf16_kv: 是否使用 A5 芯片的 BF16 KV Cache 方案（影响
            c4/c128 状态块大小与页对齐参数）。

    Returns:
        dict[int, list[list[int]]]: 对应当前硬件能力的分块大小表
        （三张候选表见函数体内注释）。
    """
    # cache_config.block_size: [mla, swa, c4 state, c128 state], [page_size_padded_t1, page_size_padded_t2]
    # （中文补充）标准分块表：供不支持 DSV4 压缩缓存的（较老）硬件使用。
    _DSV4_BLOCK_SIZES = {
        128: [[128, 128, 8, 32], [16640, 131072]],
        64: [[64, 64, 4, 16], [8320, 65536]],
        32: [[32, 32, 2, 8], [4160, 32768]],
    }
    # 压缩缓存分块表：硬件支持 DSV4_COMPRESSED_CACHE 能力时使用。
    # 与标准表相比，c128 状态块减半、页参数相应调整
    # （如逻辑块 128 时：c128 状态 32->16，t2 131072->81920，t1 略增）。
    _DSV4_COMPRESSED_BLOCK_SIZES = {
        128: [[128, 128, 8, 16], [16896, 81920]],
        64: [[64, 64, 4, 8], [8448, 40960]],
        32: [[32, 32, 2, 4], [4224, 20480]],
    }
    # A5 芯片 BF16 KV 方案分块表：A5 可用 BF16 存储 KV（判定见
    # dsa_attn_kv_plan.is_a5_bf16_kv_enabled）。c4/c128 状态块沿用压缩
    # 布局，但 t2 页大小保持标准表的值（如逻辑块 128 时 t2 = 131072）。
    _DSV4_BLOCK_SIZES_A5_BF16 = {
        128: [[128, 128, 8, 16], [16896, 131072]],
        64: [[64, 64, 4, 8], [8448, 65536]],
        32: [[32, 32, 2, 4], [4224, 32768]],
    }
    # 步骤1: 通过硬件能力档案探测当前芯片是否支持 DSV4 压缩缓存。
    # 语法点: hardware_profile.supports(能力枚举) 是能力探测接口，
    # 让同一份代码可运行在 910B / A5 等不同昇腾芯片上（能力随芯片降级）。
    if get_current_hardware_profile().supports(HardwareCapability.DSV4_COMPRESSED_CACHE):
        # 步骤2: 支持压缩缓存时，再按是否启用 A5 BF16 KV 二选一。
        if use_a5_bf16_kv:
            return _DSV4_BLOCK_SIZES_A5_BF16
        return _DSV4_COMPRESSED_BLOCK_SIZES
    # 步骤3: 不支持压缩缓存的硬件回退到标准分块表。
    return _DSV4_BLOCK_SIZES


# 模块级常量：import 时根据当前硬件能力物化两份分块表（标准/回退表与
# A5 BF16 表）。注意：由于在导入期求值，import 本模块时硬件上下文必须
# 已完成初始化；运行期若需按引擎配置重新选择，请用 dsv4_block_sizes()。
DSV4_BLOCK_SIZES = get_dsv4_block_sizes()
DSV4_BLOCK_SIZES_A5_BF16 = get_dsv4_block_sizes(use_a5_bf16_kv=True)


def dsv4_block_sizes(vllm_config: VllmConfig):
    """Return the A5 BF16 KV table when explicitly requested, else the upstream table."""
    # （中文补充）运行时查表入口：按引擎级 vllm_config 决定用哪张表。
    # 与模块级常量的区别：本函数可在引擎配置确定后再选择，避免导入期
    # 判断过早。is_a5_bf16_kv_enabled 要求显式传入引擎 vllm_config（而非
    # 进程全局的 current config），防止配置未就绪时误选 FP8 方案。
    if is_a5_bf16_kv_enabled(vllm_config):
        return DSV4_BLOCK_SIZES_A5_BF16
    return DSV4_BLOCK_SIZES


# ============================================================================
# 中文类说明（补充注释）：
# DSAAttention = DeepSeek-V4 的"多头潜在注意力（MLA）+ 稀疏注意力（DSA）"
# 层在昇腾 NPU 上的实现。
#
# 语法点: class DSAAttention(nn.Module, AttentionLayerBase) 是多继承——
#   - nn.Module: PyTorch 模块基类，提供参数注册/子模块管理/钩子机制；
#   - AttentionLayerBase: vLLM 的注意力层抽象基类（接口式约束），要求
#     子类提供 get_attn_backend()/get_kv_cache_spec() 等方法，使调度器
#     与 KV Cache 管理器无需感知硬件细节即可统一编排（Python 按 MRO
#     查找方法，两个基类无冲突方法）。
#
# 设计模式（vLLM v1 Attention 抽象）：
#   - 本层是"元数据载体"：真正的注意力计算不在本类 forward 中完成，
#     而是由注意力后端的 Impl 对象执行（本类把超参透传给 impl），
#     forward 仅返回占位输出；
#   - 通过 get_kv_cache_spec() 向上申报 KV Cache 布局（块大小/头数/
#     head_size/dtype），调度器据此分配 NPU 显存（HBM）。
#
# 使用方：vllm_ascend/ops/dsa.py 的 AscendDeepseekSparseAttention
# （MultiHeadLatentAttentionWrapper 包装器）在 __init__ 中创建本层实例。
# ============================================================================
class DSAAttention(nn.Module, AttentionLayerBase):
    """Multi-Head Latent Attention layer.

    This class takes query, and compressed key/value tensors as input.
    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """
    # （中文补充）MLA（Multi-head Latent Attention）原理：与标准 MHA
    # 为每个头缓存完整 K/V 不同，MLA 先把 K/V 投影压缩为低秩"潜在向量"
    # （kv_c，附带走 RoPE 的 k_pe 部分）再入缓存；推理时由潜在向量经
    # 上采样矩阵还原各头的 K/V，从而把 KV Cache 缩小一个数量级。代价是
    # 注意力计算需融合上采样矩阵乘，因此 NPU 上需要专门的融合算子
    # （见 vllm_ascend/attention/dsa_v1.py 中的 AscendDSAImpl）。

    def __init__(
        self,
        dim: int,
        n_heads: int,
        scale: float,
        n_local_heads: int,
        q_lora_rank: int,
        o_lora_rank: int,
        head_dim: int,
        rope_head_dim: int | None,
        nope_head_dim: int,
        n_groups: int,
        n_local_groups: int,
        window_size: int,
        compress_ratio: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        **extra_impl_args,
    ):
        """初始化 DSA 注意力层：保存超参、选择后端、创建后端实现对象。

        Args:
            dim: 模型隐藏维数。
            n_heads: 全局（TP 切分前）注意力头数。
            scale: 注意力 softmax 前的缩放系数。
            n_local_heads: 本 TP rank 持有的注意力头数。
            q_lora_rank: Query 侧低秩压缩的秩。
            o_lora_rank: 输出投影侧低秩压缩的秩。
            head_dim: 头维度（本层同时将其作为 head_size 使用，MLA 潜在
                表示的头维）。
            rope_head_dim: 走 RoPE 位置编码部分的维度；可为 None。
            nope_head_dim: 不做位置编码部分的维度。
            n_groups: KV 分组总数（MLA 解耦 RoPE 的分组）；n_local_groups
                为本 rank 持有的组数。
            window_size: SWA（滑动窗口注意力）的窗口大小。
            compress_ratio: KV 状态压缩比，决定选用哪个后端（4 -> C4，
                128 -> C128，其余 <=1 -> SWA）。
            cache_config: vLLM 缓存配置（含 cache_dtype）；None 时按
                "auto" 处理。
            quant_config: 量化配置，用于初始化 KV Cache 量化属性。
            prefix: 层名，必须全局唯一（重复注册会抛 ValueError）。
            **extra_impl_args: 透传给后端 impl 的额外关键字参数（如
                wq_a/wq_b/wkv/indexer/compressor/swa_cache_layer 等子模块
                引用）。语法点: ** 把所有未声明的关键字参数打包成 dict，
                使本层无需枚举后端专属参数即可透传。
        """
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.scale = scale
        self.n_local_heads = n_local_heads
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.nope_head_dim = nope_head_dim
        self.n_groups = n_groups
        self.n_local_groups = n_local_groups
        self.window_size = window_size
        self.compress_ratio = compress_ratio
        self.layer_name = prefix
        # 本层 head_size 直接取 head_dim（MLA 潜在向量头维）。
        self.head_size = self.head_dim
        # SWA 缓存层引用：DeepSeek-V4 的滑动窗口 KV 不走统一 KV Cache
        # 分配（见 get_kv_cache_spec 中 SWA 分支返回 None），由
        # DeepseekV4SWACache 单独管理，这里保存引用供后端 impl 使用。
        # 语法点: 变量名后加 ": 类型" 是 PEP 526 变量注解，仅供静态检查。
        self.swa_cache_layer: DeepseekV4SWACache = extra_impl_args.get("swa_cache_layer")

        # 断言 SWA 缓存层必须存在（DeepSeek-V4 每个注意力层都依赖 SWA）。
        assert self.swa_cache_layer is not None

        # 确定 KV Cache dtype 字符串："auto" 表示跟随模型权重 dtype。
        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
        else:
            kv_cache_dtype = "auto"

        # Initialize KV cache quantization attributes
        # （中文补充）从量化配置中提取 KV Cache 量化相关属性挂到 self 上，
        # 供后端量化读写缓存时使用（复用上游 vLLM 的初始化函数）。
        _init_kv_cache_quant(self, quant_config, prefix)

        # 步骤: 按压缩比选择昇腾注意力后端。原理: DeepSeek-V4 的 KV 以
        # "M 个 token 共享 1 个压缩状态"的方式存储——C4 后端对应 4:1
        # 压缩（近程/短上下文），C128 对应 128:1（超长上下文），其余情况
        # （compress_ratio <= 1）为 SWA 滑动窗口注意力。三个后端共享
        # AscendDSABackend 基类，仅名称与支持的内核块大小不同。
        if self.compress_ratio == 4:
            self.attn_backend = AscendDSAC4Backend
        elif self.compress_ratio == 128:
            self.attn_backend = AscendDSAC128Backend
        else:
            self.attn_backend = AscendDSASWABackend

        # NOTE(zxr): vllm_is_batch_invariant is delete during updating to v0.20.1
        # （中文补充）当前后端为 TRITON_MLA / FLASHINFER 时强制关闭前缀
        # 缓存（prefix caching）：这两种后端的计算结果与 batch 划分方式
        # 相关（非 batch-invariant），前缀缓存跨请求复用块可能命中不一致
        # 的结果，故禁用以保证正确性。
        if (
            cache_config is not None
            and cache_config.enable_prefix_caching
            and (self.attn_backend.get_name() == "TRITON_MLA" or self.attn_backend.get_name() == "FLASHINFER")
        ):
            cache_config.enable_prefix_caching = False

        # 步骤: 实例化后端的 Impl（真正执行注意力的对象）。语法点:
        #   - get_impl_cls() 是工厂方法，vLLM AttentionBackend 抽象规定
        #     每个后端提供 impl/metadata/builder 等类族，按需取用；
        #   - cast(type[Any], x) 是 typing.cast，仅向类型检查器断言类型，
        #     运行时不做任何转换或校验。
        impl_cls = cast(type[Any], self.attn_backend.get_impl_cls())
        self.impl = impl_cls(
            dim=self.dim,
            n_heads=self.n_heads,
            scale=self.scale,
            n_local_heads=self.n_local_heads,
            q_lora_rank=self.q_lora_rank,
            o_lora_rank=self.o_lora_rank,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            nope_head_dim=self.nope_head_dim,
            n_groups=self.n_groups,
            n_local_groups=self.n_local_groups,
            window_size=self.window_size,
            compress_ratio=self.compress_ratio,
            vllm_config=get_current_vllm_config(),
            **extra_impl_args,
        )

        # 是否采用"直接调用"而非经不透明自定义算子的注意力调用方式。
        # 原理: 平台通过 opaque_attention_op() 声明注意力是否以 torch
        # 自定义算子（torch.library/custom_ops 注册的 opaque op）形式进入
        # torch.compile 计算图——编译器把 opaque op 当黑盒，不追踪其内部；
        # 昇腾平台该值为 True，故 use_direct_call 为 False（走算子封装路径）。
        self.use_direct_call = not current_platform.opaque_attention_op()

        # 步骤: 把本层注册进 vLLM 的"静态前向上下文"。原理: vLLM v1 的
        # 注意力在 torch.compile 图中以自定义算子形式被调用，算子内部拿
        # 不到层引用，因此运行时通过"层名 -> 实例"映射
        # (static_forward_context) 按名字查找本层及其 KV cache；层名重复
        # 会破坏映射，故直接抛错。语法点: f"..." 是格式化字符串。
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        # KV Cache 占位：为每个流水线并行（PP）阶段准备一个空张量列表。
        # 原理: 真正的 KV Cache 池由 worker/ModelRunner 在模型加载后按
        # get_kv_cache_spec() 申报的规格统一分配并绑定到本属性；这里先
        # 建槽位保证各 PP 阶段都有位置。语法点: 列表推导式中 "_" 表示
        # 不使用的循环变量。
        self.kv_cache = [
            torch.tensor([]) for _ in range(get_current_vllm_config().parallel_config.pipeline_parallel_size)
        ]
        self.kv_cache_dtype = kv_cache_dtype

        # 标记本层走稀疏注意力（DSA）路径。
        self.use_sparse = True

    def forward(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        """前向占位：原样返回 q，不在本方法内计算注意力。

        原理: 在 vLLM v1 + DSA 架构下，本层只是"元数据/注册载体"——真正的
        注意力计算由外层包装器（vllm_ascend/ops/dsa.py 的
        AscendDeepseekSparseAttention）与注意力后端完成：后者通过
        static_forward_context 按层名找到本层及 KV cache，再调用昇腾
        flash attention / 稀疏注意力算子。

        Args:
            q: 查询张量。
            kv_c_normed: 归一化后的压缩 KV 潜在向量（MLA 的 kv_c 部分）。
            k_pe: 走 RoPE 的 key 部分（MLA 解耦位置编码的 k_pe）。
            output_shape: 期望的输出形状；torch.Size | None 是 PEP 604
                可选类型注解。

        Returns:
            原样返回 q（占位输出，实际注意力结果由后端路径产生）。
        """
        # 注意: 不做任何计算，见上方 docstring 的架构说明。
        return q

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        """权重加载完成后的后处理钩子（透传给后端 impl）。

        典型用途: 后端可在此做权重重排/转置（如 NPU 上的 NZ 排布）、
        量化参数离线融合等一次性处理。语法点: hasattr 动态探测方法是否
        存在——impl 不强制实现该可选接口。
        """
        if hasattr(self.impl, "process_weights_after_loading"):
            self.impl.process_weights_after_loading(act_dtype)

    def get_attn_backend(self) -> type[AttentionBackend]:
        """返回本层使用的注意力后端类（AttentionBackend 抽象的 NPU 实现）。

        Returns:
            type[AttentionBackend]: 后端"类对象"而非实例（类型注解
            type[X] 表示 X 的类），调用方可继续使用其静态工厂方法
            （get_name / get_impl_cls / get_supported_kernel_block_sizes 等）。
        """
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        """向 KV Cache 管理器申报本层的缓存规格（分配 HBM 显存的依据）。

        原理: vLLM v1 引擎启动时遍历所有注意力层的本方法，按返回的
        KVCacheSpec 规划显存池；DSA 场景返回 AscendMLAAttentionSpec
        （MLA 单头潜在 KV：num_kv_heads=1，每 token 一条压缩向量）。

        Args:
            vllm_config: 引擎级配置。

        Returns:
            KVCacheSpec 或 None:
            - compress_ratio <= 1（SWA 层）返回 None——滑动窗口 KV 由
              DeepseekV4SWACache 单独分配，不进统一缓存池；
            - 否则返回描述块大小/头数/head_size/dtype 的
              AscendMLAAttentionSpec。
        """
        # SWA 分支：返回 None 表示不参与统一 KV Cache 分配。
        if self.compress_ratio <= 1:  # SWA part. Allocated separately as DeepseekV4SWACache.
            return None
        # 步骤1: 解析缓存 dtype（"auto" -> 跟随模型 dtype），并读取两个
        # 硬件相关开关（A5 BF16 KV、压缩缓存支持）。
        kv_cache_dtype = kv_cache_dtype_str_to_dtype(self.kv_cache_dtype, vllm_config.model_config)
        use_bf16_kv = is_a5_bf16_kv_enabled(vllm_config)
        has_compressed_cache = get_current_hardware_profile().supports(HardwareCapability.DSV4_COMPRESSED_CACHE)

        # 步骤2: 计算每条缓存向量的长度。原理: 压缩缓存硬件（且非 BF16 KV）
        # 时需额外附加 128 的元信息（压缩状态的头布局/尺度等），故 +128。
        # 语法点: "A if cond else B" 是 Python 条件表达式（三元写法）。
        cached_head_size = self.head_size + 128 if has_compressed_cache and not use_bf16_kv else self.head_size
        # 步骤3: 以调度器逻辑块大小为键查分块表，取 mla 分量作为昇腾内核
        # 实际使用的物理存储块大小。
        storage_block_size = dsv4_block_sizes(vllm_config)[vllm_config.cache_config.block_size][0][0]
        # vLLM #51718 replaced MLAAttentionSpec.compress_ratio with
        # AttentionSpec.tokens_per_state on main.
        # （中文补充）上游 vLLM #51718 把规格参数从 compress_ratio 改名为
        # tokens_per_state；用 dict 打包以便兼容上游参数名的变化。
        # 语法点: ratio_kwargs: dict[str, Any] 是带键/值类型的字典类型注解。
        ratio_kwargs: dict[str, Any] = {"tokens_per_state": self.compress_ratio}
        # 步骤4: 构造并返回 MLA 缓存规格对象。
        return AscendMLAAttentionSpec(
            # The scheduler operates in raw-token units. Ascend kernels keep
            # using the compressed page exposed by storage_block_size.
            # （中文补充）调度器以"原始 token"为单位工作，而昇腾内核以
            # "压缩状态页"为单位；因此 block_size = 物理块 x 压缩比，让
            # 两侧共用同一套逻辑块大小（如 C4、物理块 64 -> 逻辑块 256）。
            block_size=storage_block_size * self.compress_ratio,
            # MLA: 所有 query 头共享同一条潜在 KV，等效单 KV 头。
            num_kv_heads=1,
            # 每条缓存向量长度（见上方步骤2）。
            head_size=cached_head_size,
            # 缓存实际存储 dtype。
            dtype=kv_cache_dtype,
            # 标记模型代际，供缓存布局选择。
            model_version="deepseek_v4",
            # 原始 dtype 字符串。
            cache_dtype_str=vllm_config.cache_config.cache_dtype,
            # 语法点: ** 把 dict 解包为关键字参数（tokens_per_state）。
            **ratio_kwargs,
        )
