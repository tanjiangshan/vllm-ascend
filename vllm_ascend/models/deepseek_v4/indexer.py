# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""DeepSeek V4 稀疏注意力索引器（Indexer，DeepSeek V3.2 式“闪电索引器”）的昇腾实现。

背景（DSA 稀疏注意力原理）: DeepSeek V3.2 起为超长上下文引入 lightning
indexer——用一个轻量“索引器”代替完整注意力做 token 选择:
  1) Indexer 有自己的低维 q/k（index_head_dim，如 128 维），由 q_lora_rank
     低秩投影 + 专用 RoPE 生成;
  2) q/k 全部 INT8 量化（本文件 quantize_query / quantize_key_and_update_cache），
     用闪电注意力核（npu_quant_lightning_indexer_v2）快速计算
     q·k 相关性分数;
  3) 对每个 query 取分数最高的 topk（index_topk，如 2048）个历史 token，
     返回它们的索引;
  4) 主注意力层（model.py 的 DSA attention）只对这些被选中的 token 计算
     精确 MLA 注意力，从而把长上下文注意力的复杂度从 O(n) 降到 O(topk)。

weights_proj 是逐 head 的分数加权（“可学习的 head 重要性”），乘在
softmax_scale 上参与 TopK 排序。rotate/Hadamard 用于打散 INT8 量化误差。

本文件的 NPU 适配点:
  - 所有量化/TopK 计算走 torch.ops._C_ascend.* / DeviceOperator 自定义算子;
  - 双流（multi-stream）重叠: Indexer 的向量算子（量化/scatter）与主注意力的
    Cube 算子（matmul）在 NPU 的 AIV/Vector 与 Cube 引擎上并行调度;
  - AscendDeepseekV4IndexerCache 重写 KVCacheSpec，使 indexer 的 k_cache
    以压缩布局（block_size = storage_block_size * compress_ratio）分配。

类结构:
  - AscendDeepseekV4IndexerCache: indexer 的 k/scale cache 伪层（继承上游
    vLLM 的 DeepseekV4IndexerCache）;
  - AscendIndexerMetadata/IndexerOverlapPlan: 请求元数据与双流重叠计划;
  - AscendIndexerOps: NPU 算子薄封装（量化/scatter/TopK）;
  - DeepseekV4Indexer: 索引器模块本体（wq_b 投影 + TopK 选择）。
"""
import math
import typing
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch_npu
from torch import nn
from transformers import DeepseekV2Config, DeepseekV3Config
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.models.deepseek_v4.attention import DeepseekV4IndexerCache
from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.kv_cache_interface import KVCacheSpec

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.models.deepseek_v4.compressor import AscendCompressorMetadata, Compressor
from vllm_ascend.ops.cv_linear import CVLinearWrapper
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.quantization.methods import (
    AscendW8A8DynamicLinearMethod,
    AscendW8A8MXFP8DynamicLinearMethod,
)
from vllm_ascend.utils import (
    npu_stream_switch,
)
from vllm_ascend.worker.device_metadata import DeviceMetadataStage, wait_for_device_metadata


def hadamard_linear(x: torch.Tensor, hadamard: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...], int]:
    """Hadamard 矩阵线性变换（维度自动 pad 到 2 的幂）。

    原理: Hadamard 旋转 H 满足 H^T H = nI，对向量做“随机旋转”般的
    线性变换，可打散相邻通道的相关性——配合逐张量 INT8 量化时能把
    量化误差均匀化（Rotate-and-quantize 技巧）。Hadamard 矩阵仅存在于
    2 的幂维度，故先 pad。

    Args:
        x: 任意形状张量（最后一维为变换维度）。
        hadamard: [dim_padded, dim_padded] Hadamard 矩阵。
    Returns:
        (out, x_shape, dim): 变换结果 + 原形状 + 原维度（供 hadamard_scale
        还原时使用）。
    """
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    # 2^ceil(log2(dim)): 不小于 dim 的最小 2 的幂。
    dim_padded = 2 ** math.ceil(math.log2(dim))
    if dim != dim_padded:
        # 右侧补零到 2 的幂维度。
        x = F.pad(x, (0, dim_padded - dim))
    # F.linear(x, H) 即 x @ H^T —— Hadamard 变换。
    return F.linear(x, hadamard), x_shape, dim


def hadamard_scale(out: torch.Tensor, x_shape: tuple[int, ...], dim: int, scale: float = 1.0) -> torch.Tensor:
    """Scale and reshape the output of hadamard_linear."""
    # 【中文】Hadamard 后处理: 乘缩放系数（如 dim^-0.5 保持范数不变），
    # 裁掉 pad 维并还原原形状——与 hadamard_linear 配套使用。
    out = out * scale
    return out[..., :dim].reshape(*x_shape)


def rotate_activation(x: torch.Tensor, hadamard: torch.Tensor) -> torch.Tensor:
    """对激活做 Hadamard 旋转（变换 + 归一化缩放 + 还原的一步封装）。"""
    out, x_shape, dim = hadamard_linear(x, hadamard)
    # dim^-0.5 缩放使 Hadamard 变换保持向量范数（正交变换的归一化因子）。
    return (out * dim**-0.5)[..., :dim].reshape(*x_shape)


def _is_w8a8_dynamic(linear) -> bool:
    """True iff ``linear`` is wired up with ``AscendW8A8DynamicLinearMethod``."""
    # 【中文】判断线性层是否使用 Ascend W8A8 动态量化（权重 INT8、激活
    # 动态量化）。getattr 三连: 无 quant_method 或为非量化方法直接 False;
    # 某些包装层把真正的 method 放在 .quant_method 属性里，需再剥一层。
    quant_method = getattr(linear, "quant_method", None)
    if quant_method is None or isinstance(quant_method, AscendUnquantizedLinearMethod):
        return False
    inner_method = getattr(quant_method, "quant_method", None)
    return isinstance(inner_method, AscendW8A8DynamicLinearMethod)


def _is_mxfp8_dynamic(linear) -> bool:
    """True iff ``linear`` is wired up with ``AscendW8A8MXFP8DynamicLinearMethod``."""
    # 【中文】判断线性层是否使用 Ascend MXFP8 动态量化（微缩放 FP8）。
    # 与 _is_w8a8_dynamic 的区别: MXFP8 的 quant_method 可能直接就是
    # AscendW8A8MXFP8DynamicLinearMethod（不需要剥内层）。
    quant_method = getattr(linear, "quant_method", None)
    if quant_method is None or isinstance(quant_method, AscendUnquantizedLinearMethod):
        return False
    if isinstance(quant_method, AscendW8A8MXFP8DynamicLinearMethod):
        return True
    inner_method = getattr(quant_method, "quant_method", None)
    return isinstance(inner_method, AscendW8A8MXFP8DynamicLinearMethod)


class AscendDeepseekV4IndexerCache(DeepseekV4IndexerCache):
    """Indexer 的 k/scale cache 伪层（NPU 版，继承上游 DeepseekV4IndexerCache）。

    作用: 让 vLLM 为 indexer 分配专用的量化 k_cache 与反量化 scale cache。
    NPU 适配: block_size 乘以 compress_ratio——cache 每个“槽位”实际存储
    compress_ratio 个 token 的压缩表示（压缩布局），块大小查表自
    DSV4_BLOCK_SIZES。
    """

    def __init__(
        self,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
        compress_ratio: int = 1,
    ):
        """初始化。

        Args:
            head_dim: indexer 头维度（如 128）。
            dtype: cache dtype（由 indexer_kv_dtype 配置解析，通常 int8）。
            prefix: 模块名前缀。
            cache_config: cache 配置。
            compress_ratio: 压缩率（本类只支持 4 的分支被创建）。
        """
        super().__init__(head_dim, dtype, prefix, cache_config, compress_ratio)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        """返回 indexer cache 的 KVCacheSpec（vLLM 据此分配 paged cache）。

        NPU 适配: 返回昇腾专有的 AscendMLAAttentionSpec，携带
        model_version="deepseek_v4"、scale 的维度/dtype 等布局细节。
        """
        from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
        from vllm_ascend.models.layer.attention.layer import DSV4_BLOCK_SIZES

        # 存储 block_size 查表（DSV4_BLOCK_SIZES[bs][0] = [mla, swa, c4, c128]）。
        storage_block_size = DSV4_BLOCK_SIZES[vllm_config.cache_config.block_size][0][0]
        # vLLM #51718 replaced MLAAttentionSpec.compress_ratio with
        # AttentionSpec.tokens_per_state on main.
        # 【中文】vLLM #51718 起上游把 compress_ratio 参数改名
        # tokens_per_state——用 dict 解包保持两版参数名兼容。
        ratio_kwargs = {"tokens_per_state": self.compress_ratio}
        # 实际 block_size = 存储 block_size x 压缩率（一槽多 token）。
        # scale_dim/scale_dtype: 反量化 scale 存放维度与精度——支持
        # DSV4_COMPRESSED_CACHE 的硬件（A5）用 float32，否则 float16。
        return AscendMLAAttentionSpec(
            block_size=storage_block_size * self.compress_ratio,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            model_version="deepseek_v4",
            cache_dtype_str=self.cache_config.cache_dtype,
            scale_dim=1 if self.head_dim == 128 else 0,
            scale_dtype=torch.float
            if get_current_hardware_profile().supports(HardwareCapability.DSV4_COMPRESSED_CACHE)
            else torch.float16,
            **ratio_kwargs,
        )

    # 伪层占位 stub（无前向计算; cache 的读写全部由自定义算子完成）。
    def forward(self): ...

    def get_attn_backend(self):
        """返回 indexer cache 配套的 NPU 注意力后端类（C4/C128 分支）。"""
        # Keep these imports lazy to avoid a model-inspection circular import.
        # （懒导入避免循环依赖。）
        # 分支1: 压缩率 4 -> C4 indexer 后端（TopK 稀疏注意力）。
        if self.compress_ratio == 4:
            from vllm_ascend.attention.dsa_v1 import AscendDSAC4Backend

            return AscendDSAC4Backend
        # 分支2: 压缩率 128 -> C128 后端（长程压缩状态注意力）。
        if self.compress_ratio == 128:
            from vllm_ascend.attention.dsa_v1 import AscendDSAC128Backend

            return AscendDSAC128Backend
        # 其他压缩率不支持。
        raise ValueError(f"Unsupported DeepSeek V4 indexer compression ratio: {self.compress_ratio}")


@dataclass(frozen=True)
class AscendIndexerMetadata:
    """Indexer 的请求级元数据容器。

    仅含一个 compressor 字段: Indexer 的 k_cache 写入依赖 Compressor 产出的
    压缩 KV 与相关元数据（cos/sin、hadamard、slot_mapping 等）。
    @dataclass(frozen=True): 不可变数据类（自动 __init__/__repr__）。
    """

    compressor: AscendCompressorMetadata


@dataclass(frozen=True)
class IndexerOverlapPlan:
    """Main-attention compressor work scheduled around Indexer selection."""

    # 【中文】双流重叠计划: 主注意力层的 compressor 工作被拆成两段回调，
    # 安排在 Indexer 选择的空档执行（NPU 上 Cube/Vector 引擎并行）:
    # - compute_attention_compressed_kv(): 计算 attention 自己的压缩 KV
    #   （可闭包捕获输入，延迟到 aux_stream 空闲时执行）;
    # - scatter_attention_compressed_kv(kv, slots): 把结果 scatter 进
    #   attention 的压缩 cache;
    # - aux_stream: 辅助 NPU 流（None 表示串行模式）。
    # 语法点: typing.Callable[[], T] 表示“无参返回 T 的可调用对象”类型。
    compute_attention_compressed_kv: typing.Callable[[], tuple[torch.Tensor, torch.Tensor]]
    scatter_attention_compressed_kv: typing.Callable[[torch.Tensor, torch.Tensor], None]
    aux_stream: torch.npu.Stream | None = None


class AscendIndexerOps:
    """NPU 算子薄封装: 把 indexer 的量化/scatter/TopK 操作统一转发给
    DeviceOperator（按硬件 profile 选择具体实现）。

    设计意图: 模型层只面对稳定的方法名，硬件差异（不同 NPU 代际的
    算子名/参数）收敛在 DeviceOperator 内。
    """

    def __init__(self, index_topk: int) -> None:
        """初始化。Args: index_topk: 每个 query 选择的 token 数。"""
        from vllm_ascend.device.device_op import DeviceOperator

        self.device_operator = DeviceOperator
        self.index_topk = index_topk

    def unpack_dsa_indexer_kv_cache(self, kv_cache: tuple[torch.Tensor, ...]):
        """把 vLLM 传入的 kv_cache 张量组解包为 indexer 的四个 cache。

        Returns:
            (state_cache, key_cache, scale_cache, full_cache):
            压缩状态 cache、量化 k cache、反量化 scale cache、全精度备份
            cache（部分硬件需要）。
        """
        return self.device_operator.unpack_dsa_indexer_kv_cache(kv_cache)

    def quantize_query(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """量化 indexer query。

        Args:
            query: [num_tokens, n_heads, head_dim]（n_heads 通常为 1）。
        Returns:
            (query_quant, query_scale): INT8 量化结果 + 逐 token 反量化 scale。
        """
        return self.device_operator.indexer_quantize_query(query)

    def quantize_key_and_update_cache(
        self,
        key: torch.Tensor,
        key_cache: torch.Tensor,
        full_cache: torch.Tensor | None,
        slot_mapping: torch.Tensor,
    ):
        """量化 indexer key 并写入 k cache（part1 融合算子）。

        Args:
            key: [num_kv, head_dim] 新写入的压缩 key。
            key_cache: 量化 k cache 张量。
            full_cache: 全精度备份 cache（可空）。
            slot_mapping: 各 key 的写入槽位。
        Returns:
            (key_quant 写入结果, key_scale): A5 等硬件会把 scale 一并
            融合写入（返回 None）。
        """
        return self.device_operator.indexer_quant_scatter_part1(
            key,
            key_cache,
            full_cache,
            slot_mapping,
        )

    def update_scale_cache(
        self,
        key_scale: torch.Tensor,
        scale_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """把 key 的反量化 scale 写入 scale cache（part3 融合算子）。"""
        self.device_operator.dsa_indexer_scatter_scale_part3(
            key_scale,
            scale_cache,
            slot_mapping,
        )

    def select_topk(
        self,
        query: torch.Tensor,
        weights: torch.Tensor,
        query_scale: torch.Tensor,
        key_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        metadata: typing.Any,
    ) -> torch.Tensor:
        """闪电 TopK 选择: 量化 q x 量化 k cache -> 每 query 的 topk 索引。

        原理: torch.ops._C_ascend.npu_quant_lightning_indexer_v2 是融合
        INT8 闪电注意力核——内部完成 q·k^T 相关性（INT8 矩阵乘 + 反量化
        修正）、weights_proj 加权、逐 query 的 TopK，一次输出索引。
        mask_mode=3 表示“按 block_table 的块级因果/上下文掩码”。

        Args:
            query: 量化后的 q。
            weights: weights_proj 输出（逐 head 分数权重）。
            query_scale: q 的反量化 scale。
            key_cache: 量化 k cache。
            scale_cache: k 的反量化 scale cache。
            metadata: 请求元数据（含 qli_* 字段与 block_table）。
        Returns:
            topk_idxs: [num_tokens, 1, topk] 被选中 token 的索引。
        """
        # 等待预计算的 DeviceMetadata（异步流水）就绪。
        wait_for_device_metadata(DeviceMetadataStage.INDEXER, id(metadata.qli_metadata))
        topk_idxs, _ = torch.ops._C_ascend.npu_quant_lightning_indexer_v2(
            query=query,
            key=key_cache,
            # prepare_*: 按 NPU 布局要求整形权重/scale（如转置、拼接）。
            weights=self.device_operator.prepare_dsa_indexer_weights(weights),
            query_dequant_scale=self.device_operator.prepare_dsa_indexer_query_scale(query_scale),
            key_dequant_scale=self.device_operator.prepare_dsa_indexer_key_scale(scale_cache),
            topk=self.index_topk,
            quant_mode=self.device_operator.get_dsa_indexer_quant_mode(),
            # 变长序列边界（cumulative query lengths）与 KV 长度。
            cu_seqlens_q=metadata.qli_cu_seqlens_q,
            seqused_k=metadata.qli_seqused_k,
            cmp_residual_k=metadata.qli_cmp_residual_k,
            # 块表: 逻辑块 -> 物理块的映射（paged KV cache 寻址）。
            block_table=metadata.block_table,
            metadata=metadata.qli_metadata,
            # 布局约定: query 为 token-major（TND）; key 为 paged block-major
            #（PA_BBND，按物理块存储）。mask_mode=3: 块级上下文掩码。
            # cmp_ratio=4: 压缩率 4; return_value=0: 只返回索引不要分数。
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=4,
            return_value=0,
        )
        return topk_idxs

    def quantize_update_cache_and_select_topk(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None,
        weights: torch.Tensor,
        key_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        full_cache: torch.Tensor | None,
        slot_mapping: torch.Tensor,
        metadata: typing.Any,
    ) -> torch.Tensor:
        """融合路径: 量化 q/k + 写 cache + TopK 选择一步完成。

        原理: indexer_quant_scatter 先量化并 scatter q/k 进 cache（同时
        返回 q 的量化结果与 scale），再调用 select_topk 完成选择——
        相比分离路径少两次 Host-Device 交互（NPU 适配: 算子融合）。

        Returns:
            topk_idxs: [num_tokens, 1, topk]。
        """
        query, query_scale, _, _ = self.device_operator.indexer_quant_scatter(
            query,
            key,
            key_cache,
            scale_cache,
            full_cache,
            slot_mapping,
        )
        return self.select_topk(
            query,
            weights,
            query_scale,
            key_cache,
            scale_cache,
            metadata,
        )


class DeepseekV4Indexer(nn.Module):
    """DSA 稀疏注意力索引器（闪电 indexer）模块本体。

    组成:
      - wq_b: q_lora_rank -> n_heads*head_dim 的 query 低秩投影
        （qr 由主注意力的 wq_a 产出，与 MLA 共享 q 压缩路径）;
      - weights_proj: hidden -> n_heads 的逐头分数权重（TopK 排序用）;
      - k_cache: AscendDeepseekV4IndexerCache 伪层（量化 k + scale）;
      - compressor: 复用 Compressor（rotate=True）产出压缩 key;
      - topk_indices_buffer: 跨层共享的 TopK 结果缓冲（IndexCache 机制，
        见 forward 的 skip_topk）。
    输出 topk_indices 直接供 DSA attention 做“精确注意力仅作用于被选
    token”的稀疏计算。
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config | DeepseekV4Config,
        compress_ratio: int,
        skip_topk: bool,
        use_index_cache: bool,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
    ):
        """初始化索引器。

        Args:
            vllm_config: vLLM 全局配置。
            config: HF 模型配置。
            compress_ratio: 压缩率（本模块只在 =4 时创建）。
            skip_topk: True 时本层不计算 TopK，直接读 topk_indices_buffer
                中前一个 indexer 层的结果（IndexCache 跨层复用）。
            use_index_cache: 是否启用 IndexCache（把 TopK 写回 buffer）。
            quant_config: 量化配置。
            cache_config: cache 配置。
            prefix: 模块名前缀。
            topk_indices_buffer: [max_num_batched_tokens, topk] 共享缓冲。
        """
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        # indexer 头数与头维度（低维，如 1 x 128）。
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        # RoPE 维度与主注意力共享配置（qk_rope_head_dim）。
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        # softmax 缩放 = head_dim^-0.5。
        self.softmax_scale = self.head_dim**-0.5
        self.compress_ratio = compress_ratio
        self.skip_topk = skip_topk
        self.use_index_cache = use_index_cache

        # wq_b: 低秩 q 的升维投影（qr -> 每 head 的 index q）。
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
            return_bias=False,
        )

        # CVLinearWrapper: 包一层支持“先量化后 matmul”两段式调用的包装器
        #（双流模式下量化与 matmul 分流执行）。
        self.cv_wq_b = CVLinearWrapper(self.wq_b)
        self.topk_indices_buffer = topk_indices_buffer
        if self.skip_topk and self.topk_indices_buffer is None:
            raise ValueError("skip_topk requires topk_indices_buffer")
        # NPU 算子封装。
        self.ops = AscendIndexerOps(index_topk=self.index_topk)
        # weights_proj: 逐头分数权重投影（不量化，保持排序精度）。
        self.weights_proj = ReplicatedLinear(
            config.hidden_size,
            self.n_heads,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
            return_bias=False,
        )
        # indexer k cache 的 dtype（通常 int8）。
        k_dtype = kv_cache_dtype_str_to_dtype(
            self.vllm_config.attention_config.indexer_kv_dtype, vllm_config.model_config
        )

        if self.compress_ratio == 4:
            # TODO(cmq): change the dtype of cache
            # 【中文】c4 模式: 创建量化 k cache 伪层。
            self.k_cache = AscendDeepseekV4IndexerCache(
                head_dim=self.head_dim,
                dtype=k_dtype,
                prefix=f"{prefix}.k_cache",
                cache_config=cache_config,
                compress_ratio=self.compress_ratio,
            )
        self.compressor = None
        if self.compress_ratio > 1:
            # indexer 的压缩 key 也由 Compressor 产出（rotate=True:
            # 后接 Hadamard 旋转以均匀化量化误差）。
            self.compressor = Compressor(
                vllm_config,
                config,
                self.compress_ratio,
                head_dim=self.head_dim,
                rotate=True,
                quant_config=quant_config,
                cache_config=cache_config,
                prefix=f"{prefix}.compressor",
            )  # Compressor(4, 128)

    @staticmethod
    def _get_indexer_cache_metadata(
        metadata: AscendIndexerMetadata,
    ) -> tuple[typing.Any, torch.Tensor]:
        """从请求元数据中取出 cache 元数据与 Hadamard 矩阵。

        语法点: @staticmethod——静态方法（不接收 self/cls），可由类直接
        调用; 这里只是个纯函数式的取值助手。

        Returns:
            (cache_req_metadata, hadamard)。
        """
        cache_metadata = metadata.compressor.cache
        cache_req_metadata = cache_metadata.req_metadata
        hadamard = cache_metadata.hadamard
        assert cache_req_metadata is not None
        assert hadamard is not None
        return cache_req_metadata, hadamard

    def update_cache(
        self,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        metadata: AscendIndexerMetadata,
    ) -> None:
        """Update Indexer caches without projecting queries or selecting TopK."""
        # 【中文】只更新 cache、不做查询/TopK 的路径（prefill 后台/辅助
        # 更新场景使用）。
        # 空批次直接返回。
        if hidden_states.shape[0] == 0:
            return

        # 解包出四个 cache + Hadamard 矩阵。
        state_cache, key_cache, scale_cache, full_cache = self.ops.unpack_dsa_indexer_kv_cache(kv_cache)
        _, hadamard = self._get_indexer_cache_metadata(metadata)
        compressor = self.compressor
        assert compressor is not None
        # 步骤1: Compressor 产出压缩 key 与写入槽位。
        key, slot_mapping = compressor(
            hidden_states=hidden_states,
            state_cache=state_cache,
            metadata=metadata.compressor,
        )
        if key.shape[0] == 0:
            return
        if compressor.rotate:
            # 步骤2: Hadamard 旋转打散量化误差。
            key = rotate_activation(key, hadamard)
        # 步骤3: 量化 key 并写入 k cache（part1），必要时写 scale（part3）。
        _, key_scale = self.ops.quantize_key_and_update_cache(
            key,
            key_cache,
            full_cache,
            slot_mapping,
        )
        if key_scale is not None:
            self.ops.update_scale_cache(
                key_scale,
                scale_cache,
                slot_mapping,
            )

    def _get_cached_topk_indices(self, num_tokens: int, offset: int = 0) -> torch.Tensor:
        """从共享缓冲读取 TopK 索引（IndexCache 跨层复用路径）。

        Args:
            num_tokens: 需要的行数。
            offset: 缓冲内偏移（投机采样 draft 步可能从中间读起）。
        Returns:
            [num_tokens, 1, topk] 索引张量。
        """
        if self.topk_indices_buffer is None:
            raise RuntimeError("topk_indices_buffer is required to read cached TopK indices")
        topk_indices = self.topk_indices_buffer[offset : offset + num_tokens]
        if topk_indices.dim() == 2:
            # 补一个 head=1 维度，统一为 3D 形状。
            topk_indices = topk_indices.unsqueeze(1)
        return topk_indices

    def _update_cached_topk_indices(self, topk_indices: torch.Tensor, offset: int = 0) -> None:
        """把本层 TopK 结果写回共享缓冲（供后续 skip_topk 层复用）。"""
        if self.topk_indices_buffer is None:
            return
        num_tokens = topk_indices.shape[0]
        topk_tokens = topk_indices.shape[-1]
        topk_indices_to_cache = topk_indices
        # 目标缓冲切片 [offset:offset+num_tokens, :topk_tokens]。
        topk_indices_buffer = self.topk_indices_buffer[offset : offset + num_tokens, :topk_tokens]
        if topk_indices_to_cache.dim() == 3 and topk_indices_buffer.dim() == 2:
            # 3D（含 head 维）-> 2D 缓冲: head 维必须为 1 才能安全 squeeze。
            if topk_indices_to_cache.shape[1] != 1:
                raise ValueError("TopK indices must have a singleton head dimension")
            topk_indices_to_cache = topk_indices_to_cache.squeeze(1)
        # 原地拷贝进共享缓冲（copy_ 保持 dtype/device 一致）。
        topk_indices_buffer.copy_(topk_indices_to_cache)

    def forward(
        self,
        layer_name: str,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        metadata: AscendIndexerMetadata,
        overlap_plan: IndexerOverlapPlan,
        *,
        qr_pertoken_scale: torch.Tensor | None = None,
        write_cache: bool = True,
    ) -> torch.Tensor:
        """索引器前向: 为本层注意力产出 TopK token 索引。

        Args:
            layer_name: 层名（用于取该层专属的 RoPE 表）。
            hidden_states: [num_tokens, hidden_size] 当前层输入。
            qr: [num_tokens, q_lora_rank] 主注意力产出的低秩 q
                （MLA 的 q 压缩向量，与主注意力共享）。
            kv_cache: indexer 的 cache 张量组。
            metadata: 请求元数据。
            overlap_plan: 双流重叠计划（aux_stream 为 None 走串行路径）。
            qr_pertoken_scale: 预量化的 q scale（prolog 已量化时复用，
                keyword-only 参数）。
            write_cache: 是否把新 key 写入 cache（False 为只查询）。
        Returns:
            topk_indices: [num_tokens, 1, index_topk] 每 query 选中的
            历史 token 索引——供 DSA attention 稀疏计算。
        """
        num_tokens = hidden_states.shape[0]
        cache_metadata, _ = self._get_indexer_cache_metadata(metadata)
        # 该层专属的 RoPE cos/sin（截取本批 token 数）。
        cos = cache_metadata.cos[layer_name][:num_tokens]
        sin = cache_metadata.sin[layer_name][:num_tokens]
        aux_stream = overlap_plan.aux_stream
        if self.skip_topk:
            # 路径1（IndexCache）: 直接复用缓冲中前一个 indexer 层的结果，
            # 本层跳过全部 TopK 计算。
            topk_indices = self._get_cached_topk_indices(num_tokens)
        elif aux_stream is not None:
            # 路径2（双流重叠）: indexer 的量化/投影在 aux_stream 上跑，
            # 与主注意力的 compressor 工作（overlap_plan 的两个回调）并行。
            indexer_q = self._cv_compute_query_and_update_cache_multistream(
                hidden_states,
                qr,
                kv_cache,
                metadata,
                cos,
                sin,
                aux_stream,
                qr_pertoken_scale,
            )
            # 主流侧: 计算 attention 的压缩 KV。
            compressed_kv, compress_slot_mapping = overlap_plan.compute_attention_compressed_kv()
            # aux_stream 上做 weights_proj + TopK，同时主流 scatter 压缩 KV。
            topk_indices = self._select_topk_multistream(
                hidden_states,
                indexer_q,
                kv_cache,
                metadata,
                aux_stream,
                lambda: overlap_plan.scatter_attention_compressed_kv(
                    compressed_kv,
                    compress_slot_mapping,
                ),
            )
        else:
            # 路径3（串行）: 单流顺序执行全部步骤。
            topk_indices = self._select_topk_serial(
                hidden_states,
                qr,
                kv_cache,
                metadata,
                cos,
                sin,
                qr_pertoken_scale,
                write_cache=write_cache,
            )

        # 串行/复用路径下，attention 的 compressor 工作在本层 forward 末尾
        # 串行执行（双流路径已在上面并行做过）。
        if write_cache and (self.skip_topk or aux_stream is None):
            compressed_kv, compress_slot_mapping = overlap_plan.compute_attention_compressed_kv()
            overlap_plan.scatter_attention_compressed_kv(compressed_kv, compress_slot_mapping)

        # IndexCache 开启时把本层 TopK 写回共享缓冲。
        if self.use_index_cache:
            self._update_cached_topk_indices(topk_indices)
        return topk_indices

    def _cv_compute_query_and_update_cache_multistream(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        metadata: AscendIndexerMetadata,
        cos: torch.Tensor,
        sin: torch.Tensor,
        aux_stream: torch.npu.Stream,
        qr_pertoken_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the Indexer query and update its cache.

        The internal multistream strategy keeps the original four-part layout:
        - Part0: Main pre-compute qr_quant[V] + compressor[C/mixed] + kv_hadamard[V]
        - Part1: Main matmul[C] ∥ Aux kv_quant[V] + scatter_k_cache[AIV]
        - Part2: Main rope[V] (serial)
        - Part3: Main q_hadamard[C] ∥ Aux scatter_scale_cache[AIV]
        """
        # 【中文】多流版“计算 indexer query 并更新 cache”。四段式流水
        #（[V]=Vector/AIV 引擎, [C]=Cube 引擎——NPU 的两类计算单元）:
        #  Part0: 主流预计算 qr 量化[V] + compressor[C/混合] + kv Hadamard[V]
        #  Part1: 主流 matmul[C] ∥ 辅流 kv 量化[V] + k_cache scatter[AIV]
        #  Part2: 主流 RoPE[V]（串行）
        #  Part3: 主流 q Hadamard[C] ∥ 辅流 scale cache scatter[AIV]
        # 目的: 让 Cube 与 Vector 引擎同时忙碌，缩短关键路径。
        # 解包四个 cache + Hadamard。
        (indexer_state_cache, indexer_k_cache, indexer_scale_cache, indexer_full_cache) = (
            self.ops.unpack_dsa_indexer_kv_cache(kv_cache)
        )
        _, hadamard = self._get_indexer_cache_metadata(metadata)
        # 当前（主流）NPU 流。
        main_stream = torch.npu.current_stream()
        compressor = self.compressor
        assert compressor is not None

        # ===== Part0: Pre-compute on main =====
        # Reuse the prolog's pre-quantized qr when this layer's scheme
        # matches (W8A8 fused quant / MXFP8 split-quant).
        # 【中文】Part0: 主流预计算。若 prolog 已按相同量化方案预量化 qr
        #（W8A8 融合量化 / MXFP8 拆分量化），直接复用结果，省一次量化。
        if qr_pertoken_scale is not None and (_is_w8a8_dynamic(self.wq_b) or _is_mxfp8_dynamic(self.wq_b)):
            qr_quant_ready = qr
            qr_scale_ready = qr_pertoken_scale
        else:
            # 否则现量化（cv_wq_b 两段式接口的量化段）。
            qr_quant_ready, qr_scale_ready = self.cv_wq_b.quantize(qr)

        # Compressor 产出压缩 kv + 槽位（Cube/混合算子，主流执行）。
        kv, slot_mapping_indexer = compressor(
            hidden_states=hidden_states,
            state_cache=indexer_state_cache,
            metadata=metadata.compressor,
        )
        if kv.numel() == 0:
            kv = None
        elif compressor.rotate:
            # Hadamard 旋转打散量化误差。
            kv = rotate_activation(kv, hadamard)

        # ===== Part1: matmul[C] ∥ kv_quant[V] + scatter_k_cache[AIV] =====
        # Record event before main stream operations for aux_stream to wait
        # 【中文】Part1: 在主流记录事件，辅流据此同步（NPU 流间同步原语:
        # event = stream.record_event(); other.wait_event(event)）。
        e_kv_ready = main_stream.record_event()

        # Aux: kv_quant + scatter_k_cache (parallel with main matmul + rope)
        # 【中文】辅流: kv 量化 + 写 k_cache（与主流 matmul+rope 并行）。
        if kv is not None:
            # npu_stream_switch: 上下文管理器——在 with 块内把当前流切到
            # aux_stream（enabled=False 时保持原流），离开时自动切回。
            with npu_stream_switch(aux_stream, enabled=True):
                torch.npu.current_stream().wait_event(e_kv_ready)
                kv, kv_scale = self.ops.quantize_key_and_update_cache(
                    kv,
                    indexer_k_cache,
                    indexer_full_cache,
                    slot_mapping_indexer,
                )

        # Main: matmul q from qr (directly submit, V/C different engines dispatch naturally)
        # 【中文】主流: qr -> q 的 matmul（Cube 引擎; 与辅流的 Vector 引擎
        # 并行，由硬件自然调度）。
        if _is_w8a8_dynamic(self.wq_b) and qr_pertoken_scale is not None:
            # W8A8 + 已有逐 token scale: npu_quant_matmul 融合
            # “量化 matmul 反量化”一次完成。
            q = torch_npu.npu_quant_matmul(
                qr_quant_ready,
                self.wq_b.weight,
                self.wq_b.weight_scale,
                pertoken_scale=qr_scale_ready,
                bias=self.wq_b.bias,
                output_dtype=hidden_states.dtype,
            )
        else:
            # 其他情况: cv_wq_b 两段式接口的 matmul 段（内部按量化方案
            # 选择算子）。
            q = self.cv_wq_b.matmul(qr_quant_ready, qr_scale_ready)  # qr_matmul

        # 等待辅流 kv scatter 完成。
        if kv is not None:
            main_stream.wait_stream(aux_stream)

        # reshape 成多头形状。
        q = q.view(-1, self.n_heads, self.head_dim)

        # ===== Part2: rope[V] (main only) =====
        # 【中文】Part2: 主流 RoPE（只旋转尾部 rope_head_dim 维，
        # interleave 模式; 原地算子 inplace_partial_rotary_mul）。
        torch.ops._C_ascend.inplace_partial_rotary_mul(  # rope
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.head_dim - self.rope_head_dim, self.head_dim],
        )

        # Wait for aux_stream kv_scatter to complete before proceeding
        # 【中文】再次等待辅流（防御性同步: RoPE 与 scatter 都完成后再进
        # Part3）。
        if kv is not None:
            main_stream.wait_stream(aux_stream)

        e_rope_done = main_stream.record_event()

        # ===== Part3: q_hadamard[C] ∥ scatter_scale_cache[AIV] =====
        # Note: On A5, indexer_compress_epilog_v2 in Part1 handles both k_cache
        # and scale_cache in one fused operation, so Part3 is skipped
        # (kv_scale is None on A5 from indexer_quant_scatter_part1).
        # 【中文】Part3: 辅流写 scale cache，主流做 q 的 Hadamard——但 A5
        # 上 Part1 的融合 epilog 已一并写好 scale cache（kv_scale 为
        # None），本段自然跳过。
        if kv is not None and kv_scale is not None:
            with npu_stream_switch(aux_stream, enabled=True):
                torch.npu.current_stream().wait_event(e_rope_done)
                self.ops.update_scale_cache(
                    kv_scale,
                    indexer_scale_cache,
                    slot_mapping_indexer,
                )

        # Main: q_hadamard[Part1 - linear] (directly submit, C/AIV different engines dispatch naturally)
        # Part1: F.linear - parallel with aux_stream kv_scatter
        # 【中文】主流: q Hadamard 第一步（F.linear，与辅流 scatter 并行）。
        hidden_size = q.size(-1)
        q_linear, q_shape, q_dim = hadamard_linear(q, hadamard)

        if kv is not None:
            main_stream.wait_stream(aux_stream)

        # Main: q_hadamard[Part2 - scale] (after aux_stream completes)
        # Part2: scale * reshape - dot multiplication
        # 【中文】主流: q Hadamard 第二步（缩放+还原形状; 等待辅流完成后
        # 执行，保证引擎占用错峰）。
        q = hadamard_scale(q_linear, q_shape, q_dim, scale=hidden_size**-0.5)

        return q

    def _select_topk_multistream(
        self,
        hidden_states: torch.Tensor,
        indexer_q: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        metadata: AscendIndexerMetadata,
        aux_stream: torch.npu.Stream,
        scatter_attention_compressed_kv: typing.Callable[[], None],
    ) -> torch.Tensor:
        """Overlap Indexer selection inputs with caller-provided main-stream work."""
        # 【中文】多流版 TopK 选择: 把 weights_proj（Cube）放到辅流，
        # 主流同时执行调用方注入的 scatter_attention_compressed_kv
        #（attention 压缩 KV 的写入），最后主流做 TopK。
        main_stream = torch.npu.current_stream()
        # 记录起点事件，辅流等待后再开工（保证读到最新 hidden_states）。
        weights_proj_start = main_stream.record_event()
        with npu_stream_switch(aux_stream, enabled=True):
            torch.npu.current_stream().wait_event(weights_proj_start)
            # 辅流: weights_proj（hidden -> 每头分数权重）。
            weights_proj_output = self.weights_proj(hidden_states)
            weights_proj_done = torch.npu.current_stream().record_event()

        # 主流: q 量化（Vector 算子，可与辅流的 Cube 并行）。
        q_quant, q_scale = self.ops.quantize_query(indexer_q)
        # Enqueue only independent Vector/AIV work on the current main stream;
        # do not switch streams or launch Cube work that would contend with the
        # auxiliary weights projection.
        # 【中文】主流只入队与辅流无依赖冲突的 Vector/AIV 工作
        #（attention 压缩 KV 的 scatter），不启动会与辅流 weights_proj
        # 抢 Cube 引擎的算子。
        scatter_attention_compressed_kv()
        # 等辅流 weights_proj 完成后再做 TopK（TopK 需要其输出）。
        main_stream.wait_event(weights_proj_done)

        # 解包 k/scale cache。
        (_, indexer_k_cache, indexer_scale_cache, _) = self.ops.unpack_dsa_indexer_kv_cache(kv_cache)
        cache_metadata, _ = self._get_indexer_cache_metadata(metadata)
        # 分数权重 = weights_proj 输出 x (softmax_scale * n_heads^-0.5):
        # 两个缩放因子——前者是注意力 softmax 的常规缩放，后者补偿多头
        # 求和后的幅度（闪电核内部按多头聚合分数）。
        weights = weights_proj_output * (self.softmax_scale * self.n_heads**-0.5)
        # 主流: 闪电 TopK 选择。
        return self.ops.select_topk(
            q_quant,
            weights,
            q_scale,
            indexer_k_cache,
            indexer_scale_cache,
            cache_metadata,
        )

    def _indexer_qkv_prepare(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        metadata: AscendIndexerMetadata,
        cos: torch.Tensor,
        sin: torch.Tensor,
        qr_pertoken_scale: torch.Tensor | None = None,
        write_cache: bool = True,
    ):
        """串行路径的 q/kv 预处理（投影 + RoPE + Hadamard + 压缩）。

        Returns:
            七元组 (q, kv, k_cache, scale_cache, full_cache, cache_metadata,
            slot_mapping): 处理好的 query、可写入的压缩 key、各 cache 与
            元数据。write_cache=False 时 kv/slot_mapping 为 None。
        """
        (indexer_state_cache, indexer_k_cache, indexer_scale_cache, indexer_full_cache) = (
            self.ops.unpack_dsa_indexer_kv_cache(kv_cache)
        )
        cache_metadata, hadamard = self._get_indexer_cache_metadata(metadata)
        compressor = self.compressor
        assert compressor is not None

        # 步骤1: q 投影 qr -> q。W8A8 动态量化 + 已有逐 token scale 且
        # 硬件不支持 FP8 attention 时，用融合的 npu_quant_matmul
        #（量化 matmul 反量化一步完成; NPU 适配分支）。
        if (
            _is_w8a8_dynamic(self.wq_b)
            and qr_pertoken_scale is not None
            and not get_current_hardware_profile().supports(HardwareCapability.FP8_ATTENTION)
        ):
            q = torch_npu.npu_quant_matmul(
                qr,
                self.wq_b.weight,
                self.wq_b.weight_scale,
                pertoken_scale=qr_pertoken_scale,
                bias=self.wq_b.bias,
                output_dtype=x.dtype,
            )
        else:
            # 普通路径: 直接线性层（内部按量化方案处理）。
            q = self.wq_b(qr)
        # reshape 成 [T, N, D] 多头形状。
        q = q.view(-1, self.n_heads, self.head_dim)  # [T, N, D]

        # 步骤2: q 的部分 RoPE——只旋转尾部 rope_head_dim 维
        #（与 MLA 的 RoPE 解耦一致），interleave 模式，原地算子。
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.head_dim - self.rope_head_dim, self.head_dim],
        )

        # 步骤3: q 的 Hadamard 旋转（与 key 的旋转同一矩阵，保证 q/k 在
        # 同一旋转域中做内积）。
        q = rotate_activation(q, hadamard)
        kv = None
        indexer_slot_mapping = None
        if write_cache:
            # 步骤4: 需要写 cache 时——Compressor 产出压缩 key + Hadamard。
            kv, indexer_slot_mapping = compressor(
                hidden_states=x,
                state_cache=indexer_state_cache,
                metadata=metadata.compressor,
            )
            if kv.numel() == 0:
                kv = None
            elif compressor.rotate:
                kv = rotate_activation(kv, hadamard)

        return (
            q,
            kv,
            indexer_k_cache,
            indexer_scale_cache,
            indexer_full_cache,
            cache_metadata,
            indexer_slot_mapping,
        )

    def _select_topk_serial(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        metadata: AscendIndexerMetadata,
        cos: torch.Tensor,
        sin: torch.Tensor,
        qr_pertoken_scale: torch.Tensor | None = None,
        write_cache: bool = True,
    ):
        """串行路径: 预处理 -> 分数权重 -> 量化写 cache + TopK（或只查询）。

        Args: 见 _indexer_qkv_prepare 与 forward。
        Returns:
            topk_indices: [num_tokens, 1, index_topk]。
        """
        # 步骤1: q/kv 预处理。
        q, kv, ik, isc, ifc, cache_metadata, indexer_slot_mapping = self._indexer_qkv_prepare(
            x,
            qr,
            kv_cache,
            metadata,
            cos,
            sin,
            qr_pertoken_scale,
            write_cache=write_cache,
        )

        # 步骤2: 逐头分数权重（含两个缩放因子，见 _select_topk_multistream）。
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)

        if write_cache:
            # 步骤3a: 融合路径——量化 q/k + 写 cache + TopK 一次完成。
            return self.ops.quantize_update_cache_and_select_topk(
                q,
                kv,
                weights,
                ik,
                isc,
                ifc,
                indexer_slot_mapping,
                cache_metadata,
            )

        # 步骤3b: 只查询路径——量化 q 后直接 TopK（不写 cache）。
        q, q_scale = self.ops.quantize_query(q)
        return self.ops.select_topk(
            q,
            weights,
            q_scale,
            ik,
            isc,
            cache_metadata,
        )
