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
"""DeepSeek V4 KV 压缩器（Compressor）的昇腾 NPU 适配实现。

背景: DeepSeek V4 采用 DSA（DeepSeek Sparse Attention，稀疏注意力）。
注意力层的 KV 不再逐 token 完整保存，而是由 Compressor 把历史 KV 递归
“压缩”成低维状态:
  - compress_ratio=4  : 每 4 个 token 压成一份 compressed KV（细粒度），
                        配合 indexer 做 TopK token 选择，即 V3.2 式闪电稀疏注意力;
  - compress_ratio=128: 每 128 个 token 压成一份 state（粗粒度长程表示）。
Compressor 内部维护递归状态 cache（kv_state + score_state 两段拼接），
每批新 token 经 wkv/wgate 门控投影 + APE 绝对位置表 + RoPE 旋转更新状态，
同时输出“压缩后的 KV”写入注意力层/indexer 的压缩 KV cache。

本文件包含:
  - AscendCompressorStateCache: 压缩状态缓存（继承上游 vLLM 的
    CompressorStateCache），提供 NPU 版 KVCacheSpec 与注意力后端选择;
  - AscendCompressorMetadata  : 请求级元数据（压缩 KV cache 与 state cache
    元数据的打包容器）;
  - Compressor                : 压缩器模块本体。核心计算全部融合进 NPU 自定义
    C++ 算子 torch.ops._C_ascend.compressor —— 这是关键 NPU 适配点：避免
    Python 端多个小 kernel 拼接，减少 Host-Device 交互与中间显存。

在 vllm-ascend 插件架构中，本模块被 model.py（注意力层）与 indexer.py
（索引器）复用：注意力 swa/compressed cache 与 indexer 的 k_cache 均由
Compressor 生成压缩表示。
"""
import typing
from dataclasses import dataclass

import torch
from torch import nn
from transformers import DeepseekV2Config, DeepseekV3Config
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.models.deepseek_v4.compressor import CompressorStateCache
from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config
from vllm.v1.kv_cache_interface import KVCacheSpec

from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.worker.device_metadata import DeviceMetadataStage, wait_for_device_metadata


class AscendCompressorStateCache(CompressorStateCache):
    """压缩状态缓存（NPU 版），继承上游 vLLM 的 CompressorStateCache。

    作用: 以“伪层”（pseudo-layer）形式挂在模型里，让 vLLM v1 的 KV cache
    管理器为其分配一块 NPU 显存，存放 Compressor 的递归状态
    （kv_state + score_state，FP32）。重写 get_kv_cache_spec() 返回昇腾
    专有的 AscendSlidingWindowMLASpec，重写 get_attn_backend() 返回
    dsa_v1 中对应的注意力后端类。

    NPU 适配点: 块大小 block_size、页对齐字节数 page_size_padded 均取自
    DSV4_BLOCK_SIZES 查表（由硬件能力 profile 决定），与 NPU 内存对齐
    约束（如 32B/页对齐）匹配。
    """

    def __init__(
        self,
        state_dim: int,
        dtype: torch.dtype,
        compress_ratio: int,
        block_size: int,
        prefix: str,
    ):
        """初始化。

        Args:
            state_dim: 状态向量维度。compress_ratio=4 的 overlap 模式为
                2*coff*head_dim（kv_state+score_state 两段，coff=2）；
                compress_ratio=128 时为 2*head_dim。
            dtype: 缓存 dtype（当前固定 float32，与压缩 kernel 约定一致）。
            compress_ratio: 压缩率（4 或 128）。
            block_size: NPU cache 块大小（每块覆盖的 token 数，随压缩率不同）。
            prefix: 模块名前缀（vLLM 用它定位权重与 cache）。
        """
        super().__init__(state_dim, dtype, compress_ratio, prefix)
        self.compress_ratio = compress_ratio
        self.block_size = block_size

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        """返回该伪层的 KVCacheSpec（vLLM 据此统一规划显存与调度组）。

        原理: vLLM v1 启动时遍历模型中所有带 get_kv_cache_spec 的模块，
        按返回的 spec 把层分组、分配 paged cache。这里返回
        AscendSlidingWindowMLASpec —— 昇腾专有 spec，携带 NPU 侧块大小、
        页对齐字节数等布局信息。
        """
        # 延迟导入以避免循环依赖（attention layer 模块反向引用本文件）。
        from vllm_ascend.models.layer.attention.layer import dsv4_block_sizes

        # 步骤1: 查表得到 DSV4 各 cache 的 NPU 布局参数。
        # pads = [page_size_padded_t1, page_size_padded_t2]，分别是 c4 小状态
        # cache 与 c128/大状态 cache 的页对齐字节数。
        pads = dsv4_block_sizes(vllm_config)[vllm_config.cache_config.block_size][1]
        # 步骤2: 按 (state_dim, compress_ratio) 选择页对齐参数。
        # state_dim==2*256 且 ratio==4 对应 indexer 的小状态 cache（pads[0]），
        # 其余（attention 的 c4/c128 大状态）用 pads[1]。
        page_size_padded = pads[0] if self.state_dim == 2 * 256 and self.compress_ratio == 4 else pads[1]

        # 步骤3: 构造 spec。num_kv_heads=1 —— 压缩状态与 MLA 一样是
        # “单 KV 头”的潜在向量；head_size 即状态维度 state_dim。
        return AscendSlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            alignment=None,
            page_size_padded=page_size_padded,
        )

    # 伪层无前向计算: forward 是占位 stub（函数体 `...` 即 Ellipsis 表达式，
    # 是合法的空函数体）。真正的状态更新在 Compressor.forward 中经自定义
    # 算子完成，vLLM 不会调用本方法。
    def forward(self): ...

    def get_attn_backend(self):
        """返回与该 cache 配套的 NPU 注意力后端类。

        原理: vLLM v1 的混合注意力机制按“每组 KVCacheSpec”选择后端；
        这里按压缩率返回 dsa_v1 中对应的 DSA 后端（C4: 细粒度压缩+TopK
        稀疏；C128: 粗粒度长程状态注意力）。
        """
        # Keep these imports lazy to avoid a model-inspection circular import.
        # （保持懒导入，避免模型检查阶段的循环导入。）
        # 分支1: 压缩率 4 -> C4 状态后端。
        if self.compress_ratio == 4:
            from vllm_ascend.attention.dsa_v1 import AscendDSAC4StateBackend

            return AscendDSAC4StateBackend
        # 分支2: 压缩率 128 -> C128 状态后端。
        if self.compress_ratio == 128:
            from vllm_ascend.attention.dsa_v1 import AscendDSAC128StateBackend

            return AscendDSAC128StateBackend
        # 其他压缩率不支持，直接抛错快速失败。
        raise ValueError(f"Unsupported DeepSeek V4 state-cache compression ratio: {self.compress_ratio}")


@dataclass(frozen=True)
class AscendCompressorMetadata:
    """Request metadata for the compressed KV and compressor state caches."""

    # 【中文补充】请求级元数据容器: 把“注意力压缩 KV cache”与“压缩器状态
    # cache”两套请求元数据打包在一起，一次性传给 Compressor.forward。
    # 语法点:
    # - @dataclass: 自动生成 __init__/__repr__/__eq__;
    # - frozen=True: 实例不可变（防止运行中被意外修改），且可哈希、可作 dict key。
    # - typing.Any: 任意类型——具体类型由注意力后端在运行时决定，这里只做透传。
    # cache: 注意力层 compressed KV cache 的请求元数据。
    # state: Compressor 递归状态 cache 的请求元数据。
    cache: typing.Any
    state: typing.Any


class Compressor(nn.Module):
    """DeepSeek V4 KV 压缩器本体（NPU 自定义融合算子的封装层）。

    原理: 维护跨 token 递归更新的压缩状态 s_t。每批新 token 的
    hidden_states 经 wkv（值投影）/wgate（门投影）两个线性层得到候选
    写入向量，与 APE 绝对位置表相加后，按门控信号融入旧状态；状态尾部
    rope_head_dim 维再做 RoPE 旋转（rotary_mode=2，压缩专用旋转组），
    得到输出的 compressed KV：
      - compress_ratio=4（overlap 模式, coff=2）: 输出 kv+score 两段拼接，
        供 indexer 的闪电稀疏注意力做 TopK token 选择;
      - compress_ratio=128: 输出单一长程 state。
    核心计算全部融合进 torch.ops._C_ascend.compressor 单个 C++ 算子——
    关键 NPU 适配点（减少 kernel 数量与 Host-Device 交互）。
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config | DeepseekV4Config,
        compress_ratio: int = 4,
        head_dim: int = 512,
        rotate: bool = False,
        *,
        cache_config: CacheConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        """初始化压缩器。

        Args:
            vllm_config: vLLM 全局配置。
            config: HF 模型配置（类型注解 A | B | C 表示联合类型，兼容
                DeepSeek V2/V3/V4 三种配置）。
            compress_ratio: 压缩率，4（token 级）或 128（块级）。
            head_dim: 压缩头维度（如 512）。
            rotate: 输出是否再做 Hadamard 旋转（indexer 路径需要，
                用于打散 INT8 量化的通道相关性/误差）。
            cache_config: cache 配置。注意 * 之后为 keyword-only 参数，
                调用时必须按关键字传递（强制显式命名，避免位置歧义）。
            quant_config: 量化配置，可选。
            prefix: 模块名前缀。
        """
        super().__init__()
        # 延迟导入避免循环依赖: DSV4_BLOCK_SIZES 定义在 attention layer 模块。
        from vllm_ascend.models.layer.attention.layer import DSV4_BLOCK_SIZES

        self.vllm_config = vllm_config
        self.config = config
        self.dim = config.hidden_size
        self.head_dim = head_dim
        # RoPE 解耦（与 MLA 思想一致）: 压缩 KV 的尾部 rope_head_dim 维
        # 承载位置信息参与旋转，前面的 nope_head_dim 维不旋转。
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = head_dim - config.qk_rope_head_dim
        self.compress_ratio = compress_ratio
        # overlap 模式仅 c4 开启: coff = 1 + overlap 为输出段数系数
        # （c4 输出 kv+score 两段 => coff=2；c128 单段 => coff=1）。
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        self.norm_eps = config.rms_norm_eps
        self.coff = 1 + self.overlap

        # APE（Absolute Position Embedding）: [compress_ratio, coff*head_dim]
        # 可学习绝对位置表，按 token 在压缩组内的相对位置查表。
        self.ape = nn.Parameter(torch.empty(compress_ratio, self.coff * self.head_dim, dtype=torch.float32))
        # wkv: hidden -> coff*head_dim 的“值”投影。
        # NPU 适配: 若硬件 profile 支持 DSV4_COMPRESSED_CACHE（如 A5），
        # 压缩路径走专有内核，权重保持高精度（quant_config=None）；否则
        # 允许按全局量化配置量化。三元表达式写成多行仅是格式化效果。
        self.wkv = ReplicatedLinear(
            self.dim,
            self.coff * self.head_dim,
            bias=False,
            quant_config=None
            if get_current_hardware_profile().supports(HardwareCapability.DSV4_COMPRESSED_CACHE)
            else quant_config,
            prefix=f"{prefix}.wkv",
            return_bias=False,
        )
        # wgate: hidden -> coff*head_dim 的“门控”投影，与 wkv 结构相同。
        self.wgate = ReplicatedLinear(
            self.dim,
            self.coff * self.head_dim,
            bias=False,
            quant_config=None
            if get_current_hardware_profile().supports(HardwareCapability.DSV4_COMPRESSED_CACHE)
            else quant_config,
            prefix=f"{prefix}.wgate",
            return_bias=False,
        )

        # The custom compressor op consumes ND weights directly.
        # 【中文】NPU 适配: 自定义压缩算子直接消费 ND（行主序）布局权重，
        # 因此跳过 vllm-ascend 常规的 NZ（列主序）权重格式转换。
        self.wkv.skip_weight_nz_conversion = True
        self.wgate.skip_weight_nz_conversion = True

        # The DSV4 compressor kernel only accepts FP32 norm_weight.
        # 【中文】NPU 约束: DSV4 压缩内核只接受 FP32 的 norm 权重。
        self.norm = RMSNorm(self.head_dim, config.rms_norm_eps, dtype=torch.float32)

        # 状态 cache 的 dtype 固定 FP32（kernel 约定）。
        state_dtype = torch.float32
        # TODO(zyj): change following codes if block_size is configurable & refactor the magic numbers
        # 【中文】按压缩率创建状态 cache 伪层，block_size 查表获得
        # （DSV4_BLOCK_SIZES[block_size] = [mla, swa, c4_state, c128_state]）。
        if compress_ratio == 4:
            # c4: state_dim = 2*coff*head_dim，即 kv_state 与 score_state 两段。
            self.state_cache = AscendCompressorStateCache(
                state_dim=2 * self.coff * self.head_dim,  # kv_state + score_state
                dtype=state_dtype,
                compress_ratio=compress_ratio,
                prefix=f"{prefix}.state_cache",
                block_size=DSV4_BLOCK_SIZES[cache_config.block_size][0][2],
            )
        elif compress_ratio == 128:
            # c128: 单段 state，dim = 2*head_dim。
            self.state_cache = AscendCompressorStateCache(
                state_dim=2 * self.head_dim,  # kv_state + score_state
                dtype=state_dtype,
                compress_ratio=compress_ratio,
                prefix=f"{prefix}.state_cache",
                block_size=DSV4_BLOCK_SIZES[cache_config.block_size][0][3],
            )
        else:
            # 不支持的压缩率直接抛错快速失败。
            raise ValueError(
                f"Only support compress_ratio in [4, 128]. Got unsupported compress_ratio: {compress_ratio}"
            )

    def _compute_metadata(
        self,
        metadata: typing.Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取（或复用已预计算的）压缩所需的 RoPE 表与 slot_mapping。

        原理: 压缩 RoPE 的 cos/sin 表与状态 cache 的 slot_mapping 依赖
        请求级信息（位置、cache 布局），既可能由调度器在 CPU 侧预先算好
        （异步流水，DeviceMetadata 机制），也可在此现算。优先取预计算结果
        以缩短 NPU 关键路径。

        Args:
            metadata: 注意力后端产生的请求元数据（含 compressor_metadata）。

        Returns:
            (compress_cos, compress_sin, slot_mapping):
            cos/sin 为压缩 RoPE 角度表；slot_mapping 为本批 token 在状态
            cache 中的写入槽位索引。
        """
        # Imported lazily to avoid a circular import at module load time.
        # （懒导入，避免模块加载期的循环依赖。）
        from vllm_ascend.attention.dsa_v1 import get_or_compute_compressor_metadata

        # 步骤1: getattr(metadata, "compressor_metadata", None) 带默认值的
        # 属性读取——属性不存在时返回 None 而非抛 AttributeError。
        precomputed = getattr(metadata, "compressor_metadata", None)
        if precomputed is not None:
            # 分支A: 已有预计算结果。等待对应的 DeviceMetadata 就绪
            # （异步计算可能仍在另一条流/主机线程上进行）。
            group_id = metadata.compressor_metadata_group_id
            assert group_id is not None
            wait_for_device_metadata(DeviceMetadataStage.COMPRESSOR, group_id)
            return precomputed

        # 分支B: 无预计算结果，现场计算。
        return get_or_compute_compressor_metadata(metadata, self.compress_ratio, self.vllm_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        state_cache: torch.Tensor,
        metadata: AscendCompressorMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """压缩前向: 更新递归状态并产出 compressed KV。

        Args:
            hidden_states: [num_tokens, hidden_size] 本批 token 的隐状态。
            state_cache: 状态 cache 张量（含 kv_state+score_state，
                形状由 AscendCompressorStateCache 的 spec 决定）。
            metadata: AscendCompressorMetadata，含压缩 KV cache 与状态
                cache 两套请求元数据。

        Returns:
            (compressed_kv, slot_mapping):
            compressed_kv: [num_tokens/compress_ratio, coff*head_dim] 压缩 KV;
            slot_mapping: 压缩 KV 在 cache 中的写入槽位（供调用方 scatter）。
        """
        # 步骤1: 取出两套请求元数据（注意力压缩 KV cache 与状态 cache）。
        compressor_metadata = metadata.cache.req_metadata
        state_metadata = metadata.state.req_metadata
        assert compressor_metadata is not None
        assert state_metadata is not None
        # 步骤2: 获取（或复用预计算的）压缩 RoPE 表与状态写入槽位。
        compress_cos, compress_sin, slot_mapping = self._compute_metadata(compressor_metadata)
        # 步骤3: 调用 NPU 自定义融合算子完成全部压缩计算:
        #   值/门投影(wkv/wgate) + APE + 状态递归更新 + RMSNorm + RoPE 旋转
        #   + 状态 scatter 回写。关键参数:
        #   - state_cache.squeeze(-2): 去掉多余的头维（单 KV 头）;
        #   - rotary_mode=2: 压缩专用 RoPE 旋转模式;
        #   - cache_mode=1: 状态 cache 的读写模式;
        #   - coff=2(overlap)/1: 输出段数;
        #   - cu_seqlens/start_pos: 变长序列边界与起始位置（前缀感知）。
        compressed_kv = torch.ops._C_ascend.compressor(
            hidden_states,
            self.wkv.weight,
            self.wgate.weight,
            state_cache.squeeze(-2),
            self.ape,
            self.norm.weight,
            compress_sin.view(-1, compress_sin.shape[-1]),
            compress_cos.view(-1, compress_cos.shape[-1]),
            state_block_table=state_metadata.block_table,
            cu_seqlens=compressor_metadata.query_start_loc,
            seqused=None,
            start_pos=compressor_metadata.start_pos,
            rope_head_dim=self.rope_head_dim,
            cmp_ratio=self.compress_ratio,
            coff=2 if self.overlap else 1,
            norm_eps=self.norm_eps,
            rotary_mode=2,
            cache_mode=1,
        )
        # 步骤4: 返回压缩 KV 与其槽位映射，调用方据此写入注意力压缩 cache。
        return compressed_kv, slot_mapping
