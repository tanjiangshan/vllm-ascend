# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA layer with separate convolutions and a bounded safe gate.

GLM-5.3-Flash 的 KDA（Kimi Delta Attention）线性注意力层：
分离式 q/k/v 因果卷积 + 有界安全门控（bounded safe gate）。

KDA 原理（Gated DeltaNet 类线性注意力）：
  - 与标准 softmax 注意力不同，线性注意力用固定大小的循环状态
    S_t = alpha_t * S_{t-1} + beta_t * k_t v_t^T 显式累积历史，
    复杂度 O(1)（每 token），不随序列长度增长，也无需 KV cache。
  - 门控 g_t（forget gate，遗忘门）控制历史信息的衰减率；
    GLM 用"有界 sigmoid" gate = lower_bound + (1-lower_bound)*sigmoid(x)
    取代 GDN 默认的无界 softplus，保证数值稳定（safe gate）。
  - beta_t（插入强度/beta 门）控制新信息的写入强度，通常 sigmoid(b)。
  - 输出对状态做 q^T S 归一化，再经 o_norm（RMSNorm+sigmoid 门控）。

本文件的 NPU 适配点：
  - 前向通过 vllm_ascend.ops 中的 AscendC 自定义算子完成：
    causal_conv1d（短卷积）、run_chunk_kda（prefill 分块）、
    run_recurrent_kda（decode 逐 token 递归）；
  - q/k/v 三个卷积合并为一个 GEMM+conv（权写在首次前向时缓存）；
  - MTP 投机解码：spec token 与非 spec token 分流，draft-verify
    用 num_accepted_tokens 做拒绝采样回滚。
"""

import torch
from torch import nn
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import divide
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.utils import set_weight_attrs

# FusedRMSNormGated is a CustomOp, so the Ascend implementation is picked up
# through the OOT registration in `vllm_ascend.utils` rather than by import.
# 语法点：FusedRMSNormGated 是 CustomOp——Ascend 实现通过 vllm_ascend.utils
# 中的 OOT（out-of-tree）注册机制被选中，而非通过本 import。
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm_ascend.models.glm5next.config import Glm5NextConfig
from vllm_ascend.models.glm5next.ops.causal_conv1d import causal_conv1d
from vllm_ascend.models.glm5next.ops.kda import KDA_MAX_RECURRENT_TOKENS, chunk_kda, recurrent_kda
from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionBackend


class _Glm5NextMergedColumnParallelLinear(MergedColumnParallelLinear):
    """Merged projection with multiple replicated output shards.

    Extends K3's ``_KimiGDNMergedColumnParallelLinear`` to support two
    replicated shards (f_a, g_a) instead of one. Pre-multiplies each
    replicated entry's output_size by tp_size so the per-rank shard
    divides back to the full size, and forces tp_rank=0 during weight
    loading for replicated shards.

    支持多个"复制型"输出分片（replicated shards）的合并投影层。

    继承上游 vLLM 的 MergedColumnParallelLinear（把多个 ColumnParallel
    投影合并为一个 GEMM）。扩展 Kimi K3 的 _KimiGDNMergedColumnParallelLinear，
    支持两个复制分片（f_a, g_a）而非一个。

    原理：普通分片按 TP 切分（每卡拿 1/tp）；复制分片每卡都要完整权重。
    这里把复制分片的 output_size 预乘 tp_size，使"每卡份额 = 完整大小"
    再除回 tp 后仍是全量；同时加载权重时强制 tp_rank=0，保证每卡加载
    完整复制分片。
    """

    # Owned by the base class; declared here so the temporary override in the
    # weight loaders below does not read the attribute before its type is known.
    # tp_rank 由基类持有；这里声明类型注解，防止下方 weight_loader 中的
    # 临时覆盖在类型未知时被读取（类型注解先行声明）。
    tp_rank: int

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        replicated_shard_ids: tuple[int, ...],
        tp_size: int,
        **kwargs,
    ) -> None:
        """初始化。

        参数：
            input_size: 输入维度。
            output_sizes: 各分片输出维度列表（会被原地复制修改）。
            replicated_shard_ids: 复制型分片的下标集合。
            tp_size: 张量并行度。
        """
        self.replicated_shard_ids = set(replicated_shard_ids)
        # 复制分片宽度预乘 tp_size（copy 避免改动调用方列表）。
        output_sizes = output_sizes.copy()
        for sid in self.replicated_shard_ids:
            output_sizes[sid] *= tp_size
        super().__init__(input_size, output_sizes, **kwargs)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        """权重加载（v1 接口）：复制分片临时置 tp_rank=0 再调基类。

        语法点：try/finally 保证无论加载成败都恢复原 tp_rank。
        """
        tp_rank = self.tp_rank
        param_tp_rank = getattr(param, "tp_rank", None)
        if loaded_shard_id in self.replicated_shard_ids:
            self.tp_rank = 0
            if param_tp_rank is not None:
                param.tp_rank = 0
        try:
            super().weight_loader(param, loaded_weight, loaded_shard_id)
        finally:
            self.tp_rank = tp_rank
            if param_tp_rank is not None:
                param.tp_rank = param_tp_rank

    def weight_loader_v2(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        """权重加载（v2 接口）：与 v1 相同的复制分片处理。"""
        tp_rank = self.tp_rank
        param_tp_rank = getattr(param, "tp_rank", None)
        if loaded_shard_id in self.replicated_shard_ids:
            self.tp_rank = 0
            if param_tp_rank is not None:
                param.tp_rank = 0
        try:
            super().weight_loader_v2(param, loaded_weight, loaded_shard_id)
        finally:
            self.tp_rank = tp_rank
            if param_tp_rank is not None:
                param.tp_rank = param_tp_rank


class Glm5NextLinearAttention(GatedDeltaNetAttention):
    """GLM-5.3-Flash 的 KDA 线性注意力层（Gated DeltaNet 变体）。

    继承上游 vLLM 的 GatedDeltaNetAttention（GDN 基类），复用其状态
    管理/元数据管道；本类负责 GLM 特定的投影结构、有界门控与
    AscendC 算子调度。

    层结构（张量流）：
      hidden [N, H]
        -> in_proj_qkvbfg_a（单 GEMM 出 q|k|v|b|f_a|g_a 六段）
        -> q/k/v 分别过短因果卷积（causal_conv1d，kernel=4）
        -> chunk_kda（prefill）/ recurrent_kda（decode/verify）更新循环状态
        -> o_norm（RMSNorm + sigmoid 门控 g2）
        -> o_proj 回 [N, H]
    """

    # 类级类型注解（语法点：仅声明类型，不赋值；属性在 __init__ 中创建）。
    head_dim: int
    num_heads: int
    conv_size: int

    def get_state_dtype(
        self,
    ) -> tuple[torch.dtype, torch.dtype]:
        """返回 (conv_state_dtype, recurrent_state_dtype)。

        委托 MambaStateDtypeCalculator：按模型 dtype 与
        mamba_cache_dtype 推导 KDA 两类状态缓存的 dtype。
        """
        if self.model_config is None or self.cache_config is None:
            raise ValueError("model_config and cache_config must be set")
        return MambaStateDtypeCalculator.kda_state_dtype(self.model_config.dtype, self.cache_config.mamba_cache_dtype)

    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """返回 (conv_state_shape, recurrent_state_shape)。

        conv_state 宽度必须包含 num_spec（投机 token 数）——使带
        num_accepted_tokens 的 AscendC 因果卷积能在 draft-verify token
        上滑动窗口而不越界读。与 qwen_gdn_linear_attn.get_state_shape
        保持一致。
        """
        # conv_state width must include num_spec so the spec-decode conv update
        # (AscendC causal-conv with num_accepted_tokens) can
        # slide the window across the draft-verify tokens without reading past
        # the allocated width. Matches qwen_gdn_linear_attn.get_state_shape.
        return MambaStateShapeCalculator.kda_state_shape(
            self.tp_size,
            self.num_heads,
            self.head_dim,
            conv_kernel_size=self.conv_size,
            num_spec=self.num_spec,
        )

    def __init__(
        self,
        config: Glm5NextConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        """初始化 KDA 层。

        参数：
            config: Glm5NextConfig（linear_* 字段配置 KDA 头）。
            vllm_config: vLLM 全局配置。
            prefix: 层名前缀（注册进静态前向上下文）。

        步骤：
            1. 临时清空 quant_config 调基类构造（KDA 投影保持 BF16，
               因为 fp8 checkpoint 不带它们的 scale）；
            2. 校验 AscendC KDA 硬约束（head_dim=128、BF16、卷积核宽、
               门控下界范围、投机 token 上限）；
            3. 构建融合投影 in_proj_qkvbfg_a 与各子投影；
            4. 注册进静态前向上下文。
        """
        # KDA projections remain BF16 because fp8 checkpoints omit their scales.
        # 步骤1: 临时关闭量化配置调用基类构造（try/finally 恢复），
        # 使 KDA 投影权重保持 BF16。
        saved_quant_config = vllm_config.quant_config
        try:
            vllm_config.quant_config = None
            super().__init__(config, vllm_config, prefix)
        finally:
            vllm_config.quant_config = saved_quant_config

        # 步骤2: AscendC 算子硬约束校验。
        if config.linear_head_dim != 128 or vllm_config.model_config.dtype != torch.bfloat16:
            raise ValueError("GLM AscendC KDA requires BF16 activations and head_dim=128.")
        num_spec = vllm_config.speculative_config.num_speculative_tokens if vllm_config.speculative_config else 0
        if num_spec + 1 > KDA_MAX_RECURRENT_TOKENS:
            raise ValueError("GLM AscendC KDA supports at most seven speculative tokens.")
        if not 2 <= config.linear_conv_kernel_dim <= 4:
            raise ValueError("GLM AscendC causal-conv requires a kernel width in [2, 4].")
        if num_spec and config.linear_conv_kernel_dim != 4:
            raise ValueError("GLM AscendC causal-conv requires kernel width=4 for MTP.")
        if not -5 <= config.linear_lower_bound < 0:
            raise ValueError("GLM AscendC KDA requires linear_lower_bound in [-5, 0).")
        self.head_dim = config.linear_head_dim
        self.num_heads = config.linear_num_heads
        self.conv_size = config.linear_conv_kernel_dim
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = divide(self.num_heads, self.tp_size)

        projection_size = self.head_dim * self.num_heads
        self.local_projection_size = divide(projection_size, self.tp_size)

        # Merge q, k, v, b, f_a, g_a projections into one GEMM (6→1 launches).
        # Order matches checkpoint's fused_qkvbfg_a_proj convention.
        # Shards 4 (f_a) and 5 (g_a) are replicated across TP ranks.
        # 步骤3a: 融合投影——把 q/k/v/b/f_a/g_a 六个投影合并为一个 GEMM
        # （6 次矩阵乘 -> 1 次）。分片顺序匹配 checkpoint 的
        # fused_qkvbfg_a_proj 约定；分片 4(f_a) 与 5(g_a) 跨 TP 复制。
        self.in_proj_qkvbfg_a = _Glm5NextMergedColumnParallelLinear(
            self.hidden_size,
            [
                projection_size,  # q (shard 0)
                projection_size,  # k (shard 1)
                projection_size,  # v (shard 2)
                self.num_heads,  # b (shard 3)
                self.head_dim,  # f_a (shard 4, replicated)
                self.head_dim,  # g_a (shard 5, replicated)
            ],
            replicated_shard_ids=(4, 5),
            tp_size=self.tp_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.in_proj_qkvbfg_a",
        )

        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.f_b_proj",
        )
        # dt_bias：时间步偏置（FP32），按头维度做 TP 分片（shard 0）。
        self.dt_bias = nn.Parameter(torch.empty(divide(projection_size, self.tp_size), dtype=torch.float32))

        # 语法点：set_weight_attrs 给参数挂载加载属性（此处指定分片加载器）。
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        # 步骤3b: q/k/v 各自的短因果卷积权重（FP32，实现为 1xconv_size 的
        # ColumnParallelLinear，随后把权重 unsqueeze 成卷积核形状）。
        self.q_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.q_conv1d",
        )
        self.k_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.k_conv1d",
        )
        self.v_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.v_conv1d",
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        # 步骤3c: 把 [out, conv_size] 的线性权重 unsqueeze 成 [out, 1, conv_size]
        # 以匹配 conv1d 权重形状。不能在 weight_loader 里做——
        # ColumnParallelLinear 已定义 weight_loader 且 set_weight_attrs
        # 不允许覆盖。
        self.q_conv1d.weight.data = self.q_conv1d.weight.data.unsqueeze(1)
        self.k_conv1d.weight.data = self.k_conv1d.weight.data.unsqueeze(1)
        self.v_conv1d.weight.data = self.v_conv1d.weight.data.unsqueeze(1)
        # Lazily-built merged q|k|v conv weight (built on first forward, after
        # weights are loaded). See _forward.
        # 合并 q|k|v 卷积权重：懒构建（首次前向、权重加载完成后拼接缓存）。
        self._merged_conv_weight: torch.Tensor | None = None

        # A_log：衰减率的对数参数（FP32，形状 [1,1,local_heads,1]），
        # 加载时按头维分片（shard 2）。
        self.A_log = nn.Parameter(torch.empty(1, 1, self.local_num_heads, 1, dtype=torch.float32))
        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(2)})

        # 步骤3d: g 门控两段式投影与输出头。
        # g_b_proj：g_a [N, head_dim] -> 每头门控 [N, heads*head_dim]；
        # o_norm：FusedRMSNormGated，输出侧带 sigmoid 激活门控；
        # o_proj：按行并行的输出投影（TP all-reduce）。
        self.g_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.g_b_proj",
        )
        self.o_norm = FusedRMSNormGated(self.head_dim, activation="sigmoid")
        self.o_proj = RowParallelLinear(
            projection_size,
            self.hidden_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # 步骤4: 注册进静态前向上下文（供注意力后端按层名查找状态缓存）。
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        # Checkpoints store A_log as 1-D; the model parameter is 4-D.
        # A_log 加载适配：checkpoint 存 1-D，模型参数是 4-D——
        # 闭包 _a_log_weight_loader 先 reshape 再按头分片加载。
        # 语法点：嵌套函数（闭包）捕获外层变量 param。
        def _a_log_weight_loader(param, loaded_weight):
            if loaded_weight.dim() == 1:
                loaded_weight = loaded_weight.view([1, 1, -1, 1])
            return sharded_weight_loader(2)(param, loaded_weight)

        self.A_log.weight_loader = _a_log_weight_loader

        # GLM-5.3-Flash uses a bounded sigmoid gate instead of the default
        # unbounded softplus gate.
        # 步骤5: 有界 sigmoid 门控下界（safe gate）：
        # g = lower_bound + (1 - lower_bound) * sigmoid(x)，
        # 替代 GDN 默认的无界 softplus 门。
        self.kda_lower_bound = config.linear_lower_bound
        # Process-global conv-state layout, resolved once here instead of on
        # every _forward call (it reads an env-derived flag each time).
        # 卷积状态布局（进程级，源自环境变量标志）只解析一次缓存，
        # 避免每次 _forward 都读环境标志。
        self._conv_state_dim_first = is_conv_state_dim_first()

    def get_attn_backend(self):
        """返回本层配套的注意力后端：AscendGDNAttentionBackend（Ascend 适配）。"""
        return AscendGDNAttentionBackend

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """KDA 层前向（投影部分；核心递归在 _forward 中）。

        参数：
            hidden_states: [num_tokens, hidden_size] 输入。
            positions: [num_tokens] token 位置（当前实现未直接使用，
                位置信息通过 GDN 元数据传递）。

        返回：
            [num_tokens, hidden_size] 输出。

        步骤：
            1. 单 GEMM 投影出 qkv|beta|f_a|g_a 六段；
            2. f_a -> f_b_proj 得 g1（状态输出门控），
               g_a -> g_b_proj 得 g2（最终输出门控）；
            3. _forward 执行卷积 + KDA 状态更新（写 core_attn_out）；
            4. o_norm 门控归一化 + o_proj 输出。
        """
        num_tokens = hidden_states.size(0)
        # One merged GEMM for q, k, v, b, f_a, g_a (replaces 6 separate GEMMs).
        # 步骤1: 单 GEMM 出全部投影段（替代 6 个独立 GEMM）。
        projected = self.in_proj_qkvbfg_a(hidden_states)[0]
        # 按宽度切分：qkv(3*local_proj) | beta(local_heads) | f_a | g_a。
        qkv, beta_raw, f_a, g_a = projected.split(
            [
                3 * self.local_projection_size,
                self.local_num_heads,
                self.head_dim,
                self.head_dim,
            ],
            dim=-1,
        )

        # Beta stays raw (bf16) here: the recurrent kernel sigmoids it in fp32
        # at load (SIGMOID_BETA), and only the chunked prefill path needs the
        # pre-computed fp32 sigmoid — computed lazily in _forward. Pure decode
        # / spec-verify steps then skip the separate sigmoid and its fp32
        # intermediate entirely.
        # 步骤2: beta 保持原始 bf16——递归内核会按 SIGMOID_BETA 模式在加载时
        # 用 fp32 sigmoid；只有分块 prefill 需要预计算 fp32 sigmoid（在
        # _forward 里惰性完成），纯 decode/spec-verify 步骤因此完全省去
        # 单独的 sigmoid 与 fp32 中间量。
        beta = beta_raw.unsqueeze(0)
        # g1: f_a 经 f_b_proj 得到"状态输出门控"（reshape 为
        # [1, N, local_heads, head_dim]）。
        g1 = self.f_b_proj(f_a)[0]
        g1 = g1.reshape(1, -1, self.local_num_heads, self.head_dim)

        # g2: g_a 经 g_b_proj 得到"最终输出门控"。
        g_proj_states = self.g_b_proj(g_a)[0]
        # Must stay 3D: rms_norm_gated reads H from g.shape[-2].
        # 必须保持 3D：rms_norm_gated 从 g.shape[-2] 读取头数 H。
        g2 = g_proj_states.reshape(-1, self.local_num_heads, self.head_dim)

        # core_attn_out: KDA 核心输出缓冲（_forward 就地写入）。
        core_attn_out = torch.empty(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        # Keep the layer's dispatch outside piecewise graph compilation.
        # 步骤3: _forward 被排除在分段图编译之外（eager 执行，见装饰器）。
        self._forward(
            qkv_proj_states=qkv,
            g1=g1,
            beta=beta,
            core_attn_out=core_attn_out,
        )
        # 步骤4: o_norm（RMSNorm + g2 的 sigmoid 门控）+ reshape + o_proj。
        core_attn_out = self.o_norm(core_attn_out, g2)
        core_attn_out = core_attn_out.reshape(core_attn_out.size(1), -1)
        return self.o_proj(core_attn_out)[0]

    @eager_break_during_capture
    def _forward(
        self,
        qkv_proj_states: torch.Tensor,
        g1: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        """KDA 核心前向：短卷积 + 状态递归更新（eager，不进编译图）。

        语法点：@eager_break_during_capture 装饰器使本方法在 CUDA/ACL
        图捕获时强制退回 eager 执行（内部有动态控制流与自定义算子）。

        参数：
            qkv_proj_states: [N, 3*local_projection] 拼接的 q|k|v 投影。
            g1: [1, N, local_heads, head_dim] 状态输出门控。
            beta: [1, N, local_heads] 原始 beta 门（未过 sigmoid）。
            core_attn_out: [1, N, local_heads, head_dim] 输出缓冲（就地写）。

        执行路径（按批次组成分流）：
            - 无元数据：清零返回（profile/warmup）；
            - spec（draft-verify）token：causal_conv1d(run_mode=1,
              num_accepted_tokens=...) + recurrent_kda（带拒绝回滚）；
            - 非 spec prefill：causal_conv1d(run_mode=0, initial_state_mode)
              + chunk_kda（分块并行）；
            - 非 spec decode：causal_conv1d(run_mode=1) + recurrent_kda。
        """
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        # 步骤1: 元数据获取与空跑保护。
        if attn_metadata_raw is None:
            core_attn_out.zero_()
            return

        assert isinstance(attn_metadata_raw, dict)
        # 按本层 prefix 取出"窄化"的 GDN 元数据。
        attn_metadata_narrowed = attn_metadata_raw.get(self.prefix)
        if attn_metadata_narrowed is None:
            # Profile/warmup dummy runs may omit mamba-family metadata.
            # profile/warmup 假跑可能不带 mamba 族元数据。
            core_attn_out.zero_()
            return
        assert isinstance(attn_metadata_narrowed, GDNAttentionMetadata)
        # 非 spec（常规 prefill/decode）元数据。
        non_spec_query_start_loc = attn_metadata_narrowed.non_spec_query_start_loc
        non_spec_state_indices_tensor = attn_metadata_narrowed.non_spec_state_indices_tensor  # noqa: E501
        num_actual_tokens = attn_metadata_narrowed.num_actual_tokens
        # Spec-decode metadata (all None when speculative decoding is disabled).
        # 投机解码元数据（未启用投机解码时全为 None）。
        spec_sequence_masks = attn_metadata_narrowed.spec_sequence_masks
        spec_query_start_loc = attn_metadata_narrowed.spec_query_start_loc
        spec_state_indices_tensor = attn_metadata_narrowed.spec_state_indices_tensor
        spec_token_indx = attn_metadata_narrowed.spec_token_indx
        non_spec_token_indx = attn_metadata_narrowed.non_spec_token_indx
        num_accepted_tokens = attn_metadata_narrowed.num_accepted_tokens
        num_spec_decodes = attn_metadata_narrowed.num_spec_decodes
        use_spec = spec_sequence_masks is not None and num_spec_decodes > 0
        # Safe-gate checkpoints use the bounded sigmoid variant.
        # 有界 sigmoid 门控下界（safe gate checkpoint）。
        lower_bound = self.kda_lower_bound
        constant_caches = self.kv_cache

        # 步骤2: 裁掉 batch 填充（只保留实际 token）。
        qkv_proj_states = qkv_proj_states[:num_actual_tokens]
        g1 = g1[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]

        # 步骤3: 取出两类持久状态缓存 (conv_state, recurrent_state)。
        (conv_state, recurrent_state) = constant_caches
        # AscendC consumes [cache, state_len, dim]. Preserve the original storage.
        # Layout is process-global and resolved once at init (see __init__).
        # AscendC 算子消费 [cache, state_len, dim] 布局；若全局布局是
        # dim-first 则转置视图（保持原存储不拷贝）。
        if self._conv_state_dim_first:
            conv_state = conv_state.transpose(-1, -2)

        # One merged short-conv over q|k|v instead of three separate calls. The
        # 1D conv is independent per channel, so concatenating q/k/v along the
        # channel dim preserves the independent q/k/v convolutions.
        # The merged weight is q|k|v conv weights concatenated;
        # built once and cached (params are fixed after load). conv_state is
        # already stored as the merged q|k|v state, so it is used directly.
        # 步骤4: 合并 q|k|v 的短因果卷积。1D 卷积逐通道独立，沿通道维拼接
        # q/k/v 等价于各自独立卷积。合并权重 = 三个卷积权重的拼接，
        # 首次调用时构建并缓存（加载后参数固定）；conv_state 本身就按
        # 合并后的 q|k|v 状态存储，直接使用。
        if self._merged_conv_weight is None:

            def _w(m):
                return m.weight.view(m.weight.size(0), m.weight.size(2))

            self._merged_conv_weight = (
                torch.cat(
                    [_w(self.q_conv1d), _w(self.k_conv1d), _w(self.v_conv1d)],
                    dim=0,
                )
                .transpose(0, 1)
                .to(dtype=qkv_proj_states.dtype)
                .contiguous()
            )
        conv_weights = self._merged_conv_weight

        # Split projections / gating into spec (draft-verify) and non-spec token
        # groups when speculative decoding is active. Spec tokens carry
        # num_spec+1 recurrent-state columns each and are advanced with
        # num_accepted_tokens for rejection-sampling rollback; non-spec tokens
        # are one-per-request. Mirrors olmo_gdn_linear_attn.py. Projections are
        # [n, *] (token dim 0); g1/beta are [1, n, h, d] (token dim 1).
        # 步骤5: 投机解码激活时，把投影/门控切分成 spec（draft-verify）与
        # 非 spec 两组。spec token 每个携带 num_spec+1 个状态列，用
        # num_accepted_tokens 做拒绝采样回滚；非 spec token 每请求一个。
        # 投影是 [n, *]（token 在 0 维）；g1/beta 是 [1, n, h, d]（token 在 1 维）。
        if use_spec:
            # In a pure spec-verify step (no non-spec tokens) the metadata
            # builder sets spec_token_indx = arange(num_actual_tokens), making
            # the index_select calls below identity copies. Skip them on this
            # steady-state decode hot path. The outputs alias the inputs here;
            # the downstream conv/recurrent kernels read them without mutating
            # in place, so the aliasing is safe.
            # 纯 spec-verify 步（无非 spec token）时，元数据构建器会把
            # spec_token_indx 设为 arange(num_actual_tokens)，使下面的
            # index_select 沦为恒等拷贝——直接跳过以省去这条稳态解码热路径
            # 上的开销。此时输出与输入别名；下游卷积/递归内核只读不改，
            # 别名安全。
            if non_spec_token_indx is None or non_spec_token_indx.numel() == 0:
                qkv_spec = qkv_proj_states
                g1_spec = g1
                beta_spec = beta
            else:
                # index_select 按索引抽出 spec token 行。
                qkv_spec = qkv_proj_states.index_select(0, spec_token_indx)
                g1_spec = g1.index_select(1, spec_token_indx)
                beta_spec = beta.index_select(1, spec_token_indx)
            if non_spec_token_indx is not None and non_spec_token_indx.numel() > 0:
                qkv_ns = qkv_proj_states.index_select(0, non_spec_token_indx)
                g1_ns = g1.index_select(1, non_spec_token_indx)
                beta_ns = beta.index_select(1, non_spec_token_indx)
            else:
                qkv_ns = g1_ns = beta_ns = None
        else:
            # 非投机解码：全部 token 走非 spec 路径。
            qkv_spec = g1_spec = beta_spec = None
            qkv_ns, g1_ns, beta_ns = qkv_proj_states, g1, beta

        # --- causal conv1d: spec (draft-verify) path ---
        # 步骤6a: spec token 的因果卷积——run_mode=1（decode/verify 模式，
        # 读旧状态滑窗），num_accepted_tokens 供内核做拒绝回滚后的窗口重放。
        if use_spec:
            assert spec_state_indices_tensor is not None
            assert num_accepted_tokens is not None
            conv_meta = attn_metadata_narrowed.spec_decode_metadata.spec_causal_conv1d
            qkv_spec = causal_conv1d(
                qkv_spec,
                conv_weights,
                conv_state,
                conv_meta.query_start_loc,
                conv_meta.cache_indices,
                run_mode=1,
                num_accepted_tokens=conv_meta.num_accepted_tokens,
            )
            q_spec, k_spec, v_spec = qkv_spec.split(self.local_projection_size, dim=-1)

        # --- causal conv1d: non-spec path (prefill or plain decode) ---
        # 步骤6b: 非 spec token 的因果卷积。
        # prefill：run_mode=0（首 token 需初始化状态，initial_state_mode
        # 标记哪些请求有历史状态）；普通 decode：run_mode=1。
        q_ns = k_ns = v_ns = None
        if attn_metadata_narrowed.num_prefills > 0:
            assert qkv_ns is not None
            conv_meta = attn_metadata_narrowed.non_spec_prefill_metadata.causal_conv1d
            qkv_ns = causal_conv1d(
                qkv_ns,
                conv_weights,
                conv_state,
                conv_meta.query_start_loc,
                conv_meta.cache_indices,
                run_mode=0,
                initial_state_mode=conv_meta.initial_state_mode,
            )
            q_ns, k_ns, v_ns = qkv_ns.split(self.local_projection_size, dim=-1)
        elif attn_metadata_narrowed.num_decodes > 0:
            assert non_spec_state_indices_tensor is not None
            conv_meta = attn_metadata_narrowed.non_spec_decode_metadata.causal_conv1d
            qkv_ns = causal_conv1d(
                qkv_ns,
                conv_weights,
                conv_state,
                conv_meta.query_start_loc,
                conv_meta.cache_indices,
                run_mode=1,
            )
            q_ns, k_ns, v_ns = qkv_ns.split(self.local_projection_size, dim=-1)

        def rearrange(x):
            """把 [N, local_proj] 重排为 [1, N, local_heads, head_dim]。"""
            return x.reshape(1, -1, self.local_num_heads, self.head_dim)

        # Pure decode writes the whole captured buffer, including padding.
        # 纯 decode（无 spec、无 prefill）时内核直接写满整个捕获缓冲
        # （含填充），无需 zero 再部分拷贝。
        direct_output = (
            not use_spec and attn_metadata_narrowed.num_prefills == 0 and attn_metadata_narrowed.num_decodes > 0
        )
        if not direct_output:
            core_attn_out.zero_()
        # 步骤7a: spec token 的 KDA 递归更新（含拒绝采样回滚：
        # num_accepted_tokens 决定状态前进几步）。
        if use_spec:
            spec_output = recurrent_kda(
                rearrange(q_spec),
                rearrange(k_spec),
                rearrange(v_spec),
                g1_spec,
                beta_spec,
                recurrent_state,
                spec_query_start_loc[: num_spec_decodes + 1],
                spec_state_indices_tensor,
                self.A_log,
                self.dt_bias,
                lower_bound,
                num_accepted_tokens,
            )
            core_attn_out[0].index_copy_(0, spec_token_indx, spec_output[0])

        if q_ns is None:
            return
        # 步骤7b: 非 spec token 的 KDA 更新。先把卷积输出重排成 4D。
        q_ns, k_ns, v_ns = rearrange(q_ns), rearrange(k_ns), rearrange(v_ns)
        metadata = attn_metadata_narrowed
        # decode token 数（混合 batch 时 prefill 段后接 decode 段）。
        decode_tokens = metadata.num_decode_tokens if metadata.num_prefills > 0 else q_ns.shape[1]
        output = None
        # 步骤7b-i: decode token -> 递归 KDA（output_buffer 直写捕获缓冲，
        # 省一次拷贝）。
        if metadata.num_decodes > 0:
            output = recurrent_kda(
                q_ns[:, :decode_tokens],
                k_ns[:, :decode_tokens],
                v_ns[:, :decode_tokens],
                g1_ns[:, :decode_tokens],
                beta_ns[:, :decode_tokens],
                recurrent_state,
                non_spec_query_start_loc[: metadata.num_decodes + 1],
                non_spec_state_indices_tensor,
                self.A_log,
                self.dt_bias,
                lower_bound,
                output_buffer=core_attn_out if direct_output else None,
            )
        # 步骤7b-ii: prefill token -> 分块 KDA（chunked，读取初始状态、
        # 写回最终状态到持久 recurrent_state 缓存）。
        if metadata.num_prefills > 0:
            prefill_output = chunk_kda(
                q_ns[:, decode_tokens:],
                k_ns[:, decode_tokens:],
                v_ns[:, decode_tokens:],
                g1_ns[:, decode_tokens:],
                beta_ns[:, decode_tokens:],
                recurrent_state,
                metadata.prefill_state_indices,
                metadata.prefill_has_initial_state,
                metadata.non_spec_prefill_metadata.chunk,
                self.A_log,
                self.dt_bias,
                lower_bound,
            )
            output = prefill_output if output is None else torch.cat((output, prefill_output), dim=1)
        assert output is not None
        # 步骤8: 结果写回 core_attn_out。
        # direct_output：递归内核已直写缓冲，直接返回。
        if direct_output:
            return
        if use_spec:
            # spec 混合 batch：按 non_spec_token_indx 把非 spec 结果散回原位。
            core_attn_out[0].index_copy_(0, non_spec_token_indx, output[0])
        else:
            # 常规路径：顺序拷贝前 output.shape[1] 个 token。
            core_attn_out[0, : output.shape[1]].copy_(output[0])
