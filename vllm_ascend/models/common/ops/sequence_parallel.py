# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""序列并行（Sequence Parallelism, SP）辅助算子（NPU 适配版）。

背景与原理：
- 在纯张量并行（TP）中，注意力是"序列敏感"的：每个 TP rank 持有完整的
  序列（token 维不切分），只切分注意力头/权重维；这导致 LayerNorm、
  残差等操作的激活在所有 rank 上重复存储。
- 序列并行（本文件实现的风格，与 DeepSeek 所用 SP / DeepSpeed-Ulysses
  思路一致）在 TP 基础上进一步把序列（token）维度切分到各 TP rank：
  * 非注意力模块（MLP、LayerNorm 等）在各 rank 的"局部 token 分片"上
    计算，激活内存降为约 1/tp_size；
  * 进入注意力层前调用 sp_all_gather 沿 dim=0 把各 rank 分片拼接为
    完整序列（注意力需要看到全序列的 Q/K/V）；
  * 注意力计算结束后调用 sp_reduce_scatter 把各 rank 的部分和结果
    规约（求和）并重新散射为本 rank 的局部 token 分片，一次通信同时
    完成 all-reduce 与序列切分。
- 配套工具：
  * sp_shard —— 无通信地本地切出本 rank 的 token 分片（模型 forward
    入口处，各 rank 拿到相同完整输入后各自取走属于自己的部分）；
  * sp_padding_mask —— 生成与分片逐 token 对齐的 padding 掩码，标记
    因对齐 tp_size 而补齐的"假 token"，供后续算子跳过无效计算。

在 vllm-ascend 中的位置：
- 位于 vllm_ascend/models/common/ops，被 deepseek_v4 / deepseek_v41 /
  kimi_k3 / glm5next 等模型的 forward 路径调用，也供
  vllm_ascend/attention/context_parallel/dsa_cp.py（DSA 上下文并行）复用。

NPU 适配点：
- 优先尝试 TP 组 DeviceCommunicator 上注册的自定义集合通信算子
  （custom_all_gather / custom_reduce_scatter，NPU 上通常为针对昇腾
  优化的 HCCL/专用通信实现），不可用时回退到 vLLM 标准分布式接口。
- reduce_scatter / shard 前会把 token 数补齐到 tp_size 的整数倍，
  因为集合通信要求各 rank 张量形状一致且能均匀切分。
"""

import torch
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)


def _custom_collective(name: str, x: torch.Tensor) -> torch.Tensor | None:
    """尝试调用设备通信器上的自定义集合通信算子，不可用时返回 None。

    语法点：
    - 返回类型 ``torch.Tensor | None`` 是 PEP 604 联合类型写法，
      等价于 Optional[torch.Tensor]（要求 Python 3.10+）。
    - getattr(obj, name, default) 按字符串名动态取属性，取不到时返回
      默认值，这里用于"探测"后端是否注册了该优化算子。

    Args:
        name: 自定义集合通信方法名，如 "custom_all_gather" /
            "custom_reduce_scatter"。
        x: 输入张量，形状 [num_tokens, ...]（首维为 token 维）。

    Returns:
        调用成功返回通信结果张量；设备通信器或该算子不存在时返回
        None，由调用方回退到标准分布式实现。
    """
    # 步骤1: 取出 TP 组的设备通信器。原理: vLLM 的 GroupCoordinator 通过
    # device_communicator 封装硬件相关的底层通信（NPU 上为 HCCL 等后端），
    # 其上可挂载融合/异步优化的自定义集合通信算子。
    device_communicator = get_tp_group().device_communicator
    if device_communicator is None:
        # 没有设备通信器（如未初始化通信后端），返回 None 走回退路径。
        return None
    # 步骤2: 按名字探测优化算子；存在则直接调用并返回结果。
    collective = getattr(device_communicator, name, None)
    return None if collective is None else collective(x)


def sp_all_gather(x: torch.Tensor) -> torch.Tensor:
    """SP 全收集：沿 token 维（dim=0）拼接各 TP rank 的分片。

    原理: 进入注意力层前，各 rank 只持有本地的 token 分片（形状
    [num_local_tokens, hidden]）；注意力需要完整序列，因此沿 dim=0 做
    all_gather，得到形状 [num_local_tokens * tp_size, hidden] 的完整序列，
    且每个 rank 持有相同的全量输入。

    Args:
        x: 本 rank 的局部张量，形状 [num_local_tokens, ...]。

    Returns:
        全收集后的张量，形状 [num_local_tokens * tp_size, ...]
        （tp_size == 1 时等于原张量）。
    """
    # 步骤1: 优先走 NPU 优化的自定义 all_gather（如融合/异步通信实现）。
    output = _custom_collective("custom_all_gather", x)
    if output is not None:
        return output
    # 步骤2: 回退到 vLLM 标准 TP all_gather；第二个参数 0 表示沿 dim=0 拼接。
    return tensor_model_parallel_all_gather(x, 0)


def sp_reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    """SP 规约散射：各 rank 的部分和先求和，再沿 token 维均匀切回各 rank。

    原理: TP 下输出投影按权重切分，各 rank 的注意力输出是"部分和"，
    本应对全部 token 做 all-reduce 才能得到完整结果；SP 模式把
    all-reduce 与"切分为各 rank 的 token 分片"合并成一次 reduce_scatter，
    既省一半通信量，又完成序列维的重新切分。输出形状 = 输入形状 / tp_size
    （沿 dim=0）。

    Args:
        x: 输入张量，形状 [num_tokens, hidden]（必须为二维，见断言）。

    Returns:
        本 rank 的分片，形状 [num_tokens / tp_size, hidden]。
    """
    # 断言输入为二维 [num_tokens, hidden]，后续的 pad/切分逻辑按此假设。
    assert x.ndim == 2
    tp_size = get_tensor_model_parallel_world_size()
    # 步骤1: 计算需补齐的 token 数。原理: Python 中 (-n) % m 恰好等于
    # "把 n 补到 m 的整数倍还差多少"（结果范围 [0, m)），如 n=5,m=4 -> 3，
    # n=8,m=4 -> 0；等价于 (m - n % m) % m 但更简洁。
    sp_pad = (-x.shape[0]) % tp_size
    # Avoid copying the full input when its token count is already aligned.
    # （中文补充）token 数已对齐时跳过 cat，避免一次全量拷贝（性能优化）。
    if sp_pad > 0:
        # 步骤2: 在末尾拼接零填充，使 token 数能被 tp_size 整除。
        # 原理: 集合通信要求各 rank 张量形状一致且可均匀切分；
        # x.new_zeros 按 x 的 dtype/device 创建零张量。
        pad_shape = [sp_pad, x.shape[1]]
        x = torch.cat([x, x.new_zeros(pad_shape)], dim=0)
    # 步骤3: 优先走 NPU 优化的自定义 reduce_scatter。
    output = _custom_collective("custom_reduce_scatter", x)
    if output is not None:
        return output
    # 步骤4: 回退到 vLLM 标准 TP reduce_scatter（沿 dim=0 切分）。
    return tensor_model_parallel_reduce_scatter(x, 0)


def sp_shard(x: torch.Tensor) -> torch.Tensor:
    """本地分片（无通信）：沿 dim=0 均匀切出本 rank 的那一份。

    原理: 与 sp_all_gather/sp_reduce_scatter 不同，本函数不发起任何
    集合通信，仅做本地切片。典型用法：模型 forward 入口处各 rank 拿到的
    是相同的完整输入（如 embedding 输出），用本函数各自取走属于自己的
    token 分片，之后非注意力层即可在局部分片上计算（进注意力层前再
    gather 回来）。支持任意维数的张量（dim=0 为 token 维即可），如
    [num_tokens, hidden] 或 [num_tokens, batch, hidden]。

    Args:
        x: 完整输入张量，形状 [num_tokens, ...]。

    Returns:
        本 rank 的连续分片，形状 [ceil(num_tokens / tp_size), ...]。
        注意：为对齐 tp_size 会先补零再切，分片可能比理论值多一行，
        多出的假 token 由 sp_padding_mask 生成的掩码标记。
    """
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    # 步骤1: 计算 dim=0 需补齐的数量（负数取模技巧，见 sp_reduce_scatter）。
    sp_pad = (-x.shape[0]) % tp_size
    # 步骤2: 构造与 x 同 dtype/device 的零张量（首维为 pad 数）并拼接。
    # 语法点: list(x.shape) 拷贝形状列表后改首维，避免修改原 shape 元组。
    pad_shape = list(x.shape)
    pad_shape[0] = sp_pad
    x = torch.cat([x, x.new_zeros(pad_shape)], dim=0)
    # 步骤3: 按 rank 序号切出连续分片 [rank*chunk, (rank+1)*chunk)。
    # 原理: 与 all_gather 的拼接顺序一致，保证各 rank 分片不重不漏。
    chunk = x.shape[0] // tp_size
    return x[tp_rank * chunk : (tp_rank + 1) * chunk]


def sp_padding_mask(
    is_padding: torch.Tensor | None,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """生成本 rank 序列分片对应的 padding 掩码。

    原理: sp_shard / sp_reduce_scatter 为对齐 tp_size 会补零"假 token"，
    注意力等算子需要知道哪些位置是假的（padding）以便忽略。本函数把
    （可能为 None 的）原始 padding 标记补齐到 tp_size 倍数后，切出与本
    rank hidden_states 分片逐 token 对齐的那一段；补齐出的位置全部标记
    为 True（是 padding）。

    Args:
        is_padding: 原始 padding 标记，形状 [num_tokens]，dtype 为
            torch.bool，True 表示该 token 是填充；可为 None（视为全 False）。
        hidden_states: 本 rank 的隐藏状态，形状 [num_tokens, ...]，
            仅用其首维确定原始 token 数。
        类型注解 torch.Tensor | None 即 PEP 604 可选类型写法。

    Returns:
        本 rank 的掩码分片，形状 [ceil(num_tokens / tp_size)]，dtype 为
        bool，与 sp_shard(hidden_states) 的分片逐 token 一一对应。
    """
    num_tokens = hidden_states.shape[0]
    if is_padding is None:
        # 未提供掩码时视为"没有 padding"，构造全 False 的默认掩码。
        is_padding = hidden_states.new_zeros(num_tokens, dtype=torch.bool)
    # 校验掩码长度与 token 数一致（防御式断言）。
    assert is_padding.shape[0] == num_tokens

    tp_size = get_tensor_model_parallel_world_size()
    # 步骤1: 计算补齐数量——必须与 sp_shard 的补齐方式完全一致，
    # 否则掩码与分片会错位。
    sp_pad = (-num_tokens) % tp_size
    # 步骤2: 末尾补 True：补出来的都是假 token，必须标记为 padding，
    # 与 sp_shard 中"补零"的位置一一对应。
    is_padding = torch.cat([is_padding, is_padding.new_ones((sp_pad,))], dim=0)
    # 步骤3: 切出本 rank 的连续分片，切片逻辑与 sp_shard 完全对称。
    chunk = is_padding.shape[0] // tp_size
    tp_rank = get_tensor_model_parallel_rank()
    return is_padding[tp_rank * chunk : (tp_rank + 1) * chunk]
