# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# 【文件职责】engram 嵌入表的 Ascend 实现（INT8 checkpoint 契约 + 头分片）。
#
# 【表结构】每个 engram 层一张表: 行数 = engram_num_embeddings（哈希桶数，
# 亿级），列数 = engram_head_dim。存储为 INT8 码 + 组 32 FP32 缩放
# （区别于上游的 FP8/UE8M0——Ascend checkpoint 的量化契约）。
#
# 【分片策略】按"哈希头桶"切分: 表的行按 head_sizes 划分成连续段（每段
# 对应一个哈希头），段集合再均分到 TP×EDP 个 rank——每 rank 拥有若干
# 完整头段的全部行（列不切分，gather 只按行）。查表流程:
#   forward(indices) → gather_engram_hashes(DP 拼批) → lookup(本分片头段
#   gather+反量化) → TP all_gather(dim=1) 拼头顺序 → 切回本副本 token。
#
# 【存储位置】设备内存（默认）/ 主机 UVA（cpu_offload）/ 节点内共享内存
# （dp_shared_memory，N 份副本共享 1 份物理表）。
#
# 【加载策略】不走常规权重迭代——bind_checkpoint 记录路径/键名，权重
# "到达"时触发 _weight_loader → load_checkpoint 按索引分片流式读取
# safetensors（每次 chunk_rows=65536 行），BF16 权重现场量化成 INT8。
# =============================================================================
"""Ascend head-sharded Engram table with the INT8 checkpoint contract.

Subclasses upstream ``ParallelEngramEmbedding`` and preserves its parameter
and loader interface. Ascend supplies initialization, uniform head layout,
storage and lookup; upstream rank selection and hash gathering are reused:

* codes are INT8 with group-32 FP32 scales (the Ascend checkpoint uses these
  instead of upstream FP8/UE8M0);
* the table is either NPU memory or CANN-registered host memory read through a
  chunked device pointer table.
"""

import json
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch import nn
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import logger
from vllm.model_executor.utils import set_weight_attrs

# Upstream #56741 normalized the V4.1 model package name.
# 【中文】继承上游 ParallelEngramEmbedding 以保留其参数与加载器接口。
from vllm.models.deepseek_v41.common.engram import ParallelEngramEmbedding

from .npu import (
    HostUvaBuffer,
    SharedUvaBuffer,
    gather_dequantize_engram_int8,
    gather_dequantize_host_uva,
    quantize_engram_rows,
)
from .parallel import (
    _gather_engram_rows,
    engram_head_shard_rank,
    gather_engram_hashes,
    get_engram_dp_group,
    get_engram_dp_size,
)


class AscendParallelEngramEmbedding(ParallelEngramEmbedding):
    """TP(+EDP) head shard of one Engram table, stored as INT8 + FP32 scales."""
    """【中文说明】一张 engram 表的 TP(+EDP) 头分片。继承上游
    ParallelEngramEmbedding，Ascend 侧提供: 初始化、均匀头布局、存储与
    查找；上游的 rank 选择与哈希 gather 逻辑直接复用。
    权重形态: weight=[本rank行数, dim] INT8 码; weight_scale_inv=
    [本rank行数, dim/32] FP32 缩放（组 32 量化）。"""

    def __init__(
        self,
        num_embeddings: int,
        dim: int,
        head_sizes: tuple[int, ...],
        layer_hash_index: int,
        block_size: int = 32,
        cpu_offload: bool = False,
        dp_shared_memory: bool = False,
    ) -> None:
        """初始化头分片。

        参数:
            num_embeddings: 总行数（各头段行数之和）。
            dim: 每行宽度（engram_head_dim）。
            head_sizes: 各哈希头的行数（表按头切连续段）。
            layer_hash_index: 本表对应的哈希槽位（layout.primes 的下标）。
            block_size: 量化组大小（默认 32）。
            cpu_offload: 表驻留主机内存（UVA 查找）。
            dp_shared_memory: 节点内 DP 副本共享一份物理表。
        步骤:
            1) 校验 DP 组同节点（共享内存/UVA 都要求）；确定有效 dp_size；
            2) 计算头分片区间: n_hash_cols 个头均分到 tp_size*dp_size 个
               rank，头数不整除则报错（建议换拓扑或开共享内存）；
            3) 分配权重（设备/主机 UVA/共享 UVA 三种形态）并注册加载器；
            4) cpu_offload 时登记 dummy 值（探测前向用）并打日志。
        """
        self.cpu_offload = cpu_offload
        self.layer_hash_index = layer_hash_index
        self._shared_group = None
        group = get_engram_dp_group()
        if group is not None and not all(in_the_same_node_as(group.cpu_group)):
            # UVA/共享内存都要求设备地址跨 rank 可达，即必须同节点。
            raise ValueError(
                "Ascend Engram requires all DP replicas to share the same node and shared-memory namespace"
            )
        if dp_shared_memory:
            if group is None or group.world_size <= 1:
                raise ValueError("dp_shared_memory needs a node-local sharing group with more than one rank")
            self._shared_group = group
            # Sharing replaces the per-step DP lookup collectives: every
            # replica looks up its own tokens over the mapped table.
            # 【中文】共享模式下逐步的 DP 集合查表被取代: 每副本直接查
            # 映射表里自己的 token → 有效 dp_size 视为 1。
            self.dp_size = 1
        else:
            self.dp_size = max(get_engram_dp_size(), 1)
        # UVA 缓冲占位（非卸载模式保持 None）。
        self._codes_uva: HostUvaBuffer | SharedUvaBuffer | None = None
        self._scales_uva: HostUvaBuffer | SharedUvaBuffer | None = None
        # The upstream constructor queries CUDA properties; keep its parameter
        # contract with Ascend INT8 storage.
        # 【中文】上游构造器会查询 CUDA 属性（NPU 上不可用）——绕开它直接
        # nn.Module.__init__，但保留同样的参数契约，改用 Ascend INT8 存储。
        nn.Module.__init__(self)
        assert head_sizes and all(size > 0 for size in head_sizes)
        assert sum(head_sizes) <= num_embeddings
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.block_size = block_size
        # n_hash_cols: 哈希列数 = 头数（每头一列哈希 → 一段行）。
        self.n_hash_cols = len(head_sizes)
        self.tp_size = get_tensor_model_parallel_world_size()
        num_shards, head_rank = self._get_shard_info()
        if self.n_hash_cols % num_shards:
            # 头必须能整分到 TP×EDP 个 rank，否则行区间无法连续。
            raise ValueError(
                f"Engram requires uniform head shards: {self.n_hash_cols} heads "
                f"cannot be divided over {num_shards} TP x EDP shards. "
                "Use a divisible topology or enable dp_shared_memory."
            )
        # 本 rank 的头区间 [head_start, head_start+part_n_hash_cols)。
        self.part_n_hash_cols = self.n_hash_cols // num_shards
        self.head_start = head_rank * self.part_n_hash_cols
        # 由头区间映射到行区间 [vocab_start, vocab_end)。
        self.vocab_start_idx = sum(head_sizes[: self.head_start])
        self.vocab_end_idx = sum(head_sizes[: self.head_start + self.part_n_hash_cols])
        self.part_num_embeddings = self.vocab_end_idx - self.vocab_start_idx
        weight, scales = self._allocate_weights()
        # 语法点: requires_grad=False——推理表参数不进 autograd 图。
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.weight_scale_inv = nn.Parameter(scales, requires_grad=False)
        for param in (self.weight, self.weight_scale_inv):
            # 注册自定义加载器与行区间起点（加载器据此切本 rank 行段）。
            set_weight_attrs(param, {"weight_loader": self._weight_loader, "engram_vocab_start": self.vocab_start_idx})
        if cpu_offload:
            # dummy 值: --load-format=dummy 探测前向时用 0/1.0 占位。
            set_weight_attrs(self.weight, {"dummy_weight_value": 0})
            set_weight_attrs(self.weight_scale_inv, {"dummy_weight_value": 1.0})
            logger.info(
                "Engram table offloaded to registered host memory: %d rows x %d, %.2f GiB per rank",
                self.part_num_embeddings,
                self.dim,
                self.part_num_embeddings * (self.dim + (self.dim // self.block_size) * 4) / 1024**3,
            )

    def _get_shard_info(self) -> tuple[int, int]:
        """返回 (总分片数, 本 rank 分片序号)。原理: 无 EDP 时按 TP 切；
        有 EDP 时按 TP×EDP 切，序号用 TP-major 的 engram_head_shard_rank。"""
        if self.dp_size == 1:
            return self.tp_size, get_tensor_model_parallel_rank()
        return self.tp_size * self.dp_size, engram_head_shard_rank()

    def _allocate_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """分配本 rank 的 INT8 码与 FP32 缩放张量（三种形态之一）。

        返回: (codes [part_rows, dim] INT8, scales [part_rows, dim/32] FP32)。
        注意: 设备路径用 zeros 而非 empty——引擎在 checkpoint 读入前会先跑
        探测前向（含 engram 查表），未初始化的行会毒化数据流导致后续
        MoE 路由出错（详见下方英文注释）。
        """
        codes_shape = (self.part_num_embeddings, self.dim)
        scales_shape = (self.part_num_embeddings, self.dim // self.block_size)
        device = torch.device("npu", torch.npu.current_device())
        if not self.cpu_offload:
            # 形态1: 设备内存。
            return (
                # Zeroed, not empty: the engine profiles the model (and runs the
                # Engram lookups) before the checkpoint is read, and a table of
                # uninitialised rows poisons the stream enough to break the MoE
                # routing later in that same forward.
                # 【中文】零初始化: 探测前向先于权重加载执行，垃圾行会污染
                # 数据流进而破坏同一次前向里后面的 MoE 路由。
                torch.zeros(codes_shape, dtype=torch.int8, device=device),
                torch.zeros(scales_shape, dtype=torch.float32, device=device),
            )
        if self._shared_group is not None:
            # One physical copy of the mapped range, registered by each
            # rank in the sharing group.
            # 【中文】形态2: 共享 UVA——全组一份物理拷贝，各 rank 注册各自映射。
            self._codes_uva = SharedUvaBuffer(codes_shape, torch.int8, device, self._shared_group)
            self._scales_uva = SharedUvaBuffer(scales_shape, torch.float32, device, self._shared_group)
            return self._codes_uva.tensor, self._scales_uva.tensor
        # 形态3: 私有主机 UVA。aclrtMallocHost 返回的页内容不确定，
        # 同设备路径的理由——探测前向会先查表，必须清零。
        self._codes_uva = HostUvaBuffer(codes_shape, torch.int8, device)
        self._scales_uva = HostUvaBuffer(scales_shape, torch.float32, device)
        # Same reason as the device path: aclrtMallocHost hands back whatever
        # was in the pages, and the profiling forward looks up before the
        # checkpoint load fills them.
        self._codes_uva.tensor.zero_()
        self._scales_uva.tensor.zero_()
        return self._codes_uva.tensor, self._scales_uva.tensor

    def close_host_offload(self) -> None:
        """Release the registered host ranges (shutdown / reload path).

        The parameters alias the host range, so a released buffer has to take
        its alias with it: otherwise a later reader follows a live tensor into
        unmapped memory.  Each buffer and its own alias are dropped together,
        and only after that buffer's close succeeded, so a failed unregister
        stays retryable and a half-finished release never leaves a live
        parameter pointing at memory the other buffer has already freed.
        """
        """【中文说明】释放已注册的主机区间（停机/重载路径）。
        原理: weight/weight_scale_inv 参数是主机区间的别名张量——释放区间
        必须同时替换掉别名，否则后续读者会拿着活张量访问已取消映射的
        内存。每个 buffer 与自己的别名成对清理，且仅在 close 成功后才清理
        ——失败的 unregister 保持可重试，半途释放不会留下指向已释放内存
        的活参数（用空参数占位）。"""
        # 别名映射: 缓冲属性 → 参数名。
        aliases = {
            "_codes_uva": "weight",
            "_scales_uva": "weight_scale_inv",
        }
        for name, alias in aliases.items():
            buffer = getattr(self, name)
            if buffer is None:
                continue
            # 先关缓冲（注销+释放），再把参数替换为同 dtype/device 的空张量。
            buffer.close()
            setattr(self, name, None)
            param = getattr(self, alias)
            setattr(
                self,
                alias,
                nn.Parameter(
                    torch.empty(0, dtype=param.dtype, device=param.device),
                    requires_grad=False,
                ),
            )

    def bind_checkpoint(self, model_path, key: str) -> None:
        """Use parameter callbacks while retaining the indexed shard reader.

        Codes and scales are read together from the selected index when the
        weight arrives. The scale callback is deliberately a no-op: iterator
        order and a separate unquantized index must not overwrite that pair.
        """
        """【中文说明】绑定 checkpoint 路径与键名（不立即读取）。真正的读取
        由 _weight_loader 在权重"到达"时触发（按索引分片流式读，码与缩放
        成对读入）。scale 的回调刻意为空操作: 权重迭代顺序或独立索引不得
        覆盖这一对已读入的数据。"""
        self._checkpoint_path = model_path
        self._checkpoint_key = key

    def _weight_loader(self, param, loaded_weight) -> None:
        """权重到达回调: 只在 weight 参数上触发一次完整加载（scale 回调为空）。"""
        if param is self.weight:
            self.load_checkpoint(self._checkpoint_path, self._checkpoint_key)

    def load_checkpoint(self, model_path, key, chunk_rows=65536):
        """Stream this rank's assigned rows, quantizing BF16 when needed.

        Shared head slices have one writer per EDP group. The final CPU
        collective synchronizes writes and propagates loading failures.
        """
        """【中文说明】流式加载本 rank 分配的行段（每批 chunk_rows 行）。
        BF16 checkpoint 现场量化成 INT8+组32 缩放。
        共享模式: 每个 EDP 组只有一个写者（rank 0），其余 rank 等待；
        末尾的 CPU 集合通信同步写入并把加载失败传播到每个 rank。"""
        if self._shared_group is None:
            # 非共享: 本 rank 独立加载自己的行段。
            self._load_into_storage(model_path, key, chunk_rows)
            return
        error = None
        if self._shared_group.rank_in_group == 0:
            try:
                self._load_into_storage(model_path, key, chunk_rows)
            except Exception as exc:  # noqa: BLE001 - propagated to every rank
                error = f"{type(exc).__name__}: {exc}"
        # 汇聚各 rank 错误；任一失败则全员抛错（避免部分 rank 带着空表运行）。
        errors: list[str | None] = [None] * self._shared_group.world_size
        dist.all_gather_object(errors, error, group=self._shared_group.cpu_group)
        failures = "; ".join(f"rank {rank}: {failure}" for rank, failure in enumerate(errors) if failure is not None)
        if failures:
            raise RuntimeError(f"Engram shared load failed: {failures}")

    # The Ascend table is streamed out of these indexed safetensors shards.
    # 【中文】Ascend 表从这些"带索引的 safetensors 分片"流式读出
    # （按优先级尝试两个索引文件名）。
    _INDEX_FILES = (
        "quant_model_weights.safetensors.index.json",
        "model.safetensors.index.json",
    )

    @classmethod
    def _checkpoint_index(cls, root: Path, key: str) -> dict[str, str]:
        """Weight map of the shard that holds ``key``.

        A checkpoint without an index, or a single-file/``pt`` one, cannot be
        served by this loader.  Inside the loader this is the last line of
        defence, after the tables are already allocated; the construction path
        runs `preflight_engram_checkpoint()` first so an unreadable checkpoint
        fails before any backing exists.
        """
        """【中文说明】解析 checkpoint 的 weight_map（键 → 分片文件名）。

        无索引、单文件或 .pt 格式的 checkpoint 无法被本加载器服务。在加载
        器内部这是"最后防线"（表已分配之后）；构造路径会先跑
        preflight_engram_checkpoint() 让不可读的 checkpoint 在任何表分配
        之前就失败。"""
        named = []
        for name in cls._INDEX_FILES:
            candidate = root / name
            if not candidate.is_file():
                continue
            named.append(name)
            weight_map = json.loads(candidate.read_text())["weight_map"]
            if key in weight_map:
                return weight_map
        raise ValueError(
            f"Engram table {key!r} is not in an indexed safetensors checkpoint "
            f"under {root} (found: {', '.join(named) or 'no index'})."
        )

    def _load_into_storage(self, model_path, key, chunk_rows) -> None:
        """实际流式读取: 逐 chunk 拷贝 INT8（或现场量化 BF16）进存储。"""
        root = Path(model_path)
        # 缩放键名约定: xxx.weight → xxx.scale。
        scale_key = key.removesuffix(".weight") + ".scale"
        index = self._checkpoint_index(root, key)
        start, end = (self.vocab_start_idx, self.vocab_end_idx)
        # 语法点: safe_open 惰性句柄（with 块内 get_slice 取张量切片，
        # 不把整个分片读进内存）。
        with safe_open(root / index[key], framework="pt", device="cpu") as file:
            tensor = file.get_slice(key)
            # 已量化 checkpoint（I8/INT8）: 码与缩放分别成对分块拷贝。
            quantized = tensor.get_dtype() in ("I8", "INT8")
            if quantized:
                with safe_open(root / index[scale_key], framework="pt", device="cpu") as sf:
                    scale = sf.get_slice(scale_key)
                    for chunk_start in range(start, end, chunk_rows):
                        stop = min(chunk_start + chunk_rows, end)
                        # 全局行号 → 本地行号偏移换算。
                        offset = chunk_start - self.vocab_start_idx
                        target_end = offset + (stop - chunk_start)
                        self.weight.data[offset:target_end].copy_(tensor[chunk_start:stop])
                        self.weight_scale_inv.data[offset:target_end].copy_(scale[chunk_start:stop])
            else:
                # BF16 checkpoint: 每 chunk 现场 quantize_engram_rows 量化。
                for chunk_start in range(start, end, chunk_rows):
                    stop = min(chunk_start + chunk_rows, end)
                    offset = chunk_start - self.vocab_start_idx
                    target_end = offset + (stop - chunk_start)
                    codes, scales = quantize_engram_rows(tensor[chunk_start:stop].to(torch.float32))
                    self.weight.data[offset:target_end].copy_(codes)
                    self.weight_scale_inv.data[offset:target_end].copy_(scales)
        logger.info("Engram rows [%d, %d) loaded from %s", start, end, index[key])

    def embed_gathered(self, gathered: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """Embed ids already gathered across the EDP group.

        ``gathered`` is ``[slot * EDP, n_hash_cols]`` rank-major; each replica
        keeps its own ``num_tokens`` window. Returns ``[num_tokens, n_hash_cols,
        dim]`` bf16 with the heads back in checkpoint order.
        """
        """【中文说明】嵌入"已完成 DP gather"的哈希 ID。
        参数:
            gathered: [slot*EDP, n_hash_cols] rank-major 的哈希批。
            num_tokens: 本副本的真实 token 数（窗口大小）。
        返回:
            [num_tokens, n_hash_cols, dim] BF16，头恢复 checkpoint 顺序。
        步骤:
            1) 连续化（gathered 可能是三维缓冲的切片，kernel 需要单位内步长）；
            2) lookup 查本分片头段 → [gathered_tokens, part_heads, dim]；
            3) DP>1: _gather_engram_rows 交换回本副本 token 窗口并拼各 rank 头；
               否则直接截前 num_tokens；
            4) TP>1: all_gather(dim=1) 拼完剩余头，最后裁掉填充头列。
        """
        # The kernel walks ids as [token, columns] with a unit inner stride, and
        # `gathered` is a slice of a [tokens, layers, columns] buffer.
        # 【中文】kernel 要求 ids 是内步长为 1 的 [token, columns] 布局，
        # 而 gathered 是 [tokens, layers, columns] 缓冲的切片——先连续化。
        gathered = gathered.contiguous()
        out = torch.zeros(
            (gathered.shape[0], self.part_n_hash_cols, self.dim),
            dtype=torch.bfloat16,
            device=gathered.device,
        )
        # 步骤2: 查本分片头段（无效 ID 的行写零）。
        self.lookup(gathered, out)
        if self.dp_size > 1:
            # 步骤3: DP 交换——取回本副本 token、拼全组头。
            out = _gather_engram_rows(out, num_tokens)
        else:
            out = out[:num_tokens]
        if self.tp_size > 1:
            # 步骤4: TP 拼头（dim=1 沿头维），头顺序 = TP-major 分片序。
            out = tensor_model_parallel_all_gather(out, dim=1)
        # 裁掉 part_n_hash_cols*tp_size 可能多出的填充列，回到 n_hash_cols。
        return out[:, : self.n_hash_cols]

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: [num_tokens, n_hash_cols] -> [num_tokens, n_hash_cols, dim].

        The shared mode has to be passed through: the replicas of one TP slot
        map the same table, so a gathered batch would make each of them embed
        EDP0's ids instead of its own window.
        """
        """【中文说明】嵌入入口: indices [num_tokens, n_hash_cols]（本副本的
        哈希 ID）→ [num_tokens, n_hash_cols, dim]。
        共享模式标志必须透传给 gather: 同一 TP 槽的各副本映射同一张表，
        若照常 gather，每副本都会去嵌 EDP0 的 ID 而不是自己的窗口。"""
        num_tokens = indices.shape[0]
        gathered = gather_engram_hashes(indices, dp_shared_memory=self._shared_group is not None)
        return self.embed_gathered(gathered, num_tokens)

    def lookup(self, indices: torch.Tensor, out: torch.Tensor, background: bool = False) -> None:
        """Look up this shard's heads into ``[tokens, padded_heads, dim]`` bf16.

        All local head columns are written, including zeroes for invalid IDs.
        """
        """【中文说明】把本分片头段查进 out [tokens, part_heads, dim] BF16。
        写入全部本地头列；无效 ID（含 DEAD_ID 填充）的行写零。
        分派顺序: 主机 UVA（offload）→ 设备表 → CPU 参考实现。"""
        tokens = indices.shape[0]
        if tokens == 0 or self.part_n_hash_cols == 0:
            return
        # launch: kernel 启动参数（头区间/输出视图），三种存储共用。
        launch = dict(
            head_start=self.head_start,
            local_heads=self.part_n_hash_cols,
            pad_heads=self.part_n_hash_cols,
            vocab_start=self.vocab_start_idx,
            vocab_end=self.vocab_end_idx,
            output=out.view(-1, self.dim),
        )
        if self._codes_uva is not None:
            # 路径1: 主机 UVA 表（指针表版 gather kernel）。
            assert self._scales_uva is not None
            codes_uva = cast(HostUvaBuffer, self._codes_uva)
            scales_uva = cast(HostUvaBuffer, self._scales_uva)
            gather_dequantize_host_uva(codes_uva, scales_uva, indices, **launch)
        elif self.weight.device.type != "cpu":
            # 路径2: 设备 INT8 表（单 kernel gather+反量化）。
            gather_dequantize_engram_int8(self.weight, self.weight_scale_inv, indices, self.dim, **launch)
        else:
            # 路径3: CPU 参考实现（测试/无设备环境）。
            _torch_lookup(self, indices, out)


def _torch_lookup(embed: AscendParallelEngramEmbedding, indices, out) -> None:
    """CPU reference used by tests and bring-up without a device table."""
    """【中文说明】CPU 参考实现（测试与无设备表环境用），与 Triton kernel
    语义一致: 取本分片头列 → 越界钳到第 0 行 → INT8×组缩放反量化 →
    越界行清零后写出。"""
    heads = embed.part_n_hash_cols
    # 只取本分片拥有的头列。
    columns = indices[:, embed.head_start : embed.head_start + heads].long()
    # 行归属判定与钳位（与 kernel 的"最后防线"一致）。
    owned = (columns >= embed.vocab_start_idx) & (columns < embed.vocab_end_idx)
    local = torch.where(owned, columns - embed.vocab_start_idx, 0).reshape(-1)
    codes = torch.index_select(embed.weight.data, 0, local)
    scales = torch.index_select(embed.weight_scale_inv.data, 0, local)
    # 反量化: 码按组乘缩放，展平转 BF16，恢复 [tokens, heads, dim]。
    decoded = codes.float().unflatten(-1, (-1, embed.block_size)) * scales.unsqueeze(-1)
    decoded = decoded.flatten(-2).bfloat16().view(indices.shape[0], heads, embed.dim)
    # 越界行写零。
    out[:, :heads].copy_(torch.where(owned.unsqueeze(-1), decoded, 0.0))


def preflight_engram_checkpoint(root, layer_ids, embed_cls=AscendParallelEngramEmbedding) -> None:
    """Check that this checkpoint can be served *before* any table is allocated.

    The shard loader streams indexed safetensors, so a checkpoint without an
    index (or with a missing Engram key, or an INT8 weight without its scale)
    cannot be loaded here.  Finding that out during weight iteration would be
    after every rank has already allocated and registered its table,
    potentially the complete node table in row shared mode. Resolve the index
    entries and read the safetensors headers first.  The loader
    keeps its own check as the last line of defence.
    """
    """【中文说明】在任何表分配之前预检 checkpoint 可服务性。
    原理: 分片加载器只支持带索引的 safetensors——缺索引、缺 engram 键、
    INT8 权重缺配套 scale 都无法加载。若拖到权重迭代阶段才发现，所有
    rank 已经分配并注册了表（行共享模式下可能是整节点的表），代价巨大。
    因此先解析索引项、读 safetensors 头部。加载器内部保留自己的检查作为
    最后防线。步骤（逐层）:
        1) 解析 weight_map 定位 weight 所在分片，分片文件必须存在；
        2) 读分片头确认权重是 INT8（非 INT8 则跳过 scale 检查）；
        3) INT8 时: scale 键必须在同一索引中、分片存在、且分片内确实含
           该 scale 张量。
    """
    root = Path(root)
    for layer_id in layer_ids:
        # engram 权重键名约定: layers.{layer_id}.engram.embed.weight。
        key = f"layers.{layer_id}.engram.embed.weight"
        index = embed_cls._checkpoint_index(root, key)
        shard = root / index[key]
        if not shard.is_file():
            raise ValueError(f"Engram layer {layer_id}: the checkpoint index points at {shard}, which does not exist.")
        with safe_open(shard, framework="pt", device="cpu") as file:
            quantized = file.get_slice(key).get_dtype() in ("I8", "INT8")
        if not quantized:
            continue
        scale_key = key.removesuffix(".weight") + ".scale"
        # The loader resolves scales from the weight's selected index too.
        # 【中文】加载器也会从 weight 选中的同一索引解析 scale——这里保持
        # 同样的查找规则做预检。
        if scale_key not in index:
            raise ValueError(
                f"Engram layer {layer_id}: {scale_key!r} is missing from the checkpoint index selected for {key!r}."
            )
        scale_shard = root / index[scale_key]
        if not scale_shard.is_file():
            raise ValueError(
                f"Engram layer {layer_id}: the checkpoint index points at "
                f"{scale_shard} for {scale_key!r}, which does not exist."
            )
        with safe_open(scale_shard, framework="pt", device="cpu") as file:
            if scale_key not in file.keys():  # noqa: SIM118 - safe_open is not a mapping
                raise ValueError(
                    f"Engram layer {layer_id}: {scale_shard} does not contain the scale tensor {scale_key!r}."
                )
