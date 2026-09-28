# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# 【文件职责】engram 的 NPU 侧存储、路由与查找原语。
#
# 【存储形态】每层 engram 表都是 INT8 码 + 组 32 FP32 缩放（对称量化），
# 按"哈希头桶"连续切分到各 rank。三种驻留方式:
#   1) 设备内存（默认）: gather_dequantize_engram_int8 单 Triton kernel
#      完成 gather+反量化；
#   2) 主机 UVA 卸载（--engram-config cpu_offload）: aclrtHostRegisterV2
#      固定主机页并发布设备地址（HostUvaBuffer），NPU kernel 直接跨 PCIe
#      读主机内存——无需 H2D 拷贝、无需主机侧 gather；
#   3) 共享内存（dp_shared_memory）: 同节点多 DP 副本共享一份物理表
#      （SharedUvaBuffer，multiprocessing.SharedMemory）。
#
# 【为何有指针表 ptrs】超大表（384M 行）会超出单个 Triton tile 能表达的
# 32 位偏移算术，因此每 CHUNK_ROWS(2^22) 行发布一个独立的设备地址，
# kernel 先查指针表再算块内偏移。
# =============================================================================
"""NPU-side Engram storage, routing and lookup.

Every layer is INT8 with group-32 FP32 scales, sharded into contiguous hash
head buckets.  With ``EngramConfig.cpu_offload`` the shard stays in host memory:
``aclrtHostRegisterV2`` pins it and ``aclrtHostGetDevicePointer`` publishes the
address the NPU gather kernel reads, so an offloaded table needs neither an H2D
copy nor a host-side gather.
"""

import ctypes
from functools import cache
from multiprocessing import shared_memory
from unittest.mock import patch

import torch
import torch.distributed as dist
from vllm.logger import logger
from vllm.triton_utils import tl, triton

# 组量化粒度: 每 32 个 INT8 元素共享一个 FP32 缩放因子。
SCALE_GROUP = 32
# A 384M row table overflows the 32 bit offset arithmetic a single Triton tile
# can express, so the device address of every group of rows is published
# separately.
# 【中文】每 2^22 行发布一个设备地址（见模块头注释）。
CHUNK_ROWS = 1 << 22
# CANN aclrtHostRegisterV2 的标志位: MAPPED=发布设备可读映射，
# PINNED=页固定（不被换出，保证设备访问安全）。
ACL_HOST_REG_MAPPED = 0x2
ACL_HOST_REG_PINNED = 0x10000000


def engram_cpu_offload(vllm_config) -> bool:
    """Whether the Engram table is offloaded to host memory (UVA lookup).

    ``--engram-config`` turns on host offload. Without it, the tables stay on
    the device, exactly like upstream.
    """
    """【中文说明】engram 表是否卸载到主机内存（UVA 查找）。--engram-config
    的 cpu_offload 选项开启；不开则与上游一致驻留设备。"""

    engram_config = getattr(vllm_config, "engram_config", None)
    return bool(engram_config is not None and engram_config.cpu_offload)


def quantize_engram_rows(rows):
    """Group32 symmetric INT8 with FP32 power-of-two scales and ties-to-even."""
    """【中文说明】组 32 对称 INT8 量化（FP32 2 的幂缩放、ties-to-even 取整）。
    步骤: 切组 → 组内绝对值最大值 → scale=max/127 并向上取整到 2 的幂 →
    码 = round(x/scale) 钳位 [-127,127]。返回 (codes, scale)。
    NPU 细节: NPU 的 exp2 可能比精确 2 的幂低一个 ULP（会改变 ties-to-even
    的取整结果），故用 ldexp 直接按指数构造精确的 2 的幂。"""
    # 切成 [rows, groups, 32] 的组。
    grouped = rows.float().unflatten(-1, (-1, SCALE_GROUP))
    maximum = grouped.abs().amax(-1, keepdim=True)
    # 全零组 scale 取 1（防除零）。
    scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum / 127)
    # NPU exp2 can return one ULP below an exact power of two, changing
    # ties-to-even codes. ldexp constructs the binary scale exactly.
    # 【中文】ceil(log2(scale)) 后用 ldexp(1, exponent) 精确构造 2 的幂。
    exponent = torch.ceil(torch.log2(scale))
    scale = torch.where(torch.isfinite(exponent), torch.ldexp(torch.ones_like(scale), exponent.int()), scale)
    # 量化取整 + 钳位，展平回 [rows, dim]。
    codes = torch.round(grouped / scale).clamp(-127, 127).to(torch.int8).flatten(-2)
    return codes, scale.squeeze(-1)


def dequantize_engram_rows(codes, scale):
    """反量化: INT8 码 × 组缩放 → BF16。
    NPU 细节: 复用一个 FP32 工作缓冲做原地乘法（decoded.mul_），避免广播
    乘法表达式额外分配一份 FP32 结果。"""
    # Keep one FP32 work buffer: in-place scaling avoids the extra FP32 result
    # allocation created by the broadcast multiply expression.
    decoded = codes.float().unflatten(-1, (-1, SCALE_GROUP))
    decoded.mul_(scale.unsqueeze(-1))
    return decoded.flatten(-2).bfloat16()


@triton.jit
def _engram_int8_gather_dequant_kernel(
    weight_ptr,
    scale_ptr,
    ids_ptr,
    output_ptr,
    rows,
    vocab_start,
    vocab_end,
    ids_stride_t,
    WIDTH: tl.constexpr,
    GROUP: tl.constexpr,
    HEAD_START: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    PAD_HEADS: tl.constexpr,
):
    # 【中文】设备表 gather+反量化 kernel。每个 program 处理一行
    # （token × 本地头），从 INT8 表取一行码、对应组缩放，乘积转 BF16 写出。
    row = tl.program_id(0)
    if row >= rows:
        return
    offsets = tl.arange(0, WIDTH)
    # Row `row` is (token, local head); ids is [tokens, n_hash_cols] and this
    # shard owns heads [HEAD_START, HEAD_START + LOCAL_HEADS). Local heads are
    # written contiguously and the rest of PAD_HEADS stays untouched, so a
    # narrower shard never writes into the padding other ranks read.
    # 【中文】行号解码: row = token*LOCAL_HEADS + 本地头序号。本地头连续写入，
    # PAD_HEADS 的其余部分不动——窄分片绝不越界写其他 rank 读的填充区。
    token = row // LOCAL_HEADS
    local = row % LOCAL_HEADS
    source_row = tl.load(ids_ptr + token * ids_stride_t + HEAD_START + local).to(tl.int64)
    # Same last line of defence as the host-uva kernel, and the same contract
    # as upstream's lookup: ids are global and only this shard's vocab range is
    # owned; anything else reads row 0 and is masked back to zero.
    # 【中文】最后防线: id 是全局编号，本分片只拥有 [vocab_start, vocab_end)
    # 区间；区间外读第 0 行占位并在写出时掩码清零（与上游 lookup 契约一致）。
    owned = (source_row >= vocab_start) & (source_row < vocab_end)
    local_row = tl.where(owned, source_row - vocab_start, 0)
    # 取一行 INT8 码转 FP32，再按组取缩放（offsets//GROUP 定组）。
    codes = tl.load(weight_ptr + local_row * WIDTH + offsets).to(tl.float32)
    scales = tl.load(scale_ptr + local_row * (WIDTH // GROUP) + offsets // GROUP)
    result = (codes * scales).to(tl.bfloat16)
    tl.store(
        output_ptr + (token * PAD_HEADS + local) * WIDTH + offsets,
        tl.where(owned, result, tl.zeros_like(result)),
    )


def gather_dequantize_engram_int8(
    weight: torch.Tensor,
    scales: torch.Tensor,
    ids: torch.Tensor,
    width: int,
    *,
    head_start: int = 0,
    local_heads: int = 1,
    pad_heads: int | None = None,
    output: torch.Tensor | None = None,
    vocab_start: int = 0,
    vocab_end: int | None = None,
) -> torch.Tensor:
    """Gather this shard's head rows from a device table, dequantize in one kernel.

    Returns ``[tokens * pad_heads, width]``; the head path views it as
    ``[tokens, pad_heads, width]``.
    """
    """【中文说明】从设备 INT8 表 gather 本分片的头行并在单 kernel 内反量化。
    参数: weight [rows, width] INT8; scales [rows, width/32] FP32; ids
    [tokens, n_hash_cols] 全局哈希行号; head_start/local_heads 圈定本分片
    拥有的头列区间; pad_heads 为输出填充头数（各分片写各自的连续段）。
    返回: [tokens*pad_heads, width] BF16。"""

    # Importing the ops package initializes the active Triton backend, so keep
    # it out of CPU-only routing and test workers.
    # 【中文】导入 ops 包会初始化 Triton 后端——延迟导入，避免 CPU 路由与
    # 测试进程背上 NPU 初始化开销。
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    pad_heads = local_heads if pad_heads is None else pad_heads
    vocab_end = weight.shape[0] if vocab_end is None else vocab_end
    tokens = ids.shape[0]
    rows = tokens * local_heads
    if output is None:
        output = torch.empty((tokens * pad_heads, width), dtype=torch.bfloat16, device=weight.device)
    if rows == 0:
        return output
    init_device_properties_triton()
    # 启动 kernel: 每 (token, 本地头) 一行。
    _engram_int8_gather_dequant_kernel[(rows,)](
        weight,
        scales,
        ids,
        output,
        rows,
        vocab_start,
        vocab_end,
        ids.stride(0),
        WIDTH=width,
        GROUP=SCALE_GROUP,
        HEAD_START=head_start,
        LOCAL_HEADS=local_heads,
        PAD_HEADS=pad_heads,
        num_warps=4,
    )
    return output


@cache
def _host_library() -> ctypes.CDLL:
    """The CANN runtime entry points that publish host memory to the device."""
    """【中文说明】加载 libascendcl.so 并声明 CANN 运行时函数原型（ctypes）。
    语法点: @cache（functools.lru_cache 的装饰器简写）——进程内只加载一次；
    argtypes/restype 声明让 ctypes 正确处理指针与返回码。"""

    lib = ctypes.CDLL("libascendcl.so")
    # aclrtMallocHost/FreeHost: 分配/释放锁页主机内存。
    lib.aclrtMallocHost.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint32]
    lib.aclrtMallocHost.restype = ctypes.c_int
    lib.aclrtFreeHost.argtypes = [ctypes.c_void_p]
    lib.aclrtFreeHost.restype = ctypes.c_int
    # aclrtHostRegisterV2: 注册主机地址区间（MAPPED|PINNED 标志）。
    lib.aclrtHostRegisterV2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32]
    lib.aclrtHostRegisterV2.restype = ctypes.c_int
    # aclrtHostGetDevicePointer: 取该主机地址对应的设备侧可读地址。
    lib.aclrtHostGetDevicePointer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32]
    lib.aclrtHostGetDevicePointer.restype = ctypes.c_int
    # aclrtHostUnregister: 注销注册。
    lib.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
    lib.aclrtHostUnregister.restype = ctypes.c_int
    return lib


class HostUvaBuffer:
    """Host memory the device gathers from directly."""
    """【中文说明】主机 UVA 缓冲: NPU 可直接 gather 的锁页主机内存。
    生命周期: aclrtMallocHost 分配 → HostRegisterV2 注册（映射+固定）→
    HostGetDevicePointer 发布设备地址 → 生成每 CHUNK_ROWS 行的设备指针表
    → close() 时注销并释放。"""

    def __init__(self, shape, dtype, device):
        """参数: shape/dtype 为表的形状与类型; device 用于放置指针表张量。"""
        self.lib = _host_library()
        rows = int(shape[0])
        row_elements = int(torch.Size(shape[1:]).numel())
        self.row_bytes = row_elements * torch.empty((), dtype=dtype).element_size()
        size = rows * self.row_bytes
        # 步骤1: 分配锁页主机内存。
        self.pointer = ctypes.c_void_p()
        rc = self.lib.aclrtMallocHost(ctypes.byref(self.pointer), size, 0)
        if rc:
            raise RuntimeError(f"aclrtMallocHost failed: rc={rc} size={size}")
        # 步骤2: 包装成 Python 缓冲并构建张量视图（零拷贝）。
        self.buffer = (ctypes.c_char * size).from_address(self.pointer.value)
        self.tensor = torch.frombuffer(self.buffer, dtype=dtype).reshape(shape)
        try:
            # 步骤3: 注册（MAPPED|PINNED）→ 取设备地址 → 建 2^22 行粒度指针表。
            rc = self.lib.aclrtHostRegisterV2(self.pointer, size, ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED)
            if rc:
                raise RuntimeError(f"aclrtHostRegisterV2 failed: rc={rc} size={size}")
            address = ctypes.c_void_p()
            rc = self.lib.aclrtHostGetDevicePointer(self.pointer, ctypes.byref(address), 0)
            if rc:
                raise RuntimeError(f"aclrtHostGetDevicePointer failed: rc={rc}")
            # 语法点: 列表推导 + 张量构造——设备地址 + 行偏移逐块发布。
            self.ptrs = torch.tensor(
                [address.value + start * self.row_bytes for start in range(0, rows, CHUNK_ROWS)],
                dtype=torch.int64,
                device=device,
            )
        except Exception:
            # A half-built buffer must not leave the host range registered: the
            # caller only sees the exception, so nothing else can release it.
            # 【中文】半构造失败必须注销已注册区间——调用方只拿到异常，
            # 没有别人能释放它；清理后置空再重抛。
            self.lib.aclrtHostUnregister(self.pointer)
            self.lib.aclrtFreeHost(self.pointer)
            self.tensor = None
            self.buffer = None
            self.pointer = ctypes.c_void_p()
            raise

    def close(self):
        """Unregister and release the host range; safe to call more than once.

        The NPU gathers out of this range through the device address the
        registration published, so the mapping must not go away while work that
        reads it is still in flight.
        """
        """【中文说明】注销注册并释放主机内存；可安全重复调用。
        原理: NPU 经注册发布的设备地址直接读该区间，故必须等在飞的读取
        结束（torch.npu.synchronize）后才能注销。"""
        # 幂等保护: 已关闭（pointer 为空）则直接返回。
        if self.pointer is None or not self.pointer.value:
            return
        if self.ptrs is not None and self.ptrs.device.type == "npu":
            # 同步等待所有在飞读取完成。
            torch.npu.synchronize()
        rc = self.lib.aclrtHostUnregister(self.pointer)
        if rc:
            raise RuntimeError(f"aclrtHostUnregister failed: rc={rc}")
        self.tensor = None
        self.buffer = None
        self.ptrs = None
        rc = self.lib.aclrtFreeHost(self.pointer)
        if rc:
            raise RuntimeError(f"aclrtFreeHost failed: rc={rc}")
        self.pointer = ctypes.c_void_p()


@triton.jit
def _engram_host_uva_gather_dequant_kernel(
    codes_ptrs,
    scales_ptrs,
    ids,
    output,
    rows,
    vocab_start,
    vocab_end,
    ids_stride_t,
    CHUNK: tl.constexpr,
    WIDTH: tl.constexpr,
    GROUP: tl.constexpr,
    HEAD_START: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    PAD_HEADS: tl.constexpr,
):
    # 【中文】主机 UVA 表的 gather+反量化 kernel。与设备表 kernel 的差别：
    # 码/缩放不是张量而是"指针表"——先按行号定位 chunk（row//CHUNK），
    # 从指针表取该 chunk 的设备地址（主机内存的映射地址），再算块内偏移。
    row = tl.program_id(0)
    if row < rows:
        token = row // LOCAL_HEADS
        head_local = row % LOCAL_HEADS
        index = tl.load(ids + token * ids_stride_t + HEAD_START + head_local).to(tl.int64)
        # The kernel owns the last line of defence: an id outside this shard
        # must not become an address the pointer table is indexed with, whoever
        # computed it.
        # 【中文】最后防线: 越界 id 绝不能变成指针表的索引（否则会读任意
        # 主机地址）——区间外钳到第 0 行并清零。
        owned = (index >= vocab_start) & (index < vocab_end)
        local_row = tl.where(owned, index - vocab_start, 0)
        # 定位 chunk 与块内行号，从指针表取码/缩放的基地址。
        chunk = local_row // CHUNK
        local = local_row % CHUNK
        # 语法点: .to(tl.pointer_type(...)) 把 int64 地址转成可解引用指针。
        codes = tl.load(codes_ptrs + chunk).to(tl.pointer_type(tl.int8))
        scales = tl.load(scales_ptrs + chunk).to(tl.pointer_type(tl.float32))
        col = tl.arange(0, WIDTH)
        value = tl.load(codes + local * WIDTH + col).to(tl.float32)
        scale = tl.load(scales + local * (WIDTH // GROUP) + col // GROUP)
        result = (value * scale).to(tl.bfloat16)
        tl.store(
            output + (token * PAD_HEADS + head_local) * WIDTH + col,
            tl.where(owned, result, tl.zeros_like(result)),
        )


def gather_dequantize_host_uva(
    codes: HostUvaBuffer,
    scales: HostUvaBuffer,
    ids: torch.Tensor,
    *,
    head_start: int = 0,
    local_heads: int = 1,
    pad_heads: int | None = None,
    output: torch.Tensor | None = None,
    vocab_start: int = 0,
    vocab_end: int | None = None,
) -> torch.Tensor:
    """Gather this shard's head rows from a registered host table on device.

    Returns ``[tokens * pad_heads, width]``; the head path views it as
    ``[tokens, pad_heads, width]``.
    """
    """【中文说明】从已注册主机表（UVA）在设备侧 gather 本分片头行并反量化。
    参数: codes/scales 为 HostUvaBuffer（含设备指针表）; 其余同设备表版本。
    返回: [tokens*pad_heads, width] BF16。数据流: kernel 经 PCIe 直接读主机
    锁页内存，无 H2D 拷贝。"""

    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    pad_heads = local_heads if pad_heads is None else pad_heads
    width = codes.tensor.shape[-1]
    vocab_end = codes.tensor.shape[0] if vocab_end is None else vocab_end
    tokens = ids.shape[0]
    rows = tokens * local_heads
    if output is None:
        output = torch.empty((tokens * pad_heads, width), dtype=torch.bfloat16, device=ids.device)
    if rows == 0:
        return output
    init_device_properties_triton()
    # 传入指针表（每 2^22 行一个设备地址）；ids 转 INT64 匹配 kernel 约定。
    _engram_host_uva_gather_dequant_kernel[(rows,)](
        codes.ptrs,
        scales.ptrs,
        ids.to(torch.int64),
        output,
        rows,
        vocab_start,
        vocab_end,
        ids.stride(0),
        CHUNK=CHUNK_ROWS,
        WIDTH=width,
        GROUP=SCALE_GROUP,
        HEAD_START=head_start,
        LOCAL_HEADS=local_heads,
        PAD_HEADS=pad_heads,
        num_warps=4,
    )
    return output


class SharedUvaBuffer:
    """One host range mapped and registered by every rank of a local group.

    The leader creates the SharedMemory segment; every rank registers its own
    mapping with CANN. The leader unlinks it after all ranks have attached.
    """
    """【中文说明】共享 UVA 缓冲: 同一节点 DP 组内所有 rank 共享一份物理表。
    流程: 组长(leader)创建 SharedMemory 段 → 广播段名 → 各 rank attach 到
    自己的地址空间并注册给 CANN → 组长在全员就绪后 unlink 段名。
    效果: N 份 DP 副本的表显存/主机内存开销从 N 份降到 1 份。"""

    def __init__(self, shape, dtype, device, group):
        """参数: group 为节点内共享组（需含 cpu_group 通信子）。"""
        self.lib = _host_library()
        self.group = group
        rows = int(shape[0])
        row_elements = int(torch.Size(shape[1:]).numel())
        self.row_bytes = row_elements * torch.empty((), dtype=dtype).element_size()
        size = rows * self.row_bytes
        self.shm = None
        self.tensor = None
        self.ptrs = None
        self.pointer = ctypes.c_void_p()

        cpu_group = group.cpu_group
        # 组长在通信组里的全局 rank 号（broadcast_object_list 的 src）。
        leader = dist.get_global_rank(cpu_group, 0)
        payload: list[str | None] = [None]
        if group.rank_in_group == 0:
            # 步骤1: 组长创建共享内存段，广播段名（或错误信息）。
            try:
                self.shm = shared_memory.SharedMemory(create=True, size=size)
                payload = [self.shm.name]
            except Exception as exc:  # noqa: BLE001 - reported to the group
                if self.shm is not None:
                    self.shm.unlink()
                    self.shm.close()
                payload = [f"ERROR: {type(exc).__name__}: {exc}"]
        dist.broadcast_object_list(payload, src=leader, group=cpu_group)
        name = payload[0]
        if name is None or name.startswith("ERROR:"):
            raise RuntimeError(f"Engram shared backing creation failed: {name}")

        error = None
        try:
            # 步骤2: 非组长 attach 到该段（拿到本进程的映射地址）。
            if self.shm is None:
                # Python 3.12 tracks attachments as owners. Match vLLM's
                # SharedMemory attach path so only the creator unlinks it.
                # 【中文】Python 3.12 会把 attach 记为 owner；这里 patch 掉
                # resource_tracker.register，对齐 vLLM 的 attach 路径——
                # 只有创建者负责 unlink。语法点: with patch(...) 临时替换函数。
                with patch("multiprocessing.resource_tracker.register", lambda *args, **kwargs: None):
                    self.shm = shared_memory.SharedMemory(name=name)
            assert self.shm.size >= size
            # 步骤3: 本进程映射地址注册给 CANN（映射+固定），取设备地址。
            address_of_mapping = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(self.shm.buf)))
            rc = self.lib.aclrtHostRegisterV2(address_of_mapping, size, ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED)
            if rc:
                raise RuntimeError(f"aclrtHostRegisterV2 failed: rc={rc} size={size}")
            # Registration succeeded: from here on the mapping owes exactly one
            # aclrtHostUnregister, and close() must keep the owner until it
            # returns success.  ``pointer`` is therefore set only
            # after the registration it has to undo exists.
            # 【中文】注册成功后才设置 pointer——它代表"欠一次 unregister"；
            # close() 失败时保留 pointer 以便重试。
            self.pointer = address_of_mapping
            address = ctypes.c_void_p()
            rc = self.lib.aclrtHostGetDevicePointer(self.pointer, ctypes.byref(address), 0)
            if rc:
                raise RuntimeError(f"aclrtHostGetDevicePointer failed: rc={rc}")
            # 步骤4: 零拷贝张量视图 + 每 2^22 行的设备指针表。
            self.tensor = torch.frombuffer(self.shm.buf, dtype=dtype, count=rows * row_elements).reshape(shape)
            self.ptrs = torch.tensor(
                [address.value + start * self.row_bytes for start in range(0, rows, CHUNK_ROWS)],
                dtype=torch.int64,
                device=device,
            )
        except Exception as exc:  # noqa: BLE001 - aggregated below
            error = f"{type(exc).__name__}: {exc}"

        # 步骤5: 全组聚合错误——任何 rank 失败则全员清理并抛错。
        errors: list[str | None] = [None] * group.world_size
        dist.all_gather_object(errors, error, group=cpu_group)
        failures = "; ".join(f"rank {rank}: {failure}" for rank, failure in enumerate(errors) if failure is not None)
        # Fence every mapping (or every failure) before the leader unlinks.
        # 【中文】barrier 确保所有映射（或失败）都完成，组长才能 unlink 段名。
        dist.barrier(group=cpu_group)
        if failures:
            try:
                self.close()
            except RuntimeError as exc:  # a rank that did register still owes it
                failures = f"{failures}; release failed: {exc}"
            if group.rank_in_group == 0:
                self._unlink()
            raise RuntimeError(f"Engram shared-memory initialization failed: {failures}")
        if group.rank_in_group == 0:
            # 全员就绪，组长 unlink 段名（段本体仍由各进程的映射维持存活）。
            self._unlink()

    def _unlink(self) -> None:
        """组长 unlink 共享内存段名（幂等；失败仅告警不抛错）。"""
        try:
            shm = self.shm
            if shm is not None:
                shm.unlink()
        except OSError as exc:
            logger.warning("Engram shared backing unlink failed: %s", exc)

    def close(self) -> None:
        """Unregister and close the mapping; shared memory is never aclrtFreeHost.

        Safe to call more than once.  A failed unregister raises and keeps the
        pointer, the CPU views and the device address table, so the caller can
        retry: the backing stays owned until CANN has really let go of it, the
        same contract as the private ``HostUvaBuffer``.  The
        mapping is only closed after the unregister succeeded -- or when
        there is nothing registered to begin with, e.g. after a failed
        ``aclrtHostRegisterV2``.
        """
        """【中文说明】注销注册并关闭映射；共享内存绝不能 aclrtFreeHost
        （它不是 aclrtMallocHost 分配的）。可重复调用。注销失败时抛错并
        保留 pointer/视图/指针表供重试——CANN 真正放手前持有底层的所有权
        （与私有 HostUvaBuffer 同一契约）。"""
        if self.pointer is not None and self.pointer.value:
            if self.ptrs is not None and self.ptrs.device.type == "npu":
                # 等待在飞的设备读取完成再注销。
                torch.npu.synchronize()
            rc = self.lib.aclrtHostUnregister(self.pointer)
            if rc:
                raise RuntimeError(f"aclrtHostUnregister failed for shared Engram: rc={rc}")
        self.pointer = ctypes.c_void_p()
        self.tensor = None
        self.ptrs = None
        if self.shm is not None:
            # 关闭本进程映射（段名已 unlink，映射关闭后段彻底释放）。
            self.shm.close()
            self.shm = None
