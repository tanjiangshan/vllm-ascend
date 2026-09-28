# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# 【engram 子包总览】engram 是 DeepSeek V4.1 的"程序性记忆"模块：
# 一种基于 n-gram 哈希检索的持久记忆机制。工作流程（与主网络的交互）：
#   1) hash_state: 对输入 token 序列做多阶 n-gram 多项式哈希（每层每头
#      一组素数取模哈希），历史来源依次为当前 chunk → runner 回看窗口
#      → SWA slot cache；
#   2) embedding: 以哈希值为行号查询 INT8+组32缩放的大嵌入表（每层一张，
#      数十 GiB 级，可 TP×EDP 头分片、可卸载到主机内存）得到记忆向量；
#   3) layer: wkv 把记忆行投影成 key/value，engram_gate 用旋转基下的
#      相似度算门控，把 value 门控写回该层的多支路残差流。
# 效果：模型可凭 n-gram 匹配"回忆"训练语料中的长程模式，属于跨层
# （配置指定若干 engram 层）、跨请求（slot cache 持久）的检索式记忆。
#
# 【模块划分（与上游 vLLM 对齐）】common=平台无关的门控/资格判定；
# hash_state=哈希状态（Triton-Ascend kernel）；embedding=嵌入表与加载；
# layer=门控层；npu=NPU 侧存储/查找（设备表/主机 UVA/共享内存）；
# parallel=EDP 数据并行交换辅助。
# =============================================================================
"""Ascend Engram for DeepSeek V4.1.

Split the way upstream vLLM splits it for other accelerators: ``common`` holds
the platform-independent n-gram hashing and gating, ``npu`` holds the storage
and lookup the NPU runs.
"""

# 【公共接口导出】__all__ 声明 from ... import * 时暴露的名单；
# 下游（model.py）从本包导入 engram 开关判定、哈希状态构造、死掩码、
# CPU 卸载判定等。
from .common import engram_enabled, engram_gate
from .hash_state import AscendEngramSlotCache, create_engram_hash_state, engram_dead_mask
from .npu import engram_cpu_offload

__all__ = [
    "AscendEngramSlotCache",
    "create_engram_hash_state",
    "engram_cpu_offload",
    "engram_dead_mask",
    "engram_enabled",
    "engram_gate",
]
