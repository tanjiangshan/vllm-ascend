# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# GLM-5.Next 模型专用的 NPU 算子封装包（ops）。
#
# 子模块清单：
#   - causal_conv1d.py   : KDA 层的短因果卷积（AscendC 自定义算子
#                          torch.ops._C_ascend.npu_causal_conv1d_custom 的
#                          Python 封装，处理非连续状态的中转拷贝）；
#   - fused_eh_norm.py   : MTP 输入融合内核（Triton：位置 0 置零 +
#                          enorm/hnorm 双 RMSNorm + 拼接，一次内核完成）；
#   - kda.py             : KDA 有界门控算子契约（recurrent_kda 递归 /
#                          chunk_kda 分块，封装 vllm_ascend.ops.kda 的
#                          AscendC 内核，含状态 gather/scatter 与回滚）；
#   - mhc_ops.py         : mHC 超连接的宽度扩展/收缩（hc_expand/hc_contract，
#                          纯形状操作）；
#   - state_ops.py       : KDA 循环状态的 gather/scatter（设备端掩码式，
#                          避免 host 同步，保持 ACL 图安全）。
#
# 本 __init__.py 不做显式导出；各模块按需被 import。
# =============================================================================
