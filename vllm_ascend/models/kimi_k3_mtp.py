# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 MTP draft model for Ascend.

Kimi K3 的 MTP（Multi-Token Prediction，多 Token 预测）草稿模型 —— 昇腾 NPU 适配版。

【在插件架构中的位置】
vllm-ascend 作为 vLLM 的硬件插件，不重复实现完整模型，而是继承上游 vLLM 的
``vllm.models.kimi_k3.amd.mtp``（AMD 实现被当作"参考实现"复用其前向逻辑），
只把草稿层内部的解码器容器替换成 NPU 版本 ``AscendKimiDecoderLayer``。

【MTP 投机解码原理】
- 草稿模型（draft）与目标模型（target）共享词表和部分结构；
- 解码时草稿模型以极小代价连续"猜"出 k 个候选 token；
- 目标模型一次前向并行验证这 k 个 token（verify），接受最长正确前缀；
- 接受的 token 直接产出，被拒绝处由目标模型自身采样重写，
  从而把"逐 token 自回归"变成"逐块验证"，显著提升解码吞吐。

【为什么需要重写 __init__ 而不是直接继承】
上游构造函数把 AMD 版解码层"硬编码"进了模块树，无法通过参数注入替换。
因此这里显式调用 ``nn.Module.__init__(self)`` 初始化裸容器
（跳过上游父类构造逻辑，避免构建出无用的 AMD 子模块），
再按相同的结构手工搭建，但关键组件换成 Ascend 实现；
前向计算（forward）逻辑则完全复用上游父类。
"""

import copy

from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.utils import maybe_prefix
# 注意导入路径：上游 vLLM 把 Kimi K3 的参考实现放在 vllm.models.kimi_k3.amd 下，
# 这里以 Upstream* 别名导入，表明"复用上游 + 换芯"的适配思路。
from vllm.models.kimi_k3.amd.mtp import (
    KimiK3MTP as UpstreamKimiK3MTP,
)
from vllm.models.kimi_k3.amd.mtp import (
    KimiK3MultiTokenPredictor as UpstreamKimiK3MultiTokenPredictor,
)
from vllm.models.kimi_k3.amd.mtp import (
    KimiK3MultiTokenPredictorLayer as UpstreamKimiK3MultiTokenPredictorLayer,
)
from vllm.models.kimi_k3.amd.mtp import SharedHead

from vllm_ascend.models.kimi_k3 import AscendKimiDecoderLayer


class AscendKimiK3MultiTokenPredictorLayer(
    UpstreamKimiK3MultiTokenPredictorLayer,
):
    """Kimi K3 的单个 MTP 草稿层（昇腾版）。

    继承上游 ``KimiK3MultiTokenPredictorLayer`` 以复用其 forward 路径，
    仅替换构造逻辑：把硬编码的 AMD 解码层换成 ``AscendKimiDecoderLayer``。

    层结构（DeepSeek-V3 式 MTP 草稿层）：
    - ``enorm`` / ``hnorm``：分别对"当前 token 嵌入"和"目标模型隐状态"做 RMSNorm；
    - ``eh_proj``：把两者拼接后投影回 hidden_size，作为草稿层的输入；
    - ``mtp_block``：一个完整的解码器子层（注意力 + MLP），是草稿计算的主体；
    - ``shared_head``：输出头（与目标模型共享 lm_head 时可省一份大矩阵）。
    """

    def __init__(self, config, vllm_config: VllmConfig, prefix: str) -> None:
        # The upstream constructor hard-codes the AMD decoder layer.  Build the
        # same container with the Ascend decoder and inherit its forward path.
        # （上游构造函数硬编码了 AMD 解码层；这里改用昇腾解码层搭建同样的容器，
        #   forward 路径直接继承上游实现。）
        nn.Module.__init__(self)
        self.config = config
        # enorm/hnorm：MTP 输入两路（token 嵌入 e 与目标隐状态 h）各自的 RMSNorm。
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # eh_proj：[N, 2*hidden] -> [N, hidden]，融合两路输入供草稿层消费。
        self.eh_proj = nn.Linear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
        )
        self.shared_head = SharedHead(
            config=config,
            prefix=prefix,
            quant_config=vllm_config.quant_config,
        )
        # copy.copy 是浅拷贝：只复制配置对象本身，不深拷贝嵌套字段，
        # 避免为草稿层额外复制整份配置内存。
        block_config = copy.copy(config)
        # 草稿层禁用 attn_res_block_size（注意力残差分块），保持逐 token 语义。
        block_config.attn_res_block_size = None
        # 核心替换点：用昇腾版 Kimi 解码层承载草稿层的主干计算。
        self.mtp_block = AscendKimiDecoderLayer(
            block_config,
            vllm_config,
            prefix=prefix,
        )


class AscendKimiK3MultiTokenPredictor(UpstreamKimiK3MultiTokenPredictor):
    """MTP 草稿模型主体：持有若干个草稿层 + 自己的词嵌入 + logits 处理器。

    继承上游 ``KimiK3MultiTokenPredictor`` 复用 forward；
    构造时同样绕开父类 __init__，把预测器层类换成上面的 Ascend 版本。
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # The upstream constructor hard-codes its predictor-layer class.
        # （上游构造函数硬编码了预测器层类，这里替换为 Ascend 版本。）
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        self.config = config
        # 草稿层的编号策略：紧跟在目标模型之后，从 num_hidden_layers 开始编号。
        # 这样 checkpoint 中 "model.layers.<idx>" 的命名空间与目标模型不冲突。
        self.mtp_start_layer_idx = config.num_hidden_layers
        # num_nextn_predict_layers：MTP 草稿层数量（DeepSeek 系惯例命名）。
        self.num_mtp_layers = config.num_nextn_predict_layers
        # nn.ModuleDict 必须用字符串键；多步投机解码时按
        # "spec_step_idx % num_mtp_layers" 轮转选用草稿层。
        self.layers = nn.ModuleDict(
            {
                str(idx): AscendKimiK3MultiTokenPredictorLayer(
                    config,
                    vllm_config,
                    f"{prefix}.layers.{idx}",
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        # VocabParallelEmbedding：词嵌入按 TP 切分到各 rank，避免单卡放不下大词表。
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        # LogitsProcessor：负责从隐状态采样/裁剪 logits（含采样参数应用）。
        self.logits_processor = LogitsProcessor(config.vocab_size)


class AscendKimiK3MTP(UpstreamKimiK3MTP):
    """Kimi K3 MTP 草稿模型顶层包装（注册名 "KimiK3MTPModel"）。

    组合 ``AscendKimiK3MultiTokenPredictor`` 并暴露统一的 forward 接口，
    供 vLLM 投机解码调度器（EAGLE 式 propose/verify 循环）调用。
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.config = vllm_config.model_config.hf_text_config
        self.quant_config = vllm_config.quant_config
        self.model = AscendKimiK3MultiTokenPredictor(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
