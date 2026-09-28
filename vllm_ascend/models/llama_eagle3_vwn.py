"""EAGLE3 VWN 草稿模型（Llama 底座）—— 昇腾 NPU 适配实现。

【VWN 原理（Variable-Width Network，可变宽度网络草稿层）】
普通解码层只有一条宽度为 hidden_size 的残差流；VWN 草稿层额外维护一条
更宽的"旁路流"（宽度 wd = hidden_size × r），两条流通过 upward/downward
投影互通：
- upward：窄流 → 宽流（扩充信息带宽）；
- downward：宽流 → 窄流（回收压缩）；
- downward_and_forgot：一次投影同时产出"子层输入（hs）+ 宽流残差（wd）"，
  拼接输出后 split 成两段，等价于两个投影融合成一个 GEMM。

m 参数（vwn_m）实现"分组共享权重"技巧：view(-1, hs//m) 把 [N, hs] 重排成
[N*m, hs/m]，让同一个 (hs/m → wd/m) 投影并行处理 m 个分组，
以块对角结构等效放大投影宽度，同时减少参数量。

【EAGLE3 接口】
草稿消费目标模型 3 层辅助隐状态（aux hidden states），经 fc 投影融合为
草稿输入；支持缩减词表（draft_vocab_size）+ id 映射回目标词表；
支持 parallel_drafting（并行多草稿）。本实现仅 layer 0 执行 VWN 计算，
其余层直通（草稿通常只配 1 层）。
"""

import torch
import torch.nn as nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.llama_eagle3 import (
    Eagle3LlamaForCausalLM,
)
from vllm.model_executor.models.llama_eagle3 import (
    LlamaDecoderLayer as Eagle3LlamaDecoderLayer,
)
from vllm.model_executor.models.llama_eagle3 import (
    LlamaModel as Eagle3LlamaModel,
)
from vllm.model_executor.models.utils import get_draft_quant_config, maybe_prefix


def _linear(inp, out, vc, qc, pfx):
    """ReplicatedLinear 工厂：构建"每 rank 复制完整权重"的线性层。

    草稿层的小投影（fc/upward/downward 等）不做 TP 切分（切分收益小于
    通信开销），ReplicatedLinear 保证所有 rank 持有相同副本。
    """
    return ReplicatedLinear(
        input_size=inp,
        output_size=out,
        bias=False,
        params_dtype=vc.model_config.dtype,
        quant_config=qc,
        prefix=pfx,
        return_bias=False,
    )


class PreVwnLayerV1(nn.Module):
    """VWN 草稿的预处理层（第 0 层专用）。

    把"token 嵌入 embeds"与"目标模型隐状态 hidden_states"两路输入融合，
    投影到窄流宽度后再 upward 升到宽流，供后续 VWN 解码层消费：
    x = fc(cat(norm(e), norm(h)))；wider = upward(x)。
    """

    def __init__(self, vllm_config, prefix="", config=None, quant_config=None):
        super().__init__()
        cfg = config or vllm_config.model_config.hf_config
        # getattr 带默认值：vwn_m/vwn_r 是可选超参，缺省 1（退化为无分组/等宽）。
        hs, m, r = cfg.hidden_size, getattr(cfg, "vwn_m", 1), getattr(cfg, "vwn_r", 1)
        wd = int(hs * r)
        self.m, self.hidden_size, self.wider_dim = m, hs, wd
        self.input_layernorm = RMSNorm(hs, eps=cfg.rms_norm_eps)
        self.hidden_norm = RMSNorm(hs, eps=cfg.rms_norm_eps)
        # fc：[N, 2*hs] -> [N, hs]，融合两路输入。
        self.fc = _linear(2 * hs, hs, vllm_config, quant_config, maybe_prefix(prefix, "fc"))
        # upward：hs -> wd（分组共享权重，实际 GEMM 形状 hs/m -> wd/m）。
        self.upward = _linear(hs // m, wd // m, vllm_config, quant_config, maybe_prefix(prefix, "upward"))

    def forward(self, embeds, hidden_states):
        """embeds/hidden_states: [N, hs]。返回宽流初始状态 [N, wd]。"""
        x = self.fc(torch.cat([self.input_layernorm(embeds), self.hidden_norm(hidden_states)], dim=-1))
        # 分组技巧：[N, hs] -> [N*m, hs/m] 过共享投影后 view 回 [N, wd]，
        # 等价于 m 个块各自做 hs/m → wd/m 的映射（权重共享的块对角 GEMM）。
        return self.upward(x.view(-1, self.hidden_size // self.m)).view(-1, self.wider_dim)


class VwnLlamaDecoderLayer(Eagle3LlamaDecoderLayer):
    """VWN 草稿解码层（继承 EAGLE3 Llama 解码层）。

    在标准 Llama 解码层（self_attn + mlp + 两个 norm）之上叠加 VWN 双流
    投影：pre_vwn_layer（仅入口用）、upward/downward 系列。
    注意：forward 中只有 layer_idx == 0 执行完整 VWN 计算，其余层直通。
    """

    def __init__(self, vllm_config, prefix="", config=None, layer_idx=0):
        super().__init__(vllm_config, prefix=prefix, config=config, layer_idx=layer_idx)
        cfg = config or vllm_config.model_config.hf_config
        qc = self.get_quant_config(vllm_config)
        m, r = getattr(cfg, "vwn_m", 1), getattr(cfg, "vwn_r", 1)
        hs, wd = self.hidden_size, int(self.hidden_size * r)
        self.m, self.wider_dim, self.layer_idx = m, wd, layer_idx

        if layer_idx == 0:
            # 第 0 层重建 qkv_proj：EAGLE3 上游构造的投影面向"拼接输入"宽度，
            # VWN 第 0 层输入是纯 hidden_size，需要标准形状的 QKV 融合投影。
            # QKVParallelLinear：Q/K/V 按 TP 切分（head 维度切分），
            # 一个 GEMM 同时产出 Q、K、V 三段（列拼接），减少 kernel 启动。
            self.self_attn.qkv_proj = QKVParallelLinear(
                hs,
                self.self_attn.head_dim,
                self.self_attn.total_num_heads,
                self.self_attn.total_num_kv_heads,
                bias=getattr(cfg, "attention_bias", False),
                quant_config=qc,
                prefix=maybe_prefix(prefix, "self_attn.qkv_proj"),
            )

        # 局部别名缩短后续构造代码。
        mp = maybe_prefix
        self.pre_vwn_layer = PreVwnLayerV1(vllm_config, mp(prefix, "layers.pre_vwn_layer"), cfg, qc)
        # downward_and_forgot：宽流一次投影出 [子层输入 hs | 宽流残差 wd]，
        # 输出宽度 (hs + wd) // m（分组），等价两个投影融合成一个 GEMM。
        self.downward_and_forgot = _linear(wd // m, (hs + wd) // m, vllm_config, qc, mp(prefix, "downward_and_forgot"))
        self.pre_attention_layernorm = RMSNorm(hs, eps=cfg.rms_norm_eps)
        # 注意力后的 upward：把注意力输出升回宽流。
        self.upward_after_attn = _linear(hs // m, wd // m, vllm_config, qc, mp(prefix, "upward_after_attn"))
        self.downward_and_forgot_after_attn = _linear(
            wd // m, (hs + wd) // m, vllm_config, qc, mp(prefix, "downward_and_forgot_after_attn")
        )
        self.post_attention_layernorm = RMSNorm(hs, eps=cfg.rms_norm_eps)
        # MLP 后的 upward 与最终 downward（宽流 → 窄流输出）。
        self.upward_after_mlp = _linear(hs // m, wd // m, vllm_config, qc, mp(prefix, "upward_after_mlp"))
        self.downward = _linear(wd // m, hs // m, vllm_config, qc, mp(prefix, "downward"))

    def forward(self, positions, embeds, hidden_states, residual):
        """VWN 解码层前向（仅第 0 层执行实际计算）。

        双流数据流（wider = 宽流 [N, wd]，hidden = 窄流 [N, hs]）：
        1) 入口：embeds + 目标隐状态 → pre_vwn_layer → wider；
        2) 注意力段：wider → downward_and_forgot → (hidden, res)；
           hidden 过 norm + self_attn；结果 upward 升宽 + 残差 res → wider；
        3) MLP 段：同构再来一次（downward_and_forgot_after_attn → mlp → upward）；
        4) 出口：downward 把宽流压回窄流 hidden_states 输出。
        """
        if self.layer_idx == 0:
            hs, wd, m = self.hidden_size, self.wider_dim, self.m
            # 步骤1：入口融合（嵌入 + 目标隐状态）→ 宽流初始状态。
            wider = self.pre_vwn_layer(embeds, hidden_states)
            # Attention
            # 步骤2：宽流 → [子层输入 | 宽流残差]，split 拆成两段。
            out = self.downward_and_forgot(wider.view(-1, wd // m)).view(-1, hs + wd)
            hidden, res = out.split([hs, wd], dim=-1)
            # 窄流过注意力（标准 pre-norm + self_attn）。
            hidden = self.self_attn(positions=positions, hidden_states=self.pre_attention_layernorm(hidden))
            # 注意力输出升宽 + 宽流残差（res 在宽流空间上累加）。
            wider = self.upward_after_attn(hidden.view(-1, hs // m)).view(-1, wd) + res
            # MLP
            out = self.downward_and_forgot_after_attn(wider.view(-1, wd // m)).view(-1, hs + wd)
            hidden, res = out.split([hs, wd], dim=-1)
            wider = (
                self.upward_after_mlp(self.mlp(self.post_attention_layernorm(hidden)).view(-1, hs // m)).view(-1, wd)
                + res
            )
            # Downward
            # 步骤3：宽流压回窄流，作为本层输出（residual 保持外层传入值不变）。
            hidden_states = self.downward(wider.view(-1, wd // m)).view(-1, hs)
        return hidden_states, residual


# @support_torch_compile：标记该类可被 torch.compile 编译加速；
# dynamic_arg_dims 声明哪些输入的哪个维度是动态的（编译图允许变化）：
# input_ids 第 0 维（token 数）、positions 最后一维动态，避免重编译。
@support_torch_compile(dynamic_arg_dims={"input_ids": 0, "positions": -1, "hidden_states": 0, "input_embeds": 0})
class VwnLlamaModel(Eagle3LlamaModel):
    """VWN 草稿模型主体（继承 EAGLE3 LlamaModel）。

    组件：embed_tokens、若干 VwnLlamaDecoderLayer、可选的 aux 隐状态融合层
    （input_norm/fc —— 消费目标模型 3 层隐状态拼接 [N, 3*hs] → [N, hs]）。
    """

    def __init__(self, *, vllm_config, start_layer_id=0, prefix=""):
        # 显式初始化 nn.Module 基类（跳过上游构造，自建 Ascend/草稿专用组件）。
        nn.Module.__init__(self)
        # 草稿配置来自 speculative_config.draft_model_config（与目标解耦）。
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.vocab_size = self.config.vocab_size
        self.quant_config = get_draft_quant_config(vllm_config)

        # EAGLE3 开关：是否消费目标模型的辅助隐状态（默认开启）。
        eagle_config = getattr(self.config, "eagle_config", None)
        if eagle_config is not None and "use_aux_hidden_state" in eagle_config:
            self.use_aux_hidden_state = eagle_config["use_aux_hidden_state"]
        else:
            self.use_aux_hidden_state = True
        # norm_before_fc：fc 之前是否先做一次整体 RMSNorm（两种 checkpoint 变体）。
        self.norm_before_fc = getattr(self.config, "norm_before_fc", False)

        vc = get_current_vllm_config()
        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        # 草稿层编号从 start_layer_id（= 目标层数）开始，避免与目标层命名冲突。
        self.layers = nn.ModuleList(
            [
                VwnLlamaDecoderLayer(vc, maybe_prefix(prefix, f"layers.{i + start_layer_id}"), self.config, layer_idx=i)
                for i in range(self.config.num_hidden_layers)
            ]
        )
        if self.use_aux_hidden_state:
            # EAGLE3 拼接 3 层目标隐状态：输入宽度 = 3 × hidden_size
            # （target_hidden_size 与目标一致时用目标宽度）。
            if hasattr(self.config, "target_hidden_size"):
                fc_input_size = self.config.target_hidden_size * 3
            else:
                fc_input_size = self.config.hidden_size * 3
            if self.norm_before_fc:
                self.input_norm = RMSNorm(
                    fc_input_size,
                    eps=self.config.rms_norm_eps,
                )
            else:
                self.input_norm = None

            self.fc_norm = None
            self.num_aux_hidden_states = 3
            # fc：[N, 3*hs] -> [N, hs]，把多层目标隐状态融合成草稿输入。
            self.fc = ReplicatedLinear(
                input_size=fc_input_size,
                output_size=self.config.hidden_size,
                bias=False,
                params_dtype=vllm_config.model_config.dtype,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "fc"),
                return_bias=False,
            )
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

    def forward(self, input_ids, positions, hidden_states, input_embeds=None):
        """草稿主体前向。返回 (最终隐状态, 倒数层隐状态)。

        hidden_states 是目标模型透传的 aux 隐状态（供第 0 层融合）。
        注意返回第二个值为未做最终 norm 的 hidden_states（EAGLE3 协议：
        上层用它可以做特征级验证/融合）。
        """
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions, embeds=input_embeds, hidden_states=hidden_states, residual=residual
            )
        # 融合"残差相加 + 最终归一化"，一次算子完成。
        return self.norm(hidden_states, residual), hidden_states


class Eagle3VwnLlamaForCausalLM(Eagle3LlamaForCausalLM):
    """EAGLE3 VWN 草稿模型顶层（注册名 "LlamaForCausalLMVwnEagle3"）。

    继承上游 ``Eagle3LlamaForCausalLM`` 复用投机解码接口；
    模型主体换成上面的 VwnLlamaModel。支持缩减词表草稿
    （draft_vocab_size + draft_id_to_target_id 映射）与 parallel_drafting。
    """

    def __init__(self, *, vllm_config, prefix=""):
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        # 草稿词表缺省时退化为完整词表（不做缩减）。
        if getattr(self.config, "draft_vocab_size", None) is None:
            base_vocab_size = getattr(self.config, "vocab_size", None)
            self.config.draft_vocab_size = base_vocab_size

        # 记录目标层数：草稿层编号从 n 开始。
        n = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        self.config.target_layer_count = n

        self.model = VwnLlamaModel(vllm_config=vllm_config, prefix="model", start_layer_id=n)

        # lm_head 基于草稿词表（可能是目标词表的高频子集，矩阵更小更快）。
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            quant_config=get_draft_quant_config(vllm_config),
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size,
            scale=logit_scale,
        )
        # 草稿词表 id → 目标词表 id 的映射表（可学习张量，requires_grad=False
        # 仅作查表；值在加载 checkpoint 时填充）。
        self.draft_id_to_target_id = nn.Parameter(
            torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
            requires_grad=False,
        )

        # 并行草稿模式：一次前向同时产出多个候选（而非串行自回归）。
        self.use_parallel_drafting = vllm_config.speculative_config.parallel_drafting

        if self.use_parallel_drafting:
            # mask_hidden：并行草稿时用于屏蔽 aux 隐状态的固定形状缓冲。
            # register_buffer(persistent=False)：注册为 buffer（随 .to() 移动、
            # 不计入 state_dict，避免污染 checkpoint 保存/加载）。
            self.register_buffer(
                "mask_hidden",
                torch.zeros(
                    1,
                    (3 if self.model.use_aux_hidden_state else 1) * self.config.hidden_size,
                ),
                persistent=False,
            )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """草稿 logits 计算并散射回目标词表空间。

        草稿用缩减词表时，logits 仅覆盖 draft_vocab_size 个 id；
        目标模型验证时需要完整词表空间的 logits 做精确比较，因此把
        草稿 logits 按 draft_id_to_target_id 映射散射到 [N, vocab_size]，
        未映射位置填 -inf（softmax 后概率为 0，等效于"草稿不支持该 token"）。
        """
        logits = self.logits_processor(self.lm_head, hidden_states)
        if self.draft_id_to_target_id is None:
            assert logits.shape[1] == self.config.vocab_size, (
                f"Expected logits to have shape (*, {self.config.vocab_size}), but got {logits.shape}"
            )
            return logits

        # 目标词表 id = 草稿 id + 偏移表（draft_id_to_target_id 存的是差值）。
        base = torch.arange(self.config.draft_vocab_size, device=logits.device)
        targets = base + self.draft_id_to_target_id
        # 初始化全 -inf 再按 targets 散射写入（scatter 的高级索引赋值）。
        logits_new = logits.new_full(
            (
                logits.shape[0],
                self.config.vocab_size,
            ),
            float("-inf"),
        )
        logits_new[:, targets] = logits
        return logits_new

    def combine_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """融合 EAGLE3 的多层辅助隐状态为草稿输入。

        不用 aux 隐状态时原样返回；否则可选 norm_before_fc（整体一次 RMSNorm）
        或 fc_norm（每段独立 RMSNorm，本实现未启用，保留接口），
        最后经 fc 投影 [N, 3*hs] → [N, hs]。
        """
        if not self.model.use_aux_hidden_state:
            return hidden_states
        # combine multiple auxiliary hidden states returned by eagle3
        # （融合 EAGLE3 返回的多路辅助隐状态。）

        if self.model.norm_before_fc:
            hidden_states = self.model.input_norm(hidden_states)

        # `norm_before_fc` adds a single RMSNorm before the FC layer, whereas `fc_norm`
        # applies separate RMSNorms to each chunk of the hidden states.
        # （norm_before_fc 是 FC 前做一次整体 RMSNorm；fc_norm 则对各段分别归一化。）
        if self.model.fc_norm is not None:
            chunks = hidden_states.chunk(self.model.num_aux_hidden_states, dim=-1)
            hidden_states = torch.cat(
                [norm(chunk) for norm, chunk in zip(self.model.fc_norm, chunks)],
                dim=-1,
            )

        return self.model.fc(hidden_states)
