"""Qwen3 DFlash2 草稿模型 —— 昇腾 NPU 适配实现。

【DFlash/DFlash2 原理】
DFlash 是面向投机解码的轻量草稿架构，两个核心机制：
1. 动态分组卷积（DFlashGroupedConv）：在注意力/MLP 前后各包一层
   "输入条件化"的分组因果卷积（prepare/finish 配对），卷积核 =
   静态基核 base_kernel + 由 kernel_projection(hidden) 动态生成的增量 δ，
   即核系数随输入内容自适应（类似 DyT/动态卷积思想），
   为无注意力的草稿层注入局部时序建模能力；
2. 候选选择器（CandidateSelector）：低秩 bigram 打分 —— 每个 token 在
   码本中有 predecessor/successor 两个低秩向量，边分数 =
   (pred_codebook[p] ⊙ W·h) · succ_codebook[c] + 一元 logits，
   用于在 top-k 候选格（lattice）中选出下一批候选 token。

DFlash2 相对 DFlash 的增强（本文件）：草稿共享目标模型 lm_head 做全词表
TopK 候选生成（compute_candidates），并支持 logit softcapping。

【NPU 适配点】
- torch._check 显式向 Dynamo 传播形状约束：NPU 的 unquantized_gemm
  fake 实现只接受 2D 输入，因此隐状态先展平投影再还原 [B, L] 形状；
- set_model_tag 隔离 torch.compile 图缓存（草稿图与选择器图分开编译）。
"""

import torch
import torch.nn.functional as F
from torch import nn
from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)
from vllm.model_executor.models.utils import maybe_prefix


def _grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    """块内因果分组卷积（动态核版），DFlash 草稿的核心算子。

    参数：
    - hidden_states: [N, hidden]（已展平的 token 流，N = B × 块内位置）；
    - delta: [N, num_groups] 由输入投影出的核增量（每组一个标量偏置）；
    - base: [taps, hidden] 静态基核（每个 tap 一套逐通道系数）；
    - block_size: 投机块大小（1 + num_speculative_tokens），卷积不跨块；
    - num_groups × group_size = hidden（分组深度卷积的组结构）；
    - taps: 卷积抽头数（看向过去的步数）。

    原理：输出[t] = Σ_tap coef[tap] ⊙ x[t-tap]，其中 coef = base + δ
    （δ 依赖当前输入 → 动态核）；位置 < tap 的块首 token 不回看（块内因果，
    用 position 掩码实现，避免看到上一投机块的尾巴）。
    """
    # 步骤1：把 hidden 维重排成 (num_groups, group_size) 的分组结构。
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    # 步骤2：动态核系数 = 静态基核 + 输入条件增量（广播到每个通道）。
    # base [taps, hidden] -> [1, taps, G, gs]；delta [N, G] -> [N, 1, G, 1]。
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    # 步骤3：tap-0 是纯逐点乘（无回看）。
    output = coefficients[:, 0] * blocks
    # 步骤4：计算每个 token 在投机块内的位置（0..block_size-1）。
    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    if block_size & (block_size - 1) == 0:
        # 位运算技巧：block_size 为 2 的幂时 x & (bs-1) 等价于 x % bs 且更快。
        position = position & (block_size - 1)
    else:
        position = position % block_size
    # 步骤5：高阶 tap —— 把序列向下平移 tap 位（当前 token 看第 t-tap 个 token），
    # 块首 position < tap 的位置用掩码清零（不跨块回看）。
    for tap in range(1, taps):
        # F.pad 第 0 维前补 tap 个零：实现序列右移（因果方向）。
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output += coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    # 步骤6：合并分组维，恢复 [N, hidden]。
    return output.flatten(-2)


class DFlashGroupedConv(nn.Module):
    """DFlash 动态分组卷积模块（权重容器 + prepare/finish 接口）。

    持有两套卷积（side 0/1）：prepare 在子层（注意力或 MLP）之前做卷积，
    finish 在子层之后做卷积 —— 两次卷积共享 kernel_projection 的输出，
    分别使用增量系数的第 0/1 份。
    """

    def __init__(
        self,
        hidden_size: int,
        taps: int,
        group_size: int,
        block_size: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        # 分组必须整除 hidden，否则 reshape 失败（快速失败）。
        if hidden_size % group_size:
            raise ValueError(f"conv_group_size={group_size} must divide hidden_size={hidden_size}.")
        self.block_size = block_size
        self.taps = taps
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        # 静态基核：[2(sides), taps, hidden]，requires_grad=False（推理-only 参数，
        # 用 nn.Parameter 是为了随模型 .to() 移动设备并被权重加载器识别）。
        self.base_kernel = nn.Parameter(
            torch.empty(2, taps, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        # 动态增量投影：hidden -> 2*taps*num_groups（两侧 × 各 tap × 各组的 δ 系数）。
        self.kernel_projection = ReplicatedLinear(
            hidden_size,
            2 * taps * self.num_groups,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "kernel_projection"),
            return_bias=False,
        )

    def _convolve(self, hidden_states: torch.Tensor, delta: torch.Tensor, side: int) -> torch.Tensor:
        """按 side（0=前卷积 / 1=后卷积）执行分组卷积。"""
        return _grouped_conv(
            hidden_states,
            delta,
            self.base_kernel[side],
            self.block_size,
            self.num_groups,
            self.group_size,
            self.taps,
        )

    def prepare(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """子层前卷积（side 0），同时返回 side 1 的动态系数供 finish 复用。

        一次 kernel_projection 同时算出两侧系数，避免重复投影：
        返回 (卷积后输出, finish 用的 δ 系数)。
        """
        coefficients = self.kernel_projection(hidden_states).reshape(
            hidden_states.shape[0], 2, self.taps, self.num_groups
        )
        return self._convolve(hidden_states, coefficients[:, 0], 0), coefficients[:, 1]

    def finish(self, hidden_states: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
        """子层后卷积（side 1），消费 prepare 返回的系数。"""
        return self._convolve(hidden_states, coefficients, 1)


class DFlash2Qwen3DecoderLayer(DFlashQwen3DecoderLayer):
    """DFlash2 草稿解码层：注意力与 MLP 各自被"前/后动态卷积"包裹。

    继承 DFlash 解码层（含 self_attn/mlp/norm），新增 attention_conv
    与 mlp_conv 两个 DFlashGroupedConv。
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config,
            config=config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        draft_config = config.dflash_config
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        # 卷积超参：抽头数/分组大小来自草稿配置；
        # block_size = 1 + 投机 token 数（草稿块宽度，卷积不跨块）。
        conv_args = dict(
            hidden_size=config.hidden_size,
            taps=int(draft_config["conv_kernel_size"]),
            group_size=int(draft_config["conv_group_size"]),
            block_size=1 + speculative_config.num_speculative_tokens,
            params_dtype=vllm_config.model_config.dtype,
        )
        # **dict 解包传参：两组卷积共用同一套超参，仅前缀不同。
        self.attention_conv = DFlashGroupedConv(**conv_args, prefix=maybe_prefix(prefix, "attention_conv"))
        self.mlp_conv = DFlashGroupedConv(**conv_args, prefix=maybe_prefix(prefix, "mlp_conv"))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """草稿层前向：norm → [卷积 → 子层 → 卷积] × (注意力, MLP) → 返回 (输出, 残差)。

        与普通解码层的差异：每个子层前后各一次动态分组卷积，
        prepare 返回的系数跨过子层传给 finish（一次投影两侧复用）。
        """
        if residual is not None:
            # 融合式 RMSNorm：一次完成残差相加 + 归一化。
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        else:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        # 注意力段：前卷积 → self_attn → 后卷积。
        hidden_states, coefficients = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states = self.attention_conv.finish(hidden_states, coefficients)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        # MLP 段：同构。
        hidden_states, coefficients = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, coefficients)
        return hidden_states, residual


def _score_edges(
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """低秩 bigram 边打分：为候选格（lattice）的每条 (前驱 p → 后继 c) 边算分。

    参数：
    - predecessor_table/successor_table: [vocab, rank] 码本（每个 token 两个低秩向量）；
    - candidate_ids: [B, L, K] 每步的 K 个候选 id；
    - unary_logits: [B, L, K] 候选的一元分数；
    - hidden: [B, L, rank] 当前隐状态的低秩投影；
    - anchor_token_ids: [B] 锚 token（块首的前驱）。

    返回 [B, L, K, K]：score[b,l,p,c] = unary + (pred_p ⊙ h) · succ_c，
    即"前驱候选 p 之后接后继候选 c"的亲和度，用于从格中选最优候选序列。
    """
    # 后继码本行：[B, L, K, rank]。
    successors = successor_table[candidate_ids]
    # 前驱 id 序列：位置 0 的前驱是锚 token，位置 l>0 的前驱是 l-1 位的候选。
    # expand 广播锚 id 到 top_k 份（位置 0 的 K 个候选共享同一锚前驱）。
    predecessor_ids = torch.cat(
        (
            anchor_token_ids[:, None, None].expand(-1, 1, top_k),
            candidate_ids[:, :-1],
        ),
        dim=1,
    )
    predecessors = predecessor_table[predecessor_ids]
    # einsum "blpr,blcr->blpc"：对 rank 维求内积；
    # predecessors * hidden[:, :, None] 先做逐元素门控（隐状态调制前驱向量），
    # 再与后继向量点积 —— 双线性形式的低秩 bigram 打分。
    return unary_logits[:, :, None] + torch.einsum("blpr,blcr->blpc", predecessors * hidden[:, :, None], successors)


# @support_torch_compile：候选选择器整体可被 torch.compile 编译（纯张量运算）。
@support_torch_compile
class CandidateSelector(nn.Module):
    """DFlash2 候选选择器：维护 pred/succ 码本并执行 bigram 边打分。"""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        rank: int,
        top_k: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        # 前驱/后继码本：[vocab, rank]，推理-only 参数（加载自 checkpoint）。
        self.predecessor_codebook = nn.Parameter(torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False)
        self.successor_codebook = nn.Parameter(torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False)
        # 隐状态低秩投影：hidden -> rank（打分在与码本同维的低秩空间进行）。
        self.hidden_projection = ReplicatedLinear(
            hidden_size,
            rank,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "hidden_projection"),
            return_bias=False,
        )

    def forward(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """候选格边打分。返回 [B, L, K, K] 的边分数矩阵。

        candidate_ids: [B, L, K]；unary_logits: [B, L, K]；
        hidden_states: [B, L, H]；anchor_token_ids: [B]。
        """
        # hidden_states is [num_reqs, num_steps, H]. NPU unquantized_gemm's Dynamo
        # fake is 2D-only, so flatten to [tokens, H], project, then restore (B, L).
        # torch._check ties independently marked dynamic batch dims so
        # predecessors [B, L, K, R] can broadcast with projected hidden.
        # （NPU 的 unquantized_gemm 的 Dynamo fake 只支持 2D：先展平投影再还原形状；
        #   torch._check 向编译器断言 B/L 维一致，使广播合法。）
        torch._check(hidden_states.shape[0] == candidate_ids.shape[0])
        torch._check(hidden_states.shape[1] == candidate_ids.shape[1])
        hidden = self.hidden_projection(hidden_states.flatten(0, 1))
        hidden = hidden.view(*hidden_states.shape[:-1], -1)
        return _score_edges(
            self.predecessor_codebook,
            self.successor_codebook,
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
            self.top_k,
        )


class DFlash2Qwen3Model(DFlashQwen3Model):
    """DFlash2 草稿模型主体。

    # vLLM #52816 switched the parent constructor from its module global to
    # this class factory. Declare the main-lane factory so DFlash2 decoder
    # layers are built.
    # （vLLM #52816 把父类构造从模块全局改为类工厂属性；
    #   在此声明主车道工厂，确保构建的是 DFlash2 解码层。）
    """
    decoder_layer_cls = DFlash2Qwen3DecoderLayer

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )

        draft_config = self.config.dflash_config
        # 嵌入缩放系数：草稿嵌入乘以固定 scale（部分 checkpoint 的训练设定）。
        self.input_embedding_scale = float(draft_config.get("input_embedding_scale", 1.0))
        # Draft load uses set_model_tag("eagle_head"); without a distinct tag the
        # selector's @support_torch_compile backend shares that npugraph cache and
        # unpacks the draft graph (67 values) with the selector's 9 inputs.
        # （草稿加载用 "eagle_head" 标签；不给选择器单独打标签会共用 npugraph
        #   编译缓存，导致 67 值的草稿图被按选择器的 9 输入错误解包。）
        with set_model_tag("dflash2_candidate_selector"):
            self.candidate_selector = CandidateSelector(
                hidden_size=self.config.hidden_size,
                vocab_size=self.config.vocab_size,
                rank=int(draft_config["selector_rank"]),
                top_k=int(draft_config["selector_top_k"]),
                params_dtype=vllm_config.model_config.dtype,
                prefix=maybe_prefix(prefix, "candidate_selector"),
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 嵌入后乘缩放系数（super() 复用上游嵌入逻辑）。
        return super().embed_input_ids(input_ids) * self.input_embedding_scale


class DFlash2Qwen3ForCausalLM(DFlashQwen3ForCausalLM):
    """DFlash2 草稿模型顶层（注册名 "DFlash2DraftModel"）。

    # vLLM #52816 likewise routes the draft model through this class factory.
    # （同样通过类工厂路由草稿模型。）
    """
    model_cls = DFlash2Qwen3Model

    # Share the target LM head so compute_candidates can top-k the full vocab.
    # （与目标模型共享 lm_head，使 compute_candidates 能做全词表 TopK。）
    has_own_lm_head = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        draft_config = self.config.dflash_config
        # 输出分数的整体乘数（校准草稿分数与目标 logits 的量级）。
        self.output_multiplier = float(draft_config.get("output_multiplier", 1.0))
        # logit softcap（Gemma 风格）：logits = tanh(x/c)·c，限制极端值；
        # 配置缺省/非正时不启用。
        softcap = float(draft_config.get("final_logit_softcapping") or 0.0)
        self.final_logit_softcapping = softcap if softcap > 0 else None

    def compute_candidates(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """全词表 TopK 候选生成：返回 (候选 id, 候选分数)。

        用共享的目标 lm_head 对草稿隐状态做全词表投影，取 top_k 个候选
        作为候选格的节点（供 CandidateSelector 做边打分与序列搜索）。
        hidden_states: [num_tokens, hidden]。
        """
        # 草稿要求 lm_head 未量化：TopK 需要精确的全词表 logits。
        if not isinstance(self.lm_head.quant_method, UnquantizedEmbeddingMethod):
            raise ValueError("DFlash2 requires an unquantized target LM head for candidate TopK.")

        selector = self.model.candidate_selector
        # 直接调用量化方法的 apply（绕过 LogitsProcessor 的采样逻辑，只要原始 logits）。
        logits = self.lm_head.quant_method.apply(self.lm_head, hidden_states, bias=None)
        # 词表并行补齐的 padding 位置置 -inf，排除出 TopK。
        num_pad = self.lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")
        # 本 rank 局部 TopK。
        values, ids = torch.topk(logits, selector.top_k, dim=-1)
        # 局部 id 加回本 rank 的词表起点，转成全局 id。
        ids = ids.to(torch.int64) + self.lm_head.shard_indices.org_vocab_start_index

        # TP > 1 时：各 rank 的局部 TopK all_gather 汇总后再全局 TopK。
        if get_tensor_model_parallel_world_size() > 1:
            values = tensor_model_parallel_all_gather(values, dim=-1)
            ids = tensor_model_parallel_all_gather(ids, dim=-1)
            # 拼接后再次 TopK 取全局前 K，gather 按索引同步 id。
            values, selected = torch.topk(values, selector.top_k, dim=-1)
            ids = ids.gather(-1, selected)

        # 分数校准：乘 output_multiplier + 可选 tanh softcap。
        values = values.float() * self.output_multiplier
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            values = torch.tanh(values / cap) * cap
        return ids, values
