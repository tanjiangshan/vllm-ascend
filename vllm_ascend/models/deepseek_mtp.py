"""DeepSeek MTP（Multi-Token Prediction，多 token 预测）投机解码草稿模型的昇腾 NPU 适配。

模块职责与架构位置：
- 本模块位于 vllm-ascend 的 models 目录（硬件插件的"模型适配层"）。上游 vLLM
  已提供 DeepSeekMTP（DeepSeek-V3 风格 MTP 投机解码草稿模型）的通用实现；
  本模块通过继承并按需重写，补充昇腾 NPU 特有逻辑。
- MTP 投机解码原理：目标模型每个解码步都会产出隐状态 hidden_states；MTP 草稿
  模型将其与 token 嵌入拼接、投影后送入一个轻量 transformer 层，快速"预测"
  未来若干个 token（草稿），随后目标模型一次前向并行验证这些投机 token。
  若全部被接受，等效于一次前向产出多个 token，显著提升解码吞吐。
- QuaRot 旋转：当目标模型采用旋转量化（W4A8/W8A8 等）时，权重与激活会被同一
  正交矩阵旋转以抹平离群值；MTP 草稿必须工作在同一旋转坐标系中，因此本类
  额外引入可加载的 rot 线性层并做相应权重名重映射。
- 本模块还包含 AscendGlmMoeDsaForCausalLM：GLM MoE（DSA 注意力）目标模型在
  昇腾上的适配，重点是 PP（流水线并行）下 MoE 层数统计修正与 rot 权重清理。
"""

from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP
from vllm.model_executor.models.deepseek_v2 import GlmMoeDsaForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.sequence import IntermediateTensors

from vllm_ascend.utils import is_rot_weight_used


# @support_torch_compile 装饰器: 标记本模型支持 torch.compile 编译。
# 原理: vLLM 会据此对模型启用分段（piecewise）编译与 NPU Graph（CUDA Graph
# 的昇腾等价物）捕获，把多个小算子融合成整图执行，减少 Host-Device 交互与
# 算子下发开销。该装饰器只收集编译元信息，不改变 forward 语义。
@support_torch_compile
class AscendDeepSeekMTP(DeepSeekMTP):
    """DeepSeek MTP 投机解码草稿模型的昇腾适配类。

    继承自上游 vLLM 的 DeepSeekMTP（注册名 DeepSeekMTPModel /
    DeepseekV32MTPModel），复用其完整 MTP 前向与权重加载逻辑；本类新增：
    1) QuaRot 旋转支持——目标模型启用旋转量化时，构造 rot 线性层把输入
       隐状态变换到旋转后的坐标系；
    2) load_weights——用 WeightsMapper 把检查点 "rot." 前缀重映射到 MTP
       层内部的实际挂载路径；
    3) _maybe_set_own_lm_head——判定 MTP 是否携带独立输出头（若无则与
       目标模型共享 lm_head，节省显存并保证草稿/目标 logits 一致）。
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        """初始化 Ascend MTP 草稿模型。

        参数:
            vllm_config: vLLM 全局配置对象（聚合模型/量化/并行/投机解码配置）。
                注意 "*," 语法: 星号写在首个参数前，其后所有参数均为
                keyword-only 参数——调用时必须用关键字传参（vllm_config=...），
                可避免参数顺序写错，是 vLLM 模型构造器的统一约定。
            prefix: 本模块在整体模型中的名称前缀，用于层级命名与权重匹配。
        """
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # 步骤1: 探测目标模型是否启用 QuaRot 旋转量化。
        # 原理: 旋转量化通过可逆正交矩阵把权重/激活的离群值（outlier）能量
        # 摊平，使低比特（INT4/INT8）量化更精确；旋转后所有下游模块（包括
        # MTP 草稿）必须工作在同一旋转坐标系，输出才能与目标模型对齐。
        self.is_rot_weight_used = is_rot_weight_used(vllm_config)
        if self.is_rot_weight_used:
            # 步骤2: 构造旋转线性层 [hidden_size, hidden_size]（无偏置）。
            # 其权重来自检查点的 rot 矩阵（load_weights 时按 "rot." 前缀加载）。
            self.rot = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        """MTP 草稿模型前向计算。

        参数（含张量形状）:
            input_ids: 输入 token id，形状 [num_tokens]；投机解码时草稿主要
                消费目标模型隐状态，此参数允许为 None（类型注解
                "X | None" 是 PEP 604 写法，等价 Optional[X]）。
            positions: 每个位置对应的 RoPE 位置 id，形状 [num_tokens]。
            hidden_states: 目标模型输出隐状态，形状 [num_tokens, hidden_size]，
                MTP 的核心输入（投机解码时由目标模型上一步产生）。
            intermediate_tensors: PP（流水线并行）非首阶段接收到的中间张量
                集合；投机草稿通常单卡执行，默认 None。
            inputs_embeds: 预先计算好的输入嵌入，形状 [num_tokens, hidden_size]，
                提供时可跳过词表查表。
            spec_step_idx: 当前投机步编号（连续投机时第几个草稿 token），默认 0。

        返回:
            torch.Tensor: 草稿模型输出隐状态，形状 [num_tokens, hidden_size]，
            后续经 lm_head/logits_processor 得到投机 token 的分布。

        算法步骤:
            步骤1: 若启用 QuaRot，先把目标隐状态旋转到旋转坐标系；
            步骤2: 委托上游 DeepSeekMTP.forward 完成 MTP 层计算
            （enorm/hnorm 归一化 + eh_proj 拼接投影 + 单层 transformer）。
        """
        if self.is_rot_weight_used:
            # 步骤1: 旋转隐状态 [num_tokens, hidden_size]，对齐旋转量化后的
            # 目标模型坐标系（正交旋转可交换次序，故可只在此处旋转一次）。
            hidden_states = self.rot(hidden_states)
        # 步骤2: 调用上游 forward；位置参数依次为
        # (input_ids, positions, hidden_states, ...)，与上游签名一致。
        return super().forward(input_ids, positions, hidden_states, intermediate_tensors, inputs_embeds, spec_step_idx)

    # 中文说明（原英文 docstring 保留于下方）: 当检查点为 MTP 提供了独立的
    # 输出头权重时，把它暴露给投机解码 runner。DeepSeekMTP 总会构造
    # shared_head 模块，因此"模块存在"并不代表"MTP 拥有自己的头"；只有当
    # 该头的权重确实从检查点加载进来时，才视其为独立头，否则标记为与目标
    # 模型共享 lm_head（避免使用随机初始化的头输出）。
    def _maybe_set_own_lm_head(self, loaded_weights: set[str]) -> None:
        """Expose a checkpoint-provided MTP head to the runner.

        DeepSeekMTP always constructs ``shared_head``, so module existence does
        not prove head ownership. Treat it as independent only when its weight
        was loaded from the checkpoint.
        """
        # 步骤1: 计算 MTP 层的起始层号（MTP 层挂在目标层之后）。
        mtp_layer_idx = self.model.mtp_start_layer_idx
        # 步骤2: 拼出 shared_head 输出头权重在检查点中的完整参数名。
        own_head_weight = f"model.layers.{mtp_layer_idx}.shared_head.head.weight"
        if own_head_weight not in loaded_weights:
            # GLM-5.3 and friends ship no MTP head, leaving shared_head.head at
            # its allocation-time contents. Record that so the proposer shares
            # the target head instead of inspecting those values.
            # （GLM-5.3 等模型不带 MTP 头: shared_head.head 保持分配时的
            # 随机内容。记录 has_own_lm_head=False，让投机 proposer 改为
            # 共享目标模型的头，而不是去读那些无意义的初始值。）
            self.has_own_lm_head = False
            return

        # 步骤3: 检查点确实带了 MTP 头权重: 标记为独立头，并把本类的
        # lm_head 指向该模块，使 logits 计算直接复用它。
        self.has_own_lm_head = True
        mtp_layer = self.model.layers[str(mtp_layer_idx)]
        self.lm_head = mtp_layer.shared_head.head

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """加载 MTP 草稿模型权重（含旋转矩阵重映射与独立头判定）。

        参数:
            weights: 权重迭代器，每个元素是 (参数名, 张量) 二元组。类型
                Iterable[tuple[str, torch.Tensor]] 表示"由 (字符串, 张量)
                元组构成的可迭代对象"——vLLM 权重加载统一使用惰性迭代器，
                避免一次性把全部权重物化到内存。
        返回:
            set[str]: 成功加载的参数名集合（框架据此校验加载完整性）。

        算法步骤:
            步骤1: KV Cache 量化场景先应用 cache scale 映射；
            步骤2: 构造 WeightsMapper 把检查点 "rot." 前缀重写成 MTP 层
                内部路径 "model.layers.{num_hidden_layers}.rot."；
            步骤3: 委托上游 DeepSeekMTP.load_weights 完成实际加载；
            步骤4: 依据加载结果判定 MTP 是否拥有独立 LM 头。
        """
        # 步骤1: "(x := f())" 是海象运算符（walrus operator，PEP 572）:
        # 在条件表达式内部完成赋值并返回值，等价于先
        # cache_scale_mapper = ... 再判断是否为 None，但代码更紧凑。
        if self.quant_config is not None and (cache_scale_mapper := self.quant_config.get_cache_scale_mapper()):
            # 原理: KV Cache 量化（如 INT8 KV）需要按层/按头加载缩放因子；
            # cache_scale_mapper 把检查点里独立的 scale 张量名映射到对应
            # 注意力层的 k_scale/v_scale 参数上，无需改动模型结构。
            weights = cache_scale_mapper.apply(weights)

        # 步骤2: 权重名前缀映射: 检查点顶层 "rot." -> MTP 层内部路径。
        # 原理: WeightsMapper 是 vLLM 的通用权重名重写器，orig_to_new_prefix
        # 按前缀做字典替换，使不同命名约定的检查点对齐到当前模块树；
        # f-string 中的 {self.config.num_hidden_layers} 即 MTP 层号
        # （MTP 层紧接在最后一个目标层之后）。
        weights_mapper = WeightsMapper(
            orig_to_new_prefix={"rot.": f"model.layers.{self.config.num_hidden_layers}.rot."},
        )
        # 步骤3: 先应用映射再交给上游加载（上游负责 MTP 其余权重的
        # 投机层名改写与共享头/嵌入的加载）。
        loaded_weights = super().load_weights(weights_mapper.apply(weights))
        # 步骤4: 根据实际加载到的参数名集合，决定草稿是否共享目标 lm_head。
        self._maybe_set_own_lm_head(loaded_weights)
        return loaded_weights

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        """把投机(MTP)层权重名改写为当前模块树中的真实路径（额外兼容 rot）。

        参数:
            spec_layer: 投机层层号（通常等于 num_hidden_layers，即紧随目标层）。
            name: 原始权重名（可能是检查点命名或中间态命名）。
        返回:
            str: 改写后的权重名。
        原理: 上游方法负责把检查点里 MTP 层的 "model.layers.{N}." 风格名字
        映射到模型内部路径；本重写额外处理 rot 权重——把
        "model.layers.{N}.rot." 还原为顶层 "rot."，使权重能落到本类
        __init__ 中创建的 self.rot 模块上（与步骤2 的正向映射互逆）。
        """
        if "rot" in name:
            # 步骤1: rot 权重: 从 MTP 层内部路径还原为本类顶层 "rot."。
            name = name.replace(f"model.layers.{spec_layer}.rot.", "rot.")
            return name
        # 步骤2: 其余权重沿用上游的投机层名字改写规则。
        return super()._rewrite_spec_layer_name(spec_layer, name)


class AscendGlmMoeDsaForCausalLM(GlmMoeDsaForCausalLM):
    """GLM MoE（DSA 注意力）语言模型的昇腾 NPU 适配。

    继承上游 vLLM 的 GlmMoeDsaForCausalLM（GLM 系列混合专家模型，DSA 为其
    注意力变体；注册名 GlmMoeDsaForCausalLM），重写两个方法：
    1) __init__——在使用 v2 ModelRunner 且 PP（流水线并行）大于 1 时，
       按本进程实际持有的 MoE 层重新统计 num_moe_layers；
    2) load_weights——用 WeightsMapper 丢弃检查点中的 "rot." 旋转权重，
       再经 AutoWeightsLoader 完成通用递归加载。
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        """初始化 GLM MoE DSA 模型。

        参数:
            vllm_config: vLLM 全局配置（"*," 之后为 keyword-only 参数）。
            prefix: 模块名称前缀。
        """
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if vllm_config.use_v2_model_runner and vllm_config.parallel_config.pipeline_parallel_size > 1:
            # EPLB maps and expert weights must describe the same local layers.
            # （EPLB=Expert Parallelism Load Balancing，专家并行负载均衡:
            # 其专家重映射表必须与本进程实际持有的 MoE 层一一对应。PP>1 时
            # 每个 rank 只分到部分层，故需用 len(self.moe_layers) 重算本地的
            # MoE 层数，覆盖上游按全局配置得出的数值。）
            self.num_moe_layers = len(self.moe_layers)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """加载 GLM MoE DSA 权重（跳过不属于本模型的 rot 旋转矩阵）。

        参数:
            weights: (参数名, 张量) 二元组的可迭代序列（惰性读取检查点分片）。
        返回:
            set[str]: 成功加载的参数名集合。

        算法步骤:
            步骤1: 构造映射 "rot." -> None。WeightsMapper 把目标前缀映射为
                None 表示"丢弃该权重"——旋转矩阵属于量化方案而非模型参数，
                跳过加载以避免产生"意外权重"错误；
            步骤2: AutoWeightsLoader 按模块树递归匹配参数名并加载（自动
                处理 MoE 专家权重展开、packed 权重切分等）。
        """
        # 步骤1: 丢弃 rot 前缀权重（映射值为 None 即跳过）。
        mapper = WeightsMapper(orig_to_new_prefix={"rot.": None})
        # 步骤2: 通用加载器递归加载其余权重。
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=mapper)
