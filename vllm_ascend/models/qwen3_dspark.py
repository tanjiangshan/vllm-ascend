"""Qwen3 DSpark 草稿模型 —— 昇腾 NPU 适配版（含 QuaRot 权重对齐与置信头）。

【DSpark 原理】
DSpark 是块级（block-level）投机解码草稿模型：一次草拟一整块 token
（而非 EAGLE/MTP 的逐 token 串行草拟）。它消费目标模型若干层的辅助隐状态
（aux hidden states），把它们投影融合为草稿上下文，再对整块候选并行打分。
本文件在昇腾侧补充三件事：
1. QuaRot 旋转对齐（草稿输入投影 fc 与目标模型隐空间基对齐）；
2. 置信头（confidence head）推理：为每个草稿 token 输出接受概率，
   供 verify 阶段做自适应接受（高置信位置放行，低置信提前截止）；
3. 目标模型辅助隐状态捕获开关：让目标模型前向时物化 DSpark 所需的中间层输出。

继承上游 ``Qwen3DSparkForCausalLM``，复用其主体结构。
"""

import torch
from vllm.config import VllmConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkForCausalLM

from vllm_ascend.models.llama_eagle3 import load_quarot_target_layer
from vllm_ascend.utils import (
    get_rotation_matrix,
    get_rotation_path,
)

# 目标模型 checkpoint 中词嵌入可能出现的命名（纯文本模型与多模态模型前缀不同），
# 按顺序探测第一个命中的名字。
TARGET_EMBED_WEIGHT_NAMES = (
    "language_model.model.embed_tokens.weight",
    "model.embed_tokens.weight",
)
# 目标模型输出头的候选命名，同上。
TARGET_LM_HEAD_WEIGHT_NAMES = (
    "language_model.lm_head.weight",
    "lm_head.weight",
)


# Process the first linear weight with rotation matrix, if the target model uses rotary quantization
def process_weight(linear_weight: torch.Tensor, rotation_weight: torch.Tensor):
    """对线性权重做逐块右乘旋转矩阵（QuaRot 反旋转）。

    DSpark 草稿的 fc 投影输入宽度 = 目标隐状态层数 × hidden_size，
    而旋转矩阵 Q 只覆盖单个 hidden_size 段，因此按 hidden_size 分块循环旋转：
    W_aligned[:, s:s+H] = W[:, s:s+H] · Q。
    分块而非 block_diag 的好处：无需物化放大 3 倍的块对角矩阵，省显存。
    """
    # 断言输入宽度是 hidden_size 的整数倍，否则旋转段切分不完整（快速失败）。
    assert linear_weight.shape[1] % rotation_weight.shape[0] == 0, (
        f"Linear weight shape[1] must be a multiple of rotation weight shape[0],"
        f" but get {linear_weight.shape[1]=} and {rotation_weight.shape[0]=}"
    )
    # 统一到 FP32 精度做矩阵乘，避免 bf16 下误差累积。
    rotation_weight = rotation_weight.to(device=linear_weight.device, dtype=torch.float32)
    hidden_size = rotation_weight.shape[0]
    ori_dtype = linear_weight.dtype
    # 预分配结果缓冲区（empty 不做初始化，稍后逐块覆写）。
    processed_weight = torch.empty(linear_weight.shape, dtype=torch.float32, device=linear_weight.device)
    # 按列分块循环：每次旋转一个 hidden_size 宽的纵向切片。
    for start_pos in range(0, linear_weight.shape[1], hidden_size):
        linear_weight_chunked = linear_weight[:, start_pos : start_pos + hidden_size].to(torch.float32)
        processed_weight[:, start_pos : start_pos + hidden_size].copy_(
            torch.matmul(linear_weight_chunked, rotation_weight)
        )
    # 还原原始 dtype（通常 bf16）。
    return processed_weight.to(ori_dtype)


# @torch.no_grad()：装饰器，函数内禁用梯度记录（纯推理期权重变换）。
@torch.no_grad()
def align_draft_weights(model, projection, vllm_config):
    """Align draft inputs with the rotated target without modifying shared weights."""
    # QuaRot 对齐入口：只在目标模型使用旋转量化时生效（rotation_path 非空）。
    # 原则："不修改共享权重" —— 草稿与目标可能共享 embed/lm_head 实例，
    # 因此不能原地改共享矩阵，而是为草稿新建一份旋转对齐后的独立副本。
    rotation_path = get_rotation_path(vllm_config)
    if rotation_path is None:
        return
    rotation = get_rotation_matrix(rotation_path).cpu()
    # 步骤1：旋转草稿的 fc 输入投影（消费目标多层隐状态的第一个线性层），
    # 使草稿能直接消费旋转后的目标隐状态。先搬到 CPU 做变换再放回原设备。
    weight = projection.weight
    weight.copy_(process_weight(weight.cpu(), rotation).to(weight.device))
    target_config = vllm_config.model_config.hf_text_config
    # 步骤2：若草稿不拥有自己的 embed_tokens / lm_head（与目标共享），
    # 则从目标 checkpoint 取出对应权重、旋转对齐后装配为草稿私有层。
    # 元组字段：(宿主模块, 属性名, 层类, 目标权重候选名, "已拥有"标志属性名)。
    for owner, name, layer_cls, weight_names, own_flag in (
        (model.model, "embed_tokens", VocabParallelEmbedding, TARGET_EMBED_WEIGHT_NAMES, "has_own_embed_tokens"),
        (model, "lm_head", ParallelLMHead, TARGET_LM_HEAD_WEIGHT_NAMES, "has_own_lm_head"),
    ):
        # getattr 三参数形式：属性不存在时返回默认值 False，不抛异常。
        if getattr(model, own_flag, False):
            continue
        # torch.device 上下文：在其中创建的层会直接分配在指定设备上。
        with torch.device(weight.device):
            layer = layer_cls(target_config.vocab_size, target_config.hidden_size, params_dtype=weight.dtype)
        # 复用 llama_eagle3 的对齐加载器：读目标分片 + 反旋转 + 写入本层。
        load_quarot_target_layer(layer, vllm_config.model_config.model, weight_names, rotation, f"draft {name}.weight")
        # 量化方法钩子：权重装填后执行量化预处理（如按 per-channel 缩放重排）。
        layer.quant_method.process_weights_after_loading(layer)
        # setattr 动态挂载：把新的对齐层替换到草稿模型上，并标记"已拥有"，
        # 避免重复加载（该方法可能被多次调用）。
        setattr(owner, name, layer)
        setattr(model, own_flag, True)


class AscendQwen3DSparkForCausalLM(Qwen3DSparkForCausalLM):
    """Qwen3 DSpark 草稿模型（昇腾版）。

    注册名 "Qwen3DSparkModel" / "Qwen3OmniDSparkModel"（两者复用同一实现）。
    继承上游 ``Qwen3DSparkForCausalLM``，扩展：置信头前向、目标辅助隐状态
    捕获配置、加载后 QuaRot 对齐（post_process）。
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config = self.config
        # 置信头开关：checkpoint 未提供 confidence_head 时关闭，走普通 verify。
        self.enable_confidence_head = bool(getattr(config, "enable_confidence_head", False))

    def compute_confidence(self, head_hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        """Per-position acceptance probability for each drafted token."""
        # 置信头推理：sigmoid(confidence_head(h, e)) ∈ (0,1) 输出每个草稿 token
        # 被目标模型接受的概率估计。投机采样 verify 阶段可用它自适应决定
        # 接受长度（期望意义下更优），而非固定阈值。
        if not self.enable_confidence_head:
            raise RuntimeError("The DSpark confidence head is disabled.")
        assert self.model.confidence_head is not None
        return torch.sigmoid(self.model.confidence_head(head_hidden, markov_embed))

    def configure_target_aux_hidden_capture(self, target_model: torch.nn.Module) -> None:
        """Select draft auxiliary inputs, without changing target Eager/Graph mode."""
        # 打开目标模型的 DSpark 辅助隐状态"物化"开关：目标前向时把草稿所需的
        # 中间层输出缓存下来供草稿消费。目标模型可能是纯文本（直接有该接口）
        # 或多模态包装（需先 get_language_model() 拿到语言主干再找接口），
        # 因此逐级探测；接口不存在则静默跳过（目标不配合时由上层报错）。
        set_capture_mode = getattr(target_model, "set_dspark_aux_capture_materialized", None)
        if set_capture_mode is None:
            get_language_model = getattr(target_model, "get_language_model", None)
            if callable(get_language_model):
                set_capture_mode = getattr(get_language_model(), "set_dspark_aux_capture_materialized", None)
        if set_capture_mode is not None:
            set_capture_mode(True)

    def post_process(self, vllm_config: VllmConfig) -> None:
        # 权重加载后的收尾钩子：执行 QuaRot 对齐（fc 旋转 + 私有 embed/lm_head 装配）。
        align_draft_weights(self, self.model.fc, vllm_config)
