"""EAGLE3 Llama 投机解码草稿模型 —— 昇腾 NPU 适配版（含 QuaRot 旋转对齐）。

【EAGLE3 原理】
EAGLE3 是自回归投机解码（speculative decoding）的草稿架构：
- 草稿模型每步基于"目标模型多层隐状态 + 已生成的 token 嵌入"预测下一个 token；
- 与 EAGLE/EAGLE-2 不同，EAGLE3 直接消费目标模型**低/中/高多个层**的隐状态
  （feature-level 融合），信息量更大，草稿接受率显著提升；
- 草稿连续生成 k 个候选 token，目标模型一次前向并行验证并接受最长正确前缀，
  从而把逐 token 解码加速为逐块解码。

【QuaRot 旋转对齐原理】
若目标模型经过 QuaRot 量化训练/变换，其隐空间被一个正交旋转矩阵 Q 旋转过
（W' = W·Q^T，h' = Q·h）。EAGLE3 草稿模型若要与目标模型共享隐状态/词表，
两侧必须处于同一隐空间基（basis）。本文件在加载权重时做"反旋转"对齐：
- fc 层（消费目标隐状态的投影）：W_draft_aligned = W_draft · block_diag(Q,Q,Q)；
- embed_tokens：E_aligned = E · Q^T；
- 若草稿 checkpoint 缺词嵌入，则从目标模型 checkpoint 里取出嵌入并旋转对齐。

【在插件架构中的位置】
继承上游 vLLM 的 ``Eagle3LlamaForCausalLM``，仅在权重加载阶段
注入 QuaRot 旋转处理；前向计算完全复用上游实现（NPU 上无需特殊算子）。
"""

import json
import logging
import os
from collections.abc import Iterable
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM

from vllm_ascend.utils import (
    get_rotation_matrix,
    get_rotation_path,
)

logger = logging.getLogger(__name__)


def get_embedding_tensor(directory_path):
    """Scans the directory and returns the first tensor found that contains 'embed' in its key."""
    # 扫描 checkpoint 目录，返回第一个 key 中含 'embed' 的张量（用于补齐草稿模型缺失的词嵌入）。
    if not os.path.isdir(directory_path):
        return None
    for filename in os.listdir(directory_path):
        if filename.endswith(".safetensors"):
            file_path = os.path.join(directory_path, filename)
            # load_file 会把整个分片加载进内存；此处只在找不到 index 时兜底使用。
            state_dict = load_file(file_path)
            for key, tensor in state_dict.items():
                if "embed" in key.lower():
                    return tensor
    return None


def _find_safetensors_weight(
    model_path: Path,
    weight_names: tuple[str, ...],
) -> tuple[Path, str]:
    """Locate one target tensor without loading unrelated checkpoint shards."""
    # 定位目标张量所在的分片文件，且不加载无关分片（大模型 checkpoint 动辄数百 GB 分片，
    # 全量扫描代价高）。策略：优先查 *.safetensors.index.json 的 weight_map（O(1) 定位），
    # 没有 index 时才逐分片打开看 key 集合（只读 key 元数据，不读张量数据）。
    for index_path in sorted(model_path.glob("*.safetensors.index.json")):
        with index_path.open(encoding="utf-8") as index_file:
            weight_map = json.load(index_file).get("weight_map", {})
        for weight_name in weight_names:
            # 海象运算符 :=：在条件表达式中赋值并立即使用，等价于
            # shard_name = weight_map.get(weight_name); if shard_name: ...
            if shard_name := weight_map.get(weight_name):
                return model_path / shard_name, weight_name

    # 兜底路径：无 index 文件时逐分片检查（safe_open 只读元数据即可列出 keys）。
    for shard_path in sorted(model_path.glob("*.safetensors")):
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            shard_keys = set(shard.keys())
        for weight_name in weight_names:
            if weight_name in shard_keys:
                return shard_path, weight_name

    raise KeyError(f"None of {weight_names!r} was found in the target checkpoint at {model_path}.")


# @torch.inference_mode()：装饰器，函数内关闭梯度追踪与版本计数，
# 推理期权重加载的标准做法（比 no_grad 更彻底，省去 autograd 记录开销）。
@torch.inference_mode()
def load_quarot_target_layer(
    layer: nn.Module,
    target_model_path: Path | str,
    weight_names: tuple[str, ...],
    rotation: torch.Tensor,
    label: str,
) -> None:
    """Load one target vocab shard into the draft's unrotated hidden basis."""
    # 从目标模型 checkpoint 取出一个"词表维度"权重分片（lm_head 或 embed_tokens），
    # 施加旋转对齐后写入草稿模型对应层 —— 使草稿的输出头/嵌入与目标模型隐空间一致。
    target_model_path = Path(target_model_path)
    # 步骤1：定位目标权重所在的 safetensors 分片。
    shard_path, weight_name = _find_safetensors_weight(
        target_model_path,
        weight_names,
    )
    # 步骤2：确定本 rank 负责的词表行区间。词表并行（TP）时每个 rank 只持有
    # 一个切片，shard_indices 记录了该切片在原始（未切分）词表中的起止位置。
    shard_indices = getattr(layer, "shard_indices", None)
    if shard_indices is None:
        # 无切分信息：整表加载。
        start_index = 0
        end_index = layer.weight.shape[0]
    else:
        start_index = shard_indices.org_vocab_start_index
        end_index = shard_indices.org_vocab_end_index

    # 步骤3：按行区间读取目标权重切片。
    # get_slice 返回惰性切片视图，只把需要的行读入内存（大词表友好）。
    with safe_open(shard_path, framework="pt", device="cpu") as shard:
        target_weight = shard.get_slice(weight_name)[start_index:end_index]

    # 步骤4：旋转对齐。统一升到 FP32 做矩阵乘（bf16 下大矩阵乘误差会被放大），
    # 再转回目标设备。W_aligned = W_target · Q^T（把目标隐空间旋回草稿的基）。
    rotation = rotation.to(
        device=layer.weight.device,
        dtype=torch.float32,
    )
    target_weight = target_weight.to(
        device=layer.weight.device,
        dtype=torch.float32,
    )
    aligned_weight = torch.matmul(target_weight, rotation.T)
    # 步骤5：写入草稿层权重。若词表并行补齐（padded vocab）导致本层行数多于
    # 实际加载行数，多余行清零（padding 行不参与有效计算）。
    loaded_rows = aligned_weight.shape[0]
    layer.weight.data[:loaded_rows].copy_(aligned_weight.to(layer.weight.dtype))
    layer.weight.data[loaded_rows:].zero_()
    logger.info(
        "[spec_decode/quarot] Loaded and aligned %s from %s (%s).",
        label,
        shard_path.name,
        tuple(layer.weight.shape),
    )


def compute_rotation_matrix3(Q: torch.Tensor) -> torch.Tensor:
    """Anti-rotate matrix for 3 layers of hidden_states."""
    # EAGLE3 拼接目标模型 3 个层的隐状态（low/mid/high），每段都需要独立旋转一次。
    # torch.block_diag(Q, Q, Q) 构造块对角矩阵：
    #   [Q 0 0]
    #   [0 Q 0]   —— 拼接向量 [h1; h2; h3] 乘以它等价于分别旋转三段再拼回，
    #   [0 0 Q]      一次 GEMM 完成三次旋转。
    return torch.block_diag(Q, Q, Q)


class AscendEagle3LlamaForCausalLM(Eagle3LlamaForCausalLM):
    """EAGLE3 Llama 草稿模型（昇腾版，注册名 "Eagle3LlamaForCausalLM"）。

    继承上游 ``Eagle3LlamaForCausalLM``，仅扩展权重加载：
    当检测到 QuaRot 旋转矩阵（rotation_path 非空）时，对草稿权重做旋转对齐，
    使草稿与目标模型共享同一隐空间；否则完全走上游加载路径。
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # 目标模型路径：QuaRot 对齐需要回读目标模型的嵌入/输出头权重。
        self.target_model_path = Path(vllm_config.model_config.model)
        # 旋转矩阵文件路径（由 VLLM_ASCEND_* 环境变量/配置决定，见 utils.get_rotation_path）。
        self.rotation_path = get_rotation_path(vllm_config)
        self.is_quarot_used = self.rotation_path is not None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """权重加载入口：QuaRot 模式下先旋转对齐再交给上游加载器。

        weights 是 (参数名, 张量) 的可迭代流（vLLM 权重加载协议），
        各模型按需消费/重映射，未被消费的权重代表 checkpoint 冗余。
        """
        if self.is_quarot_used:
            # 步骤1：读取旋转矩阵 Q 并构造 3 层拼接用的块对角旋转 Q3。
            Q = get_rotation_matrix(self.rotation_path)
            Q3 = compute_rotation_matrix3(Q)
            # 步骤2：确定嵌入的目标 dtype（config.dtype 可能是 "bfloat16" 字符串或 torch.dtype）。
            if isinstance(self.config.dtype, str):
                embed_dtype = getattr(torch, self.config.dtype)
            else:
                embed_dtype = self.config.dtype
            processed_weights: list[tuple[str, torch.Tensor]] = []
            includes_embed_tokens = False
            # 步骤3：逐项流式处理 —— fc 层（EAGLE3 消费目标隐状态的投影）右乘 Q3
            # 完成反旋转；FP32 计算后再还原原 dtype，避免精度损失。
            for name, loaded_weight in weights:
                if "fc." in name:
                    dtype = loaded_weight.dtype
                    loaded_weight = (loaded_weight.to(torch.float32) @ Q3.to(torch.float32)).to(dtype)
                if "embed_tokens" in name:
                    includes_embed_tokens = True
                processed_weights.append((name, loaded_weight))

            # 步骤4：草稿 checkpoint 不含词嵌入时，从目标模型取嵌入并旋转
            # （E · Q^T），作为 "embed_tokens.weight" 补给草稿模型。
            if not includes_embed_tokens:
                embed_weight = (
                    get_embedding_tensor(self.target_model_path).to(torch.float32) @ Q.T.to(torch.float32)
                ).to(embed_dtype)
                processed_weights.append(("embed_tokens.weight", embed_weight))
            # 步骤5：交给上游 EAGLE3 加载器完成最终装配。
            super().load_weights(processed_weights)
        else:
            # 非 QuaRot 场景：直接走上游加载路径。
            super().load_weights(weights)
