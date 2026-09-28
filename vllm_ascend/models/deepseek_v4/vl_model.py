# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
# ============================================================================
# 【模块职责】DeepSeek-V4-Flash-Vision-Exp 多模态模型的昇腾 NPU 顶层包装。
# 处理器（processor）、哨兵 token 布局（sentinel layout）、ViT 与 Aligner
# 均与上游 vLLM 实现（vllm-project/vllm#54566）共享；本模块只包含昇腾侧
# 的“语言主干集成”与“权重加载边界”:
#   - 视觉塔（vision/aligner/哨兵嵌入）为复制式，不参与 TP 切分;
#   - 语言主干复用 AscendDeepseekV4ForCausalLM（见 model.py）;
#   - load_weights 把 checkpoint 权重按命名空间分发给视觉塔与语言主干。
# 多模态推理张量流: patches -> ViT -> Aligner -> 图像嵌入 -> 按哨兵掩码
# 替换/合并进 inputs_embeds -> 语言主干解码。
# ============================================================================
"""Ascend wrapper for DeepSeek-V4-Flash-Vision-Exp.

The processor, sentinel layout, ViT, and aligner are shared with the upstream
vLLM implementation from vllm-project/vllm#54566. This module contains only
the Ascend language-backbone integration and weight-loading boundary.
"""

from collections.abc import Iterable, Iterator

import torch
from torch import nn
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_ascend.models.deepseek_v4.mm_preprocess import (
    IMAGE_PLACEHOLDER,
    IMAGE_SENTINEL_BASE_ID,
    DeepseekV4VLDummyInputsBuilder,
    DeepseekV4VLMultiModalProcessor,
    DeepseekV4VLProcessingInfo,
    image_sentinel_mask,
)
from vllm_ascend.models.deepseek_v4.model import AscendDeepseekV4ForCausalLM
from vllm_ascend.models.deepseek_v4.vision import (
    DeepseekV4Aligner,
    DeepseekV4ViT,
)


def _vision_parameter_name(name: str) -> str | None:
    """Map a checkpoint vision tensor to the wrapper parameter namespace."""

    # 【中文】判断一条 checkpoint 权重是否属于视觉塔；是则返回映射后的
    # 参数名，否则返回 None（表示属于语言主干）。
    # 语法点: str | None 是“联合类型注解”（Python 3.10+ 写法，
    # 等价 Optional[str]），表示返回值可能是字符串或 None。
    # 步骤1: 剥掉 "model." 前缀 —— 语言主干在本类中以 language_model.model.*
    # 命名，而 checkpoint 中视觉权重也可能带 model. 前缀。
    if name.startswith("model."):
        name = name.removeprefix("model.")
    # 步骤2: 以这些前缀开头的视为视觉塔参数（vision./aligner./image_* 哨兵）。
    # startswith 接受元组，表示“匹配任一前缀即真”。
    if name.startswith(("vision.", "aligner.", "image_")):
        return name
    # 步骤3: 其余权重归属语言主干。
    return None


# 装饰器语法点: @MULTIMODAL_REGISTRY.register_processor(...) 把下面的类注册
# 进 vLLM 的多模态注册表，绑定三件套——
#   processor: 多模态预处理器（构建哨兵块、替换占位符）;
#   info    : 处理信息类（配置读取、token 上限计算）;
#   dummy_inputs: 哑输入构造器（ACL Graph Capture / profile run 用）。
@MULTIMODAL_REGISTRY.register_processor(
    DeepseekV4VLMultiModalProcessor,
    info=DeepseekV4VLProcessingInfo,
    dummy_inputs=DeepseekV4VLDummyInputsBuilder,
)
class AscendDeepseekV4ForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsEagle3,
):
    """DeepSeek-V4 vision entry point using the Ascend text backbone."""

    # 【中文补充】多模态条件生成模型的昇腾入口类。多重继承的各混入
    # （mixin）接口声明能力:
    #   - SupportsMultiModal: 支持多模态输入（实现 embed_multimodal 等）;
    #   - SupportsPP        : 支持流水线并行（提供 intermediate_tensors 协议）;
    #   - SupportsEagle3    : 支持作为 EAGLE3 风格投机采样的目标模型
    #                         （暴露 MTP 目标隐状态、专家映射等接口）。

    # 语法点: 类属性。要求调度器传入“原始 input_ids”（而非已合并的
    # inputs_embeds），因为哨兵 token 的嵌入替换必须在模型内部完成。
    requires_raw_input_tokens = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        """返回指定模态在 prompt 中的占位符字符串。

        Args:
            modality: 模态名（仅支持 "image"）。
            i: 第几个占位符（本模型所有图像共用同一占位符，故忽略）。
        Returns:
            占位符字符串 "<｜deepseek_image｜>"。
        Raises:
            ValueError: 不支持的模态。
        """
        # del i 显式丢弃参数（占位符与序号无关）。
        del i
        if modality == "image":
            return IMAGE_PLACEHOLDER
        raise ValueError(f"Unsupported modality: {modality!r}")

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        """初始化多模态包装模型。

        Args:
            vllm_config: vLLM 全局配置。签名中的 * 表示其后所有参数
                （vllm_config、prefix）为 keyword-only，必须按关键字传递。
            prefix: 模块名前缀。

        结构: 视觉塔（复制式） + 哨兵嵌入参数 + 语言主干
              AscendDeepseekV4ForCausalLM。
        """
        super().__init__()
        model_config = vllm_config.model_config
        config = model_config.hf_config
        # 步骤1: 多模态场景的压缩对齐配置——哨兵块起始位置需对齐到
        # COMPRESS_PAD_TO 的倍数，避免压缩组跨越“图像块/文本”边界。
        if getattr(config, "vision_n_layers", 0) > 0:
            config.mm_prefix_clamp_sliding_window = True
            config.mm_prefix_span_leading_pad_modulus = 4
        self.config = config
        self.multimodal_config = model_config.multimodal_config
        assert self.multimodal_config is not None

        # 步骤2: 判断图像输入是否启用（配置有视觉层 且 每条 prompt 允许
        # 至少 1 张图像）。get_limit_per_prompt 返回该模态的配额。
        image_enabled = config.vision_n_layers > 0 and self.multimodal_config.get_limit_per_prompt("image") > 0
        # 步骤3: with 语法——在上下文管理器内创建视觉子模块，便于框架
        # 标记“塔模型”参数（影响 TP/PP 划分与权重加载分组）。
        with self._mark_tower_model(vllm_config, {"image"}):
            # 先声明全部可选属性为 None（类型注解 A | None 表示可空），
            # 保证禁用图像时访问这些属性也不会 AttributeError。
            self.vision: DeepseekV4ViT | None = None
            self.aligner: DeepseekV4Aligner | None = None
            self.image_start: nn.Parameter | None = None
            self.image_end: nn.Parameter | None = None
            self.image_newline: nn.Parameter | None = None
            self.image_pad: nn.Parameter | None = None
            if image_enabled:
                # 视觉塔 + 对齐器（复制式，不切 TP）。
                self.vision = DeepseekV4ViT(config)
                self.aligner = DeepseekV4Aligner(config)
                # 五个可学习哨兵嵌入: 图像块首/填充/换行/块尾 + pad，
                # 维度 = hidden_size，float32 存储（与语言模型 dtype 解耦）。
                for name in (
                    "image_start",
                    "image_end",
                    "image_newline",
                    "image_pad",
                ):
                    setattr(
                        self,
                        name,
                        nn.Parameter(torch.empty(config.hidden_size, dtype=torch.float32)),
                    )
                # 塔权重统一转换为模型运行 dtype（如 bfloat16）。
                self.vision.to(dtype=model_config.dtype)
                self.aligner.to(dtype=model_config.dtype)

        # 步骤4: 在“语言模型”上下文中创建文本主干（复用 model.py 的实现）。
        with self._mark_language_model(vllm_config):
            self.language_model = AscendDeepseekV4ForCausalLM(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )
        # 步骤5: 把语言主干的 PP 中间张量构造器提升到本类（PP 调度器
        # 直接调用本类的该方法）。
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    def _parse_and_validate_image_input(self, **kwargs: object) -> dict | None:
        """从 forward 的关键字参数中解析并校验图像输入。

        Args:
            **kwargs: 语法点——收集所有剩余关键字参数为字典。
                含 patches（ViT patch 像素）、vit_grid/llm_grid（网格尺寸）、
                perm（排列索引）。
        Returns:
            图像输入字典；无图像时返回 None。
        Raises:
            ValueError: 缺少必要字段时。
        """
        # kwargs.pop(name, None): 弹出指定键（不存在时返回默认 None），
        # 既取值又把它从 kwargs 中移除。
        patches = kwargs.pop("patches", None)
        if patches is None:
            return None
        vit_grid = kwargs.pop("vit_grid", None)
        llm_grid = kwargs.pop("llm_grid", None)
        perm = kwargs.pop("perm", None)
        if vit_grid is None or llm_grid is None or perm is None:
            raise ValueError("DeepSeek-V4 vision input requires patches, vit_grid, llm_grid, and perm.")
        return {
            "patches": patches,
            "vit_grid": vit_grid,
            "llm_grid": llm_grid,
            "perm": perm,
        }

    def _process_image_input(
        self,
        patches: torch.Tensor,
        vit_grid: torch.Tensor,
        llm_grid: torch.Tensor,
        perm: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """逐图运行视觉塔 + 对齐器，并按 perm 重排为最终 N 布局顺序。

        Args:
            patches: [sum(n_vit_h*n_vit_w), 3, p, p] 所有图像 patch 拼接。
            vit_grid: [num_images, 2] 每张图的 ViT 网格 [n_vit_h, n_vit_w]。
            llm_grid: [num_images, 2] 每张图的 LLM 网格 [n_llm_h, n_llm_w]。
            perm: [sum(n_llm_h*n_llm_w)] 把 Aligner 输出重排到哨兵块
                最终顺序的索引。

        Returns:
            元组，每个元素为一张图的嵌入 [n_llm_h*n_llm_w, hidden_size]。
        """
        assert self.vision is not None and self.aligner is not None
        # 权重 dtype 对齐（patch 输入可能来自 CPU/其他 dtype）。
        patches = patches.to(self.aligner.w1.weight.dtype)

        embeds: list[torch.Tensor] = []
        # 游标: patches 按 ViT 网格拼接、perm 按 LLM 网格拼接，
        # 逐图处理时需要分别记录当前图的起始偏移。
        vit_offset = 0
        llm_offset = 0
        # zip(..., strict=True): 语法点——严格模式 zip，两序列长度不等时
        # 抛 ValueError，防止静默截断。tolist() 把网格张量转成 Python 列表。
        for (n_vit_h, n_vit_w), (n_llm_h, n_llm_w) in zip(vit_grid.tolist(), llm_grid.tolist(), strict=True):
            n_vit = n_vit_h * n_vit_w
            n_llm = n_llm_h * n_llm_w
            # 步骤1: 切出当前图的 patch -> ViT 编码 -> Aligner 投影，
            # 得到 [n_llm_h*n_llm_w, hidden_size]（行主序 N 布局）。
            image_embeds = self.aligner(
                self.vision(
                    patches[vit_offset : vit_offset + n_vit],
                    n_vit_h,
                    n_vit_w,
                ),
                n_vit_h,
                n_vit_w,
            )
            # 步骤2: 用该图的 perm 索引重排嵌入——mm_preprocess 构造的
            # 哨兵块内部是“两行交错”的特殊 N 布局，perm 把 Aligner 的
            # 行主序输出映射到该布局的 IMAGE 槽位顺序。
            item_perm = perm[llm_offset : llm_offset + n_llm].to(image_embeds.device)
            embeds.append(image_embeds[item_perm])
            # 步骤3: 前移两个游标。
            vit_offset += n_vit
            llm_offset += n_llm
        return tuple(embeds)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """计算多模态（图像）嵌入——SupportsMultiModal 接口方法。

        Args:
            **kwargs: 含图像输入字段（patches/vit_grid/llm_grid/perm）。
        Returns:
            图像嵌入元组（每图一个张量）；无图像或视觉塔未启用时返回 []。
        """
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None or self.vision is None:
            return []
        return self._process_image_input(
            image_input["patches"],
            image_input["vit_grid"],
            image_input["llm_grid"],
            image_input["perm"],
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """构造最终输入嵌入: 词嵌入 + 哨兵嵌入替换 + 图像嵌入合并。

        原理: prompt 中每个 <｜deepseek_image｜> 已被预处理器展开为一段
        哨兵 token（IMAGE_START/IMAGE_PAD/IMAGE/IMAGE_NEWLINE/IMAGE_END），
        对应 id = IMAGE_SENTINEL_BASE_ID + type。这里:
          1) 查表得到 5 种哨兵的可学习嵌入（table）;
          2) 对所有哨兵位置按 type 查表替换词嵌入;
          3) 再把视觉塔输出嵌入按 is_multimodal 掩码覆盖到 IMAGE 位置。

        Args:
            input_ids: [num_tokens] token id 序列。
            multimodal_embeddings: 图像嵌入元组（可空）。
            is_multimodal: [num_tokens] bool 掩码，True 处将被图像嵌入覆盖
                （keyword-only 参数）。
        Returns:
            inputs_embeds: [num_tokens, hidden_size]。
        """
        # 局部导入避免模块级循环依赖（_merge_multimodal_embeddings 在
        # vllm.model_executor.models.utils 中）。
        from vllm.model_executor.models.utils import (
            _merge_multimodal_embeddings,
        )

        # 步骤1: 常规词嵌入查表。
        inputs_embeds = self.language_model.embed_input_ids(input_ids)
        if self.image_start is not None:
            # 步骤2: 计算哨兵位置掩码（id 落在保留区间内的 token）。
            sentinel_mask = image_sentinel_mask(input_ids)
            if is_multimodal is not None:
                # 若提供了 is_multimodal 掩码，从哨兵掩码中剔除将被
                # 图像嵌入覆盖的 IMAGE 位（~ 按位取反后按位与）。
                sentinel_mask = sentinel_mask & ~is_multimodal.to(input_ids.device)
            # 步骤3: 构造 5 行查表矩阵，行序与 IMAGE_START..IMAGE_END 对应:
            # [image_start, image_pad, image_pad, image_newline, image_end]。
            # 注意第 2/3 行都是 image_pad —— type==IMAGE(2) 的位置稍后由
            # 图像嵌入覆盖，这里先填 pad 保证无空档。
            table = torch.stack(
                [
                    self.image_start,
                    self.image_pad,
                    self.image_pad,
                    self.image_newline,
                    self.image_end,
                ]
            ).to(inputs_embeds.dtype)
            # 步骤4: (input_ids - BASE).clamp(0,4) 把哨兵 id 映射到 0..4
            # 行号（非哨兵位置被 clamp 后结果无意义，但会被掩码屏蔽）;
            # torch.where 按掩码逐位置选择: 哨兵处用 table[idx]，否则词嵌入。
            idx = (input_ids - IMAGE_SENTINEL_BASE_ID).clamp(0, 4)
            inputs_embeds = torch.where(sentinel_mask.unsqueeze(-1), table[idx], inputs_embeds)

        # 步骤5: 无图像嵌入时直接返回。
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        if is_multimodal is None:
            raise ValueError("is_multimodal is required when merging image embeddings.")
        # 步骤6: 把图像嵌入覆盖到 is_multimodal=True 的位置。
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """前向: 直接委托给语言主干（视觉计算已在 embed 阶段完成）。

        Args:
            input_ids: [num_tokens] token id。
            positions: [num_tokens] 位置 id。
            intermediate_tensors: PP 并行时来自上一流水级的中间张量。
            inputs_embeds: 预合并的输入嵌入（多模态路径）。
        Returns:
            隐状态或（PP 非最后一级的）中间张量。
        """
        # del kwargs 丢弃多余参数（视觉字段已在 embed 阶段消费）。
        del kwargs
        return self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """由隐状态计算 logits（委托语言主干）。Args: hidden_states:
        [num_tokens, hidden_size]。Returns: [num_tokens, vocab_size] 或 None。"""
        return self.language_model.compute_logits(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """返回 MoE 专家参数映射（(param_name, weight_name, expert_id,
        shard_id) 四元组列表），供 EPLB/权重加载使用（委托语言主干）。"""
        return self.language_model.get_expert_mapping()

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        """返回 MTP 草稿所需的“hc_head 前”目标隐状态缓冲
        （SupportsEagle3 接口，委托语言主干）。"""
        return self.language_model.get_mtp_target_hidden_states()

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        """设置需要导出辅助隐状态的层号集合（供 DSpark 草稿模型取用）。"""
        self.language_model.set_aux_hidden_state_layers(layers)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """加载 checkpoint 权重: 视觉塔直接装载，语言权重透传。

        策略: 用生成器函数 language_weights() 做流式分拣——遍历权重迭代器，
        属于视觉塔的当场加载并记录，其余的 yield 给语言主干的
        load_weights（惰性迭代，无需把整个 checkpoint 读入内存）。

        Args:
            weights: (名字, 张量) 可迭代对象，通常来自 safetensors 流。

        Returns:
            成功加载的参数名集合（视觉名 + "language_model." 前缀的语言名）。
        """
        # dict(self.named_parameters()): 参数名 -> Parameter 的映射，
        # 供视觉权重按名查找目标参数。
        params = dict(self.named_parameters())
        loaded_vision: set[str] = set()

        def language_weights() -> Iterator[tuple[str, torch.Tensor]]:
            """生成器: 逐条分拣，视觉权重就地加载，语言权重向下透传。"""
            for name, loaded_weight in weights:
                vision_name = _vision_parameter_name(name)
                if vision_name is None:
                    # 非视觉权重: yield 给语言主干（生成器语法点——
                    # 每次迭代在此暂停并产出一个值）。
                    yield name, loaded_weight
                    continue
                if vision_name not in params:
                    raise KeyError(f"Vision weight {name!r} has no parameter {vision_name!r}.")
                # getattr(param, "weight_loader", default_weight_loader):
                # 参数自带的加载器（量化参数有专用 loader），无则用默认。
                param = params[vision_name]
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, loaded_weight)
                loaded_vision.add(vision_name)

        # 语言主干消费生成器分拣出的语言权重。
        loaded_language = self.language_model.load_weights(language_weights())
        # 集合推导式 + | 并集: 语言主干返回的名字补上
        # "language_model." 前缀后与视觉名合并。
        return loaded_vision | {f"language_model.{name}" for name in loaded_language}

    def process_weights_after_loading(self) -> None:
        """权重加载完成后的后处理钩子（委托给语言主干的同名钩子，
        如权重转置/量化参数 finalize 等）。"""
        # getattr(obj, name, None): 探测可选方法，存在才调用——
        # 语言主干不总是实现该钩子。
        hook = getattr(
            self.language_model,
            "process_weights_after_loading",
            None,
        )
        if hook is not None:
            hook()
