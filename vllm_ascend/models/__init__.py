"""vllm-ascend 插件的模型注册中心（模型入口模块）。

架构位置与职责：
- vLLM 上游通过 ModelRegistry 维护"架构名(architecture) -> 模型实现类"的全局
  映射，推理时依据 checkpoint 配置里的 architectures 字段查找并实例化模型。
- vllm-ascend 作为硬件插件，不修改上游模型文件，而是在本目录提供针对昇腾 NPU
  适配/增强的模型实现，并通过本模块把自定义架构名注册进 ModelRegistry，
  使 vLLM 在昇腾硬件上加载这些模型时自动路由到 Ascend 版本实现。

注册机制原理：
- ModelRegistry.register_model(architecture, "模块路径:类名") 的第二个参数是
  字符串而非类对象，注册表据此做延迟导入（lazy import），只有真正实例化某个
  模型时才 import 对应模块，可显著缩短插件加载时间。
- 条目分三类：① 上游同名架构的 NPU 覆盖（如 Gemma4/Kimi/Eagle3）；② 兼容旧
  checkpoint 命名的别名（如 KimiK3ForCausalLM）；③ 投机解码草稿模型。
- 草稿模型（Draft Model）与目标模型配合实现投机采样（Speculative Decoding）：
  MTP（Multi-Token Prediction，多 token 预测头，一次前向预测多个未来 token）、
  DSpark（基于马尔可夫转移矩阵的轻量草稿网络）、EAGLE3（在特征层自回归的
  高接受率草稿）、DFlash2（分组卷积 + 候选选择器草稿）等；草稿廉价地"猜"出
  多个 token，目标模型一次前向并行验证，从而提升解码吞吐。
"""


from vllm import ModelRegistry


def register_model():
    """向 vLLM 的 ModelRegistry 批量注册 vllm-ascend 的全部模型实现。

    原理: vLLM 启动并加载硬件插件时会调用本函数，把下方每个
    "架构名 -> 模块:类名" 映射写入全局注册表；ModelRegistry 之后按
    checkpoint 的 architectures 字段动态导入并实例化对应类。
    注册字符串格式 "包.模块:类名" 实现延迟加载（lazy import），
    避免启动时导入全部模型模块。
    """
    # Gemma4 多模态（图/视频/音频）模型: NPU 适配 + ModelSlim
    # MXFP4/MXFP8 动态量化支持（详见 gemma4_mm.py）。
    ModelRegistry.register_model(
        "Gemma4ForConditionalGeneration",
        "vllm_ascend.models.gemma4_mm:AscendGemma4ForConditionalGeneration",
    )
    # Kimi K3（KimiLinear 架构）纯文本模型: MLA 注意力 + MoE + KDA
    # 注意力残差 + 可选序列并行的 NPU 适配（详见 kimi_k3.py）。
    ModelRegistry.register_model(
        "KimiLinearForCausalLM",
        "vllm_ascend.models.kimi_k3:AscendKimiLinearForCausalLM",
    )
    # Keep the release-branch text architecture as a compatibility alias for
    # checkpoints whose config predates vLLM's KimiLinear rename.
    # （保留旧架构名 KimiK3ForCausalLM 作为兼容别名: 部分早期发布的
    # checkpoint 配置仍使用该名字，路由到与 KimiLinear 相同的实现。）
    ModelRegistry.register_model(
        "KimiK3ForCausalLM",
        "vllm_ascend.models.kimi_k3:AscendKimiLinearForCausalLM",
    )
    # Kimi K3 多模态（图文）版本: 视觉塔 MoonViT3d + 多模态投影器 + 文本主干。
    ModelRegistry.register_model(
        "KimiK3ForConditionalGeneration",
        "vllm_ascend.models.kimi_k3:AscendKimiK3ForConditionalGeneration",
    )
    # Kimi K3 MTP 草稿模型: 投机解码用的多 token 预测头（见 kimi_k3_mtp.py）。
    ModelRegistry.register_model(
        "KimiK3MTPModel",
        "vllm_ascend.models.kimi_k3_mtp:AscendKimiK3MTP",
    )
    # Kimi K3 DSpark 草稿模型: 马尔可夫式轻量草稿网络（见 kimi_k3_dspark.py）。
    ModelRegistry.register_model(
        "K3DSparkModel",
        "vllm_ascend.models.kimi_k3_dspark:AscendK3DSparkForCausalLM",
    )
    # DeepSeek V4 系列目标模型（文本 + 多模态 VL）。
    ModelRegistry.register_model(
        "DeepseekV4ForCausalLM", "vllm_ascend.models.deepseek_v4.model:AscendDeepseekV4ForCausalLM"
    )
    ModelRegistry.register_model(
        "DeepseekV4ForConditionalGeneration",
        "vllm_ascend.models.deepseek_v4.vl_model:AscendDeepseekV4ForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "DeepseekV41ForCausalLM",
        "vllm_ascend.models.deepseek_v41.vl_model:AscendDeepseekV41ForCausalLM",
    )
    # MiniMax M3 稀疏注意力模型（文本 + 多模态）。
    ModelRegistry.register_model(
        "MiniMaxM3SparseForCausalLM",
        "vllm_ascend.models.minimax_m3:MiniMaxM3SparseForCausalLM",
    )
    ModelRegistry.register_model(
        "MiniMaxM3SparseForConditionalGeneration",
        "vllm_ascend.models.minimax_m3:MiniMaxM3SparseForConditionalGeneration",
    )
    # DeepSeek V4 的 MTP 投机解码草稿模型。
    ModelRegistry.register_model("DeepSeekV4MTPModel", "vllm_ascend.models.deepseek_v4.mtp:DeepSeekV4MTP")
    # DSpark 草稿: DeepSeek V4/V4.1 目标模型的马尔可夫草稿网络。
    ModelRegistry.register_model(
        "DSparkDraftModel",
        "vllm_ascend.models.deepseek_v4.dspark:DSparkDeepseekV4ForCausalLM",
    )
    ModelRegistry.register_model(
        "DeepseekV41DSparkModel",
        "vllm_ascend.models.deepseek_v41.dspark:DSparkDeepseekV41ForCausalLM",
    )
    # EAGLE3 VWN 草稿: 变宽网络（Variable-Width Network）版 EAGLE3（见
    # llama_eagle3_vwn.py）。
    ModelRegistry.register_model(
        "LlamaForCausalLMVwnEagle3", "vllm_ascend.models.llama_eagle3_vwn:Eagle3VwnLlamaForCausalLM"
    )
    # Qwen3 DSpark 草稿模型（与 Qwen3-Omni DSpark 共用同一实现）。
    ModelRegistry.register_model("Qwen3DSparkModel", "vllm_ascend.models.qwen3_dspark:AscendQwen3DSparkForCausalLM")
    ModelRegistry.register_model(
        "Qwen3OmniDSparkModel",
        "vllm_ascend.models.qwen3_dspark:AscendQwen3DSparkForCausalLM",
    )
    # DFlash2 草稿: 分组卷积 + 候选选择器式草稿模型（见 qwen3_dflash2.py）。
    ModelRegistry.register_model(
        "DFlash2DraftModel",
        "vllm_ascend.models.qwen3_dflash2:DFlash2Qwen3ForCausalLM",
    )
    # DeepSeek 系 MTP 草稿: DeepSeekMTPModel 与 DeepseekV32MTPModel 共用实现。
    ModelRegistry.register_model("DeepSeekMTPModel", "vllm_ascend.models.deepseek_mtp:AscendDeepSeekMTP")
    ModelRegistry.register_model("DeepseekV32MTPModel", "vllm_ascend.models.deepseek_mtp:AscendDeepSeekMTP")
    # GLM MoE（DSA 注意力）目标模型（见 deepseek_mtp.py）。
    ModelRegistry.register_model("GlmMoeDsaForCausalLM", "vllm_ascend.models.deepseek_mtp:AscendGlmMoeDsaForCausalLM")
    # EAGLE3 草稿: Llama 目标模型的 EAGLE3 特征级草稿（见 llama_eagle3.py）。
    ModelRegistry.register_model(
        "Eagle3LlamaForCausalLM", "vllm_ascend.models.llama_eagle3:AscendEagle3LlamaForCausalLM"
    )
    # GLM-5-Next 系列目标模型及其 MTP 草稿。
    ModelRegistry.register_model(
        "Glm5NextForCausalLM",
        "vllm_ascend.models.glm5next.model:Glm5NextForCausalLM",
    )
    ModelRegistry.register_model(
        "Glm5NextForConditionalGeneration",
        "vllm_ascend.models.glm5next.model:Glm5NextForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "Glm5NextMTPModel",
        "vllm_ascend.models.glm5next.mtp:Glm5NextMTP",
    )
    # LlamaForCausalLMEagle3: EAGLE3LlamaForCausalLM 的旧注册名，同样路由到
    # 昇腾实现（与上方 Eagle3LlamaForCausalLM 条目一致）。
    ModelRegistry.register_model(
        "LlamaForCausalLMEagle3", "vllm_ascend.models.llama_eagle3:AscendEagle3LlamaForCausalLM"
    )
