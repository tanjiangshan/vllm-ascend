# vllm-ascend 源码学习路线图

> 行号基准：vllm `main @ 75dc588269`、vllm-ascend `main @ 10b4fb2a9`（2026-09-17）。
> 代码更新后行号会漂移，以"类名/函数名 + 文件"为锚点搜索为准。

## 〇、先建立的认知：vllm 与 vllm-ascend 的关系

vllm-ascend **不是独立引擎**，是 vllm 的硬件平台插件（OOT plugin）：

- 引擎主循环、调度器、KV cache 管理、OpenAI 服务 —— 都在 **vllm** 仓库
- NPU 专属实现（worker、model runner、attention backend、算子、量化）—— 在 **vllm-ascend** 仓库
- vllm 通过 `Platform` 接口把"设备相关决策"外包给插件；**vllm 主干在哪调用 platform 钩子，vllm-ascend 就在哪接管**

```
┌───────────────────────── vllm（主干） ─────────────────────────┐
│ api_server → AsyncLLM → EngineCore → Scheduler → Executor    │
│                     │                      │                  │
│              Platform 接口 ◄──────────────┘ 调用钩子做决策     │
└─────────────────────────┬──────────────────────────────────────┘
                          │ 插件接管
┌───────────────────── vllm-ascend ─▼───────────────────────────┐
│ NPUPlatform → NPUWorker → NPUModelRunner                      │
│   → AscendAttentionBackend → NPU 算子(ops) / 量化 / 通信      │
└───────────────────────────────────────────────────────────────┘
```

---

## 阶段 0：插件挂载机制（1~2 天）

目标：回答"vllm 怎么知道跑在 NPU 上、attention backend 是谁选的、补丁在哪打的"。

| 顺序 | 文件:行号 | 看什么 |
|---|---|---|
| 1 | `setup.py:519` | entry point：`vllm.platform_plugins → ascend = vllm_ascend:register`，另有 `vllm.general_plugins`（kv connector、model loader） |
| 2 | `vllm_ascend/__init__.py:68` | `register()` 返回 platform 类路径；`register_connector` / `register_model_loader` 等其它插件入口 |
| 3 | `vllm_ascend/platform.py:80` | `NPUPlatform(Platform)`：设备名 `npu`、支持的量化列表、`simple_compile_backend` 等身份声明 |
| 4 | `platform.py:226` | `get_attn_backend_cls()`：**attention backend 的分发中枢**。FA3 优先（`:234`），然后按 `(use_mla, use_sparse, use_compress)` 查表分发到 MLA/SFA/DSA/通用 backend；310P 走兼容表（`:242`） |
| 5 | `platform.py:443` | `check_and_update_config()`：启动时改写 vllm 配置（block size、图捕获尺寸等），体会"插件修正主干配置"的模式 |
| 6 | `platform.py:146,162` | `get_device_capability` / `register_custom_kv_cache_specs`：KV cache 规格（KVPP）如何上报给主干 |
| 7 | `vllm_ascend/utils.py:583` | `adapt_patch()`：补丁机制总入口，全局补丁与 worker 级补丁的加载方式 |
| 8 | `vllm_ascend/ascend_config.py` | NPU 专属配置体系（`--ascend-scheduler-config` 等）；`device/hardware*.py` 硬件能力画像 |

对照（vllm 侧）：`vllm/platforms/__init__.py` 中搜索 `platform_plugins`，看插件发现与 `current_platform` 的确定过程。启动日志里的 `Available plugins for group vllm.platform_plugins` 就是这条链路。

**检验**：能画出"从 `vllm serve` 命令到确定使用 NPUPlatform"的调用链。

---

## 阶段 1：一条请求走通（只看骨架，别陷细节）

以一次 chat completion 请求 + 一次 decode step 为主线。

### 1.1 服务端与异步引擎入口（vllm 侧）

| 文件:行号 | 看什么 |
|---|---|
| `vllm/entrypoints/launchers/api_server/entry.py:36` | `build_async_engine_client()`：服务启动、创建 AsyncLLM |
| `vllm/entrypoints/openai/chat_completion/api_router.py:29` | `/v1/chat/completions` 路由 |
| `vllm/entrypoints/openai/chat_completion/serving.py:118` | `OpenAIServingChat`：OpenAI 协议 → 引擎请求的转换 |
| `vllm/v1/engine/async_llm.py:77` | `AsyncLLM`：API 进程持有的引擎客户端 |
| `async_llm.py:346` | `add_request()`：请求进入引擎 |
| `async_llm.py:759` | `output_handler` 协程：输出异步回传、流式吐 token 的地方 |

### 1.2 引擎核心进程（vllm 侧）

| 文件:行号 | 看什么 |
|---|---|
| `vllm/v1/engine/core.py:105` | `EngineCore`：独立进程，引擎心脏 |
| `core.py:463` | `add_request()`：请求入队 |
| `core.py:1479` | `_process_engine_step()`：**主循环每一步**——调度 → 执行 → 产出 |
| `core.py:620` | `model_executor.execute_model(scheduler_output)`：交给执行器 |
| `vllm/v1/engine/input_processor.py` / `output_processor.py` / `detokenizer.py` | 输入预处理、增量 detokenize（输出侧三件套） |

### 1.3 调度器与 KV cache（vllm 侧）

| 文件:行号 | 看什么 |
|---|---|
| `vllm/v1/core/sched/scheduler.py:74` | `Scheduler` |
| `scheduler.py:509` | `schedule()`：**核心中的核心**——prefill/decode 组 batch、抢占、预算控制 |
| `scheduler.py:2398` | `add_request()` |
| `vllm/v1/core/sched/request_queue.py` | 请求排队策略（waiting/running 队列） |
| `vllm/v1/core/kv_cache_manager.py` + `block_pool.py` + `kv_cache_coordinator.py` | KV block 分配/释放/迁移（prefix caching 也在这） |

### 1.4 跨进程到 worker（vllm 侧）

| 文件:行号 | 看什么 |
|---|---|
| `vllm/v1/executor/uniproc_executor.py:51` | `UniProcExecutor`：单进程执行器（单卡走这）；多卡看 `multiproc_executor.py` / `ray_executor.py` |
| `vllm/v1/worker/gpu_worker.py:177` | `Worker`：GPU 参照物，后面 NPUWorker 与它对齐 |
| `vllm/v1/worker/gpu_model_runner.py:503` | `GPUModelRunner`：**NPUModelRunner 的父类**，先通读它的 execute_model 流程 |

### 1.5 NPU 侧 worker（vllm-ascend）

| 文件:行号 | 看什么 |
|---|---|
| `vllm_ascend/worker/worker.py:126` | `NPUWorker.__init__`：adapt_patch → 注册自定义算子 → `init_ascend_config`，NPU 侧接管的第一现场 |
| `worker.py:496` | `init_device()`：NPU 设备初始化 |
| `worker.py:842` | `load_model()`：触发权重加载 |
| `worker.py:764` | `execute_model()`：每步执行的 worker 入口 |
| `vllm_ascend/worker/model_runner_v1.py:342` | `NPUModelRunner(GPUModelRunner)`（5000+ 行，本仓库最重要文件） |
| `model_runner_v1.py:4065` | `load_model()`：权重加载与层替换 |
| `model_runner_v1.py:3665` | `_dummy_run()`：预热/profiling/图捕获前的空跑 |
| `model_runner_v1.py:2145` | `execute_model()`：**每步核心**——准备输入 → 模型 forward → 采样 |
| `model_runner_v1.py:5893` | `capture_model()`：aclgraph 捕获（对应 CUDA Graph） |
| `worker/npu_input_batch.py` / `block_table.py` / `device_metadata.py` | NPU 侧输入 batch、block table、设备元信息 |

读法：先通读父类 `GPUModelRunner` 的 execute_model，再 diff 式看 NPU 子类改了哪些——改的就是 NPU 差异点。

### 1.6 模型 forward 与 attention（两边交界）

| 文件:行号 | 看什么 |
|---|---|
| `vllm/model_executor/models/llama.py` | 拿一个最简单的模型通读 forward；模型注册表在 `models/registry.py` |
| `vllm/model_executor/layers/attention/attention.py:218` | `Attention` 层：模型里的 attention 调用入口，内部按 selector 选 backend |
| `vllm/v1/attention/selector.py` | backend 选择逻辑（最终回到 `NPUPlatform.get_attn_backend_cls`） |
| `vllm_ascend/attention/attention_v1.py:75` | `AscendAttentionBackend`：通用 MHA backend（总入口先读这个） |
| `attention_v1.py:506` | `AscendAttentionBackendImpl`：实际实现 |
| `vllm_ascend/ascend_forward_context.py` | NPU 前向上下文：MoE 通信类型、KV cache 绑定等执行期状态 |

### 1.7 采样与输出回流

| 文件:行号 | 看什么 |
|---|---|
| `vllm_ascend/sample/sampler.py:48` | `AscendSampler(Sampler)`：NPU 采样实现 |
| `sampler.py:111` | `AscendTopKTopPSampler`：top-k/top-p 在 NPU 上的实现 |
| `vllm/v1/sample/sampler.py` | 父类对照 |
| 输出回流 | `EngineCoreOutputs` → core_client → `AsyncLLM.output_handler` → `detokenizer` → OpenAI SSE 响应 |

**检验**：画一张完整时序图（api → engine core → scheduler → executor → NPU worker → forward → sample → 输出），标出跨进程边界。建议同时在 NPU 环境跑 `examples/offline_inference/` 里的最小示例 + `py-spy dump` 印证调用栈。

---

## 阶段 2：模块精读（按依赖顺序）

每个模块给"入口 → 重点 → 深入"三步。阶段 1 已覆盖的部分不再重复。

### 2.1 platform / config / device（入门即阶段 0，此步补全）
- `device/hardware.py`、`device/hardware_profile.py`：`AttentionBackendFamily` / `QuantizationBackendFamily` 硬件画像，解释"为什么 A5 和 910B 走不同代码路径"
- `envs.py`：所有 `VLLM_ASCEND_*` 环境变量，相当于功能开关清单

### 2.2 worker/（继续深挖）
- `model_runner_v1.py` 中 logits 处理、aclgraph 相关分支（`patch/worker/patch_cudagraph.py` 配合读）
- `worker/v2/model_runner.py:83`：v2 引擎版本，v1 读完后对比差异
- `worker/kvpp_cache.py`、`encoder_acl_graph.py`、`dcp_utils.py`

### 2.3 attention/（NPU 最核心差异点）
| 文件 | 内容 |
|---|---|
| `attention_v1.py` | 通用 MHA（先读，建立 Metadata/MetadataBuilder 模式） |
| `fa3_v1.py:11` | `AscendFABackend`：FlashAttention 路径 |
| `mla_v1.py:82` | `AscendMLABackend`；`:785 AscendMLAImpl`（DeepSeek/GLM 系 MLA，吸收式 KV） |
| `sfa_v1.py:268` | `AscendSFABackend`：稀疏注意力 |
| `dsa_v1.py:214` | `AscendDSABackend` + `indexer.py` / `indexer_kpool.py`：DSA（lightning indexer） |
| `attention_mask.py` | mask 构造 |
| `context_parallel/mla_cp.py` | 上下文并行（CP）的 MLA |
| `sparse_flash_mla.py`、`dsa_attn_kv_plan.py` | 稀疏 MLA 规划 |

结合 `platform.py:226` 的分发逻辑串起来：什么模型/什么配置走哪条路径。

### 2.4 ops/（算子层，attention 的地基）
- 直连文件：`layernorm.py`、`activation.py`、`mla.py`、`rotary_embedding.py`、`linear.py`、`vocab_parallel_embedding.py`
- `fused_moe/`：MoE 算子（配 `patch/platform/patch_fused_moe.py`）
- `triton/`：NPU 上的 Triton kernel（`v2/` 是新版），看 `docs/` 子目录有说明
- `register_custom_ops.py`：算子注册机制
- C++ 侧对应 `csrc/`（Ascend C kernel），可选深入

### 2.5 distributed/
- `parallel_state.py`：NPU 的 TP/PP/EP 世界状态（对照 vllm 的 `vllm/distributed/parallel_state.py`）
- `device_communicators/`：HCCL 通信封装
- `eplb/`：专家负载均衡
- `kv_transfer/`：**PD 分离全家桶**——`kv_p2p/`（mooncake、sfa_pd_rd2h）、`kv_pool/`（ascend_store、kv_offload、ucm_connector）。入口 `distributed/kv_transfer/__init__.py:21 register_connector()`

### 2.6 quantization/
- `methods/w8a8`、`w4a4`、`wna16`：NPU 量化方案
- `methods/kv_cache`：KV cache 量化
- `methods/registry.py:24`：量化方案注册机制

### 2.7 model_loader/（权重加载）
- 默认 loader 由 platform 指定（回看 `platform.py`）
- `netloader/`：网络加载（`executor/`、`interaction/elastic.py:213 register`）
- `rfork/`：rfork 加载（`transfer_backend.py:387`）

### 2.8 models/ + patch/（模型定制与对主干的适配）
- `models/glm5next/`（衔接你的 GLM 学习分支）、`models/deepseek_v4/`、`models/minimax_m3/`、`models/layer/`
- `patch/platform/`：对主干调度/分布式/KV cache 的补丁（`patch_kv_cache_utils.py`、`patch_balance_schedule.py`、`patch_fused_moe.py`…）
- `patch/worker/`：模型行为补丁（`patch_deepseek_v2.py` 是典型样例：直接替换 upstream 类方法）
- **patch/ 是"为什么需要适配 vllm"答案最密集的地方**，每读一个补丁问一句"这为什么不能进 upstream"

### 2.9 spec_decode/ + worker/v2/spec_decode/（投机解码）
- `spec_decode/` 静态部分 + `worker/v2/spec_decode/`（eagle、mtp、dspark、dflash、autoregressive）
- 配合 `sample/rejection_sampler.py:37 AscendRejectionSampler`

### 2.10 进阶/按需
- `compilation/`：torch.compile 集成（`updatable_graph.py`、`passes/` 图优化模式）
- `lora/`、`profiler/`、`observability/`、`eplb/`、`weight_switch/`（在线换权重）、`batch_invariant.py`
- `core/`：NPU 侧调度增强（`batch_job_aware_scheduler.py`、`dyntra_lb_scheduler.py`、`recompute_scheduler.py` 等）
- `_310p/`：旧硬件兼容层，最后看或跳过

---

## 学习方法

1. **跑起来再读**：NPU 环境用 `examples/offline_inference/` 最小示例跑通，比纯读码效率高一个量级
2. **对照读**：vllm-ascend 几乎每个类都继承/补丁 vllm 的 GPU 对应物，`git diff` 思维读"NPU 改了什么、为什么"
3. **动态印证**：`VLLM_LOGGING_LEVEL=DEBUG`、IDE 断点（多进程注意 attach 到 EngineCore 进程）、`py-spy dump --pid`
4. **边读边注释**：就在当前分支 `huamus-0917-learn-annotations` 上做，一个模块一个 commit
5. **检验驱动**：每阶段末尾的"检验"能独立讲出来再进入下一阶段
