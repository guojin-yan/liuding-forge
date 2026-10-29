# 本地推理与版本切换

`jyppx::liuding::serving` 是默认关闭的 Preview 组件，通过 `<jyppx/liuding/serving/serving.hpp>` 使用。它提供经过验证的 native safetensors 模型加载、显式可信的 TorchScript 兼容路径、单进程预测、有界动态批处理和传输无关的服务适配器。当前 Serving 后端未接入 ONNX Runtime；HTTP/gRPC/TLS、流式服务、实例池和部署控制面由宿主或后续版本负责。

## 加载与请求

`NativeStateBackend::load` 校验 bundle 文件清单、大小、摘要、可选签名策略、精确 backend/version、环境锁、依赖版本、设备、dtype、动态形状和平台声明，再快照文件并严格导入 safetensors。目录在整个加载期间必须由可信文件系统所有者保持不可变；重新打开路径不等同于抵抗恶意并发修改的文件系统快照。`serving_platform_id()` 返回当前构建的 OS/架构标识。环境锁由调用方认证，本库不运行包管理器重建环境。

模型工厂必须返回全新的 C++ `ModelAdapter` 和 `Pipeline`，自行解析并绑定已验证的 input/output/pre/post 描述符，持有返回后仍需使用的数据。bundle 不选择类、插件或可执行代码。描述符摘要用于确认身份，具体预处理语义仍由可信工厂实现。native 模型使用 eval/inference 模式；拒绝所选设备的环境 autocast。

至少一个调用方提供的独立已知向量必须通过完整 pre/invoke/post、输出 schema、资源和数值检查，加载才返回 ready predictor。`load_torchscript_predictor` 实际加载和调用显式可信的 JIT 归档，独立固定归档摘要与环境锁，继承兼容运行时的设备和失败语义；它不把可执行归档混入数据型 bundle，也不支持本组件的批处理或 OOM 拆批重试。

`PredictRequest` 复用 `nn::Invocation`，支持 CPU/CUDA 稠密 Tensor、精确标量、None、tuple、同质 list 和字符串键 dict。输入与输出分别检查逻辑字节、Tensor 数、rank、维度、形状乘积、递归节点/深度、字符串、有限数及可选数值区间；schema/dtype/device 必须精确匹配，不自动转换或回退。直接 `Predictor` 调用需要在 `context.model_version` 填写精确版本；版本化 owner 和 lease 允许空值并绑定当前或已持有版本。

`Predictor` 串行执行；`max_active_requests` 包括等待串行执行的准入请求。快照期间输入必须保持静止。输入队列与结果使用独立连续 Tensor 快照，不保留值间 storage alias；结果可在实例关闭后使用。CUDA 成功返回前会同步所选设备，因此也可能等待该设备上的其他任务。

上述限制覆盖声明的边界、状态和快照 payload，不是进程 RSS/显存硬限制，不能约束后端 workspace、可信回调分配或 kernel 时长。deadline/cancel 在阶段边界协作检查，不能抢占同步工厂、codec 或 kernel；需要硬隔离时由外部进程监督器提供。

## 同 runtime 的受控切换

`ReloadablePredictor` 在旁路调用现有 `RuntimeBackend::load`，完成 bundle 校验、导入和已知向量 warmup，再比较完整输入输出 schema，原子发布新版本。backend ID/version、格式、环境锁、依赖、精确设备、dtype 和执行能力必须与初始实例一致。仅支持 native bundle 的同 runtime/设备切换；不支持跨后端迁移、跨设备 placement 或版本化 `DynamicBatcher`。

```cpp
using namespace jyppx::liuding::serving;
LoadOptions load;
load.max_state_bytes = 64ULL << 20U;
load.max_snapshot_bytes = 128ULL << 20U;
const auto reservation = 2 * load.max_state_bytes + load.max_snapshot_bytes;
ReloadablePredictor current(std::move(initial),
    {.initial_reservation_bytes = reservation,
     .max_resident_bytes = 2 * reservation});
auto old_request = current.acquire();
const auto change = current.reload(backend, new_bundle, verify, known_vectors,
                                  load, {.drain_timeout = std::chrono::milliseconds(0)});
// old_request.predict(...) 始终使用原版本；current.predict(...) 绑定新版本。
// 释放旧 lease 后调用 drain_previous(deadline)，完成原实例回收。
```

`PredictionLease` 为 move-only，准入时固定版本，覆盖 decode、predict、encode 的整个请求生命周期。新请求只绑定当前版本；显式指定旧版本的新请求被拒绝。单个 lease 的调用需要外部串行化。`max_active_leases` 同时限制当前及退役版本的请求持有者，避免无限堆积尚未进入模型的请求。

最多同时保留两槽：一个当前版本和一个候选或退役版本。加载前预留 `2 * max_state_bytes + max_snapshot_bytes`，覆盖声明的候选状态、导入 staging 与快照 payload；初始预留由拥有者明确声明。两槽总预留超过 `max_resident_bytes`，或已有候选/退役版本时，拒绝下一次加载。`statistics()` 的 resident/peak/reserved 反映槽和声明预算，不等同于实测 RSS/显存，工厂的实际峰值仍须独立测量。

候选的验证、warmup、schema 或协作 deadline 失败不会替换旧版本。发布后的 drain 超时返回 `previous_drained=false`，新版本仍已发布，旧实例及预留继续保留；释放旧 lease 后重试 `drain_previous`。候选超时必须为正，默认 30 秒；旧版本 drain 默认 5 秒，可设 0 进行即时检查，两者上限 24 小时。超时不能抢占可信同步代码，失败清理同样可能超过预算。

`close(deadline, stop)` 停止新准入，等待候选及所有 lease，再释放两槽；返回 false 时仍保留正在使用的实例。析构仅停止准入，已有 lease 延长相关实例寿命，最后一个持有者释放时关闭模型，避免同线程持有 lease 再析构 owner 的死锁。应用必须主动释放 lease 才能确定性回收资源。不要通过保留的原 predictor 别名继续调用或关闭模型，也不要让 owner 析构与其成员调用并发。

## 批处理与关闭

`DynamicBatcher` 为单 worker、有界队列，支持一个 leading dimension 为 1 的位置 Tensor 输入和一个 batch 维一致的 Tensor 输出。schema 必须声明固定 rank/尾部形状与动态首维，后端必须明确声明逐行独立且 pre/post 保持 batch。其他结构化输入输出使用直接 predictor 调用。

队列按模型/pipeline/schema、dtype/精确设备/形状、优先级/deadline class、project/principal/deployment 分组，只合并兼容前缀。容量在快照前预留，字节预算覆盖排队和活动请求；`max_batch_size` 与 `max_wait` 有界，但等待上限不能抢占前一个 compute。过期/取消成员在新 batch 前移除，每个结果交付时再检查；一个成员取消不终止其他成员。未提供结果缓存。

只有显式 retry-safe native forward 中确认的分配失败产生 `RetryableOutOfMemory`，按整个原始 batch 的有限预算迭代拆分，最多 8 次，不重试 singleton。普通资源/schema/pre/post/输出错误不触发重试；JIT 不允许此策略。

predictor 关闭先拒绝新请求，再等待活动 lease；超时返回 false，保持 draining，不卸载运行中的 session。batcher 先停止准入、取消队列并等待 compute/快照，再关闭 predictor；不 detach worker。进入 forward 后的失败会保守地将实例降级，显式 retry-safe OOM 除外；forward 前的 schema/资源/cancel/deadline 拒绝不损害模型。readiness 在 draining/degraded/closed 时为 false。

## 服务适配与观测

`InferenceServiceAdapter` 使用可信、有版本的 decoder/encoder/authorizer，宿主负责 TLS、身份认证及请求上下文。未认证或缺少 project/principal 时拒绝，授权先于解码。请求字节在 decode 前限制，响应在 encode 后限制；codec 本身必须控制展开分配。绑定 `ReloadablePredictor` 的构造函数在授权后、decode 前获取 lease，并持有至 encode 完成；此路径直接预测，不共享固定版本 batcher。并发服务使用的协议回调必须线程安全。

公共响应仅返回固定状态：`invalid_argument`、`unauthenticated`、`permission_denied`、`resource_exhausted`、`cancelled`、`deadline_exceeded`、`unavailable`、`internal`。私有 backend 异常、输入值或 token 不进入公共回复。直接 C++ 异常可保留嵌套原因，供受权本地诊断。

`ObservationHook` 接收已删减的 foundation log/correlation/attributes，由应用接入遥测。回调须有界、非阻塞且不重入；异常只增加失败计数。请求/trace 等高基数身份放 logs/traces，避免直接作为 metric label。响应包含各阶段实际计时，调度统计提供请求数、活动深度、驻留逻辑字节和 OOM 重试数。

参见[本地 JSONL 与切换示例](../../examples/serving/README.md)。示例不启动网络监听器，不下载预训练权重。单元与安装消费者覆盖真实 bundle 导入、数值预测、版本 lease、并发切换和关闭；具体平台支持、延迟、吞吐、RSS/显存峰值与长时负载资格以实际验收结果为准。
