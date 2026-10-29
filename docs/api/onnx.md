# ONNX 导出与推理

部署功能集中在 `deployment/`：`onnx_export` 将原生 C++ 模型导出为 ONNX，`onnx_runtime` 加载 ONNX 并执行推理。两个组件分别启用、默认关闭，安装后的导出与推理均无需 Python。

完整的原生检查点恢复、双输入双输出导出和动态 batch 数值核对，见 [C++ 部署示例](../../examples/deployment/README.md)。

| 组件 | 用途 | 公共头文件 | CMake target |
| --- | --- | --- | --- |
| ONNX 导出 | 执行样例 forward，捕获计算图，按算子转换并序列化 | `<jyppx/liuding/deployment/onnx_export/exporter.hpp>` | `jyppx::liuding::onnx_export` |
| ONNX Runtime | 校验模型字节与输入，使用显式 Execution Provider 推理 | `<jyppx/liuding/deployment/onnx_runtime/session.hpp>` | `jyppx::liuding::onnx_runtime` |

## 构建与接入

准备 LibTorch 2.14.1；需要推理时再准备官方 ONNX Runtime 1.23.2 SDK。在现有 CMake 配置中启用需要的组件，两者互不强制依赖，也不要求启用项目的 `models` 组件。

```sh
cmake -S . -B out/build/your-profile \
  -DJYPPX_LIUDING_BUILD_ONNX_EXPORT=ON \
  -DJYPPX_LIUDING_BUILD_ONNX_RUNTIME=ON \
  -DJYPPX_LIUDING_ONNX_RUNTIME_ROOT=/path/to/onnxruntime-sdk
```

安装后的消费者显式请求组件：

```cmake
find_package(jyppx CONFIG REQUIRED COMPONENTS onnx_export onnx_runtime)
target_link_libraries(my_app PRIVATE
    jyppx::liuding::onnx_export jyppx::liuding::onnx_runtime)
```

Windows 应将安装目录 `bin/onnxruntime*.dll` 放在应用程序目录；仅调整 `PATH` 可能仍先加载系统自带的旧版 ORT。Linux 应确保加载安装包对应的共享库。

## 从 C++ 模型导出并执行

```cpp
#include <jyppx/liuding/deployment/onnx_export/exporter.hpp>
#include <jyppx/liuding/deployment/onnx_runtime/session.hpp>
#include <torch/nn/modules/linear.h>

#include <array>

torch::nn::Linear model(3, 2);  // 可使用已训练、已加载权重的模型。
model->eval();
const auto example = at::zeros({1, 3}, at::TensorOptions().dtype(at::kFloat));
const auto artifact = jyppx::liuding::onnx_export::export_model(model, example);

jyppx::liuding::onnx_runtime::SessionOptions options;
options.provider.name = "CPUExecutionProvider";
options.max_batch_size = 1024;
jyppx::liuding::onnx_runtime::Session session(
    artifact.onnx, artifact.digest, options);
const std::array<std::int64_t, 2> shape{1, 3};
const std::array<float, 3> input{1.0F, 2.0F, 3.0F};
const auto output = session.run(input, shape);
```

单 Tensor 输入输出的模型使用 `export_model()`；多个输入输出或自定义调用方式使用 `export_module(module, forward, examples, options)`，其中 `forward` 接收 `std::span<const at::Tensor>` 并返回 `std::vector<at::Tensor>`。注册的参数与 buffer 随图保存；输入输出名称由 `input_names`、`output_names` 指定。`export_linear()` 和 `export_resnet18()` 是调用相同通用入口的便利函数，不维护另一套模型整图实现。

导出以算子覆盖为边界，不按模型名称设置白名单。已支持算子的组合可直接复用；缺失或自定义算子通过 `custom_lowerings` 注册原生 C++ 转换，未注册时报告具体算子与原因。导出器使用 LibTorch 官方 ONNX serializer，当前输出为 opset 17、自包含权重，不支持 external data 或直接转换任意 `.pt` 文件。

导出会真实执行 forward，并可能为动态 batch 校验再次执行。调用方负责设定 eval 模式、CPU 设备、数据类型和预处理，保证 forward 确定且不修改输入、模型状态或外部资源。trace 只捕获实际执行路径；动态 batch 会用另一 batch 检查图结构，遇到形状常量或路径变化明确拒绝，不能据此推导任意数据相关控制流均可导出。当前动态轴仅支持 batch 维；其他维度须固定。

`ExportOptions` 设置 batch 上下界、命名和字节上限；`dynamic_batch=false` 导出固定形状。符号 batch 不会在 ONNX 中编码业务上下界，运行端还应设置 `max_batch_size` 并检查自己的最小 batch。输出大小限制不等于进程内存或显存硬配额。启用 `quantization` 后，可导出 canonical Q0 Linear 的 U8U8/int32 图，见[量化](quantization.md)。

## Runtime 与后端选择

Runtime 接收一个或多个具名 FP32 稠密输入，返回独立持有数据的具名输出，秩为 1–8。`inputs()`、`outputs()` 返回图声明顺序的签名；多输入调用 `run(std::span<const TensorInput>)`，每个 `TensorInput` 提供 `name`、`values` 和 `shape`，输入顺序可变，输出按声明顺序返回。名称必须完整、唯一且与图一致。单输入输出模型仍可使用上面的 `run(values, shape)`；该简便入口遇到多输入或多输出时明确报错。

只有前导 batch 维可以动态变化。同名 batch 符号必须取相同大小，不同输入的独立符号可取不同大小；输出的动态符号必须由输入绑定，其他维度固定。I/O 位于主机内存，输入数据由调用方持有且在调用期间保持不变，adapter 不复制 FP32 输入数据；设备传输由所选 Provider 处理。

`available_providers()` 查询实际加载的 ORT 库，`provider_append_supported(name)` 查询 adapter 是否接入对应的追加 API。CPU、CUDA、TensorRT、OpenVINO 及通用 Provider 均显式指定名称和参数；实际可用性取决于 ORT 构建、驱动与 SDK。列出 Provider 或创建 session 成功，均不代表已验证模型实际使用该后端执行。

非 CPU 请求默认 `allow_cpu_fallback=false`，缺失后端或不支持的图明确失败；调用方显式允许后才接受 CPU 分区回退。`profile_prefix` 启用 ORT profiling，`end_profiling()` 返回可核对实际 Provider 与算子的文件。各后端的数值、kernel 与资源资格分别验证，CPU 结果不代表 GPU 结果。

Runtime 在解析前检查完整字节 SHA-256 和大小，执行前检查形状、batch、输入大小与有限数值，输出独立拥有；取消与 deadline 为协作式中断。`max_io_tensors` 限制输入与输出的总个数，`max_input_bytes`、`max_output_bytes` 分别限制一次调用的全部输入、全部输出字节数，所有输出大小在分配输出数据前统一检查。I/O 上限不覆盖元数据、图中间量与设备分配，哈希不代表来源可信；不可信模型须额外隔离。

## 导出报告与训练来源

通用导出的 `ExportResult::report` 保存模型 SHA-256、构建身份、实际输入输出、参数与 buffer 的名称/类型/形状/字节摘要、算子和已解析的动态 batch。报告不保存样本或参数原文；计算摘要会额外读取这些已有 Tensor，不复制完整参数。Q0 INT8 转换使用自己的源制品关联，不提供 trace 报告。

使用 `<jyppx/liuding/deployment/onnx_export/report.hpp>` 的 `serialize_export_report()` 保存报告，`deserialize_export_report()` 严格读取 schema v1，`verify_export_report_model()` 核对完整模型字节并重新调用 LibTorch 官方 checker。报告明确标记 Runtime 和数值验证为 `not_checked`，不能代替实际 ORT 测试或来源认证。

调用方可在保存前填写 `report->source`：`run_id` 是原训练 run，其他三项从其 lineage 原样复制。源码身份可为 Git commit，配置与数据身份使用完整 SHA-256。将 ONNX 和报告保存为两个文件，并记录报告文件的 SHA-256 后，可以登记到已有 run store：

```sh
liuding-forge model onnx register \
  --path model.onnx --report export-report.json --report-sha256 REPORT_SHA256 \
  --root ./runs --id export-001 --reference deployment
```

`register` 要求启用 `onnx_export`，核对报告、模型与原 run lineage 后创建独立的导出 run，其 `parent_run_id` 指向原训练。模型和报告通过同一事务登记到 CAS，成功后导出 run 进入 completed；同一参数可安全重试，失败不留下半对模型/报告引用。原训练 run 可以已经完成。导出 run ID 应独立保留给本次交付，报告 SHA-256 只确认字节身份。`register` 只登记原生导出结果；它不从权重推断用户 C++ forward，也不执行 ORT。实际执行使用下面的 `verify`；自动组包使用本页的 `model onnx pack`，已有完整清单则可用通用 [`model bundle pack`](model_bundle.md#发布)。

## 从训练检查点导出并组包

`model onnx export` 从原训练 run 的 CAS 引用读取模型定义与 `.jyck` 检查点，核对完整字节摘要和引用类型，按定义构造 CPU 模型并严格恢复指定 participant，然后调用同一个通用 `export_module()`。命令由恢复后的原生模型计算全部参考输出，执行 checker、ORT 数值比较和 Provider trace，通过后发布可独立 `smoke` 的 bundle；失败不发布，已有目录拒绝覆盖。

```sh
liuding-forge model onnx export \
  --root ./runs --source-run training-001 \
  --checkpoint-reference final-checkpoint --definition-reference model-definition \
  --inputs inputs.json --inputs-sha256 INPUTS_SHA256 \
  --tensors inputs.safetensors --tensors-sha256 TENSORS_SHA256 \
  --metadata delivery.json --metadata-sha256 DELIVERY_SHA256 \
  --provider CPUExecutionProvider --output ./deployment-bundle --format json
```

训练生产者通过 `SQLiteRunStore` 保存类型为 `checkpoint` 的检查点引用，以及 role 为 `model.definition` 的普通制品引用。模型定义使用 `jyppx.liuding.onnx_model_definition` v1，必须包含 `participant_id`，例如：

```json
{"format":"jyppx.liuding.onnx_model_definition","schema_version":1,"participant_id":"model.main","factory":{"id":"torch.nn.Linear","in_features":3,"out_features":2,"bias":true}}
```

新建的 ResNet18 ImageFolder 训练配方自动保存 `model-definition` 和 `final-checkpoint` 引用；迁移学习配方和历史已完成 run 不自动补写定义。导出不创建新训练 run，不更改原 run 状态或制品引用，也不隐式迁移旧版数据库；打开现有 SQLite/CAS 仍需可写存储，不支持只读介质。

启用 `models` 后也可使用 `{"id":"jyppx.liuding.models","name":"resnet18","num_classes":2,"input_channels":3}`；名称由现有模型注册表解析，其余构造选项采用该模型的原生默认值。这个声明必须符合训练模型的实际构造；命令不猜测任意权重的 forward，也不从同名网络推导算子支持。自定义模型、不同构造选项或多输出调用继续使用通用 C++ API，并由调用方恢复自身状态。CLI 对注册表的大模型有构造前保守检查，并在恢复时再次限制状态字节；超限明确拒绝。

`inputs.json` 使用 `jyppx.liuding.onnx_export_inputs` v1，声明全部名称、batch 范围、是否动态、数值容差和 1–64 个样例。每个输入按名称引用 `inputs.safetensors` 中的 FP32 Tensor；文件不得含未引用的 Tensor。参考输出由 C++ 生成，无需调用方填写，例如：

```json
{"format":"jyppx.liuding.onnx_export_inputs","schema_version":1,"input_names":["input"],"output_names":["output"],"minimum_batch":1,"maximum_batch":8,"dynamic_batch":true,"absolute_tolerance":0.00001,"relative_tolerance":0.0001,"cases":[{"id":"batch-2","inputs":[{"name":"input","tensor":"example"}]}]}
```

模型定义和输入 JSON 分别最多 4 MiB，检查点最多 256 MiB，输入与生成参考输出的合计 payload 最多 64 MiB；生成的 safetensors 文件也受 64 MiB 限制。这些是逻辑字节限制，不是 RSS 上界；命令在创建 ORT Session 前释放原模型、检查点快照和参考 Tensor。`delivery.json` 的检查点摘要和大小必须与实际恢复的原 run 引用一致；其他交付字段见下文。`timeout-ms` 限定 ORT 验证阶段，不构成模型构造和 trace 的进程级超时。

输出保存原始模型定义、输入 JSON/Tensor、恢复关联证据及自动生成的模型、报告、参考输出、环境锁和执行证据。恢复证据确认该次执行从指定 run 引用恢复了模型状态；它不认证原代码、生产者或原配置语义，也不证明任务准确率。后续独立 `smoke` 重验模型、验证输入和环境锁，不重新加载原检查点或认证恢复声明；可信签名和独立安装环境的验收仍需完成。

## 实际数值验证与执行证据

`<jyppx/liuding/deployment/onnx_runtime/verification.hpp>` 提供 `verify_session()`：传入现有 Session、具名输入和预期输出，按 `|actual - expected| <= absolute_tolerance + relative_tolerance * |expected|` 比较全部元素。返回的报告包含每个样例的输入/预期/实际输出摘要、误差、失败数和调用耗时，不保留 Tensor 或样本原文；数值不一致返回失败观测，输入/schema/资源错误抛出异常。`inspect_profile()` 解析 `end_profiling()` 生成的原始 ORT trace，统计实际 kernel 的 Provider/算子；仅设置 Provider 或建立 Session 不构成执行证据。耗时属于观测值，不是吞吐、任务准确率或内存资格。

启用导出与 Runtime 后，CLI 可执行并登记验证。样例 JSON 使用 `jyppx.liuding.onnx_verification_cases` v1，字段为 `format`、`schema_version`、`reference`、`absolute_tolerance`、`relative_tolerance`、`cases`；每个 case 含 `id`、`inputs`、`expected`，输入及预期输出以 `{"name":"图中名称","tensor":"safetensors 中的键"}` 绑定。使用库的 safetensors 编码接口保存连续 FP32 数据；`reference` 由调用方注明预期值的计算来源，不能用模型自己的同次输出冒充独立参考。最多 64 个 case，样例数据及展开后的全部 payload 分别限制为 64 MiB。

```sh
liuding-forge model onnx verify \
  --path model.onnx --report export-report.json --report-sha256 REPORT_SHA256 \
  --cases cases.json --cases-sha256 CASES_SHA256 \
  --tensors cases.safetensors --tensors-sha256 TENSORS_SHA256 \
  --provider CPUExecutionProvider --root ./runs --id verify-001 --reference deployment
```

命令核对模型和原训练来源，要求实际 trace 中的 kernel 全部属于显式 Provider，不允许静默 CPU 回退。需要参数时重复使用 `--provider-option key=value`；`--timeout-ms` 默认 60000，属于协作式执行期限。每次验证使用新的 run ID，六个制品（模型、导出报告、样例 JSON、样例 Tensor、验证报告、原始 trace）成批登记到独立 run；数值通过后 completed，数值失败后 failed 并返回退出码 4，输入或后端错误返回 2。原训练保持原状态。验证制品含实际样本和预期值；字节摘要、调用方参考和 run lineage 均不认证生产者，完整 bundle 发布、真实签名及其他设备/Provider 的资格仍需单独验收。

模型已在 [bundle](model_bundle.md) 中时，用下面三个参数替换 `--path model.onnx`，其余报告、样例、Provider 和 run 参数相同：

```sh
--bundle ./model-bundle --bundle-sha256 ORIGINAL_MANIFEST_SHA256 --model-id primary
```

命令完整校验 bundle，按显式模型 ID 选择单文件 ONNX，使用原生句柄读取并保留已校验的同一份模型字节，再进行官方 checker 与实际数值验证。manifest SHA-256 对应原始 `manifest.json` 的完整字节；大小限制为 manifest 4 MiB、单文件 256 MiB、数据文件合计 512 MiB。不得同时传 `--path`，其他模型、未声明文件或任何文件摘要错误均明确拒绝。

使用包外样例且没有下述环境锁时，验证证据封装 `jyppx.liuding.onnx_verification_evidence` 使用 v2，包含 `input_bundle`：普通文件输入时为 null，bundle 输入时记录原始 manifest 摘要、bundle/model/file ID 和签名状态；内部执行报告仍使用原有 v1。bundle 模式额外将原始 manifest 字节登记为第七个制品，数值失败也保留关联。该模式使用调用方提供的原始 FP32 case，不解释包中的预处理、后处理和任务描述符；未检查项在报告中明确为 false。实际执行只授予选中模型和样例的证据，不授予包内其他模型或全部部署环境资格。原生 CLI 检查点导出见上一节；可信签名仍需另行完成。

## 包内证据复验

bundle v2 可同时提供模型、导出报告、case JSON 和 safetensors。后三者在 manifest 的 `evidence` 中各有独立文件 ID，owner schema 和版本必须符合下表；实际内容还会严格解析并与模型及本次输入绑定。

| 参数 | owner schema ID | 版本 | 内容上限 |
| --- | --- | --- | --- |
| `--report-id` | `jyppx.liuding.onnx_export_report` | 1 | 4 MiB |
| `--cases-id` | `jyppx.liuding.onnx_verification_cases` | 1 | 4 MiB |
| `--tensors-id` | `jyppx.liuding.onnx_verification_tensors` | 1 | 64 MiB，FP32 safetensors |

```sh
liuding-forge model onnx smoke \
  --bundle ./model-bundle --bundle-sha256 ORIGINAL_MANIFEST_SHA256 --model-id primary \
  --report-id export-report --cases-id cases --tensors-id tensors \
  --provider CPUExecutionProvider --format json
```

`smoke` 不需要原训练 run 数据库或包外模型/样例文件。它完整校验目录，保留同一次校验读取的模型和三件输入，重新运行官方 checker、严格解析全部 case/Tensor 关联，再执行 ORT 并比较全部具名输出。模型、manifest 和目录总额沿用上述限制；快照保留选中的四文件，已知字节预算最多 332 MiB；新环境锁另计至多 64 KiB。这不是 RSS 上限，发现文件 ID 与保留最终字节时仍会各做一次完整校验。

数值通过返回 0，数值不一致返回 4；输入/后端错误返回 2，I/O 错误返回 3。JSON 输出包含新执行报告和原始 `runtime_profile` 字符串，后者的 UTF-8 字节摘要与报告中的 profile SHA-256 相同；原始 trace 最多 16 MiB。命令不修改模型包、不创建 run；需要留存时保存完整输出。包内预期值和导出来源仍是声明，不认证生产者，也不证明任务准确率、完整 runtime requirements、其他模型或性能资格。

需要把本次复验登记到已有原训练 lineage 时，将同一组参数用于 `model onnx verify`，另加 `--root`、`--id`、`--reference`。不要混用包内 ID 与包外 `--path`、`--report`、`--cases`、`--tensors` 或其摘要。成功和数值失败均登记原始 manifest 与六件执行制品，有环境锁时另登记其精确字节；仍须能在指定 run store 中找到报告声明的训练来源。

包内输入且没有新环境锁时生成验证证据封装 v3，增加 `verification_inputs` 的文件 ID 与逐项语义检查结果，以及 `source_run_lineage_verified`：`smoke` 为 false，核对原训练的 `verify` 为 true。外部输入模式保持 v2，内部 Runtime 执行报告保持 v1。`descriptor_payloads_validated=false` 表示未解析包内全部通用描述符；已校验的验证输入单独列明。预后处理、任务评估、SBOM 内容与真实签名仍须按各自契约验收。

## 自动组包与环境锁

`model onnx pack` 接受与 `verify` 相同的模型、报告、样例和 Provider 输入，额外提供交付元数据与新输出目录。它重新执行 checker 和全部数值比较，通过后自动生成 bundle v2 的文件清单、实际图 I/O 描述、provenance、环境锁及新执行证据，原子发布到新目录；数值失败返回 4 并在输出中保留报告和原始 trace，不发布目录。已有输出始终拒绝覆盖。整个流程由 C++ 完成。

```sh
liuding-forge model onnx pack \
  --path model.onnx --report export-report.json --report-sha256 REPORT_SHA256 \
  --cases cases.json --cases-sha256 CASES_SHA256 \
  --tensors cases.safetensors --tensors-sha256 TENSORS_SHA256 \
  --metadata delivery.json --metadata-sha256 DELIVERY_SHA256 \
  --provider CPUExecutionProvider --output ./deployment-bundle --format json
```

`delivery.json` 使用 `jyppx.liuding.onnx_delivery_metadata` v1，最多 4 MiB，拒绝未知字段。模型的导出报告须显式包含原训练 run/code/config/data 引用。交付元数据字段如下，调用方提供实际内容，无需编写 manifest、文件列表、摘要或 Runtime 声明。

| 字段 | 内容 |
| --- | --- |
| `format`、`schema_version` | 上述格式 ID 与整数 1 |
| `bundle_id` | 本次模型包身份 |
| `checkpoint` | 原检查点的 `sha256` 与正整数 `size_bytes`；这是调用方声明的关联，不内嵌检查点 |
| `preprocessing`、`postprocessing`、`evaluation`、`sbom` | 各为 `schema_id`、正整数 `schema_version`、JSON 对象 `content`；填写所属 schema 的真实内容 |
| `labels` | 同样的 owner 描述，或显式 null |

输出固定选择 ID 为 `primary`、`export-report`、`cases`、`tensors`，可直接用于上一节的 `smoke`。包内保存原模型、报告、case、Tensor 和元数据的精确字节，以及本次验证报告与 ORT trace。预后处理、任务报告和 SBOM 按调用方 schema 保存，通用组包命令不替它们授予任务质量、完整 SBOM 或来源认证资格。生成的 provenance 关联 run/code/config/data/checkpoint、元数据和导出/执行证据；检查点与原 run 的关联仍须在来源重建流程中独立证明。

环境锁使用 `jyppx.liuding.onnx_runtime_lock` v1，作为独立 evidence 保存，最多 64 KiB；manifest 的 `environment_lock` 是其原始字节摘要。它绑定框架与 LibTorch 构建版本、实际加载的 ORT 版本、平台、FP32 I/O、实际图的动态 batch、显式 Provider 名称/参数与 CPU 回退策略。执行域在 CPU EP 下为 `cpu`，CUDA/TensorRT EP 下为 `cuda`，其他 EP 使用完整 Provider 名称的小写形式；它不表达硬件型号或驱动资格。

当 backend ID 为 `jyppx.liuding.onnx_runtime` 时，`smoke`/`verify` 要求恰好一件受支持的锁，在同次完整校验中保留，执行前检查其与 manifest 及本次 Runtime 观察完全一致。版本、平台、依赖、dtype、动态声明或 Provider 参数不匹配均拒绝；未知依赖不被忽略。这类复验生成证据封装 v4，并明确记录 `runtime_requirements_checked=true`、锁 ID 和摘要；`verify` 共登记八件制品。其他 owner 的声明继续明确未检查，不获得兼容资格。锁只覆盖上述执行点，驱动/ABI 的完整支持矩阵、设备资源和性能、其他平台与真实签名仍需分别验收。

## 公共环境锁与 Serving

`<jyppx/liuding/deployment/onnx_runtime/environment.hpp>` 提供 `EnvironmentLock` 与 `EnvironmentRequirements`：`observe_environment(session)` 观察实际执行环境，`serialize_environment_lock()`、`parse_environment_lock()` 保存和严格解析上述 v1 格式，`verify_environment_lock()` 核对原始字节摘要、声明及实际 Session 的版本、平台、依赖和 Provider/options/fallback。该 API 属于 `onnx_runtime`，可独立于导出器与 bundle 模块使用；环境匹配不替代来源认证或数值验收。

需要 bundle 加载、已知向量 warmup 与受控版本切换时，使用独立可选的 [`onnx_serving`](serving.md) 组件；完整应用见 [C++ ONNX Serving 示例](../../examples/serving/README.md#onnx-bundle-与多输入输出)。
