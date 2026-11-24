# 参与六丁炼算（Liuding Forge）开发

提交问题时，请提供复现步骤、构建配置、LibTorch/编译器版本，以及必要的错误输出。修改代码时，围绕一个明确问题提交可审查的改动，并说明行为变化、验证结果和已知限制。

产品版本只按明确的发布安排修改。功能开发、模块调整和分支合并不自动修改 CMake、兼容清单或打包版本；尚未发布的用户可见变更记入 `CHANGELOG.md` 的 `Unreleased`。

## 构建与验证

项目使用 C++20 和 CMake 3.18+。先查看 [compatibility.json](compatibility.json) 和 [dependency-manifest.json](dependency-manifest.json)，准备对应的编译器、LibTorch 和依赖；已验证的版本组合与待验证组合分别记录，不能从版本范围推导支持。Windows 官方 Release LibTorch 要求匹配 Release CRT 和 `_ITERATOR_DEBUG_LEVEL=0`。

兼容矩阵中的 `verified_lanes` 仅记录各自源码、组件 profile、设备和实际测试范围；局部验证通过不代表完整支持矩阵或正式发布资格。`pending_lanes` 是尚未验证的精确候选，继承所属版本组的 LibTorch 版本，但必须显式声明自己的 `distribution`、环境和所需验证，保持 `status=candidate`、`verification_status=unverified`，不得填写测试成绩、验证证据或验证提交；只有真实验证完成后才能登记到 `verified_lanes`。

在仓库根目录分别运行以下两条命令，检查兼容矩阵和依赖声明：

```sh
python scripts/validation/compatibility.py --file compatibility.json --validate
python scripts/validation/dependency_manifest.py --file dependency-manifest.json
```

组件矩阵的子配置继承父构建明确选择的编译器、toolchain 文件、构建程序、开发用 Python 解释器和 SDK 搜索根，以及离线依赖策略；子配置使用独立于父构建的构建和安装暂存，不复用父构建的可变依赖目录或产品组件开关。检查 ROCm 等需要特定编译器的配置时，同样核对子配置的 cache。

复用兼容的现有构建目录，保留其中的 generator、工具链和依赖路径；先检查目录内实际 `CMakeCache.txt` 的版本、配置和依赖身份，不能仅凭目录存在推定可复用。以下是已有兼容 Windows cache 时的增量构建示例：

```sh
cmake -S . -B out/build/windows-release -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_UPDATES_DISCONNECTED=ON
cmake --build out/build/windows-release --config Release --target jyppx_test_core --parallel 1
ctest --test-dir out/build/windows-release -C Release --output-on-failure -j 1 -R "^(DeviceTest|TensorUtilsTest)\."
```

上述命令中的 `python` 应指向实际安装的 Python 3 解释器。Linux 使用对应的兼容构建目录；单配置 generator 可省略 `--config Release` 和 `-C Release`。没有可复用目录时，先按兼容矩阵准备依赖，再显式配置 `CMAKE_PREFIX_PATH` 和需要的组件。配置阶段不下载依赖；可选组件只按当前任务启用。

ONNX 独立验证可显式设置 `JYPPX_LIUDING_ONNX_REFERENCE_PYTHON`，使用已准备的测试环境执行 `OfficialNativeOnnxExport` 和 `OfficialOnnxRuntimeInterchange`；它只参与 checker/参考数据测试，不进入产品导出或安装依赖。

Linux 原生 HTTP 的 `NativeOtlpHttpTrustedTls` 与 `NativeOtlpHttpTrustedDns` 使用独立 user/mount namespace 验证可信 TLS、过期和 IP/DNS 名称拒绝。`JYPPX_LIUDING_TELEMETRY_HTTP_TEST_CA_BUNDLE` 必须指向所选 curl SDK 实际使用的默认 CA 文件，默认是 `/etc/ssl/certs/ca-certificates.crt`；可用该 SDK 的 `curl-config --ca` 核对。此选项仅选择测试的私有信任视图，不设置产品 CA、关闭证书验证或修改宿主信任库；缺少 namespace 能力时测试失败。

`NativeOtlpHttpTrustedTlsCancellation` 使用同一私有信任方式，将原响应头、响应体停滞时的取消与截止合同分别运行在已建立的 TLS 1.2 `localhost` 连接上。它保留原八轮取消、两轮截止测量、有限预热、严格文件描述符恢复、未确认事件原字节重试及进程内重复检查，原六场景 `NativeOtlpHttpCancellation` 独立保留；两项各保持 180 秒 CTest 上限。

启用 `JYPPX_LIUDING_TEST_NATIVE_OTLP_COLLECTOR` 并指定锁定的 `JYPPX_OTELCOL_EXECUTABLE`、`JYPPX_TELEMETRY_PYTHON` 后，Linux 的 `NativeOtlpCollectorTrustedTls` 使用同一私有信任方式，将原 C++ API、SQLite/CLI 与 worker 的三组事件发送到官方 Collector 的 TLS 1.2 `localhost` 接收器，并由官方 protobuf reader 核对完整信号。测试保留证书、Collector 配置与日志、原始输出和读回报告；原明文互通测试独立保留。HTTPS 互通不授予证书吊销、持久 ACK 或 HTTPS 崩溃恢复资格。

同一配置的 `NativeOtlpCollectorTrustedTlsDurableQueue` 独立验证 HTTPS 接收器与原 `fsync` 队列的组合，保留三信号各一次持久队列强杀恢复和易失队列丢失对照、原 90 秒 CTest 及进程/内容断言。每次 Collector 启动核对 TLS 版本与证书；下游仍使用 HTTP 回环，不能推导下游 HTTPS、通用持久 ACK 或断电恢复。

`NativeOtlpCollectorTrustedTlsQueueFull` 和 `NativeOtlpCollectorTrustedTlsLostReply` 分别保留原 16 事件队列满与 3 事件未知确认重试合同，原生客户端至记录端点、记录端点至官方 Collector 均使用 TLS 1.2 `localhost`。记录端点只转发原 protobuf 字节与真实响应，并在丢回复场景丢弃一次实际成功响应；两项各保留原 120 秒 CTest、原确认前缀与逐字节重试断言，不提供 exactly-once。

外部 MIGraphX 的原生 GPU 互操作测试默认关闭。已激活匹配的 ROCm SDK 并备妥 MIGraphX 2.18.0 时，可在兼容构建中增加 `-DJYPPX_LIUDING_TEST_MIGRAPHX=ON -DJYPPX_LIUDING_MIGRAPHX_TEST_ROOT=<原生SDK绝对路径> -DJYPPX_LIUDING_MIGRAPHX_TEST_FIXTURE_DIR=<固定参考绝对路径>`，构建 `jyppx_test_migraphx_interchange` 后串行运行 `ctest --test-dir <构建目录> -C Release --output-on-failure -j 1 -R "^MIGraphXInterchange[.]"`。参考目录的 `linear/`、`resnet18/` 各含原 `model.onnx` 及 batch 2/5 的 `case-{0,1}-{input,output}.f32`；测试内固定完整 SHA，不下载、重新生成或更新参考。四项测试核对原 FP32 容差、实际 GPU 指令及同步主存输出，每次向构建目录的 `evidence/migraphx-interchange/` 新目录保留完整输出、编译程序、路线与加载模块。每项最多 600 秒，JIT 并行度为 1；缺硬件或错误 SDK 直接失败。外部图执行测试不提供框架 MIGraphX Session、Serving 或 ORT EP，也不授予动态形状、其他图、精度或性能资格。

同一配置下运行 `ctest --test-dir <构建目录> -C Release --output-on-failure -j 1 -R "^MIGraphXReload[.]"`，另四项测试先在原生进程中编译并执行，再将程序和完整输出的 SHA 传给独立重载进程。重载分支不调用 ONNX 解析或程序编译 API，核验后从证据目录的私有副本加载，保持原 shape、数值和 GPU 路线检查，并要求输出逐字节等于首次执行。编译子进程最多 600 秒，重载子进程最多 60 秒，每项 CTest 最多 700 秒；两次执行的日志、程序、输出、路线及模块留在 `evidence/migraphx-reload/` 新目录。此契约仅适用于同一设备和精确 SDK，不推导编译程序可跨设备、SDK 或操作系统移植。

已登记的 WSL2/W7900 原生 SDK 组合另通过关闭 MLIR 后的 ResNet18 batch 2/5 编译与独立进程严格重载，保留相同输入、原 FP32 容差及 GPU 算术/拷贝/同步要求。可在启动测试进程时选择该路线：

```sh
MIGRAPHX_DISABLE_MLIR=1 ctest --test-dir <构建目录> -C Release --output-on-failure -j 1 -R '^MIGraphXReload[.]resnet18_[01]$'
```

此变量由 MIGraphX 在首次使用时读取并缓存，影响进程内后续编译；不能改变已编译程序。该范围仅覆盖上述原生测试及精确 SDK，默认 MLIR 路线的严格重载失败继续保留；不保证其他模型、形状、产品 Session、Serving 或扩展 SDK 的逐位确定性。编译时保持同一进程环境，并在需要逐位输出时独立验收。

将示例中的 target 和测试选择替换为实际受影响的范围。公共头、CMake target 或组件依赖变化还需运行以下检查，并验证安装后消费者及相关组件关闭组合：

Windows ROCm 的 `VisionTasksInstallConsumer`、`DataImageInstallConsumer` 和 `SdkInstallConsumer` 继承父构建显式启用的 `JYPPX_LIUDING_STAGE_ROCM_RUNTIME_DLLS`，在其独立消费者构建目录暂存相同 HIP 导入闭包及 SDK `.kpack` 归档；重复构建及关闭该选择会更新暂存清单，并只移除清单拥有且未修改的旧文件。该测试设置不写入安装包或更改应用的 SDK 部署要求。

```sh
ctest --test-dir out/build/windows-release -C Release --output-on-failure -j 1 -R "^(ComponentDependencyGraph|RepositoryHygiene|PublicHeaderGuards)$"
```

新增行为与缺陷修复应有能验证契约的测试，覆盖适用的错误、取消、资源边界和恢复路径。测试报告应注明源码身份、环境、通过/失败/跳过数量；跳过硬件测试不代表该环境已支持。覆盖率遵循 [coverage-policy.json](coverage-policy.json) 的组件策略。

## 代码与性能要求

项目是 C++20 训练平台，所有交付给用户的功能（训练、数据准备、权重转换、ONNX 导出、推理和工具）必须由 C++ 实现；不得通过 Python/sidecar/subprocess 替代实现，也不得让安装后的用户必须具备 Python 环境。Python 仅用于开发维护脚本、官方 checker、独立参考和测试辅助，不能成为产品功能的生产入口。存量 Python 产品路径需按模块迁移并重新验收，历史通过不豁免此约束。

- 公共接口使用 `jyppx::liuding`、`<jyppx/liuding/...>` 和导出目标 `jyppx::liuding::<component>`；配置选项使用 `JYPPX_LIUDING_*`。组件依赖遵循 [依赖图约定](docs/development/component_dependencies.md)。
- 使用仓库 `.clang-format`。公共声明的 Doxygen 注释应说明必要的所有权、生命周期、线程、错误、状态、单位和资源契约；严格质量配置还需通过 clang-tidy 和 Doxygen 检查。
- C++ 文件保留产品名和项目别名文件头。修改已有接口时同步安装面、示例和用户文档，不保留已退役入口的隐式转发。
- worker 异常应回传到消费者；提前退出、取消和析构须回收线程、队列与持有的资源。错误路径应保持声明的状态保证。
- 按模块组织头文件、源码和相关测试：同一模块先集中在各自对应的大目录下，再按职责分子目录，私有 `.hpp` 与 `.cpp` 就近存放，共享代码归入实际所属模块。现有公开 include 路径和安装接口保持稳定；搬迁时同步 CMake 和引用路径，内部实现、工具及示例沿用此布局。
- 数据准备与训练路径应控制复制、暂存、预取和设备传输，关注端到端耗时、峰值内存和显存。性能声明须绑定相同工作负载、环境和源码身份，并说明资源取舍。

性能采样方法见 [benchmark 指南](docs/benchmarks/README.md)，CUDA 训练测量使用 `scripts/benchmarks/run_cuda_baseline.py`，coverage 使用 `scripts/validation/run-coverage.sh`。保留规定的预热和独立采样；编译时间比较需要对应的干净构建。不同 GPU、驱动、LibTorch、编译器或工作负载的结果不能直接套用同一回归阈值。

## 文档与文件归属

文档优先面向库的使用者：教程提供完整步骤，API、设计和维护说明保留当前契约、必要操作与支持边界。先更新已有页面并使用链接引用共用内容；README 中英文分文件，其他文档当前使用中文。自然语言段落和列表项各占一行，由 Markdown 渲染器自动换行；代码、表格和目录树保留结构换行。

源码、测试、产品工具及可复用维护流程放在仓库；开发报告、审计、历史测量和原始验收证据放在仓库外的方案记录目录。生成物、下载数据、隔离环境和运行日志使用被忽略的目录。复用现有构建，不清理其他任务正在使用的产物。详细规则见 [仓库卫生与测试数据](docs/development/repository_hygiene.md)。

仓库级 AI 工作流位于 [.agents/skills/](.agents/skills/)：构建验证、模型接入和文档维护分别使用对应技能；技能不能替代测试或人工审查。模型架构、预训练权重、预处理和任务质量应分别举证，不能用形状或单步训练通过推导准确率和完整设备支持。

DINOv2 独立随机架构参考使用 `tests/unit/models/reference/generate_dinov2_reference.py --capacity small|base|large|giant --output-dir <空目录> --jyppx-commit <完整提交>`；默认 Small，Giant 另需 `--giant-gate` 并串行执行。Base/Large/Giant 保留原 Python 3.11.9、CPU Torch 2.14.1、torchvision 0.29.1、safetensors 0.6.2 和脚本锁定的 timm 源码身份、seed 与完整文件 SHA；生成器按张量写入大容量状态并用独立 safetensors reader 核对，拒绝覆盖已有产物。先检查可用物理内存、提交额度与磁盘空间，再运行对应原生门禁；此开发工具不进入安装依赖或产品运行流程。

主存不足以执行 Giant CPU FP32 反向时，可单独启用 `JYPPX_LIUDING_DINOV2_GIANT_CPU_BF16_STORAGE_GATE=1` 并运行 `^Models[.]DinoV2GiantCpuBfloat16StorageTrainingResourceGate$`。该用例将参数和梯度存为 BF16，使用 `1×3×14×14` 输入、FP32 loss 与一次 SGD 更新，检查全部梯度有限以及分类头和 patch 权重实际改变，峰值 RSS 上限为 8 GiB；构造时仍先分配 FP32 权重。须串行运行并检查可用物理内存和提交额度；此独立用例不替代原 FP32 反向、独立数值参考或任务质量门禁。

排查普通架构固定参考时，可将 `JYPPX_LIUDING_MODELS_REFERENCE_CAPTURE_DIR` 设为仓库外已存在的父目录，再单独运行对应 CPU 或 GPU 参考用例。测试仅在显式启用时将已计算的输出和适用中间特征写入 `<架构>-cpu/outputs.safetensors` 或 `<架构>-cuda0/outputs.safetensors`，打印完整文件 SHA；子目录必须不存在，编码文件上限为 64 MiB。采集保留原数值断言，超差用例仍失败；产物只用于开发诊断，不替换固定参考，也不进入安装包。

DINOv2-small 真实首批次对照可显式设置 `JYPPX_LIUDING_DINOV2_REAL_FIRST_STEP_CAPTURE_DIR`，再单独运行 `ModelsPretrainedBackward.DinoV2SmallRealFirstBatchCpuGradientsAndSgd` 或对应的 `CudaGradientsAndSgd`。父目录须已存在，测试将 logits、loss、六份完整梯度及六份更新参数保存到新建的 `dinov2-small-cpu/outputs.safetensors` 或 `dinov2-small-cuda0/outputs.safetensors`，打印完整 SHA，编码文件上限为 64 MiB；原固定权重、输入、容差及失败结果保持。与官方同后端输出比较时，分别记录双方的确定性设置，不能将指定张量一致推导为所有参数梯度或任务质量一致。

## 提交与交付

当前主线为 `main`。提交前检查差异，只包含当前任务的文件，并补齐必要验证和用户可见变更的 [CHANGELOG.md](CHANGELOG.md)。提交消息采用 `feat(scope): ...`、`fix(scope): ...`、`docs(scope): ...` 等明确形式；交付说明先写解决的问题与结果，再写验证和限制。

GitHub 工作流当前只通过 `workflow_dispatch` 手动触发，开发期推送不自动运行 Actions。正式发布另需完成 [发布流程](docs/development/releasing.md) 中的支持矩阵、安全、制品和升级回退验收。

## 贡献许可

除非事先明确说明，主动提交并纳入本项目的贡献采用 [Apache License 2.0](LICENSE)，不附加其他条款。只提交自己拥有或获准贡献的内容，并保留适用的第三方版权、专利和署名声明。

复制、翻译或实质改写上游源码时，应同时记录原始项目、准确版本或提交、文件路径、修改范围和适用许可，并保留原文；项目文件头和语言转换不改变上游材料的授权条件。模型架构实现、参考测试、权重及数据集分别记录来源；涉及模型的现有边界见 [模型来源与许可](third_party/licenses/MODEL_PROVENANCE.md)，尚未完成的来源核查不得直接标记为已审查。
