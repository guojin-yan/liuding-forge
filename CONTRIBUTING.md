# 参与六丁炼算（Liuding Forge）开发

提交问题时，请提供复现步骤、构建配置、LibTorch/编译器版本，以及必要的错误输出。修改代码时，围绕一个明确问题提交可审查的改动，并说明行为变化、验证结果和已知限制。

## 构建与验证

项目使用 C++20 和 CMake 3.18+。先查看 [compatibility.json](compatibility.json) 和 [dependency-manifest.json](dependency-manifest.json)，准备对应的编译器、LibTorch 和依赖；已验证的版本组合与待验证组合分别记录，不能从版本范围推导支持。Windows 官方 Release LibTorch 要求匹配 Release CRT 和 `_ITERATOR_DEBUG_LEVEL=0`。

兼容矩阵中的 `verified_lanes` 仅记录各自源码、组件 profile、设备和实际测试范围；开发切片通过不代表当前完整支持矩阵或 GA。`pending_lanes` 是尚未验证的精确候选，继承所属版本组的 LibTorch 版本，但必须显式声明自己的 `distribution`、环境和所需验证，保持 `status=candidate`、`verification_status=unverified`，不得填写测试成绩、验证证据或验证提交；只有真实验证完成后才能登记到 `verified_lanes`。

LibTorch 2.14.1+cpu / GCC 11.4.0 / Ubuntu 22.04.5 x86_64 的选定 Debug coverage profile 已在干净提交 `4136aab` 上验证，包含原覆盖率策略和独立安装 SDK 消费者；2039 项合同分阶段为 1847 通过、192 条件跳过。该 profile 关闭 examples/docs 并排除 packaging 标签，不授予被跳过路径、其他平台/设备或完整发布资格；具体范围见兼容矩阵对应的 `verified_lanes`。启用通用 `onnx_export` 的 profile 要求精确 LibTorch 2.14.1，不能为复用历史 coverage 树降级依赖。

同一 CPU/编译器组合的 Release SDK 已在干净提交 `57e304c` 完成18组件安装、独立八Core消费者和完整选定 2076 项合同分阶段验收（1887 通过、189 条件跳过、零未解决失败）；tests/examples/docs开启。保留维护中断，两组件关闭矩阵及被跳过路径仍需各自验收；实际安装指纹保持 `57e304c`，不随后续元数据提交变更。

在仓库根目录分别运行以下两条命令，检查兼容矩阵和依赖声明：

```sh
python scripts/validation/compatibility.py --file compatibility.json --validate
python scripts/validation/dependency_manifest.py --file dependency-manifest.json
```

复用兼容的现有构建目录，保留其中的 generator、工具链和依赖路径；先检查目录内实际 `CMakeCache.txt` 的版本、配置和依赖身份，不能仅凭目录存在推定可复用。以下是已有兼容 Windows cache 时的增量构建示例：

```sh
cmake -S . -B out/build/config-windows -DFETCHCONTENT_FULLY_DISCONNECTED=ON
cmake --build out/build/config-windows --config Release --target jyppx_test_core --parallel 2
ctest --test-dir out/build/config-windows -C Release --output-on-failure -R "^(DeviceTest|TensorUtilsTest)\."
```

上述命令中的 `python` 应指向实际安装的 Python 3 解释器。Linux 使用对应的兼容构建目录；单配置 generator 可省略 `--config Release` 和 `-C Release`。没有可复用目录时，先按兼容矩阵准备依赖，再显式配置 `CMAKE_PREFIX_PATH` 和需要的组件。配置阶段不下载依赖；可选组件只按当前任务启用。

ONNX 独立验证可显式设置 `JYPPX_LIUDING_ONNX_REFERENCE_PYTHON`，使用已准备的测试环境执行 `OfficialNativeOnnxExport` 和 `OfficialOnnxRuntimeInterchange`；它只参与 checker/参考数据测试，不进入产品导出或安装依赖。

将示例中的 target 和测试选择替换为实际受影响的范围。公共头、CMake target 或组件依赖变化还需运行以下检查，并验证安装后消费者及相关组件关闭组合：

```sh
ctest --test-dir out/build/config-windows -C Release --output-on-failure -R "^(ComponentDependencyGraph|RepositoryHygiene|PublicHeaderGuards)$"
```

新增行为与缺陷修复应有能验证契约的测试，覆盖适用的错误、取消、资源边界和恢复路径。测试报告应注明源码身份、环境、通过/失败/跳过数量；跳过硬件测试不代表该环境已支持。覆盖率遵循 [coverage-policy.json](coverage-policy.json) 的组件策略。

## 代码与性能要求

项目是 C++20 训练平台，所有交付给用户的功能（训练、数据准备、权重转换、ONNX 导出、推理和工具）必须由 C++ 实现；不得通过 Python/sidecar/subprocess 替代实现，也不得让安装后的用户必须具备 Python 环境。Python 仅用于开发维护脚本、官方 checker、独立参考和测试辅助，不能成为产品功能的生产入口。存量 Python 产品路径需按模块迁移并重新验收，历史通过不豁免此约束。

- 公共接口使用 `jyppx::liuding`、`<jyppx/liuding/...>` 和导出目标 `jyppx::liuding::<component>`；配置选项使用 `JYPPX_LIUDING_*`。组件依赖遵循 [依赖图约定](docs/development/component_dependencies.md)。
- 使用仓库 `.clang-format`。公共声明的 Doxygen 注释应说明必要的所有权、生命周期、线程、错误、状态、单位和资源契约；严格质量配置还需通过 clang-tidy 和 Doxygen 检查。
- C++ 文件保留产品名和项目别名文件头。修改已有接口时同步安装面、示例和用户文档，不保留已退役入口的隐式转发。
- worker 异常应回传到消费者；提前退出、取消和析构须回收线程、队列与持有的资源。错误路径应保持声明的状态保证。
- 数据准备与训练路径应控制复制、暂存、预取和设备传输，关注端到端耗时、峰值内存和显存。性能声明须绑定相同工作负载、环境和源码身份，并说明资源取舍。

性能采样方法见 [benchmark 指南](docs/benchmarks/README.md)，CUDA 训练测量使用 `scripts/benchmarks/run_cuda_baseline.py`，coverage 使用 `scripts/validation/run-coverage.sh`。保留规定的预热和独立采样；编译时间比较需要对应的干净构建。不同 GPU、驱动、LibTorch、编译器或工作负载的结果不能直接套用同一回归阈值。

## 文档与文件归属

文档优先面向库的使用者：教程提供完整步骤，API、设计和维护说明保留当前契约、必要操作与支持边界。先更新已有页面并使用链接引用共用内容；README 中英文分文件，其他文档当前使用中文。自然语言段落和列表项各占一行，由 Markdown 渲染器自动换行；代码、表格和目录树保留结构换行。

源码、测试、产品工具及可复用维护流程放在仓库；开发报告、审计、历史测量和原始验收证据放在仓库外的方案记录目录。生成物、下载数据、隔离环境和运行日志使用被忽略的目录。复用现有构建，不清理其他任务正在使用的产物。详细规则见 [仓库卫生与测试数据](docs/development/repository_hygiene.md)。

仓库级 AI 工作流位于 [.agents/skills/](.agents/skills/)：构建验证、模型接入和文档维护分别使用对应技能；技能不能替代测试或人工审查。模型架构、预训练权重、预处理和任务质量应分别举证，不能用形状或单步训练通过推导准确率和完整设备支持。

## 提交与交付

当前主线为 `main`。提交前检查差异，只包含当前任务的文件，并补齐必要验证和用户可见变更的 [CHANGELOG.md](CHANGELOG.md)。提交消息采用 `feat(scope): ...`、`fix(scope): ...`、`docs(scope): ...` 等明确形式；交付说明先写解决的问题与结果，再写验证和限制。

GitHub 工作流当前只通过 `workflow_dispatch` 手动触发，开发期推送不自动运行 Actions。正式发布另需完成 [发布流程](docs/development/releasing.md) 中的支持矩阵、安全、制品和升级回退验收。

## 贡献许可

除非事先明确说明，主动提交并纳入本项目的贡献采用 [Apache License 2.0](LICENSE)，不附加其他条款。只提交自己拥有或获准贡献的内容，并保留适用的第三方版权、专利和署名声明。

复制、翻译或实质改写上游源码时，应同时记录原始项目、准确版本或提交、文件路径、修改范围和适用许可，并保留原文；项目文件头和语言转换不改变上游材料的授权条件。模型架构实现、参考测试、权重及数据集分别记录来源；涉及模型的现有边界见 [模型来源与许可](third_party/licenses/MODEL_PROVENANCE.md)，尚未完成的来源核查不得直接标记为已审查。
