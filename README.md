# 六丁炼算（Liuding Forge）

[English README](README.en.md)

六丁炼算（Liuding Forge）是一个基于 LibTorch 的 C++20 深度学习训练框架，面向需要原生 C++ 训练、可恢复生命周期、本地实验和交付工具的工程团队。JYPPX 是产品群总品牌，JYPPX Training Framework 是本项目的内部别名；对外产品名称、内部别名和代码技术身份分别承担不同职责。

> 当前状态：项目仍处于 v1.0 开发阶段。已有功能和验证结果只代表对应的代码范围与环境，不代表正式 GA 发布或与 PyTorch 全面等价。

## 项目定位

项目对外名称为六丁炼算（Liuding Forge）。项目重点是可靠的训练生命周期、数据管线、优化器与调度器、评估、检查点、实验制品、诊断和交付工具，不负责实现 Web 页面、团队服务、RBAC、多租户或远程控制面。

公共 C++ 根命名空间为 `jyppx::liuding`，公共头文件使用 `<jyppx/liuding/...>`，安装包通过 `find_package(jyppx)` 发现，导出 target 使用 `jyppx::liuding::<component>`。逻辑 CMake target、`JYPPX_*` 配置变量和协议字段保持技术身份；生成库、CLI、示例和归档使用 `liuding-forge` 产品前缀。

JYPPX Training Framework 作为内部项目别名保留，用于方案、工程协作和产品群内部识别；它不是对外产品名，也不是新的 C++ 命名空间。

## 我们解决的问题

PyTorch 以 Python 生态和研究效率见长，LibTorch 提供了 C++ tensor 与神经网络基础，但工程团队在 C++ 中还需要自己处理训练生命周期、可恢复状态、数据并发、实验制品、环境身份和交付边界。Liuding Forge 在 LibTorch 之上补充这些工程能力，并通过显式合约、版本化状态和可验证证据降低长期维护成本。

项目的主要优势是：

- 面向 C++ 生产集成，公共 API、CMake package、安装消费和 ABI/运行环境边界一起设计。
- 将模型、优化器、调度器、数据、策略、RNG、AMP、回调和任务状态纳入显式 checkpoint domain，避免依赖隐式全局状态。
- 为 map/iterable 数据、worker 生命周期、结构化 batch、能力描述和失败清理定义有界行为。
- 用 BuildFingerprint、依赖锁、工作负载和设备身份绑定性能与发布证据，避免把不同环境的数字直接比较。
- 将本地运行、制品、评估、遥测、bundle 和 serving 设计为可替换的边界，便于与外部平台集成。

## 功能模块对比与开发进度

下表按工程基础、基础计算、数据管线、模型系统、训练系统、实验与评估、交付工具、扩展能力和当前范围外能力划分。它同时展示 PyTorch 的对应能力和 Liuding Forge 当前的开发状态；状态只使用“已具备”“部分具备”“开发中”“暂不具备”。所有判断均限于当前 v1.0 代码范围和声明环境。

| 功能模块 | 详细说明 | 当前库是否具备 | PyTorch 是否具备 | 其他说明 |
| --- | --- | --- | --- | --- |
| **工程基础：构建与安装** | 命名边界、依赖方向、CMake package、安装消费和仓库卫生 | 已具备 | 部分具备 | Liuding Forge 的公共 C++ 身份为 `jyppx::liuding`；逻辑 CMake target 和 `JYPPX_*` 变量保留技术身份。 |
| **基础计算：Tensor 与自动微分** | 张量运算、神经网络模块和 autograd 基础 | 已具备（依赖 LibTorch） | 已具备 | Liuding Forge 直接使用 LibTorch，不重新实现 tensor/autograd；能力边界由匹配的 LibTorch 版本决定。 |
| **基础计算：基础组件与损失函数** | foundation 合约、错误/RNG/build fingerprint 和常用 loss 垂直切片 | 已具备 | 已具备 | 属于 C++ 工程基础层；不宣称覆盖 PyTorch 全部损失函数和扩展生态。 |
| **数据管线：数据集与数据加载** | map/iterable pipeline、worker、sampler、确定性 cursor 和数据生命周期 | 已具备 | 已具备 | 核心 C++ 管线已实现；多进程 distributed 数据加载不在当前范围。 |
| **数据管线：结构化 Batch 与能力描述** | batch schema、字段访问、能力声明和受约束的 collate 行为 | 已具备 | 部分具备 | PyTorch 可通过 Dataset、collate 和第三方类型组合实现，但没有同等的本项目能力描述合约。 |
| **模型系统：模型适配与模型生态** | ModelAdapter、模型 checkpoint participant、受控参考模型和模型预览 | 部分具备（持续开发） | 生态具备 | Liuding Forge 提供工程适配边界，不宣称与 PyTorch 官方及社区 model zoo 等价。 |
| **训练系统：优化器** | SGD、Adam、AdamW、RMSprop、Adagrad、NAdam、Adadelta、LAMB adapter 及状态恢复 | 已具备 | 已具备 | 主要优化器已接入；跨版本迁移、稀疏参数和更大扩展生态仍有限。 |
| **训练系统：学习率调度** | descriptor-first scheduler、组合调度、训练事件适配和精确恢复 | 已具备 | 已具备 | 调度器集合在显式合约下可恢复；迁移语义受版本和配置约束。 |
| **训练系统：Trainer 与训练编排** | Trainer、callback、metric、strategy、AMP、多 optimizer 和 checkpoint domain | 已具备 | 生态具备 | 当前交付的是 C++ 训练闭环；PyTorch 核心提供训练原语，上层编排通常来自生态项目。 |
| **训练系统：检查点与恢复** | storage-neutral participant、事务协调、文件存储、CAS/SQLite 元数据和有界恢复 | 已具备 | 已具备 | 不兼容任意 PyTorch pickle；恢复范围由显式 participant 和 artifact manifest 决定。 |
| **训练系统：混合精度** | CPU bfloat16、CUDA float16/bfloat16、GradScaler 和 RNG participant | 部分具备 | 已具备 | 只覆盖声明组合；设备、编译器、CUDA 和 LibTorch 必须与兼容矩阵匹配。 |
| **训练系统：指标、回调与事件** | 训练指标、生命周期回调、事件分发和调度器事件适配 | 已具备 | 生态具备 | 当前能力服务于本项目 Trainer；更广泛的实验跟踪和社区回调仍由外部生态提供。 |
| **实验与评估：本地实验与制品** | run/artifact、SQLite/CAS、遥测、环境身份和离线工作流 | 已具备 | 生态具备 | 面向可审计本地运行；不替代远程 tracking、团队协作或多租户平台。 |
| **实验与评估：评估与数据质量** | evaluation suite、数据质量检查、结果 manifest 和 CLI JSON 契约 | 已具备 | 生态具备 | 已有本地工程链路；具体任务指标和数据集适配仍按项目范围扩展。 |
| **交付工具：模型 Bundle 与互操作** | 有界 model bundle、interop backend、权重导入和安装后消费 | 部分具备 | 生态具备 | 支持声明的 bundle/后端边界；不兼容任意 PyTorch 导出物或任意 pickle。 |
| **交付工具：C++ CLI 与交付工具链** | liuding-forge CLI、迁移、评估、检查、打包和安装消费者工具 | 已具备 | 部分具备 | 这是面向原生 C++ 和离线交付的补充能力；PyTorch 主要以 Python 工具链为中心。 |
| **交付工具：性能观测与资源基线** | BuildFingerprint、设备/工作负载身份、数据管线资源观测和基线协议 | 已具备 | 部分具备 | 性能结果必须绑定精确环境，不能把不同硬件、编译器或依赖组合的数字直接比较。 |
| **交付工具：质量、恢复与发布门槛** | 当前训练生命周期的质量/恢复验证、coverage、sanitizer、fuzz、资源矩阵和发布回退 | 部分具备（持续完善） | 生态具备 | 基础质量与恢复能力已有验证；完整覆盖率、依赖安全、发布回退和签名交付仍在完善。 |
| **扩展能力：图像输入与标注工作流** | data_image、mask 语义、联合几何、批量 transforms、renderer/output 和 benchmark | 部分具备（开发中） | 生态具备 | 首个 data_image 切片已完成；完整图像工作流仍在开发。 |
| **扩展能力：量化** | storage quantization 与 FP32 fallback 的独立预览边界 | 开发中（预览） | 已具备 | 当前不是 compute quantization，也不承诺量化感知训练（QAT）。 |
| **扩展能力：导出、Serving 与部署** | 有界 model bundle、互操作后端和本地 serving preview | 开发中（预览） | 生态具备 | 不宣称完整部署平台；TorchScript、torch.export、TorchServe 等能力不等价映射。 |
| **当前范围外：分布式训练** | 多进程、多后端和跨节点训练控制面 | 暂不具备 | 已具备 | distributed 不属于当前 v1.0 代码仓库交付范围。 |
| **当前范围外：远程平台与控制面** | Web、团队服务、RBAC、多租户和远程运行控制 | 暂不具备 | 不适用 | 这些属于产品平台层，不作为当前 Liuding Forge 训练库能力承诺。 |

核心训练、数据、检查点、实验制品和 C++ 工具链已经具备声明范围内的实现；模型生态、完整图像工作流、量化、Serving 以及发布质量能力仍在持续开发；分布式训练和远程平台不属于当前版本范围。Liuding Forge 补充的是 LibTorch 在工程化训练和交付方面的能力，不重新实现 PyTorch，也不与其 Python 研究生态或模型生态做等价承诺。

当前代码主线为 main，远程仓库为 [github.com/guojin-yan/liuding-forge](https://github.com/guojin-yan/liuding-forge)。后续提交和推送统一进入 main。

## 环境要求

- C++20。
- CMake 3.18 或更高版本。
- Python 3，用于配置阶段校验和部分工具。
- 精确的 LibTorch、编译器、操作系统和 CUDA 组合，以 [compatibility.json](compatibility.json) 为准。
- CUDA 还需要匹配的 GPU、驱动、toolkit 和 LibTorch 构建信息。

未列入兼容矩阵的组合可以用于开发排查，但不得写成正式支持。Windows 官方 LibTorch 使用 Release CRT，构建时保持 `CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL` 和 `_ITERATOR_DEBUG_LEVEL=0`。

## 快速开始

### 获取源码

~~~bash
git clone https://github.com/guojin-yan/liuding-forge.git
cd liuding-forge
~~~

### 配置、构建和安装

~~~bash
cmake -S . -B out/build/release -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build out/build/release --config Release --parallel 2
cmake --install out/build/release --config Release --prefix out/install/release
~~~

### 使用已安装的 SDK

~~~cmake
find_package(jyppx CONFIG REQUIRED COMPONENTS sdk)
target_link_libraries(my_app PRIVATE jyppx::liuding::sdk)
~~~

只需要基础能力、错误、RNG 和构建指纹时使用 `jyppx::liuding::foundation`；只需要 checkpoint artifact manifest codec 时使用 `jyppx::liuding::interop`。安装后的公开 CLI 名称为 `liuding-forge`，旧的 `jyppx` CLI 名称不会安装。

### 运行测试和 benchmark

~~~bash
cmake -S . -B out/build/release -DJYPPX_LIUDING_BUILD_TESTS=ON -DJYPPX_LIUDING_BUILD_BENCHMARKS=ON -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build out/build/release --config Release --parallel 2
ctest --test-dir out/build/release -C Release --output-on-failure
~~~

benchmark 操作说明见 [docs/benchmarks/README.md](docs/benchmarks/README.md)，资源基线协议见 [scripts/benchmarks/RESOURCE_BASELINE.md](scripts/benchmarks/RESOURCE_BASELINE.md)。历史结果、审计报告和原始验收证据位于代码仓库外的 v1.0-plan/06_records/。

## 文档与仓库边界

- [文档索引](docs/index.md)
- [基础训练教程](docs/tutorials/01_basic_training.md)
- [数据加载教程](docs/tutorials/02_data_loading.md)
- [优化器与调度器教程](docs/tutorials/03_optimizers.md)
- [回调与检查点教程](docs/tutorials/04_callbacks.md)
- [混合精度教程](docs/tutorials/05_mixed_precision.md)
- [数据 API](docs/api/data.md)、[优化器 API](docs/api/optim.md)、[训练 API](docs/api/training.md)
- [CLI 指南](docs/user-guide/cli.md)、[部署指南](docs/user-guide/deployment.md)、[性能调优](docs/user-guide/performance_tuning.md)

仓库只提交源码、测试、构建/CI 配置、脚本、随库交付的操作说明和必要机器可读配置。开发交付报告、依赖/许可证审计、历史 benchmark 结果、验证日志和原始验收证据统一归档到外部 v1.0-plan/06_records/；构建、运行日志和临时产物归被忽略的 out/。

## 许可与贡献

本项目主体采用 [Apache License 2.0](LICENSE)。第三方依赖保留各自许可，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。贡献前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)，并为代码变更补充必要测试和面向使用者的文档。
