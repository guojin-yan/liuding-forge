# 六丁炼算（Liuding Forge）

[English README](README.en.md)

六丁炼算是基于 LibTorch 的 C++20 深度学习训练框架，提供可恢复训练、数据管线、视觉模型、本地实验和模型交付工具。

项目尚未发布，源码版本为 `0.3.0`。组件与模型的使用边界见 [SDK 使用范围与限制](docs/user-guide/beta_limits.md)，平台、设备和依赖组合见 [compatibility.json](compatibility.json)。

## 当前功能

| 模块 | 提供的接口与工作流 |
| --- | --- |
| [训练](docs/api/training.md) | Trainer、任务 step、回调、多优化器、梯度累积、裁剪与显式 AMP |
| [数据](docs/api/data.md) | map/iterable 管线、采样、线程 worker、结构化批次与游标恢复 |
| [图像数据](docs/api/data_image.md) | 解码、统一标注、联合几何变换、组批、设备传输与有界图像输出 |
| [优化器与调度](docs/api/optim.md) | SGD、Adam、AdamW、RMSprop、Adagrad、NAdam、Adadelta、LAMB 及可恢复调度器 |
| [检查点](docs/api/durable_checkpoint.md) | 模型、优化器、调度器、RNG、数据、GradScaler 与回调的协调保存及恢复 |
| [模型](docs/api/models.md) | ResNet 等分类骨干、ViT/Swin、DINOv2/v3、特征提取、分类头替换与显式权重导入 |
| [视觉训练套件](docs/api/vision/index.md) | 原始 YOLOv5 P5/P6、YOLOv8/11/12/26、Faster/Fast R-CNN、FCOS、RetinaNet、SSD、RT-DETR 等模型；数据转换、增强、训练、验证、预测和调参 |
| [扩展任务](docs/api/vision/extensions.md) | 多标签、语义分割、深度、SimMIM、蒸馏、开放词汇、ReID 与序列跟踪 |
| [本地实验](docs/api/local_run_storage.md) | SQLite/CAS、运行与制品、原生进程执行、有界搜索、TensorBoard 与 OTLP 遥测 |
| [评估](docs/api/evaluation.md) | 分类、框 AP、分割指标、按来源汇总、质量规则与 JSON/HTML 报告 |
| [模型交付](docs/api/onnx.md) | C++ ONNX 导出、ONNX Runtime、模型 bundle、签名与本地 Serving |

模型选择以当前构建的注册表为准。仓库不附带预训练权重；架构实现、权重许可、任务精度和设备支持分别判断。训练与搜索模板位于 [recipes](recipes/README.md)，完整 C++ 程序见[示例目录](examples/README.md)。

## 接入方式

公共头文件使用 `<jyppx/liuding/...>`，命名空间为 `jyppx::liuding`；CMake 包为 `jyppx`，组件目标为 `jyppx::liuding::<component>`，构建选项使用 `JYPPX_LIUDING_*` 前缀。`sdk` 聚合八个基础组件，Models、Vision、部署和遥测组件按需开启。产品命令为 `liuding-forge` 与独立的 `liuding-forge-vision`。

## 环境要求

- C++20。
- CMake 3.18 或更高版本。
- Python 3 仅用于构建/开发校验与独立参考；安装后的 C++ 库、CLI 和原生示例不需要 Python。
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

在 `main` 主线程中，首次配置或使用 LibTorch、启动 worker 之前调用运行时初始化；CLI 已自动接入。该入口规避当前 Windows LibTorch 2.14.1 OpenMP 配置的退出挂起，调用顺序和适用范围见[故障排查](docs/user-guide/troubleshooting.md#windows-训练完成后进程不退出)。

~~~cpp
#include <jyppx/liuding/core/runtime.hpp>

int main() {
  jyppx::liuding::initialize_runtime();
  // 创建模型、数据管线并执行训练。
}
~~~

只需要基础能力、错误、RNG 和构建指纹时使用 `jyppx::liuding::foundation`；只需要 checkpoint artifact manifest codec 时使用 `jyppx::liuding::interop`。安装后的公开 CLI 名称为 `liuding-forge`，旧的 `jyppx` CLI 名称不会安装。

### 运行测试和 benchmark

~~~bash
cmake -S . -B out/build/release -DJYPPX_LIUDING_BUILD_TESTS=ON -DJYPPX_LIUDING_BUILD_BENCHMARKS=ON -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build out/build/release --config Release --parallel 2
ctest --test-dir out/build/release -C Release --output-on-failure
~~~

benchmark 操作说明见 [docs/benchmarks/README.md](docs/benchmarks/README.md)，资源基线协议见 [scripts/benchmarks/RESOURCE_BASELINE.md](scripts/benchmarks/RESOURCE_BASELINE.md)。性能结果应绑定实际构建、设备与工作负载，运行输出保存在被忽略的 `out/` 或使用者指定的工作目录。

## 文档与仓库边界

- [文档索引](docs/index.md)、[训练与搜索配方](recipes/README.md)、[开发维护脚本](scripts/README.md)
- [基础训练教程](docs/tutorials/01_basic_training.md)
- [数据加载教程](docs/tutorials/02_data_loading.md)
- [优化器与调度器教程](docs/tutorials/03_optimizers.md)
- [回调与检查点教程](docs/tutorials/04_callbacks.md)
- [混合精度教程](docs/tutorials/05_mixed_precision.md)
- [图像分类与跨进程续训](docs/tutorials/image_classification_recipe.md)
- [双输入分类、回归与跨进程续训](docs/tutorials/multi_io_recipe.md)
- [数据 API](docs/api/data.md)、[优化器 API](docs/api/optim.md)、[训练 API](docs/api/training.md)
- [CLI 指南](docs/user-guide/cli.md)、[部署指南](docs/user-guide/deployment.md)、[性能调优](docs/user-guide/performance_tuning.md)

仓库保存源码、测试、构建配置、可复用维护工具与使用说明。构建、数据、日志和模型输出使用被忽略的 `out/` 或使用者指定的工作目录；目录职责见[仓库卫生](docs/development/repository_hygiene.md)。

## 许可与贡献

本项目贡献者拥有并有权授权的贡献采用 [Apache License 2.0](LICENSE)。第三方依赖、上游材料和模型权重保留各自许可，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 与 [模型来源与许可](third_party/licenses/MODEL_PROVENANCE.md)；DINOv3 的独立条款及尚未完成的源码来源核查不由项目许可替代。贡献前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)，并为代码变更补充必要测试和面向使用者的文档。
