# Liuding Forge（六丁炼算）

[中文 README](README.md)

Liuding Forge is a C++20 deep learning training framework built on LibTorch, with resumable training, data pipelines, vision models, local experiments, and model delivery tools.

The project has not been released. The source version is `0.3.0`. See [SDK scope and limitations](docs/user-guide/beta_limits.md) for component and model boundaries, and [compatibility.json](compatibility.json) for platform, device, and dependency combinations.

## Current capabilities

| Module | Interfaces and workflows |
| --- | --- |
| [Training](docs/api/training.md) | Trainer, task steps, callbacks, multiple optimizers, accumulation, clipping, and explicit AMP |
| [Data](docs/api/data.md) | Map/iterable pipelines, sampling, thread workers, structured batches, and cursor recovery |
| [Image data](docs/api/data_image.md) | Decoding, unified annotations, joint geometry, batching, device transfer, and bounded image output |
| [Optimizers and scheduling](docs/api/optim.md) | SGD, Adam, AdamW, RMSprop, Adagrad, NAdam, Adadelta, LAMB, and resumable schedulers |
| [Checkpoints](docs/api/durable_checkpoint.md) | Coordinated model, optimizer, scheduler, RNG, data, GradScaler, and callback state |
| [Models](docs/api/models.md) | Classification backbones, ViT/Swin, DINOv2/v3, feature extraction, classifier replacement, and explicit weight imports |
| [Vision suite](docs/api/vision/index.md) | Original YOLOv5 P5/P6, YOLOv8/11/12/26, Faster/Fast R-CNN, FCOS, RetinaNet, SSD, RT-DETR, and other registered models; data conversion, augmentation, training, validation, prediction, and tuning |
| [Extended tasks](docs/api/vision/extensions.md) | Multilabel classification, semantic segmentation, depth, SimMIM, distillation, open vocabulary, ReID, and sequence tracking |
| [Local experiments](docs/api/local_run_storage.md) | SQLite/CAS, runs and artifacts, native process execution, bounded search, TensorBoard, and OTLP telemetry |
| [Evaluation](docs/api/evaluation.md) | Classification, box AP, segmentation metrics, source summaries, quality rules, and JSON/HTML reports |
| [Model delivery](docs/api/onnx.md) | C++ ONNX export, ONNX Runtime, model bundles, signing, and local serving |

The registry in the selected build determines available model combinations. The repository includes no pretrained weights. Architecture implementation, weight licensing, task quality, and device support have separate boundaries. See [recipes](recipes/README.md) for configuration templates and [examples](examples/README.md) for complete C++ programs.

## Integration

Public headers use `<jyppx/liuding/...>` and namespace `jyppx::liuding`. The CMake package is `jyppx`, component targets are `jyppx::liuding::<component>`, and build options use `JYPPX_LIUDING_*`. The `sdk` target aggregates eight base components; Models, Vision, deployment, and telemetry components are optional. Product commands are `liuding-forge` and the separate `liuding-forge-vision`.

## Requirements

- C++20.
- CMake 3.18 or newer.
- Python 3 only for build/development validation and independent references; installed C++ libraries, the CLI, and native examples do not require Python.
- An exact LibTorch, compiler, operating-system, and CUDA combination listed in [compatibility.json](compatibility.json).
- A matching GPU, driver, toolkit, and LibTorch build for CUDA.

An unlisted combination may be useful for development investigation but must not be described as officially supported. Official Windows LibTorch binaries use the Release CRT; keep `CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL` and `_ITERATOR_DEBUG_LEVEL=0`.

## Quick start

### Get the source

~~~bash
git clone https://github.com/guojin-yan/liuding-forge.git
cd liuding-forge
~~~

### Configure, build, and install

~~~bash
cmake -S . -B out/build/release -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build out/build/release --config Release --parallel 2
cmake --install out/build/release --config Release --prefix out/install/release
~~~

### Consume the installed SDK

~~~cmake
find_package(jyppx CONFIG REQUIRED COMPONENTS sdk)
target_link_libraries(my_app PRIVATE jyppx::liuding::sdk)
~~~

Initialize the runtime on the main thread before configuring or using LibTorch or starting workers; the CLI already calls this entry point. It mitigates the exit hang in the current Windows LibTorch 2.14.1 OpenMP profile. See [troubleshooting](docs/user-guide/troubleshooting.md#windows-训练完成后进程不退出) for initialization order and scope.

~~~cpp
#include <jyppx/liuding/core/runtime.hpp>

int main() {
  jyppx::liuding::initialize_runtime();
  // Create the model and data pipeline, then run training.
}
~~~

Use `jyppx::liuding::foundation` for capability, error, RNG, and build-fingerprint contracts, and `jyppx::liuding::interop` for the bounded checkpoint artifact manifest codec. The installed public CLI is `liuding-forge`; the former `jyppx` CLI name is not installed.

### Run tests and benchmarks

~~~bash
cmake -S . -B out/build/release -DJYPPX_LIUDING_BUILD_TESTS=ON -DJYPPX_LIUDING_BUILD_BENCHMARKS=ON -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build out/build/release --config Release --parallel 2
ctest --test-dir out/build/release -C Release --output-on-failure
~~~

See [docs/benchmarks/README.md](docs/benchmarks/README.md) for benchmark operation and [scripts/benchmarks/RESOURCE_BASELINE.md](scripts/benchmarks/RESOURCE_BASELINE.md) for the resource protocol. Bind performance results to the actual build, device, and workload. Save generated output under ignored `out/` or a working directory you choose.

## Documentation and repository boundary

Start with the [documentation index](docs/index.md), [tutorials](docs/tutorials/), [image classification and process restart](docs/tutorials/image_classification_recipe.md), [paired-input classification/regression and recovery](docs/tutorials/multi_io_recipe.md), [API reference](docs/api/), [CLI guide](docs/user-guide/cli.md), and [benchmark instructions](docs/benchmarks/README.md).

See [training and search recipes](recipes/README.md) for native CLI configuration templates and [development scripts](scripts/README.md) for maintenance, validation, and release tools.

The repository contains source, tests, build configuration, reusable maintenance tools, and usage instructions. Keep builds, data, logs, and model output under ignored `out/` or a working directory you choose. See [repository hygiene](docs/development/repository_hygiene.md) for directory responsibilities.

## License and contribution

Contributions owned by project contributors and within their authority to license are distributed under the [Apache License 2.0](LICENSE). Third-party dependencies, upstream materials and model weights retain their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [model provenance and licensing](third_party/licenses/MODEL_PROVENANCE.md). The project license does not replace DINOv3's separate terms or resolve the pending C++ source provenance review. Read [CONTRIBUTING.md](CONTRIBUTING.md) before contributing and include the tests and user-facing documentation required for code changes.
