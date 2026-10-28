# Liuding Forge

[中文 README](README.md)

Liuding Forge is a C++20 deep-learning training framework built on LibTorch for engineering teams that need native C++ training, durable lifecycle state, local experimentation, and delivery tooling.

> Status: the project is under active v1.0 development. Existing features and validation results apply only to their declared code scope and environment; they do not claim GA release readiness or full parity with PyTorch.

## Product scope

Liuding Forge targets engineering teams that build, recover, evaluate, and deliver training workloads in native C++. The project focuses on reliable training lifecycle management, data pipelines, optimizers and schedulers, evaluation, checkpoints, experiment artifacts, diagnostics, and delivery tools. Web UI, team services, RBAC, multi-tenancy, and remote control planes are outside this repository.

When using the C++ API, include public headers from `<jyppx/liuding/...>` and call the interfaces in the `jyppx::liuding` namespace. For CMake integration, find the installed package with `find_package(jyppx)` and link targets such as `jyppx::liuding::<component>`; build options continue to use the `JYPPX_*` prefix. The libraries, CLI, examples, and archive files that users run or distribute use the `liuding-forge` product prefix.

## What the project adds

PyTorch is strongest in its Python ecosystem and research velocity. LibTorch provides the C++ tensor and neural-network foundation, but engineering teams still need explicit lifecycle state, recoverable training, bounded data concurrency, experiment artifacts, environment identity, and delivery boundaries. Liuding Forge adds these engineering contracts on top of LibTorch and makes them reviewable through versioned state and reproducible evidence.

The project focuses on:

- Native C++ integration with a single design for public APIs, CMake packages, installation consumers, ABI, and runtime boundaries.
- Explicit checkpoint domains covering model, optimizer, scheduler, data, strategy, RNG, AMP, callbacks, and task state.
- Bounded behavior for map/iterable data, worker lifecycle, structured batches, capability descriptors, and failure cleanup.
- BuildFingerprint, dependency locks, workload identity, and device identity bound to performance and release evidence.
- Replaceable boundaries for local runs, artifacts, evaluation, telemetry, bundles, and serving.

## Feature Module Development Progress

The table is organized by engineering foundation, core compute, data pipelines, model system, training system, experiments and evaluation, delivery tools, extensions, and capabilities outside the current scope. It shows only Liuding Forge capability status and development progress; the notes identify the corresponding PyTorch modules or commonly used capabilities. Every judgment is limited to the current v1.0 code scope and declared environment.

| Feature module | Detailed scope | Liuding Forge status | Development progress | Other notes |
| --- | --- | --- | --- | --- |
| **Engineering foundation: build and installation** | Naming boundaries, dependency direction, CMake package, installation consumers, and repository hygiene | Complete | Complete | PyTorch counterpart: the LibTorch C++ frontend, CMake package, and C++ extension build path; the public API uses `jyppx::liuding`. |
| **Core compute: tensor and autograd** | Tensor operations, neural-network modules, and autograd foundations | Complete (via LibTorch) | Ongoing validation | PyTorch counterpart: `torch.Tensor`, `torch.autograd`, `torch.nn`, and the LibTorch C++ frontend; Liuding Forge does not reimplement tensor/autograd. |
| **Core compute: foundation and loss functions** | Foundation contracts, error/RNG/build fingerprint, and common loss vertical slices | Complete | Complete | PyTorch counterpart: `torch.nn` losses, random utilities, and device foundations; coverage follows this project's C++ contracts. |
| **Data pipelines: datasets and loading** | Map/iterable pipelines, workers, samplers, deterministic cursors, and data lifecycle | Complete | Complete | PyTorch counterpart: `torch.utils.data.Dataset`, `DataLoader`, `Sampler`, and worker mechanisms; multi-process distributed loading is outside the current scope. |
| **Data pipelines: structured batches and capability descriptors** | Batch schemas, field access, capability declarations, and constrained collate behavior | Complete | Complete | PyTorch counterpart: `collate_fn`, batch composition, and data-field handling; this project adds an explicit capability contract. |
| **Model system: model adapters and ecosystem** | ModelAdapter, model checkpoint participant, ResNet/ResNeXt/DenseNet/MobileNet/EfficientNet, `tf_efficientnetv2_s`, ConvNeXt, ViT, Swin-T, four DINOv2 capacities, and eleven DINOv3 factories | Partial | All 33 vision variants provide unified feature extraction, classifier queries/replacement, and headless embeddings. Selected variants have fixed random-state, feature-differential, and training checks. Preprocessing, pretrained weights, and full device/task matrices remain open | PyTorch counterpart: `torch.nn.Module`, torchvision, timm, and the DINO ecosystem. Architecture checks do not establish pretrained-model availability; see the [model API (Chinese)](docs/api/models.md). |
| **Training system: optimizers** | SGD, Adam, AdamW, RMSprop, Adagrad, NAdam, Adadelta, and LAMB adapters with state recovery | Complete | Complete | PyTorch counterpart: `torch.optim` and optimizer state; cross-version migration, sparse parameters, and the wider extension ecosystem remain limited. |
| **Training system: learning-rate schedulers** | Descriptor-first schedulers, composition, lifecycle-event integration, and exact restore | Complete | Complete | PyTorch counterpart: `torch.optim.lr_scheduler`; this project adds explicit descriptors, event binding, and recoverable state. |
| **Training system: Trainer and orchestration** | Trainer, callbacks, metrics, strategies, AMP, multiple optimizers, and checkpoint domains | Complete | Complete | PyTorch counterpart: training loops, model/optimizer composition, and ecosystem Trainers; this project provides a C++ training loop. |
| **Training system: checkpointing and recovery** | Storage-neutral participants, transactional coordination, file storage, CAS/SQLite metadata, and bounded recovery | Complete | Complete | PyTorch counterpart: `state_dict`, `torch.save`, and `torch.load`; arbitrary PyTorch pickle is not supported, and recovery is defined by explicit participants and manifests. |
| **Training system: mixed precision** | CPU bfloat16, CUDA float16/bfloat16, GradScaler, and RNG participants | Partial | Ongoing improvement | PyTorch counterpart: `torch.amp`, `torch.cuda.amp`, and `GradScaler`; only declared combinations are covered and device/LibTorch versions must match. |
| **Training system: metrics, callbacks, and events** | Training metrics, lifecycle callbacks, event dispatch, and scheduler event integration | Complete | Complete | PyTorch counterpart: training-loop hooks and ecosystem capabilities such as torchmetrics and Lightning; current interfaces serve this project's Trainer. |
| **Training system: basic diagnostics** | Loss NaN/Inf, nonfinite gradients, AMP skips, optional gradient norms and data-wait timing with recoverable sampling | Scoped diagnostics available | Ongoing improvement | PyTorch counterpart: training checks and profiling hooks; extra scans/timing are off by default. Full profiling and diagnostic reports remain open. |
| **Experiments and evaluation: local runs and artifacts** | Runs/artifacts, SQLite/CAS, telemetry, environment identity, and offline workflows | Complete | Complete | PyTorch counterpart: TensorBoard, `torch.profiler`, experiment tracking, and artifact-management ecosystems; this does not replace remote tracking or multi-tenant platforms. |
| **Experiments and evaluation: evaluation and data quality** | Classification, semantic IoU/Dice and axis-aligned box AP inference, class/source slices, recoverable classification evaluation and quality reports | Three scoped evaluators available | In progress | PyTorch counterpart: evaluation loops and torchmetrics; real detection/segmentation tasks and full task qualification remain in development. |
| **Delivery tools: model bundles and interoperability** | Bounded model bundles, interop backends, weight import, and installed-consumer workflows | Partial | Ongoing improvement | PyTorch counterpart: `state_dict`, TorchScript, `torch.export`, and weight-import paths; arbitrary exports and pickle are not accepted. |
| **Delivery tools: C++ CLI and toolchain** | liuding-forge CLI, migration, evaluation, inspection, packaging, and installation-consumer tools | Complete | Complete | PyTorch counterpart: the LibTorch C++ frontend, `torchrun` surroundings, and extension-build tools; this project focuses on native C++ and offline delivery. |
| **Delivery tools: performance observability and resource baselines** | BuildFingerprint, device/workload identity, data-pipeline resource observation, and baseline protocol | Complete | Ongoing improvement | PyTorch counterpart: Profiler, CUDA observation, and benchmark tools; performance results require exact environment identity and cannot be compared across environments directly. |
| **Delivery tools: quality, recovery, and release gates** | Training lifecycle quality/recovery validation, coverage, sanitizers, fuzzing, resource matrices, and release rollback | Partial | Ongoing improvement | PyTorch counterpart: testing, benchmark, and release-validation ecosystems; full coverage, dependency security, rollback, and signing remain under development. |
| **Extensions: image input and annotation workflow** | Image/annotation readers, pixel semantics, coordinate tracing, masks, joint transforms, batching, and rendering/output | Partial | Lossless PNG labels, CPU joint geometry and sample-keyed randomness, stack/pad/ragged collation, CPU/CUDA numeric and pixel-batch geometry, precommit recovery, CPU annotation rendering, and bounded PNG/JPEG encoding with local file output have working slices; CUDA rich annotations, asynchronous output, and end-to-end task integration remain open | PyTorch counterpart: torchvision datasets, transforms, masks/annotations, and visual data pipelines; see the [image data API (Chinese)](docs/api/data_image.md) and [image rendering API (Chinese)](docs/api/vision_render.md). |
| **Extensions: quantization** | An independent preview boundary for storage quantization and FP32 fallback | Partial | In progress (preview) | PyTorch counterpart: `torch.ao.quantization`; this is not compute quantization and makes no quantization-aware training (QAT) claim. |
| **Extensions: export, serving, and deployment** | Bounded model bundles, interop backends, and local serving preview | Partial | In progress (preview) | PyTorch counterpart: TorchScript, `torch.export`, TorchServe, and deployment ecosystems; this is not a complete platform or one-to-one mapping. |
| **Outside current scope: distributed training** | Multi-process, multi-backend, and cross-node training control | Not available | Outside current scope | PyTorch counterpart: `torch.distributed`, DDP, and multi-backend communication; this is outside the current v1.0 repository delivery. |
| **Outside current scope: remote platform and control plane** | Web, team services, RBAC, multi-tenancy, and remote run control | Not available | Outside current scope | PyTorch does not itself provide a complete remote control plane; related capabilities belong to platform and MLOps ecosystems. |

Core training, data, checkpointing, experiment artifacts, and the C++ toolchain are available within their declared scope. The model ecosystem, complete image workflow, quantization, serving, and release-quality capabilities are still developing; distributed training and remote platforms are outside the current version scope. Liuding Forge adds engineering capabilities around LibTorch rather than reimplementing PyTorch.

The code mainline is main and the remote repository is [github.com/guojin-yan/liuding-forge](https://github.com/guojin-yan/liuding-forge). Future commits and pushes target main.

## Requirements

- C++20.
- CMake 3.18 or newer.
- Python 3 for configure-time validation and selected tools.
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

Use `jyppx::liuding::foundation` for capability, error, RNG, and build-fingerprint contracts, and `jyppx::liuding::interop` for the bounded checkpoint artifact manifest codec. The installed public CLI is `liuding-forge`; the former `jyppx` CLI name is not installed.

### Run tests and benchmarks

~~~bash
cmake -S . -B out/build/release -DJYPPX_LIUDING_BUILD_TESTS=ON -DJYPPX_LIUDING_BUILD_BENCHMARKS=ON -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build out/build/release --config Release --parallel 2
ctest --test-dir out/build/release -C Release --output-on-failure
~~~

See [docs/benchmarks/README.md](docs/benchmarks/README.md) for benchmark operation and [scripts/benchmarks/RESOURCE_BASELINE.md](scripts/benchmarks/RESOURCE_BASELINE.md) for the resource protocol. Historical results, audits, and raw acceptance evidence are kept outside the code repository under v1.0-plan/06_records/.

## Documentation and repository boundary

Start with the [documentation index](docs/index.md), [tutorials](docs/tutorials/), [API reference](docs/api/), [CLI guide](docs/user-guide/cli.md), and [benchmark instructions](docs/benchmarks/README.md).

The repository contains source code, tests, build/CI configuration, scripts, shipped operation instructions, and required machine-readable configuration. Delivery reports, dependency and license audits, historical benchmark results, validation logs, and raw acceptance evidence are archived outside the code repository under v1.0-plan/06_records/. Builds, run logs, and temporary products belong under the ignored out/ directory.

## License and contribution

The project is distributed under the [Apache License 2.0](LICENSE). Third-party dependencies retain their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Read [CONTRIBUTING.md](CONTRIBUTING.md) before contributing and include the tests and user-facing documentation required for code changes.
