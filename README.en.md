# Liuding Forge

[中文 README](README.md)

Liuding Forge is a C++20 deep-learning training framework built on LibTorch for engineering teams that need native C++ training, durable lifecycle state, local experimentation, and delivery tooling. JYPPX is the product-group brand, while JYPPX Training Framework is the internal project alias. The public product name, internal alias, and technical identifiers have separate responsibilities.

> Status: the project is under active v1.0 development. Existing features and validation results apply only to their declared code scope and environment; they do not claim GA release readiness or full parity with PyTorch.

## Product scope

The public product is Liuding Forge. The project focuses on reliable training lifecycle management, data pipelines, optimizers and schedulers, evaluation, checkpoints, experiment artifacts, diagnostics, and delivery tools. Web UI, team services, RBAC, multi-tenancy, and remote control planes are outside this repository.

The public C++ root namespace is `jyppx::liuding`. Public headers use `<jyppx/liuding/...>`, packages are discovered with `find_package(jyppx)`, and exported targets use `jyppx::liuding::<component>`. Logical CMake targets, `JYPPX_*` options, and protocol fields retain their technical identity; generated libraries, the CLI, examples, and archives use the `liuding-forge` product prefix.

JYPPX Training Framework remains the internal project alias for planning, engineering coordination, and product-group identification. It is not the public product name and does not define another C++ namespace.

## What the project adds

PyTorch is strongest in its Python ecosystem and research velocity. LibTorch provides the C++ tensor and neural-network foundation, but engineering teams still need explicit lifecycle state, recoverable training, bounded data concurrency, experiment artifacts, environment identity, and delivery boundaries. Liuding Forge adds these engineering contracts on top of LibTorch and makes them reviewable through versioned state and reproducible evidence.

The project focuses on:

- Native C++ integration with a single design for public APIs, CMake packages, installation consumers, ABI, and runtime boundaries.
- Explicit checkpoint domains covering model, optimizer, scheduler, data, strategy, RNG, AMP, callbacks, and task state.
- Bounded behavior for map/iterable data, worker lifecycle, structured batches, capability descriptors, and failure cleanup.
- BuildFingerprint, dependency locks, workload identity, and device identity bound to performance and release evidence.
- Replaceable boundaries for local runs, artifacts, evaluation, telemetry, bundles, and serving.

## Feature modules, comparison, and development progress

The table is organized by engineering foundation, core compute, data pipelines, model system, training system, experiments and evaluation, delivery tools, extensions, and capabilities outside the current scope. It shows both the corresponding PyTorch capability and the current Liuding Forge status. Statuses use Complete, Partial, In progress, or Not available. Every judgment is limited to the current v1.0 code scope and declared environment.

| Feature module | Detailed scope | Liuding Forge status | PyTorch status | Other notes |
| --- | --- | --- | --- | --- |
| **Engineering foundation: build and installation** | Naming boundaries, dependency direction, CMake package, installation consumers, and repository hygiene | Complete | Partial | Liuding Forge uses `jyppx::liuding` as its public C++ identity; logical CMake targets and `JYPPX_*` variables retain their technical identity. |
| **Core compute: tensor and autograd** | Tensor operations, neural-network modules, and autograd foundations | Complete (via LibTorch) | Complete | Liuding Forge uses LibTorch directly and does not reimplement tensor/autograd; the boundary follows the matched LibTorch version. |
| **Core compute: foundation and loss functions** | Foundation contracts, error/RNG/build fingerprint, and common loss vertical slices | Complete | Complete | This is the C++ engineering foundation; it does not claim coverage of every PyTorch loss or extension. |
| **Data pipelines: datasets and loading** | Map/iterable pipelines, workers, samplers, deterministic cursors, and data lifecycle | Complete | Complete | The core C++ pipeline is implemented; multi-process distributed data loading is outside the current scope. |
| **Data pipelines: structured batches and capability descriptors** | Batch schemas, field access, capability declarations, and constrained collate behavior | Complete | Partial | PyTorch can compose Dataset, collate, and third-party types, but does not provide the same project-specific capability contract. |
| **Model system: model adapters and ecosystem** | ModelAdapter, model checkpoint participant, controlled reference models, and model previews | Partial (in progress) | Ecosystem | Liuding Forge provides an engineering adapter boundary and makes no model-zoo parity claim. |
| **Training system: optimizers** | SGD, Adam, AdamW, RMSprop, Adagrad, NAdam, Adadelta, and LAMB adapters with state recovery | Complete | Complete | The main optimizer set is integrated; cross-version migration, sparse parameters, and the wider extension ecosystem remain limited. |
| **Training system: learning-rate schedulers** | Descriptor-first schedulers, composition, lifecycle-event integration, and exact restore | Complete | Complete | The current scheduler set is recoverable under explicit contracts; migration semantics depend on version and configuration. |
| **Training system: Trainer and orchestration** | Trainer, callbacks, metrics, strategies, AMP, multiple optimizers, and checkpoint domains | Complete | Ecosystem | The delivered scope is a C++ training loop; PyTorch core provides primitives while higher-level orchestration commonly comes from ecosystem projects. |
| **Training system: checkpointing and recovery** | Storage-neutral participants, transactional coordination, file storage, CAS/SQLite metadata, and bounded recovery | Complete | Complete | Arbitrary PyTorch pickle compatibility is not supported; recovery is defined by explicit participants and artifact manifests. |
| **Training system: mixed precision** | CPU bfloat16, CUDA float16/bfloat16, GradScaler, and RNG participants | Partial | Complete | Only declared combinations are covered; device, compiler, CUDA, and LibTorch must match the compatibility matrix. |
| **Training system: metrics, callbacks, and events** | Training metrics, lifecycle callbacks, event dispatch, and scheduler event integration | Complete | Ecosystem | The current capability serves the project Trainer; broader tracking and community callback ecosystems remain external. |
| **Experiments and evaluation: local runs and artifacts** | Runs/artifacts, SQLite/CAS, telemetry, environment identity, and offline workflows | Complete | Ecosystem | It targets auditable local runs and does not replace remote tracking, team collaboration, or multi-tenant platforms. |
| **Experiments and evaluation: evaluation and data quality** | Evaluation suites, data-quality checks, result manifests, and CLI JSON contracts | Complete | Ecosystem | The local engineering chain exists; task-specific metrics and dataset adapters continue to expand by scope. |
| **Delivery tools: model bundles and interoperability** | Bounded model bundles, interop backends, weight import, and installed-consumer workflows | Partial | Ecosystem | Only declared bundle and backend boundaries are supported; arbitrary PyTorch exports and pickles are not accepted. |
| **Delivery tools: C++ CLI and toolchain** | liuding-forge CLI, migration, evaluation, inspection, packaging, and installation-consumer tools | Complete | Partial | This is a complementary capability for native C++ and offline delivery; PyTorch is primarily Python-tool driven. |
| **Delivery tools: performance observability and resource baselines** | BuildFingerprint, device/workload identity, data-pipeline resource observation, and baseline protocol | Complete | Partial | Performance results must carry exact environment identity; numbers from different hardware, compilers, or dependencies cannot be compared directly. |
| **Delivery tools: quality, recovery, and release gates** | Training lifecycle quality/recovery validation, coverage, sanitizers, fuzzing, resource matrices, and release rollback | Partial (continuing) | Ecosystem | Core quality and recovery checks are in place; full coverage, dependency security, release rollback, and signing remain under development. |
| **Extensions: image input and annotation workflow** | Data-image input, mask semantics, joint geometry, batch transforms, renderer/output, and benchmarks | Partial (in progress) | Ecosystem | The first data-image slice is complete; the full image workflow remains in development. |
| **Extensions: quantization** | An independent preview boundary for storage quantization and FP32 fallback | In progress (preview) | Complete | This is not compute quantization and makes no quantization-aware training (QAT) claim. |
| **Extensions: export, serving, and deployment** | Bounded model bundles, interop backends, and local serving preview | In progress (preview) | Ecosystem | This is not a complete deployment platform; TorchScript, torch.export, TorchServe, and related capabilities are not mapped one-to-one. |
| **Outside current scope: distributed training** | Multi-process, multi-backend, and cross-node training control | Not available | Complete | Distributed training is outside the current v1.0 repository delivery scope. |
| **Outside current scope: remote platform and control plane** | Web, team services, RBAC, multi-tenancy, and remote run control | Not available | Not applicable | These are platform-layer concerns and are not promised as Liuding Forge training-library capabilities. |

Core training, data, checkpointing, experiment artifacts, and the C++ toolchain are available within their declared scope. The model ecosystem, complete image workflow, quantization, serving, and release-quality capabilities are still developing; distributed training and remote platforms are outside the current version scope. Liuding Forge adds engineering capabilities around LibTorch rather than reimplementing PyTorch, and it makes no equivalence claim for PyTorch's Python research or model ecosystems.

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
