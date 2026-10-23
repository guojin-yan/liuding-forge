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

## Feature comparison and development progress

The table combines the PyTorch comparison, the current Liuding Forge implementation, and the corresponding plan stage. “Liuding Forge status” uses Complete, Partial, In progress, or Not available; “PyTorch status” indicates whether the capability exists in PyTorch core or its official ecosystem. Every judgment is limited to the current v1.0 code scope and declared environment. Progress is measured against plan gates and evidence rather than source-line count or raw test count.

| Feature module | Detailed scope | Liuding Forge status | PyTorch status | Other notes |
| --- | --- | --- | --- | --- |
| Plan baseline, build, and installation | Naming boundaries, dependency direction, CMake package, installation consumers, and repository hygiene gates | Complete (M0-M1) | Partial | Liuding Forge uses `jyppx::liuding` as its public C++ identity; logical CMake targets and `JYPPX_*` variables retain their technical identity. |
| Tensor and autograd | Tensor operations, neural-network modules, and autograd foundations | Complete (via LibTorch) | Complete | Liuding Forge uses LibTorch directly and does not reimplement tensor/autograd; the boundary follows the matched LibTorch version. |
| Foundation and loss functions | Foundation contracts, error/RNG/build fingerprint, and four loss vertical slices | Complete (M2) | Complete | This is the C++ engineering foundation; it does not claim coverage of every PyTorch loss or extension. |
| Datasets and data loading | Map/iterable pipelines, workers, samplers, deterministic cursors, and data lifecycle | Complete (M3) | Complete | The core C++ pipeline is implemented; multi-process distributed data loading is deferred. |
| Structured batches and capability descriptors | Batch schemas, field access, capability declarations, and constrained collate behavior | Complete (M3) | Partial | PyTorch can compose Dataset, collate, and third-party types, but does not provide the same project-specific capability contract. |
| Model adapters and model ecosystem | ModelAdapter, model checkpoint participant, controlled reference models, and model previews | Partial (M8 in progress) | Ecosystem | Liuding Forge provides an engineering adapter boundary and makes no model-zoo parity claim. |
| Optimizers | SGD, Adam, AdamW, RMSprop, Adagrad, NAdam, Adadelta, and LAMB adapters with state recovery | Complete | Complete | The main optimizer set is integrated; cross-version migration, sparse parameters, and the wider extension ecosystem remain limited. |
| Learning-rate schedulers | Descriptor-first schedulers, composition, lifecycle-event integration, and exact restore | Complete | Complete | The current scheduler set is recoverable under explicit contracts; migration semantics depend on version and configuration. |
| Trainer and training orchestration | Trainer, callbacks, metrics, strategies, AMP, multiple optimizers, and checkpoint domains | Complete | Ecosystem | The delivered scope is a C++ training loop; PyTorch core provides primitives while higher-level orchestration commonly comes from ecosystem projects. |
| Checkpointing and recovery | Storage-neutral participants, transactional coordination, file storage, CAS/SQLite metadata, and bounded recovery | Complete (M4) | Complete | Arbitrary PyTorch pickle compatibility is not supported; recovery is defined by explicit participants and artifact manifests. |
| Mixed precision | CPU bfloat16, CUDA float16/bfloat16, GradScaler, and RNG participants | Partial | Complete | Only declared combinations are covered; device, compiler, CUDA, and LibTorch must match the compatibility matrix. |
| Metrics, callbacks, and events | Training metrics, lifecycle callbacks, event dispatch, and scheduler event integration | Complete | Ecosystem | The current capability serves the project Trainer; broader tracking and community callback ecosystems remain external. |
| Local experiments and artifacts | Runs/artifacts, SQLite/CAS, telemetry, environment identity, and offline workflows | Complete (M5) | Ecosystem | It targets auditable local runs and does not replace remote tracking, team collaboration, or multi-tenant platforms. |
| Evaluation and data quality | Evaluation suites, data-quality checks, result manifests, and CLI JSON contracts | Complete (M6) | Ecosystem | The local engineering chain exists; task-specific metrics and dataset adapters continue to expand by scope. |
| Model bundles and interoperability | Bounded model bundles, interop backends, weight import, and installed-consumer workflows | Partial (M6) | Ecosystem | Only declared bundle and backend boundaries are supported; arbitrary PyTorch exports and pickles are not accepted. |
| C++ CLI and delivery toolchain | liuding-forge CLI, migration, evaluation, inspection, packaging, and installation-consumer tools | Complete (M6) | Partial | This is a complementary capability for native C++ and offline delivery; PyTorch is primarily Python-tool driven. |
| Performance observability and resource baselines | BuildFingerprint, device/workload identity, data-pipeline resource observation, and baseline protocol | Complete (F3) | Partial | Performance results must carry exact environment identity; numbers from different hardware, compilers, or dependencies cannot be compared directly. |
| Quality, recovery, and release gates | Training lifecycle quality/recovery validation, coverage, sanitizers, fuzzing, resource matrices, and release rollback | Partial (F4 complete; M7 in progress) | Ecosystem | F4 is validated within declared scope; M7 coverage, primary dependency security, Beta/RC rollback, and signing remain open. |
| Image input and annotation workflow | First data-image slice, mask semantics, joint geometry, batch transforms, renderer/output, and benchmarks | Partial (in progress) | Ecosystem | Only the first data-image slice is complete; the full image workflow remains in development. |
| Quantization | An independent preview boundary for storage quantization and FP32 fallback | In progress (M8 preview) | Complete | This is not compute quantization and makes no quantization-aware training (QAT) claim. |
| Export, serving, and deployment | Bounded model bundles, interop backends, and local serving preview | In progress (M8 preview) | Ecosystem | This is not a complete deployment platform; TorchScript, torch.export, TorchServe, and related capabilities are not mapped one-to-one. |
| Distributed training | Multi-process, multi-backend, and cross-node training control | Not available (deferred) | Complete | Distributed training is outside the current v1.0 repository delivery scope. |
| Remote platform and control plane | Web, team services, RBAC, multi-tenancy, and remote run control | Not available (out of scope) | Not applicable | These are platform-layer concerns and are not promised as Liuding Forge training-library capabilities. |

M0-M6, F3, and F4 are complete within their declared scope. The complete image workflow, M7, and M8 remain in development; distributed training and remote platforms are deferred by plan. Liuding Forge adds engineering capabilities around LibTorch rather than reimplementing PyTorch, and it makes no equivalence claim for PyTorch's Python research or model ecosystems.

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
