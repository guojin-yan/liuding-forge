# JYPPX Training Framework

A C++ deep learning training framework built on LibTorch, providing PyTorch-style APIs for training neural networks in C++.

## Features

- 🚀 **PyTorch-Compatible API**: Familiar interface for PyTorch users
- ⚡ **High Performance**: Native C++ performance with LibTorch backend
- 🔧 **Modular Design**: Clean separation of optimizers, data loading, and training
- 📊 **Rich Training Tools**: Callbacks, metrics, learning rate schedulers
- 🧪 **Well Tested**: Comprehensive unit and integration tests

## Requirements

- **C++20** or later (required by the bundled LibTorch 2.14.x)
- **CMake 3.18+**
- **Python 3** for configure-time architecture and dependency validation
- An exact LibTorch combination listed in [`compatibility.json`](compatibility.json)
  (the current development baseline is 2.14.1 CUDA; 2.5.1 CPU remains a
  compatibility candidate until its CI evidence is green)
- CUDA is supported only for an exact LibTorch, driver, and GPU combination
  with recorded evidence; an unlisted CUDA version is not implicitly supported

Validate the compatibility source before configuring a local build:

```bash
python scripts/validation/compatibility.py --file compatibility.json --validate
```

## Quick Start

### Installation

```bash
git clone https://github.com/guojin-yan/jyppx-training-framework.git
cd jyppx-training-framework

# Configure a persistent build directory (specify LibTorch path)
cmake -S . -B out/build/release -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/libtorch

# Reuse this directory for incremental builds
cmake --build out/build/release --config Release --parallel 2

# Install to a local prefix (optional)
cmake --install out/build/release --config Release --prefix out/install/release
```

On Windows, the example targets copy the LibTorch runtime DLLs next to the
generated executables, so they can be launched directly from the build tree.

LibTorch's official Windows binaries are Release-built with `/MD`. The project
therefore builds every configuration against the release CRT
(`CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL`, `_ITERATOR_DEBUG_LEVEL=0`); do
not override these, or Debug builds will crash inside the torch DLLs. Use the
Debug configuration for source-level debugging of framework code only.

After installation, consumers can load the package and link the exported
target:

```cmake
find_package(jyppx CONFIG REQUIRED COMPONENTS sdk)
target_link_libraries(my_app PRIVATE jyppx::sdk)
```

Use `jyppx::foundation` when a tool only needs the capability, error, RNG, and
build-fingerprint contracts. Use `jyppx::interop` for the bounded checkpoint
artifact manifest codec without linking training orchestration. Public C++
headers, namespaces, and exported CMake targets all use the `jyppx` name;
legacy target aliases are intentionally not provided.

### Basic Usage

The current Trainer uses explicit contracts instead of inferring task
semantics. Adapt the model with `ModelAdapter`, create a map- or iterable-style
`BatchPipeline`, bind a descriptor-first optimizer and `TrainingStrategy`, and
provide mappers plus a `StepFunction`. See
[`examples/basic_trainer.cpp`](examples/basic_trainer.cpp) for a complete
linear-regression program.

`Trainer::prepare()` registers model, every optimizer/scheduler binding, data,
strategy, RNG, callback and task state as one checkpoint domain. `fit()` then
returns a `FitResult` containing committed epoch/step counters and final metric
maps. `save_checkpoint()` and `restore_checkpoint()` use storage-neutral
transaction interfaces; file/container formats belong to interop adapters.

## Documentation

Repository directories have distinct owners:

| Directory | Contents |
| --- | --- |
| `include/`, `src/` | Public library contracts and their implementations. |
| [`tools/`](tools/README.md) | Tools delivered to library users, including the CLI and optional runtime adapters. |
| [`scripts/`](scripts/README.md) | Repository build checks, CI, coverage and performance baseline orchestration. These are not installed as library tools. |
| `tests/` | Unit, integration, installation and release acceptance tests, with their fixtures. |
| `out/` | Ignored incremental builds, reports, logs, downloaded dependencies and virtual environments. |

- [Documentation index](docs/index.md) — tutorials and API reference
- Tutorials: [basic training](docs/tutorials/01_basic_training.md),
  [data loading](docs/tutorials/02_data_loading.md),
  [optimizers and schedulers](docs/tutorials/03_optimizers.md),
  [callbacks and checkpoints](docs/tutorials/04_callbacks.md),
  [mixed precision](docs/tutorials/05_mixed_precision.md)
- API reference: [data](docs/api/data.md), [optim](docs/api/optim.md),
  [training](docs/api/training.md), [utils](docs/api/utils.md)

### User guides

- [Beta scope and migration limits](docs/user-guide/beta_limits.md)
- [Troubleshooting](docs/user-guide/troubleshooting.md)
- [Performance tuning](docs/user-guide/performance_tuning.md)
- [Deployment](docs/user-guide/deployment.md)
- Optional Preview: [models](docs/api/models.md),
  [storage quantization](docs/api/quantization.md), [serving](docs/api/serving.md)

## Known Limitations

- Precision modes include full precision, CPU bfloat16, CUDA float16 and CUDA
  bfloat16. CUDA float16 requires GradScaler; bfloat16 modes use no loss
  scaling. Hardware and dependency qualification follows `compatibility.json`.
- DataPipeline workers use threads in one process. Dataset access and captured
  batch-factory state must be thread-safe; cross-process distributed Gloo
  support is deferred until a LibTorch package with the required development
  headers is available.
- Performance benchmark output is hardware-dependent and should be compared
  on matched environments.

## Project Structure

```
jyppx-training-framework/
├── include/jyppx/        # Public headers
│   ├── core/             # Core utilities
│   ├── optim/            # Optimizers
│   ├── data/             # Data loading
│   ├── training/         # Training abstractions
│   ├── utils/            # Utilities
│   └── nn/               # Neural network utilities
├── src/                  # Implementation
├── tests/                # Tests
├── examples/             # Example programs
└── docs/                 # Documentation
```

## Examples

See the [examples/](examples/) directory for complete examples:

- Basic training
- Custom datasets
- Learning rate scheduling
- Callbacks and metrics
- Mixed precision training

## Building Tests

```bash
cmake -S . -B out/build/release -DJYPPX_BUILD_TESTS=ON
cmake --build out/build/release --config Release --parallel 2
ctest --test-dir out/build/release -C Release --output-on-failure
```

The configuration arguments also support multi-config generators such as
Visual Studio. GoogleTest 1.14.0 is fetched from `third_party/downloads/` when
present (offline builds work), otherwise from GitHub.

## Local Release Baseline

Run the reproducible local baseline from the repository root:

```powershell
pwsh -NoProfile -File scripts/benchmarks/Run-Baseline.ps1
```

The command reuses the persistent `out/build/baseline-release` directory, runs
the full Release test suite and benchmarks, and writes machine-readable timing,
artifact size, throughput, and isolated peak-memory evidence under
`out/baseline/release`. Pass `-FreshBuild` only when an intentionally isolated
clean-build measurement is required; clean and incremental build timings are
not directly comparable.

CUDA Trainer baselines require a Release build linked to CUDA LibTorch and a
visible NVIDIA GPU. After building with `JYPPX_BUILD_BENCHMARKS=ON`, run:

```bash
LD_LIBRARY_PATH=/path/to/libtorch/lib \
python3 scripts/benchmarks/run_cuda_baseline.py \
  --benchmark out/build/cuda/tests/benchmarks/jyppx_bench_cuda_training \
  --output-dir out/baseline/cuda \
  --write-reference out/baseline/cuda/reference.json
```

The runner executes one independent warmup and five independent samples for
both the FP32/single-SGD and FP16/two-SGD/AMP paths. It rejects missing CUDA,
incomplete Trainer or optimizer steps, scaler state outside a checkpoint
boundary, process-to-process identity drift, CV above 10%, and references from
a different build, GPU, driver, power limit, workload, or runtime environment.
Results and logs remain under the ignored `out/` tree.

The same gate records isolated CUDA performance and allocator behavior for all
eight supported optimizers. Each algorithm is compared only with its own
environment-bound reference; the results are not a cross-algorithm ranking:

```bash
LD_LIBRARY_PATH=/path/to/libtorch/lib \
python3 scripts/benchmarks/run_cuda_baseline.py \
  --suite optimizer \
  --benchmark out/build/cuda/tests/benchmarks/jyppx_bench_cuda_optimizer \
  --output-dir out/baseline/cuda-optimizer \
  --write-reference out/baseline/cuda-optimizer/reference.json
```

Every optimizer process uses the same FP32 CUDA tensors and fixed dense
gradients, verifies 200 committed steps, finite outputs and a real parameter
change, and publishes element throughput, mean step time, and CUDA allocator
peaks. The default run collects one warmup plus five independent samples for
SGD, Adam, AdamW, RMSprop, Adagrad, NAdam, Adadelta, and LAMB.

## Debug, Coverage, and Sanitizers

Debug uses the same release CRT and iterator-debug level as LibTorch on Windows;
this is required for safe source-level debugging against the official binaries.
Coverage and sanitizer options require GCC or Clang and fail at configure time
on unsupported compilers instead of silently producing an uninstrumented build.
The Linux coverage lane is `bash scripts/validation/run-coverage.sh`; it requires `gcovr 5.0`
or newer and
`CMAKE_PREFIX_PATH` pointing to a CPU LibTorch installation. `JYPPX_ENABLE_ASAN`,
`JYPPX_ENABLE_UBSAN`, and `JYPPX_ENABLE_TSAN` instrument both framework
libraries and test/benchmark executables. AddressSanitizer and ThreadSanitizer
cannot be combined; coverage and ThreadSanitizer are also rejected. The
coverage runner limits compilation to two jobs by default; set
`JYPPX_BUILD_PARALLEL_LEVEL` to a positive integer when the host has more
memory or needs an even smaller resource footprint. The runner also applies
the checked-in `coverage-policy.json`: every instrumented project file must
belong to exactly one module, and line/function/branch covered counts,
denominators, and percentages may not fall below that module's reviewed
baseline. A deliberate source removal or baseline reduction therefore needs
an explicit policy update in the same review.

## Building Documentation

```bash
cmake -S . -B out/build/release -DJYPPX_BUILD_DOCS=ON
cmake --build out/build/release --config Release --target jyppx_api_docs --parallel 2
```

## License

Licensed exclusively under the [Apache License 2.0](LICENSE). Third-party
dependencies retain their own licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

Built with [LibTorch](https://pytorch.org/cppdocs/) - the C++ frontend for PyTorch.

## Links

- GitHub: https://github.com/guojin-yan/jyppx-training-framework
- Documentation: [docs/](docs/)
- Issue Tracker: https://github.com/guojin-yan/jyppx-training-framework/issues
