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
- An exact LibTorch combination listed in [`compatibility.json`](compatibility.json)
  (the current development baseline is 2.14.1 CUDA; 2.5.1 CPU remains a
  compatibility candidate until its CI evidence is green)
- CUDA is supported only for an exact LibTorch, driver, and GPU combination
  with recorded evidence; an unlisted CUDA version is not implicitly supported

Validate the compatibility source before configuring a local build:

```bash
python tools/compatibility.py --file compatibility.json --validate
```

## Quick Start

### Installation

```bash
git clone https://github.com/guojin-yan/jyppx-training-framework.git
cd jyppx-training-framework

# Create build directory
mkdir build && cd build

# Configure (specify LibTorch path)
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch

# Build
cmake --build . --parallel

# Install (optional)
sudo cmake --install .
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
find_package(jyppx_training_framework CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE jyppx::sdk)
```

Use `jyppx::foundation` when a tool only needs the error, RNG, and build
fingerprint contracts. Public C++ headers, namespaces, and exported CMake
targets all use the `jyppx` name; legacy target aliases are intentionally not
provided.

### Basic Usage

```cpp
#include <jyppx/jyppx.hpp>

int main() {
    // Create dataset
    auto data = torch::randn({100, 10});
    auto targets = torch::rand({100, 2});
    auto dataset = std::make_shared<jyppx::data::TensorDataset>(data, targets);
    
    // Create dataloader
    jyppx::data::DataLoader loader(dataset, {
        .batch_size = 32,
        .shuffle = true
    });
    
    // Create model
    auto model = std::make_shared<torch::nn::LinearImpl>(10, 2);
    
    // Create optimizer
    auto optimizer = std::make_shared<jyppx::optim::Adam>(
        model->parameters(),
        jyppx::optim::AdamOptions{.lr = 0.001}
    );
    
    // Create trainer
    jyppx::training::Trainer trainer(model, optimizer, {
        .max_epochs = 10,
        .device = torch::kCPU
    });
    
    // Train
    trainer.fit(loader);
    
    return 0;
}
```

`Trainer::save_checkpoint()` stores model weights, optimizer state, scheduler
and AMP scaler state, recent losses, and loop counters. Construct a compatible
trainer and call `load_checkpoint()` before `fit()` to continue from the saved
epoch.

## Documentation

- [Documentation index](docs/index.md) — tutorials and API reference
- Tutorials: [basic training](docs/tutorials/01_basic_training.md),
  [data loading](docs/tutorials/02_data_loading.md),
  [optimizers and schedulers](docs/tutorials/03_optimizers.md),
  [callbacks and checkpoints](docs/tutorials/04_callbacks.md),
  [mixed precision](docs/tutorials/05_mixed_precision.md)
- API reference: [data](docs/api/data.md), [optim](docs/api/optim.md),
  [training](docs/api/training.md), [utils](docs/api/utils.md)

### User guides

- [Troubleshooting](docs/user-guide/troubleshooting.md)
- [Performance tuning](docs/user-guide/performance_tuning.md)
- [Deployment](docs/user-guide/deployment.md)

## Known Limitations

- AMP currently requires CUDA; CPU training uses full precision.
- DataLoader workers use threads in one process. Dataset access must be
  thread-safe, and cross-process distributed Gloo support is deferred until a
  LibTorch package with the required development headers is available.
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

## Documentation

- [Installation Guide](docs/tutorials/installation.md)
- [Quick Start Tutorial](docs/tutorials/quickstart.md)
- [API Documentation](docs/api/)

## Examples

See the [examples/](examples/) directory for complete examples:

- Basic training
- Custom datasets
- Learning rate scheduling
- Callbacks and metrics
- Mixed precision training

## Building Tests

```bash
cmake .. -DJYPPX_BUILD_TESTS=ON
cmake --build .
ctest --output-on-failure
```

On multi-config generators (Visual Studio), pass the configuration to both
steps: `cmake --build . --config Release` and `ctest -C Release`. GoogleTest
1.14.0 is fetched from `third_party/downloads/` when present (offline builds
work), otherwise from GitHub.

## Local Release Baseline

Run the reproducible local baseline from the repository root:

```powershell
pwsh -NoProfile -File tools/Run-Baseline.ps1
```

The command uses a clean `out/build/baseline-release` directory, runs the full
Release test suite and benchmarks, and writes machine-readable timing, artifact
size, throughput, and isolated peak-memory evidence under
`out/baseline/release`. Use `-ReuseBuild` only for diagnosing the baseline
tool itself; reused build time is not comparable to a clean compile baseline.

## Debug, Coverage, and Sanitizers

Debug uses the same release CRT and iterator-debug level as LibTorch on Windows;
this is required for safe source-level debugging against the official binaries.
Coverage and sanitizer options require GCC or Clang and fail at configure time
on unsupported compilers instead of silently producing an uninstrumented build.
The Linux coverage lane is `bash tools/run-coverage.sh`; it requires `gcovr` and
`CMAKE_PREFIX_PATH` pointing to a CPU LibTorch installation. `JYPPX_ENABLE_ASAN`,
`JYPPX_ENABLE_UBSAN`, and `JYPPX_ENABLE_TSAN` instrument both framework
libraries and test/benchmark executables. AddressSanitizer and ThreadSanitizer
cannot be combined; coverage and ThreadSanitizer are also rejected. The
coverage runner limits compilation to two jobs by default; set
`JYPPX_BUILD_PARALLEL_LEVEL` to a positive integer when the host has more
memory or needs an even smaller resource footprint.

## Building Documentation

```bash
cmake .. -DJYPPX_BUILD_DOCS=ON
cmake --build . --target docs
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
