# Deployment Guide

六丁炼算（Liuding Forge）is distributed as a CMake package backed by LibTorch.
The installed package keeps the historical `jyppx` name for compatibility:

```cmake
find_package(jyppx CONFIG REQUIRED COMPONENTS training)
add_executable(train train.cpp)
target_link_libraries(train PRIVATE jyppx::training)
```

Public headers remain under `include/jyppx` and the namespace is `jyppx`. On
Windows, ship the matching LibTorch DLLs and MSVC runtime. On Linux, put the
LibTorch `lib` directory in the loader path with an rpath or
`LD_LIBRARY_PATH`. Qualified combinations are listed in
[`compatibility.json`](../../compatibility.json); macOS remains unsupported
pending qualification on real hardware.

Build Release for production and keep compiler, C++20 standard library,
LibTorch, and CUDA runtime versions consistent. Pin these versions in the
application build to avoid ABI changes after a system upgrade. Verify a small
CPU job and checkpoint save/load cycle in the target image before shipping a
GPU image.

Production services should validate input schemas, cap DataPipeline workers
and prefetch capacity, handle structured pipeline failures at the job
boundary, and store checkpoints on durable storage with framework, LibTorch,
device, data-lineage, and optimizer metadata. `SingleDeviceStrategy` supports
one process on CPU or one GPU. Multi-process distributed strategies remain
deferred until a LibTorch package with the required development headers is
available.

The optional [Serving Preview](../api/serving.md) supplies verified bundle
loading, request limits and dynamic batching. Its
[JSONL host example](../../examples/serving/README.md) demonstrates local
integration. Applications own HTTP/TLS endpoints, authentication and service
control.
