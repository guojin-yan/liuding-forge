# Deployment Guide

JYPPX is distributed as a CMake package backed by LibTorch:

```cmake
find_package(Torch REQUIRED)
find_package(jyppx_training_framework CONFIG REQUIRED)
add_executable(train train.cpp)
target_link_libraries(train PRIVATE
    jyppx_training_framework::jyppx_training_framework)
```

Public headers remain under `include/jyppx` and the namespace is `jyppx`. On
Windows, ship the matching LibTorch DLLs and MSVC runtime. On Linux, put the
LibTorch `lib` directory in the loader path with an rpath or
`LD_LIBRARY_PATH`. On macOS, use a package matching the target architecture
and configure its library search path before signing.

Build Release for production and keep compiler, C++20 standard library,
LibTorch, and CUDA runtime versions consistent. Pin these versions in the
application build to avoid ABI changes after a system upgrade. Verify a small
CPU job and checkpoint save/load cycle in the target image before shipping a
GPU image.

Production services should validate input shapes, cap DataLoader workers,
handle worker exceptions at the job boundary, and store checkpoints on
durable storage with framework, LibTorch, device, and optimizer metadata. The
distributed API currently provides the single-process reference path; Gloo
cross-process development is deferred until a package with its development
headers is available.
