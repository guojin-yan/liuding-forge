// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/core/runtime.hpp>

#include <torch/version.h>

#include <ATen/Parallel.h>
#if defined(USE_ROCM) && defined(_WIN32)
#include <torch/cuda.h>

#include <ATen/hip/HIPContextLight.h>
#endif
#include <mutex>

namespace jyppx::liuding {

void initialize_runtime() {
#if defined(USE_ROCM) && defined(_WIN32)
  // Keep a real torch_hip import: otherwise the Windows linker can discard
  // the backend DLL and its operator/hook registrations before main starts.
  // No device context is requested when the selected SDK has no visible GPU.
  if (torch::cuda::is_available()) {
    (void)at::cuda::getDeviceProperties(0);
  }
#endif
#if defined(_WIN32) && AT_PARALLEL_OPENMP && TORCH_VERSION_MAJOR == 2 && \
    TORCH_VERSION_MINOR == 14 &&                                         \
    (TORCH_VERSION_PATCH == 1 ||                                         \
     (TORCH_VERSION_PATCH == 0 && defined(USE_ROCM)))
  static std::once_flag initialized;
  std::call_once(initialized, [] {
    const auto threads = at::get_num_threads();
    // set_num_threads constructs Caffe2's process-wide pthreadpool only once;
    // later changes update OpenMP/MKL but do not resize that auxiliary pool.
    // Construct it inline before any Windows worker can survive into DllMain.
    at::set_num_threads(1);
    at::set_num_threads(threads);
  });
#endif
}

}  // namespace jyppx::liuding
