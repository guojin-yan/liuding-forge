// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/core/runtime.hpp>

#include <torch/version.h>

#include <ATen/Parallel.h>
#include <mutex>

namespace jyppx::liuding {

void initialize_runtime() {
#if defined(_WIN32) && AT_PARALLEL_OPENMP && TORCH_VERSION_MAJOR == 2 && \
    TORCH_VERSION_MINOR == 14 && TORCH_VERSION_PATCH == 1
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
