// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "mnist_download.hpp"

#include <jyppx/liuding/foundation/error.hpp>

namespace jyppx::liuding::datasets {
std::filesystem::path download_prepare_mnist(const std::filesystem::path&) {
  throw NotImplemented(
      "Native MNIST HTTPS download is disabled; enable "
      "JYPPX_LIUDING_DOWNLOAD_MNIST with its explicit system TLS SDK, "
      "or use offline --source-dir");
}
}  // namespace jyppx::liuding::datasets
