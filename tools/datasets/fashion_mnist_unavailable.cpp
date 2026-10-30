// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "fashion_mnist.hpp"

#include <jyppx/liuding/foundation/error.hpp>
namespace jyppx::liuding::datasets {
std::filesystem::path prepare_fashion_mnist(const std::filesystem::path&,
                                            const std::filesystem::path&,
                                            std::string_view) {
  throw CompatibilityError(
      "Fashion-MNIST preparation is unavailable; build with "
      "JYPPX_LIUDING_PREPARE_FASHION_MNIST=ON and an explicit pinned static "
      "zlib SDK");
}
}  // namespace jyppx::liuding::datasets
