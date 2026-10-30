// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "mnist.hpp"

#include <jyppx/liuding/foundation/error.hpp>
namespace jyppx::liuding::datasets {
std::filesystem::path prepare_mnist(const std::filesystem::path&,
                                    const std::filesystem::path&) {
  throw NotImplemented(
      "MNIST preparation requires JYPPX_LIUDING_PREPARE_MNIST and the explicit "
      "private dataset codec SDK");
}
}  // namespace jyppx::liuding::datasets
