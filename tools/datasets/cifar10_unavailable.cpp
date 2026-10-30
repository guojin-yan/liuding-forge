// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "cifar10.hpp"

#include <jyppx/liuding/foundation/error.hpp>

namespace jyppx::liuding::datasets {
std::filesystem::path prepare_cifar10(const std::filesystem::path&,
                                      const std::filesystem::path&) {
  throw NotImplemented(
      "CIFAR-10 preparation requires JYPPX_LIUDING_PREPARE_CIFAR10 and an "
      "explicit independently qualified dataset codec SDK");
}
}  // namespace jyppx::liuding::datasets
