// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/core/runtime.hpp>

#include <gtest/gtest.h>

namespace {
class RuntimeEnvironment final : public testing::Environment {
 public:
  void SetUp() override { jyppx::liuding::initialize_runtime(); }
};
[[maybe_unused]] auto* runtime_environment =
    testing::AddGlobalTestEnvironment(new RuntimeEnvironment);
}  // namespace
