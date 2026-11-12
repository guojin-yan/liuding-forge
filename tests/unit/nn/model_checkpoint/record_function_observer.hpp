// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_TESTS_NN_MODEL_CHECKPOINT_RECORD_FUNCTION_OBSERVER_HPP
#define JYPPX_LIUDING_TESTS_NN_MODEL_CHECKPOINT_RECORD_FUNCTION_OBSERVER_HPP

#include <ATen/core/Tensor.h>
#include <cstdint>

namespace jyppx::liuding::tests::model_checkpoint {
using CopyObserver = void (*)(const at::Tensor&);
std::uint64_t register_copy_observer(CopyObserver observer);
void unregister_copy_observer(std::uint64_t handle) noexcept;
}  // namespace jyppx::liuding::tests::model_checkpoint

#endif
