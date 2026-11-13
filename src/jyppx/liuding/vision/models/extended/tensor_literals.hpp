// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#pragma once
#include <torch/torch.h>
#include <vector>

namespace jyppx::liuding::models::extended::detail {
// C++ tensor(vector) may trace an allocation without its host-side writes.
// Full + stack records the values and also works during ordinary execution.
template <typename T>
Tensor tensor_literals(const std::vector<T>& values,
                       const torch::TensorOptions& options) {
  std::vector<Tensor> entries;
  entries.reserve(values.size());
  for (const auto value : values)
    entries.push_back(torch::full({}, value, options));
  return entries.empty() ? torch::empty({0}, options) : torch::stack(entries);
}
}  // namespace jyppx::liuding::models::extended::detail
