// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_PRIVATE_CLASS_WEIGHTS_HPP
#define JYPPX_LIUDING_VISION_TASKS_PRIVATE_CLASS_WEIGHTS_HPP
#include <jyppx/liuding/core/types.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <cmath>
namespace jyppx::liuding::vision::tasks::detail {
inline void check_class_weights(const std::vector<double>& weights,
                                std::int64_t classes) {
  if (weights.empty()) return;
  if (weights.size() != static_cast<std::size_t>(classes))
    throw InvalidArgument("Class weights differ from class count");
  double total = 0;
  for (double value : weights) {
    if (!std::isfinite(value) || value < 0)
      throw InvalidArgument("Class weights must be finite nonnegative");
    total += value;
  }
  if (!std::isfinite(total) || total <= 0)
    throw InvalidArgument("Class weights need positive mass");
}
inline Tensor class_weight_tensor(const std::vector<double>& weights,
                                  const Tensor& reference) {
  if (weights.empty()) return {};
  return torch::tensor(weights, reference.options().dtype(torch::kFloat32));
}
}  // namespace jyppx::liuding::vision::tasks::detail
#endif
