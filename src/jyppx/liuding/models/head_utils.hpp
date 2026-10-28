// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_MODELS_HEAD_UTILS_HPP
#define JYPPX_LIUDING_MODELS_HEAD_UTILS_HPP
#include "detail.hpp"

#include <torch/nn/modules/linear.h>

namespace jyppx::liuding::models::detail {
inline void ValidateClasses(std::int64_t classes) {
  if (classes < 0 || classes > 1'000'000)
    Invalid("classifier.classes", "classes in [0,1000000]");
}
inline void ResetLinear(torch::nn::Module& owner, torch::nn::Linear& head,
                        const char* name, std::int64_t width,
                        std::int64_t classes, const Tensor& reference) {
  ValidateClasses(classes);
  if (classes == 0) {
    if (head) {
      owner.unregister_module(name);
      head = nullptr;
    }
    return;
  }
  auto replacement = torch::nn::Linear(width, classes);
  replacement->to(reference.device(), reference.scalar_type());
  replacement->train(owner.is_training());
  if (head)
    head = owner.replace_module(name, replacement);
  else
    head = owner.register_module(name, replacement);
}
}  // namespace jyppx::liuding::models::detail
#endif
