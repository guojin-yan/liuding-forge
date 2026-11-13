// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "blocks.hpp"

#include <jyppx/liuding/vision/models/yolo/model.hpp>
namespace jyppx::liuding::models::yolo {
std::size_t fuse_for_inference(torch::nn::Module& model) {
  if (model.is_training())
    throw InvalidArgument("Fusion requires an evaluation clone");
  std::size_t count = 0;
  for (const auto& module : model.modules(false))
    if (auto conv = std::dynamic_pointer_cast<detail::Conv>(module))
      count += conv->fuse_inference();
  return count;
}
}  // namespace jyppx::liuding::models::yolo
