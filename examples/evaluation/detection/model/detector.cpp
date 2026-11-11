// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "detector.hpp"

#include <torch/nn/modules/activation.h>
#include <torch/utils.h>

#include <array>
#include <cstdint>

namespace jyppx::liuding::examples::detection {

Detector::Detector() {
  body = register_module("body", torch::nn::Sequential());
  std::int64_t input = 3;
  const std::array<std::int64_t, 6> channels{16, 32, 64, 64, 64, 64};
  for (std::size_t i = 0; i < channels.size(); ++i) {
    body->push_back(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(input, channels[i], 3)
                              .stride(i < 4 ? 2 : 1)
                              .padding(1)));
    body->push_back(torch::nn::ReLU());
    input = channels[i];
  }
  head = register_module("head", torch::nn::Conv2d(64, 15, 1));
  torch::NoGradGuard guard;
  for (std::int64_t a = 0; a < 3; ++a) head->bias[a * 5 + 4].fill_(-2.);
}

torch::Tensor Detector::forward(const torch::Tensor& images) {
  return head->forward(body->forward(images));
}

}  // namespace jyppx::liuding::examples::detection
