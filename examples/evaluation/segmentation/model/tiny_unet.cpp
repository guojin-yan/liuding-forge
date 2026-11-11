// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "tiny_unet.hpp"

#include <torch/torch.h>
namespace jyppx::liuding::examples::segmentation {
using torch::Tensor;
namespace {
torch::nn::Sequential block(std::int64_t in, std::int64_t out) {
  return torch::nn::Sequential(
      torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, 3).padding(1)),
      torch::nn::ReLU(),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(out, out, 3).padding(1)),
      torch::nn::ReLU());
}
}  // namespace
TinyUNet::TinyUNet()
    : e1(register_module("e1", block(3, 8))),
      e2(register_module("e2", block(8, 16))),
      middle(register_module("middle", block(16, 32))),
      d2(register_module("d2", block(48, 16))),
      d1(register_module("d1", block(24, 8))),
      head(register_module("head", torch::nn::Conv2d(8, 2, 1))) {}
torch::Tensor TinyUNet::forward(const Tensor& x) {
  const auto a = e1->forward(x);
  const auto b = e2->forward(torch::max_pool2d(a, 2));
  const auto c = middle->forward(torch::max_pool2d(b, 2));
  const auto up = [](const Tensor& v) {
    // Exact nearest 2x, with a deterministic reduction in backward.
    return v.unsqueeze(3)
        .unsqueeze(5)
        .expand({v.size(0), v.size(1), v.size(2), 2, v.size(3), 2})
        .reshape({v.size(0), v.size(1), v.size(2) * 2, v.size(3) * 2});
  };
  const auto d = d2->forward(torch::cat({up(c), b}, 1));
  return head->forward(d1->forward(torch::cat({up(d), a}, 1)));
}
}  // namespace jyppx::liuding::examples::segmentation
