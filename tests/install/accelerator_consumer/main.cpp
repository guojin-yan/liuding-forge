// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/core/device.hpp>
#include <jyppx/liuding/core/runtime.hpp>
#include <jyppx/liuding/foundation/version.hpp>

#include <torch/torch.h>

#include <iostream>
#include <string>

int main() {
  using namespace jyppx::liuding;
  initialize_runtime();
  if (!cuda_is_available() || cuda_device_count() < 1) return 1;
  const auto device = parse_device("cuda:0");
  const auto input = torch::arange(16, torch::kFloat32).reshape({4, 4}) / 16;
  auto weight = torch::eye(4, torch::TensorOptions().device(device));
  weight.requires_grad_(true);
  const auto result = torch::matmul(input.to(device), weight);
  if (!torch::allclose(result.cpu(), input)) return 2;
  result.square().mean().backward();
  if (!weight.grad().defined() ||
      !torch::isfinite(weight.grad()).all().item<bool>())
    return 3;
  const auto fingerprint = current_build_fingerprint();
#if defined(USE_ROCM)
  if (!fingerprint.libtorch_build.starts_with("rocm-") ||
      !fingerprint.accelerator_runtime.starts_with("rocm:available:"))
    return 4;
#endif
  std::cout << build_fingerprint_json(fingerprint) << '\n';
}
