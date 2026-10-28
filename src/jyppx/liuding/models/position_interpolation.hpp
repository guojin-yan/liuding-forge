// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

#ifndef JYPPX_LIUDING_MODELS_POSITION_INTERPOLATION_HPP
#define JYPPX_LIUDING_MODELS_POSITION_INTERPOLATION_HPP

#include <torch/nn/functional/upsampling.h>

#include <ATen/autocast_mode.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <cstdint>
#include <vector>

namespace jyppx::liuding::models::detail {

// Bicubic resize is separable and linear. Derive its small coefficient matrices
// with the CPU ATen implementation, then retain activations and gradients on
// the original device. Matmul backward avoids CUDA resize's atomic
// accumulation. This private path returns FP32 even inside an autocast scope.
inline torch::Tensor SeparableBicubicPositions(const torch::Tensor& input,
                                               std::int64_t height,
                                               std::int64_t width,
                                               bool antialias) {
  const c10::impl::ExcludeDispatchKeyGuard no_autocast(
      at::autocast::get_autocast_dispatch_key_from_device_type(
          input.device().type()));
  const auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  const auto coefficients = [&](std::int64_t source, std::int64_t target,
                                bool vertical) {
    auto basis = torch::eye(source, options).reshape({1, 1, source, source});
    return torch::nn::functional::interpolate(
               basis,
               torch::nn::functional::InterpolateFuncOptions()
                   .size(std::vector<std::int64_t>{vertical ? target : source,
                                                   vertical ? source : target})
                   .mode(torch::kBicubic)
                   .align_corners(false)
                   .antialias(antialias))
        .select(0, 0)
        .select(0, 0)
        .to(input.device());
  };
  const auto vertical = coefficients(input.size(2), height, true);
  const auto horizontal = coefficients(input.size(3), width, false);
  return torch::matmul(vertical,
                       torch::matmul(input.to(torch::kFloat32), horizontal));
}
}  // namespace jyppx::liuding::models::detail

#endif
