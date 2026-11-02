// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

#ifndef JYPPX_LIUDING_MODELS_POSITION_INTERPOLATION_HPP
#define JYPPX_LIUDING_MODELS_POSITION_INTERPOLATION_HPP

#include <torch/csrc/autograd/custom_function.h>
#include <torch/nn/functional/upsampling.h>

#include <ATen/autocast_mode.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <cstdint>
#include <vector>

namespace jyppx::liuding::models::detail {

// Retain ATen CUDA bicubic forward rounding. Its deterministic linear adjoint
// uses the same coefficient matrices as the existing separable implementation,
// avoiding the atomic accumulation in CUDA resize backward.
class BicubicPositionsAutograd final
    : public torch::autograd::Function<BicubicPositionsAutograd> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext *context,
                               const torch::Tensor &input,
                               const torch::Tensor &vertical,
                               const torch::Tensor &horizontal,
                               bool antialias) {
    context->save_for_backward({vertical, horizontal});
    return torch::nn::functional::interpolate(
        input, torch::nn::functional::InterpolateFuncOptions()
                   .size(std::vector<std::int64_t>{vertical.size(0),
                                                   horizontal.size(1)})
                   .mode(torch::kBicubic)
                   .align_corners(false)
                   .antialias(antialias));
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *context,
      torch::autograd::variable_list gradients) {
    const auto coefficients = context->get_saved_variables();
    const c10::impl::ExcludeDispatchKeyGuard no_autocast(
        at::autocast::get_autocast_dispatch_key_from_device_type(
            gradients.front().device().type()));
    auto incoming =
        torch::matmul(coefficients[0].transpose(0, 1), gradients.front());
    auto input_gradient =
        torch::matmul(incoming, coefficients[1].transpose(0, 1));
    return {input_gradient, torch::Tensor{}, torch::Tensor{}, torch::Tensor{}};
  }
};

// Return FP32 even inside autocast. The CPU path and the linear coefficient
// definition remain unchanged; CUDA uses native forward with the same adjoint.
inline torch::Tensor SeparableBicubicPositions(const torch::Tensor &input,
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
  if (input.is_cuda())
    return BicubicPositionsAutograd::apply(input.to(torch::kFloat32), vertical,
                                           horizontal, antialias);
  return torch::matmul(vertical,
                       torch::matmul(input.to(torch::kFloat32), horizontal));
}
}  // namespace jyppx::liuding::models::detail

#endif
