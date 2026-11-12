// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_TEST_UTILS_DEPLOYMENT_ONNX_EXPORT_NORMALIZATION_CASES_HPP
#define JYPPX_TEST_UTILS_DEPLOYMENT_ONNX_EXPORT_NORMALIZATION_CASES_HPP

#include <jyppx/liuding/deployment/onnx_export/exporter.hpp>

#include <torch/torch.h>

#include <ATen/ops/_native_batch_norm_legit_no_training.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace jyppx::liuding::test {

struct OnnxNormalizationCase {
  std::string name;
  onnx_export::ForwardFunction forward;
  std::vector<std::int64_t> input_sample_shape;
  std::vector<std::int64_t> output_sample_shape;
  std::vector<float> input_values;
  std::vector<float> expected_values;
};

inline std::vector<OnnxNormalizationCase> OnnxNormalizationCases(
    torch::nn::Module& module, std::string_view group) {
  const bool internal = group != "frontend";
  const bool flatten = group == "internal_flatten";
  if (group != "frontend" && group != "internal" && !flatten)
    throw std::invalid_argument("unknown normalization reference group");
  const auto mean =
      module.register_buffer("bn_mean", at::tensor({1.F, -2.F, 0.5F}));
  const auto variance =
      module.register_buffer("bn_var", at::tensor({4.F, 9.F, 16.F}));
  const auto weight =
      module.register_buffer("bn_weight", at::tensor({2.F, -3.F, 0.5F}));
  const auto bias =
      module.register_buffer("bn_bias", at::tensor({0.25F, -1.F, 2.F}));
  std::vector<OnnxNormalizationCase> result;
  for (const int rank : {2, 3, 4, 5}) {
    std::vector<std::int64_t> shape{3};
    for (int axis = 2; axis < rank; ++axis) shape.push_back(2);
    const auto spatial = std::int64_t{1} << (rank - 2);
    for (const double epsilon : {0.0, 0.25}) {
      for (const int affine : {0, 1, 2, 3}) {
        if (flatten && affine != 3) continue;
        const std::optional<at::Tensor> scale =
            affine & 1 ? std::optional(weight) : std::nullopt;
        const std::optional<at::Tensor> offset =
            affine & 2 ? std::optional(bias) : std::nullopt;
        std::vector<float> input;
        std::vector<float> expected;
        constexpr std::array<double, 3> means{1.0, -2.0, 0.5};
        constexpr std::array<double, 3> variances{4.0, 9.0, 16.0};
        constexpr std::array<double, 3> scales{2.0, -3.0, 0.5};
        constexpr std::array<double, 3> offsets{0.25, -1.0, 2.0};
        for (std::size_t channel = 0; channel < 3; ++channel) {
          for (std::int64_t position = 0; position < spatial; ++position) {
            const double value = static_cast<double>(channel) * 3.0 - 4.0 +
                                 static_cast<double>(position) * 0.5;
            input.push_back(static_cast<float>(value));
            const double scale_value = affine & 1 ? scales[channel] : 1.0;
            const double offset_value = affine & 2 ? offsets[channel] : 0.0;
            expected.push_back(static_cast<float>(
                (value - means[channel]) /
                    std::sqrt(variances[channel] + epsilon) * scale_value +
                offset_value));
          }
        }
        result.push_back(
            {std::to_string(rank) + "/" + std::to_string(epsilon) + "/" +
                 std::to_string(affine),
             [internal, flatten, epsilon, scale, offset, mean,
              variance](std::span<const at::Tensor> inputs) {
               auto output =
                   internal
                       ? std::get<0>(at::_native_batch_norm_legit_no_training(
                             inputs[0], scale, offset, mean, variance, 0.1,
                             epsilon))
                       : at::batch_norm(inputs[0], scale, offset, mean,
                                        variance, false, 0.1, epsilon, false);
               if (flatten) output = output.flatten(1);
               return std::vector<at::Tensor>{output};
             },
             shape, flatten ? std::vector<std::int64_t>{3 * spatial} : shape,
             std::move(input), std::move(expected)});
      }
    }
  }
  return result;
}

inline std::vector<OnnxNormalizationCase> OnnxLayerNormalizationCases(
    torch::nn::Module& module, bool double_precision) {
  const auto dtype = double_precision ? at::kDouble : at::kFloat;
  std::vector<OnnxNormalizationCase> result;
  for (const int dimensions : {1, 2}) {
    const std::vector<std::int64_t> normalized =
        dimensions == 1 ? std::vector<std::int64_t>{4}
                        : std::vector<std::int64_t>{2, 2};
    const auto weight = module.register_buffer(
        "ln_weight_" + std::to_string(dimensions),
        at::tensor({1.0, -2.0, 0.5, 3.0}, at::TensorOptions().dtype(dtype))
            .reshape(normalized));
    const auto bias = module.register_buffer(
        "ln_bias_" + std::to_string(dimensions),
        at::tensor({0.25, -1.0, 2.0, 0.0}, at::TensorOptions().dtype(dtype))
            .reshape(normalized));
    const std::vector<std::int64_t> shape =
        dimensions == 1 ? std::vector<std::int64_t>{2, 4}
                        : std::vector<std::int64_t>{2, 2, 2};
    for (const double epsilon : {0.0, 0.25}) {
      for (const int affine : {0, 1, 2, 3}) {
        const std::optional<at::Tensor> scale =
            affine & 1 ? std::optional(weight) : std::nullopt;
        const std::optional<at::Tensor> offset =
            affine & 2 ? std::optional(bias) : std::nullopt;
        constexpr std::array<double, 4> centered{-3.0, -1.0, 1.0, 3.0};
        constexpr std::array<double, 4> scales{1.0, -2.0, 0.5, 3.0};
        constexpr std::array<double, 4> offsets{0.25, -1.0, 2.0, 0.0};
        std::vector<float> expected;
        for (int group = 0; group < 2; ++group) {
          for (std::size_t index = 0; index < centered.size(); ++index) {
            const auto scale_value = affine & 1 ? scales[index] : 1.0;
            const auto offset_value = affine & 2 ? offsets[index] : 0.0;
            expected.push_back(static_cast<float>(
                centered[index] / std::sqrt(5.0 + epsilon) * scale_value +
                offset_value));
          }
        }
        result.push_back(
            {std::to_string(dimensions) + "/" + std::to_string(epsilon) + "/" +
                 std::to_string(affine),
             [dtype, double_precision, normalized, scale, offset,
              epsilon](std::span<const at::Tensor> inputs) {
               auto input = inputs[0].to(dtype);
               // A large common offset isolates FP64 accumulation: lowering
               // through FP32 destroys the small differences before variance.
               if (double_precision) input = at::add(input, 100000000.0);
               return std::vector<at::Tensor>{at::layer_norm(input, normalized,
                                                             scale, offset,
                                                             epsilon, false)
                                                  .to(at::kFloat)};
             },
             shape,
             shape,
             {-3.F, -1.F, 1.F, 3.F, 7.F, 9.F, 11.F, 13.F},
             std::move(expected)});
      }
    }
  }
  return result;
}

inline at::Tensor OnnxNormalizationTensor(
    const std::vector<float>& values,
    const std::vector<std::int64_t>& sample_shape, std::int64_t batch) {
  std::vector<std::int64_t> shape{batch};
  shape.insert(shape.end(), sample_shape.begin(), sample_shape.end());
  return at::tensor(values).repeat({batch}).reshape(shape);
}

}  // namespace jyppx::liuding::test

#endif  // JYPPX_TEST_UTILS_DEPLOYMENT_ONNX_EXPORT_NORMALIZATION_CASES_HPP
