// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_TEST_UTILS_DEPLOYMENT_ONNX_EXPORT_SEQUENCE_CASES_HPP
#define JYPPX_TEST_UTILS_DEPLOYMENT_ONNX_EXPORT_SEQUENCE_CASES_HPP

#include <jyppx/liuding/deployment/onnx_export/exporter.hpp>

#include <torch/torch.h>

#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace jyppx::liuding::test {

struct OnnxSequenceCase {
  std::string name;
  onnx_export::ForwardFunction forward;
  std::vector<at::Tensor> rows;
  std::vector<std::int64_t> sample_shape;
  std::vector<float> expected_values;
};

inline std::vector<OnnxSequenceCase> OnnxSequenceCases(
    torch::nn::Module& module, std::string_view group) {
  const bool concatenate = group == "cat";
  const bool columns = group == "stack_columns";
  if (!concatenate && !columns && group != "stack_rows")
    throw std::invalid_argument("unknown ONNX sequence reference group");
  const std::array<std::int64_t, 2> axes =
      concatenate ? std::array<std::int64_t, 2>{1, -1}
      : columns   ? std::array<std::int64_t, 2>{2, -1}
                  : std::array<std::int64_t, 2>{1, -2};
  const auto base = module.register_buffer(
      "sequence_integer_base",
      at::scalar_tensor(INT64_C(9007199254740993),
                        at::TensorOptions().dtype(at::kLong)));
  std::vector<OnnxSequenceCase> result;
  for (const auto axis : axes) {
    for (const bool reverse : {false, true}) {
      for (int kind = 0; kind < 3; ++kind) {
        std::vector<at::Tensor> rows;
        std::array<float, 3> left;
        std::array<float, 3> right;
        if (kind == 0) {
          rows = {at::tensor({16777216.F, 16777216.F, 16777216.F}),
                  at::tensor({0.F, 1.F, 2.F})};
          // FP32 + FP64 promotes before assembly. Premature FP32 conversion
          // of the second branch would produce 0/2/4 instead of 1/2/3.
          left = {0.F, 0.F, 0.F};
          right = {1.F, 2.F, 3.F};
        } else if (kind == 1) {
          rows = {at::tensor({0.F, 1.F, 0.F}), at::tensor({0.F, 1.F, 2.F})};
          // Bool + int64 keeps every integer bit; verify small residuals of
          // values above 2^53 before the final public FP32 output conversion.
          left = {-9007199254740992.F, -9007199254740992.F,
                  -9007199254740992.F};
          right = {0.F, 1.F, 2.F};
        } else {
          rows = {at::tensor({-1.5F, 2.25F, -0.5F}),
                  at::tensor({0.F, 1.F, 1.F})};
          left = {-1.5F, 2.25F, -0.5F};
          right = {0.F, 1.F, 1.F};
        }
        if (reverse) std::swap(left, right);
        std::vector<float> expected;
        if (columns) {
          for (std::size_t i = 0; i < left.size(); ++i) {
            expected.push_back(left[i]);
            expected.push_back(right[i]);
          }
        } else {
          expected.insert(expected.end(), left.begin(), left.end());
          expected.insert(expected.end(), right.begin(), right.end());
        }
        result.push_back(
            {std::to_string(axis) + "/" + std::to_string(reverse) + "/" +
                 std::to_string(kind),
             [concatenate, reverse, axis, kind,
              base](std::span<const at::Tensor> inputs) {
               auto left = kind == 1 ? inputs[0].to(at::kBool) : inputs[0];
               auto right = kind == 0
                                ? at::add(inputs[1].to(at::kDouble), 16777217.0)
                            : kind == 1 ? at::add(inputs[1].to(at::kLong), base)
                                        : inputs[1].to(at::kBool);
               if (reverse) std::swap(left, right);
               const std::vector<at::Tensor> values{left, right};
               auto output = concatenate ? at::cat(values, axis)
                                         : at::stack(values, axis);
               if (kind == 0) output = at::sub(output, 16777216.0);
               if (kind == 1) output = at::sub(output, base);
               return std::vector<at::Tensor>{output.to(at::kFloat)};
             },
             std::move(rows),
             concatenate ? std::vector<std::int64_t>{6}
             : columns   ? std::vector<std::int64_t>{3, 2}
                         : std::vector<std::int64_t>{2, 3},
             std::move(expected)});
      }
    }
  }
  return result;
}

inline at::Tensor OnnxSequenceExpected(const OnnxSequenceCase& probe,
                                       std::int64_t batch) {
  std::vector<std::int64_t> shape{batch};
  shape.insert(shape.end(), probe.sample_shape.begin(),
               probe.sample_shape.end());
  return at::tensor(probe.expected_values).repeat({batch}).reshape(shape);
}

}  // namespace jyppx::liuding::test

#endif  // JYPPX_TEST_UTILS_DEPLOYMENT_ONNX_EXPORT_SEQUENCE_CASES_HPP
