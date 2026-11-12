// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_TEST_UTILS_DEPLOYMENT_ONNX_EXPORT_ARITHMETIC_CASES_HPP
#define JYPPX_TEST_UTILS_DEPLOYMENT_ONNX_EXPORT_ARITHMETIC_CASES_HPP

#include <jyppx/liuding/deployment/onnx_export/exporter.hpp>

#include <torch/torch.h>

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace jyppx::liuding::test {
namespace exporter = jyppx::liuding::onnx_export;

struct OnnxArithmeticCase {
  std::string name;
  exporter::ForwardFunction forward;
  std::vector<at::Tensor> rows;
  at::Tensor expected_row;
};

inline std::vector<OnnxArithmeticCase> OnnxArithmeticCases(
    torch::nn::Module& module, std::string_view group) {
  std::vector<OnnxArithmeticCase> result;
  if (group == "numeric") {
    const std::array<std::array<float, 3>, 6> expected{
        {{{2.5F, -2.75F, 2.5F}},
         {{0.5F, -1.75F, -1.5F}},
         {{3.F, 2.25F, 2.F}},
         {{0.75F, 2.25F, 0.125F}},
         {{2.F, -1.F, 4.F}},
         {{1.5F, -2.25F, 0.5F}}}};
    for (int operation = 0; operation < 6; ++operation) {
      for (const bool reverse : {false, true}) {
        const exporter::ForwardFunction forward =
            [operation, reverse](std::span<const at::Tensor> inputs) {
              const auto left = reverse ? inputs[0].to(at::kDouble) : inputs[0];
              const auto right =
                  reverse ? inputs[1] : inputs[1].to(at::kDouble);
              const auto output = operation == 0   ? at::add(left, right, 0.5)
                                  : operation == 1 ? at::sub(left, right, 0.5)
                                  : operation == 2 ? at::mul(left, right)
                                  : operation == 3 ? at::div(left, right)
                                  : operation == 4 ? at::maximum(left, right)
                                                   : at::minimum(left, right);
              return std::vector<at::Tensor>{output.to(at::kFloat)};
            };
        const auto& row = expected[static_cast<std::size_t>(operation)];
        result.push_back(
            {std::to_string(operation) + "/" + std::to_string(reverse),
             forward,
             {at::tensor({1.5F, -2.25F, 0.5F}), at::tensor({2.F, -1.F, 4.F})},
             at::tensor({row[0], row[1], row[2]})});
      }
    }
    const auto baseline = module.register_buffer(
        "numeric_baseline", at::tensor({100000000.0}, at::kDouble));
    for (const bool reverse : {false, true}) {
      const exporter::ForwardFunction forward =
          [baseline, reverse](std::span<const at::Tensor> inputs) {
            const auto left = reverse ? inputs[0].to(at::kDouble) : inputs[0];
            const auto right = reverse ? inputs[1] : inputs[1].to(at::kDouble);
            return std::vector<at::Tensor>{
                at::sub(at::add(left, right), baseline).to(at::kFloat)};
          };
      result.push_back({"cancellation/" + std::to_string(reverse),
                        forward,
                        {at::tensor({100000000.F, 100000000.F, 100000000.F}),
                         at::tensor({1.F, 2.F, 3.F})},
                        at::tensor({1.F, 2.F, 3.F})});
    }
  } else if (group == "division") {
    for (const bool scalar : {false, true}) {
      const exporter::ForwardFunction forward =
          [scalar](std::span<const at::Tensor> inputs) {
            const auto left = inputs[0].to(at::kLong);
            const auto output = scalar ? at::div(left, 2)
                                       : at::div(left, inputs[1].to(at::kLong));
            return std::vector<at::Tensor>{output.to(at::kFloat)};
          };
      result.push_back(
          {std::to_string(scalar), forward,
           scalar ? std::vector<at::Tensor>{at::tensor({3.F, -5.F, 7.F})}
                  : std::vector<at::Tensor>{at::tensor({3.F, -5.F, 7.F}),
                                            at::tensor({2.F, -2.F, 4.F})},
           scalar ? at::tensor({1.5F, -2.5F, 3.5F})
                  : at::tensor({1.5F, 2.5F, 1.75F})});
    }
  } else if (group == "comparison") {
    for (int kind = 0; kind < 3; ++kind) {
      const auto scalar = module.register_buffer(
          "comparison_" + std::to_string(kind),
          at::tensor({16777217.0}, at::kDouble)
              .reshape(kind == 0 ? at::IntArrayRef{} : at::IntArrayRef{1}));
      for (int operation = 0; operation < 3; ++operation) {
        const exporter::ForwardFunction forward =
            [kind, scalar, operation](std::span<const at::Tensor> inputs) {
              const auto& left = inputs[0];
              const auto output =
                  kind == 2 ? (operation == 0   ? at::eq(left, 16777217.0)
                               : operation == 1 ? at::gt(left, 16777217.0)
                                                : at::lt(left, 16777217.0))
                            : (operation == 0   ? at::eq(left, scalar)
                               : operation == 1 ? at::gt(left, scalar)
                                                : at::lt(left, scalar));
              return std::vector<at::Tensor>{output.to(at::kFloat)};
            };
        const auto expected = operation == 0
                                  ? (kind == 1 ? at::tensor({0.F, 0.F, 0.F})
                                               : at::tensor({1.F, 0.F, 0.F}))
                              : operation == 1 ? at::tensor({0.F, 1.F, 0.F})
                              : kind == 1      ? at::tensor({1.F, 0.F, 1.F})
                                               : at::tensor({0.F, 0.F, 1.F});
        result.push_back(
            {std::to_string(kind) + "/" + std::to_string(operation),
             forward,
             {at::tensor({16777216.F, 16777218.F, 16777214.F})},
             expected});
      }
    }
  } else if (group == "integer") {
    for (const std::int64_t large :
         {INT64_C(9007199254740993), -INT64_C(9007199254740993)}) {
      const auto base = module.register_buffer(
          "integer_" + std::to_string(large), at::tensor({large}, at::kLong));
      for (const bool alpha : {false, true}) {
        const exporter::ForwardFunction forward =
            [large, base, alpha](std::span<const at::Tensor> inputs) {
              const auto left = inputs[0].to(at::kLong);
              const auto right = inputs[1].to(at::kLong);
              const auto delta =
                  alpha ? at::sub(at::add(left, right, large), right * base)
                        : at::sub(at::add(left, large), base);
              return std::vector<at::Tensor>{delta.to(at::kFloat)};
            };
        result.push_back(
            {std::to_string(large) + "/" + std::to_string(alpha),
             forward,
             {at::tensor({0.F, 1.F, -1.F}), at::tensor({1.F, 0.F, -1.F})},
             at::tensor({0.F, 1.F, -1.F})});
      }
    }
  } else if (group == "boolean") {
    const std::array<std::array<float, 3>, 7> expected{{{{1.F, 1.F, 1.F}},
                                                        {{0.F, 1.F, 0.F}},
                                                        {{1.F, 1.F, 1.F}},
                                                        {{0.F, 1.F, 0.F}},
                                                        {{0.F, 1.F, 0.F}},
                                                        {{0.F, 0.F, 1.F}},
                                                        {{1.F, 0.F, 0.F}}}};
    for (int operation = 0; operation < 8; ++operation) {
      const exporter::ForwardFunction forward =
          [operation](std::span<const at::Tensor> inputs) {
            const auto left = inputs[0].to(at::kBool);
            const auto right = inputs[1].to(at::kBool);
            const auto output = operation == 0   ? at::add(left, right)
                                : operation == 1 ? at::mul(left, right)
                                : operation == 2 ? at::maximum(left, right)
                                : operation == 3 ? at::minimum(left, right)
                                : operation == 4 ? at::eq(left, right)
                                : operation == 5 ? at::gt(left, right)
                                : operation == 6 ? at::lt(left, right)
                                                 : at::add(left, right, 0);
            return std::vector<at::Tensor>{output.to(at::kFloat)};
          };
      const auto row = operation == 7
                           ? std::array<float, 3>{0.F, 1.F, 1.F}
                           : expected[static_cast<std::size_t>(operation)];
      result.push_back(
          {std::to_string(operation),
           forward,
           {at::tensor({0.F, 1.F, 1.F}), at::tensor({1.F, 1.F, 0.F})},
           at::tensor({row[0], row[1], row[2]})});
    }
  }
  return result;
}
}  // namespace jyppx::liuding::test

#endif  // JYPPX_TEST_UTILS_DEPLOYMENT_ONNX_EXPORT_ARITHMETIC_CASES_HPP
