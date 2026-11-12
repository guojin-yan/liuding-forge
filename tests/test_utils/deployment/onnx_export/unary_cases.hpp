// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_TEST_UTILS_DEPLOYMENT_ONNX_EXPORT_UNARY_CASES_HPP
#define JYPPX_TEST_UTILS_DEPLOYMENT_ONNX_EXPORT_UNARY_CASES_HPP

#include <jyppx/liuding/deployment/onnx_export/exporter.hpp>

#include <torch/torch.h>

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace jyppx::liuding::test {

struct OnnxUnaryCase {
  std::string name;
  onnx_export::ForwardFunction forward;
  at::Tensor row;
  at::Tensor expected_row;
  bool exact;
  bool cpu_kernel = true;
};

inline at::Tensor OnnxUnaryOperation(std::string_view operation,
                                     const at::Tensor& input) {
  if (operation == "sigmoid") return at::sigmoid(input);
  if (operation == "tanh") return at::tanh(input);
  if (operation == "exp") return at::exp(input);
  if (operation == "log") return at::log(input);
  if (operation == "sqrt") return at::sqrt(input);
  if (operation == "erf") return at::erf(input);
  if (operation == "floor") return at::floor(input);
  if (operation == "ceil") return at::ceil(input);
  if (operation == "round") return at::round(input);
  throw std::invalid_argument("unknown unary reference case");
}

inline std::vector<OnnxUnaryCase> OnnxUnaryCases(torch::nn::Module& module,
                                                 std::string_view operation) {
  std::vector<OnnxUnaryCase> result;
  const bool rounding =
      operation == "floor" || operation == "ceil" || operation == "round";
  if (rounding) {
    for (const auto dtype :
         {at::kByte, at::kChar, at::kShort, at::kInt, at::kLong}) {
      // A float conversion would corrupt the int64 residuals above 2^53.
      const auto base = module.register_buffer(
          "rounding_base_" + std::to_string(static_cast<int>(dtype)),
          at::scalar_tensor(
              dtype == at::kLong ? INT64_C(9007199254740993) : INT64_C(0),
              at::TensorOptions().dtype(dtype)));
      const auto row = dtype == at::kByte   ? at::tensor({0.F, 128.F, 255.F})
                       : dtype == at::kLong ? at::tensor({0.F, 1.F, 2.F})
                                            : at::tensor({-3.F, 0.F, 7.F});
      result.push_back(
          {std::to_string(static_cast<int>(dtype)),
           [name = std::string(operation), dtype,
            base](std::span<const at::Tensor> inputs) {
             const auto input = at::add(inputs[0].to(dtype), base);
             return std::vector<at::Tensor>{
                 at::sub(OnnxUnaryOperation(name, input), base).to(at::kFloat)};
           },
           row, row.clone(), true});
    }
    for (const auto dtype : {at::kFloat, at::kDouble}) {
      const auto expected = operation == "floor" ? at::tensor({-3.F, -2.F, 2.F})
                            : operation == "ceil"
                                ? at::tensor({-2.F, -1.F, 3.F})
                                : at::tensor({-2.F, -2.F, 3.F});
      result.push_back(
          {std::to_string(static_cast<int>(dtype)),
           [name = std::string(operation),
            dtype](std::span<const at::Tensor> inputs) {
             return std::vector<at::Tensor>{
                 OnnxUnaryOperation(name, inputs[0].to(dtype)).to(at::kFloat)};
           },
           at::tensor({-2.5F, -1.5F, 2.75F}), expected, true});
    }
    return result;
  }

  for (const auto dtype :
       {at::kLong, at::kInt, at::kBool, at::kDouble, at::kFloat}) {
    auto row = at::tensor({-1.F, 0.F, 1.F});
    at::Tensor expected;
    const bool boolean = dtype == at::kBool;
    // Constants come from real-valued mathematical definitions, independently
    // of the ATen operation being exported. Each is rounded once to float32.
    if (operation == "sigmoid") {
      expected = at::tensor({0.2689414213699951F, 0.5F, 0.7310585786300049F});
      if (boolean)
        expected = at::tensor({0.5F, 0.7310585786300049F, 0.7310585786300049F});
    } else if (operation == "tanh") {
      expected = at::tensor({-0.7615941559557649F, 0.F, 0.7615941559557649F});
      if (boolean)
        expected = at::tensor({0.F, 0.7615941559557649F, 0.7615941559557649F});
    } else if (operation == "exp") {
      expected = at::tensor({0.3678794411714423F, 1.F, 2.718281828459045F});
      if (boolean)
        expected = at::tensor({1.F, 2.718281828459045F, 2.718281828459045F});
    } else if (operation == "log") {
      row = at::tensor({1.F, 2.F, 8.F});
      expected = at::tensor({0.F, 0.6931471805599453F, 2.079441541679836F});
      if (boolean) expected = at::zeros({3}, at::kFloat);
    } else if (operation == "sqrt") {
      row = at::tensor({0.F, 4.F, 9.F});
      expected =
          boolean ? at::tensor({0.F, 1.F, 1.F}) : at::tensor({0.F, 2.F, 3.F});
    } else if (operation == "erf") {
      expected = at::tensor({-0.8427007929497149F, 0.F, 0.8427007929497149F});
      if (boolean)
        expected = at::tensor({0.F, 0.8427007929497149F, 0.8427007929497149F});
    } else {
      throw std::invalid_argument("unknown unary reference case");
    }
    if (boolean && operation != "log" && operation != "sqrt")
      row = at::tensor({0.F, 1.F, 1.F});
    result.push_back(
        {std::to_string(static_cast<int>(dtype)),
         [name = std::string(operation),
          dtype](std::span<const at::Tensor> inputs) {
           return std::vector<at::Tensor>{
               OnnxUnaryOperation(name, inputs[0].to(dtype)).to(at::kFloat)};
         },
         row, expected, operation == "sqrt",
         !(operation == "erf" && dtype == at::kDouble)});
  }
  return result;
}

}  // namespace jyppx::liuding::test

#endif  // JYPPX_TEST_UTILS_DEPLOYMENT_ONNX_EXPORT_UNARY_CASES_HPP
