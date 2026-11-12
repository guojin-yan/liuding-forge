// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "../../test_utils/deployment/onnx_export/arithmetic_cases.hpp"
#include "../../test_utils/deployment/onnx_export/unary_cases.hpp"

#include <jyppx/liuding/deployment/onnx_export/exporter.hpp>
#include <jyppx/liuding/deployment/onnx_runtime/session.hpp>
#include <jyppx/liuding/deployment/onnx_runtime/verification.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace exporter = jyppx::liuding::onnx_export;
namespace runtime = jyppx::liuding::onnx_runtime;

void Require(bool condition, const std::string& operation) {
  if (!condition) throw std::runtime_error(operation);
}

// This type exists only in the downstream consumer, with no model registration.
class UserNetworkImpl final : public torch::nn::Module {
 public:
  UserNetworkImpl()
      : hidden_(register_module("hidden", torch::nn::Linear(4, 6))),
        output_(register_module("output", torch::nn::Linear(6, 3))) {
    torch::NoGradGuard guard;
    const auto options =
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    hidden_->weight.copy_(torch::arange(24, options).reshape({6, 4}) / 12.0F -
                          0.9F);
    hidden_->bias.copy_(torch::linspace(-0.4F, 0.6F, 6, options));
    output_->weight.copy_(torch::arange(18, options).reshape({3, 6}) / 15.0F -
                          0.5F);
    output_->bias.copy_(torch::tensor({0.1F, -0.3F, 0.5F}, options));
  }

  torch::Tensor forward(const torch::Tensor& input) {
    return output_->forward(torch::relu(hidden_->forward(input)));
  }

 private:
  torch::nn::Linear hidden_;
  torch::nn::Linear output_;
};
TORCH_MODULE(UserNetwork);

torch::Tensor Input(std::int64_t batch) {
  const auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  return (torch::arange(batch * 4, options).reshape({batch, 4}) * 0.175F -
          1.25F)
      .contiguous();
}

void Compare(const runtime::TensorOutput& actual,
             const torch::Tensor& expected) {
  Require(actual.shape == std::vector<std::int64_t>(expected.sizes().begin(),
                                                    expected.sizes().end()),
          "installed dynamic output shape");
  Require(actual.values.size() == static_cast<std::size_t>(expected.numel()),
          "installed output element count");
  const auto* reference = expected.const_data_ptr<float>();
  for (std::size_t index = 0; index < actual.values.size(); ++index) {
    const auto tolerance = 1e-5F + 1e-4F * std::abs(reference[index]);
    Require(std::isfinite(actual.values[index]) &&
                std::isfinite(reference[index]) &&
                std::abs(actual.values[index] - reference[index]) <= tolerance,
            "installed ONNX/native numerical mismatch at element " +
                std::to_string(index));
  }
}

void CheckUnaryOperators() {
  for (const std::string_view operation :
       {"sigmoid", "tanh", "exp", "log", "sqrt", "erf", "floor", "ceil",
        "round"}) {
    torch::nn::Module module;
    const auto probes = jyppx::liuding::test::OnnxUnaryCases(module, operation);
    Require(probes.size() == (operation == "floor" || operation == "ceil" ||
                                      operation == "round"
                                  ? 7
                                  : 5),
            "installed unary fixture count");
    for (const auto& probe : probes) {
      const auto label = std::string(operation) + "/" + probe.name;
      const std::vector<at::Tensor> examples{probe.row.repeat({2, 1})};
      const auto artifact =
          exporter::export_module(module, probe.forward, examples);
      if (!probe.cpu_kernel) {
        const auto native = probe.forward(examples);
        Require(native.size() == 1 &&
                    at::allclose(native.front(),
                                 probe.expected_row.repeat({2, 1}), 1e-6, 1e-6),
                "installed Double Erf native reference");
        bool refused = false;
        try {
          runtime::Session unsupported(artifact.onnx, artifact.digest);
        } catch (const jyppx::liuding::CompatibilityError& error) {
          refused = error.details().operation == "load" &&
                    error.details().actual.find("Erf") != std::string::npos;
        }
        Require(refused, "ORT 1.23.2 CPU must refuse Double Erf " + label);
        continue;
      }
      runtime::Session session(artifact.onnx, artifact.digest);
      for (const auto batch : {1, 2, 5}) {
        const auto input = probe.row.repeat({batch, 1});
        const runtime::TensorInput binding{
            .name = artifact.input_names.front(),
            .values = std::span(input.const_data_ptr<float>(),
                                static_cast<std::size_t>(input.numel())),
            .shape = std::span(input.sizes().data(), input.sizes().size())};
        const auto actual = session.run(std::span(&binding, 1));
        const auto expected = probe.expected_row.repeat({batch, 1});
        const auto native = probe.forward(std::vector<at::Tensor>{input});
        Require(native.size() == 1 &&
                    (probe.exact
                         ? at::equal(native.front(), expected)
                         : at::allclose(native.front(), expected, 1e-6, 1e-6)),
                "installed native unary reference " + label);
        Require(actual.size() == 1, "installed unary output count " + label);
        Compare(actual.front(), expected);
        if (probe.exact) {
          for (std::size_t index = 0; index < actual.front().values.size();
               ++index)
            Require(actual.front().values[index] ==
                        expected.const_data_ptr<float>()[index],
                    "installed exact integer unary " + label);
        }
      }
    }
  }
}

void CheckArithmetic() {
  for (const std::string_view group :
       {"numeric", "division", "comparison", "integer", "boolean"}) {
    torch::nn::Module module;
    const auto probes =
        jyppx::liuding::test::OnnxArithmeticCases(module, group);
    Require(!probes.empty(), "installed arithmetic fixture group");
    for (const auto& probe : probes) {
      const auto inputs_for = [&](int batch) {
        std::vector<at::Tensor> inputs;
        for (const auto& row : probe.rows)
          inputs.push_back(row.repeat({batch, 1}));
        return inputs;
      };
      const auto examples = inputs_for(2);
      const auto artifact =
          exporter::export_module(module, probe.forward, examples);
      runtime::Session session(artifact.onnx, artifact.digest);
      for (const auto batch : {1, 2, 5}) {
        const auto inputs = inputs_for(batch);
        std::vector<runtime::TensorInput> bindings;
        for (std::size_t index = 0; index < inputs.size(); ++index) {
          const auto& input = inputs[index];
          bindings.push_back(
              {.name = artifact.input_names[index],
               .values = std::span(input.const_data_ptr<float>(),
                                   static_cast<std::size_t>(input.numel())),
               .shape = std::span(input.sizes().data(), input.sizes().size())});
        }
        const auto actual = session.run(bindings);
        Require(actual.size() == 1 &&
                    actual.front().shape == std::vector<std::int64_t>{batch, 3},
                "installed arithmetic output shape " + std::string(group) +
                    "/" + probe.name);
        const auto expected = probe.expected_row.repeat({batch, 1});
        const auto native = probe.forward(inputs);
        Require(native.size() == 1 && torch::equal(native.front(), expected),
                "installed native arithmetic reference " + std::string(group) +
                    "/" + probe.name);
        Require(actual.front().values.size() ==
                    static_cast<std::size_t>(expected.numel()),
                "installed arithmetic output count");
        for (std::size_t index = 0; index < actual.front().values.size();
             ++index)
          Require(actual.front().values[index] ==
                      expected.const_data_ptr<float>()[index],
                  "installed exact arithmetic " + std::string(group) + "/" +
                      probe.name + " element=" + std::to_string(index));
      }
    }
  }
}

void CheckFlatten() {
  torch::nn::Module module;
  for (const auto start : {1, 2, 3, -3, -2, -1}) {
    for (const auto end : {-1, 3}) {
      const exporter::ForwardFunction forward =
          [start, end](std::span<const at::Tensor> inputs) {
            return std::vector<at::Tensor>{
                at::flatten(inputs.front(), start, end)};
          };
      const std::array examples{
          at::arange(48, at::kFloat).reshape({2, 2, 3, 4})};
      const auto artifact = exporter::export_module(module, forward, examples);
      runtime::Session session(artifact.onnx, artifact.digest);
      const auto axis = start < 0 ? start + 4 : start;
      for (const auto batch : {1, 2, 5}) {
        const auto input =
            at::arange(batch * 24, at::kFloat).reshape({batch, 2, 3, 4});
        const auto actual =
            session.run(std::span(input.const_data_ptr<float>(),
                                  static_cast<std::size_t>(input.numel())),
                        std::span(input.sizes().data(), input.sizes().size()));
        const std::vector<std::int64_t> expected_shape =
            axis == 1   ? std::vector<std::int64_t>{batch, 24}
            : axis == 2 ? std::vector<std::int64_t>{batch, 2, 12}
                        : std::vector<std::int64_t>{batch, 2, 3, 4};
        Require(
            actual.shape == expected_shape &&
                actual.values.size() == static_cast<std::size_t>(batch * 24),
            "installed suffix flatten preserves prefix dimensions");
        for (std::size_t index = 0; index < actual.values.size(); ++index)
          Require(actual.values[index] == static_cast<float>(index),
                  "installed suffix flatten preserves exact row-major data");
      }
    }
  }
  exporter::ExportOptions options;
  options.dynamic_batch = false;
  for (const std::vector<std::int64_t> shape :
       {std::vector<std::int64_t>{24}, {2, 3, 4}}) {
    for (const auto start : {0, -static_cast<int>(shape.size())}) {
      const exporter::ForwardFunction forward =
          [start](std::span<const at::Tensor> inputs) {
            return std::vector<at::Tensor>{
                at::flatten(inputs.front(), start, -1)};
          };
      const auto input = at::arange(24, at::kFloat).reshape(shape);
      const std::array examples{input};
      const auto artifact =
          exporter::export_module(module, forward, examples, options);
      runtime::Session session(artifact.onnx, artifact.digest);
      const auto actual =
          session.run(std::span(input.const_data_ptr<float>(),
                                static_cast<std::size_t>(input.numel())),
                      std::span(input.sizes().data(), input.sizes().size()));
      Require(actual.shape == std::vector<std::int64_t>{24} &&
                  actual.values.size() == 24U,
              "installed all-axis flatten returns rank one");
      for (std::size_t index = 0; index < actual.values.size(); ++index)
        Require(actual.values[index] == static_cast<float>(index),
                "installed all-axis flatten preserves exact data");
    }
  }
}
void CheckExact(runtime::Session& session, const at::Tensor& input,
                const std::vector<std::int64_t>& shape,
                const std::vector<float>& values) {
  const auto actual =
      session.run(std::span(input.const_data_ptr<float>(),
                            static_cast<std::size_t>(input.numel())),
                  std::span(input.sizes().data(), input.sizes().size()));
  Require(actual.shape == shape && actual.values == values,
          "installed shape operator exact independent reference");
}

void CheckShapeOperators() {
  torch::nn::Module module;
  for (const auto axis : {1, -3, 2, -2, 3, -1}) {
    const exporter::ForwardFunction forward =
        [axis](std::span<const at::Tensor> inputs) {
          return std::vector<at::Tensor>{at::squeeze(inputs.front(), axis)};
        };
    const std::array examples{at::arange(6, at::kFloat).reshape({2, 1, 3, 1})};
    const auto artifact = exporter::export_module(module, forward, examples);
    runtime::Session session(artifact.onnx, artifact.digest);
    const auto normalized = axis < 0 ? axis + 4 : axis;
    for (const auto batch : {1, 2, 5}) {
      const auto input =
          at::arange(batch * 3, at::kFloat).reshape({batch, 1, 3, 1});
      const std::vector<std::int64_t> shape =
          normalized == 1   ? std::vector<std::int64_t>{batch, 3, 1}
          : normalized == 3 ? std::vector<std::int64_t>{batch, 1, 3}
                            : std::vector<std::int64_t>{batch, 1, 3, 1};
      std::vector<float> values(static_cast<std::size_t>(batch * 3));
      for (std::size_t index = 0; index < values.size(); ++index)
        values[index] = static_cast<float>(index);
      CheckExact(session, input, shape, values);
    }
  }
  for (const std::vector<std::int64_t> axes :
       {std::vector<std::int64_t>{0, -1, -2}, {0, -2, -1}}) {
    const exporter::ForwardFunction forward =
        [axes](std::span<const at::Tensor> inputs) {
          return std::vector<at::Tensor>{
              at::permute(inputs.front(), axes).contiguous()};
        };
    const std::array examples{at::arange(12, at::kFloat).reshape({2, 2, 3})};
    const auto artifact = exporter::export_module(module, forward, examples);
    runtime::Session session(artifact.onnx, artifact.digest);
    for (const auto batch : {1, 2, 5}) {
      const auto input =
          at::arange(batch * 6, at::kFloat).reshape({batch, 2, 3});
      const std::vector<std::int64_t> shape =
          axes[1] == -1 ? std::vector<std::int64_t>{batch, 3, 2}
                        : std::vector<std::int64_t>{batch, 2, 3};
      std::vector<float> values(static_cast<std::size_t>(batch * 6));
      for (std::int64_t n = 0; n < batch; ++n)
        for (std::int64_t row = 0; row < 2; ++row)
          for (std::int64_t column = 0; column < 3; ++column) {
            const auto offset = axes[1] == -1 ? n * 6 + column * 2 + row
                                              : n * 6 + row * 3 + column;
            values[static_cast<std::size_t>(offset)] =
                static_cast<float>(n * 6 + row * 3 + column);
          }
      CheckExact(session, input, shape, values);
    }
  }
  exporter::ExportOptions options;
  options.dynamic_batch = false;
  for (const auto mode : {0, 1, 2}) {
    const exporter::ForwardFunction forward =
        [mode](std::span<const at::Tensor> inputs) {
          const auto scalar = at::mean(inputs.front());
          const auto output = mode == 0   ? at::transpose(scalar, 0, -1)
                              : mode == 1 ? at::t(scalar)
                                          : at::squeeze(scalar, 0);
          return std::vector<at::Tensor>{at::unsqueeze(output, 0)};
        };
    const auto input = at::arange(6, at::kFloat).reshape({2, 3});
    const std::array examples{input};
    const auto artifact =
        exporter::export_module(module, forward, examples, options);
    runtime::Session session(artifact.onnx, artifact.digest);
    CheckExact(session, input, {1}, {2.5F});
  }
  for (const auto axis : {0, -3, 99}) {
    const exporter::ForwardFunction forward =
        [axis](std::span<const at::Tensor> inputs) {
          return std::vector<at::Tensor>{
              axis == 99 ? at::squeeze(inputs.front())
                         : at::squeeze(inputs.front(), axis)};
        };
    const std::array examples{at::ones({2, 3, 4})};
    bool rejected = false;
    try {
      static_cast<void>(exporter::export_module(module, forward, examples));
    } catch (const jyppx::liuding::CompatibilityError& error) {
      rejected = std::string(error.what()).find("dynamic") != std::string::npos;
    }
    Require(rejected, "installed dynamic squeeze rejects variable output rank");
  }
}

void CheckReductions() {
  torch::nn::Module module;
  exporter::ExportOptions fixed;
  fixed.dynamic_batch = false;
  for (const auto mean : {false, true}) {
    for (const auto empty : {false, true}) {
      for (const auto keepdim : {true, false}) {
        const exporter::ForwardFunction forward =
            [mean, empty, keepdim](std::span<const at::Tensor> inputs) {
              const std::vector<std::int64_t> dimensions;
              const auto axes = empty ? at::OptionalIntArrayRef(dimensions)
                                      : at::OptionalIntArrayRef{};
              const auto output = mean ? at::mean(inputs.front(), axes, keepdim)
                                       : at::sum(inputs.front(), axes, keepdim);
              return std::vector<at::Tensor>{
                  keepdim ? output : at::unsqueeze(output, 0)};
            };
        const auto input = at::arange(24, at::kFloat).reshape({2, 3, 4});
        const std::array examples{input};
        const auto artifact =
            exporter::export_module(module, forward, examples, fixed);
        runtime::Session session(artifact.onnx, artifact.digest);
        const std::vector<std::int64_t> shape =
            keepdim ? std::vector<std::int64_t>{1, 1, 1}
                    : std::vector<std::int64_t>{1};
        CheckExact(session, input, shape, {mean ? 11.5F : 276.0F});
      }
    }
  }
  for (const auto mean : {false, true}) {
    for (const auto dimensioned : {false, true}) {
      const exporter::ForwardFunction forward =
          [mean, dimensioned](std::span<const at::Tensor> inputs) {
            const auto output =
                dimensioned
                    ? (mean ? at::mean(inputs.front(), {1}, true, at::kDouble)
                            : at::sum(inputs.front(), {1}, true, at::kDouble))
                    : (mean ? at::mean(inputs.front(), at::kDouble)
                            : at::sum(inputs.front(), at::kDouble));
            const auto fp32 = output.to(at::kFloat);
            return std::vector<at::Tensor>{
                dimensioned ? fp32 : at::unsqueeze(fp32, 0)};
          };
      exporter::ExportOptions options;
      options.dynamic_batch = dimensioned;
      const std::array examples{
          at::tensor({100000000.0F, 1.0F, -100000000.0F}).repeat({2, 1})};
      const auto artifact =
          exporter::export_module(module, forward, examples, options);
      runtime::Session session(artifact.onnx, artifact.digest);
      const std::vector<int> batches =
          dimensioned ? std::vector<int>{1, 2, 5} : std::vector<int>{2};
      for (const auto batch : batches) {
        const auto input =
            at::tensor({100000000.0F, 1.0F, -100000000.0F}).repeat({batch, 1});
        const std::vector<std::int64_t> shape =
            dimensioned ? std::vector<std::int64_t>{batch, 1}
                        : std::vector<std::int64_t>{1};
        const auto value =
            mean ? 1.0F / 3.0F : static_cast<float>(dimensioned ? 1 : batch);
        CheckExact(session, input, shape,
                   std::vector<float>(dimensioned ? batch : 1, value));
      }
    }
  }
  for (const auto mode : {0, 1, 2}) {
    const exporter::ForwardFunction forward =
        [mode](std::span<const at::Tensor> inputs) {
          const auto output =
              mode == 0   ? at::sum(inputs.front().to(at::kInt), {1}, true)
              : mode == 1 ? at::sum(inputs.front(), {1}, true, at::kLong)
                          : at::sum(inputs.front(), {1}, true, at::kInt);
          return std::vector<at::Tensor>{output.to(at::kFloat)};
        };
    // Every value fits int32, but the first row's exact integer sum does not.
    const auto row = mode == 0 ? at::tensor({2147483520.0F, 256.0F, 0.0F})
                               : at::tensor({1.75F, 2.5F, -1.25F});
    const std::array examples{row.repeat({2, 1})};
    const auto artifact = exporter::export_module(module, forward, examples);
    runtime::Session session(artifact.onnx, artifact.digest);
    for (const auto batch : {1, 2, 5}) {
      const auto input = row.repeat({batch, 1});
      // 2147483776 rounds to 2147483648 when the public output casts to FP32.
      const auto value = mode == 0 ? 2147483648.0F : 2.0F;
      CheckExact(session, input, {batch, 1}, std::vector<float>(batch, value));
    }
  }
}

void CheckActivations() {
  torch::nn::Module module;
  const std::vector<float> points{-6.0F, -3.5F, -3.0F, -2.5F, -1.5F,
                                  -1.0F, -0.5F, 0.0F,  0.5F,  1.0F,
                                  1.5F,  2.5F,  3.0F,  3.5F,  6.0F};
  const auto row = at::tensor(points);
  for (const bool swish : {false, true}) {
    const exporter::ForwardFunction forward =
        [swish](std::span<const at::Tensor> inputs) {
          return std::vector<at::Tensor>{swish
                                             ? at::hardswish(inputs.front())
                                             : at::hardsigmoid(inputs.front())};
        };
    const std::array examples{row.repeat({2, 1})};
    const auto artifact = exporter::export_module(module, forward, examples);
    runtime::Session session(artifact.onnx, artifact.digest);
    for (const auto batch : {1, 2, 5}) {
      const auto input = row.repeat({batch, 1});
      auto reference = at::empty_like(input);
      auto* values = reference.data_ptr<float>();
      for (std::int64_t n = 0; n < batch; ++n)
        for (std::size_t k = 0; k < points.size(); ++k) {
          const double x = points[k];
          const auto bounded = std::clamp((x + 3.0) / 6.0, 0.0, 1.0);
          values[static_cast<std::size_t>(n) * points.size() + k] =
              static_cast<float>(swish ? x * bounded : bounded);
        }
      const std::array<std::int64_t, 2> shape{
          batch, static_cast<std::int64_t>(points.size())};
      Compare(session.run(std::span(input.const_data_ptr<float>(),
                                    static_cast<std::size_t>(input.numel())),
                          shape),
              reference);
    }
  }
  const auto logits = at::tensor({0.0F, -104.0F, -105.0F});
  const auto weights = module.register_buffer(
      "weights", at::tensor({0.0, 1e40, 1e40}, at::kDouble));
  const double normalizer = 1.0 + std::exp(-104.0) + std::exp(-105.0);
  for (const bool internal : {false, true}) {
    for (const bool logarithmic : {false, true}) {
      for (const auto axis : {1, -1}) {
        const exporter::ForwardFunction forward =
            [internal, logarithmic, axis,
             weights](std::span<const at::Tensor> inputs) {
              const auto input = inputs.front();
              const auto output =
                  internal
                      ? (logarithmic
                             ? at::_log_softmax(input.to(at::kDouble), axis,
                                                false)
                             : at::_softmax(input.to(at::kDouble), axis, false))
                      : (logarithmic ? at::log_softmax(input, axis, at::kDouble)
                                     : at::softmax(input, axis, at::kDouble));
              const auto probability = logarithmic ? at::exp(output) : output;
              return std::vector<at::Tensor>{
                  (probability * weights).to(at::kFloat)};
            };
        const std::array examples{logits.repeat({2, 1})};
        const auto artifact =
            exporter::export_module(module, forward, examples);
        runtime::Session session(artifact.onnx, artifact.digest);
        for (const auto batch : {1, 2, 5}) {
          const auto input = logits.repeat({batch, 1});
          auto reference = at::empty_like(input);
          auto* values = reference.data_ptr<float>();
          for (std::int64_t n = 0; n < batch; ++n)
            for (std::size_t k = 0; k < 3; ++k) {
              values[static_cast<std::size_t>(n) * 3 + k] =
                  k == 0
                      ? 0.0F
                      : static_cast<float>(std::exp(k == 1 ? -104.0 : -105.0) /
                                           normalizer * 1e40);
            }
          const std::array<std::int64_t, 2> shape{batch, 3};
          Compare(
              session.run(std::span(input.const_data_ptr<float>(),
                                    static_cast<std::size_t>(input.numel())),
                          shape),
              reference);
        }
      }
    }
  }
  exporter::ExportOptions options;
  options.dynamic_batch = false;
  for (const auto mode : {0, 1, 2, 3}) {
    for (const auto axis : {0, -1}) {
      const exporter::ForwardFunction forward =
          [mode, axis](std::span<const at::Tensor> inputs) {
            const auto scalar = at::mean(inputs.front());
            const auto output = mode == 0   ? at::softmax(scalar, axis)
                                : mode == 1 ? at::log_softmax(scalar, axis)
                                : mode == 2
                                    ? at::_softmax(scalar, axis, false)
                                    : at::_log_softmax(scalar, axis, false);
            return std::vector<at::Tensor>{at::unsqueeze(output, 0)};
          };
      const auto input = at::arange(6, at::kFloat).reshape({2, 3});
      const std::array examples{input};
      const auto artifact =
          exporter::export_module(module, forward, examples, options);
      runtime::Session session(artifact.onnx, artifact.digest);
      CheckExact(session, input, {1}, {mode % 2 == 0 ? 1.0F : 0.0F});
    }
  }
}

}  // namespace

int main() {
  try {
    at::set_num_threads(1);
    torch::NoGradGuard guard;
    UserNetwork model;
    model->eval();

    exporter::ExportOptions export_options;
    export_options.input_names = {"features"};
    export_options.output_names = {"logits"};
    export_options.dynamic_batch = true;
    export_options.minimum_batch = 1;
    export_options.maximum_batch = 8;
    export_options.dynamic_axes = {{"features", {{0, "batch"}}},
                                   {"logits", {{0, "batch"}}}};
    const auto artifact =
        exporter::export_model(model, Input(2), export_options);
    const auto report = exporter::deserialize_export_report(
        exporter::serialize_export_report(artifact.report.value()));
    exporter::verify_export_report_model(report, artifact.onnx);
    Require(report.inputs.front().name == "features" &&
                report.outputs.front().name == "logits" &&
                report.dynamic_axes.at("features").at(0) == "batch",
            "installed native export report");
    Require(!artifact.onnx.empty() &&
                artifact.model == exporter::ModelKind::generic &&
                !artifact.external_data,
            "installed generic self-contained export");

    runtime::SessionOptions session_options;
    session_options.provider.name = "CPUExecutionProvider";
    session_options.allow_cpu_fallback = false;
    session_options.cpu_threads = 1;
    session_options.max_batch_size = 8;
    runtime::Session session(artifact.onnx, artifact.digest, session_options);
    Require(session.execution_provider().name == "CPUExecutionProvider" &&
                !session.cpu_fallback_allowed(),
            "installed explicit CPU provider");
    Require(session.model_digest().hex == artifact.digest.hex,
            "installed artifact identity");
    Require(
        session.input().name == "features" && session.output().name == "logits",
        "installed public input/output names");
    Require(session.input().shape == std::vector<std::int64_t>{-1, 4} &&
                session.output().shape == std::vector<std::int64_t>{-1, 3} &&
                session.input().batch_symbol == "batch" &&
                session.output().batch_symbol == "batch",
            "installed symbolic batch signature");

    for (const auto batch : {1, 2, 5}) {
      const auto input = Input(batch);
      const auto expected = model->forward(input).contiguous();
      const auto actual =
          session.run(std::span(input.const_data_ptr<float>(),
                                static_cast<std::size_t>(input.numel())),
                      std::span(input.sizes().data(), input.sizes().size()));
      Compare(actual, expected);
      const std::array verification_inputs{runtime::TensorInput{
          .name = "features",
          .values = std::span(input.const_data_ptr<float>(),
                              static_cast<std::size_t>(input.numel())),
          .shape = std::span(input.sizes().data(), input.sizes().size())}};
      const std::array verification_outputs{runtime::TensorInput{
          .name = "logits",
          .values = std::span(expected.const_data_ptr<float>(),
                              static_cast<std::size_t>(expected.numel())),
          .shape =
              std::span(expected.sizes().data(), expected.sizes().size())}};
      const std::array cases{
          runtime::VerificationCase{.id = "consumer",
                                    .inputs = verification_inputs,
                                    .expected = verification_outputs}};
      const auto verification = runtime::verify_session(
          session, cases, {.reference = "installed-user-network-forward"});
      Require(runtime::verification_passed(verification) &&
                  !runtime::serialize_verification_report(verification).empty(),
              "installed numerical verification report");
    }
    export_options.input_names = {"features", "offset"};
    export_options.output_names = {"shifted", "scaled"};
    export_options.dynamic_axes = {{"features", {{0, "batch"}}},
                                   {"offset", {{0, "batch"}}},
                                   {"shifted", {{0, "batch"}}},
                                   {"scaled", {{0, "batch"}}}};
    const std::vector<torch::Tensor> examples{Input(2), torch::ones({2, 3})};
    const auto multi_artifact = exporter::export_module(
        *model,
        [&](std::span<const torch::Tensor> inputs) {
          const auto logits = model->forward(inputs[0]);
          return std::vector<torch::Tensor>{logits + inputs[1],
                                            logits * inputs[1]};
        },
        examples, export_options);
    runtime::Session multi(multi_artifact.onnx, multi_artifact.digest,
                           session_options);
    Require(multi.inputs().size() == 2U && multi.outputs().size() == 2U,
            "installed multi-I/O metadata");
    for (const auto batch : {1, 2, 5}) {
      const auto features = Input(batch);
      const auto offset =
          (torch::arange(batch * 3, torch::kFloat32).reshape({batch, 3}) /
               4.0F +
           1.0F)
              .contiguous();
      const auto logits = model->forward(features);
      const std::array<runtime::TensorInput, 2> inputs{
          runtime::TensorInput{
              .name = "offset",
              .values = std::span(offset.const_data_ptr<float>(),
                                  static_cast<std::size_t>(offset.numel())),
              .shape = std::span(offset.sizes().data(), offset.sizes().size())},
          runtime::TensorInput{
              .name = "features",
              .values = std::span(features.const_data_ptr<float>(),
                                  static_cast<std::size_t>(features.numel())),
              .shape =
                  std::span(features.sizes().data(), features.sizes().size())}};
      const auto outputs = multi.run(inputs);
      Require(outputs.size() == 2U && outputs[0].name == "shifted" &&
                  outputs[1].name == "scaled",
              "installed multi-I/O output identity");
      Compare(outputs[0], (logits + offset).contiguous());
      Compare(outputs[1], (logits * offset).contiguous());
    }
    CheckFlatten();
    CheckShapeOperators();
    CheckReductions();
    CheckActivations();
    CheckArithmetic();
    CheckUnaryOperators();
    std::cout
        << "Native deployment consumer: single/multi-I/O dynamic batches "
           "1,2,5 and 40 exact suffix/full flatten cases match on CPU; "
           "27 exact squeeze/permutation/scalar cases and 3 dynamic squeeze "
           "rejections; "
           "25 exact reduction cases including Int32 promotion overflow "
           "boundary; "
           "38 activation/dtype/scalar and 111 arithmetic/promotion cases "
           "with independent references; 150 unary dtype/integer cases "
           "and one Double Erf CPU refusal with independent references "
           "and 151 native forwards\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Deployment install consumer failed: " << error.what() << '\n';
    return 1;
  }
}
