// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

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
        const auto actual = session.run(
            std::span(input.const_data_ptr<float>(),
                      static_cast<std::size_t>(input.numel())),
            std::span(input.sizes().data(), input.sizes().size()));
        const std::vector<std::int64_t> expected_shape =
            axis == 1 ? std::vector<std::int64_t>{batch, 24}
            : axis == 2 ? std::vector<std::int64_t>{batch, 2, 12}
                        : std::vector<std::int64_t>{batch, 2, 3, 4};
        Require(actual.shape == expected_shape &&
                    actual.values.size() ==
                        static_cast<std::size_t>(batch * 24),
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
      const auto actual = session.run(
          std::span(input.const_data_ptr<float>(),
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
  const auto actual = session.run(
      std::span(input.const_data_ptr<float>(),
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
    const std::array examples{
        at::arange(6, at::kFloat).reshape({2, 1, 3, 1})};
    const auto artifact = exporter::export_module(module, forward, examples);
    runtime::Session session(artifact.onnx, artifact.digest);
    const auto normalized = axis < 0 ? axis + 4 : axis;
    for (const auto batch : {1, 2, 5}) {
      const auto input =
          at::arange(batch * 3, at::kFloat).reshape({batch, 1, 3, 1});
      const std::vector<std::int64_t> shape =
          normalized == 1 ? std::vector<std::int64_t>{batch, 3, 1}
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
    const std::array examples{
        at::arange(12, at::kFloat).reshape({2, 2, 3})};
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
          const auto output = mode == 0 ? at::transpose(scalar, 0, -1)
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
              const auto axes = empty
                                    ? at::OptionalIntArrayRef(dimensions)
                                    : at::OptionalIntArrayRef{};
              const auto output = mean
                                      ? at::mean(inputs.front(), axes, keepdim)
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
            const auto output = dimensioned
                                    ? (mean ? at::mean(inputs.front(), {1},
                                                       true, at::kDouble)
                                            : at::sum(inputs.front(), {1},
                                                      true, at::kDouble))
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
        const auto value = mean ? 1.0F / 3.0F
                               : static_cast<float>(dimensioned ? 1 : batch);
        CheckExact(session, input, shape,
                   std::vector<float>(dimensioned ? batch : 1, value));
      }
    }
  }
  for (const auto mode : {0, 1, 2}) {
    const exporter::ForwardFunction forward =
        [mode](std::span<const at::Tensor> inputs) {
          const auto output =
              mode == 0 ? at::sum(inputs.front().to(at::kInt), {1}, true)
              : mode == 1 ? at::sum(inputs.front(), {1}, true, at::kLong)
                          : at::sum(inputs.front(), {1}, true, at::kInt);
          return std::vector<at::Tensor>{output.to(at::kFloat)};
        };
    // Every value fits int32, but the first row's exact integer sum does not.
    const auto row = mode == 0
                         ? at::tensor({2147483520.0F, 256.0F, 0.0F})
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
    std::cout << "Native deployment consumer: single/multi-I/O dynamic batches "
                 "1,2,5 and 40 exact suffix/full flatten cases match on CPU; "
                 "27 exact squeeze/permutation/scalar cases and 3 dynamic squeeze rejections; "
                 "25 exact reduction cases including Int32 promotion overflow boundary\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Deployment install consumer failed: " << error.what() << '\n';
    return 1;
  }
}
