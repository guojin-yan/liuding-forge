// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/deployment/onnx_export/exporter.hpp>
#include <jyppx/liuding/deployment/onnx_runtime/session.hpp>

#include <torch/torch.h>

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
    }
    std::cout
        << "Native deployment consumer: dynamic batches 1,2,5 match on CPU\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Deployment install consumer failed: " << error.what() << '\n';
    return 1;
  }
}
