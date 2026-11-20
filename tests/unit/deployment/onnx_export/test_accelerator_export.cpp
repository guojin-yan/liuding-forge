// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/core/device.hpp>
#include <jyppx/liuding/deployment/onnx_export/exporter.hpp>
#include <jyppx/liuding/deployment/onnx_runtime/session.hpp>

#include <torch/torch.h>

#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

namespace {
namespace j = jyppx::liuding;

class AcceleratorOnnxExport : public testing::TestWithParam<torch::ScalarType> {
};

TEST_P(AcceleratorOnnxExport, TrainedGpuStateExportsThroughIndependentCpuCopy) {
  // This is an opt-in hardware gate: a missing GPU must fail qualification.
  ASSERT_TRUE(j::cuda_is_available());
  const auto dtype = GetParam();
  const torch::Device device(torch::kCUDA, 0);
  torch::nn::Linear model(3, 2);
  model->to(device, dtype);
  {
    torch::NoGradGuard guard;
    model->weight.copy_(
        torch::tensor({1.F, -.5F, .25F, -.25F, .5F, 1.F}).reshape({2, 3}));
    model->bias.copy_(torch::tensor({.25F, -.5F}));
  }
  const auto before = model->weight.detach().clone();
  torch::optim::SGD optimizer(model->parameters(), .125);
  const auto input = torch::tensor({1.F, 2.F, 3.F, -1.F, .5F, 2.F})
                         .reshape({2, 3})
                         .to(device, dtype);
  model->forward(input).to(torch::kFloat32).square().mean().backward();
  ASSERT_TRUE(torch::isfinite(model->weight.grad()).all().item<bool>());
  optimizer.step();
  ASSERT_FALSE(torch::equal(before, model->weight));
  const auto trained_weight =
      model->weight.detach().to(torch::kCPU, torch::kFloat32);
  const auto trained_bias =
      model->bias.detach().to(torch::kCPU, torch::kFloat32);
  const auto gpu_gradient = model->weight.grad().clone();

  // The public serializer requires explicit CPU state, including on ROCm.
  EXPECT_THROW(j::onnx_export::export_model(model, torch::zeros({2, 3})),
               j::CompatibilityError);
  torch::nn::Linear cpu(3, 2);
  cpu->eval();
  {
    torch::NoGradGuard guard;
    cpu->weight.copy_(trained_weight);
    cpu->bias.copy_(trained_bias);
  }
  const auto artifact = j::onnx_export::export_model(cpu, torch::zeros({2, 3}));
  j::onnx_runtime::Session session(artifact.onnx, artifact.digest);
  const auto* weights = trained_weight.const_data_ptr<float>();
  const auto* biases = trained_bias.const_data_ptr<float>();
  for (const std::int64_t batch : {1, 2, 5}) {
    std::vector<float> values(static_cast<std::size_t>(batch * 3));
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] = static_cast<float>(static_cast<int>(i % 9) - 4) / 4.F;
    const std::array<std::int64_t, 2> shape{batch, 3};
    const auto actual = session.run(values, shape);
    ASSERT_EQ(actual.shape, (std::vector<std::int64_t>{batch, 2}));
    for (std::int64_t row = 0; row < batch; ++row) {
      for (int output = 0; output < 2; ++output) {
        // Scalar reference uses the actual trained state without Torch matmul.
        float expected = biases[output];
        for (int col = 0; col < 3; ++col)
          expected += weights[output * 3 + col] * values[row * 3 + col];
        EXPECT_NEAR(actual.values[row * 2 + output], expected, 1e-6F);
      }
    }
  }
  EXPECT_TRUE(model->is_training());
  EXPECT_TRUE(model->weight.requires_grad());
  EXPECT_EQ(model->weight.device(), device);
  EXPECT_EQ(model->weight.scalar_type(), dtype);
  EXPECT_TRUE(torch::equal(
      model->weight.detach().to(torch::kCPU, torch::kFloat32), trained_weight));
  EXPECT_TRUE(torch::equal(model->weight.grad(), gpu_gradient));
}

INSTANTIATE_TEST_SUITE_P(
    Precision, AcceleratorOnnxExport,
    testing::Values(torch::kFloat32, torch::kFloat16, torch::kBFloat16),
    [](const testing::TestParamInfo<torch::ScalarType>& info) {
      if (info.param == torch::kFloat16) return "Float16";
      if (info.param == torch::kBFloat16) return "BFloat16";
      return "Float32";
    });
}  // namespace
