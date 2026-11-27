// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/core/runtime.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/vision/ops/deformable.hpp>

#include <ATen/Context.h>
#ifdef JYPPX_VISION_GPU_HIP
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>
#else
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/cuda.h>
#include <torch/nn/functional/vision.h>

#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace jyppx::liuding::vision_ops::tests {
namespace {
using I = std::int64_t;
using Shapes = std::vector<std::array<I, 2>>;

// Independent per-level ATen oracle, including the normalized-grid operations.
Tensor Oracle(const Tensor& value, const Shapes& shapes,
              const Tensor& locations, const Tensor& weights) {
  std::vector<Tensor> sampled;
  I offset = 0;
  for (I level = 0; level < static_cast<I>(shapes.size()); ++level) {
    const auto [h, w] = shapes[level];
    auto feature =
        value.narrow(1, offset, h * w)
            .flatten(2)
            .permute({0, 2, 1})
            .reshape({value.size(0) * value.size(2), value.size(3), h, w});
    auto grid = (locations.select(3, level) * 2 - 1)
                    .permute({0, 2, 1, 3, 4})
                    .flatten(0, 1);
    sampled.push_back(torch::nn::functional::grid_sample(
        feature, grid,
        torch::nn::functional::GridSampleFuncOptions()
            .mode(torch::kBilinear)
            .padding_mode(torch::kZeros)
            .align_corners(false)));
    offset += h * w;
  }
  auto packed = torch::stack(sampled, -2).flatten(-2);
  auto factors =
      weights.permute({0, 2, 1, 3, 4})
          .reshape({value.size(0) * value.size(2), 1, locations.size(1), -1});
  return (packed * factors)
      .sum(-1)
      .reshape(
          {value.size(0), value.size(2) * value.size(3), locations.size(1)})
      .permute({0, 2, 1});
}

class NativeDeformGpu : public ::testing::Test {
 protected:
  void SetUp() override {
    torch::set_num_threads(1);
    ASSERT_TRUE(torch::cuda::is_available());
  }

  void Compare(const Tensor& value, const Shapes& shapes,
               const Tensor& locations, const Tensor& weights,
               bool native = true) {
    std::vector<Tensor> inputs{value.detach().requires_grad_(true),
                               locations.detach().requires_grad_(true),
                               weights.detach().requires_grad_(true)},
        reference;
    for (const auto& input : inputs)
      reference.push_back(input.clone().detach().requires_grad_(true));
    auto output = multiscale_deformable_attention(inputs[0], shapes, inputs[1],
                                                  inputs[2]);
    ASSERT_EQ(
        output.grad_fn()->name().find("NativeDeform") != std::string::npos,
        native);
    auto expected = Oracle(reference[0], shapes, reference[1], reference[2]);
    ASSERT_TRUE(torch::allclose(output, expected, 2e-4, 2e-5, true));
    auto adjoint =
        (torch::cos(torch::arange(output.numel(), output.options())) * .07)
            .reshape_as(output)
            .transpose(1, 2)
            .contiguous()
            .transpose(1, 2);
    output.backward(adjoint);
    expected.backward(adjoint);
    for (int i = 0; i < 3; ++i) {
      SCOPED_TRACE(i);
      EXPECT_TRUE(torch::allclose(inputs[i].grad(), reference[i].grad(), 2e-4,
                                  2e-5, true));
    }
  }
};

TEST_F(NativeDeformGpu, FullOutputsAndAllAdjointsMatchIndependentOracle) {
  c10::cuda::CUDAStreamGuard stream(c10::cuda::getStreamFromPool(false, 0));
  const Shapes shapes{{3, 4}, {1, 3}, {2, 1}};
  for (auto dtype : {torch::kFloat32, torch::kFloat64})
    for (int layout : {0, 1, 2})
      for (int channels : {1, 3, 32}) {
        SCOPED_TRACE(::testing::Message()
                     << int(dtype) << '/' << layout << '/' << channels);
        auto options = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);
        auto value =
            torch::sin(torch::arange(2 * 17 * 2 * channels * 2, options))
                .view({2, 17, 2, channels * 2})
                .slice(3, 1, channels * 2, 2);
        auto locations =
            (torch::sin(torch::arange(2 * 5 * 2 * 3 * 4 * 4, options)) * .8 +
             .5)
                .view({2, 5, 2, 3, 4, 4})
                .slice(5, 1, 4, 2);
        auto weights = torch::cos(torch::arange(2 * 5 * 2 * 3 * 8, options))
                           .view({2, 5, 2, 3, 8})
                           .slice(4, 1, 8, 2);
        if (layout == 0) {
          value = value.contiguous();
          locations = locations.contiguous();
          weights = weights.contiguous();
        } else if (layout == 2) {
          value = value.narrow(0, 0, 1).expand({2, 17, 2, channels});
          locations = locations.narrow(0, 0, 1).expand({2, 5, 2, 3, 4, 2});
          weights = weights.narrow(0, 0, 1).expand({2, 5, 2, 3, 4});
        }
        Compare(value, shapes, locations, weights);
      }
}

TEST_F(NativeDeformGpu, PaddingNonfiniteInputsAndBoundedFallbacks) {
  const Shapes shapes{{3, 4}, {1, 3}, {2, 1}};
  auto options = torch::TensorOptions().device(torch::kCUDA);
  auto value = torch::ones({2, 17, 2, 3}, options);
  auto locations = torch::full({2, 5, 2, 3, 4, 2}, .5, options);
  auto weights = torch::full({2, 5, 2, 3, 4}, .1, options);
  for (double point :
       {-.5, 0., .5, 1., 1.5, 1e30, std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()}) {
    SCOPED_TRACE(point);
    Compare(value, shapes, torch::full_like(locations, point), weights);
  }
  for (double nf : {std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::infinity(),
                    -std::numeric_limits<double>::infinity()})
    for (int input : {0, 1, 2}) {
      SCOPED_TRACE(::testing::Message() << "nonfinite " << nf << '/' << input);
      auto a = value.clone(), b = locations.clone(), c = weights.clone();
      if (input == 0) a.index_put_({0, 0, 0, 0}, nf);
      if (input == 1) b.index_put_({0, 0, 0, 0, 0, 0}, nf);
      if (input == 2) c.index_put_({0, 0, 0, 0, 0}, nf);
      Compare(a, shapes, b, c);
    }
  {
    SCOPED_TRACE("CPU fallback");
    Compare(value.cpu(), shapes, locations.cpu(), weights.cpu(), false);
  }
  {
    SCOPED_TRACE("FP16 fallback");
    // One contribution per feature avoids legacy grid_sample's overlapping
    // half-atomic roundoff while checking the unchanged fallback and adjoints.
    Compare(value.to(torch::kFloat16), shapes,
            locations.narrow(1, 0, 1).narrow(4, 0, 1).to(torch::kFloat16),
            weights.narrow(1, 0, 1).narrow(4, 0, 1).to(torch::kFloat16), false);
  }
  {
    SCOPED_TRACE("negative value fallback");
    Compare(torch::_neg_view(value), shapes, locations, weights, false);
  }
  {
    SCOPED_TRACE("negative locations fallback");
    Compare(value, shapes, torch::_neg_view(locations), weights, false);
  }
  {
    SCOPED_TRACE("negative weights fallback");
    Compare(value, shapes, locations, torch::_neg_view(weights), false);
  }
  const Shapes many(17, {1, 1});
  Compare(value, many, torch::full({2, 5, 2, 17, 4, 2}, .5, options),
          torch::full({2, 5, 2, 17, 4}, .1, options), false);
  const Shapes maximum(16, {1, 1});
  Compare(value.narrow(1, 0, 16), maximum,
          torch::full({2, 5, 2, 16, 4, 2}, .5, options),
          torch::full({2, 5, 2, 16, 4}, .1, options));
  EXPECT_THROW(
      multiscale_deformable_attention(value, shapes, locations, weights, 1),
      ResourceExhausted);
}

TEST_F(NativeDeformGpu, RejectsMutationsHigherDerivativesAndStrictDeterminism) {
  const Shapes shapes{{2, 2}};
  auto options =
      torch::TensorOptions().device(torch::kCUDA).requires_grad(true);
  for (int mutation = 0; mutation < 3; ++mutation) {
    std::vector<Tensor> inputs{torch::ones({1, 4, 1, 3}, options),
                               torch::full({1, 2, 1, 1, 2, 2}, .5, options),
                               torch::ones({1, 2, 1, 1, 2}, options)};
    auto result = multiscale_deformable_attention(inputs[0], shapes, inputs[1],
                                                  inputs[2]);
    {
      torch::NoGradGuard guard;
      inputs[mutation].add_(1);
    }
    EXPECT_THROW(result.sum().backward(), c10::Error);
  }
  auto value = torch::ones({1, 4, 1, 3}, options);
  auto locations = torch::full({1, 2, 1, 1, 2, 2}, .5, options);
  auto weights = torch::ones({1, 2, 1, 1, 2}, options);
  auto result =
      multiscale_deformable_attention(value, shapes, locations, weights);
  EXPECT_THROW(result.sum().backward({}, false, true), c10::Error);
  const auto was_enabled = at::globalContext().deterministicAlgorithms();
  const auto was_warn_only =
      at::globalContext().deterministicAlgorithmsWarnOnly();
  at::globalContext().setDeterministicAlgorithms(true, false);
  try {
    auto strict =
        multiscale_deformable_attention(value, shapes, locations, weights);
    EXPECT_THROW(strict.sum().backward(), c10::Error);
  } catch (...) {
    at::globalContext().setDeterministicAlgorithms(was_enabled, was_warn_only);
    throw;
  }
  at::globalContext().setDeterministicAlgorithms(was_enabled, was_warn_only);
}
TEST_F(NativeDeformGpu, TracingReplaysAllThreeRuntimeInputs) {
  torch::NoGradGuard no_grad;
  const Shapes shapes{{2, 3}, {3, 2}};
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto value = torch::arange(12, options).view({1, 12, 1, 1});
  auto locations = torch::full({1, 2, 1, 2, 2, 2}, .5, options);
  auto weights = torch::full({1, 2, 1, 2, 2}, .25, options);
  const auto traced = torch::jit::tracer::trace(
      {value, locations, weights},
      [&](torch::jit::Stack inputs) {
        return torch::jit::Stack{multiscale_deformable_attention(
            inputs[0].toTensor(), shapes, inputs[1].toTensor(),
            inputs[2].toTensor())};
      },
      [](const auto&) { return std::string(); }, true, false);
  torch::jit::GraphExecutor executor(traced.first->graph,
                                     "multiscale_sampling");
  auto next_value = value * 1.5 + 7;
  auto next_locations = locations * .75;
  auto next_weights = weights * .5;
  torch::jit::Stack inputs{next_value, next_locations, next_weights};
  executor.run(inputs);
  ASSERT_EQ(inputs.size(), 1);
  EXPECT_TRUE(torch::allclose(
      inputs[0].toTensor(),
      Oracle(next_value, shapes, next_locations, next_weights), 2e-4, 2e-5));
  EXPECT_FALSE(torch::allclose(inputs[0].toTensor(),
                               Oracle(value, shapes, locations, weights), 2e-4,
                               2e-5));
}
}  // namespace
}  // namespace jyppx::liuding::vision_ops::tests
