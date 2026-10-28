// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

#include "../../../src/jyppx/liuding/models/position_interpolation.hpp"

#include <jyppx/liuding/models/dino.hpp>

#include <torch/torch.h>

#include <array>
#include <gtest/gtest.h>
#include <utility>

namespace {
class DeterministicPositions final {
 public:
  DeterministicPositions()
      : algorithms_(at::globalContext().deterministicAlgorithms()),
        warn_only_(at::globalContext().deterministicAlgorithmsWarnOnly()),
        cudnn_(at::globalContext().deterministicCuDNN()),
        benchmark_(at::globalContext().benchmarkCuDNN()),
        tf32_(at::globalContext().allowTF32CuBLAS()) {
    at::globalContext().setDeterministicAlgorithms(true, false);
    at::globalContext().setDeterministicCuDNN(true);
    at::globalContext().setBenchmarkCuDNN(false);
    at::globalContext().setAllowTF32CuBLAS(false);
  }
  ~DeterministicPositions() {
    at::globalContext().setDeterministicAlgorithms(algorithms_, warn_only_);
    at::globalContext().setDeterministicCuDNN(cudnn_);
    at::globalContext().setBenchmarkCuDNN(benchmark_);
    at::globalContext().setAllowTF32CuBLAS(tf32_);
  }

 private:
  bool algorithms_, warn_only_, cudnn_, benchmark_, tf32_;
};

void ComparePositionInterpolation(bool cuda) {
  const DeterministicPositions deterministic;
  for (const auto antialias : {false, true}) {
    for (const auto& [height, width] :
         std::array<std::pair<std::int64_t, std::int64_t>, 5>{
             {{2, 3}, {16, 16}, {39, 43}, {1, 1}, {37, 16}}}) {
      SCOPED_TRACE(::testing::Message()
                   << "antialias=" << antialias << " grid=" << height << 'x'
                   << width << " cuda=" << cuda);
      auto reference_input = torch::linspace(-1.0, 1.0, 2 * 4 * 37 * 37)
                                 .reshape({2, 4, 37, 37})
                                 .set_requires_grad(true);
      auto actual_input = reference_input.detach().clone();
      if (cuda) actual_input = actual_input.to(torch::kCUDA);
      actual_input.set_requires_grad(true);
      const auto expected = torch::nn::functional::interpolate(
          reference_input, torch::nn::functional::InterpolateFuncOptions()
                               .size(std::vector<std::int64_t>{height, width})
                               .mode(torch::kBicubic)
                               .align_corners(false)
                               .antialias(antialias));
      const auto actual =
          jyppx::liuding::models::detail::SeparableBicubicPositions(
              actual_input, height, width, antialias);
      EXPECT_TRUE(torch::allclose(actual.cpu(), expected, 2e-5, 2e-6));
      const auto upstream = torch::linspace(-0.7, 0.3, expected.numel())
                                .reshape(expected.sizes());
      expected.backward(upstream);
      actual.backward(upstream.to(actual.device()));
      EXPECT_TRUE(torch::allclose(actual_input.grad().cpu(),
                                  reference_input.grad(), 2e-5, 3e-6));
    }
  }
}

TEST(ModelsPositions, BicubicValuesAndGradientsMatchCpuReference) {
  ComparePositionInterpolation(false);
}

TEST(ModelsCuda, BicubicPositionValuesAndGradientsMatchCpuReference) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA unavailable";
  ComparePositionInterpolation(true);
}

TEST(ModelsCuda, BicubicPositionsRetainFloat32InsideAutocast) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA unavailable";
  const auto previous = at::autocast::is_autocast_enabled(at::kCUDA);
  struct Restore {
    bool enabled;
    ~Restore() { at::autocast::set_autocast_enabled(at::kCUDA, enabled); }
  } restore{previous};
  at::autocast::set_autocast_enabled(at::kCUDA, true);
  auto input = torch::linspace(-1.0, 1.0, 4 * 37 * 37)
                   .reshape({1, 4, 37, 37})
                   .to(torch::kCUDA)
                   .set_requires_grad(true);
  const auto actual = jyppx::liuding::models::detail::SeparableBicubicPositions(
      input, 16, 16, false);
  EXPECT_EQ(actual.scalar_type(), torch::kFloat32);
  at::autocast::set_autocast_enabled(at::kCUDA, false);
  const auto expected =
      jyppx::liuding::models::detail::SeparableBicubicPositions(input, 16, 16,
                                                                false);
  EXPECT_TRUE(torch::equal(actual, expected));
  actual.sum().backward();
  EXPECT_TRUE(torch::isfinite(input.grad()).all().item<bool>());
}

TEST(ModelsCuda, DinoV2ResizedPositionsHaveRepeatableGradients) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA unavailable";
  const DeterministicPositions deterministic;
  torch::manual_seed(238);
  const auto input = torch::linspace(-0.5, 0.5, 3 * 28 * 42)
                         .reshape({1, 3, 28, 42})
                         .to(torch::kCUDA);
  for (const auto antialias : {false, true}) {
    auto model = jyppx::liuding::models::dinov2_vits14(
        {.num_classes = 3, .position_antialias = antialias});
    model->to(torch::kCUDA);
    model->forward(input).square().mean().backward();
    const auto gradient = model->named_parameters()["pos_embed"].grad().clone();
    ASSERT_TRUE(torch::isfinite(gradient).all().item<bool>());
    EXPECT_GT(gradient.slice(1, 1).abs().sum().item<double>(), 0.0);
    model->zero_grad();
    model->forward(input).square().mean().backward();
    EXPECT_TRUE(
        torch::equal(gradient, model->named_parameters()["pos_embed"].grad()));
  }
}
}  // namespace
