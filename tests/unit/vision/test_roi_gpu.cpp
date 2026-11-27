// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "../../../src/jyppx/liuding/vision/ops/roi_gpu.hpp"

#include <jyppx/liuding/core/runtime.hpp>

#include <ATen/Context.h>
#ifdef JYPPX_VISION_GPU_HIP
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>
#else
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif
#include <torch/cuda.h>
#include <torch/nn/functional/vision.h>

#include <algorithm>
#include <gtest/gtest.h>
#include <limits>

namespace jyppx::liuding::vision_ops::tests {
namespace {
using I = std::int64_t;

// Independent per-ROI ATen oracle: no calls to the packed/native
// implementation.
Tensor Oracle(const Tensor& input, const Tensor& rows,
              const RoiAlignOptions& o) {
  const auto values = rows.accessor<double, 2>();
  const auto ph = o.output_size[0], pw = o.output_size[1], r = o.sampling_ratio;
  std::vector<Tensor> outputs;
  for (I i = 0; i < rows.size(0); ++i) {
    const double offset = o.aligned ? .5 : 0.;
    const double x = values[i][1] * o.spatial_scale - offset;
    const double y = values[i][2] * o.spatial_scale - offset;
    const double w = (std::max)(values[i][3] * o.spatial_scale - offset - x,
                                o.aligned ? 0. : 1.);
    const double h = (std::max)(values[i][4] * o.spatial_scale - offset - y,
                                o.aligned ? 0. : 1.);
    auto options = input.options().requires_grad(false);
    auto xx = ((torch::arange(pw * r, options) + .5) * (w / (pw * r)) + x)
                  .view({1, -1})
                  .expand({ph * r, pw * r});
    auto yy = ((torch::arange(ph * r, options) + .5) * (h / (ph * r)) + y)
                  .view({-1, 1})
                  .expand({ph * r, pw * r});
    auto valid =
        (xx >= -1) & (xx <= input.size(3)) & (yy >= -1) & (yy <= input.size(2));
    auto grid = torch::stack({(xx + .5) * (2. / input.size(3)) - 1,
                              (yy + .5) * (2. / input.size(2)) - 1},
                             -1)
                    .unsqueeze(0);
    auto sampled = torch::nn::functional::grid_sample(
        input.narrow(0, static_cast<I>(values[i][0]), 1), grid,
        torch::nn::functional::GridSampleFuncOptions()
            .mode(torch::kBilinear)
            .padding_mode(torch::kBorder)
            .align_corners(false));
    outputs.push_back((sampled * valid.to(input.scalar_type()))
                          .view({input.size(1), ph, r, pw, r})
                          .mean({2, 4}));
  }
  return torch::stack(outputs);
}

class NativeRoiGpu : public ::testing::Test {
 protected:
  void SetUp() override {
    torch::set_num_threads(1);
    ASSERT_TRUE(torch::cuda::is_available());
  }
};

TEST_F(NativeRoiGpu, FullOutputsAndFeatureGradientsMatchIndependentOracle) {
  const Device device(torch::kCUDA, 0);
  const auto rows = torch::tensor({{1., -4., -5., 2., 3.},
                                   {0., 1., 2., 10., 8.},
                                   {1., 0., 0., 0., 0.},
                                   {0., 20., 30., 21., 31.},
                                   {1., -1., -1., 11., 9.},
                                   {0., 3., 2., 3., 2.}},
                                  torch::kFloat64);
  // Exercise allocation, launches and backward on a non-default stream.
  c10::cuda::CUDAStreamGuard stream(c10::cuda::getStreamFromPool(false, 0));
  for (auto dtype : {torch::kFloat32, torch::kFloat64})
    for (int layout : {0, 1, 2})
      for (bool aligned : {false, true})
        for (I ratio : {I{1}, I{2}, I{8}}) {
          SCOPED_TRACE(::testing::Message() << dtype << ' ' << layout << ' '
                                            << aligned << ' ' << ratio);
          auto image =
              torch::sin(torch::arange(2 * 3 * 9 * 22,
                                       torch::TensorOptions().dtype(dtype)))
                  .view({2, 3, 9, 22})
                  .to(device);
          if (layout == 0) image = image.narrow(3, 1, 11).contiguous();
          if (layout == 1) image = image.slice(3, 1, 22, 2);
          if (layout == 2)
            image =
                image.narrow(0, 0, 1).narrow(3, 1, 11).expand({2, 3, 9, 11});
          auto a = image.detach().requires_grad_(true);
          auto b = image.clone().detach().requires_grad_(true);
          RoiAlignOptions o;
          o.output_size = {2, 3};
          o.sampling_ratio = ratio;
          o.aligned = aligned;
          o.spatial_scale = aligned ? .5 : 2.;
          ASSERT_TRUE(detail::try_roi_align_gpu(a, rows, o).defined());
          auto rois = rows.to(device, dtype)
                          .t()
                          .contiguous()
                          .t()
                          .detach()
                          .requires_grad_(true);
          auto actual = roi_align(a, rois, o), expected = Oracle(b, rows, o);
          ASSERT_NE(actual.grad_fn()->name().find("NativeRoi"),
                    std::string::npos);
          ASSERT_TRUE(torch::isfinite(actual).all().item<bool>());
          EXPECT_TRUE(torch::allclose(actual, expected, 2e-4, 2e-5));
          auto adjoint =
              torch::linspace(-.03, .07, actual.numel(), actual.options())
                  .reshape_as(actual)
                  .transpose(2, 3)
                  .contiguous()
                  .transpose(2, 3);
          actual.backward(adjoint);
          expected.backward(adjoint);
          ASSERT_TRUE(torch::isfinite(a.grad()).all().item<bool>());
          EXPECT_TRUE(torch::allclose(a.grad(), b.grad(), 2e-4, 2e-5));
          EXPECT_FALSE(rois.grad().defined());
        }
}

TEST_F(NativeRoiGpu, NonfiniteFeaturesAndFallbacksPreserveSemantics) {
  const Device device(torch::kCUDA, 0);
  auto rows = torch::tensor(
      {{0., -4., -4., -2., -2.}, {0., 4., 4., 6., 6.}, {0., 0., 0., 4., 4.}},
      torch::kFloat64);
  for (double value : {std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity()}) {
    auto image =
        torch::ones({1, 2, 4, 4}, torch::TensorOptions().device(device));
    image.index_put_({0, 0, 0, 0}, value);
    image.index_put_({0, 1, 3, 3}, value);
    ASSERT_TRUE(detail::try_roi_align_gpu(image, rows, {}).defined());
    EXPECT_TRUE(torch::allclose(roi_align(image, rows.to(device), {}),
                                Oracle(image, rows, {}), 2e-4, 2e-5, true));
  }
  auto image = torch::ones({1, 1, 4, 4}, torch::TensorOptions().device(device));
  auto huge =
      torch::tensor({{0., 1e12, 1e12, 1e12 + 1., 1e12 + 1.}}, torch::kFloat64);
  EXPECT_FALSE(detail::try_roi_align_gpu(image, huge, {}).defined());
  EXPECT_TRUE(
      torch::equal(roi_align(image, huge.to(device)), Oracle(image, huge, {})));
  EXPECT_FALSE(detail::try_roi_align_gpu(image.cpu(), rows, {}).defined());
  for (auto dtype : {torch::kFloat16, torch::kBFloat16})
    EXPECT_FALSE(
        detail::try_roi_align_gpu(image.to(dtype), rows, {}).defined());
}

TEST_F(NativeRoiGpu, AutogradRejectsMutationDeterminismAndHigherDerivatives) {
  const Device device(torch::kCUDA, 0);
  auto rois = torch::tensor({{0., 0., 0., 3., 3.}}).to(device);
  for (int mode : {0, 1, 2}) {
    auto image =
        torch::ones({1, 1, 4, 4}, torch::TensorOptions().device(device))
            .requires_grad_(true);
    auto output = roi_align(image, rois);
    if (mode == 0) {
      torch::NoGradGuard guard;
      image.add_(1);
    }
    const auto deterministic = at::globalContext().deterministicAlgorithms();
    const auto warn = at::globalContext().deterministicAlgorithmsWarnOnly();
    if (mode == 1) at::globalContext().setDeterministicAlgorithms(true, false);
    if (mode == 2)
      EXPECT_THROW(output.sum().backward({}, false, true), c10::Error);
    else
      EXPECT_THROW(output.sum().backward(), c10::Error);
    at::globalContext().setDeterministicAlgorithms(deterministic, warn);
  }
}
}  // namespace
}  // namespace jyppx::liuding::vision_ops::tests
