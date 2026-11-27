// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
#include "roi_gpu.hpp"

#include "roi_gpu_launch.hpp"

#include <ATen/Context.h>
#ifdef JYPPX_VISION_GPU_HIP
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>
#else
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif
#include <torch/csrc/autograd/custom_function.h>

#include <algorithm>
#include <cmath>

namespace jyppx::liuding::vision_ops::detail {
namespace {
using I = std::int64_t;

class NativeRoi : public torch::autograd::Function<NativeRoi> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* context, Tensor input,
                        Tensor parameters, Tensor batches, I ph, I pw,
                        I ratio) {
    c10::cuda::CUDAGuard guard(input.device());
    auto output =
        torch::empty({batches.numel(), input.size(1), ph, pw}, input.options());
    RoiGpuShape shape{batches.numel(),
                      input.size(1),
                      input.size(2),
                      input.size(3),
                      ph,
                      pw,
                      ratio,
                      input.stride(0),
                      input.stride(1),
                      input.stride(2),
                      input.stride(3)};
    auto status = launch_roi_gpu(
        input.const_data_ptr(), parameters.const_data_ptr<double>(),
        batches.const_data_ptr<I>(), output.mutable_data_ptr(), shape,
        input.scalar_type() == torch::kFloat64, false,
        c10::cuda::getCurrentCUDAStream(input.get_device()).stream());
    TORCH_CHECK(status == 0, "Native ROIAlign GPU launch failed: ", status);
    // Retain the feature version check as well as the tensors used by backward.
    context->save_for_backward({parameters, batches, input});
    context->saved_data["shape"] = input.sizes().vec();
    context->saved_data["ph"] = ph;
    context->saved_data["pw"] = pw;
    context->saved_data["ratio"] = ratio;
    return output;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* context,
      torch::autograd::variable_list gradients) {
    TORCH_CHECK(
        !torch::GradMode::is_enabled(),
        "Native ROIAlign GPU backward supports first-order gradients only");
    auto values = context->get_saved_variables();
    auto grad = gradients[0].contiguous();
    c10::cuda::CUDAGuard guard(grad.device());
    at::globalContext().alertNotDeterministic("liuding_roi_align_gpu_backward");
    auto dims = context->saved_data["shape"].toIntVector();
    auto output = torch::zeros(dims, grad.options());
    RoiGpuShape shape{values[1].numel(),
                      dims[1],
                      dims[2],
                      dims[3],
                      context->saved_data["ph"].toInt(),
                      context->saved_data["pw"].toInt(),
                      context->saved_data["ratio"].toInt(),
                      0,
                      0,
                      0,
                      0};
    auto status = launch_roi_gpu(
        grad.const_data_ptr(), values[0].const_data_ptr<double>(),
        values[1].const_data_ptr<I>(), output.mutable_data_ptr(), shape,
        grad.scalar_type() == torch::kFloat64, true,
        c10::cuda::getCurrentCUDAStream(grad.get_device()).stream());
    TORCH_CHECK(status == 0,
                "Native ROIAlign GPU backward launch failed: ", status);
    return {output, Tensor{}, Tensor{}, Tensor{}, Tensor{}, Tensor{}};
  }
};
}  // namespace

Tensor try_roi_align_gpu(const Tensor& features, const Tensor& rows,
                         const RoiAlignOptions& options) {
  if (!features.is_cuda() || features.is_neg() || features.is_conj() ||
      (features.scalar_type() != torch::kFloat32 &&
       features.scalar_type() != torch::kFloat64))
    return {};
  c10::cuda::CUDAGuard guard(features.device());
  if (!vision_gpu_supported()) return {};
  auto values = rows.accessor<double, 2>();
  const auto count = rows.size(0), ph = options.output_size[0],
             pw = options.output_size[1], ratio = options.sampling_ratio;
  if (!count) return {};
  std::vector<double> parameters;
  std::vector<I> batches;
  parameters.reserve(count * 4);
  batches.reserve(count);
  for (I i = 0; i < count; ++i) {
    const double offset = options.aligned ? .5 : 0.;
    const double x1 = values[i][1] * options.spatial_scale - offset;
    const double y1 = values[i][2] * options.spatial_scale - offset;
    const double width =
        (std::max)(values[i][3] * options.spatial_scale - offset - x1,
                   options.aligned ? 0. : 1.);
    const double height =
        (std::max)(values[i][4] * options.spatial_scale - offset - y1,
                   options.aligned ? 0. : 1.);
    for (double v : {width / (pw * ratio), height / (ph * ratio), x1, y1})
      if (!std::isfinite(v) || std::abs(v) > 1e10) return {};
    parameters.insert(parameters.end(),
                      {width / (pw * ratio), height / (ph * ratio), x1, y1});
    batches.push_back(static_cast<I>(values[i][0]));
  }
  return NativeRoi::apply(
      features,
      torch::tensor(parameters, torch::kFloat64)
          .view({count, 4})
          .to(features.device()),
      torch::tensor(batches, torch::kInt64).to(features.device()), ph, pw,
      ratio);
}
}  // namespace jyppx::liuding::vision_ops::detail
