// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "deformable_gpu.hpp"

#include "deformable_gpu_launch.hpp"
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
#include <torch/csrc/jit/frontend/tracer.h>

#include <algorithm>
#include <vector>

namespace jyppx::liuding::vision_ops::detail {
namespace {
using I = std::int64_t;

DeformGpuShape Shape(const Tensor& value, const Tensor& locations,
                     const Tensor& weights, const std::vector<I>& levels) {
  DeformGpuShape shape{value.size(0),
                       value.size(1),
                       value.size(2),
                       value.size(3),
                       locations.size(1),
                       locations.size(3),
                       locations.size(4),
                       {},
                       {},
                       {},
                       {}};
  std::copy_n(value.strides().begin(), 4, shape.vs);
  std::copy_n(locations.strides().begin(), 6, shape.ls);
  std::copy_n(weights.strides().begin(), 5, shape.ws);
  std::copy(levels.begin(), levels.end(), shape.spatial);
  return shape;
}

class NativeDeform : public torch::autograd::Function<NativeDeform> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* context, Tensor value,
                        Tensor locations, Tensor weights,
                        std::vector<I> levels) {
    c10::cuda::CUDAGuard guard(value.device());
    auto shape = Shape(value, locations, weights, levels);
    auto output =
        torch::empty({shape.batch, shape.queries, shape.heads * shape.channels},
                     value.options());
    const auto status = launch_deform_gpu_forward(
        value.const_data_ptr(), locations.const_data_ptr(),
        weights.const_data_ptr(), output.mutable_data_ptr(), shape,
        value.scalar_type() == torch::kFloat64,
        c10::cuda::getCurrentCUDAStream(value.get_device()).stream());
    TORCH_CHECK(status == 0,
                "Native deformable sampling GPU launch failed: ", status);
    context->save_for_backward({value, locations, weights});
    context->saved_data["levels"] = std::move(levels);
    return output;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* context,
      torch::autograd::variable_list gradients) {
    TORCH_CHECK(
        !torch::GradMode::is_enabled(),
        "Native deformable sampling supports first-order gradients only");
    auto inputs = context->get_saved_variables();
    const auto& value = inputs[0];
    c10::cuda::CUDAGuard guard(value.device());
    at::globalContext().alertNotDeterministic(
        "liuding_deformable_gpu_backward");
    auto gradient = gradients[0].contiguous();
    auto dv = torch::zeros(value.sizes(), value.options());
    auto dl = torch::empty(inputs[1].sizes(), inputs[1].options());
    auto dw = torch::empty(inputs[2].sizes(), inputs[2].options());
    const auto status = launch_deform_gpu_backward(
        value.const_data_ptr(), inputs[1].const_data_ptr(),
        inputs[2].const_data_ptr(), gradient.const_data_ptr(),
        dv.mutable_data_ptr(), dl.mutable_data_ptr(), dw.mutable_data_ptr(),
        Shape(value, inputs[1], inputs[2],
              context->saved_data["levels"].toIntVector()),
        value.scalar_type() == torch::kFloat64,
        c10::cuda::getCurrentCUDAStream(value.get_device()).stream());
    TORCH_CHECK(
        status == 0,
        "Native deformable sampling GPU backward launch failed: ", status);
    return {dv, dl, dw, Tensor{}};
  }
};
}  // namespace

Tensor try_deform_gpu(const Tensor& value,
                      std::span<const std::array<I, 2>> shapes,
                      const Tensor& locations, const Tensor& weights) {
  if (!value.is_cuda() || torch::jit::tracer::getTracingState() ||
      shapes.size() > kDeformGpuMaxLevels ||
      (value.scalar_type() != torch::kFloat32 &&
       value.scalar_type() != torch::kFloat64) ||
      value.is_neg() || value.is_conj() || locations.is_neg() ||
      locations.is_conj() || weights.is_neg() || weights.is_conj())
    return {};
  c10::cuda::CUDAGuard guard(value.device());
  if (!vision_gpu_supported()) return {};
  std::vector<I> levels;
  levels.reserve(shapes.size() * 3);
  I offset = 0;
  for (const auto& shape : shapes) {
    levels.insert(levels.end(), {shape[0], shape[1], offset});
    offset += shape[0] * shape[1];
  }
  // Kernel arguments retain the bounded host metadata. Uploading a tiny Tensor
  // here synchronizes every call and can outweigh the savings inside a model.
  return NativeDeform::apply(value, locations, weights, std::move(levels));
}
}  // namespace jyppx::liuding::vision_ops::detail
