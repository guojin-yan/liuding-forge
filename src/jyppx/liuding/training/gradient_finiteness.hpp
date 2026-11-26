// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_TRAINING_GRADIENT_FINITENESS_HPP
#define JYPPX_LIUDING_TRAINING_GRADIENT_FINITENESS_HPP

#include <jyppx/liuding/core/types.hpp>

#include <ATen/ops/_foreach_norm.h>
#include <algorithm>
#include <iterator>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace jyppx::liuding::training::detail {

// Batch dense real GPU reductions without changing gradient values. The
// infinity norm cannot overflow for finite real elements, unlike an L2 norm.
// Read only one flag vector per device, retaining the first failing index.
inline std::optional<std::size_t> first_nonfinite_gradient(
    std::span<const Tensor> gradients) {
  torch::NoGradGuard guard;
  struct DeviceChecks {
    Device device;
    std::vector<Tensor> gradients;
    std::vector<std::size_t> indices;
  };
  std::vector<DeviceChecks> devices;
  for (std::size_t index = 0; index < gradients.size(); ++index) {
    const auto& gradient = gradients[index];
    if (!gradient.defined()) continue;
    auto device = std::find_if(
        devices.begin(), devices.end(),
        [&](auto& entry) { return entry.device == gradient.device(); });
    if (device == devices.end()) {
      devices.push_back({gradient.device(), {}, {}});
      device = std::prev(devices.end());
    }
    device->gradients.push_back(gradient);
    device->indices.push_back(index);
  }
  std::optional<std::size_t> first;
  for (const auto& device : devices) {
    const auto can_batch =
        device.device.is_cuda() &&
        std::all_of(
            device.gradients.begin(), device.gradients.end(),
            [](const Tensor& gradient) {
              const auto dtype = gradient.scalar_type();
              return gradient.layout() == c10::Layout::Strided &&
                     gradient.numel() > 0 &&
                     gradient.is_non_overlapping_and_dense() &&
                     (dtype == torch::kFloat16 || dtype == torch::kBFloat16 ||
                      dtype == torch::kFloat32 || dtype == torch::kFloat64);
            });
    Tensor flags;
    if (can_batch) {
      const auto norms = at::_foreach_norm(
          device.gradients, std::numeric_limits<double>::infinity());
      flags = torch::isfinite(torch::stack(norms)).to(torch::kCPU);
    } else {
      std::vector<Tensor> checks;
      checks.reserve(device.gradients.size());
      for (const auto& gradient : device.gradients)
        checks.push_back(torch::isfinite(gradient).all());
      flags = torch::stack(checks).to(torch::kCPU);
    }
    const auto* finite = flags.const_data_ptr<bool>();
    for (std::size_t index = 0; index < device.indices.size(); ++index) {
      if (!finite[index] && (!first || device.indices[index] < *first))
        first = device.indices[index];
    }
  }
  return first;
}

}  // namespace jyppx::liuding::training::detail
#endif  // JYPPX_LIUDING_TRAINING_GRADIENT_FINITENESS_HPP
