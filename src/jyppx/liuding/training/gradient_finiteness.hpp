// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_TRAINING_GRADIENT_FINITENESS_HPP
#define JYPPX_LIUDING_TRAINING_GRADIENT_FINITENESS_HPP

#include <jyppx/liuding/core/types.hpp>

#include <algorithm>
#include <iterator>
#include <optional>
#include <span>
#include <vector>

namespace jyppx::liuding::training::detail {

// Launch every reduction before reading flags. A per-parameter item<bool>()
// serializes the host and accelerator hundreds of times for typical models.
// Copy only one boolean vector per device, retaining the first failing index
// for the Trainer's parameter-specific error. No gradient values are changed.
inline std::optional<std::size_t> first_nonfinite_gradient(
    std::span<const Tensor> gradients) {
  struct DeviceChecks {
    Device device;
    std::vector<Tensor> flags;
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
    device->flags.push_back(torch::isfinite(gradient).all());
    device->indices.push_back(index);
  }
  std::optional<std::size_t> first;
  for (const auto& device : devices) {
    const auto flags = torch::stack(device.flags).to(torch::kCPU);
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
