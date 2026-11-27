// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_PRIVATE_TARGET_GROUPING_HPP
#define JYPPX_LIUDING_VISION_PRIVATE_TARGET_GROUPING_HPP

#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace jyppx::liuding::vision::detail {
// Callers check the target matrix shape. Preserve equality-selection semantics,
// including ignored IDs, row order within each image, and target gradients.
inline std::vector<torch::Tensor> group_targets_by_image(
    const torch::Tensor& targets, std::int64_t batch) {
  using I = std::int64_t;
  if (batch <= 0) return {};
  if (batch > 32 || targets.layout() != torch::kStrided ||
      targets.is_complex()) {
    std::vector<torch::Tensor> result;
    for (I b = 0; b < batch; ++b)
      result.push_back(targets.index({targets.select(1, 0) == b}));
    return result;
  }
  std::vector<I> sizes(batch, 0);
  if (!targets.size(0)) return targets.split_with_sizes(sizes, 0);
  // Only the ID column crosses the device boundary, once for the batch.
  const auto ids = targets.select(1, 0)
                       .detach()
                       .to(torch::kCPU, torch::kFloat64)
                       .contiguous();
  const auto* data = ids.const_data_ptr<double>();
  const auto valid = [batch](double id) {
    return id >= 0 && id < batch && id == std::floor(id);
  };
  bool ordered = true;
  double previous = -1;
  I count = 0;
  for (I row = 0; row < ids.numel(); ++row) {
    const auto id = data[row];
    if (!valid(id)) {
      ordered = false;
      continue;
    }
    ++sizes[static_cast<I>(id)];
    ++count;
    if (id < previous) ordered = false;
    previous = id;
  }
  // The normal data mapper already emits image-ordered targets: no reorder or
  // target copy is needed, and all per-image views share one split backward.
  if (ordered) return targets.split_with_sizes(sizes, 0);
  auto indices = torch::empty({count}, torch::kInt64);
  auto* order = indices.mutable_data_ptr<I>();
  std::vector<I> cursor(batch, 0);
  for (I b = 1; b < batch; ++b) cursor[b] = cursor[b - 1] + sizes[b - 1];
  for (I row = 0; row < ids.numel(); ++row)
    if (valid(data[row])) order[cursor[static_cast<I>(data[row])]++] = row;
  return targets.index_select(0, indices.to(targets.device()))
      .split_with_sizes(sizes, 0);
}
}  // namespace jyppx::liuding::vision::detail
#endif
