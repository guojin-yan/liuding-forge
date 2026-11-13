// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/vision/ops/deformable.hpp>
#include <jyppx/liuding/vision/ops/boxes.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <torch/nn/functional/vision.h>
#include <algorithm>
#include <cmath>
#include <vector>
namespace jyppx::liuding::vision_ops {
namespace { using I = std::int64_t; }
Tensor multiscale_deformable_attention(
    const Tensor& value, std::span<const std::array<I, 2>> shapes,
    const Tensor& locations, const Tensor& weights, I cap) {
  if (!value.defined() || value.dim() != 4 || !value.is_floating_point() ||
      !locations.defined() || locations.dim() != 6 || !weights.defined() ||
      weights.dim() != 5 || shapes.empty() || cap < 1 ||
      locations.size(5) != 2 || locations.size(0) != value.size(0) ||
      locations.size(2) != value.size(2) || locations.size(3) != shapes.size() ||
      weights.sizes() != locations.sizes().slice(0, 5) ||
      locations.device() != value.device() || weights.device() != value.device() ||
      locations.scalar_type() != value.scalar_type() || weights.scalar_type() != value.scalar_type())
    throw InvalidArgument("Invalid deformable attention shapes/dtypes/devices");
  I total = 0;
  for (auto shape : shapes) {
    if (shape[0] < 1 || shape[1] < 1 || shape[0] > cap / shape[1] || total > cap - shape[0] * shape[1])
      throw ResourceExhausted("Deformable attention spatial limit exceeded");
    total += shape[0] * shape[1];
  }
  if (total != value.size(1)) throw InvalidArgument("Deformable value length differs from spatial shapes");
  I count = 1;
  for (I size : {value.size(0), locations.size(1), value.size(2), value.size(3),
                 static_cast<I>(shapes.size()), locations.size(4)}) {
    if (size < 1 || count > cap / size) throw ResourceExhausted("Deformable sampling budget exceeded");
    count *= size;
  }
  std::vector<Tensor> sampled;
  I offset = 0;
  for (I level = 0; level < static_cast<I>(shapes.size()); ++level) {
    auto [h, w] = shapes[level];
    auto feature = value.slice(1, offset, offset + h * w).flatten(2).permute({0, 2, 1})
                       .reshape({value.size(0) * value.size(2), value.size(3), h, w});
    auto grid = (locations.select(3, level) * 2 - 1).permute({0, 2, 1, 3, 4}).flatten(0, 1);
    sampled.push_back(torch::nn::functional::grid_sample(feature, grid,
        torch::nn::functional::GridSampleFuncOptions().mode(torch::kBilinear)
            .padding_mode(torch::kZeros).align_corners(false)));
    offset += h * w;
  }
  auto packed = torch::stack(sampled, -2).flatten(-2);
  auto factors = weights.permute({0, 2, 1, 3, 4}).reshape(
      {value.size(0) * value.size(2), 1, locations.size(1), -1});
  return (packed * factors).sum(-1).reshape(
      {value.size(0), value.size(2) * value.size(3), locations.size(1)}).permute({0, 2, 1});
}
Tensor roi_pool(const Tensor& x, const Tensor& rois, std::array<I, 2> out,
                double scale, I cap) {
  if (!x.defined() || x.dim() != 4 || !x.is_floating_point() || !rois.defined() ||
      rois.dim() != 2 || rois.size(1) != 5 || !rois.is_floating_point() ||
      out[0] < 1 || out[1] < 1 || !std::isfinite(scale) || scale <= 0 || cap < 1)
    throw InvalidArgument("Invalid ROI pool inputs");
  I count = (std::max)(I{1}, rois.size(0));
  for (I size : {x.size(1), out[0], out[1]}) {
    if (size < 1 || count > cap / size) throw ResourceExhausted("ROI pool output budget exceeded");
    count *= size;
  }
  if (!rois.size(0)) return x.sum() * 0 + torch::empty({0, x.size(1), out[0], out[1]}, x.options());
  auto r = rois.detach().to(torch::kCPU, torch::kFloat64).contiguous();
  if (!torch::isfinite(r).all().item<bool>()) throw InvalidArgument("Nonfinite ROI pool coordinates");
  auto a = r.accessor<double, 2>();
  std::vector<Tensor> pooled;
  for (I row = 0; row < r.size(0); ++row) {
    I batch = static_cast<I>(a[row][0]);
    if (batch < 0 || batch >= x.size(0) || a[row][0] != batch)
      throw InvalidArgument("ROI pool batch index out of range");
    for (I j = 1; j < 5; ++j)
      if (std::abs(a[row][j] * scale) > 1e8) throw InvalidArgument("ROI pool coordinate overflow");
    I x1 = std::llround(a[row][1] * scale), y1 = std::llround(a[row][2] * scale);
    I x2 = std::llround(a[row][3] * scale), y2 = std::llround(a[row][4] * scale);
    double rh = (std::max)(I{1}, y2 - y1 + 1), rw = (std::max)(I{1}, x2 - x1 + 1);
    std::vector<Tensor> cells;
    for (I h = 0; h < out[0]; ++h) for (I w = 0; w < out[1]; ++w) {
      I ys = std::clamp<I>(y1 + static_cast<I>(std::floor(h * rh / out[0])), 0, x.size(2));
      I ye = std::clamp<I>(y1 + static_cast<I>(std::ceil((h + 1) * rh / out[0])), 0, x.size(2));
      I xs = std::clamp<I>(x1 + static_cast<I>(std::floor(w * rw / out[1])), 0, x.size(3));
      I xe = std::clamp<I>(x1 + static_cast<I>(std::ceil((w + 1) * rw / out[1])), 0, x.size(3));
      cells.push_back(ye <= ys || xe <= xs ? x[batch].sum({1, 2}) * 0
          : std::get<0>(x[batch].slice(1, ys, ye).slice(2, xs, xe).flatten(1).max(1)));
    }
    pooled.push_back(torch::stack(cells, 1).view({x.size(1), out[0], out[1]}));
  }
  return torch::stack(pooled);
}
Tensor aligned_generalized_iou(const Tensor& a, const Tensor& b, double epsilon) {
  auto iou = aligned_box_iou(a, b, epsilon);
  auto tl = torch::minimum(a.slice(1, 0, 2), b.slice(1, 0, 2));
  auto br = torch::maximum(a.slice(1, 2, 4), b.slice(1, 2, 4));
  auto enclosure = (br - tl).clamp_min(0).prod(1).clamp_min(epsilon);
  auto itl = torch::maximum(a.slice(1, 0, 2), b.slice(1, 0, 2));
  auto ibr = torch::minimum(a.slice(1, 2, 4), b.slice(1, 2, 4));
  auto intersection = (ibr - itl).clamp_min(0).prod(1);
  auto united = box_area(a) + box_area(b) - intersection;
  return iou - (enclosure - united) / enclosure;
}
}  // namespace jyppx::liuding::vision_ops
