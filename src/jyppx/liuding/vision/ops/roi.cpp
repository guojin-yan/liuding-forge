// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/vision/ops/roi.hpp>

#include <torch/nn/functional/vision.h>

#include <algorithm>
#include <cmath>
namespace jyppx::liuding::vision_ops {
namespace {
void CheckBoxes(const Tensor& a, const Tensor& b) {
  if (!a.defined() || !b.defined() || a.dim() != 2 || b.dim() != 2 ||
      a.size(1) != 4 || a.size(0) != b.size(0) || !a.is_floating_point() ||
      !b.is_floating_point() || a.device() != b.device())
    throw InvalidArgument(
        "Box coding requires floating corresponding box/delta rows on one "
        "device");
}
void CheckWeights(const std::array<double, 4>& weights) {
  for (auto w : weights)
    if (!std::isfinite(w) || w <= 0)
      throw InvalidArgument("Box coder weights must be positive finite");
}
}  // namespace
Tensor roi_align(const Tensor& features, const Tensor& rois,
                 const RoiAlignOptions& o) {
  if (!features.defined() || features.dim() != 4 ||
      !features.is_floating_point() || features.size(0) < 1 ||
      features.size(1) < 1 || features.size(2) < 1 || features.size(3) < 1 ||
      !rois.defined() || rois.dim() != 2 || rois.size(1) != 5 ||
      !rois.is_floating_point() || rois.device() != features.device() ||
      !std::isfinite(o.spatial_scale) || o.spatial_scale <= 0 ||
      o.output_size[0] < 1 || o.output_size[1] < 1 || o.output_size[0] > 256 ||
      o.output_size[1] > 256 || o.sampling_ratio < 1 || o.sampling_ratio > 8 ||
      o.max_rois < 1 || o.max_output_elements < 1)
    throw InvalidArgument("Invalid ROIAlign feature/ROI metadata or options");
  const auto count = rois.size(0), channels = features.size(1);
  const auto ph = o.output_size[0], pw = o.output_size[1],
             ratio = o.sampling_ratio;
  if (count > o.max_rois || channels > o.max_output_elements / ph / pw ||
      count > o.max_output_elements / ph / pw / channels / ratio / ratio)
    throw ResourceExhausted(
        "ROIAlign ROI/output/sampling element bound exceeded");
  if (count == 0)
    return features.sum() * 0 +
           torch::empty({0, channels, ph, pw}, features.options());
  auto rows = rois.detach().to(torch::kCPU, torch::kFloat64).contiguous();
  auto values = rows.accessor<double, 2>();
  std::vector<Tensor> output;
  const auto tensor_options = features.options().requires_grad(false);
  for (std::int64_t i = 0; i < count; ++i) {
    for (int j = 0; j < 5; ++j)
      if (!std::isfinite(values[i][j]))
        throw InvalidArgument("Nonfinite ROIAlign coordinate");
    const auto batch = static_cast<std::int64_t>(values[i][0]);
    if (values[i][0] != static_cast<double>(batch) || batch < 0 ||
        batch >= features.size(0) || values[i][3] < values[i][1] ||
        values[i][4] < values[i][2])
      throw InvalidArgument("Invalid ROIAlign batch index or box extent");
    const double offset = o.aligned ? .5 : 0.;
    const double x1 = values[i][1] * o.spatial_scale - offset;
    const double y1 = values[i][2] * o.spatial_scale - offset;
    const double width = (std::max)(
        values[i][3] * o.spatial_scale - offset - x1, o.aligned ? 0. : 1.);
    const double height = (std::max)(
        values[i][4] * o.spatial_scale - offset - y1, o.aligned ? 0. : 1.);
    auto xs = (torch::arange(pw * ratio, tensor_options) + .5) *
                  (width / (pw * ratio)) +
              x1;
    auto ys = (torch::arange(ph * ratio, tensor_options) + .5) *
                  (height / (ph * ratio)) +
              y1;
    auto xx = xs.view({1, -1}).expand({ph * ratio, pw * ratio});
    auto yy = ys.view({-1, 1}).expand({ph * ratio, pw * ratio});
    auto valid = (xx >= -1) & (xx <= features.size(3)) & (yy >= -1) &
                 (yy <= features.size(2));
    auto grid = torch::stack({(xx + .5) * (2. / features.size(3)) - 1,
                              (yy + .5) * (2. / features.size(2)) - 1},
                             -1)
                    .unsqueeze(0);
    auto sampled = torch::nn::functional::grid_sample(
        features.slice(0, batch, batch + 1), grid,
        torch::nn::functional::GridSampleFuncOptions()
            .mode(torch::kBilinear)
            .padding_mode(torch::kBorder)
            .align_corners(false));
    sampled = sampled * valid.to(sampled.scalar_type());
    output.push_back(
        sampled.view({channels, ph, ratio, pw, ratio}).mean({2, 4}));
  }
  return torch::stack(output);
}
Tensor multiscale_roi_align(const std::vector<Tensor>& features,
                            const Tensor& rois,
                            const std::vector<std::int64_t>& strides,
                            RoiAlignOptions o) {
  if (features.empty() || features.size() != strides.size() ||
      !rois.defined() || rois.dim() != 2 || rois.size(1) != 5)
    throw InvalidArgument("Invalid multiscale ROI feature/stride/ROI contract");
  for (std::size_t i = 0; i < features.size(); ++i) {
    if (strides[i] < 1 || (strides[i] & (strides[i] - 1)) ||
        (i && strides[i] != 2 * strides[i - 1]) || features[i].dim() != 4 ||
        features[i].size(0) != features[0].size(0) ||
        features[i].size(1) != features[0].size(1))
      throw InvalidArgument("ROI feature reductions/channels/batch mismatch");
  }
  const auto wh = (rois.slice(1, 3, 5) - rois.slice(1, 1, 3)).clamp_min(0);
  auto levels = torch::floor(
      4 +
      torch::log2(torch::sqrt(wh.select(1, 0) * wh.select(1, 1)) / 224 + 1e-6));
  const auto minimum = static_cast<std::int64_t>(std::log2(strides.front()));
  levels = (levels.clamp(minimum, minimum + features.size() - 1) - minimum)
               .to(torch::kInt64);
  Tensor output;
  for (std::size_t i = 0; i < features.size(); ++i) {
    auto selected =
        (levels == static_cast<std::int64_t>(i)).nonzero().flatten();
    o.spatial_scale = 1. / strides[i];
    auto pooled = roi_align(features[i], rois.index_select(0, selected), o);
    if (!output.defined())
      output = torch::zeros({rois.size(0), features[0].size(1),
                             o.output_size[0], o.output_size[1]},
                            pooled.options());
    output = output.index_copy(0, selected, pooled);
  }
  return output;
}
Tensor encode_box_deltas(const Tensor& anchors, const Tensor& targets,
                         std::array<double, 4> weights) {
  CheckBoxes(anchors, targets);
  CheckWeights(weights);
  if (targets.size(1) != 4)
    throw InvalidArgument("Box targets require four columns");
  auto a_wh = (anchors.slice(1, 2) - anchors.slice(1, 0, 2)).clamp_min(1e-6);
  auto t_wh = (targets.slice(1, 2) - targets.slice(1, 0, 2)).clamp_min(1e-6);
  auto shift = ((targets.slice(1, 0, 2) + t_wh * .5) -
                (anchors.slice(1, 0, 2) + a_wh * .5)) /
               a_wh;
  return torch::cat({shift, torch::log(t_wh / a_wh)}, 1) *
         torch::tensor(std::vector<double>(weights.begin(), weights.end()),
                       anchors.options());
}
Tensor decode_box_deltas(const Tensor& anchors, const Tensor& deltas,
                         std::array<double, 4> weights, double max_log_scale) {
  CheckBoxes(anchors, deltas);
  CheckWeights(weights);
  if (deltas.size(1) % 4 || !std::isfinite(max_log_scale) || max_log_scale <= 0)
    throw InvalidArgument("Box delta width/log-scale bound is invalid");
  auto values =
      deltas.reshape({deltas.size(0), -1, 4}) /
      torch::tensor(std::vector<double>(weights.begin(), weights.end()),
                    deltas.options());
  auto wh = (anchors.slice(1, 2) - anchors.slice(1, 0, 2))
                .clamp_min(1e-6)
                .unsqueeze(1);
  auto center = anchors.slice(1, 0, 2).unsqueeze(1) + wh * .5;
  auto pcenter = center + values.slice(2, 0, 2) * wh;
  auto pwh = values.slice(2, 2).clamp_max(max_log_scale).exp() * wh;
  return torch::cat({pcenter - pwh * .5, pcenter + pwh * .5}, 2)
      .reshape_as(deltas);
}
}  // namespace jyppx::liuding::vision_ops
