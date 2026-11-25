// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_DATA_RTDETR_PREPARATION_DETAIL_HPP
#define JYPPX_LIUDING_VISION_TASKS_DATA_RTDETR_PREPARATION_DETAIL_HPP
#include <jyppx/liuding/vision/tasks/data/rtdetr_preparation.hpp>

#include <array>
#include <optional>
namespace jyppx::liuding::vision::tasks::detail {
// Fixed parameters separate pixel/box reference checks from RNG sampling.
struct RtDetrSamplePlan {
  std::optional<double> brightness, contrast, saturation, hue;
  bool contrast_before = false;
  std::optional<std::array<int, 3>> permutation;
  std::array<std::uint32_t, 4> padding{};  // left, top, right, bottom
  std::optional<data::image::ImageGeometryStep> crop;
  bool horizontal_flip = false;
  bool sanitize_boxes = true;
};
data::image::VisionSample apply_rtdetr_plan(
    const data::image::VisionSample& source, data::image::ImageExtent canvas,
    const RtDetrSamplePlan& plan,
    const data::image::JointTransformOptions& geometry,
    std::stop_token stop = {});
}  // namespace jyppx::liuding::vision::tasks::detail
#endif
