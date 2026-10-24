// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_DATA_IMAGE_JOINT_ANNOTATION_DETAIL_HPP
#define JYPPX_LIUDING_DATA_IMAGE_JOINT_ANNOTATION_DETAIL_HPP

#include <jyppx/liuding/data_image/joint_transform.hpp>

namespace jyppx::liuding::data::image::detail {

struct JointAnnotationState {
  std::size_t geometry_checks = 0;
  std::size_t output_points = 0;
  std::size_t output_fragments = 0;
  std::size_t output_runs = 0;
  std::optional<VisionAnnotationRecord> panoptic;
};

JointAnnotationState CheckJointAnnotations(
    const VisionSample& sample, std::span<const ImageGeometryStep> steps,
    std::span<const ImageExtent> extents, const JointTransformOptions& options,
    const GeometryTrace& local, std::stop_token stop);

void MapJointAnnotations(const VisionSample& source, VisionSample& output,
                         std::span<const ImageGeometryStep> steps,
                         std::span<const ImageExtent> extents,
                         const JointTransformOptions& options,
                         const GeometryTrace& local,
                         JointAnnotationState& budget, std::stop_token stop);

Tensor TransformDiscrete(const Tensor& input,
                         std::span<const ImageGeometryStep> steps,
                         std::span<const ImageExtent> extents,
                         std::int64_t fill, std::stop_token stop);

}  // namespace jyppx::liuding::data::image::detail

#endif  // JYPPX_LIUDING_DATA_IMAGE_JOINT_ANNOTATION_DETAIL_HPP
