// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_DATA_IMAGE_PRIVATE_TASK_GEOMETRY_HPP
#define JYPPX_LIUDING_DATA_IMAGE_PRIVATE_TASK_GEOMETRY_HPP
#include <jyppx/liuding/data_image/joint_transform.hpp>
namespace jyppx::liuding::data::image::detail {
void MapTaskPayload(const VisionSample& source, VisionSample& result,
                    std::span<const ImageGeometryStep> steps,
                    const JointTransformOptions& options,
                    const GeometryTrace& trace, std::stop_token stop);
}
#endif
