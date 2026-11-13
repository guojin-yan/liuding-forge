// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_DATA_COMPOSITION_HPP
#define JYPPX_LIUDING_VISION_TASKS_DATA_COMPOSITION_HPP
/** @file composition.hpp @brief Native multi-source and projective training
 * augmentation. */
#include <jyppx/liuding/vision/tasks/data/preparation.hpp>
namespace jyppx::liuding::vision::tasks {
using SampleLoader =
    std::function<data::image::VisionSample(std::size_t, std::stop_token)>;
/** @brief Deterministic partner selection and joint target composition.
 * All inputs are CPU RGB uint8. Mosaic uses four quadrants, MixUp a bounded
 * uniform coefficient [.35,.65], CutMix retains polygon/RLE visible geometry,
 * CopyPaste requires instance masks and rejects unsupported pose/OBB mixing.
 * Source mappings remain on canonical objects; synthetic images have no global
 * inverse GeometryTrace. No Torch/global random state is consumed.
 */
data::image::VisionSample augment_multisource(
    const data::image::VisionSample& primary, std::size_t dataset_size,
    const SampleLoader& load, const MultiSourceOptions& options,
    const LetterBoxOptions& letterbox, const data::image::SampleRandom& random,
    std::uint64_t epoch = 0, std::stop_token stop = {});
/** @brief Homography warp with linked boxes/polygons/RLE/pose and provenance.
 * OBB accepts similarity transforms; shear/perspective explicitly rejects OBB
 * rather than fitting a different rectangle. Photos use bilinear sampling,
 * masks nearest, coordinates continuous pixel boundaries. No inverse inference
 * trace is advertised for this training-only warp.
 */
data::image::VisionSample augment_projective(
    const data::image::VisionSample& sample, const ProjectiveOptions& options,
    const data::image::SampleRandom& random, double padding = 114,
    std::stop_token stop = {});
std::string multisource_identity(const MultiSourceOptions& options);
std::string projective_identity(const ProjectiveOptions& options);
}  // namespace jyppx::liuding::vision::tasks
#endif  // JYPPX_LIUDING_VISION_TASKS_DATA_COMPOSITION_HPP
