// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_DATA_EXTENDED_HPP
#define JYPPX_LIUDING_VISION_TASKS_DATA_EXTENDED_HPP
/** @file extended.hpp @brief Supplementary manifests and shared batch projections. */
#include <jyppx/liuding/vision/tasks/data/targets.hpp>
#include <filesystem>
namespace jyppx::liuding::vision::tasks {
struct TaskDatasetImport {
  std::shared_ptr<data::image::ImageDataset> dataset;
  std::filesystem::path image_root;
  std::vector<data::image::VisionCategory> categories;
  std::string identity;
  data::image::DepthRepresentation depth_representation=data::image::DepthRepresentation::depth_meters;
  std::int64_t semantic_ignore=255;
  std::vector<std::string> sequence_ids;
  std::vector<std::int64_t> identity_labels;
};
/** @brief Reads bounded native vision.dataset/v1 JSON. All layouts produce
 * the existing ImageDataset/VisionSample; supplementary fields use one typed
 * payload. Semantic masks use the native discrete decoder; depth uses F32
 * safetensors keys depth [1,H,W], valid BOOL [1,H,W]. No executable formats.
 * The digest binds the manifest and every declared target file. Paths must
 * remain under image_root after canonical resolution. */
TaskDatasetImport read_task_dataset(const std::filesystem::path& manifest,
    data::image::ImageDatasetOptions decoding={},std::stop_token stop={});
/** @brief Projects typed fields on the image device, preserving unknowns. */
training::TargetMapper make_extended_target_mapper(TaskKind task,
    std::int64_t classes,std::int64_t patch_size=4,
    std::int64_t mask_patch_size=32,double mask_ratio=.6,std::uint64_t seed=0);
/** @brief Adds deterministic SimMIM mask or external proposals to images.
 * Fast R-CNN requires every item to declare a proposal generator/inventory. */
training::BatchMapper make_extended_image_mapper(TaskKind task,
    const YoloDataOptions& data={},bool external_proposals=false,
    std::int64_t patch_size=4,std::int64_t mask_patch_size=32,
    double mask_ratio=.6,std::uint64_t seed=0);
}
#endif
