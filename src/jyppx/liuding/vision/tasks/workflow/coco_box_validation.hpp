// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_PRIVATE_COCO_BOX_VALIDATION_HPP
#define JYPPX_LIUDING_VISION_TASKS_PRIVATE_COCO_BOX_VALIDATION_HPP
#include <jyppx/liuding/training/metrics/detection.hpp>
#include <jyppx/liuding/vision/tasks/workflow/recipe.hpp>

#include <map>
#include <span>
namespace jyppx::liuding::vision::tasks::detail {
class CocoBoxValidation {
 public:
  CocoBoxValidation(std::shared_ptr<const data::image::AnnotationIndex> targets,
                    std::string dataset_identity, const RecipeSpec& recipe,
                    std::string execution);
  void update(const data::image::ImageBatch& batch,
              std::span<const ImageResult> predictions, std::uint64_t epoch);
  std::string finish(std::stop_token stop);

 protected:
  const ResultOptions& result_options() const noexcept { return results_; }

 private:
  void reset();
  std::shared_ptr<const data::image::AnnotationIndex> targets_;
  std::map<std::string, std::size_t> by_path_;
  training::metrics::DetectionMetricOptions options_;
  ResultOptions results_;
  std::string execution_;
  std::vector<
      std::unique_ptr<training::metrics::DetectionAveragePrecisionMetric>>
      metrics_;
  std::size_t images_ = 0;
  std::optional<std::uint64_t> epoch_;
};
}  // namespace jyppx::liuding::vision::tasks::detail
#endif
