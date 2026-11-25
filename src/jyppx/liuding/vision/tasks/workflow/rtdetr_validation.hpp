// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_PRIVATE_RTDETR_VALIDATION_HPP
#define JYPPX_LIUDING_VISION_TASKS_PRIVATE_RTDETR_VALIDATION_HPP
#include <jyppx/liuding/training/metrics/detection.hpp>
#include <jyppx/liuding/vision/tasks/workflow/binding.hpp>

#include <map>
namespace jyppx::liuding::vision::tasks::detail {
class RtDetrValidation {
 public:
  RtDetrValidation(std::shared_ptr<const data::image::AnnotationIndex> targets,
                   std::string dataset_identity, const RecipeSpec& recipe);
  training::StepOutput step(TaskBinding& binding,
                            const training::StepInput& input);
  std::string finish(std::stop_token stop);

 private:
  void reset();
  std::shared_ptr<const data::image::AnnotationIndex> targets_;
  std::map<std::string, std::size_t> by_path_;
  training::metrics::DetectionMetricOptions options_;
  ResultOptions results_;
  std::unique_ptr<training::metrics::DetectionAveragePrecisionMetric> metric_;
  std::size_t images_ = 0;
  std::optional<std::uint64_t> epoch_;
};
}  // namespace jyppx::liuding::vision::tasks::detail
#endif
