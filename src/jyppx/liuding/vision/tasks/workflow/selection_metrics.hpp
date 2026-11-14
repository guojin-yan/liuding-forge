// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_WORKFLOW_SELECTION_METRICS_HPP
#define JYPPX_LIUDING_VISION_TASKS_WORKFLOW_SELECTION_METRICS_HPP

#include <jyppx/liuding/vision/tasks/workflow/recipe.hpp>

#include <string_view>

namespace jyppx::liuding::vision::tasks::detail {
inline bool selection_metric_available(const RecipeSpec& recipe,
                                       std::string_view metric) {
  if (metric == "validation.loss") return true;
  switch (recipe.model.task) {
    case TaskKind::detect:
    case TaskKind::instance_segment:
    case TaskKind::pose:
    case TaskKind::obb:
      return metric == "validation.map" || metric == "validation.ap50";
    case TaskKind::classify:
      return metric == "validation.top1" || metric == "validation.top5";
    case TaskKind::classify_multilabel:
      return metric == "validation.map" || metric == "validation.macro_f1" ||
             metric == "validation.micro_f1";
    case TaskKind::semantic_segment:
      return metric == "validation.miou" ||
             metric == "validation.pixel_accuracy";
    case TaskKind::depth:
      return metric == "validation.mae" || metric == "validation.rmse" ||
             (!recipe.extension.relative_inverse_depth &&
              (metric == "validation.absrel" || metric == "validation.delta1"));
    case TaskKind::reid:
      return metric == "validation.map" || metric == "validation.rank1";
    case TaskKind::pretrain_simmim:
      return false;
  }
  return false;
}
}  // namespace jyppx::liuding::vision::tasks::detail
#endif
