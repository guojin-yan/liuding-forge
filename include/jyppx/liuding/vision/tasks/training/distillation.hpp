// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_TRAINING_DISTILLATION_HPP
#define JYPPX_LIUDING_VISION_TASKS_TRAINING_DISTILLATION_HPP
/** @file distillation.hpp @brief Frozen same-view teacher training strategy. */
#include <jyppx/liuding/vision/tasks/training/advanced.hpp>
namespace jyppx::liuding::vision::tasks {
struct DistillationOptions {
  double temperature=2.,logits_weight=1.,feature_weight=0.;
  bool multilabel=false;
};
/** @brief Freezes teacher parameters and mode. The optional second output is
 * the projected student/global teacher feature, already registered in student
 * state before optimizer creation. Teacher shares exactly the invocation image
 * tensor, executes without autograd and is checked for mode/state mutation.
 * Validation uses only the supervised student loss. Caller checkpoints teacher
 * state separately and includes verified asset/configuration identity in run.
 */
training::StepFunction make_distillation_step(training::StepFunction supervised,
    std::shared_ptr<nn::ModelAdapter> teacher,DistillationOptions options={});
}
#endif
