// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_WORKFLOW_BINDING_HPP
#define JYPPX_LIUDING_VISION_TASKS_WORKFLOW_BINDING_HPP
/** @file binding.hpp @brief Explicit catalog and model/task workflow
 * composition. */
#include <jyppx/liuding/nn/model_adapter.hpp>
#include <jyppx/liuding/vision/tasks/workflow/recipe.hpp>
namespace jyppx::liuding::vision::tasks {
/** @brief Separate implementation facts and qualification facts. */
struct CapabilityRecord {
  std::string architecture;
  std::string revision;
  TaskKind task = TaskKind::detect;
  std::string variant;
  std::string component;
  std::int64_t input_divisor = 32;
  bool native_training = true;
  bool pretrained_weights = false;
  bool reference_qualified = false;
  bool quality_qualified = false;
  bool end_to_end = false;
  std::array<std::int64_t,2> fixed_input_size{0,0};  ///< H/W; zero is dynamic.
  std::int64_t minimum_input_size=32;
  std::string output_contract="objects/v1";
  bool requires_external_proposals=false;
};
/** @brief Fixed binding retaining one model state and its task-specific
 * contracts. */
struct TaskBinding {
  CapabilityRecord capability;
  std::shared_ptr<nn::ModelAdapter> model;
  training::BatchMapper image_mapper;
  training::TargetMapper target_mapper;
  training::StepFunction step;
  YoloTaskOptions yolo;
  std::function<std::size_t()>
      fuse_inference;  ///< Mutates this binding's eval instance only.
  std::function<std::vector<ImageResult>(
      const c10::IValue&, std::array<std::int64_t, 2>, const ResultOptions&)>
      decode;
};
/** @brief Lists only installed pack selections; never allocates model
 * parameters. */
std::vector<CapabilityRecord> vision_catalog();
/** @brief Resolves one exact model/task/variant selection, without allocating
 * parameters. */
CapabilityRecord describe_binding(const ModelSpec& spec);
/** @brief Creates an owning native adapter and criterion; no weights are
 * fetched. Placement/initialization/freeze must finish before optimizer and EMA
 * binding.
 */
TaskBinding bind_task(const RecipeSpec& recipe);
/** @brief Flattens a fixed tensor or tensor-tuple adapter result; rejects other
 * schemas. */
std::vector<Tensor> raw_tensors(const c10::IValue& value);
}  // namespace jyppx::liuding::vision::tasks
#endif  // JYPPX_LIUDING_VISION_TASKS_WORKFLOW_BINDING_HPP
