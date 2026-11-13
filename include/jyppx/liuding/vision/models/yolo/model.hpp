// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_VISION_MODELS_YOLO_MODEL_HPP
#define JYPPX_LIUDING_VISION_MODELS_YOLO_MODEL_HPP
/** @file model.hpp @brief Typed native YOLO architecture catalog and raw heads.
 */

#include <jyppx/liuding/nn/model_adapter.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace jyppx::liuding::models::yolo {

/** @brief Independently identified model families; no implicit aliases. */
enum class Family { v8, yolo11, yolo12, yolo26 };
/** @brief Compound scale. Windows-safe enumerators avoid the SDK small macro.
 */
enum class Capacity {
  nano_capacity,
  small_capacity,
  medium_capacity,
  large_capacity,
  extra_large_capacity
};
/** @brief Explicit output head and task contract. */
enum class Task { detect, classify, instance_segment, pose, obb };
/** @brief Only configurations present in the frozen reference inventory. */
enum class Variant { p5, p2, p6, ghost_p5, ghost_p2, ghost_p6 };

/** @brief Complete construction options. Factories never download weights. */
struct ModelOptions {
  Family family = Family::v8;
  Capacity capacity = Capacity::nano_capacity;
  Task task = Task::detect;
  Variant variant = Variant::p5;
  std::int64_t num_classes = 80;    ///< [1,10000], contiguous task class IDs.
  std::int64_t input_channels = 3;  ///< [1,16].
  std::int64_t reg_max =
      0;  ///< 0 selects 16, or 1 for YOLO26; otherwise [1,32].
  std::int64_t mask_channels =
      32;  ///< [1,256], instance coefficients/prototypes.
  std::int64_t prototype_channels =
      256;                      ///< Base channels before width scaling.
  std::int64_t keypoints = 17;  ///< [1,256].
  std::int64_t keypoint_dimensions = 3;  ///< 2 (xy) or 3 (xy/visibility logit).
  double dropout = 0;  ///< Classification dropout probability in [0,1).
};

/** @brief Architecture facts, distinct from pretrained/task qualification. */
struct ModelDefinition {
  std::string id;
  std::string revision;  ///< Native graph/output semantics revision.
  std::string reference_path;
  std::string reference_sha256;  ///< Exact local configuration snapshot.
  std::vector<std::int64_t>
      strides;  ///< Ordered raw levels; empty for classify.
  std::int64_t input_divisor = 32;
  std::int64_t reg_max = 16;
  bool end_to_end = false;  ///< Independent detached-feature one-to-one branch.
  std::vector<std::string> output_names;
};

/** @brief Checks options without allocating parameters and resolves definition.
 * @throws InvalidArgument Unknown enum, absent variant or invalid head option.
 */
ModelDefinition describe_model(const ModelOptions& options = {});
/** @brief Lists the 145 closed reference configurations without constructing
 * models. */
std::vector<ModelOptions> model_catalog();
/** @brief Folds native Conv/BN pairs in an eval-only disposable inference
 * clone. Changes state topology; fused weights cannot be used for exact
 * training resume.
 */
std::size_t fuse_for_inference(torch::nn::Module& model);

/** @brief Native graph with stable outputs in both training and evaluation.
 *
 * Detection raw is [B,4*R+C+E,A]. E is 0, mask_channels, K*D (plus 2*K
 * uncertainty logits for YOLO26), or one angle logit. Classification returns
 * one [B,C] logits tensor. Other outputs are ordered by
 * definition.output_names: raw_one_to_many; optional raw_one_to_one; optional
 * mask prototypes; optional YOLO26 semantic logits. YOLO26 one-to-one features
 * are detached. No decoding, NMS, loss, mode-dependent schema, weights or
 * quality claims are implicit. Caller serializes state mutation and bounds
 * input/workspace memory.
 */
class ModelImpl final : public torch::nn::Module {
 public:
  explicit ModelImpl(const ModelOptions& options = {});
  ~ModelImpl() override;
  /** @brief Floating NCHW, positive H/W divisible by definition.input_divisor.
   */
  std::vector<Tensor> forward(const Tensor& images);
  /** @brief Returns raw logits only for detection/classification with one
   * branch.
   * @throws InvalidArgument The definition has more than one raw output.
   */
  Tensor forward_tensor(const Tensor& images);
  /** @brief YOLO26 pose flow log-density for differentiable [N,2] residuals.
   * @throws InvalidArgument This is not a YOLO26 pose model or shape is
   * invalid.
   */
  Tensor pose_log_probability(const Tensor& residuals);
  const ModelOptions& options() const noexcept;
  const ModelDefinition& definition() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
TORCH_MODULE(Model);

/** @brief Owning adapter with a fixed tuple schema, including one-element
 * tuples. */
nn::ModelAdapter create_model(const ModelOptions& options = {});
/** @brief Tensor adapter for single-output detection/classification only. */
nn::ModelAdapter create_tensor_model(const ModelOptions& options = {});

}  // namespace jyppx::liuding::models::yolo
#endif  // JYPPX_LIUDING_VISION_MODELS_YOLO_MODEL_HPP
