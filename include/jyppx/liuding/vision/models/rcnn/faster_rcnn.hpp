// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
#ifndef JYPPX_LIUDING_VISION_MODELS_RCNN_FASTER_RCNN_HPP
#define JYPPX_LIUDING_VISION_MODELS_RCNN_FASTER_RCNN_HPP
/** @file faster_rcnn.hpp @brief Native feature-bound Faster R-CNN/FPN/RPN/ROI.
 */
#include <jyppx/liuding/models/features.hpp>
#include <jyppx/liuding/nn/model_adapter.hpp>

#include <memory>
#include <vector>
namespace jyppx::liuding::models::rcnn {
/** @brief Randomly initialized detector settings; class count excludes
 * background. */
struct FasterRcnnOptions {
  std::int64_t num_classes = 80;
  std::int64_t fpn_channels = 256;
  std::int64_t representation_size = 1024;
  std::vector<std::int64_t> anchor_sizes{32, 64, 128, 256, 512};
  std::vector<double> aspect_ratios{.5, 1., 2.};
  std::int64_t rpn_pre_nms_train = 2000;
  std::int64_t rpn_pre_nms_eval = 1000;
  std::int64_t rpn_post_nms_train = 2000;
  std::int64_t rpn_post_nms_eval = 1000;
  double rpn_nms_threshold = .7;
  std::int64_t roi_batch_size = 512;
  double roi_positive_fraction = .25;
  std::int64_t max_targets = 4096;
  std::int64_t max_anchor_elements = 16777216;
};
/** @brief Fixed raw output schema, with variable proposal count.
 * proposals=[R,5] batch/xyxy; labels and matched_boxes have R rows when targets
 * are supplied in training, otherwise -1/zero placeholders. All outputs remain
 * present in both modes; proposal selection and matched identities are
 * detached.
 */
struct FasterRcnnOutput {
  Tensor rpn_logits;     ///< [B,A].
  Tensor rpn_deltas;     ///< [B,A,4].
  Tensor anchors;        ///< [A,4].
  Tensor proposals;      ///< [R,5].
  Tensor class_logits;   ///< [R,C+1], background index 0.
  Tensor box_deltas;     ///< [R,4*(C+1)], weights 10/10/5/5.
  Tensor labels;         ///< [R] int64, -1 when not sampled for training.
  Tensor matched_boxes;  ///< [R,4].
  Tensor image_sizes;    ///< [B,2], height/width in canvas pixels.
  std::vector<Tensor> tensors() const;
};
/** @brief Detector retaining one existing backbone state and explicit features.
 *
 * Requires exactly four NCHW features at reductions 4/8/16/32, declared
 * channels and stable names. It registers the original backbone module without
 * copying parameters. Images are already normalized floating NCHW; padding and
 * original coordinate restoration belong to the task. No pretrained weights are
 * loaded.
 */
class FasterRcnnImpl final : public torch::nn::Module {
 public:
  FasterRcnnImpl(nn::ModelAdapter backbone, FeatureSpec features,
                 const FasterRcnnOptions& options = {});
  ~FasterRcnnImpl() override;
  /** @brief Runs FPN/RPN/ROI; optional targets=[T,6] batch/class/xyxy.
   * @param image_sizes Optional [B,2] actual extents within padded images.
   * @note Training with targets consumes Torch RNG for balanced ROI sampling.
   */
  FasterRcnnOutput forward(const Tensor& images, const Tensor& targets = {},
                           const Tensor& image_sizes = {});
  /** @brief Deterministic inference stages: P2-P5, five logits and five deltas.
   * Exposes the same registered state for native graph export; no proposals or
   * CPU control flow are captured. Model must already be in evaluation mode.
   */
  std::vector<Tensor> forward_rpn(const Tensor& images);
  /** @brief Shared ROI dense head accepting already pooled [R,C,7,7] features.
   */
  std::vector<Tensor> forward_box_head(const Tensor& pooled);
  const FasterRcnnOptions& options() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
TORCH_MODULE(FasterRcnn);
/** @brief Creates the required ResNet50 v1.5 + FPN combination using models
 * registry. Adapter accepts images, targets [T,6], and image_sizes [B,2];
 * returns a fixed nine-tensor tuple. Empty targets and full image extents are
 * valid inference inputs. Model mode controls proposal budgets and training ROI
 * sampling.
 */
nn::ModelAdapter create_faster_rcnn_resnet50_fpn(
    const FasterRcnnOptions& options = {});
}  // namespace jyppx::liuding::models::rcnn
#endif  // JYPPX_LIUDING_VISION_MODELS_RCNN_FASTER_RCNN_HPP
