// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_MODELS_EXTENDED_DETECTORS_HPP
#define JYPPX_LIUDING_VISION_MODELS_EXTENDED_DETECTORS_HPP
/** @file detectors.hpp @brief Native FCOS and RetinaNet v1 heads. */
#include <jyppx/liuding/vision/models/extended/pyramid.hpp>
namespace jyppx::liuding::models::extended {
enum class DenseDetectorKind { fcos, retinanet_v1 };
struct DenseDetectorOptions {
  DenseDetectorKind kind=DenseDetectorKind::fcos;
  PyramidOptions pyramid;
  std::int64_t num_classes=80;
  std::int64_t max_anchors=1048576;
};
/** @brief Fixed tuple: class logits [B,A,C], regressions [B,A,4],
 * centerness [B,A,1] (zero columns for RetinaNet), anchors [A,4],
 * per-level counts [5]. FCOS ltrb values are normalized by stride;
 * RetinaNet deltas use unit weights. Images are normalized by the task.
 */
class DenseDetectorImpl final:public torch::nn::Module{
 public:
  explicit DenseDetectorImpl(DenseDetectorOptions options={});
  std::vector<Tensor> forward(const Tensor& images);
  const DenseDetectorOptions& options()const noexcept{return options_;}
 private:
  DenseDetectorOptions options_;FeaturePyramid pyramid_{nullptr};
  torch::nn::Sequential cls_tower_{nullptr},box_tower_{nullptr};
  torch::nn::Conv2d classes_{nullptr},boxes_{nullptr},center_{nullptr};
};
TORCH_MODULE(DenseDetector);
nn::ModelAdapter adapt(DenseDetector model);
}
#endif
