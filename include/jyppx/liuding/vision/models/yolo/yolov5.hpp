// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
#ifndef JYPPX_LIUDING_VISION_MODELS_YOLO_YOLOV5_HPP
#define JYPPX_LIUDING_VISION_MODELS_YOLO_YOLOV5_HPP
/** @file yolov5.hpp @brief Original anchor/objectness YOLOv5, distinct from
 * v5u. */
#include <jyppx/liuding/vision/models/yolo/model.hpp>

#include <array>
namespace jyppx::liuding::models::yolo {
/** @brief Original YOLOv5 P5/P6 n/s/m/l/x construction options.
 * Source: ultralytics/yolov5 tag v7.0, commit
 * 915bbf294bb74c859f0b41f1c23bc395014ea679. The source tag is a YOLOv5
 * release tag; it does not introduce the excluded YOLOv7 model family.
 */
struct OriginalV5Options {
  Capacity capacity = Capacity::nano_capacity;
  bool p6 = false;
  std::int64_t num_classes = 80;
  std::int64_t input_channels = 3;
  std::vector<std::array<double, 6>>
      anchors;  ///< Empty selects pinned defaults.
};
/** @brief Returns the explicit pixel anchor table, without constructing state.
 */
std::vector<std::array<double, 6>> original_v5_anchors(bool p6 = false);
/** @brief Native C3/SPPF/PAN detector with persistent raw output schema.
 * Returns [B,A,C+5], ordered level/anchor/y/x. Columns are raw center offsets,
 * width/height logits, objectness and class logits. No DFL or TAL is involved.
 */
class OriginalV5Impl final : public torch::nn::Module {
 public:
  explicit OriginalV5Impl(const OriginalV5Options& options = {});
  ~OriginalV5Impl() override;
  Tensor forward(const Tensor& images);
  const OriginalV5Options& options() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
TORCH_MODULE(OriginalV5);
/** @brief Tensor-to-Tensor adapter; no weights or network access. */
nn::ModelAdapter create_original_v5(const OriginalV5Options& options = {});
/** @brief Stable original-source identity, including capacity and P5/P6. */
std::string original_v5_model_id(const OriginalV5Options& options = {});
}  // namespace jyppx::liuding::models::yolo
#endif  // JYPPX_LIUDING_VISION_MODELS_YOLO_YOLOV5_HPP
