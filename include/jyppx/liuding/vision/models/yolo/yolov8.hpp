// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_VISION_MODELS_YOLO_YOLOV8_HPP
#define JYPPX_LIUDING_VISION_MODELS_YOLO_YOLOV8_HPP
/** @file yolov8.hpp
 * @brief Native YOLOv8 P5 architecture with a stable raw-output boundary.
 */

#include <jyppx/liuding/core/types.hpp>
#include <jyppx/liuding/nn/model_adapter.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <string>

namespace jyppx::liuding::models::yolo {

/** @brief Standard P5 compound-scaling choices; capacities are independent. */
enum class YoloV8Capacity {
  nano_capacity,         ///< Depth 0.33, width 0.25, base channel cap 1024.
  small_capacity,        ///< Depth 0.33, width 0.50, base channel cap 1024.
  medium_capacity,       ///< Depth 0.67, width 0.75, base channel cap 768.
  large_capacity,        ///< Depth 1.00, width 1.00, base channel cap 512.
  extra_large_capacity,  ///< Depth 1.00, width 1.25, base channel cap 512.
};

/** @brief Typed architecture options; no weights or implicit downloads. */
struct YoloV8Options {
  YoloV8Capacity capacity = YoloV8Capacity::nano_capacity;  ///< Compound scale.
  std::int64_t num_classes = 80;    ///< Positive class count, at most 10000.
  std::int64_t reg_max = 16;        ///< Distribution bins per edge, in [2,32].
  std::int64_t input_channels = 3;  ///< Positive channel count, at most 16.
};

/**
 * @brief Owns a randomly initialized native P5 backbone, PAN neck and head.
 *
 * Module state follows LibTorch ownership. Forward has the same raw schema in
 * train/eval; decoding, assignment and loss belong to vision_tasks. Concurrent
 * execution/mutation requires caller synchronization. This development model
 * does not advertise upstream weight compatibility or task quality.
 */
class YoloV8Impl final : public torch::nn::Module {
 public:
  /** @brief Builds the selected architecture and initializes detection biases.
   * @param options Explicit capacity, channels, classes and regression bins.
   * @throws InvalidArgument An option is outside its supported range.
   */
  explicit YoloV8Impl(const YoloV8Options& options = {});
  /** @brief Releases the owned implementation and registered child handles. */
  ~YoloV8Impl() override;
  /** @brief Returns packed raw P3/P4/P5 outputs.
   * @param images Floating NCHW input; positive batch/H/W, H/W divisible by 32.
   * @return [B,4*reg_max+num_classes,A], A=HW/64+HW/256+HW/1024;
   * P3 then P4 then P5, row-major spatial order. Gradient graph is retained.
   * @throws InvalidArgument Input rank, channels or stride alignment is
   * invalid.
   */
  Tensor forward(const Tensor& images);
  /** @brief Returns each raw scale before packing.
   * @param images Same input contract as forward().
   * @return P3/P4/P5 tensors [B,4*reg_max+num_classes,H/s,W/s], s=8/16/32.
   * @throws InvalidArgument The input metadata is invalid.
   */
  std::array<Tensor, 3> forward_levels(const Tensor& images);
  /** @brief Returns the immutable options used for construction.
   * @return Borrowed options, valid for this module's lifetime.
   */
  const YoloV8Options& options() const noexcept;

 private:
  struct Impl;  ///< Private module wiring; children are registered on this
                ///< module.
  std::unique_ptr<Impl> impl_;  ///< Sole owner of architecture wiring/options.
};

/** @brief Shared LibTorch holder for the native P5 model. */
TORCH_MODULE(YoloV8);

/** @brief Creates the existing framework's owning Tensor-to-Tensor adapter.
 * @param options Explicit randomly initialized architecture selection.
 * @return ModelAdapter retaining one module state and the raw output contract.
 * @throws InvalidArgument Options are invalid.
 */
nn::ModelAdapter create_yolov8(const YoloV8Options& options = {});

/** @brief Returns the stable architecture ID for one selected P5 capacity.
 * @param capacity One declared compound scale.
 * @return Native catalog identity; not a pretrained weight qualification.
 * @throws InvalidArgument The enum value is invalid.
 */
std::string yolov8_model_id(YoloV8Capacity capacity);

}  // namespace jyppx::liuding::models::yolo

#endif  // JYPPX_LIUDING_VISION_MODELS_YOLO_YOLOV8_HPP
