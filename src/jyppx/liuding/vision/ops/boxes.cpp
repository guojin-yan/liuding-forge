// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/vision/ops/boxes.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace jyppx::liuding::vision_ops {
namespace {

[[noreturn]] void Invalid(const std::string& operation,
                          const std::string& expected) {
  throw InvalidArgument(
      "Invalid visual box operation input",
      ErrorDetails{
          .component = "vision_ops.boxes",
          .operation = operation,
          .expected = expected,
          .actual = "input does not satisfy the declared contract",
          .hint = "Check coordinate format, tensor metadata and bounds."});
}

void ValidateBoxes(const Tensor& boxes, const std::string& operation) {
  if (!boxes.defined() || boxes.layout() != torch::kStrided ||
      boxes.dim() != 2 || boxes.size(1) != 4 ||
      (boxes.scalar_type() != torch::kFloat32 &&
       boxes.scalar_type() != torch::kFloat64)) {
    Invalid(operation, "a defined strided float32/float64 [N,4] tensor");
  }
  if (!boxes.is_cpu() && !boxes.is_cuda()) {
    throw DeviceError("Visual box geometry requires CPU or CUDA tensors");
  }
}

void ValidatePair(const Tensor& left, const Tensor& right,
                  const std::string& operation) {
  ValidateBoxes(left, operation + ".left");
  ValidateBoxes(right, operation + ".right");
  if (left.scalar_type() != right.scalar_type() ||
      left.device() != right.device()) {
    Invalid(operation, "matching input dtype and device");
  }
}

void ValidateEpsilon(double epsilon, Dtype dtype,
                     const std::string& operation) {
  if (!std::isfinite(epsilon) || epsilon <= 0.0 ||
      (dtype == torch::kFloat32 &&
       (epsilon < std::numeric_limits<float>::denorm_min() ||
        epsilon > std::numeric_limits<float>::max()))) {
    Invalid(operation,
            "a positive finite epsilon representable in input dtype");
  }
}

Tensor Area(const Tensor& boxes) {
  return (boxes.select(1, 2) - boxes.select(1, 0)).clamp_min(0) *
         (boxes.select(1, 3) - boxes.select(1, 1)).clamp_min(0);
}

bool ValidFormat(BoxFormat format) {
  return format == BoxFormat::xyxy || format == BoxFormat::cxcywh;
}

}  // namespace

Tensor convert_boxes(const Tensor& boxes, BoxFormat source, BoxFormat target) {
  ValidateBoxes(boxes, "convert_boxes");
  if (!ValidFormat(source) || !ValidFormat(target)) {
    Invalid("convert_boxes.format", "xyxy or cxcywh");
  }
  if (source == target) {
    return boxes.clone();
  }
  const auto first = boxes.slice(1, 0, 2);
  const auto second = boxes.slice(1, 2, 4);
  if (source == BoxFormat::xyxy) {
    // Halve first to avoid overflowing a finite center when both corners
    // are large and have the same sign.
    return torch::cat({first * 0.5 + second * 0.5, second - first}, 1);
  }
  const auto half_extent = second * 0.5;
  return torch::cat({first - half_extent, first + half_extent}, 1);
}

Tensor box_area(const Tensor& boxes) {
  ValidateBoxes(boxes, "box_area");
  return Area(boxes);
}

Tensor aligned_box_iou(const Tensor& left, const Tensor& right,
                       double epsilon) {
  ValidatePair(left, right, "aligned_box_iou");
  ValidateEpsilon(epsilon, left.scalar_type(), "aligned_box_iou.epsilon");
  if (left.size(0) != right.size(0)) {
    Invalid("aligned_box_iou.rows", "the same number of boxes");
  }
  const auto lower = torch::maximum(left.slice(1, 0, 2), right.slice(1, 0, 2));
  const auto upper = torch::minimum(left.slice(1, 2, 4), right.slice(1, 2, 4));
  const auto extent = (upper - lower).clamp_min(0);
  const auto intersection = extent.select(1, 0) * extent.select(1, 1);
  return intersection /
         (Area(left) + Area(right) - intersection).clamp_min(epsilon);
}

Tensor pairwise_box_iou(const Tensor& left, const Tensor& right,
                        const PairwiseBoxOptions& options) {
  ValidatePair(left, right, "pairwise_box_iou");
  ValidateEpsilon(options.epsilon, left.scalar_type(),
                  "pairwise_box_iou.epsilon");
  if (options.max_pairs <= 0) {
    Invalid("pairwise_box_iou.max_pairs", "a positive pair bound");
  }
  const auto rows = left.size(0);
  const auto columns = right.size(0);
  if (columns != 0 && rows > options.max_pairs / columns) {
    throw ResourceExhausted(
        "Pairwise box IoU exceeds the pair bound",
        ErrorDetails{
            .component = "vision_ops.boxes",
            .operation = "pairwise_box_iou.max_pairs",
            .expected = "at most " + std::to_string(options.max_pairs),
            .actual = std::to_string(rows) + " * " + std::to_string(columns),
            .hint = "Use smaller groups or explicitly raise max_pairs."});
  }
  if (rows == 0 || columns == 0) {
    // Broadcasting empty views avoids O(N+M) intermediates and preserves the
    // autograd edges to both inputs, including the nonempty axis.
    return left.select(1, 0).unsqueeze(1) + right.select(1, 0).unsqueeze(0);
  }
  const auto lower = torch::maximum(left.slice(1, 0, 2).unsqueeze(1),
                                    right.slice(1, 0, 2).unsqueeze(0));
  const auto upper = torch::minimum(left.slice(1, 2, 4).unsqueeze(1),
                                    right.slice(1, 2, 4).unsqueeze(0));
  const auto extent = (upper - lower).clamp_min(0);
  const auto intersection = extent.select(2, 0) * extent.select(2, 1);
  return intersection /
         (Area(left).unsqueeze(1) + Area(right).unsqueeze(0) - intersection)
             .clamp_min(options.epsilon);
}

Tensor nms(const Tensor& boxes, const Tensor& scores,
           const NmsOptions& options) {
  ValidateBoxes(boxes, "nms.boxes");
  if (!boxes.is_cpu() || (scores.defined() && !scores.is_cpu())) {
    throw DeviceError("This NMS profile requires CPU boxes and scores",
                      "Select a CPU postprocessing profile explicitly.");
  }
  if (!scores.defined() || scores.layout() != torch::kStrided ||
      scores.dim() != 1 || scores.size(0) != boxes.size(0) ||
      scores.scalar_type() != boxes.scalar_type()) {
    Invalid("nms.scores", "a matching strided CPU [N] score tensor");
  }
  if (!std::isfinite(options.iou_threshold) || options.iou_threshold < 0.0 ||
      options.iou_threshold > 1.0 || options.max_candidates <= 0 ||
      options.max_output <= 0) {
    Invalid("nms.options",
            "IoU threshold in [0,1] and positive input/output bounds");
  }
  const auto count = boxes.size(0);
  if (count > options.max_candidates) {
    throw ResourceExhausted("NMS exceeds max_candidates before copying inputs");
  }
  const torch::NoGradGuard no_grad;
  const auto coordinates = boxes.to(torch::kFloat64).contiguous();
  const auto confidence = scores.to(torch::kFloat64).contiguous();
  const auto* coordinate_data = coordinates.data_ptr<double>();
  const auto* score_data = confidence.data_ptr<double>();
  std::vector<double> areas(static_cast<std::size_t>(count));
  for (std::int64_t index = 0; index < count; ++index) {
    const auto* box = coordinate_data + index * 4;
    if (!std::isfinite(score_data[index]) ||
        !std::all_of(box, box + 4,
                     [](double value) { return std::isfinite(value); })) {
      Invalid("nms.values", "finite coordinates and scores");
    }
    const auto width = box[2] - box[0];
    const auto height = box[3] - box[1];
    const auto area = width * height;
    if (width < 0.0 || height < 0.0 || !std::isfinite(width) ||
        !std::isfinite(height) || !std::isfinite(area)) {
      Invalid("nms.boxes", "ordered corners and finite edge lengths/areas");
    }
    areas[static_cast<std::size_t>(index)] = area;
  }
  std::vector<std::int64_t> order(static_cast<std::size_t>(count));
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [score_data](auto left, auto right) {
    return score_data[left] == score_data[right]
               ? left < right
               : score_data[left] > score_data[right];
  });
  std::vector<std::uint8_t> suppressed(static_cast<std::size_t>(count), 0);
  std::vector<std::int64_t> kept;
  kept.reserve(static_cast<std::size_t>((std::min)(count, options.max_output)));
  for (std::int64_t position = 0; position < count; ++position) {
    if (suppressed[static_cast<std::size_t>(position)] != 0) {
      continue;
    }
    const auto selected = order[static_cast<std::size_t>(position)];
    kept.push_back(selected);
    if (static_cast<std::int64_t>(kept.size()) == options.max_output) {
      break;
    }
    const auto* first = coordinate_data + selected * 4;
    for (auto next = position + 1; next < count; ++next) {
      if (suppressed[static_cast<std::size_t>(next)] != 0) {
        continue;
      }
      const auto candidate = order[static_cast<std::size_t>(next)];
      const auto* second = coordinate_data + candidate * 4;
      const auto width = (std::max)(0.0, (std::min)(first[2], second[2]) -
                                             (std::max)(first[0], second[0]));
      const auto height = (std::max)(0.0, (std::min)(first[3], second[3]) -
                                              (std::max)(first[1], second[1]));
      // Scale before adding areas so two individually finite areas cannot
      // overflow the union and silently turn overlap into zero.
      const auto scale = (std::max)(areas[selected], areas[candidate]);
      const auto overlap = scale == 0.0 ? 0.0 : width * height / scale;
      const auto iou = scale == 0.0
                           ? 0.0
                           : overlap / (areas[selected] / scale +
                                        areas[candidate] / scale - overlap);
      if (iou > options.iou_threshold) {
        suppressed[static_cast<std::size_t>(next)] = 1;
      }
    }
  }
  auto result = torch::empty({static_cast<std::int64_t>(kept.size())},
                             torch::TensorOptions().dtype(torch::kInt64));
  if (!kept.empty()) {
    std::copy(kept.begin(), kept.end(), result.data_ptr<std::int64_t>());
  }
  return result;
}

}  // namespace jyppx::liuding::vision_ops
