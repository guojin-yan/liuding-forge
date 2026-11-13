// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_VISION_OPS_BOXES_HPP
#define JYPPX_LIUDING_VISION_OPS_BOXES_HPP
/**
 * @file boxes.hpp
 * @brief Differentiable axis-aligned box geometry and bounded CPU suppression.
 */

#include <jyppx/liuding/core/types.hpp>

#include <cstdint>

namespace jyppx::liuding::vision_ops {

/** @brief Coordinate ordering of a box in a continuous coordinate system. */
enum class BoxFormat {
  xyxy,    ///< Left, top, right, bottom; no inclusive-pixel +1 correction.
  cxcywh,  ///< Center x/y and full width/height, in the same coordinate units.
};

/**
 * @brief Bounds the dense pair matrix before allocation.
 *
 * Plain value options may be copied between threads. The pair bound limits
 * N*M, not allocator workspace, autograd storage, process RSS or CUDA VRAM.
 */
struct PairwiseBoxOptions {
  std::int64_t max_pairs = 1048576;  ///< Positive maximum number of N*M pairs.
  double epsilon = 1e-7;  ///< Positive finite lower bound on the union area.
};

/**
 * @brief Bounds deterministic, class-agnostic CPU non-maximum suppression.
 *
 * Inputs are snapshotted into contiguous CPU float64 storage. Scratch is O(N),
 * worst-case comparison work is O(N*N); no dense IoU matrix is allocated.
 * Instances own no resources and may be copied between threads.
 */
struct NmsOptions {
  double iou_threshold = 0.45;  ///< Suppress only if IoU is strictly greater.
  std::int64_t max_candidates =
      4096;                       ///< Positive input bound, before copying.
  std::int64_t max_output = 300;  ///< Positive maximum number of kept indices.
};

/**
 * @brief Converts [N,4] float32/float64 boxes without mutating their storage.
 * @param boxes Strided CPU/CUDA box tensor; noncontiguous and empty are valid.
 * @param source Input coordinate ordering.
 * @param target Output coordinate ordering.
 * @return Independently stored [N,4] tensor, preserving dtype, device and
 * graph.
 * @throws InvalidArgument Tensor metadata or a format value is invalid.
 * @throws DeviceError The tensor is on an unsupported device backend.
 *
 * Coordinates are continuous, with no clipping or normalization. Callers must
 * supply finite coordinates and nonnegative widths/heights. Numeric values
 * are not inspected on the host; invalid values propagate through arithmetic.
 * Concurrent calls are safe if callers do not mutate shared input storage.
 */
Tensor convert_boxes(const Tensor& boxes, BoxFormat source, BoxFormat target);

/**
 * @brief Computes nonnegative continuous areas of [N,4] xyxy boxes.
 * @param boxes Strided CPU/CUDA float32/float64 boxes, including empty tensors.
 * @return New [N] tensor with negative edge lengths clamped to zero; autograd,
 * dtype and device are preserved. Coordinates must be finite; no host scan.
 * @throws InvalidArgument The input metadata is invalid.
 * @throws DeviceError The device backend is unsupported.
 * @note Safe for concurrent reads; input storage is never modified.
 */
Tensor box_area(const Tensor& boxes);

/**
 * @brief Computes IoU between corresponding rows without broadcasting.
 * @param left Strided [N,4] xyxy CPU/CUDA float32/float64 boxes.
 * @param right Same shape, dtype and device as left.
 * @param epsilon Positive finite floor for union area, in squared input units.
 * @return New [N] IoU tensor; zero-area pairs have zero IoU. Keeps autograd.
 * @throws InvalidArgument Metadata, row counts or epsilon are invalid.
 * @throws DeviceError A device backend is unsupported.
 *
 * Finite coordinates are a caller precondition. Negative edge lengths are
 * clamped to zero; NaN/overflow propagate. No host scan or input mutation is
 * performed. Concurrent read-only calls are safe.
 */
Tensor aligned_box_iou(const Tensor& left, const Tensor& right,
                       double epsilon = 1e-7);

/**
 * @brief Computes the dense [N,M] intersection-over-union matrix.
 * @param left Strided [N,4] xyxy CPU/CUDA float32/float64 boxes.
 * @param right Strided [M,4] boxes with the same dtype and device.
 * @param options Pair-count bound and positive finite union-area floor.
 * @return New [N,M] tensor with graph, dtype and device preserved; empty axes
 * remain connected to both inputs' autograd graphs and allocate no pair data.
 * @throws InvalidArgument Tensor metadata or options are invalid.
 * @throws DeviceError A device backend is unsupported.
 * @throws ResourceExhausted N*M exceeds max_pairs; rejected before arithmetic.
 *
 * Numeric conventions and concurrent-read safety match aligned_box_iou().
 */
Tensor pairwise_box_iou(const Tensor& left, const Tensor& right,
                        const PairwiseBoxOptions& options = {});

/**
 * @brief Returns original row indices selected by deterministic CPU NMS.
 * @param boxes Strided CPU [N,4] float32/float64 xyxy boxes. Coordinates, edge
 * differences and areas must be finite; corners must be nondecreasing.
 * @param scores CPU [N] finite scores with the same dtype as boxes.
 * @param options Explicit comparison threshold and candidate/output bounds.
 * @return Owned CPU int64 vector ordered by descending score; equal scores
 * use ascending input index. Zero-area boxes have zero IoU. No gradients.
 * @throws InvalidArgument Metadata, coordinates, scores or options are invalid.
 * @throws DeviceError A tensor is not on CPU; there is no implicit CPU copy.
 * @throws ResourceExhausted Input exceeds max_candidates before copying.
 *
 * Class filtering/grouping belongs to the caller. Never offsets coordinates
 * by class IDs, mutates inputs or allocates a pair matrix. Concurrent calls
 * are safe when input storage is not concurrently modified.
 */
Tensor nms(const Tensor& boxes, const Tensor& scores,
           const NmsOptions& options = {});

}  // namespace jyppx::liuding::vision_ops

#endif  // JYPPX_LIUDING_VISION_OPS_BOXES_HPP
