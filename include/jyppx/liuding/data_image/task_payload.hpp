// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_DATA_IMAGE_TASK_PAYLOAD_HPP
#define JYPPX_LIUDING_DATA_IMAGE_TASK_PAYLOAD_HPP
/** @file task_payload.hpp @brief Typed supplementary targets for one canonical image. */
#include <jyppx/liuding/core/value_schema.hpp>
#include <jyppx/liuding/data_image/resource_budget.hpp>
#include <memory>
#include <string>
#include <vector>
namespace jyppx::liuding::data::image {
/** @brief Numeric meaning of a depth field, independent of its display. */
enum class DepthRepresentation : std::int64_t {
  depth_meters = 0, inverse_depth = 1, relative_inverse_depth = 2
};
/** @brief Supplementary task fields sharing the existing VisionSample owner.
 * Undefined tensors mean absent fields. Values use current sample canvas
 * coordinates; semantic class IDs remain in the existing mask_target field.
 * Tensor handles are shared, not cloned. Concurrent mutation is caller-owned.
 * All tensor leaves participate in the existing batch staging/budget paths.
 */
struct VisionTaskPayload {
  std::shared_ptr<ImageResourceLeaseSet> resource_leases;
  ///< Runtime reservations retained with transformed supplementary tensors.
  Tensor multilabel_values;  ///< Floating [C], probabilities in [0,1].
  Tensor known_labels;       ///< Bool [C], unknown differs from a negative.
  Tensor depth;              ///< Floating [1,H,W], representation below.
  Tensor depth_valid;        ///< Bool [1,H,W]; invalid values may be nonfinite.
  Tensor proposals;          ///< Floating [R,4] xyxy on the current canvas.
  Tensor proposal_scores;    ///< Optional floating [R].
  Tensor prompt_embeddings;  ///< Floating [C,D], ordered label space.
  Tensor visual_prompts;     ///< Floating [P,5] class/xyxy on current canvas.
  Tensor pretrain_mask;      ///< Bool [grid_h,grid_w], true denotes masked.
  Tensor track_ids;          ///< Int64 [objects], aligned with annotation rows.
  Tensor camera_intrinsics;  ///< Optional floating [3,3].
  std::vector<std::string> text_prompts;
  std::string label_space_id, text_encoder_id, proposal_generator;
  std::string proposal_inventory, sequence_id, camera_id;
  std::int64_t frame_id = 0;
  double timestamp_seconds = 0;
  DepthRepresentation depth_representation = DepthRepresentation::depth_meters;
  double depth_unit_scale = 1.;  ///< Stored value multiplier to declared unit.
  std::int64_t semantic_ignore = 255;
  /** @brief Stable tensor order used by typed serialization and staging. */
  std::vector<Tensor> tensors() const;
  /** @brief Validates metadata, shape and finite supervised values, bounded
   * before host synchronization. Returns aggregate logical tensor bytes. */
  std::uint64_t validate(std::uint64_t max_bytes = 1ULL << 30U,
                          std::size_t max_strings = 10000,
                          std::size_t max_string_bytes = 16384) const;
};
/** @brief Schema-checked versioned tuple; tensor leaves retain storage. */
c10::IValue task_payload_value(const VisionTaskPayload& payload);
/** @brief Reads the typed tuple and rejects unknown versions/fields. */
std::shared_ptr<const VisionTaskPayload> task_payload_from_value(const c10::IValue& value);
/** @brief Shape/type schema, without constructing model or target tensors. */
ValueSchema task_payload_schema();
}  // namespace jyppx::liuding::data::image
#endif
