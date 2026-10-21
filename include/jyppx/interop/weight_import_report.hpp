// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_INTEROP_WEIGHT_IMPORT_REPORT_HPP
#define JYPPX_INTEROP_WEIGHT_IMPORT_REPORT_HPP

/**
 * @file weight_import_report.hpp
 * @brief Plans complete weight mappings without accessing or mutating tensors.
 */

#include <jyppx/foundation/artifact.hpp>
#include <jyppx/interop/safetensors.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace jyppx::interop {

/** @brief Explicit policy for matching imported state to destination state. */
enum class WeightImportMode : std::uint8_t {
  strict = 1,   ///< Exact keys, shapes, and dtypes; conversion is forbidden.
  partial = 2,  ///< Explicitly permit omissions and declared dtype conversions.
};

/** @brief Stable rejection/retention reasons for unassigned destination state.
 */
enum class WeightSkipReason : std::uint8_t {
  missing = 1,           ///< No source key has this destination identity.
  shape_mismatch = 2,    ///< The source dimensions differ from the destination.
  dtype_mismatch = 3,    ///< No exact declared dtype conversion authorizes it.
  duplicate_source = 4,  ///< The source identity is ambiguous; always rejected.
  duplicate_target = 5,  ///< The destination identity is ambiguous; rejected.
};

/** @brief One storage-neutral named state entry in a complete mapping. */
struct WeightTensorDescriptor {
  std::string name;  ///< Exact UTF-8 logical key; no prefix stripping occurs.
  SafeTensorDtype dtype = SafeTensorDtype::f32;  ///< Scalar encoding identity.
  std::vector<std::uint64_t> shape;  ///< Empty means scalar; zeros are valid.
};

/**
 * @brief An explicit, per-key dtype conversion supplied by a partial adapter.
 *
 * The planner checks the declaration only. It never executes a conversion;
 * the consuming adapter must implement this exact versioned rule or fail
 * before modifying the model. Unused or stale declarations are rejected.
 */
struct WeightConversionRule {
  std::string key;  ///< Exact shared source/destination key.
  SafeTensorDtype source_dtype =
      SafeTensorDtype::f32;  ///< Required input type.
  SafeTensorDtype target_dtype =
      SafeTensorDtype::f32;  ///< Required output type.
  std::string rule_id;  ///< Versioned machine identity, e.g. float-cast.v1.
};

/** @brief Bounded metadata work; tensor allocation is owned by the adapter. */
struct WeightImportLimits {
  std::size_t max_tensors = 65536U;  ///< Bound independently for each side.
  std::size_t max_rank = 32U;        ///< Largest source or destination rank.
  std::uint64_t max_dimension = 1ULL << 32U;        ///< Bound on one dimension.
  std::uint64_t max_tensor_elements = 1ULL << 32U;  ///< One state entry.
  std::uint64_t max_total_elements = 1ULL << 40U;   ///< Each complete side.
  std::size_t max_string_bytes = 4096U;  ///< One name or rule identifier.
  std::size_t max_total_string_bytes = 16U * 1024U * 1024U;  ///< Input text.
  std::size_t max_report_bytes = 32U * 1024U * 1024U;  ///< Encoded JSON report.
};

/** @brief Explicit import policy; defaults preserve strict loading semantics.
 */
struct WeightImportOptions {
  WeightImportMode mode = WeightImportMode::strict;  ///< Never inferred.
  double minimum_tensor_coverage =
      0.0;  ///< Partial-only count threshold [0,1].
  std::vector<WeightConversionRule> conversions;  ///< Explicit partial rules.
};

/** @brief Records both dimensions for an exact-key shape incompatibility. */
struct WeightShapeMismatch {
  std::string key;  ///< Unambiguous shared state identity.
  std::vector<std::uint64_t> source_shape;  ///< Imported dimensions.
  std::vector<std::uint64_t> target_shape;  ///< Destination dimensions.
};

/** @brief Records a dtype incompatibility and any authorized conversion. */
struct WeightDtypeMismatch {
  std::string key;  ///< Unambiguous shared state identity.
  SafeTensorDtype source_dtype = SafeTensorDtype::f32;  ///< Imported dtype.
  SafeTensorDtype target_dtype = SafeTensorDtype::f32;  ///< Destination dtype.
  std::string conversion_rule;  ///< Explicit planned rule, or empty if skipped.
};

/** @brief Names a destination entry that a partial adapter must retain. */
struct WeightRetainedTarget {
  std::string key;  ///< Exact destination identity whose value stays unchanged.
  WeightSkipReason reason = WeightSkipReason::missing;  ///< Machine reason.
};

/**
 * @brief Versioned preflight diagnostics for lineage and model bundles.
 *
 * This is a plan, not evidence that any weight was loaded. `ready` authorizes
 * staging only. The nn adapter must validate payload digests, stage all data
 * and declared conversions, quiesce mutation, recheck destination topology,
 * then commit transactionally. On failure it must leave every value unchanged.
 * This contract never silently reinitializes retained destination entries.
 */
struct WeightImportReport {
  static constexpr std::uint32_t format_version = 1U;  ///< Preflight schema.
  std::uint32_t schema_version = format_version;       ///< Serialized schema.
  ArtifactDigest source_digest;  ///< Exact independently verified input bytes.
  WeightImportMode mode = WeightImportMode::strict;  ///< Explicit request mode.
  bool ready = false;  ///< Entire mapping satisfies the declared policy.
  bool coverage_satisfied = false;  ///< Partial minimum count threshold passes.
  std::uint64_t source_tensor_count = 0U;    ///< Includes duplicate entries.
  std::uint64_t target_tensor_count = 0U;    ///< Includes duplicate entries.
  std::uint64_t source_element_count = 0U;   ///< Source bounded shape products.
  std::uint64_t target_element_count = 0U;   ///< Destination shape products.
  std::uint64_t matched_element_count = 0U;  ///< Planned destination elements.
  double tensor_coverage = 0.0;  ///< Matched keys / target entries; empty is 1.
  double element_coverage =
      0.0;  ///< Matched / target elements; zero total is 1.
  double minimum_tensor_coverage = 0.0;  ///< Requested partial count threshold.
  std::vector<std::string> matched;  ///< Assigned keys, including conversions.
  std::vector<std::string> missing;  ///< Unique target keys with no source.
  std::vector<std::string> unexpected;  ///< Unique source keys with no target.
  std::vector<std::string> duplicate_source;  ///< Ambiguous source identities.
  std::vector<std::string> duplicate_target;  ///< Ambiguous target identities.
  std::vector<WeightShapeMismatch> shape_mismatches;  ///< Incompatible shapes.
  std::vector<WeightDtypeMismatch> dtype_mismatches;  ///< Includes conversions.
  std::vector<WeightConversionRule> conversions;  ///< Planned conversions only.
  std::vector<WeightRetainedTarget> retained_targets;  ///< Unchanged values.
};

/** @brief One fully preflighted source-to-destination index assignment. */
struct WeightAssignment {
  std::string key;  ///< Exact shared identity, sorted in deterministic order.
  std::size_t source_index =
      0U;  ///< Index in the original supplied source span.
  std::size_t target_index = 0U;  ///< Index in the original destination span.
  std::string conversion_rule;    ///< Explicit adapter conversion, or empty.
};

/**
 * @brief Immutable complete mapping plan whose rejected assignments are hidden.
 *
 * Planning reads metadata only and never touches model state. Inputs may be
 * released after return. Original indices refer to the caller's supplied
 * order; adapters must retain that same ordering and stable topology to commit.
 * Concurrent const inspection is safe.
 */
class WeightImportPlan final {
 public:
  /** @brief Returns complete deterministic diagnostics, including rejections.
   * @return Owned report valid for this plan's lifetime.
   */
  const WeightImportReport& report() const noexcept;

  /** @brief Returns the complete accepted assignment list for staging.
   * @return Accepted mappings in destination-key order.
   * @throws CompatibilityError If the complete plan is rejected.
   */
  std::span<const WeightAssignment> assignments() const;

  /** @brief Encodes this validated preflight report for lineage/bundles.
   * @return Canonical UTF-8 schema-v1 JSON; never claims applied success.
   * @throws ResourceExhausted If allocation or report byte limits fail.
   */
  std::string report_json() const;

 private:
  /** @brief Retains only a fully computed report and candidate assignments.
   * @param report Complete validated diagnostics.
   * @param assignments Complete unambiguous mappings.
   * @param maximum_report_bytes Bound for subsequent report serialization.
   */
  WeightImportPlan(WeightImportReport report,
                   std::vector<WeightAssignment> assignments,
                   std::size_t maximum_report_bytes);

  /** @brief Grants the complete-mapping factory construction access. */
  friend WeightImportPlan plan_weight_import(
      const ArtifactDigest&, std::span<const WeightTensorDescriptor>,
      std::span<const WeightTensorDescriptor>, const WeightImportOptions&,
      WeightImportLimits);

  WeightImportReport report_;  ///< Owned immutable complete report.
  std::vector<WeightAssignment> assignments_;  ///< Accepted candidates only.
  std::size_t maximum_report_bytes_;  ///< Explicit encoding allocation bound.
};

/**
 * @brief Validates all source/target metadata before producing any assignments.
 * @param source_digest Digest of the exact source artifact, verified by caller.
 * @param source Complete imported key/dtype/shape inventory.
 * @param target Complete destination inventory in stable adapter order.
 * @param options Explicit strict or partial policy and conversion rules.
 * @param limits Positive bounds checked before metadata expansion.
 * @return Immutable accepted or rejected plan with a complete versioned report.
 * @throws InvalidArgument For invalid options, UTF-8, dtypes, shapes, digests,
 * or unused/stale conversion declarations.
 * @throws ResourceExhausted If metadata counts, products, or allocation exceed
 * a declared bound. Duplicate keys are reported and always reject the plan.
 *
 * Keys are matched exactly; this is not a key-migration or fuzzy-name API.
 * No tensor bytes are read and no parameter is copied, cast, or reinitialized.
 */
WeightImportPlan plan_weight_import(
    const ArtifactDigest& source_digest,
    std::span<const WeightTensorDescriptor> source,
    std::span<const WeightTensorDescriptor> target,
    const WeightImportOptions& options = {}, WeightImportLimits limits = {});

}  // namespace jyppx::interop

#endif  // JYPPX_INTEROP_WEIGHT_IMPORT_REPORT_HPP
