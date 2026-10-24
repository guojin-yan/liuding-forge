// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_DATA_BATCH_TRANSFORM_HPP
#define JYPPX_LIUDING_DATA_BATCH_TRANSFORM_HPP
/**
 * @file batch_transform.hpp
 * @brief Declares consumer-side transforms executed before batch commit.
 */

#include <jyppx/liuding/data/batch_schema.hpp>

#include <cstdint>
#include <functional>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>

namespace jyppx::liuding::data {

/**
 * @brief Stable logical identity of the batch being delivered.
 *
 * Views are valid only during the callback. Derive randomness from immutable
 * transform/source identity and this context; process-global mutable RNG and
 * physical thread IDs do not provide exact replay.
 */
struct BatchConsumerContext {
  std::string_view dataset_version;  ///< Immutable source lineage.
  std::uint64_t epoch = 0;           ///< Selected logical epoch.
  std::uint64_t sequence = 0;        ///< Next uncommitted batch sequence.
  std::span<const std::uint64_t> sample_ids;  ///< Ordered stable source IDs.
  std::string_view
      source_watermark;  ///< Candidate iterable position, or empty.
};

/**
 * @brief Optional synchronous consumer transform, after transfer and before
 * commit.
 *
 * `input_schema` describes the value after H2D; pipeline `schema` describes the
 * returned value. `id` must bind the implementation and all configuration,
 * including randomness policies. Callbacks run only on the next() caller and
 * must honor its stop token, preserve caller stream ordering, treat input
 * storage/containers as read-only, and avoid external side effects. They must
 * not reenter the pipeline. The framework cannot undo user side effects or
 * asynchronous device failures observed after commit.
 *
 * Only committed source position is checkpointed. Mutable callback state is
 * not saved. Mark a callback nondeterministic unless context and immutable
 * configuration alone reproduce its values. Resource checks bound logical
 * input/output payloads, not callback scratch allocation, global memory, or
 * RSS.
 */
struct BatchConsumerTransform {
  std::string id;  ///< Required stable implementation/configuration identity.
  BatchSchema input_schema = BatchSchema::any(
      "data.consumer.input");  ///< Post-transfer input contract.
  bool deterministic = true;   ///< Whether replay reproduces the output value.
  std::function<c10::IValue(const c10::IValue&, const BatchConsumerContext&,
                            std::stop_token)>
      apply;  ///< Required callback producing the final consumer value.
};

}  // namespace jyppx::liuding::data

#endif  // JYPPX_LIUDING_DATA_BATCH_TRANSFORM_HPP
