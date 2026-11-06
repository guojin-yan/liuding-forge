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

#include <jyppx/liuding/data/batch_memory.hpp>
#include <jyppx/liuding/data/batch_schema.hpp>

#include <cstdint>
#include <functional>
#include <memory>
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
 * @brief Acquires an opaque aggregate-resource lease for one batch value.
 *
 * The data pipeline owns the returned object while a batch is prefetched or
 * delivered. The callback is intentionally type-erased so the codec/image
 * adapter can account for annotation, pixel, pinned, GPU, and queue resources
 * without making the codec-free data component depend on `data_image`.
 * Returning an empty pointer means that the value has no additional aggregate
 * reservation. The callback must be bounded, thread-safe for worker calls,
 * and honor the stop token.
 */
using BatchResourceLeaseFactory = std::function<std::shared_ptr<void>(
    const c10::IValue&, const BatchMemoryMetadata&, std::stop_token)>;

/** @brief Pipeline boundary whose retained output needs resource accounting. */
enum class BatchResourceStage : std::uint8_t {
  produced,  ///< Factory output, before optional memory preparation.
  prepare,   ///< Projected pinning output, before the first allocation.
  transfer,  ///< Projected destination output, before the first allocation.
  consumer_output,  ///< Callback output, before replacing the input value.
};

/**
 * @brief Accounts source values and admits known memory-stage outputs.
 *
 * produced and consumer_output receive the actual output. prepare and
 * transfer receive the unchanged source and memory policy after Core has
 * validated every leaf, before any destination allocation. The old value and
 * its lease remain alive while the callback reserves a replacement. Returning
 * an empty owner retains the source lease only for prepare and transfer. For
 * produced and consumer_output it means the output needs no reservation.
 * Callbacks must be thread-safe, bounded and respond to cancellation; avoid
 * waiting while retaining resources needed to satisfy the request. Consumer
 * callbacks allocate their own output before consumer_output; this factory
 * cannot admit arbitrary callback scratch or allocations in advance.
 * This runtime-only policy is mutually exclusive with
 * BatchResourceLeaseFactory.
 */
using BatchStageResourceFactory = std::function<std::shared_ptr<void>(
    const c10::IValue&, BatchResourceStage, const BatchMemoryOptions&,
    std::stop_token)>;

/**
 * @brief Admits one batch before its source or collator starts allocating.
 *
 * The pipeline retains the returned lease through the delivered envelope.
 * The callback may wait for capacity, but must honor the stop token so close,
 * epoch replacement, and consumer timeout can cancel blocked producers.
 * Map workers invoke it in batch sequence order. This runtime-only policy is
 * separate from the post-production resource lease factory.
 */
using BatchAdmissionFactory =
    std::function<std::shared_ptr<void>(std::stop_token)>;

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
