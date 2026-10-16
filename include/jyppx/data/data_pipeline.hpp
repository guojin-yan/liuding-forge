// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_DATA_DATA_PIPELINE_HPP
#define JYPPX_DATA_DATA_PIPELINE_HPP

/**
 * @file data_pipeline.hpp
 * @brief Defines the explicit-epoch, cursor-aware map-style data pipeline.
 */

#include <jyppx/data/samplers/sampler.hpp>
#include <jyppx/foundation/checkpoint.hpp>

#include <ATen/core/ivalue.h>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <vector>

namespace jyppx::data {

/**
 * @brief Type-erased batch produced by a DataPipeline.
 *
 * The data module intentionally carries an opaque `c10::IValue` and a stable
 * schema label instead of depending on the nn invocation schema. A later
 * mapper may validate the value against the consumer-owned schema.
 */
struct BatchEnvelope {
  c10::IValue value;      ///< Collated batch value owned by the consumer.
  std::string schema_id;  ///< Stable producer schema label.
  std::vector<std::uint64_t>
      sample_ids;  ///< Dataset-local IDs in the committed batch order.
  std::string dataset_version;  ///< Immutable dataset lineage identifier.
  std::uint64_t epoch = 0;      ///< Logical epoch that produced this batch.
  std::uint64_t sequence = 0;   ///< Zero-based committed batch sequence.
};

/**
 * @brief Reports the capabilities of a configured DataPipeline.
 */
struct DataCapability {
  bool map_style = true;  ///< The source is addressed by finite sample indices.
  bool deterministic_epoch = false;  ///< Epoch selection is reproducible.
  bool in_epoch_cursor = true;       ///< Batch commit cursor is checkpointable.
  bool worker_prefetch = false;      ///< Worker prefetch is not in this target.
  RestoreLevel restore_level =
      RestoreLevel::bounded;  ///< Strongest declared restore guarantee.
  std::vector<std::string>
      degradation_reasons;  ///< Reasons the guarantee is weaker than exact.
};

/**
 * @brief Configures a map-style DataPipeline.
 */
struct DataPipelineOptions {
  std::string pipeline_id = "data.pipeline";  ///< Stable participant ID.
  std::string dataset_version;   ///< Required immutable dataset version.
  std::size_t dataset_size = 0;  ///< Positive legal sample-index upper bound.
  std::string schema_id =
      "data.batch/v1";         ///< Stable batch value schema label.
  std::size_t batch_size = 1;  ///< Samples committed by one call to next().
  bool drop_last = false;  ///< Whether to discard an incomplete final batch.
};

/**
 * @brief Storage-neutral cursor state for a prepared or active epoch.
 */
struct DataPipelineState {
  static constexpr std::uint32_t format_version =
      1;  ///< Current state schema version.

  std::uint32_t version = format_version;  ///< Serialized state version.
  std::string pipeline_id;       ///< Participant identity at snapshot time.
  std::string dataset_version;   ///< Dataset lineage identity at snapshot time.
  std::size_t dataset_size = 0;  ///< Exclusive upper bound for sample indices.
  std::string schema_id;         ///< Batch schema label at snapshot time.
  std::size_t batch_size = 0;    ///< Batch size at snapshot time.
  bool drop_last = false;        ///< Tail policy at snapshot time.
  bool prepared = false;    ///< Whether an epoch has been selected explicitly.
  std::uint64_t epoch = 0;  ///< Selected logical epoch.
  std::uint64_t next_cursor = 0;    ///< Next uncommitted sample position.
  std::uint64_t next_sequence = 0;  ///< Sequence assigned to the next batch.
  bool sampler_state_supported =
      false;  ///< Whether sampler state gives exact continuation.
  std::string sampler_identity;  ///< Stable immutable sampler configuration.
  std::string sampler_state;     ///< Opaque sampler state at epoch preparation.
  std::vector<std::size_t>
      epoch_indices;  ///< Materialized order for the selected epoch.
};

/**
 * @brief Converts an ordered index span into one opaque batch value.
 * @param indices Dataset-local indices for the next batch.
 * @return Collated value placed in a BatchEnvelope.
 */
using DataBatchFactory =
    std::function<c10::IValue(std::span<const std::size_t> indices)>;

/**
 * @brief Explicit-epoch, cursor-aware map-style data pipeline.
 *
 * `set_epoch()` is the only operation that materializes a new sampler order.
 * `next()` never advances to another epoch implicitly. A successful `next()`
 * is the commit point for one batch; a factory exception leaves the cursor
 * unchanged. This first target is single-threaded and has no prefetch queue,
 * so its in-epoch checkpoint is exact when the sampler exposes state and a
 * stable configuration identity.
 */
class DataPipeline final : public CheckpointParticipant {
 public:
  /**
   * @brief Creates a pipeline over a caller-owned sampler.
   * @param sampler Non-null sampler producing dataset-local indices.
   * @param options Validated identity, lineage, schema and batching options.
   * @param batch_factory Collates one ordered index span into a value.
   * @throws InvalidArgument A required object or option is invalid.
   */
  DataPipeline(std::shared_ptr<samplers::Sampler> sampler,
               DataPipelineOptions options, DataBatchFactory batch_factory);

  /** @brief Releases the pipeline without advancing sampler state. */
  ~DataPipeline() override;

  DataPipeline(const DataPipeline&) = delete;  ///< Pipeline is non-copyable.
  DataPipeline& operator=(const DataPipeline&) = delete;  ///< Non-copyable.

  /**
   * @brief Materializes exactly one new epoch and resets its cursor.
   * @param epoch Non-negative logical epoch number.
   * @throws InvalidArgument If the sampler produces an invalid epoch shape.
   */
  void set_epoch(std::uint64_t epoch);

  /**
   * @brief Commits and returns the next batch in the selected epoch.
   * @param stop_token Cancellation requested by the consumer.
   * @return A committed batch, or empty at epoch exhaustion/cancellation.
   * @throws InvalidArgument No epoch has been selected yet.
   */
  std::optional<BatchEnvelope> next(std::stop_token stop_token = {});

  /**
   * @brief Reports the configured pipeline capabilities.
   * @return Capability declaration owned by this pipeline.
   */
  DataCapability capability() const;

  /**
   * @brief Returns a storage-neutral snapshot without changing state.
   * @return Current cursor and materialized epoch order.
   */
  DataPipelineState state() const;

  /** @brief Returns the stable checkpoint participant identity. */
  std::string participant_id() const override;

  /** @brief Returns the current state manifest and restore guarantee. */
  StateManifest describe_state() const override;

  /**
   * @brief Saves topology and cursor fields atomically from one snapshot.
   * @param writer Storage-neutral destination owned by the coordinator.
   */
  void save_state(StateWriter& writer) const override;

  /**
   * @brief Restores a validated cursor without changing it on failure.
   * @param reader Storage-neutral source owned by the coordinator.
   * @param policy Weakest accepted restore guarantee.
   * @throws CompatibilityError, SerializationError, or ResourceExhausted on
   *     incompatible or malformed state.
   */
  void load_state(const StateReader& reader, RestorePolicy policy) override;

 private:
  /** @brief Validates a decoded state against immutable pipeline options. */
  void validate_state(const DataPipelineState& state) const;

  std::shared_ptr<samplers::Sampler>
      sampler_;  ///< Sampler whose state owns future epoch ordering.
  DataPipelineOptions options_;     ///< Immutable pipeline configuration.
  DataBatchFactory batch_factory_;  ///< Collator invoked at batch commit.
  DataPipelineState state_;         ///< Current epoch order and commit cursor.
};

}  // namespace jyppx::data

#endif  // JYPPX_DATA_DATA_PIPELINE_HPP
