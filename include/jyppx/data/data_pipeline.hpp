// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_DATA_DATA_PIPELINE_HPP
#define JYPPX_DATA_DATA_PIPELINE_HPP

/**
 * @file data_pipeline.hpp
 * @brief Defines the explicit-epoch, worker-prefetched map-style data pipeline.
 */

#include <jyppx/data/batch_memory.hpp>
#include <jyppx/data/batch_schema.hpp>
#include <jyppx/data/samplers/sampler.hpp>
#include <jyppx/foundation/capability.hpp>
#include <jyppx/foundation/checkpoint.hpp>

#include <ATen/core/ivalue.h>
#include <chrono>
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
 * The data module carries an IValue plus the complete versioned schema that
 * was validated immediately before commit. Consumers can inspect semantic,
 * dtype, device, and shape constraints without depending on the nn target.
 */
struct BatchEnvelope {
  c10::IValue value;   ///< Collated batch value owned by the consumer.
  BatchSchema schema;  ///< Complete validated producer value contract.
  std::vector<std::uint64_t>
      sample_ids;  ///< Dataset-local IDs in the committed batch order.
  std::string dataset_version;   ///< Immutable dataset lineage identifier.
  std::uint64_t epoch = 0;       ///< Logical epoch that produced this batch.
  std::uint64_t sequence = 0;    ///< Zero-based committed batch sequence.
  std::string source_watermark;  ///< Opaque iterable position; empty for maps.
  BatchMemoryMetadata
      memory;  ///< Pinning, transfer, tensor-count, and byte metadata.
};

/**
 * @brief Identifies the worker execution model used by one pipeline.
 */
enum class DataWorkerModel : std::uint8_t {
  inline_execution = 0,  ///< Collation executes on the consumer thread.
  thread_pool = 1,       ///< A fixed in-process thread pool performs collation.
  producer_thread = 2,   ///< One in-process thread owns an iterable cursor.
};

/**
 * @brief Identifies how committed batch sequences map to logical shards.
 *
 * Logical shards are checkpoint-stable identities and are independent of the
 * physical thread that happens to execute a factory invocation.
 */
enum class DataWorkerShardPolicy : std::uint8_t {
  single_shard = 0,       ///< All batches belong to one logical shard.
  round_robin_batch = 1,  ///< Batch `s` belongs to shard `s % worker_count`.
};

/**
 * @brief Reports the capabilities of a configured DataPipeline.
 */
struct DataCapability {
  bool map_style = true;  ///< The source is addressed by finite sample indices.
  bool iterable_style =
      false;  ///< The source is consumed through an independent cursor.
  bool finite_epoch = true;  ///< Whether every selected epoch must terminate.
  bool deterministic_epoch = false;  ///< Epoch selection is reproducible.
  bool in_epoch_cursor = true;       ///< Batch commit cursor is checkpointable.
  DataWorkerModel worker_model =
      DataWorkerModel::inline_execution;  ///< In-process execution topology.
  DataWorkerShardPolicy worker_shard_policy =
      DataWorkerShardPolicy::single_shard;  ///< Stable logical assignment.
  bool process_workers =
      false;  ///< Process/fork/spawn workers are not implemented.
  bool worker_prefetch = false;  ///< Whether background workers build batches.
  std::size_t worker_count = 0;  ///< Number of configured worker threads.
  std::size_t prefetch_batches =
      0;  ///< Maximum scheduled but uncommitted batches.
  bool persistent_workers =
      false;  ///< Whether worker threads survive epoch boundaries.
  bool batch_timeout = false;  ///< Whether consumer waits have a deadline.
  bool memory_stage = false;   ///< Whether pinning or transfer is configured.
  bool pin_memory = false;     ///< Whether CPU tensor pinning is requested.
  bool non_blocking_transfer =
      false;  ///< Whether device transfers request non-blocking execution.
  BatchTransferStreamPolicy transfer_stream_policy =
      BatchTransferStreamPolicy::caller_current;  ///< Transfer stream owner.
  std::optional<Device>
      target_device;  ///< Configured batch destination, when present.
  std::size_t memory_stage_capacity =
      0;  ///< Maximum staged but uncommitted batches; zero when disabled.
  std::size_t max_batch_tensors =
      0;  ///< Maximum tensor occurrences accepted by the memory stage.
  std::size_t max_batch_value_nodes =
      0;  ///< Maximum recursive IValue nodes accepted by memory staging.
  std::uint64_t max_batch_bytes =
      0;  ///< Maximum logical tensor bytes accepted by memory staging.
  bool schema_validation = true;  ///< Every consumer-visible batch is checked.
  BatchSchema batch_schema;  ///< Complete versioned producer value contract.
  std::string batch_schema_fingerprint;  ///< Compact diagnostic schema digest.
  ValueSchemaLimits schema_limits;       ///< Schema/runtime traversal bounds.
  std::size_t max_replayed_batches =
      0;  ///< Maximum uncommitted batches rebuilt after restore.
  std::optional<std::uint64_t>
      max_replayed_samples;  ///< Source-level duplicate bound, if finite.
  std::optional<std::uint64_t>
      max_omitted_samples;  ///< Source-level omission bound, if finite.
  RestoreLevel restore_level =
      RestoreLevel::bounded;  ///< Strongest declared restore guarantee.
  std::vector<std::string>
      degradation_reasons;  ///< Reasons the guarantee is weaker than exact.
};

/**
 * @brief Maps one configured pipeline capability into manifest schema v1.
 *
 * `DataCapability` remains the data-domain fact source. This conversion does
 * not infer process workers, devices, dtypes, restore guarantees, or limits
 * that the configured pipeline did not declare.
 *
 * @param capability Valid configured map or iterable pipeline capability.
 * @return Structured configured-instance descriptor for cross-domain
 * aggregation.
 * @throws InvalidArgument If style, enum, schema, or restore semantics are
 * inconsistent.
 * @throws ResourceExhausted If capability metadata exceeds manifest bounds.
 */
CapabilityDescriptor describe_data_capability(const DataCapability& capability);

/**
 * @brief Defines the common batch stream consumed by the training engine.
 *
 * Map-style and iterable-style implementations remain separate concrete
 * contracts. This interface exposes only their shared lifecycle and committed
 * batch boundary; it does not add a synthetic size, sampler, or random-access
 * operation to iterable sources. Public calls are single-owner unless a
 * concrete implementation documents a stronger guarantee.
 */
class BatchPipeline : public CheckpointParticipant {
 public:
  /** @brief Enables destruction through the shared pipeline contract. */
  ~BatchPipeline() override = default;

  /**
   * @brief Selects exactly one logical epoch and prepares its first batch.
   * @param epoch Zero-based logical epoch selected by the coordinator.
   */
  virtual void set_epoch(std::uint64_t epoch) = 0;

  /**
   * @brief Commits and returns the next batch at the pipeline boundary.
   * @param stop_token Cooperative cancellation request from the consumer.
   * @return A committed batch, or empty at cancellation or exhaustion.
   */
  virtual std::optional<BatchEnvelope> next(
      std::stop_token stop_token = {}) = 0;

  /** @brief Permanently cancels and releases active pipeline resources. */
  virtual void close() noexcept = 0;

  /**
   * @brief Reports whether permanent shutdown has begun.
   * @return `true` when no future epoch or batch operation is accepted.
   */
  virtual bool closed() const noexcept = 0;

  /**
   * @brief Reports map/iterable, prefetch, and restore capabilities.
   * @return Immutable capability declaration for the concrete pipeline.
   */
  virtual DataCapability capability() const = 0;
};

/**
 * @brief Configures a map-style DataPipeline.
 */
struct DataPipelineOptions {
  std::string pipeline_id = "data.pipeline";  ///< Stable participant ID.
  std::string dataset_version;   ///< Required immutable dataset version.
  std::size_t dataset_size = 0;  ///< Positive legal sample-index upper bound.
  BatchSchema schema =
      BatchSchema::any("data.batch");  ///< Validated consumer batch contract.
  std::string batch_factory_id;  ///< Required collator/configuration identity.
  std::size_t batch_size = 1;    ///< Samples committed by one call to next().
  bool drop_last = false;  ///< Whether to discard an incomplete final batch.
  std::size_t num_workers = 0;  ///< Background workers; zero is inline mode.
  std::size_t prefetch_batches =
      2;  ///< Positive total uncommitted-batch capacity.
  bool persistent_workers =
      false;  ///< Keeps worker threads alive between selected epochs.
  bool deterministic_batch_factory =
      true;  ///< Whether replay with the supplied context is value-stable.
  std::uint64_t worker_seed =
      0;  ///< Stable seed mixed with dataset, epoch, worker, and sequence.
  std::chrono::milliseconds batch_timeout{
      0};  ///< Positive worker wait deadline; zero disables timeout.
  BatchMemoryOptions
      memory;  ///< Optional bounded host-pinning and device-transfer stage.
};

/**
 * @brief Storage-neutral cursor state for a prepared or active epoch.
 */
struct DataPipelineState {
  static constexpr std::uint32_t format_version =
      4;  ///< Current writer schema; the reader also accepts explicit N-1 v3.

  std::uint32_t version = format_version;  ///< Serialized state version.
  std::string pipeline_id;       ///< Participant identity at snapshot time.
  std::string dataset_version;   ///< Dataset lineage identity at snapshot time.
  std::size_t dataset_size = 0;  ///< Exclusive upper bound for sample indices.
  std::string schema_descriptor;  ///< Complete canonical batch schema bytes.
  std::string batch_factory_id;   ///< Collator identity at snapshot time.
  std::size_t batch_size = 0;     ///< Batch size at snapshot time.
  bool drop_last = false;         ///< Tail policy at snapshot time.
  std::size_t num_workers = 0;    ///< Worker topology at snapshot time.
  std::size_t prefetch_batches =
      0;  ///< Maximum uncommitted work at snapshot time.
  bool persistent_workers =
      false;  ///< Worker lifecycle policy at snapshot time.
  bool deterministic_batch_factory =
      true;  ///< Whether prefetched work is exactly replayable.
  std::uint64_t worker_seed = 0;  ///< Base deterministic worker seed.
  std::uint64_t batch_timeout_ms =
      0;                    ///< Consumer wait deadline encoded as milliseconds.
  bool pin_memory = false;  ///< Host-pinning topology at snapshot time.
  std::string target_device;  ///< Empty or canonical destination device text.
  bool non_blocking_transfer =
      false;  ///< Transfer scheduling policy at snapshot time.
  std::size_t memory_stage_capacity =
      0;  ///< Staged-but-uncommitted batch bound at snapshot time.
  std::size_t max_batch_tensors =
      0;  ///< Per-batch tensor occurrence limit at snapshot time.
  std::size_t max_batch_value_nodes =
      0;  ///< Per-batch recursive value-node limit at snapshot time.
  std::uint64_t max_batch_bytes =
      0;  ///< Per-batch logical tensor byte limit at snapshot time.
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
 * @brief Provides deterministic identity and cancellation to a batch factory.
 *
 * A factory that declares deterministic behavior must derive all random work
 * from `worker_seed` instead of mutable process-global RNG state. Logical
 * worker assignment is stable even when a different physical thread executes
 * the task.
 */
struct DataWorkerContext {
  std::uint32_t worker_id = 0;     ///< Logical shard assigned to this batch.
  std::uint32_t worker_count = 1;  ///< Logical shards in the active topology.
  std::uint64_t shard_sequence =
      0;  ///< Zero-based sequence local to the assigned logical shard.
  std::uint64_t worker_seed = 0;  ///< Fully mixed per-batch deterministic seed.
  std::uint64_t epoch = 0;     ///< Explicit epoch selected by the coordinator.
  std::uint64_t sequence = 0;  ///< Batch sequence within the selected epoch.
};

/**
 * @brief Converts an ordered index span into one opaque batch value.
 * @param indices Dataset-local indices for the next batch.
 * @param context Stable worker, epoch, sequence, and seed metadata.
 * @param stop Cooperative epoch shutdown request.
 * @return Collated value placed in a BatchEnvelope.
 */
using DataBatchFactory = std::function<c10::IValue(
    std::span<const std::size_t> indices, const DataWorkerContext& context,
    std::stop_token stop)>;

/**
 * @brief Explicit-epoch, cursor-aware map-style data pipeline.
 *
 * `set_epoch()` is the only operation that materializes a new sampler order.
 * `next()` never advances to another epoch implicitly. A successful `next()`
 * is the only batch commit point; cancellation, timeout, or a factory failure
 * leaves the cursor unchanged and stops the active worker epoch. A caller may
 * then save committed state, restore it, or explicitly select a new epoch.
 *
 * In worker mode, batches may finish out of order but are observed and
 * committed in sequence order. At most `prefetch_batches` are submitted ahead
 * of the committed cursor. Checkpoints persist only committed state;
 * prefetched work is rebuilt from deterministic worker context after restore.
 * A non-deterministic factory therefore declares bounded rather than exact
 * recovery. Instances are single-owner and are not safe for concurrent public
 * method calls. The factory itself must be safe for `num_workers` concurrent
 * calls and must honor its stop token for bounded close.
 */
class DataPipeline final : public BatchPipeline {
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

  /** @brief Requests worker shutdown and releases pipeline resources. */
  ~DataPipeline() override;

  DataPipeline(const DataPipeline&) = delete;  ///< Pipeline is non-copyable.
  DataPipeline& operator=(const DataPipeline&) = delete;  ///< Non-copyable.

  /**
   * @brief Materializes exactly one new epoch and resets its cursor.
   * @param epoch Non-negative logical epoch number.
   * @throws InvalidArgument If the sampler produces an invalid epoch shape.
   */
  void set_epoch(std::uint64_t epoch) override;

  /**
   * @brief Commits and returns the next batch in the selected epoch.
   * @param stop_token Cancellation requested by the consumer.
   * @return A committed batch, or empty at epoch exhaustion/cancellation.
   * @throws InvalidArgument No epoch has been selected yet.
   * @throws Cancelled A previous wait cancelled the current worker epoch.
   * @throws DeadlineExceeded A worker batch exceeded `batch_timeout`.
   * @throws BackendError A worker batch factory raised a non-JYPPX exception.
   */
  std::optional<BatchEnvelope> next(std::stop_token stop_token = {}) override;

  /**
   * @brief Cooperatively stops workers and permanently closes the pipeline.
   *
   * The operation is idempotent. A running factory must honor its stop token;
   * C++ cannot forcibly terminate a user callback that ignores cancellation.
   */
  void close() noexcept override;

  /**
   * @brief Reports whether explicit or destructor shutdown began.
   * @return `true` after `close()` permanently disables the pipeline.
   */
  bool closed() const noexcept override;

  /**
   * @brief Returns scheduled but not yet committed worker batches.
   * @return Zero in inline, unprepared, exhausted, failed, or closed state.
   */
  std::size_t outstanding_batches() const noexcept;

  /**
   * @brief Returns immutable pipeline, worker, and timeout options.
   * @return A reference valid for the pipeline lifetime.
   */
  const DataPipelineOptions& options() const noexcept;

  /**
   * @brief Reports the configured pipeline capabilities.
   * @return Capability declaration owned by this pipeline.
   */
  DataCapability capability() const override;

  /**
   * @brief Returns a storage-neutral snapshot without changing state.
   * @return Current cursor and materialized epoch order.
   */
  DataPipelineState state() const;

  /**
   * @brief Returns the stable checkpoint participant identity.
   * @return Configured pipeline identifier used by the coordinator.
   */
  std::string participant_id() const override;

  /**
   * @brief Returns the current state manifest and restore guarantee.
   * @return Bounded topology/cursor schema and declared restore level.
   */
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
  /** @brief Hides worker queues, threads, and checkpoint codec state. */
  struct Impl;

  std::unique_ptr<Impl> impl_;  ///< Owned sampler, worker runtime, and cursor.
};

}  // namespace jyppx::data

#endif  // JYPPX_DATA_DATA_PIPELINE_HPP
