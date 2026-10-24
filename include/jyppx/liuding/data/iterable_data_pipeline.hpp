// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_DATA_ITERABLE_DATA_PIPELINE_HPP
#define JYPPX_LIUDING_DATA_ITERABLE_DATA_PIPELINE_HPP
/**
 * @file iterable_data_pipeline.hpp
 * @brief Defines a checkpoint-aware pipeline for sequential data sources.
 */

#include <jyppx/liuding/data/data_pipeline.hpp>
#include <jyppx/liuding/data/iterable_source.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stop_token>
#include <string>

namespace jyppx::liuding::data {

/**
 * @brief Configures batching and bounded prefetch for an iterable source.
 */
struct IterableDataPipelineOptions {
  std::string pipeline_id = "data.iterable_pipeline";  ///< Participant ID.
  std::string dataset_version;  ///< Required immutable dataset identity.
  BatchSchema schema = BatchSchema::any(
      "data.iterable_batch");    ///< Validated consumer batch contract.
  std::string batch_factory_id;  ///< Stable collator/configuration identity.
  std::size_t batch_size = 1;    ///< Positive number of source items per batch.
  bool drop_last = false;        ///< Drops an incomplete finite final batch.
  std::size_t prefetch_batches =
      2;  ///< Positive total produced-but-uncommitted batch capacity.
  std::chrono::milliseconds batch_timeout{
      0};  ///< Consumer wait deadline; zero waits without a deadline.
  BatchMemoryOptions
      memory;  ///< Optional bounded host-pinning and device-transfer stage.
  bool deterministic_batch_factory =
      true;  ///< Whether collator replay is stable.
  std::optional<BatchConsumerTransform>
      consumer_transform = std::nullopt;  ///< Optional post-transfer, precommit
                                          ///< consumer stage.
};

/**
 * @brief Current committed state for an iterable source pipeline.
 */
struct IterableDataPipelineState {
  static constexpr std::uint32_t format_version =
      4;  ///< Writer schema; readers retain v3 and constrained label-only v2.

  std::uint32_t version = format_version;  ///< Serialized schema version.
  std::string pipeline_id;        ///< Participant identity at snapshot time.
  std::string dataset_version;    ///< Dataset identity at snapshot time.
  std::string schema_descriptor;  ///< Complete canonical batch schema bytes.
  std::string batch_factory_id;   ///< Collator identity at snapshot time.
  bool deterministic_batch_factory = true;  ///< Producer replay declaration.
  std::string consumer_transform_id;  ///< Empty when consumer transform is off.
  std::string
      consumer_input_schema_descriptor;  ///< Full post-transfer contract.
  bool deterministic_consumer_transform =
      true;                          ///< Consumer replay declaration.
  std::string source_id;             ///< Source topology identity at snapshot.
  bool finite_epoch = true;          ///< Source finiteness at snapshot time.
  bool deterministic_epoch = false;  ///< Source determinism at snapshot time.
  bool checkpointable = false;  ///< Cursor state capability at snapshot time.
  RestoreLevel source_restore_level =
      RestoreLevel::best_effort;  ///< Source guarantee at snapshot time.
  std::optional<std::uint64_t>
      max_replayed_samples;  ///< Declared duplicate bound at snapshot time.
  std::optional<std::uint64_t>
      max_omitted_samples;     ///< Declared omission bound at snapshot time.
  std::size_t batch_size = 0;  ///< Batch size at snapshot time.
  bool drop_last = false;      ///< Tail policy at snapshot time.
  std::size_t prefetch_batches = 0;    ///< Bounded producer queue capacity.
  std::uint64_t batch_timeout_ms = 0;  ///< Encoded consumer wait timeout.
  bool pin_memory = false;    ///< Host-pinning topology at snapshot time.
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
  bool prepared = false;    ///< Whether an epoch cursor was opened.
  bool exhausted = false;   ///< Whether exhaustion was consumer-visible.
  std::uint64_t epoch = 0;  ///< Explicitly selected logical epoch.
  std::uint64_t next_sequence = 0;  ///< Sequence for the next visible batch.
  std::uint64_t committed_samples =
      0;  ///< Number of source items in committed visible batches.
  std::string source_watermark;  ///< Last committed opaque source watermark.
  std::string source_state;      ///< Cursor state at the committed boundary.
};

/**
 * @brief Collates ordered iterable samples into one opaque batch value.
 * @param samples Ordered source items assigned to the next batch.
 * @param epoch Explicit logical epoch selected by the coordinator.
 * @param sequence Batch sequence within the selected epoch.
 * @param stop Cooperative cancellation for source and collator work.
 * @return Collated value stored in the common `BatchEnvelope`.
 */
using IterableBatchFactory = std::function<c10::IValue(
    std::span<const IterableSample> samples, std::uint64_t epoch,
    std::uint64_t sequence, std::stop_token stop)>;

/**
 * @brief Sequential-source pipeline with bounded producer backpressure.
 *
 * This class does not emulate random access, sampler indices, or a synthetic
 * size. One producer thread owns one isolated source cursor and may prefetch at
 * most `prefetch_batches` completed or in-flight batches. Only successful
 * consumer delivery advances checkpoint state. Exact/bounded cursor state is
 * captured after each produced batch; best-effort restore reopens the saved
 * epoch and explicitly permits unbounded replay or omission.
 *
 * Public methods are single-owner and must not be called concurrently. Source
 * cursors and the batch factory must honor their stop token/close contract so
 * cancellation and destruction can join the producer.
 */
class IterableDataPipeline final : public BatchPipeline {
 public:
  /**
   * @brief Creates an unprepared pipeline over an immutable source factory.
   * @param source Non-null factory for isolated epoch cursors.
   * @param options Validated identity, batching, queue, and timeout options.
   * @param batch_factory Non-empty ordered sample collator.
   * @throws InvalidArgument or ResourceExhausted for invalid contracts.
   */
  IterableDataPipeline(std::shared_ptr<const IterableSource> source,
                       IterableDataPipelineOptions options,
                       IterableBatchFactory batch_factory);

  /** @brief Cancels and joins the producer before releasing its cursor. */
  ~IterableDataPipeline() override;

  IterableDataPipeline(const IterableDataPipeline&) =
      delete;  ///< Pipeline state is unique and non-copyable.
  IterableDataPipeline& operator=(const IterableDataPipeline&) =
      delete;  ///< Pipeline state is unique and non-copyable.

  /**
   * @brief Opens exactly one replacement cursor for a logical epoch.
   * @param epoch Explicit non-negative logical epoch number.
   * @throws Error If cursor creation or initial state capture fails.
   */
  void set_epoch(std::uint64_t epoch) override;

  /**
   * @brief Commits and returns the next source batch.
   * @param stop_token Consumer cancellation request.
   * @return A committed batch, or empty for cancellation/exhaustion.
   * @throws InvalidArgument If no epoch has been selected.
   * @throws Cancelled If the active producer epoch was cancelled earlier.
   * @throws DeadlineExceeded If `batch_timeout` expires.
   * @throws BackendError If source or collator user code fails.
   */
  std::optional<BatchEnvelope> next(std::stop_token stop_token = {}) override;

  /**
   * @brief Permanently closes the cursor and joins the producer.
   *
   * The operation is idempotent and never starts another source operation.
   */
  void close() noexcept override;

  /**
   * @brief Returns whether permanent close has begun.
   * @return `true` after `close()` permanently disables the pipeline.
   */
  bool closed() const noexcept override;

  /**
   * @brief Returns produced or in-flight batches occupying queue capacity.
   * @return Zero when unprepared, exhausted, failed, cancelled, or closed.
   */
  std::size_t outstanding_batches() const noexcept;

  /**
   * @brief Returns immutable construction options.
   * @return A reference valid for the pipeline lifetime.
   */
  const IterableDataPipelineOptions& options() const noexcept;

  /**
   * @brief Reports source, prefetch, and restore capabilities.
   * @return Immutable source guarantees and runtime queue configuration.
   */
  DataCapability capability() const override;

  /**
   * @brief Returns the last consumer-committed source state.
   * @return Storage-neutral committed cursor snapshot.
   */
  IterableDataPipelineState state() const;

  /**
   * @brief Returns the stable checkpoint participant ID.
   * @return Configured participant identity.
   */
  std::string participant_id() const override;

  /**
   * @brief Describes current-writer topology and committed cursor fields.
   * @return Bounded manifest with the source restore guarantee.
   */
  StateManifest describe_state() const override;

  /**
   * @brief Saves only consumer-committed state, never live queue objects.
   * @param writer Storage-neutral destination owned by the coordinator.
   */
  void save_state(StateWriter& writer) const override;

  /**
   * @brief Restores through a newly opened cursor before retiring live state.
   * @param reader Storage-neutral current-schema source.
   * @param policy Weakest restore guarantee accepted by the caller.
   */
  void load_state(const StateReader& reader, RestorePolicy policy) override;

 private:
  /** @brief Hides the producer queue, cursor, codec, and committed state. */
  struct Impl;

  std::unique_ptr<Impl> impl_;  ///< Owned source contract and active runtime.
};

}  // namespace jyppx::liuding::data

#endif  // JYPPX_LIUDING_DATA_ITERABLE_DATA_PIPELINE_HPP
