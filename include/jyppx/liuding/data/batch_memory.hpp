// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_DATA_BATCH_MEMORY_HPP
#define JYPPX_LIUDING_DATA_BATCH_MEMORY_HPP
/**
 * @file batch_memory.hpp
 * @brief Defines bounded host pinning and device staging for batch values.
 */

#include <jyppx/liuding/core/types.hpp>

#include <ATen/core/ivalue.h>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace jyppx::liuding::data {

/**
 * @brief Identifies who owns the stream used for a device transfer.
 *
 * Data pipelines prepare and optionally pin values on background threads, but
 * transfer them only while `next()` runs on the consumer. This prevents a
 * user batch factory from accidentally selecting the publication stream by
 * changing thread-local backend state.
 */
enum class BatchTransferStreamPolicy : std::uint8_t {
  caller_current =
      0,  ///< Uses the backend current stream of the transfer API caller.
};

/**
 * @brief Configures the optional memory stage applied before batch commit.
 *
 * The stage recursively visits tensors in tuples, homogeneous lists, and
 * dictionaries. Host pinning occurs before an optional device transfer so a
 * CUDA transfer may use LibTorch's asynchronous path. Resource limits are
 * checked for the complete value before the first tensor allocation.
 */
struct BatchMemoryOptions {
  bool pin_memory = false;  ///< Pins CPU tensors before optional transfer.
  std::optional<Device>
      target_device;  ///< Destination device; empty preserves each device.
  bool non_blocking = false;  ///< Requests non-blocking device transfers.
  std::size_t stage_capacity =
      2;  ///< Maximum staged but uncommitted batches in a pipeline.
  std::size_t max_tensors =
      4096;  ///< Maximum tensor occurrences accepted in one batch value.
  std::size_t max_value_nodes =
      65536;  ///< Maximum recursively visited IValue nodes per batch.
  std::uint64_t max_bytes =
      1ULL << 30U;  ///< Maximum logical tensor bytes accepted per batch.
};

/**
 * @brief Reports the work performed by one batch memory stage invocation.
 */
struct BatchMemoryMetadata {
  bool staged = false;  ///< Whether the optional memory stage was enabled.
  bool pin_memory_requested = false;    ///< Requested host-pinning policy.
  bool non_blocking_requested = false;  ///< Requested transfer scheduling.
  std::optional<Device>
      target_device;  ///< Configured destination copied into the envelope.
  BatchTransferStreamPolicy transfer_stream_policy =
      BatchTransferStreamPolicy::caller_current;  ///< Device stream owner.
  std::size_t tensor_count = 0;  ///< Tensor occurrences in the batch value.
  std::size_t value_node_count =
      0;  ///< Recursively visited values, including container roots.
  std::uint64_t tensor_bytes = 0;  ///< Total logical bytes across occurrences.
  std::size_t pinned_tensor_count =
      0;  ///< CPU tensor occurrences pinned by or before this stage.
  std::size_t transferred_tensor_count =
      0;  ///< Tensor occurrences moved to a different device.
};

/**
 * @brief Owns a recursively staged value and its observable memory metadata.
 */
struct BatchMemoryStageResult {
  c10::IValue value;             ///< Value with supported tensor leaves staged.
  BatchMemoryMetadata metadata;  ///< Bounded work summary for the envelope.
};

/**
 * @brief Reports whether host pinning or device movement is configured.
 * @param options Memory-stage configuration to inspect.
 * @return `true` when the pipeline must invoke the memory stage.
 */
bool batch_memory_stage_enabled(const BatchMemoryOptions& options) noexcept;

/**
 * @brief Validates memory-stage policy and resource limits.
 * @param options Configuration supplied to a pipeline or direct stage call.
 * @throws InvalidArgument If a limit is zero or non-blocking has no target.
 */
void validate_batch_memory_options(const BatchMemoryOptions& options);

/**
 * @brief Validates a batch and optionally pins its CPU tensor leaves.
 *
 * This phase never performs the configured device transfer. When a target is
 * configured, every source tensor must be on the CPU. It is safe to run on a
 * background producer because the returned value owns every pinned tensor
 * until the consumer performs the transfer phase.
 *
 * @param value Batch value to validate and optionally pin.
 * @param options Validated pinning, transfer, capacity, and resource policy.
 * @return Prepared value and complete pre-transfer resource metadata.
 * @throws InvalidArgument For unsupported values or tensor layouts.
 * @throws ResourceExhausted If tensor, node, or byte limits are exceeded.
 * @throws DeviceError If LibTorch cannot pin a requested CPU tensor.
 */
BatchMemoryStageResult prepare_batch_memory(const c10::IValue& value,
                                            const BatchMemoryOptions& options);

/**
 * @brief Transfers a prepared batch on the calling thread's current stream.
 *
 * The function revalidates options, metadata, resource counts, and pinning
 * before allocating a destination tensor. Pipelines call it from `next()` so
 * the training consumer, rather than a background worker, owns ordering on the
 * backend current stream. No hidden stream or implicit cross-stream event is
 * created.
 *
 * @param prepared Value returned by `prepare_batch_memory`.
 * @param options Same immutable policy used during preparation.
 * @return Consumer-ready value and finalized transfer metadata.
 * @throws CompatibilityError If metadata and options do not match.
 * @throws InvalidArgument If the prepared value violates the stage contract.
 * @throws ResourceExhausted If revalidation exceeds a resource bound.
 * @throws DeviceError If LibTorch cannot perform the requested transfer.
 */
BatchMemoryStageResult transfer_batch_memory(BatchMemoryStageResult prepared,
                                             const BatchMemoryOptions& options);

/**
 * @brief Pins and/or transfers every supported tensor leaf in a batch value.
 *
 * Supported values are tensors, tuples (including named tuples), homogeneous
 * lists, dictionaries with scalar keys, and scalar/None leaves. Undefined or
 * non-strided tensors and opaque IValue composites are rejected explicitly.
 * A configured target device requires CPU source tensors so publication never
 * inherits an untracked producer stream.
 * The input IValue and its containers are not mutated. Device transfer uses
 * the calling thread's backend current stream. This convenience function is
 * equivalent to `transfer_batch_memory(prepare_batch_memory(...), options)`.
 *
 * @param value Batch value to inspect and stage.
 * @param options Validated pinning, transfer, capacity, and resource policy.
 * @return Independently owned staged value plus per-batch memory metadata.
 * @throws InvalidArgument For unsupported values or tensor layouts.
 * @throws ResourceExhausted If tensor count or logical bytes exceed limits.
 * @throws DeviceError If LibTorch cannot pin or transfer a tensor.
 */
BatchMemoryStageResult stage_batch_memory(const c10::IValue& value,
                                          const BatchMemoryOptions& options);

}  // namespace jyppx::liuding::data

#endif  // JYPPX_LIUDING_DATA_BATCH_MEMORY_HPP