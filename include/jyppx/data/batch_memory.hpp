// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_DATA_BATCH_MEMORY_HPP
#define JYPPX_DATA_BATCH_MEMORY_HPP

/**
 * @file batch_memory.hpp
 * @brief Defines bounded host pinning and device staging for batch values.
 */

#include <jyppx/core/types.hpp>

#include <ATen/core/ivalue.h>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace jyppx::data {

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
 * @brief Pins and/or transfers every supported tensor leaf in a batch value.
 *
 * Supported values are tensors, tuples (including named tuples), homogeneous
 * lists, dictionaries with scalar keys, and scalar/None leaves. Undefined or
 * non-strided tensors and opaque IValue composites are rejected explicitly.
 * The input IValue and its containers are not mutated.
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

}  // namespace jyppx::data

#endif  // JYPPX_DATA_BATCH_MEMORY_HPP
