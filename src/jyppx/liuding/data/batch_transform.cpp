// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "batch_memory_detail.hpp"
#include "batch_transform_detail.hpp"

#include <jyppx/liuding/foundation/error.hpp>

#include <exception>
#include <utility>

namespace jyppx::liuding::data::detail {

void validate_consumer_transform(
    const std::optional<BatchConsumerTransform>& transform) {
  if (!transform) return;
  if (transform->id.empty() || !transform->apply) {
    throw InvalidArgument(
        "Consumer transform requires an identity and callback");
  }
  if (transform->id.size() > 4096U) {
    throw ResourceExhausted("Consumer transform identity exceeds 4096 bytes");
  }
  validate_batch_schema(transform->input_schema);
}

BatchMemoryUsage run_consumer_transform(
    c10::IValue& value, const BatchConsumerTransform& transform,
    const BatchConsumerContext& context, const BatchMemoryOptions& limits,
    std::stop_token stop, const BatchStageResourceFactory& factory,
    std::shared_ptr<void>* retained) {
  static_cast<void>(inspect_batch_memory(value, limits));
  validate_batch_value(value, transform.input_schema);
  if (stop.stop_requested()) throw Cancelled("Consumer transform cancelled");
  // Declared first so rejected/cancelled output storage dies before its lease.
  std::shared_ptr<void> output_lease;
  c10::IValue output;
  try {
    output = transform.apply(value, context, stop);
  } catch (const Error&) {
    throw;
  } catch (const std::exception& cause) {
    std::throw_with_nested(BackendError(
        "Consumer batch transform failed",
        ErrorDetails{.component = "data.consumer_transform",
                     .operation = transform.id,
                     .expected = "one complete consumer batch",
                     .actual = cause.what(),
                     .hint = "Inspect the nested callback error."}));
  } catch (...) {
    std::throw_with_nested(
        BackendError("Consumer batch transform raised a non-standard exception",
                     ErrorDetails{.component = "data.consumer_transform",
                                  .operation = transform.id}));
  }
  if (stop.stop_requested()) throw Cancelled("Consumer transform cancelled");
  const auto usage = inspect_batch_memory(output, limits);
  run_stage_resource_factory(factory, output,
                             BatchResourceStage::consumer_output, limits, stop,
                             output_lease);
  value = std::move(output);
  if (retained && factory) *retained = std::move(output_lease);
  return usage;
}

void run_stage_resource_factory(const BatchStageResourceFactory& factory,
                                const c10::IValue& value,
                                BatchResourceStage stage,
                                const BatchMemoryOptions& options,
                                std::stop_token stop,
                                std::shared_ptr<void>& retained) {
  if (!factory) return;
  // Reuse the common callback error/cancellation semantics. Publish the new
  // owner before checking cancellation so unwinding preserves its payload.
  run_resource_lease_factory(
      [&](const c10::IValue& input, const BatchMemoryMetadata&,
          std::stop_token token) {
        return factory(input, stage, options, token);
      },
      value, {}, stop, retained);
}

void run_pipeline_memory(c10::IValue& value, BatchMemoryMetadata& metadata,
                         const BatchMemoryOptions& options,
                         const BatchStageResourceFactory& factory,
                         BatchResourceStage stage, std::stop_token stop,
                         std::shared_ptr<void>& retained) {
  std::shared_ptr<void> output_lease;
  BatchMemoryAdmission admit;
  if (factory)
    admit = [&] {
      run_stage_resource_factory(factory, value, stage, options, stop,
                                 output_lease);
    };
  // Keep the source handle and reservation until all backend temporaries have
  // been released and the complete replacement is ready.
  auto output =
      stage == BatchResourceStage::prepare
          ? prepare_batch_memory_with_admission(value, options, admit)
          : transfer_batch_memory_with_admission(
                {.value = value, .metadata = metadata}, options, admit);
  value = std::move(output.value);
  metadata = output.metadata;
  // An opaque lease may own a control block with a null stored pointer.
  if (output_lease.use_count() != 0) retained = std::move(output_lease);
}

void run_resource_lease_factory(const BatchResourceLeaseFactory& factory,
                                const c10::IValue& value,
                                const BatchMemoryMetadata& metadata,
                                std::stop_token stop,
                                std::shared_ptr<void>& retained) {
  if (!factory) return;
  if (stop.stop_requested()) throw Cancelled("Batch resource lease cancelled");
  try {
    // Publish ownership before observing cancellation: unwinding must keep
    // this reservation until the caller has destroyed the retained payload.
    retained = factory(value, metadata, stop);
    if (stop.stop_requested()) {
      throw Cancelled("Batch resource lease cancelled");
    }
  } catch (const Error&) {
    throw;
  } catch (const std::exception& cause) {
    std::throw_with_nested(BackendError(
        "Batch resource lease factory failed",
        ErrorDetails{.component = "data.resource_lease",
                     .operation = "acquire",
                     .expected = "one bounded opaque lease",
                     .actual = cause.what(),
                     .hint = "Inspect the nested adapter error."}));
  } catch (...) {
    std::throw_with_nested(BackendError(
        "Batch resource lease factory raised a non-standard exception",
        ErrorDetails{.component = "data.resource_lease",
                     .operation = "acquire"}));
  }
}

std::shared_ptr<void> run_admission_factory(
    const BatchAdmissionFactory& factory, std::stop_token stop) {
  if (!factory) return {};
  if (stop.stop_requested()) throw Cancelled("Batch admission cancelled");
  try {
    auto lease = factory(stop);
    if (stop.stop_requested()) throw Cancelled("Batch admission cancelled");
    return lease;
  } catch (const Error&) {
    throw;
  } catch (const std::exception& cause) {
    std::throw_with_nested(
        BackendError("Batch admission factory failed",
                     ErrorDetails{.component = "data.batch_admission",
                                  .operation = "acquire",
                                  .actual = cause.what()}));
  } catch (...) {
    std::throw_with_nested(
        BackendError("Batch admission factory raised a non-standard exception",
                     ErrorDetails{.component = "data.batch_admission",
                                  .operation = "acquire"}));
  }
}

}  // namespace jyppx::liuding::data::detail
