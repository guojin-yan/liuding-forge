// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

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

BatchMemoryUsage run_consumer_transform(c10::IValue& value,
                                        const BatchConsumerTransform& transform,
                                        const BatchConsumerContext& context,
                                        const BatchMemoryOptions& limits,
                                        std::stop_token stop) {
  static_cast<void>(inspect_batch_memory(value, limits));
  validate_batch_value(value, transform.input_schema);
  if (stop.stop_requested()) throw Cancelled("Consumer transform cancelled");
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
  value = std::move(output);
  return usage;
}

std::shared_ptr<void> run_resource_lease_factory(
    const BatchResourceLeaseFactory& factory, const c10::IValue& value,
    const BatchMemoryMetadata& metadata, std::stop_token stop) {
  if (!factory) return {};
  if (stop.stop_requested()) throw Cancelled("Batch resource lease cancelled");
  try {
    auto lease = factory(value, metadata, stop);
    if (stop.stop_requested()) {
      lease.reset();
      throw Cancelled("Batch resource lease cancelled");
    }
    return lease;
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

}  // namespace jyppx::liuding::data::detail
