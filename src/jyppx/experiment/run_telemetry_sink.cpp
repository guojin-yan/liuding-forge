// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/experiment/run_telemetry_sink.hpp>
#include <jyppx/foundation/error.hpp>
#include <jyppx/foundation/sha256.hpp>

#include <array>
#include <limits>
#include <utility>

namespace jyppx::experiment {
namespace {

std::string operation_batch_key(std::span<const TelemetryEnvelope> envelopes,
                                const TelemetryLimits& limits) {
  Sha256 hash;
  constexpr std::string_view domain = "jyppx.sqlite-telemetry-operation/v1";
  hash.update(std::as_bytes(std::span(domain.data(), domain.size())));
  std::array<std::byte, 8U> count{};
  const auto size = static_cast<std::uint64_t>(envelopes.size());
  for (std::size_t index = 0U; index < count.size(); ++index)
    count[index] = static_cast<std::byte>(size >> ((7U - index) * 8U));
  hash.update(count);
  for (const auto& envelope : envelopes) {
    const auto identity =
        telemetry_idempotency_digest(envelope.identity, limits).hex;
    hash.update(std::as_bytes(std::span(identity.data(), identity.size())));
  }
  return sha256_digest_hex(hash.finish());
}

}  // namespace

SQLiteRunTelemetrySink::SQLiteRunTelemetrySink(
    std::shared_ptr<SQLiteRunStore> store, std::string run_id,
    std::string sink_id, std::string destination_id)
    : store_(std::move(store)),
      run_id_(std::move(run_id)),
      descriptor_{.schema_version = 1U,
                  .type_id = "jyppx.telemetry-sink.sqlite/v1",
                  .sink_id = std::move(sink_id),
                  .destination_id = std::move(destination_id),
                  .role = TelemetrySinkRole::evidence} {
  validate_telemetry_sink_descriptor(descriptor_);
  if (!store_) throw InvalidArgument("SQLite telemetry sink requires a store");
  if (!store_->find(run_id_))
    throw InvalidArgument("SQLite telemetry sink requires an existing run");
}

const TelemetrySinkDescriptor& SQLiteRunTelemetrySink::descriptor()
    const noexcept {
  return descriptor_;
}

TelemetryEmitResult SQLiteRunTelemetrySink::emit(
    std::span<const TelemetryEnvelope> envelopes, std::stop_token stop,
    TelemetryDeadline deadline) {
  check_telemetry_operation(stop, deadline);
  if (closed_) throw NotReady("SQLite telemetry sink is closed");
  const auto& limits = store_->event_limits().telemetry;
  validate_telemetry_batch(envelopes, limits);
  // This conservative overflow check precedes durable effects. The store
  // performs the authoritative validation against its configured limits.
  if (envelopes.size() >
      std::numeric_limits<std::uint64_t>::max() - counters_.exported_events)
    throw ResourceExhausted("SQLite telemetry sink counter exhausted");
  TelemetryEmitResult result;
  result.dispositions.reserve(envelopes.size());
  const auto stored =
      store_->append(run_id_, operation_batch_key(envelopes, limits), envelopes,
                     stop, deadline);
  for (const auto& receipt : stored.receipts) {
    result.dispositions.push_back(receipt.disposition);
    if (receipt.disposition == TelemetryDisposition::accepted)
      ++counters_.exported_events;
  }
  return result;
}

TelemetryFlushResult SQLiteRunTelemetrySink::flush(std::stop_token stop,
                                                   TelemetryDeadline deadline) {
  check_telemetry_operation(stop, deadline);
  return counters_;
}

TelemetryFlushResult SQLiteRunTelemetrySink::close(std::stop_token stop,
                                                   TelemetryDeadline deadline) {
  const auto result = flush(stop, deadline);
  closed_ = true;
  return result;
}

}  // namespace jyppx::experiment
