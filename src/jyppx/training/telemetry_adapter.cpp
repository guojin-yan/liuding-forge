// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/foundation/sha256.hpp>
#include <jyppx/training/telemetry_adapter.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>

namespace jyppx::training {
namespace {

[[noreturn]] void reject(std::string field) {
  throw CompatibilityError(
      "Training telemetry adaptation or N-1 migration rejected",
      ErrorDetails{
          .component = "training.telemetry_adapter",
          .operation = std::move(field),
          .expected = "valid immutable source and explicitly compatible "
                      "semantics/topology",
          .actual = "unsupported or inconsistent source; content omitted",
          .hint = "Keep the original checkpoint and validate the complete "
                  "migration candidate before installing live state."});
}

bool terminal(CallbackEventKind kind) {
  return kind == CallbackEventKind::run_finished ||
         kind == CallbackEventKind::run_failed ||
         kind == CallbackEventKind::run_cancelled;
}

bool phase_matches(CallbackEventKind kind, CallbackPhase phase) {
  switch (kind) {
    case CallbackEventKind::run_started:
      return phase == CallbackPhase::preparing;
    case CallbackEventKind::run_finished:
      return phase == CallbackPhase::completed;
    case CallbackEventKind::run_failed:
      return phase == CallbackPhase::failed;
    case CallbackEventKind::run_cancelled:
      return phase == CallbackPhase::cancelled;
    case CallbackEventKind::epoch_started:
    case CallbackEventKind::epoch_finished:
    case CallbackEventKind::optimizer_step_started:
    case CallbackEventKind::optimizer_step_finished:
      return phase == CallbackPhase::training;
    case CallbackEventKind::batch_started:
    case CallbackEventKind::batch_finished:
      return phase == CallbackPhase::training ||
             phase == CallbackPhase::validation;
    case CallbackEventKind::validation_started:
    case CallbackEventKind::validation_finished:
      return phase == CallbackPhase::validation;
    case CallbackEventKind::checkpoint_started:
    case CallbackEventKind::checkpoint_committed:
    case CallbackEventKind::checkpoint_failed:
      return phase == CallbackPhase::checkpointing;
  }
  return false;
}

bool same(const CallbackCommand& left, const CallbackCommand& right) {
  return left.kind == right.kind && left.callback_id == right.callback_id &&
         left.reason == right.reason;
}

void source_fields(const EventEnvelope& source, const TelemetryLimits& limits) {
  if (source.schema_version != 1U || source.event.schema_version != 1U ||
      source.event.event_index == 0U || source.event.run_id.empty() ||
      source.event.trace_id.empty() ||
      !phase_matches(source.event.kind, source.event.phase)) {
    reject("source.envelope");
  }
  if (source.event.metrics.size() > limits.max_metrics_per_envelope ||
      source.callback_result.commands.size() >
          limits.max_decisions_per_envelope) {
    throw ResourceExhausted(
        "Training telemetry source exceeds collection limits");
  }
  for (const auto& [name, value] : source.event.metrics) {
    if (name.empty() || !std::isfinite(value)) reject("source.metrics");
  }
  const CallbackCommand* effective = nullptr;
  std::set<std::string> actors;
  for (const auto& command : source.callback_result.commands) {
    const auto priority = static_cast<std::uint8_t>(command.kind);
    if (priority < 1U || priority > 3U || command.callback_id.empty() ||
        command.reason.empty() || !actors.insert(command.callback_id).second) {
      reject("source.commands");
    }
    if (effective == nullptr || command.kind > effective->kind)
      effective = &command;
  }
  if ((effective != nullptr) !=
          source.callback_result.effective_command.has_value() ||
      (effective != nullptr &&
       !same(*effective, *source.callback_result.effective_command))) {
    reject("source.effective_command");
  }
}

std::string derived_id(std::string_view domain, std::string_view value) {
  Sha256 hash;
  const auto append = [&](std::string_view part) {
    std::array<std::byte, 8> length{};
    const auto size = static_cast<std::uint64_t>(part.size());
    for (std::size_t index = 0U; index < length.size(); ++index) {
      length[index] = static_cast<std::byte>((size >> (index * 8U)) & 0xFFU);
    }
    hash.update(length);
    hash.update({reinterpret_cast<const std::byte*>(part.data()), part.size()});
  };
  append(domain);
  append(value);
  return sha256_digest_hex(hash.finish());
}

TelemetryEnvelope convert(const EventEnvelope& source,
                          const TrainingTelemetryAdapterOptions& options,
                          const TelemetryTime& time, bool legacy) {
  validate_telemetry_limits(options.limits);
  source_fields(source, options.limits);
  TelemetryEnvelope result;
  result.identity = {
      .stream_id = options.stream_id,
      .event_id = "training.event/" + std::to_string(source.event.event_index),
      .idempotency_key =
          "training.callback/" + std::to_string(source.event.event_index),
      .sequence = source.event.event_index};
  result.time = time;
  result.correlation = {.project_id = options.project_id,
                        .run_id = source.event.run_id,
                        .attempt_id = options.attempt_id,
                        .trace_id = source.event.trace_id,
                        .span_id = {},
                        .parent_span_id = {}};
  result.reliability = TelemetryReliability::required;
  result.attributes = {
      {"training.phase", std::string(callback_phase_name(source.event.phase))},
      {"training.epoch", std::to_string(source.event.epoch)},
      {"training.global_step", std::to_string(source.event.global_step)}};
  if (legacy) {
    result.attributes["migration.source"] = "training.event-envelope/v1";
    result.attributes["migration.time"] = "unavailable";
    result.attributes["migration.trace_id"] = source.event.trace_id;
    result.correlation.trace_id =
        "legacy.trace/" +
        derived_id("jyppx.training.trace-v1", source.event.trace_id);
  }
  TelemetryLifecycle payload;
  payload.name =
      "training." + std::string(callback_event_kind_name(source.event.kind));
  payload.terminal = terminal(source.event.kind);
  for (const auto& [name, value] : source.event.metrics) {
    TelemetryMetricPoint point;
    if (legacy) {
      point.definition = {
          .metric_id = "jyppx.legacy.training.scalar/" +
                       derived_id("jyppx.training.metric-name-v1", name),
          .unit = TelemetryMetricUnit::unspecified,
          .aggregation = TelemetryAggregation::unspecified,
          .direction = TelemetryMetricDirection::none,
          .parameters = {{"source_metric_name", name},
                         {"source_schema", "training.callback-event/v1"}},
          .context_keys = {"phase"}};
      point.context = {
          {"phase", std::string(callback_phase_name(source.event.phase))}};
    } else {
      const auto binding = options.metrics.find(name);
      if (binding == options.metrics.end()) reject("source.metric_binding");
      point.definition = binding->second.definition;
      point.context = binding->second.context;
      if (point.definition.unit == TelemetryMetricUnit::unspecified ||
          point.definition.aggregation == TelemetryAggregation::unspecified) {
        reject("source.metric_binding.semantics");
      }
    }
    point.step = {.kind = TelemetryStepKind::global_step,
                  .value = source.event.global_step};
    point.observation = make_telemetry_metric_value(value);
    payload.metrics.push_back(std::move(point));
  }
  for (const auto& command : source.callback_result.commands) {
    payload.decisions.push_back(
        {.actor_id = command.callback_id,
         .action = std::string(callback_command_kind_name(command.kind)),
         .reason = command.reason,
         .effective =
             source.callback_result.effective_command.has_value() &&
             same(command, *source.callback_result.effective_command)});
  }
  result.payload = std::move(payload);
  validate_telemetry_envelope(result, options.limits);
  return result;
}

}  // namespace

TelemetryEnvelope adapt_training_event(
    const EventEnvelope& source, const TrainingTelemetryAdapterOptions& options,
    const TelemetryTime& captured_time) {
  return convert(source, options, captured_time, false);
}

TelemetryEnvelope migrate_training_envelope_v1(const EventEnvelope& source,
                                               const TelemetryLimits& limits) {
  TrainingTelemetryAdapterOptions options;
  options.stream_id =
      "jyppx.training.outbox-v1/" +
      derived_id("jyppx.training.outbox-v1.stream", source.event.run_id);
  options.limits = limits;
  return convert(source, options, {}, true);
}

TelemetryEnvelope adapt_training_compatibility_event(
    const EventEnvelope& source, std::string_view stream_id,
    const TelemetryTime& captured_time, const TelemetryLimits& limits) {
  TrainingTelemetryAdapterOptions options;
  options.stream_id = stream_id;
  options.limits = limits;
  auto result = convert(source, options, captured_time, true);
  result.attributes.erase("migration.source");
  result.attributes.erase("migration.time");
  result.attributes.erase("migration.trace_id");
  result.attributes["compatibility.source"] = "training.event-envelope/v1";
  result.attributes["compatibility.trace_id"] = source.event.trace_id;
  validate_telemetry_envelope(result, limits);
  return result;
}

TrainingOutboxMigrationCandidate migrate_training_outbox_v1(
    const TrainingOutboxV1MigrationInput& source,
    std::span<const TelemetrySinkDescriptor> current_sinks,
    std::span<const std::pair<std::string, std::string>> allowed_type_changes,
    const TelemetryLimits& limits, std::size_t max_pending_events,
    std::uint64_t max_pending_bytes) {
  validate_telemetry_limits(limits);
  if (max_pending_events == 0U || max_pending_bytes == 0U ||
      source.pending.size() > max_pending_events) {
    throw ResourceExhausted("Outbox N-1 migration exceeds new queue limits");
  }
  if (source.sinks.size() != current_sinks.size() || current_sinks.empty()) {
    reject("topology.sink_count");
  }
  std::set<std::string> sink_ids;
  for (std::size_t index = 0U; index < current_sinks.size(); ++index) {
    const auto& old = source.sinks[index];
    const auto& current = current_sinks[index];
    validate_telemetry_sink_descriptor(current, limits);
    const auto pair = std::make_pair(old.type_id, current.type_id);
    if (old.schema_version != 1U ||
        old.delivery_guarantee != EventDeliveryGuarantee::at_least_once ||
        old.sink_id != current.sink_id ||
        old.destination_id != current.destination_id ||
        !sink_ids.insert(current.sink_id).second ||
        (old.type_id != current.type_id &&
         std::find(allowed_type_changes.begin(), allowed_type_changes.end(),
                   pair) == allowed_type_changes.end())) {
      reject("topology.identity_or_type");
    }
    if (index != 0U && current_sinks[index - 1U].sink_id >= current.sink_id) {
      reject("topology.order");
    }
  }
  const auto& cursor = source.cursor;
  if (cursor.last_delivered_event_index > cursor.last_enqueued_event_index ||
      source.pending.size() != cursor.last_enqueued_event_index -
                                   cursor.last_delivered_event_index ||
      cursor.last_delivered_epoch > cursor.last_epoch ||
      cursor.last_delivered_global_step > cursor.last_global_step ||
      (cursor.last_delivered_event_index == 0U &&
       (cursor.last_delivered_epoch != 0U ||
        cursor.last_delivered_global_step != 0U)) ||
      (!cursor.initialized &&
       (cursor.terminal || !cursor.run_id.empty() ||
        cursor.last_enqueued_event_index != 0U || cursor.last_epoch != 0U ||
        cursor.last_global_step != 0U)) ||
      (cursor.initialized &&
       (cursor.run_id.empty() || cursor.last_enqueued_event_index == 0U))) {
    reject("cursor");
  }
  TrainingOutboxMigrationCandidate candidate;
  candidate.cursor = cursor;
  candidate.pending.reserve(source.pending.size());
  auto previous_epoch = cursor.last_delivered_epoch;
  auto previous_step = cursor.last_delivered_global_step;
  for (std::size_t index = 0U; index < source.pending.size(); ++index) {
    const auto& pending = source.pending[index];
    const auto& event = pending.envelope.event;
    if (pending.next_sink_index > current_sinks.size() ||
        event.run_id != cursor.run_id ||
        event.event_index != cursor.last_delivered_event_index + index + 1U ||
        event.epoch < previous_epoch || event.global_step < previous_step ||
        (event.event_index == 1U &&
         event.kind != CallbackEventKind::run_started) ||
        (terminal(event.kind) && index + 1U != source.pending.size())) {
      reject("pending.order_or_cursor");
    }
    auto migrated = migrate_training_envelope_v1(pending.envelope, limits);
    const auto encoded = encode_telemetry_envelope(migrated, limits);
    constexpr std::uint64_t node_charge = 16U;
    const auto encoded_bytes = static_cast<std::uint64_t>(encoded.size());
    if (encoded_bytes >
            std::numeric_limits<std::uint64_t>::max() - node_charge ||
        encoded_bytes + node_charge >
            max_pending_bytes - candidate.canonical_pending_bytes) {
      throw ResourceExhausted("Outbox N-1 migration exceeds new byte budget");
    }
    candidate.canonical_pending_bytes += encoded_bytes + node_charge;
    candidate.pending.push_back({std::move(migrated), pending.next_sink_index});
    previous_epoch = event.epoch;
    previous_step = event.global_step;
  }
  if (previous_epoch != cursor.last_epoch ||
      previous_step != cursor.last_global_step ||
      (!source.pending.empty() &&
       terminal(source.pending.back().envelope.event.kind) !=
           cursor.terminal)) {
    reject("cursor.suffix");
  }
  return candidate;
}

}  // namespace jyppx::training
