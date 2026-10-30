// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "native_telemetry_protocol.hpp"

#include <jyppx/liuding/experiment/telemetry_exporters.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace jyppx::liuding::cli {
namespace {
using Json = nlohmann::json;

[[noreturn]] void invalid() {
  // No untrusted event body, endpoint or configuration is exposed.
  throw SerializationError("Invalid native telemetry worker protocol");
}

void object(const Json& value, std::initializer_list<std::string_view> names) {
  if (!value.is_object() || value.size() != names.size()) invalid();
  for (const auto name : names)
    if (!value.contains(name)) invalid();
}

const std::string& text(const Json& value, std::size_t maximum) {
  if (!value.is_string()) invalid();
  const auto& result = value.get_ref<const std::string&>();
  if (result.size() > maximum) throw ResourceExhausted("Worker text bound");
  return result;
}

std::uint64_t number(const Json& value, std::uint64_t maximum = UINT64_MAX) {
  if (!value.is_number_unsigned()) invalid();
  const auto result = value.get<std::uint64_t>();
  if (result > maximum) invalid();
  return result;
}

std::int64_t signed_number(const Json& value) {
  if (value.is_number_unsigned())
    return static_cast<std::int64_t>(number(value, INT64_MAX));
  if (!value.is_number_integer()) invalid();
  return value.get<std::int64_t>();
}

template <typename Enum>
Enum tag(const Json& value, std::uint64_t maximum) {
  const auto result = number(value, maximum);
  if (!result) invalid();
  return static_cast<Enum>(result);
}

bool boolean(const Json& value) {
  if (!value.is_boolean()) invalid();
  return value.get<bool>();
}

void array(const Json& value, std::size_t maximum) {
  if (!value.is_array()) invalid();
  if (value.size() > maximum)
    throw ResourceExhausted("Worker collection bound");
}

TelemetryAttributes attributes(const Json& value, std::size_t maximum,
                               const TelemetryLimits& limits) {
  if (!value.is_object()) invalid();
  if (value.size() > maximum) throw ResourceExhausted("Worker attribute bound");
  TelemetryAttributes result;
  for (const auto& [name, field] : value.items()) {
    if (name.size() > limits.max_identifier_bytes)
      throw ResourceExhausted("Worker attribute key bound");
    result.emplace(name, text(field, limits.max_text_bytes));
  }
  return result;
}

TelemetryMetricDefinition definition(const Json& value,
                                     const TelemetryLimits& limits) {
  object(value,
         {"schema_version", "metric_id", "version", "unit", "aggregation",
          "direction", "parameters", "context_keys", "definition_digest"});
  TelemetryMetricDefinition result;
  result.schema_version = static_cast<std::uint32_t>(
      number(value.at("schema_version"), UINT32_MAX));
  result.metric_id = text(value.at("metric_id"), limits.max_identifier_bytes);
  result.version =
      static_cast<std::uint32_t>(number(value.at("version"), UINT32_MAX));
  result.unit = tag<TelemetryMetricUnit>(value.at("unit"), 9U);
  result.aggregation = tag<TelemetryAggregation>(value.at("aggregation"), 6U);
  result.direction = tag<TelemetryMetricDirection>(value.at("direction"), 3U);
  result.parameters =
      attributes(value.at("parameters"), limits.max_attributes, limits);
  const auto& keys = value.at("context_keys");
  array(keys, limits.max_context_fields);
  for (const auto& key : keys)
    if (!result.context_keys.insert(text(key, limits.max_identifier_bytes))
             .second)
      invalid();
  if (telemetry_metric_definition_digest(result, limits).hex !=
      text(value.at("definition_digest"), 64U))
    invalid();
  return result;
}

std::vector<TelemetryMetricPoint> points(const Json& value,
                                         const TelemetryLimits& limits) {
  array(value, limits.max_metrics_per_envelope);
  std::vector<TelemetryMetricPoint> result;
  result.reserve(value.size());
  for (const auto& field : value) {
    object(field, {"definition", "step", "status", "value", "reason", "context",
                   "sample_count"});
    TelemetryMetricPoint point;
    point.definition = definition(field.at("definition"), limits);
    const auto& step = field.at("step");
    object(step, {"kind", "value"});
    point.step.kind = tag<TelemetryStepKind>(step.at("kind"), 5U);
    point.step.value = number(step.at("value"));
    point.observation.status =
        tag<TelemetryValueStatus>(field.at("status"), 5U);
    const auto& scalar = field.at("value");
    if (!scalar.is_null()) {
      // The producer emits binary64 JSON numbers, including -0.0. Accepting
      // integer coercion here could round full-width integers silently.
      if (!scalar.is_number_float()) invalid();
      const auto scalar_value = scalar.get<double>();
      if (!std::isfinite(scalar_value)) invalid();
      point.observation.value = scalar_value;
    }
    point.observation.reason = text(field.at("reason"), limits.max_text_bytes);
    point.context =
        attributes(field.at("context"), limits.max_context_fields, limits);
    if (!field.at("sample_count").is_null())
      point.sample_count = number(field.at("sample_count"));
    result.push_back(std::move(point));
  }
  return result;
}

TelemetryPayload payload(const Json& value, const TelemetryLimits& limits) {
  if (!value.is_object() || !value.contains("kind")) invalid();
  switch (number(value.at("kind"), 6U)) {
    case 1U: {
      object(value, {"kind", "name", "terminal", "metrics", "decisions"});
      TelemetryLifecycle result;
      result.name = text(value.at("name"), limits.max_identifier_bytes);
      result.terminal = boolean(value.at("terminal"));
      result.metrics = points(value.at("metrics"), limits);
      const auto& decisions = value.at("decisions");
      array(decisions, limits.max_decisions_per_envelope);
      result.decisions.reserve(decisions.size());
      for (const auto& field : decisions) {
        object(field, {"actor_id", "action", "reason", "effective"});
        result.decisions.push_back(
            {text(field.at("actor_id"), limits.max_identifier_bytes),
             text(field.at("action"), limits.max_identifier_bytes),
             text(field.at("reason"), limits.max_text_bytes),
             boolean(field.at("effective"))});
      }
      return result;
    }
    case 2U:
      object(value, {"kind", "points"});
      return TelemetryMetricBatch{points(value.at("points"), limits)};
    case 3U:
      object(value, {"kind", "severity", "body"});
      return TelemetryLog{tag<TelemetryLogSeverity>(value.at("severity"), 4U),
                          text(value.at("body"), limits.max_text_bytes)};
    case 4U:
      object(value, {"kind", "name", "start_monotonic_ns", "end_monotonic_ns",
                     "status", "reason"});
      return TelemetrySpan{text(value.at("name"), limits.max_identifier_bytes),
                           number(value.at("start_monotonic_ns")),
                           number(value.at("end_monotonic_ns")),
                           tag<TelemetrySpanStatus>(value.at("status"), 4U),
                           text(value.at("reason"), limits.max_text_bytes)};
    case 5U:
      object(value, {"kind", "resource_id", "measurements"});
      return TelemetryResourceSample{
          text(value.at("resource_id"), limits.max_identifier_bytes),
          points(value.at("measurements"), limits)};
    case 6U: {
      object(value, {"kind", "role", "artifact"});
      const auto& artifact = value.at("artifact");
      object(artifact, {"algorithm", "digest", "size_bytes", "logical_name",
                        "media_type", "producer"});
      if (text(artifact.at("algorithm"), 6U) != "sha256") invalid();
      return TelemetryArtifactReference{
          text(value.at("role"), limits.max_identifier_bytes),
          {{ArtifactDigestAlgorithm::sha256, text(artifact.at("digest"), 64U)},
           number(artifact.at("size_bytes"),
                  limits.max_artifact_reference_bytes),
           text(artifact.at("logical_name"), limits.max_text_bytes),
           text(artifact.at("media_type"), limits.max_text_bytes),
           text(artifact.at("producer"), limits.max_text_bytes)}};
    }
    default:
      invalid();
  }
}

Json counters(const TelemetryFlushResult& value) {
  return {{"exported_events", value.exported_events},
          {"sampled_out_events", value.sampled_out_events},
          {"unsupported_events", value.unsupported_events}};
}
}  // namespace

nlohmann::json parse_native_telemetry_json(std::string_view value,
                                           std::size_t maximum_bytes) {
  if (value.size() > maximum_bytes)
    throw ResourceExhausted("Worker JSON byte bound");
  std::vector<std::set<std::string>> keys;
  try {
    return Json::parse(
        value.begin(), value.end(),
        [&](int depth, Json::parse_event_t event, Json& field) {
          if (depth > 32) invalid();
          if (event == Json::parse_event_t::object_start) keys.emplace_back();
          if (event == Json::parse_event_t::key &&
              (keys.empty() ||
               !keys.back().insert(field.get<std::string>()).second))
            invalid();
          if (event == Json::parse_event_t::object_end) keys.pop_back();
          return true;
        });
  } catch (const Json::exception&) {
    invalid();
  }
}

TelemetryEnvelope decode_native_telemetry_projection(
    const nlohmann::json& value, std::string_view content_digest,
    const TelemetryLimits& limits) {
  try {
    validate_telemetry_limits(limits);
    validate_artifact_digest(
        {ArtifactDigestAlgorithm::sha256, std::string(content_digest)});
    object(value, {"schema_version", "identity", "time", "correlation",
                   "reliability", "attributes", "payload"});
    TelemetryEnvelope result;
    result.schema_version = static_cast<std::uint32_t>(
        number(value.at("schema_version"), UINT32_MAX));
    const auto& id = value.at("identity");
    object(id, {"stream_id", "event_id", "idempotency_key", "sequence"});
    result.identity = {
        text(id.at("stream_id"), limits.max_identifier_bytes),
        text(id.at("event_id"), limits.max_identifier_bytes),
        text(id.at("idempotency_key"), limits.max_identifier_bytes),
        number(id.at("sequence"))};
    const auto& time = value.at("time");
    object(time, {"wall_time_unix_ns", "monotonic_time_ns", "clock_id"});
    if (!time.at("wall_time_unix_ns").is_null())
      result.time.wall_time_unix_ns =
          signed_number(time.at("wall_time_unix_ns"));
    if (!time.at("monotonic_time_ns").is_null())
      result.time.monotonic_time_ns = number(time.at("monotonic_time_ns"));
    result.time.clock_id =
        text(time.at("clock_id"), limits.max_identifier_bytes);
    const auto& correlation = value.at("correlation");
    object(correlation, {"project_id", "run_id", "attempt_id", "trace_id",
                         "span_id", "parent_span_id"});
    result.correlation = {
        text(correlation.at("project_id"), limits.max_identifier_bytes),
        text(correlation.at("run_id"), limits.max_identifier_bytes),
        text(correlation.at("attempt_id"), limits.max_identifier_bytes),
        text(correlation.at("trace_id"), limits.max_identifier_bytes),
        text(correlation.at("span_id"), limits.max_identifier_bytes),
        text(correlation.at("parent_span_id"), limits.max_identifier_bytes)};
    result.reliability = tag<TelemetryReliability>(value.at("reliability"), 2U);
    result.attributes =
        attributes(value.at("attributes"), limits.max_attributes, limits);
    result.payload = payload(value.at("payload"), limits);
    validate_telemetry_envelope(result, limits);
    if (telemetry_content_digest(result, limits).hex != content_digest)
      invalid();
    // Also compare the complete projection, whose fingerprint participates in
    // the native sink's retry identity. In particular, never fold signed zero.
    if (experiment::telemetry_projection_json(result, limits) != value.dump())
      invalid();
    return result;
  } catch (const Json::exception&) {
    invalid();
  }
}

NativeTelemetryProtocol::NativeTelemetryProtocol(TelemetrySink& sink,
                                                 TelemetryLimits limits)
    : sink_(sink), limits_(limits) {
  validate_telemetry_limits(limits_);
}

nlohmann::json NativeTelemetryProtocol::exchange(const nlohmann::json& request,
                                                 TelemetryDeadline deadline) {
  if (!request.is_object() || !request.contains("request_id")) invalid();
  const auto id = text(request.at("request_id"), 64U);
  if (id.empty()) invalid();
  Json response{{"protocol_version", 1U}, {"request_id", id}};
  try {
    if (!request.contains("protocol_version") ||
        number(request.at("protocol_version"), 1U) != 1U ||
        !request.contains("backend") ||
        text(request.at("backend"), 32U) != "opentelemetry" ||
        !request.contains("operation"))
      invalid();
    const auto operation = text(request.at("operation"), 8U);
    if (operation == "emit") {
      object(request, {"protocol_version", "request_id", "backend", "operation",
                       "events"});
      if (closed_) throw NotReady("Native telemetry worker is closed");
      const auto& events = request.at("events");
      array(events, limits_.max_batch_events);
      if (events.empty()) invalid();
      std::vector<TelemetryEnvelope> batch;
      batch.reserve(events.size());
      std::string identity = "jyppx.telemetry.sidecar-batch/v1";
      for (const auto& event : events) {
        object(event, {"content_digest", "envelope"});
        const auto& digest = text(event.at("content_digest"), 64U);
        batch.push_back(decode_native_telemetry_projection(event.at("envelope"),
                                                           digest, limits_));
        identity += digest;
      }
      if (sha256_hex({reinterpret_cast<const std::byte*>(identity.data()),
                      identity.size()}) != id)
        invalid();
      validate_telemetry_batch(batch, limits_);
      // No request-ID result cache: the sink owns per-envelope seen state.
      const auto result = sink_.emit(batch, {}, deadline);
      validate_telemetry_emit_result(sink_.descriptor(), batch, result,
                                     limits_);
      response["receipts"] = Json::array();
      for (const auto receipt : result.dispositions)
        response["receipts"].push_back(static_cast<unsigned>(receipt));
      counters_ = sink_.flush({}, deadline);
    } else if (operation == "flush" || operation == "close") {
      object(request,
             {"protocol_version", "request_id", "backend", "operation"});
      if (id != operation) invalid();
      if (!closed_) {
        if (operation == "close") {
          counters_ = sink_.close({}, deadline);
          closed_ = true;
        } else
          counters_ = sink_.flush({}, deadline);
      }
    } else
      invalid();
    response["counters"] = counters(counters_);
  } catch (const Error&) {
    // Emit can have completed a prefix before failing. Preserve the sink in
    // this process and return only an error, never partial/fabricated receipts
    // or stale counters. A later identical request consults the sink's seen.
    response.erase("receipts");
    response.erase("counters");
    response["error"] = "native_otlp.operation_rejected";
  } catch (const Json::exception&) {
    response.erase("receipts");
    response.erase("counters");
    response["error"] = "native_otlp.invalid_request";
  }
  return response;
}

}  // namespace jyppx::liuding::cli
