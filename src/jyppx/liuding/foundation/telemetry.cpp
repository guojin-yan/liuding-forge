// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/foundation/telemetry.hpp>

#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

namespace jyppx::liuding {
namespace {

constexpr std::string_view kContentDomain = "jyppx.telemetry.content/v1";
constexpr std::string_view kDefinitionDomain =
    "jyppx.telemetry.metric-definition/v1";
constexpr std::string_view kIdempotencyDomain =
    "jyppx.telemetry.idempotency/v1";

[[noreturn]] void invalid(std::string field, std::string expected) {
  throw InvalidArgument(
      "Invalid neutral telemetry contract",
      ErrorDetails{.component = "foundation.telemetry",
                   .operation = std::move(field),
                   .expected = std::move(expected),
                   .actual = "invalid field; input content omitted",
                   .hint = "Validate and redact producer observations before "
                           "durable acceptance."});
}

[[noreturn]] void incompatible(std::string field, std::string expected) {
  throw CompatibilityError(
      "Incompatible telemetry observation or receipt",
      ErrorDetails{.component = "foundation.telemetry",
                   .operation = std::move(field),
                   .expected = std::move(expected),
                   .actual = "unsupported schema or conflicting identity",
                   .hint = "Use the declared schema and replay immutable "
                           "content with its original operation key."});
}

[[noreturn]] void exhausted(std::string field, std::uint64_t limit,
                            std::uint64_t actual) {
  throw ResourceExhausted(
      "Telemetry resource bound exceeded",
      ErrorDetails{.component = "foundation.telemetry",
                   .operation = std::move(field),
                   .expected = std::to_string(limit),
                   .actual = std::to_string(actual),
                   .hint = "Reduce observation size or change the explicit "
                           "resource policy before retrying."});
}

[[noreturn]] void malformed(std::string field, std::string expected) {
  throw SerializationError(
      "Malformed canonical telemetry encoding",
      ErrorDetails{.component = "foundation.telemetry",
                   .operation = std::move(field),
                   .expected = std::move(expected),
                   .actual = "invalid canonical input; content omitted",
                   .hint = "Reject the complete input without changing any "
                           "live run, metric, or delivery state."});
}

void bounded(std::size_t size, std::size_t limit, std::string_view field) {
  if (size > limit) {
    exhausted(std::string(field), static_cast<std::uint64_t>(limit),
              static_cast<std::uint64_t>(size));
  }
}

bool valid_utf8(std::string_view text) noexcept {
  std::size_t offset = 0U;
  while (offset < text.size()) {
    const auto lead = static_cast<unsigned char>(text[offset]);
    if (lead == 0U) return false;
    if (lead <= 0x7FU) {
      ++offset;
      continue;
    }
    std::size_t length = 0U;
    std::uint32_t point = 0U;
    std::uint32_t minimum = 0U;
    if (lead >= 0xC2U && lead <= 0xDFU) {
      length = 2U;
      point = lead & 0x1FU;
      minimum = 0x80U;
    } else if (lead >= 0xE0U && lead <= 0xEFU) {
      length = 3U;
      point = lead & 0x0FU;
      minimum = 0x800U;
    } else if (lead >= 0xF0U && lead <= 0xF4U) {
      length = 4U;
      point = lead & 0x07U;
      minimum = 0x10000U;
    } else {
      return false;
    }
    if (length > text.size() - offset) return false;
    for (std::size_t index = 1U; index < length; ++index) {
      const auto next = static_cast<unsigned char>(text[offset + index]);
      if ((next & 0xC0U) != 0x80U) return false;
      point = (point << 6U) | (next & 0x3FU);
    }
    if (point < minimum || point > 0x10FFFFU ||
        (point >= 0xD800U && point <= 0xDFFFU)) {
      return false;
    }
    offset += length;
  }
  return true;
}

void text(std::string_view value, std::size_t limit, std::string_view field,
          bool allow_empty = true) {
  bounded(value.size(), limit, field);
  if ((!allow_empty && value.empty()) || !valid_utf8(value)) {
    invalid(std::string(field), "bounded valid UTF-8 without embedded NUL");
  }
}

void identifier(std::string_view value, const TelemetryLimits& limits,
                std::string_view field, bool allow_empty = false) {
  text(value, limits.max_identifier_bytes, field, allow_empty);
  for (const auto character : value) {
    const auto byte = static_cast<unsigned char>(character);
    if (byte <= 0x20U || byte == 0x7FU) {
      invalid(std::string(field),
              "an opaque identity without ASCII whitespace "
              "or control characters");
    }
  }
}

template <typename Enum>
bool in_range(Enum value, Enum first, Enum last) noexcept {
  using Underlying = std::underlying_type_t<Enum>;
  const auto number = static_cast<Underlying>(value);
  return number >= static_cast<Underlying>(first) &&
         number <= static_cast<Underlying>(last);
}

void attributes(const TelemetryAttributes& values, std::size_t count_limit,
                const TelemetryLimits& limits, std::string_view field) {
  bounded(values.size(), count_limit, field);
  for (const auto& [key, value] : values) {
    identifier(key, limits, field);
    text(value, limits.max_text_bytes, field);
  }
}

void identity_fields(const TelemetryIdentity& identity,
                     const TelemetryLimits& limits) {
  identifier(identity.stream_id, limits, "identity.stream_id");
  identifier(identity.event_id, limits, "identity.event_id");
  identifier(identity.idempotency_key, limits, "identity.idempotency_key");
  if (identity.sequence == 0U) {
    invalid("identity.sequence", "a positive producer sequence");
  }
}

void definition_fields(const TelemetryMetricDefinition& definition,
                       const TelemetryLimits& limits) {
  if (definition.schema_version != 1U) {
    incompatible("metric.definition.schema_version", "schema 1");
  }
  identifier(definition.metric_id, limits, "metric.definition.metric_id");
  if (definition.version == 0U) {
    invalid("metric.definition.version", "a positive immutable version");
  }
  if (!in_range(definition.unit, TelemetryMetricUnit::dimensionless,
                TelemetryMetricUnit::unspecified) ||
      !in_range(definition.aggregation, TelemetryAggregation::last,
                TelemetryAggregation::unspecified) ||
      !in_range(definition.direction, TelemetryMetricDirection::none,
                TelemetryMetricDirection::maximize)) {
    invalid("metric.definition.semantics", "known unit/aggregation/direction");
  }
  attributes(definition.parameters, limits.max_attributes, limits,
             "metric.definition.parameters");
  bounded(definition.context_keys.size(), limits.max_context_fields,
          "metric.definition.context_keys");
  for (const auto& key : definition.context_keys) {
    identifier(key, limits, "metric.definition.context_keys");
  }
}

void point_fields(const TelemetryMetricPoint& point,
                  const TelemetryLimits& limits) {
  definition_fields(point.definition, limits);
  if (!in_range(point.step.kind, TelemetryStepKind::micro_step,
                TelemetryStepKind::elapsed_nanoseconds)) {
    invalid("metric.step.kind", "a declared counter kind");
  }
  const auto& observation = point.observation;
  if (!in_range(observation.status, TelemetryValueStatus::finite,
                TelemetryValueStatus::missing)) {
    invalid("metric.observation.status", "a declared scalar status");
  }
  const bool finite = observation.status == TelemetryValueStatus::finite;
  if (finite != observation.value.has_value() ||
      (finite && !std::isfinite(*observation.value))) {
    invalid("metric.observation.value",
            "a present finite scalar exactly for "
            "finite status");
  }
  text(observation.reason, limits.max_text_bytes, "metric.observation.reason",
       observation.status != TelemetryValueStatus::missing);
  attributes(point.context, limits.max_context_fields, limits,
             "metric.context");
  if (point.context.size() != point.definition.context_keys.size()) {
    invalid("metric.context", "the exact declared context key set");
  }
  for (const auto& [key, value] : point.context) {
    static_cast<void>(value);
    if (!point.definition.context_keys.contains(key)) {
      invalid("metric.context", "the exact declared context key set");
    }
  }
  if (finite && point.sample_count == std::optional<std::uint64_t>{0U}) {
    invalid("metric.sample_count",
            "a positive count for a present finite value");
  }
}

void points_fields(const std::vector<TelemetryMetricPoint>& points,
                   const TelemetryLimits& limits, bool allow_empty) {
  bounded(points.size(), limits.max_metrics_per_envelope, "payload.metrics");
  if (!allow_empty && points.empty()) {
    invalid("payload.metrics", "at least one metric observation");
  }
  for (const auto& point : points) point_fields(point, limits);
}

void envelope_fields(const TelemetryEnvelope& envelope,
                     const TelemetryLimits& limits) {
  if (envelope.schema_version != 1U) {
    incompatible("envelope.schema_version", "schema 1");
  }
  identity_fields(envelope.identity, limits);
  identifier(envelope.time.clock_id, limits, "time.clock_id", true);
  if (envelope.time.monotonic_time_ns.has_value() !=
      !envelope.time.clock_id.empty()) {
    invalid("time.clock_id",
            "a nonempty clock ID exactly when monotonic "
            "time is present");
  }
  const auto& correlation = envelope.correlation;
  identifier(correlation.project_id, limits, "correlation.project_id", true);
  identifier(correlation.run_id, limits, "correlation.run_id", true);
  identifier(correlation.attempt_id, limits, "correlation.attempt_id", true);
  identifier(correlation.trace_id, limits, "correlation.trace_id", true);
  identifier(correlation.span_id, limits, "correlation.span_id", true);
  identifier(correlation.parent_span_id, limits, "correlation.parent_span_id",
             true);
  if ((!correlation.attempt_id.empty() && correlation.run_id.empty()) ||
      (!correlation.span_id.empty() && correlation.trace_id.empty()) ||
      (!correlation.parent_span_id.empty() &&
       (correlation.span_id.empty() ||
        correlation.parent_span_id == correlation.span_id))) {
    invalid("correlation", "run/attempt and trace/span/parent consistency");
  }
  if (!in_range(envelope.reliability, TelemetryReliability::required,
                TelemetryReliability::best_effort)) {
    invalid("envelope.reliability", "a declared reliability policy");
  }
  attributes(envelope.attributes, limits.max_attributes, limits, "attributes");
  if (envelope.payload.valueless_by_exception()) {
    invalid("payload", "one supported typed payload alternative");
  }
  std::visit(
      [&](const auto& payload) {
        using Payload = std::decay_t<decltype(payload)>;
        if constexpr (std::is_same_v<Payload, TelemetryLifecycle>) {
          identifier(payload.name, limits, "lifecycle.name");
          points_fields(payload.metrics, limits, true);
          bounded(payload.decisions.size(), limits.max_decisions_per_envelope,
                  "lifecycle.decisions");
          std::size_t effective = 0U;
          for (const auto& decision : payload.decisions) {
            identifier(decision.actor_id, limits, "lifecycle.decision.actor");
            identifier(decision.action, limits, "lifecycle.decision.action");
            text(decision.reason, limits.max_text_bytes,
                 "lifecycle.decision.reason", false);
            if (decision.effective) ++effective;
          }
          if (effective > 1U) {
            invalid("lifecycle.decisions", "at most one effective decision");
          }
          if (payload.terminal &&
              envelope.reliability != TelemetryReliability::required) {
            invalid("lifecycle.terminal", "required terminal evidence");
          }
        } else if constexpr (std::is_same_v<Payload, TelemetryMetricBatch>) {
          points_fields(payload.points, limits, false);
        } else if constexpr (std::is_same_v<Payload, TelemetryLog>) {
          if (!in_range(payload.severity, TelemetryLogSeverity::debug,
                        TelemetryLogSeverity::error)) {
            invalid("log.severity", "a declared log severity");
          }
          text(payload.body, limits.max_text_bytes, "log.body", false);
        } else if constexpr (std::is_same_v<Payload, TelemetrySpan>) {
          identifier(payload.name, limits, "span.name");
          if (!in_range(payload.status, TelemetrySpanStatus::unset,
                        TelemetrySpanStatus::cancelled)) {
            invalid("span.status", "a declared span outcome");
          }
          text(payload.reason, limits.max_text_bytes, "span.reason");
          if (correlation.trace_id.empty() || correlation.span_id.empty() ||
              !envelope.time.monotonic_time_ns.has_value() ||
              payload.end_monotonic_ns < payload.start_monotonic_ns ||
              payload.end_monotonic_ns > *envelope.time.monotonic_time_ns) {
            invalid("span.time",
                    "an identified completed span in one declared "
                    "monotonic clock domain");
          }
        } else if constexpr (std::is_same_v<Payload, TelemetryResourceSample>) {
          identifier(payload.resource_id, limits, "resource.resource_id");
          points_fields(payload.measurements, limits, false);
        } else if constexpr (std::is_same_v<Payload,
                                            TelemetryArtifactReference>) {
          identifier(payload.role, limits, "artifact.role");
          try {
            validate_artifact_digest(payload.artifact.digest);
          } catch (const InvalidArgument&) {
            invalid("artifact.digest", "a canonical SHA-256 content identity");
          }
          text(payload.artifact.logical_name, limits.max_text_bytes,
               "artifact.logical_name", false);
          identifier(payload.artifact.media_type, limits,
                     "artifact.media_type");
          identifier(payload.artifact.producer, limits, "artifact.producer");
          if (payload.artifact.size_bytes >
              limits.max_artifact_reference_bytes) {
            exhausted("artifact.size_bytes",
                      limits.max_artifact_reference_bytes,
                      payload.artifact.size_bytes);
          }
        }
      },
      envelope.payload);
}

// This internal persistence codec is also the domain-separated hash preimage.
// The counting path enforces byte bounds without another unbounded buffer. All
// multibyte values have fixed little-endian encodings; all variable-length
// fields carry an unsigned 64-bit size. Exporters remain separate adapters.
class CanonicalEncoder final {
 public:
  explicit CanonicalEncoder(std::uint64_t limit, Sha256* hash = nullptr,
                            std::vector<std::byte>* output = nullptr)
      : limit_(limit), hash_(hash), output_(output) {}

  void bytes(std::span<const std::byte> value) {
    const auto count = static_cast<std::uint64_t>(value.size());
    if (count > limit_ - used_) {
      exhausted("canonical.bytes", limit_,
                count > std::numeric_limits<std::uint64_t>::max() - used_
                    ? std::numeric_limits<std::uint64_t>::max()
                    : used_ + count);
    }
    used_ += count;
    if (hash_ != nullptr) hash_->update(value);
    if (output_ != nullptr) {
      output_->insert(output_->end(), value.begin(), value.end());
    }
  }

  void u8(std::uint8_t value) {
    const std::array<std::byte, 1> encoded{static_cast<std::byte>(value)};
    bytes(encoded);
  }

  void u32(std::uint32_t value) {
    std::array<std::byte, 4> encoded{};
    for (std::size_t index = 0U; index < encoded.size(); ++index) {
      encoded[index] = static_cast<std::byte>((value >> (index * 8U)) & 0xFFU);
    }
    bytes(encoded);
  }

  void u64(std::uint64_t value) {
    std::array<std::byte, 8> encoded{};
    for (std::size_t index = 0U; index < encoded.size(); ++index) {
      encoded[index] = static_cast<std::byte>((value >> (index * 8U)) & 0xFFU);
    }
    bytes(encoded);
  }

  void boolean(bool value) { u8(value ? 1U : 0U); }

  void string(std::string_view value) {
    u64(static_cast<std::uint64_t>(value.size()));
    bytes({reinterpret_cast<const std::byte*>(value.data()), value.size()});
  }

  void number(double value) {
    static_assert(sizeof(double) == sizeof(std::uint64_t));
    static_assert(std::numeric_limits<double>::is_iec559);
    // The observation validator rejects NaN/Inf; both signed zeros have one
    // semantic representation, independent of the producer's arithmetic path.
    u64(value == 0.0 ? 0U : std::bit_cast<std::uint64_t>(value));
  }

  template <typename Enum>
  void enumeration(Enum value) {
    u8(static_cast<std::uint8_t>(value));
  }

  std::uint64_t used() const noexcept { return used_; }

 private:
  std::uint64_t limit_;
  Sha256* hash_;
  std::vector<std::byte>* output_;
  std::uint64_t used_ = 0U;
};

void encode_attributes(CanonicalEncoder& encoder,
                       const TelemetryAttributes& attributes) {
  encoder.u64(static_cast<std::uint64_t>(attributes.size()));
  for (const auto& [key, value] : attributes) {
    encoder.string(key);
    encoder.string(value);
  }
}

void encode_definition(CanonicalEncoder& encoder,
                       const TelemetryMetricDefinition& definition) {
  encoder.u32(definition.schema_version);
  encoder.string(definition.metric_id);
  encoder.u32(definition.version);
  encoder.enumeration(definition.unit);
  encoder.enumeration(definition.aggregation);
  encoder.enumeration(definition.direction);
  encode_attributes(encoder, definition.parameters);
  encoder.u64(static_cast<std::uint64_t>(definition.context_keys.size()));
  for (const auto& key : definition.context_keys) encoder.string(key);
}

void encode_point(CanonicalEncoder& encoder,
                  const TelemetryMetricPoint& point) {
  encode_definition(encoder, point.definition);
  encoder.enumeration(point.step.kind);
  encoder.u64(point.step.value);
  encoder.enumeration(point.observation.status);
  encoder.boolean(point.observation.value.has_value());
  if (point.observation.value.has_value()) {
    encoder.number(*point.observation.value);
  }
  encoder.string(point.observation.reason);
  encode_attributes(encoder, point.context);
  encoder.boolean(point.sample_count.has_value());
  if (point.sample_count.has_value()) encoder.u64(*point.sample_count);
}

void encode_points(CanonicalEncoder& encoder,
                   const std::vector<TelemetryMetricPoint>& points) {
  encoder.u64(static_cast<std::uint64_t>(points.size()));
  for (const auto& point : points) encode_point(encoder, point);
}

void encode_envelope(CanonicalEncoder& encoder,
                     const TelemetryEnvelope& envelope) {
  encoder.string(kContentDomain);
  encoder.u32(envelope.schema_version);
  encoder.string(envelope.identity.stream_id);
  encoder.string(envelope.identity.event_id);
  encoder.string(envelope.identity.idempotency_key);
  encoder.u64(envelope.identity.sequence);
  encoder.boolean(envelope.time.wall_time_unix_ns.has_value());
  if (envelope.time.wall_time_unix_ns.has_value()) {
    encoder.u64(std::bit_cast<std::uint64_t>(*envelope.time.wall_time_unix_ns));
  }
  encoder.boolean(envelope.time.monotonic_time_ns.has_value());
  if (envelope.time.monotonic_time_ns.has_value()) {
    encoder.u64(*envelope.time.monotonic_time_ns);
  }
  encoder.string(envelope.time.clock_id);
  encoder.string(envelope.correlation.project_id);
  encoder.string(envelope.correlation.run_id);
  encoder.string(envelope.correlation.attempt_id);
  encoder.string(envelope.correlation.trace_id);
  encoder.string(envelope.correlation.span_id);
  encoder.string(envelope.correlation.parent_span_id);
  encoder.enumeration(envelope.reliability);
  encode_attributes(encoder, envelope.attributes);
  encoder.enumeration(*telemetry_payload_kind(envelope));
  std::visit(
      [&](const auto& payload) {
        using Payload = std::decay_t<decltype(payload)>;
        if constexpr (std::is_same_v<Payload, TelemetryLifecycle>) {
          encoder.string(payload.name);
          encoder.boolean(payload.terminal);
          encode_points(encoder, payload.metrics);
          encoder.u64(static_cast<std::uint64_t>(payload.decisions.size()));
          for (const auto& decision : payload.decisions) {
            encoder.string(decision.actor_id);
            encoder.string(decision.action);
            encoder.string(decision.reason);
            encoder.boolean(decision.effective);
          }
        } else if constexpr (std::is_same_v<Payload, TelemetryMetricBatch>) {
          encode_points(encoder, payload.points);
        } else if constexpr (std::is_same_v<Payload, TelemetryLog>) {
          encoder.enumeration(payload.severity);
          encoder.string(payload.body);
        } else if constexpr (std::is_same_v<Payload, TelemetrySpan>) {
          encoder.string(payload.name);
          encoder.u64(payload.start_monotonic_ns);
          encoder.u64(payload.end_monotonic_ns);
          encoder.enumeration(payload.status);
          encoder.string(payload.reason);
        } else if constexpr (std::is_same_v<Payload, TelemetryResourceSample>) {
          encoder.string(payload.resource_id);
          encode_points(encoder, payload.measurements);
        } else if constexpr (std::is_same_v<Payload,
                                            TelemetryArtifactReference>) {
          encoder.string(payload.role);
          encoder.enumeration(payload.artifact.digest.algorithm);
          encoder.string(payload.artifact.digest.hex);
          encoder.u64(payload.artifact.size_bytes);
          encoder.string(payload.artifact.logical_name);
          encoder.string(payload.artifact.media_type);
          encoder.string(payload.artifact.producer);
        }
      },
      envelope.payload);
}

class CanonicalDecoder final {
 public:
  explicit CanonicalDecoder(std::span<const std::byte> bytes) : bytes_(bytes) {}

  std::uint8_t u8() {
    require(1U);
    return std::to_integer<std::uint8_t>(bytes_[offset_++]);
  }

  std::uint32_t u32() {
    require(4U);
    std::uint32_t value = 0U;
    for (std::size_t index = 0U; index < 4U; ++index) {
      value |= static_cast<std::uint32_t>(u8()) << (index * 8U);
    }
    return value;
  }

  std::uint64_t u64() {
    require(8U);
    std::uint64_t value = 0U;
    for (std::size_t index = 0U; index < 8U; ++index) {
      value |= static_cast<std::uint64_t>(u8()) << (index * 8U);
    }
    return value;
  }

  bool boolean() {
    const auto value = u8();
    if (value > 1U) malformed("encoding.boolean", "exactly zero or one");
    return value == 1U;
  }

  std::size_t count(std::size_t limit, std::string_view field,
                    std::size_t minimum_item_bytes = 0U) {
    const auto value = u64();
    if (value > static_cast<std::uint64_t>(limit)) {
      exhausted(std::string(field), static_cast<std::uint64_t>(limit), value);
    }
    if (minimum_item_bytes != 0U &&
        value > (bytes_.size() - offset_) / minimum_item_bytes) {
      malformed(std::string(field), "enough input for the declared item count");
    }
    return static_cast<std::size_t>(value);
  }

  std::string string(std::size_t limit, std::string_view field) {
    const auto size = count(limit, field);
    require(size);
    std::string value;
    if (size != 0U) {
      value.assign(reinterpret_cast<const char*>(bytes_.data() + offset_),
                   size);
    }
    offset_ += size;
    return value;
  }

  double number() {
    const auto bits = u64();
    const auto value = std::bit_cast<double>(bits);
    if (!std::isfinite(value) || (value == 0.0 && bits != 0U)) {
      malformed("encoding.number",
                "finite binary64 with canonical positive zero");
    }
    return value;
  }

  template <typename Enum>
  Enum enumeration() {
    return static_cast<Enum>(u8());
  }

  void finish() const {
    if (offset_ != bytes_.size()) {
      malformed("encoding.trailing", "exactly one complete canonical object");
    }
  }

 private:
  void require(std::size_t count) const {
    if (count > bytes_.size() - offset_) {
      malformed("encoding.truncation", "complete length-delimited fields");
    }
  }

  std::span<const std::byte> bytes_;
  std::size_t offset_ = 0U;
};

TelemetryAttributes decode_attributes(CanonicalDecoder& decoder,
                                      std::size_t count_limit,
                                      const TelemetryLimits& limits) {
  const auto count =
      decoder.count(count_limit, "encoding.attributes.count", 16U);
  TelemetryAttributes result;
  std::string previous;
  for (std::size_t index = 0U; index < count; ++index) {
    auto key =
        decoder.string(limits.max_identifier_bytes, "encoding.attribute.key");
    auto value =
        decoder.string(limits.max_text_bytes, "encoding.attribute.value");
    if (index != 0U && !(previous < key)) {
      malformed("encoding.attributes.order",
                "strictly ordered unique UTF-8 keys");
    }
    previous = key;
    result.emplace(std::move(key), std::move(value));
  }
  return result;
}

TelemetryMetricDefinition decode_definition(CanonicalDecoder& decoder,
                                            const TelemetryLimits& limits) {
  TelemetryMetricDefinition result;
  result.schema_version = decoder.u32();
  if (result.schema_version != 1U) {
    incompatible("metric.definition.schema_version", "schema 1");
  }
  result.metric_id =
      decoder.string(limits.max_identifier_bytes, "encoding.metric.id");
  result.version = decoder.u32();
  result.unit = decoder.enumeration<TelemetryMetricUnit>();
  result.aggregation = decoder.enumeration<TelemetryAggregation>();
  result.direction = decoder.enumeration<TelemetryMetricDirection>();
  result.parameters = decode_attributes(decoder, limits.max_attributes, limits);
  const auto count = decoder.count(limits.max_context_fields,
                                   "encoding.metric.context_keys", 8U);
  std::string previous;
  for (std::size_t index = 0U; index < count; ++index) {
    auto key = decoder.string(limits.max_identifier_bytes,
                              "encoding.metric.context_key");
    if (index != 0U && !(previous < key)) {
      malformed("encoding.metric.context_keys", "strictly ordered unique keys");
    }
    previous = key;
    result.context_keys.emplace(std::move(key));
  }
  return result;
}

TelemetryMetricPoint decode_point(CanonicalDecoder& decoder,
                                  const TelemetryLimits& limits) {
  TelemetryMetricPoint result;
  result.definition = decode_definition(decoder, limits);
  result.step.kind = decoder.enumeration<TelemetryStepKind>();
  result.step.value = decoder.u64();
  result.observation.status = decoder.enumeration<TelemetryValueStatus>();
  if (decoder.boolean()) result.observation.value = decoder.number();
  result.observation.reason =
      decoder.string(limits.max_text_bytes, "encoding.metric.reason");
  result.context =
      decode_attributes(decoder, limits.max_context_fields, limits);
  if (decoder.boolean()) result.sample_count = decoder.u64();
  return result;
}

std::vector<TelemetryMetricPoint> decode_points(CanonicalDecoder& decoder,
                                                const TelemetryLimits& limits) {
  const auto count = decoder.count(limits.max_metrics_per_envelope,
                                   "encoding.metrics.count", 63U);
  std::vector<TelemetryMetricPoint> result;
  result.reserve(count);
  for (std::size_t index = 0U; index < count; ++index) {
    result.push_back(decode_point(decoder, limits));
  }
  return result;
}

TelemetryEnvelope decode_envelope(CanonicalDecoder& decoder,
                                  const TelemetryLimits& limits) {
  if (decoder.string(kContentDomain.size(), "encoding.domain") !=
      kContentDomain) {
    incompatible("encoding.domain", "the schema-v1 telemetry content domain");
  }
  TelemetryEnvelope result;
  result.schema_version = decoder.u32();
  if (result.schema_version != 1U) {
    incompatible("envelope.schema_version", "schema 1");
  }
  result.identity.stream_id =
      decoder.string(limits.max_identifier_bytes, "encoding.identity.stream");
  result.identity.event_id =
      decoder.string(limits.max_identifier_bytes, "encoding.identity.event");
  result.identity.idempotency_key =
      decoder.string(limits.max_identifier_bytes, "encoding.identity.key");
  result.identity.sequence = decoder.u64();
  if (decoder.boolean()) {
    result.time.wall_time_unix_ns = std::bit_cast<std::int64_t>(decoder.u64());
  }
  if (decoder.boolean()) result.time.monotonic_time_ns = decoder.u64();
  result.time.clock_id =
      decoder.string(limits.max_identifier_bytes, "encoding.time.clock_id");
  result.correlation.project_id = decoder.string(
      limits.max_identifier_bytes, "encoding.correlation.project");
  result.correlation.run_id =
      decoder.string(limits.max_identifier_bytes, "encoding.correlation.run");
  result.correlation.attempt_id = decoder.string(
      limits.max_identifier_bytes, "encoding.correlation.attempt");
  result.correlation.trace_id =
      decoder.string(limits.max_identifier_bytes, "encoding.correlation.trace");
  result.correlation.span_id =
      decoder.string(limits.max_identifier_bytes, "encoding.correlation.span");
  result.correlation.parent_span_id = decoder.string(
      limits.max_identifier_bytes, "encoding.correlation.parent_span");
  result.reliability = decoder.enumeration<TelemetryReliability>();
  result.attributes = decode_attributes(decoder, limits.max_attributes, limits);
  switch (decoder.enumeration<TelemetryPayloadKind>()) {
    case TelemetryPayloadKind::lifecycle: {
      TelemetryLifecycle payload;
      payload.name = decoder.string(limits.max_identifier_bytes,
                                    "encoding.lifecycle.name");
      payload.terminal = decoder.boolean();
      payload.metrics = decode_points(decoder, limits);
      const auto count = decoder.count(limits.max_decisions_per_envelope,
                                       "encoding.lifecycle.decisions", 25U);
      payload.decisions.reserve(count);
      for (std::size_t index = 0U; index < count; ++index) {
        TelemetryDecision decision;
        decision.actor_id = decoder.string(limits.max_identifier_bytes,
                                           "encoding.decision.actor");
        decision.action = decoder.string(limits.max_identifier_bytes,
                                         "encoding.decision.action");
        decision.reason =
            decoder.string(limits.max_text_bytes, "encoding.decision.reason");
        decision.effective = decoder.boolean();
        payload.decisions.push_back(std::move(decision));
      }
      result.payload = std::move(payload);
      break;
    }
    case TelemetryPayloadKind::metric_batch:
      result.payload = TelemetryMetricBatch{decode_points(decoder, limits)};
      break;
    case TelemetryPayloadKind::log: {
      TelemetryLog payload;
      payload.severity = decoder.enumeration<TelemetryLogSeverity>();
      payload.body = decoder.string(limits.max_text_bytes, "encoding.log.body");
      result.payload = std::move(payload);
      break;
    }
    case TelemetryPayloadKind::span: {
      TelemetrySpan payload;
      payload.name =
          decoder.string(limits.max_identifier_bytes, "encoding.span.name");
      payload.start_monotonic_ns = decoder.u64();
      payload.end_monotonic_ns = decoder.u64();
      payload.status = decoder.enumeration<TelemetrySpanStatus>();
      payload.reason =
          decoder.string(limits.max_text_bytes, "encoding.span.reason");
      result.payload = std::move(payload);
      break;
    }
    case TelemetryPayloadKind::resource_sample: {
      TelemetryResourceSample payload;
      payload.resource_id =
          decoder.string(limits.max_identifier_bytes, "encoding.resource.id");
      payload.measurements = decode_points(decoder, limits);
      result.payload = std::move(payload);
      break;
    }
    case TelemetryPayloadKind::artifact_reference: {
      TelemetryArtifactReference payload;
      payload.role =
          decoder.string(limits.max_identifier_bytes, "encoding.artifact.role");
      payload.artifact.digest.algorithm =
          decoder.enumeration<ArtifactDigestAlgorithm>();
      payload.artifact.digest.hex =
          decoder.string(64U, "encoding.artifact.digest");
      payload.artifact.size_bytes = decoder.u64();
      payload.artifact.logical_name =
          decoder.string(limits.max_text_bytes, "encoding.artifact.name");
      payload.artifact.media_type = decoder.string(limits.max_identifier_bytes,
                                                   "encoding.artifact.media");
      payload.artifact.producer = decoder.string(limits.max_identifier_bytes,
                                                 "encoding.artifact.producer");
      result.payload = std::move(payload);
      break;
    }
    default:
      incompatible("encoding.payload_tag", "a supported schema-v1 payload tag");
  }
  return result;
}

ArtifactDigest digest(Sha256& hash) {
  return {.algorithm = ArtifactDigestAlgorithm::sha256,
          .hex = sha256_digest_hex(hash.finish())};
}

template <typename Key>
void remember_content(std::map<Key, std::string>& known, Key key,
                      const std::string& content, std::string_view field) {
  const auto [iterator, inserted] = known.emplace(std::move(key), content);
  if (!inserted && iterator->second != content) {
    incompatible(std::string(field), "identical content for a reused identity");
  }
}

}  // namespace

std::optional<TelemetryPayloadKind> telemetry_payload_kind(
    const TelemetryEnvelope& envelope) noexcept {
  if (envelope.payload.valueless_by_exception()) return std::nullopt;
  // The alternatives' persisted tags are deliberately defined here instead of
  // exposing std::variant's implementation-dependent representation.
  switch (envelope.payload.index()) {
    case 0U:
      return TelemetryPayloadKind::lifecycle;
    case 1U:
      return TelemetryPayloadKind::metric_batch;
    case 2U:
      return TelemetryPayloadKind::log;
    case 3U:
      return TelemetryPayloadKind::span;
    case 4U:
      return TelemetryPayloadKind::resource_sample;
    case 5U:
      return TelemetryPayloadKind::artifact_reference;
    default:
      return std::nullopt;
  }
}

TelemetryMetricValue make_telemetry_metric_value(std::optional<double> value,
                                                 std::string reason) {
  if (!value.has_value()) {
    if (reason.empty())
      invalid("metric.observation.reason", "a missing reason");
    return {.status = TelemetryValueStatus::missing,
            .value = std::nullopt,
            .reason = std::move(reason)};
  }
  if (std::isnan(*value)) {
    return {.status = TelemetryValueStatus::not_a_number,
            .value = std::nullopt,
            .reason = std::move(reason)};
  }
  if (std::isinf(*value)) {
    return {.status = std::signbit(*value)
                          ? TelemetryValueStatus::negative_infinity
                          : TelemetryValueStatus::positive_infinity,
            .value = std::nullopt,
            .reason = std::move(reason)};
  }
  return {.status = TelemetryValueStatus::finite,
          .value = *value == 0.0 ? 0.0 : *value,
          .reason = std::move(reason)};
}

void validate_telemetry_limits(const TelemetryLimits& limits) {
  if (limits.max_identifier_bytes == 0U || limits.max_text_bytes == 0U ||
      limits.max_attributes == 0U || limits.max_context_fields == 0U ||
      limits.max_metrics_per_envelope == 0U ||
      limits.max_decisions_per_envelope == 0U ||
      limits.max_batch_events == 0U || limits.max_envelope_bytes == 0U ||
      limits.max_batch_bytes < limits.max_envelope_bytes ||
      limits.max_artifact_reference_bytes == 0U) {
    invalid("limits",
            "positive limits with batch bytes at least envelope bytes");
  }
}

void validate_telemetry_metric_definition(
    const TelemetryMetricDefinition& definition,
    const TelemetryLimits& limits) {
  validate_telemetry_limits(limits);
  definition_fields(definition, limits);
  CanonicalEncoder encoder(limits.max_envelope_bytes);
  encoder.string(kDefinitionDomain);
  encode_definition(encoder, definition);
}

void validate_telemetry_envelope(const TelemetryEnvelope& envelope,
                                 const TelemetryLimits& limits) {
  validate_telemetry_limits(limits);
  envelope_fields(envelope, limits);
  CanonicalEncoder encoder(limits.max_envelope_bytes);
  encode_envelope(encoder, envelope);
}

void validate_telemetry_batch(std::span<const TelemetryEnvelope> envelopes,
                              const TelemetryLimits& limits) {
  validate_telemetry_limits(limits);
  bounded(envelopes.size(), limits.max_batch_events, "batch.events");
  std::map<std::pair<std::string, std::string>, std::string> by_operation;
  std::map<std::pair<std::string, std::string>, std::string> by_event;
  std::map<std::pair<std::string, std::uint64_t>, std::string> by_sequence;
  std::uint64_t bytes = 0U;
  for (const auto& envelope : envelopes) {
    envelope_fields(envelope, limits);
    Sha256 hash;
    CanonicalEncoder encoder(limits.max_envelope_bytes, &hash);
    encode_envelope(encoder, envelope);
    if (encoder.used() > limits.max_batch_bytes - bytes) {
      exhausted(
          "batch.bytes", limits.max_batch_bytes,
          encoder.used() > std::numeric_limits<std::uint64_t>::max() - bytes
              ? std::numeric_limits<std::uint64_t>::max()
              : bytes + encoder.used());
    }
    bytes += encoder.used();
    const auto content = sha256_digest_hex(hash.finish());
    remember_content(by_operation,
                     std::make_pair(envelope.identity.stream_id,
                                    envelope.identity.idempotency_key),
                     content, "batch.idempotency_key");
    remember_content(
        by_event,
        std::make_pair(envelope.identity.stream_id, envelope.identity.event_id),
        content, "batch.event_id");
    remember_content(
        by_sequence,
        std::make_pair(envelope.identity.stream_id, envelope.identity.sequence),
        content, "batch.sequence");
  }
}

std::vector<std::byte> encode_telemetry_envelope(
    const TelemetryEnvelope& envelope, const TelemetryLimits& limits) {
  validate_telemetry_limits(limits);
  envelope_fields(envelope, limits);
  CanonicalEncoder count(limits.max_envelope_bytes);
  encode_envelope(count, envelope);
  std::vector<std::byte> result;
  if (count.used() > static_cast<std::uint64_t>(result.max_size())) {
    exhausted("encoding.allocation",
              static_cast<std::uint64_t>(result.max_size()), count.used());
  }
  result.reserve(static_cast<std::size_t>(count.used()));
  CanonicalEncoder output(limits.max_envelope_bytes, nullptr, &result);
  encode_envelope(output, envelope);
  return result;
}

TelemetryEnvelope decode_telemetry_envelope(std::span<const std::byte> bytes,
                                            const TelemetryLimits& limits) {
  validate_telemetry_limits(limits);
  if (bytes.size() > limits.max_envelope_bytes) {
    exhausted("decoding.bytes", limits.max_envelope_bytes,
              static_cast<std::uint64_t>(bytes.size()));
  }
  CanonicalDecoder decoder(bytes);
  auto result = decode_envelope(decoder, limits);
  decoder.finish();
  try {
    validate_telemetry_envelope(result, limits);
  } catch (const InvalidArgument&) {
    malformed("decoding.fields", "valid schema-v1 envelope field semantics");
  }
  return result;
}

std::vector<std::byte> encode_telemetry_metric_definition(
    const TelemetryMetricDefinition& definition,
    const TelemetryLimits& limits) {
  validate_telemetry_limits(limits);
  definition_fields(definition, limits);
  CanonicalEncoder count(limits.max_envelope_bytes);
  count.string(kDefinitionDomain);
  encode_definition(count, definition);
  std::vector<std::byte> result;
  if (count.used() > static_cast<std::uint64_t>(result.max_size())) {
    exhausted("encoding.allocation",
              static_cast<std::uint64_t>(result.max_size()), count.used());
  }
  result.reserve(static_cast<std::size_t>(count.used()));
  CanonicalEncoder output(limits.max_envelope_bytes, nullptr, &result);
  output.string(kDefinitionDomain);
  encode_definition(output, definition);
  return result;
}

TelemetryMetricDefinition decode_telemetry_metric_definition(
    std::span<const std::byte> bytes, const TelemetryLimits& limits) {
  validate_telemetry_limits(limits);
  if (bytes.size() > limits.max_envelope_bytes) {
    exhausted("decoding.bytes", limits.max_envelope_bytes,
              static_cast<std::uint64_t>(bytes.size()));
  }
  CanonicalDecoder decoder(bytes);
  if (decoder.string(kDefinitionDomain.size(), "encoding.domain") !=
      kDefinitionDomain) {
    incompatible("encoding.domain", "the schema-v1 metric-definition domain");
  }
  auto result = decode_definition(decoder, limits);
  decoder.finish();
  try {
    validate_telemetry_metric_definition(result, limits);
  } catch (const InvalidArgument&) {
    malformed("decoding.fields", "valid schema-v1 metric definition semantics");
  }
  return result;
}

ArtifactDigest telemetry_metric_definition_digest(
    const TelemetryMetricDefinition& definition,
    const TelemetryLimits& limits) {
  validate_telemetry_limits(limits);
  definition_fields(definition, limits);
  Sha256 hash;
  CanonicalEncoder encoder(limits.max_envelope_bytes, &hash);
  encoder.string(kDefinitionDomain);
  encode_definition(encoder, definition);
  return digest(hash);
}

ArtifactDigest telemetry_content_digest(const TelemetryEnvelope& envelope,
                                        const TelemetryLimits& limits) {
  validate_telemetry_limits(limits);
  envelope_fields(envelope, limits);
  Sha256 hash;
  CanonicalEncoder encoder(limits.max_envelope_bytes, &hash);
  encode_envelope(encoder, envelope);
  return digest(hash);
}

ArtifactDigest telemetry_idempotency_digest(const TelemetryIdentity& identity,
                                            const TelemetryLimits& limits) {
  validate_telemetry_limits(limits);
  identity_fields(identity, limits);
  Sha256 hash;
  CanonicalEncoder encoder(limits.max_envelope_bytes, &hash);
  encoder.string(kIdempotencyDomain);
  encoder.string(identity.stream_id);
  encoder.string(identity.idempotency_key);
  return digest(hash);
}

void check_telemetry_operation(std::stop_token stop,
                               TelemetryDeadline deadline) {
  if (stop.stop_requested()) {
    throw Cancelled(
        "Telemetry operation cancelled",
        ErrorDetails{
            .component = "foundation.telemetry",
            .operation = "operation.cancel",
            .expected = "an active producer/consumer token",
            .actual = "stop requested",
            .hint = "Retain immutable pending observations for retry."});
  }
  if (deadline.has_value() && std::chrono::steady_clock::now() >= *deadline) {
    throw DeadlineExceeded(
        "Telemetry operation deadline reached",
        ErrorDetails{.component = "foundation.telemetry",
                     .operation = "operation.deadline",
                     .expected = "a future absolute steady-clock deadline",
                     .actual = "deadline reached",
                     .hint = "Keep the original retry identities and inspect "
                             "the sink's durable acknowledgement boundary."});
  }
}

void validate_telemetry_sink_descriptor(
    const TelemetrySinkDescriptor& descriptor, const TelemetryLimits& limits) {
  validate_telemetry_limits(limits);
  if (descriptor.schema_version != 1U) {
    incompatible("sink.schema_version", "schema 1");
  }
  identifier(descriptor.type_id, limits, "sink.type_id");
  identifier(descriptor.sink_id, limits, "sink.sink_id");
  identifier(descriptor.destination_id, limits, "sink.destination_id");
  if (!in_range(descriptor.role, TelemetrySinkRole::evidence,
                TelemetrySinkRole::observer)) {
    invalid("sink.role", "a declared evidence/observer role");
  }
}

void validate_telemetry_emit_result(
    const TelemetrySinkDescriptor& descriptor,
    std::span<const TelemetryEnvelope> envelopes,
    const TelemetryEmitResult& result, const TelemetryLimits& limits) {
  validate_telemetry_sink_descriptor(descriptor, limits);
  validate_telemetry_batch(envelopes, limits);
  if (result.dispositions.size() != envelopes.size()) {
    incompatible("sink.receipts", "one complete receipt per input envelope");
  }
  for (std::size_t index = 0U; index < result.dispositions.size(); ++index) {
    const auto disposition = result.dispositions[index];
    if (!in_range(disposition, TelemetryDisposition::accepted,
                  TelemetryDisposition::unsupported)) {
      incompatible("sink.receipt", "a declared acceptance disposition");
    }
    const bool required =
        envelopes[index].reliability == TelemetryReliability::required;
    if ((required && disposition == TelemetryDisposition::sampled_out) ||
        (required && descriptor.role == TelemetrySinkRole::evidence &&
         disposition == TelemetryDisposition::unsupported)) {
      incompatible("sink.receipt",
                   "preserved required evidence or an explicit "
                   "unsupported optional observer receipt");
    }
  }
}

}  // namespace jyppx::liuding
