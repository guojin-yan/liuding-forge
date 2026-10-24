// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

// Draft tests: deliberately not registered or executed while the production
// M4/M5 storage chapter is under its unified validation freeze.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/foundation/telemetry.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
namespace jy = jyppx::liuding;

jy::TelemetryMetricDefinition loss_definition() {
  return {.schema_version = 1U,
          .metric_id = "training.loss",
          .version = 1U,
          .unit = jy::TelemetryMetricUnit::dimensionless,
          .aggregation = jy::TelemetryAggregation::mean,
          .direction = jy::TelemetryMetricDirection::minimize,
          .parameters = {{"reduction", "mean"}},
          .context_keys = {"split"}};
}

jy::TelemetryMetricPoint loss_point() {
  return {.definition = loss_definition(),
          .step = {.kind = jy::TelemetryStepKind::global_step, .value = 7U},
          .observation = jy::make_telemetry_metric_value(0.25),
          .context = {{"split", "validation"}},
          .sample_count = 32U};
}

jy::TelemetryEnvelope observation() {
  return {.schema_version = 1U,
          .identity = {.stream_id = "run.test/attempt.1",
                       .event_id = "event.1",
                       .idempotency_key = "callback.1",
                       .sequence = 1U},
          .time = {.wall_time_unix_ns = 1000,
                   .monotonic_time_ns = 120U,
                   .clock_id = "clock.test"},
          .correlation = {.project_id = "project.test",
                          .run_id = "run.test",
                          .attempt_id = "attempt.1",
                          .trace_id = "trace.test"},
          .reliability = jy::TelemetryReliability::required,
          .attributes = {{"producer", "training"}, {"phase", "validation"}},
          .payload = jy::TelemetryLifecycle{
              .name = "training.epoch_finished",
              .terminal = false,
              .metrics = {loss_point()},
              .decisions = {{.actor_id = "early_stopping",
                             .action = "stop",
                             .reason = "patience exhausted",
                             .effective = true}}}};
}

// The test fixture intentionally leaves optional metadata empty so this offset
// follows the written v1 schema directly. It targets a critical wire tag rather
// than depending on a private production decoder or searching ambiguous bytes.
std::size_t simple_payload_tag_offset(const jy::TelemetryEnvelope& envelope) {
  constexpr std::string_view domain = "jyppx.telemetry.content/v1";
  return 8U + domain.size() + 4U + 8U + envelope.identity.stream_id.size() +
         8U + envelope.identity.event_id.size() + 8U +
         envelope.identity.idempotency_key.size() + 8U + 1U + 1U + 8U +
         6U * 8U + 1U + 8U;
}

TEST(TelemetryContractTest, RoundTripsEveryTypedPayloadWithoutBackendTypes) {
  std::vector<jy::TelemetryEnvelope> values;
  auto value = observation();
  values.push_back(value);
  value.payload = jy::TelemetryMetricBatch{{loss_point()}};
  values.push_back(value);
  value.payload = jy::TelemetryLog{jy::TelemetryLogSeverity::warning,
                                   "bounded redacted diagnostic\nsecond line"};
  values.push_back(value);
  value.correlation.span_id = "span.test";
  value.correlation.parent_span_id = "span.parent";
  value.payload = jy::TelemetrySpan{.name = "validation.forward",
                                    .start_monotonic_ns = 100U,
                                    .end_monotonic_ns = 110U,
                                    .status = jy::TelemetrySpanStatus::ok};
  values.push_back(value);
  value.payload = jy::TelemetryResourceSample{"cpu.0", {loss_point()}};
  values.push_back(value);
  value.payload = jy::TelemetryArtifactReference{
      .role = "checkpoint.last",
      .artifact = {.digest = {.hex = std::string(64U, 'a')},
                   .size_bytes = 123U,
                   .logical_name = "last-checkpoint",
                   .media_type = "application/octet-stream",
                   .producer = "training.checkpoint"}};
  values.push_back(value);

  for (std::size_t index = 0U; index < values.size(); ++index) {
    SCOPED_TRACE(index);
    EXPECT_NO_THROW(jy::validate_telemetry_envelope(values[index]));
    EXPECT_EQ(jy::telemetry_payload_kind(values[index]),
              static_cast<jy::TelemetryPayloadKind>(index + 1U));
    const auto bytes = jy::encode_telemetry_envelope(values[index]);
    const auto restored = jy::decode_telemetry_envelope(bytes);
    EXPECT_EQ(jy::encode_telemetry_envelope(restored), bytes);
    EXPECT_EQ(jy::telemetry_content_digest(restored).hex,
              jy::sha256_hex(bytes));
  }
}

TEST(TelemetryContractTest, RetainsFiniteNonfiniteAndMissingSemantics) {
  const auto finite = jy::make_telemetry_metric_value(1.25);
  EXPECT_EQ(finite.status, jy::TelemetryValueStatus::finite);
  EXPECT_EQ(finite.value, 1.25);

  const std::array inputs{std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::infinity(),
                          -std::numeric_limits<double>::infinity()};
  const std::array statuses{jy::TelemetryValueStatus::not_a_number,
                            jy::TelemetryValueStatus::positive_infinity,
                            jy::TelemetryValueStatus::negative_infinity};
  for (std::size_t index = 0U; index < inputs.size(); ++index) {
    auto value = observation();
    auto& point =
        std::get<jy::TelemetryLifecycle>(value.payload).metrics.front();
    point.observation = jy::make_telemetry_metric_value(inputs[index]);
    EXPECT_EQ(point.observation.status, statuses[index]);
    EXPECT_FALSE(point.observation.value.has_value());
    const auto restored =
        jy::decode_telemetry_envelope(jy::encode_telemetry_envelope(value));
    const auto& actual =
        std::get<jy::TelemetryLifecycle>(restored.payload).metrics.front();
    EXPECT_EQ(actual.observation.status, statuses[index]);
    EXPECT_FALSE(actual.observation.value.has_value());
  }
  const auto missing =
      jy::make_telemetry_metric_value(std::nullopt, "empty validation split");
  EXPECT_EQ(missing.status, jy::TelemetryValueStatus::missing);
  EXPECT_FALSE(missing.value.has_value());
  EXPECT_THROW(jy::make_telemetry_metric_value(std::nullopt),
               jy::InvalidArgument);
}

TEST(TelemetryContractTest, RejectsRawNanAndContradictorySymbolicValues) {
  auto value = observation();
  auto& point = std::get<jy::TelemetryLifecycle>(value.payload).metrics.front();
  point.observation.value = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(jy::validate_telemetry_envelope(value), jy::InvalidArgument);
  point.observation.status = jy::TelemetryValueStatus::positive_infinity;
  point.observation.value = 3.0;
  EXPECT_THROW(jy::validate_telemetry_envelope(value), jy::InvalidArgument);
  point.observation.value.reset();
  point.observation.status = jy::TelemetryValueStatus::missing;
  point.observation.reason.clear();
  EXPECT_THROW(jy::validate_telemetry_envelope(value), jy::InvalidArgument);
  point.observation.reason = "no samples";
  point.sample_count = 0U;
  EXPECT_NO_THROW(jy::validate_telemetry_envelope(value));
}

TEST(TelemetryContractTest,
     DefinitionsIdentifySemanticUnitVersionAndParameters) {
  const auto original = loss_definition();
  const auto original_digest = jy::telemetry_metric_definition_digest(original);
  const auto bytes = jy::encode_telemetry_metric_definition(original);
  EXPECT_EQ(original_digest.hex, jy::sha256_hex(bytes));
  EXPECT_EQ(jy::encode_telemetry_metric_definition(
                jy::decode_telemetry_metric_definition(bytes)),
            bytes);
  for (const auto mutation : {0U, 1U, 2U, 3U, 4U}) {
    auto changed = original;
    if (mutation == 0U) changed.unit = jy::TelemetryMetricUnit::ratio;
    if (mutation == 1U) ++changed.version;
    if (mutation == 2U) changed.parameters["reduction"] = "sum";
    if (mutation == 3U) changed.aggregation = jy::TelemetryAggregation::sum;
    if (mutation == 4U)
      changed.direction = jy::TelemetryMetricDirection::maximize;
    EXPECT_NE(jy::telemetry_metric_definition_digest(changed).hex,
              original_digest.hex);
  }
}

TEST(TelemetryContractTest, RejectsContextMismatchAndUnknownSemanticEnums) {
  auto value = observation();
  auto& point = std::get<jy::TelemetryLifecycle>(value.payload).metrics.front();
  point.context.clear();
  EXPECT_THROW(jy::validate_telemetry_envelope(value), jy::InvalidArgument);
  point.context = {{"different", "validation"}};
  EXPECT_THROW(jy::validate_telemetry_envelope(value), jy::InvalidArgument);
  point.context = {{"split", "validation"}};
  point.sample_count = 0U;
  EXPECT_THROW(jy::validate_telemetry_envelope(value), jy::InvalidArgument);
  point.sample_count = 1U;
  point.definition.aggregation = static_cast<jy::TelemetryAggregation>(255U);
  EXPECT_THROW(jy::validate_telemetry_envelope(value), jy::InvalidArgument);
}

TEST(TelemetryContractTest, CanonicalizesSignedZeroAndMapInsertionOrder) {
  auto left = observation();
  auto right = left;
  std::get<jy::TelemetryLifecycle>(left.payload).metrics.front().observation =
      jy::make_telemetry_metric_value(-0.0);
  std::get<jy::TelemetryLifecycle>(right.payload).metrics.front().observation =
      jy::make_telemetry_metric_value(0.0);
  right.attributes.clear();
  right.attributes.emplace("phase", "validation");
  right.attributes.emplace("producer", "training");
  EXPECT_EQ(jy::encode_telemetry_envelope(left),
            jy::encode_telemetry_envelope(right));
  EXPECT_EQ(jy::telemetry_content_digest(left),
            jy::telemetry_content_digest(right));
}

TEST(TelemetryContractTest, MatchesIndependentLengthDelimitedOperationVector) {
  // SHA-256 of three UTF-8 strings, each prefixed by its uint64 little-endian
  // byte length: domain, run.test/attempt.1, callback.1. Reference computed
  // with .NET System.Security.Cryptography.SHA256, not the draft C++ codec.
  const auto value = observation();
  EXPECT_EQ(jy::telemetry_idempotency_digest(value.identity).hex,
            "14e621e6b6b65bab13456db988bbaa19cbe6e27d77c4d0d5a090cb801ff4de47");
  auto left = value.identity;
  auto right = value.identity;
  left.stream_id = "ab";
  left.idempotency_key = "c";
  right.stream_id = "a";
  right.idempotency_key = "bc";
  EXPECT_NE(jy::telemetry_idempotency_digest(left),
            jy::telemetry_idempotency_digest(right));
}

TEST(TelemetryContractTest, SeparatesOperationIdentityFromImmutableContent) {
  auto original = observation();
  auto changed = original;
  ++*changed.time.wall_time_unix_ns;
  EXPECT_EQ(jy::telemetry_idempotency_digest(original.identity),
            jy::telemetry_idempotency_digest(changed.identity));
  EXPECT_NE(jy::telemetry_content_digest(original),
            jy::telemetry_content_digest(changed));
  const std::array conflict{original, changed};
  EXPECT_THROW(jy::validate_telemetry_batch(conflict), jy::CompatibilityError);
}

TEST(TelemetryContractTest,
     AllowsIdenticalReplayButRejectsConflictingIdentities) {
  const auto original = observation();
  const std::array retries{original, original};
  EXPECT_NO_THROW(jy::validate_telemetry_batch(retries));

  auto changed = original;
  changed.identity.idempotency_key = "different.operation";
  changed.identity.event_id = "different.event";
  const std::array sequence_conflict{original, changed};
  EXPECT_THROW(jy::validate_telemetry_batch(sequence_conflict),
               jy::CompatibilityError);
  changed.identity.sequence = 2U;
  changed.identity.event_id = original.identity.event_id;
  const std::array event_conflict{original, changed};
  EXPECT_THROW(jy::validate_telemetry_batch(event_conflict),
               jy::CompatibilityError);
  changed.identity.stream_id = "run.test/attempt.2";
  changed.identity.sequence = 1U;
  const std::array separate_stream{original, changed};
  EXPECT_NO_THROW(jy::validate_telemetry_batch(separate_stream));
}

TEST(TelemetryContractTest,
     RejectsFutureSchemaAndZeroSequenceBeforeAcceptance) {
  auto value = observation();
  value.schema_version = 2U;
  EXPECT_THROW(jy::validate_telemetry_envelope(value), jy::CompatibilityError);
  value.schema_version = 1U;
  value.identity.sequence = 0U;
  EXPECT_THROW(jy::validate_telemetry_envelope(value), jy::InvalidArgument);
  auto definition = loss_definition();
  definition.schema_version = 2U;
  EXPECT_THROW(jy::validate_telemetry_metric_definition(definition),
               jy::CompatibilityError);
}

TEST(TelemetryContractTest,
     PreservesUnknownHistoricalTimeWithoutClockFabrication) {
  auto value = observation();
  value.time = {};
  value.attributes["migration.source"] = "training.event-envelope/v1";
  value.attributes["migration.time"] = "unavailable";
  const auto restored =
      jy::decode_telemetry_envelope(jy::encode_telemetry_envelope(value));
  EXPECT_FALSE(restored.time.wall_time_unix_ns.has_value());
  EXPECT_FALSE(restored.time.monotonic_time_ns.has_value());
  EXPECT_TRUE(restored.time.clock_id.empty());
  EXPECT_EQ(jy::telemetry_content_digest(value),
            jy::telemetry_content_digest(restored));
  value.time.monotonic_time_ns = 3U;
  EXPECT_THROW(jy::validate_telemetry_envelope(value), jy::InvalidArgument);
}

TEST(TelemetryContractTest, ValidatesSpanClockAndParentRelationships) {
  auto value = observation();
  value.correlation.span_id = "span.child";
  value.correlation.parent_span_id = "span.parent";
  value.payload = jy::TelemetrySpan{
      .name = "forward", .start_monotonic_ns = 100U, .end_monotonic_ns = 110U};
  EXPECT_NO_THROW(jy::validate_telemetry_envelope(value));
  std::get<jy::TelemetrySpan>(value.payload).end_monotonic_ns = 99U;
  EXPECT_THROW(jy::validate_telemetry_envelope(value), jy::InvalidArgument);
  std::get<jy::TelemetrySpan>(value.payload).end_monotonic_ns = 121U;
  EXPECT_THROW(jy::validate_telemetry_envelope(value), jy::InvalidArgument);
  std::get<jy::TelemetrySpan>(value.payload).end_monotonic_ns = 110U;
  value.correlation.parent_span_id = value.correlation.span_id;
  EXPECT_THROW(jy::validate_telemetry_envelope(value), jy::InvalidArgument);
}

TEST(TelemetryContractTest,
     KeepsDecisionAuditOrderedAndTerminalEvidenceRequired) {
  auto value = observation();
  auto& payload = std::get<jy::TelemetryLifecycle>(value.payload);
  payload.decisions.push_back(
      {"another.actor", "checkpoint", "requested", true});
  EXPECT_THROW(jy::validate_telemetry_envelope(value), jy::InvalidArgument);
  payload.decisions.back().effective = false;
  EXPECT_NO_THROW(jy::validate_telemetry_envelope(value));
  const auto original = jy::telemetry_content_digest(value);
  std::reverse(payload.decisions.begin(), payload.decisions.end());
  EXPECT_NE(jy::telemetry_content_digest(value), original);
  payload.terminal = true;
  value.reliability = jy::TelemetryReliability::best_effort;
  EXPECT_THROW(jy::validate_telemetry_envelope(value), jy::InvalidArgument);
}

TEST(TelemetryContractTest, RejectsInvalidUtf8AndNulWithoutExposingInputText) {
  for (const auto& invalid :
       {std::string("\xC0\xAF", 2U), std::string("\xED\xA0\x80", 3U),
        std::string("\xF4\x90\x80\x80", 4U),
        std::string("secret\0value", 12U)}) {
    auto value = observation();
    value.attributes["test"] = invalid;
    try {
      jy::validate_telemetry_envelope(value);
      FAIL() << "Invalid UTF-8/NUL observation was accepted";
    } catch (const jy::InvalidArgument& error) {
      EXPECT_EQ(std::string(error.what()).find("secret"), std::string::npos);
    }
  }
}

TEST(TelemetryContractTest, EnforcesPerFieldCountEnvelopeAndBatchLimits) {
  auto value = observation();
  auto limits = jy::TelemetryLimits{};
  limits.max_identifier_bytes = 2U;
  EXPECT_THROW(jy::validate_telemetry_envelope(value, limits),
               jy::ResourceExhausted);
  limits = {};
  limits.max_attributes = 1U;
  EXPECT_THROW(jy::validate_telemetry_envelope(value, limits),
               jy::ResourceExhausted);
  const auto bytes = jy::encode_telemetry_envelope(value);
  limits = {};
  limits.max_envelope_bytes = bytes.size();
  EXPECT_NO_THROW(jy::validate_telemetry_envelope(value, limits));
  --limits.max_envelope_bytes;
  EXPECT_THROW(jy::validate_telemetry_envelope(value, limits),
               jy::ResourceExhausted);
  limits = {};
  limits.max_envelope_bytes = bytes.size();
  limits.max_batch_bytes = bytes.size() * 2U - 1U;
  const std::array values{value, value};
  EXPECT_THROW(jy::validate_telemetry_batch(values, limits),
               jy::ResourceExhausted);
  limits.max_batch_events = 1U;
  EXPECT_THROW(jy::validate_telemetry_batch(values, limits),
               jy::ResourceExhausted);
  limits.max_batch_bytes = 1U;
  EXPECT_THROW(jy::validate_telemetry_limits(limits), jy::InvalidArgument);
}

TEST(TelemetryContractTest,
     RejectsTruncationTrailingBytesAndFutureEncodedSchema) {
  const auto bytes = jy::encode_telemetry_envelope(observation());
  for (std::size_t size = 0U; size < bytes.size(); ++size) {
    EXPECT_THROW(jy::decode_telemetry_envelope(
                     std::span<const std::byte>(bytes).first(size)),
                 jy::Error);
  }
  auto trailing = bytes;
  trailing.push_back(std::byte{0});
  EXPECT_THROW(jy::decode_telemetry_envelope(trailing), jy::SerializationError);
  auto future = bytes;
  future[8U + std::string_view("jyppx.telemetry.content/v1").size()] =
      std::byte{2};
  EXPECT_THROW(jy::decode_telemetry_envelope(future), jy::CompatibilityError);
}

TEST(TelemetryContractTest, RejectsUnknownCriticalTagAndImpossibleCounts) {
  auto value = observation();
  value.time = {};
  value.correlation = {};
  value.attributes.clear();
  value.payload = jy::TelemetryMetricBatch{{loss_point()}};
  auto bytes = jy::encode_telemetry_envelope(value);
  const auto tag = simple_payload_tag_offset(value);
  ASSERT_LT(tag, bytes.size());
  ASSERT_EQ(bytes[tag], std::byte{2});
  bytes[tag] = std::byte{255};
  EXPECT_THROW(jy::decode_telemetry_envelope(bytes), jy::CompatibilityError);
  bytes[tag] = std::byte{2};
  std::fill_n(bytes.begin() + static_cast<std::ptrdiff_t>(tag + 1U), 8U,
              std::byte{255});
  EXPECT_THROW(jy::decode_telemetry_envelope(bytes), jy::ResourceExhausted);
}

TEST(TelemetryContractTest, RejectsDuplicateOrUnsortedCanonicalDefinitionKeys) {
  auto definition = loss_definition();
  definition.parameters = {{"alpha", "left"}, {"omega", "right"}};
  const auto original = jy::encode_telemetry_metric_definition(definition);
  const auto key_position = [&](std::string_view key) {
    return std::search(
        original.begin(), original.end(),
        reinterpret_cast<const std::byte*>(key.data()),
        reinterpret_cast<const std::byte*>(key.data() + key.size()));
  };
  const auto alpha = key_position("alpha");
  const auto omega = key_position("omega");
  ASSERT_NE(alpha, original.end());
  ASSERT_NE(omega, original.end());
  auto duplicate = original;
  std::copy_n(alpha, 5U, duplicate.begin() + (omega - original.begin()));
  EXPECT_THROW(jy::decode_telemetry_metric_definition(duplicate),
               jy::SerializationError);
  auto unsorted = original;
  std::copy_n(omega, 5U, unsorted.begin() + (alpha - original.begin()));
  std::copy_n(alpha, 5U, unsorted.begin() + (omega - original.begin()));
  EXPECT_THROW(jy::decode_telemetry_metric_definition(unsorted),
               jy::SerializationError);
}

TEST(TelemetryContractTest, DistinguishesObserverOmissionsFromDurableEvidence) {
  const std::array input{observation()};
  const jy::TelemetrySinkDescriptor evidence{
      .type_id = "jyppx.telemetry-sink.sqlite/v1",
      .sink_id = "sqlite.events",
      .destination_id = "local.run-store",
      .role = jy::TelemetrySinkRole::evidence};
  auto observer = evidence;
  observer.role = jy::TelemetrySinkRole::observer;
  const jy::TelemetryEmitResult accepted{{jy::TelemetryDisposition::accepted}};
  const jy::TelemetryEmitResult duplicate{
      {jy::TelemetryDisposition::duplicate}};
  const jy::TelemetryEmitResult unsupported{
      {jy::TelemetryDisposition::unsupported}};
  const jy::TelemetryEmitResult sampled{
      {jy::TelemetryDisposition::sampled_out}};
  EXPECT_NO_THROW(
      jy::validate_telemetry_emit_result(evidence, input, accepted));
  EXPECT_NO_THROW(
      jy::validate_telemetry_emit_result(evidence, input, duplicate));
  EXPECT_THROW(jy::validate_telemetry_emit_result(evidence, input, unsupported),
               jy::CompatibilityError);
  EXPECT_NO_THROW(
      jy::validate_telemetry_emit_result(observer, input, unsupported));
  EXPECT_THROW(jy::validate_telemetry_emit_result(observer, input, sampled),
               jy::CompatibilityError);
  auto best_effort = input;
  best_effort.front().reliability = jy::TelemetryReliability::best_effort;
  EXPECT_NO_THROW(
      jy::validate_telemetry_emit_result(evidence, best_effort, sampled));
  const jy::TelemetryEmitResult incomplete{};
  EXPECT_THROW(jy::validate_telemetry_emit_result(evidence, input, incomplete),
               jy::CompatibilityError);
}

TEST(TelemetryContractTest, ChecksCancellationBeforeDeadlineWithoutSleeping) {
  std::stop_source stop;
  stop.request_stop();
  const auto expired =
      std::chrono::steady_clock::now() - std::chrono::seconds(1);
  EXPECT_THROW(jy::check_telemetry_operation(stop.get_token(), expired),
               jy::Cancelled);
  EXPECT_THROW(jy::check_telemetry_operation({}, expired),
               jy::DeadlineExceeded);
  EXPECT_NO_THROW(jy::check_telemetry_operation());
}

}  // namespace
