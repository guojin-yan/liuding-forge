// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/training/telemetry_adapter.hpp>

#include <array>
#include <gtest/gtest.h>
#include <limits>

namespace {
namespace jt = jyppx::liuding::training;

jt::EventEnvelope legacy(std::uint64_t index = 2U) {
  jt::EventEnvelope result;
  result.event = {.kind = index == 1U ? jt::CallbackEventKind::run_started
                                      : jt::CallbackEventKind::epoch_finished,
                  .phase = index == 1U ? jt::CallbackPhase::preparing
                                       : jt::CallbackPhase::training,
                  .event_index = index,
                  .run_id = "run.test",
                  .trace_id = "native trace with spaces",
                  .epoch = index - 1U,
                  .global_step = index - 1U,
                  .metrics = {{"loss", 0.25}}};
  result.callback_result.commands = {
      {jt::CallbackCommandKind::request_checkpoint, "checkpoint", "periodic"},
      {jt::CallbackCommandKind::stop, "early_stopping", "patience exhausted"}};
  result.callback_result.effective_command =
      result.callback_result.commands.back();
  return result;
}

TEST(TrainingTelemetryAdapterTest,
     MigratesOldEnvelopeDeterministicallyWithoutInventedSemantics) {
  const auto source = legacy();
  const auto first = jt::migrate_training_envelope_v1(source);
  const auto second = jt::migrate_training_envelope_v1(source);
  EXPECT_EQ(jyppx::liuding::encode_telemetry_envelope(first),
            jyppx::liuding::encode_telemetry_envelope(second));
  EXPECT_EQ(first.identity.sequence, source.event.event_index);
  EXPECT_EQ(first.correlation.run_id, source.event.run_id);
  EXPECT_FALSE(first.time.wall_time_unix_ns.has_value());
  EXPECT_FALSE(first.time.monotonic_time_ns.has_value());
  EXPECT_EQ(first.attributes.at("migration.trace_id"), source.event.trace_id);
  const auto& payload = std::get<jyppx::liuding::TelemetryLifecycle>(first.payload);
  ASSERT_EQ(payload.metrics.size(), 1U);
  EXPECT_EQ(payload.metrics.front().definition.unit,
            jyppx::liuding::TelemetryMetricUnit::unspecified);
  EXPECT_EQ(payload.metrics.front().definition.aggregation,
            jyppx::liuding::TelemetryAggregation::unspecified);
  EXPECT_EQ(
      payload.metrics.front().definition.parameters.at("source_metric_name"),
      "loss");
  ASSERT_EQ(payload.decisions.size(), 2U);
  EXPECT_FALSE(payload.decisions[0].effective);
  EXPECT_TRUE(payload.decisions[1].effective);
}

TEST(TrainingTelemetryAdapterTest,
     LiveAdapterRequiresRegisteredTypedSemanticsAndCapturedTime) {
  auto source = legacy();
  source.event.trace_id = "trace.test";
  jt::TrainingTelemetryAdapterOptions options;
  options.stream_id = "run.test/attempt.2";
  options.attempt_id = "attempt.2";
  EXPECT_THROW(jt::adapt_training_event(source, options, {}),
               jyppx::liuding::CompatibilityError);
  options.metrics["loss"] = {
      .definition = {.metric_id = "training.loss",
                     .unit = jyppx::liuding::TelemetryMetricUnit::dimensionless,
                     .aggregation = jyppx::liuding::TelemetryAggregation::mean,
                     .direction = jyppx::liuding::TelemetryMetricDirection::minimize},
      .context = {}};
  const jyppx::liuding::TelemetryTime time{.wall_time_unix_ns = 123,
                                  .monotonic_time_ns = 40U,
                                  .clock_id = "clock.2"};
  const auto adapted = jt::adapt_training_event(source, options, time);
  EXPECT_EQ(adapted.time.wall_time_unix_ns, 123);
  EXPECT_EQ(adapted.correlation.attempt_id, "attempt.2");
  EXPECT_EQ(adapted.identity.stream_id, "run.test/attempt.2");
}

TEST(TrainingTelemetryAdapterTest,
     RejectsCorruptPhasePriorityAndSourceNonfinite) {
  auto source = legacy();
  source.event.phase = jt::CallbackPhase::validation;
  EXPECT_THROW(jt::migrate_training_envelope_v1(source),
               jyppx::liuding::CompatibilityError);
  source = legacy();
  source.callback_result.effective_command =
      source.callback_result.commands.front();
  EXPECT_THROW(jt::migrate_training_envelope_v1(source),
               jyppx::liuding::CompatibilityError);
  source = legacy();
  source.event.metrics["loss"] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(jt::migrate_training_envelope_v1(source),
               jyppx::liuding::CompatibilityError);
}

TEST(TrainingTelemetryAdapterTest,
     PreservesN1PendingPrefixAndPerSinkRetryCursor) {
  jt::TrainingOutboxV1MigrationInput source;
  source.cursor = {.initialized = true,
                   .terminal = false,
                   .run_id = "run.test",
                   .last_enqueued_event_index = 2U,
                   .last_epoch = 1U,
                   .last_global_step = 1U,
                   .last_delivered_event_index = 1U};
  source.sinks = {
      {.type_id = "legacy.jsonl/v1", .sink_id = "a", .destination_id = "jsonl"},
      {.type_id = "legacy.console/v1",
       .sink_id = "b",
       .destination_id = "console"}};
  source.pending.push_back({legacy(), 1U});
  const std::array current{
      jyppx::liuding::TelemetrySinkDescriptor{.type_id = "neutral.jsonl/v1",
                                     .sink_id = "a",
                                     .destination_id = "jsonl"},
      jyppx::liuding::TelemetrySinkDescriptor{.type_id = "neutral.console/v1",
                                     .sink_id = "b",
                                     .destination_id = "console"}};
  const std::array pairs{std::make_pair(std::string("legacy.jsonl/v1"),
                                        std::string("neutral.jsonl/v1")),
                         std::make_pair(std::string("legacy.console/v1"),
                                        std::string("neutral.console/v1"))};
  const auto candidate = jt::migrate_training_outbox_v1(source, current, pairs);
  EXPECT_EQ(candidate.cursor.last_delivered_event_index, 1U);
  ASSERT_EQ(candidate.pending.size(), 1U);
  EXPECT_EQ(candidate.pending.front().next_sink_index, 1U);
  EXPECT_EQ(candidate.pending.front().envelope.identity.sequence, 2U);
  EXPECT_GT(candidate.canonical_pending_bytes, 16U);
  auto changed = current;
  changed[1].destination_id = "a-different-console";
  EXPECT_THROW(jt::migrate_training_outbox_v1(source, changed, pairs),
               jyppx::liuding::CompatibilityError);
  EXPECT_THROW(jt::migrate_training_outbox_v1(source, current, {}),
               jyppx::liuding::CompatibilityError);
  source.pending.front().next_sink_index = 3U;
  EXPECT_THROW(jt::migrate_training_outbox_v1(source, current, pairs),
               jyppx::liuding::CompatibilityError);
  EXPECT_EQ(source.cursor.last_enqueued_event_index, 2U);
}

TEST(TrainingTelemetryAdapterTest,
     MigrationFailureDoesNotMutateDetachedSource) {
  jt::TrainingOutboxV1MigrationInput source;
  source.cursor = {.initialized = true,
                   .run_id = "run.test",
                   .last_enqueued_event_index = 1U};
  source.sinks = {
      {.type_id = "same/v1", .sink_id = "a", .destination_id = "same"}};
  source.pending = {{legacy(1U), 0U}};
  const std::array current{jyppx::liuding::TelemetrySinkDescriptor{
      .type_id = "same/v1", .sink_id = "a", .destination_id = "same"}};
  EXPECT_THROW(
      jt::migrate_training_outbox_v1(source, current, {}, {}, 4096U, 1U),
      jyppx::liuding::ResourceExhausted);
  ASSERT_EQ(source.pending.size(), 1U);
  EXPECT_EQ(source.pending.front().next_sink_index, 0U);
  EXPECT_EQ(source.pending.front().envelope.event.event_index, 1U);
}

TEST(TrainingTelemetryAdapterTest,
     LegacySinkCompatibilityCapturesRealTimeWithoutInventingMetrics) {
  const auto source = legacy();
  const jyppx::liuding::TelemetryTime time{.wall_time_unix_ns = 456,
                                  .monotonic_time_ns = 90U,
                                  .clock_id = "clock.live"};
  const auto result =
      jt::adapt_training_compatibility_event(source, "stream.live", time);
  EXPECT_EQ(result.identity.stream_id, "stream.live");
  EXPECT_EQ(result.time.wall_time_unix_ns, 456);
  EXPECT_EQ(result.time.monotonic_time_ns, 90U);
  EXPECT_EQ(result.attributes.at("compatibility.source"),
            "training.event-envelope/v1");
  EXPECT_FALSE(result.attributes.contains("migration.time"));
  EXPECT_FALSE(result.attributes.contains("migration.source"));
  const auto& payload = std::get<jyppx::liuding::TelemetryLifecycle>(result.payload);
  EXPECT_EQ(payload.metrics.at(0U).definition.unit,
            jyppx::liuding::TelemetryMetricUnit::unspecified);
  EXPECT_EQ(
      jyppx::liuding::encode_telemetry_envelope(result),
      jyppx::liuding::encode_telemetry_envelope(
          jt::adapt_training_compatibility_event(source, "stream.live", time)));
}

}  // namespace
