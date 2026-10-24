// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_TESTS_FUZZ_TELEMETRY_SEED_HPP
#define JYPPX_TESTS_FUZZ_TELEMETRY_SEED_HPP

#include <jyppx/liuding/foundation/telemetry.hpp>

#include <cstdint>
#include <string>

namespace jyppx::liuding::fuzz {

inline TelemetryLimits telemetry_limits() {
  return {.max_identifier_bytes = 256U,
          .max_text_bytes = 512U,
          .max_attributes = 16U,
          .max_context_fields = 8U,
          .max_metrics_per_envelope = 32U,
          .max_decisions_per_envelope = 16U,
          .max_batch_events = 4U,
          .max_envelope_bytes = 64U * 1024U,
          .max_batch_bytes = 256U * 1024U,
          .max_artifact_reference_bytes = 1024U * 1024U};
}

inline TelemetryMetricDefinition seed_definition() {
  return {.schema_version = 1U,
          .metric_id = "classification.accuracy",
          .version = 1U,
          .unit = TelemetryMetricUnit::ratio,
          .aggregation = TelemetryAggregation::mean,
          .direction = TelemetryMetricDirection::maximize,
          .parameters = {{"average", "micro"}},
          .context_keys = {"split"}};
}

inline TelemetryMetricPoint seed_point() {
  return {.definition = seed_definition(),
          .step = {.kind = TelemetryStepKind::sample, .value = 10U},
          .observation = make_telemetry_metric_value(0.9),
          .context = {{"split", "validation"}},
          .sample_count = 10U};
}

inline TelemetryEnvelope seed_envelope(unsigned kind = 2U) {
  TelemetryEnvelope result{.identity = {.stream_id = "fuzz.stream",
                                        .event_id = "event.1",
                                        .idempotency_key = "operation.1",
                                        .sequence = 1U},
                           .time = {.wall_time_unix_ns = 1700000000123456789LL,
                                    .monotonic_time_ns = 100U,
                                    .clock_id = "fuzz.clock"},
                           .correlation = {.project_id = "project.fuzz",
                                           .run_id = "run.fuzz",
                                           .attempt_id = "attempt.1",
                                           .trace_id = "trace.fuzz",
                                           .span_id = "span.fuzz"},
                           .attributes = {{"source", "fuzzer"}}};
  switch (kind % 6U) {
    case 0U:
      result.payload =
          TelemetryLifecycle{.name = "training.epoch_finished",
                             .metrics = {seed_point()},
                             .decisions = {{.actor_id = "early_stop",
                                            .action = "continue",
                                            .reason = "budget remains",
                                            .effective = true}}};
      break;
    case 1U:
      result.payload = TelemetryMetricBatch{{seed_point()}};
      break;
    case 2U:
      result.payload = TelemetryLog{TelemetryLogSeverity::info, "redacted log"};
      break;
    case 3U:
      result.payload = TelemetrySpan{.name = "training.step",
                                     .start_monotonic_ns = 1U,
                                     .end_monotonic_ns = 100U,
                                     .status = TelemetrySpanStatus::ok};
      break;
    case 4U:
      result.payload = TelemetryResourceSample{"cpu.0", {seed_point()}};
      break;
    default:
      result.payload = TelemetryArtifactReference{
          .role = "checkpoint.last",
          .artifact = {.digest = {.hex = std::string(64U, 'a')},
                       .size_bytes = 16U,
                       .logical_name = "checkpoint",
                       .media_type = "application/octet-stream",
                       .producer = "fuzz.fixture"}};
      break;
  }
  return result;
}

}  // namespace jyppx::liuding::fuzz

#endif  // JYPPX_TESTS_FUZZ_TELEMETRY_SEED_HPP
