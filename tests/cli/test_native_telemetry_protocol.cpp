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
#include <gtest/gtest.h>
#include <map>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace jy = jyppx::liuding;
namespace cli = jy::cli;
namespace ex = jy::experiment;
using Json = nlohmann::json;

jy::TelemetryMetricPoint point(double number = -0.0) {
  return {.definition = {.metric_id = "validation.loss",
                         .version = UINT32_MAX,
                         .unit = jy::TelemetryMetricUnit::dimensionless,
                         .aggregation = jy::TelemetryAggregation::mean,
                         .direction = jy::TelemetryMetricDirection::minimize,
                         .parameters = {{"average", "mean"}},
                         .context_keys = {"dataset", "split"}},
          .step = {jy::TelemetryStepKind::global_step, UINT64_MAX},
          .observation = {jy::TelemetryValueStatus::finite, number, ""},
          .context = {{"dataset", "fixture"}, {"split", "validation"}},
          .sample_count = UINT64_MAX};
}

jy::TelemetryEnvelope event(std::uint64_t sequence = 1U) {
  return {
      .identity = {.stream_id = "worker.test",
                   .event_id = "event." + std::to_string(sequence),
                   .idempotency_key = "operation." + std::to_string(sequence),
                   .sequence = sequence},
      .time = {.wall_time_unix_ns = INT64_MIN,
               .monotonic_time_ns = UINT64_MAX,
               .clock_id = "clock.fixture"},
      .correlation = {.project_id = "project.fixture",
                      .run_id = "run.fixture",
                      .attempt_id = "attempt.fixture",
                      .trace_id = "opaque.trace",
                      .span_id = "opaque.span",
                      .parent_span_id = "opaque.parent"},
      .attributes = {{"origin", "immutable"}},
      .payload =
          jy::TelemetryLog{jy::TelemetryLogSeverity::info, "already redacted"}};
}

Json projection(const jy::TelemetryEnvelope& value) {
  return cli::parse_native_telemetry_json(ex::telemetry_projection_json(value),
                                          16U * 1024U * 1024U);
}

Json request(const std::vector<jy::TelemetryEnvelope>& values) {
  Json events = Json::array();
  std::string identity = "jyppx.telemetry.sidecar-batch/v1";
  for (const auto& value : values) {
    const auto digest = jy::telemetry_content_digest(value).hex;
    identity += digest;
    events.push_back(
        {{"content_digest", digest}, {"envelope", projection(value)}});
  }
  return {{"protocol_version", 1U},
          {"backend", "opentelemetry"},
          {"operation", "emit"},
          {"request_id",
           jy::sha256_hex({reinterpret_cast<const std::byte*>(identity.data()),
                           identity.size()})},
          {"events", std::move(events)}};
}

class PrefixSink final : public jy::TelemetrySink {
 public:
  const jy::TelemetrySinkDescriptor& descriptor() const noexcept override {
    return descriptor_;
  }
  jy::TelemetryEmitResult emit(std::span<const jy::TelemetryEnvelope> values,
                               std::stop_token,
                               jy::TelemetryDeadline) override {
    ++calls;
    jy::TelemetryEmitResult result;
    for (const auto& value : values) {
      const auto key =
          std::pair(value.identity.stream_id, value.identity.idempotency_key);
      const auto content = ex::telemetry_projection_json(value);
      if (const auto found = seen.find(key); found != seen.end()) {
        if (found->second != content)
          throw jy::InvalidArgument("Immutable projection changed");
        result.dispositions.push_back(jy::TelemetryDisposition::duplicate);
      } else {
        if (fail_after_prefix && counters_.exported_events == 1U) {
          fail_after_prefix = false;
          throw jy::BackendError("Injected second-envelope rejection");
        }
        seen.emplace(key, content);
        ++counters_.exported_events;
        result.dispositions.push_back(jy::TelemetryDisposition::accepted);
      }
    }
    return result;
  }
  jy::TelemetryFlushResult flush(std::stop_token,
                                 jy::TelemetryDeadline) override {
    return counters_;
  }
  jy::TelemetryFlushResult close(std::stop_token stop,
                                 jy::TelemetryDeadline deadline) override {
    return flush(stop, deadline);
  }
  unsigned calls = 0U;
  bool fail_after_prefix = false;
  jy::TelemetryFlushResult counters_;

 private:
  jy::TelemetrySinkDescriptor descriptor_{
      .type_id = "jyppx.telemetry-sink.native-worker-test/v1",
      .sink_id = "test.native-worker",
      .destination_id = "test.native-worker",
      .role = jy::TelemetrySinkRole::observer};
  std::map<std::pair<std::string, std::string>, std::string> seen;
};

TEST(NativeTelemetryProtocol, AllPayloadsKeepCanonicalAndProjectionIdentity) {
  auto base = event(UINT64_MAX);
  std::vector<jy::TelemetryPayload> payloads{
      jy::TelemetryLifecycle{"training.finished",
                             true,
                             {point()},
                             {{"callback.stop", "stop", "complete", true}}},
      jy::TelemetryMetricBatch{{point()}},
      jy::TelemetryLog{jy::TelemetryLogSeverity::warning, "fixed log"},
      jy::TelemetrySpan{"train.batch", UINT64_MAX - 1U, UINT64_MAX,
                        jy::TelemetrySpanStatus::ok, ""},
      jy::TelemetryResourceSample{"device.cpu", {point()}},
      jy::TelemetryArtifactReference{
          "checkpoint.best",
          {{jy::ArtifactDigestAlgorithm::sha256, std::string(64U, 'a')},
           123U,
           "best.model",
           "application/octet-stream",
           "fixture"}}};
  for (const auto& payload : payloads) {
    base.payload = payload;
    const auto projected = projection(base);
    const auto decoded = cli::decode_native_telemetry_projection(
        projected, jy::telemetry_content_digest(base).hex);
    EXPECT_EQ(jy::encode_telemetry_envelope(decoded),
              jy::encode_telemetry_envelope(base));
    EXPECT_EQ(ex::telemetry_projection_json(decoded), projected.dump());
    EXPECT_EQ(decoded.identity.sequence, UINT64_MAX);
    EXPECT_EQ(decoded.time.wall_time_unix_ns, INT64_MIN);
    EXPECT_EQ(decoded.time.monotonic_time_ns, UINT64_MAX);
  }
}

TEST(NativeTelemetryProtocol, SignedZeroAndSymbolicNonfiniteRemainDistinct) {
  auto value = event();
  auto values = std::vector{point()};
  for (const auto status : {jy::TelemetryValueStatus::not_a_number,
                            jy::TelemetryValueStatus::positive_infinity,
                            jy::TelemetryValueStatus::negative_infinity,
                            jy::TelemetryValueStatus::missing}) {
    auto missing = point();
    missing.observation = {
        status, std::nullopt,
        status == jy::TelemetryValueStatus::missing ? "not measured" : ""};
    missing.sample_count = std::nullopt;
    values.push_back(missing);
  }
  value.payload = jy::TelemetryMetricBatch{values};
  const auto decoded = cli::decode_native_telemetry_projection(
      projection(value), jy::telemetry_content_digest(value).hex);
  const auto& points =
      std::get<jy::TelemetryMetricBatch>(decoded.payload).points;
  ASSERT_TRUE(points.front().observation.value);
  EXPECT_TRUE(std::signbit(*points.front().observation.value));
  EXPECT_FALSE(points.back().observation.value);
  EXPECT_EQ(points.back().observation.reason, "not measured");
}

TEST(NativeTelemetryProtocol, RejectsNarrowingUnknownFieldsAndDigestDrift) {
  auto value = event();
  value.payload = jy::TelemetryMetricBatch{{point()}};
  const auto original = projection(value);
  const auto digest = jy::telemetry_content_digest(value).hex;
  std::vector<Json> invalid;
  auto add = [&](auto mutation) {
    auto changed = original;
    mutation(changed);
    invalid.push_back(std::move(changed));
  };
  add([](Json& v) { v["schema_version"] = UINT64_MAX; });
  add([](Json& v) { v["time"]["wall_time_unix_ns"] = UINT64_MAX; });
  add([](Json& v) { v["identity"]["sequence"] = -1; });
  add([](Json& v) { v["reliability"] = 257U; });
  add([](Json& v) { v["payload"]["points"][0]["value"] = UINT64_MAX; });
  add([](Json& v) {
    v["payload"]["points"][0]["definition"]["definition_digest"] =
        std::string(64U, '0');
  });
  add([](Json& v) {
    v["payload"]["points"][0]["definition"]["context_keys"] =
        Json::array({"split", "dataset"});
  });
  add([](Json& v) { v["payload"]["future_field"] = true; });
  for (const auto& changed : invalid)
    EXPECT_THROW(cli::decode_native_telemetry_projection(changed, digest),
                 jy::Error);
  EXPECT_THROW(
      cli::decode_native_telemetry_projection(original, std::string(64U, '0')),
      jy::Error);
}

TEST(NativeTelemetryProtocol, BoundedJsonRejectsDuplicatesDepthAndBadUtf8) {
  EXPECT_THROW(cli::parse_native_telemetry_json("{\"x\":1,\"x\":2}", 100U),
               jy::Error);
  const std::string deep = std::string(34U, '[') + "0" + std::string(34U, ']');
  EXPECT_THROW(cli::parse_native_telemetry_json(deep, 100U), jy::Error);
  EXPECT_THROW(cli::parse_native_telemetry_json("{}", 1U), jy::Error);
  const std::string malformed = "\"" + std::string(1U, '\xff') + "\"";
  EXPECT_THROW(cli::parse_native_telemetry_json(malformed, 100U), jy::Error);
}

TEST(NativeTelemetryProtocol, InvalidWholeBatchNeverReachesSink) {
  PrefixSink sink;
  cli::NativeTelemetryProtocol protocol(sink);
  auto batch = request({event(1U), event(2U)});
  batch["events"][1]["envelope"]["payload"]["body"] = "digest drift";
  const auto reply = protocol.exchange(batch);
  EXPECT_TRUE(reply.contains("error"));
  EXPECT_FALSE(reply.contains("receipts"));
  EXPECT_FALSE(reply.contains("counters"));
  EXPECT_EQ(sink.calls, 0U);
}

TEST(NativeTelemetryProtocol, FailedPrefixHasNoFabricatedAckAndRetryKeepsSeen) {
  PrefixSink sink;
  sink.fail_after_prefix = true;
  cli::NativeTelemetryProtocol protocol(sink);
  const auto batch = request({event(1U), event(2U)});
  const auto failed = protocol.exchange(batch);
  EXPECT_TRUE(failed.contains("error"));
  EXPECT_FALSE(failed.contains("receipts"));
  EXPECT_FALSE(failed.contains("counters"));
  EXPECT_EQ(sink.counters_.exported_events, 1U);
  const auto retried = protocol.exchange(batch);
  EXPECT_FALSE(retried.contains("error"));
  EXPECT_EQ(retried.at("receipts"), Json::array({2U, 1U}));
  EXPECT_EQ(retried.at("counters").at("exported_events"), 2U);
  EXPECT_EQ(protocol.exchange(batch).at("receipts"), Json::array({2U, 2U}));
}

TEST(NativeTelemetryProtocol, RequestIdCacheCannotHideSignedZeroConflict) {
  PrefixSink sink;
  cli::NativeTelemetryProtocol protocol(sink);
  auto value = event();
  value.payload = jy::TelemetryMetricBatch{{point(-0.0)}};
  const auto negative = request({value});
  EXPECT_FALSE(protocol.exchange(negative).contains("error"));
  value.payload = jy::TelemetryMetricBatch{{point(0.0)}};
  const auto positive = request({value});
  EXPECT_EQ(positive.at("request_id"), negative.at("request_id"));
  EXPECT_TRUE(protocol.exchange(positive).contains("error"));
  EXPECT_EQ(sink.calls, 2U);
  EXPECT_EQ(sink.counters_.exported_events, 1U);
}

TEST(NativeTelemetryProtocol, DefaultBackendAndErrorsRemainOtlp) {
  PrefixSink sink;
  cli::NativeTelemetryProtocol protocol(sink);
  auto batch = request({event()});
  batch["backend"] = "tensorboard";
  const auto rejected = protocol.exchange(batch);
  EXPECT_EQ(rejected.at("error"), "native_otlp.operation_rejected");
  EXPECT_FALSE(rejected.contains("receipts"));
  EXPECT_FALSE(rejected.contains("counters"));
  EXPECT_EQ(sink.calls, 0U);
  batch["backend"] = "opentelemetry";
  EXPECT_FALSE(protocol.exchange(batch).contains("error"));
  EXPECT_EQ(sink.calls, 1U);
}

TEST(NativeTelemetryProtocol, TensorBoardBackendKeepsReplayAndCloseContract) {
  PrefixSink sink;
  const std::string backend = "tensorboard";
  cli::NativeTelemetryProtocol protocol(sink, {}, backend);
  auto batch = request({event()});
  const auto wrong_backend = protocol.exchange(batch);
  EXPECT_EQ(wrong_backend.at("error"), "native_tensorboard.operation_rejected");
  EXPECT_EQ(sink.calls, 0U);
  batch["backend"] = "tensorboard";
  EXPECT_EQ(protocol.exchange(batch).at("receipts"), Json::array({1U}));
  EXPECT_EQ(protocol.exchange(batch).at("receipts"), Json::array({2U}));
  const Json close{{"protocol_version", 1U},
                   {"backend", "tensorboard"},
                   {"request_id", "close"},
                   {"operation", "close"}};
  const auto closed = protocol.exchange(close);
  EXPECT_EQ(closed.at("counters").at("exported_events"), 1U);
  EXPECT_EQ(protocol.exchange(close), closed);
  const auto late = protocol.exchange(batch);
  EXPECT_EQ(late.at("error"), "native_tensorboard.operation_rejected");
  EXPECT_FALSE(late.contains("receipts"));
  EXPECT_FALSE(late.contains("counters"));
  EXPECT_EQ(sink.calls, 2U);
}

TEST(NativeTelemetryProtocol, TensorBoardFailureDoesNotAcknowledgePrefix) {
  PrefixSink sink;
  sink.fail_after_prefix = true;
  cli::NativeTelemetryProtocol protocol(sink, {}, "tensorboard");
  auto batch = request({event(1U), event(2U)});
  batch["backend"] = "tensorboard";
  const auto failed = protocol.exchange(batch);
  EXPECT_EQ(failed.at("error"), "native_tensorboard.operation_rejected");
  EXPECT_FALSE(failed.contains("receipts"));
  EXPECT_FALSE(failed.contains("counters"));
  EXPECT_EQ(sink.counters_.exported_events, 1U);
  EXPECT_EQ(protocol.exchange(batch).at("receipts"), Json::array({2U, 1U}));
}

TEST(NativeTelemetryProtocol, UnknownBackendIsRejectedBeforeSinkUse) {
  PrefixSink sink;
  EXPECT_THROW(cli::NativeTelemetryProtocol(sink, {}, "future"),
               jy::InvalidArgument);
  EXPECT_EQ(sink.calls, 0U);
}

}  // namespace
