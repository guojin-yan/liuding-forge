// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "otlp_preflight.hpp"

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/telemetry_native/otlp_projection.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <stop_token>
#include <utility>
#include <vector>

namespace {
namespace forge = jyppx::liuding;
namespace ex = forge::experiment;
namespace detail = ex::detail;

forge::TelemetryEnvelope Event(std::uint64_t sequence = 1U) {
  forge::TelemetryMetricPoint point;
  point.definition.metric_id = "validation.loss";
  point.step.value = sequence;
  point.observation = forge::make_telemetry_metric_value(0.125);
  return {.identity = {.stream_id = "otlp.test",
                       .event_id = std::to_string(sequence),
                       .idempotency_key = std::to_string(sequence),
                       .sequence = sequence},
          .time = {.wall_time_unix_ns = 1700000000000000000LL},
          .payload = forge::TelemetryMetricBatch{{point}}};
}

class ControlledSender final : public detail::OtlpRequestSender {
 public:
  std::size_t fail_at = 0U;
  std::vector<ex::OtlpSignal> calls;
  std::string send(const ex::PreparedOtlpRequest& request, std::size_t,
                   std::stop_token, forge::TelemetryDeadline) override {
    calls.push_back(request.signal);
    if (fail_at != 0U && calls.size() == fail_at)
      throw forge::IoError("Controlled sender failure");
    return {};  // Empty official response protobuf, not a simulated HTTP
                // success claim.
  }
};

detail::OtlpPreflightOptions Options() {
  detail::OtlpPreflightOptions options;
  options.mapping.resource_instance_id = "test.instance";
  return options;
}

TEST(NativeOtlpProjection, OptionsPreserveReferenceBounds) {
  EXPECT_THROW(ex::OtlpProjectionEncoder{ex::OtlpProjectionOptions{}},
               forge::InvalidArgument);
  auto options = Options().mapping;
  options.resource_instance_id = std::string(1U, static_cast<char>(255));
  EXPECT_THROW(ex::OtlpProjectionEncoder{options}, forge::InvalidArgument);
  options = Options().mapping;
  ++options.limits.max_text_bytes;
  EXPECT_THROW(ex::OtlpProjectionEncoder{options}, forge::InvalidArgument);
  options = Options().mapping;
  options.max_prepared_bytes = 1U;
  EXPECT_THROW(ex::OtlpProjectionEncoder{options}, forge::InvalidArgument);
}

TEST(NativeOtlpProjection,
     MissingWallRemainsLogOnlyAndNegativeMetricsUnsupported) {
  const ex::OtlpProjectionEncoder encoder(Options().mapping);
  auto event = Event();
  event.time.wall_time_unix_ns.reset();
  EXPECT_TRUE(encoder.encode(event).empty());
  event.payload =
      forge::TelemetryLog{forge::TelemetryLogSeverity::info, "no wall"};
  const auto request = encoder.encode(event);
  ASSERT_EQ(request.size(), 1U);
  EXPECT_EQ(request.front().signal, ex::OtlpSignal::logs);
  event.time.wall_time_unix_ns = -1;
  EXPECT_TRUE(encoder.encode(event).empty());
}

TEST(NativeOtlpProjection, MissingValuesKeepTheMetricAndAuditPair) {
  const ex::OtlpProjectionEncoder encoder(Options().mapping);
  auto event = Event();
  std::get<forge::TelemetryMetricBatch>(event.payload)
      .points.front()
      .observation =
      forge::make_telemetry_metric_value(std::nullopt, "not measured");
  const auto requests = encoder.encode(event);
  ASSERT_EQ(requests.size(), 2U);
  EXPECT_EQ(requests[0].signal, ex::OtlpSignal::metrics);
  EXPECT_EQ(requests[1].signal, ex::OtlpSignal::logs);
  EXPECT_FALSE(requests[0].body.empty());
  EXPECT_FALSE(requests[1].body.empty());
}

TEST(NativeOtlpProjection,
     ExplicitSpanAnchorsRejectUnsignedUnderflowAndOverflow) {
  auto event = Event();
  event.time = {
      .wall_time_unix_ns = -1, .monotonic_time_ns = 2U, .clock_id = "clock"};
  event.correlation = {.trace_id = "trace", .span_id = "span"};
  event.payload =
      forge::TelemetrySpan{.name = "phase",
                           .start_monotonic_ns = 1U,
                           .end_monotonic_ns = 2U,
                           .status = forge::TelemetrySpanStatus::ok};
  auto options = Options().mapping;
  options.clock_mappings = {{"clock", {3U, 0U}}};
  EXPECT_TRUE(ex::OtlpProjectionEncoder(options).encode(event).empty());
  options.clock_mappings = {
      {"clock", {0U, (std::numeric_limits<std::uint64_t>::max)()}}};
  EXPECT_TRUE(ex::OtlpProjectionEncoder(options).encode(event).empty());
  options.clock_mappings = {{"clock", {0U, 0U}}};
  const auto encoded = ex::OtlpProjectionEncoder(options).encode(event);
  ASSERT_EQ(encoded.size(), 1U);
  EXPECT_EQ(encoded.front().signal, ex::OtlpSignal::traces);
}

TEST(NativeOtlpPreflight, WholeBatchValidationPrecedesEverySenderCall) {
  ControlledSender sender;
  detail::OtlpDeliveryState state(Options(), sender);
  auto invalid = Event(2U);
  invalid.identity.event_id.clear();
  EXPECT_THROW(state.emit(std::array{Event(), invalid}),
               forge::InvalidArgument);
  EXPECT_TRUE(sender.calls.empty());
  EXPECT_EQ(state.flush().exported_events, 0U);
}

TEST(NativeOtlpPreflight, EncodedAndIndexLimitsFailBeforeEffects) {
  for (unsigned mode = 0U; mode < 3U; ++mode) {
    ControlledSender sender;
    auto options = Options();
    if (mode == 0U) options.mapping.max_request_bytes = 32U;
    if (mode == 1U) options.max_batch_prepared_bytes = 32U;
    if (mode == 2U) options.max_seen_bytes = 1U;
    detail::OtlpDeliveryState state(options, sender);
    EXPECT_THROW(state.emit(std::array{Event()}), forge::ResourceExhausted);
    EXPECT_TRUE(sender.calls.empty());
    EXPECT_EQ(state.flush().exported_events, 0U);
  }
}

TEST(NativeOtlpPreflight, AcceptedUnsupportedAndReplayAreCountedExactlyOnce) {
  ControlledSender sender;
  detail::OtlpDeliveryState state(Options(), sender);
  auto unsupported = Event(2U);
  unsupported.time.wall_time_unix_ns = -1;
  const auto result =
      state.emit(std::array{Event(), Event(), unsupported, unsupported});
  EXPECT_EQ(result.dispositions,
            (std::vector{forge::TelemetryDisposition::accepted,
                         forge::TelemetryDisposition::duplicate,
                         forge::TelemetryDisposition::unsupported,
                         forge::TelemetryDisposition::unsupported}));
  ASSERT_EQ(sender.calls.size(), 2U);
  state.emit(std::array{Event(), unsupported});
  EXPECT_EQ(sender.calls.size(), 2U);
  EXPECT_EQ(state.close().exported_events, 1U);
  EXPECT_EQ(state.flush().unsupported_events, 1U);
}

TEST(NativeOtlpPreflight, ImmutableConflictIsRefusedBeforeNewEffects) {
  ControlledSender sender;
  detail::OtlpDeliveryState state(Options(), sender);
  state.emit(std::array{Event()});
  auto conflicting = Event();
  conflicting.attributes["changed"] = "yes";
  EXPECT_THROW(state.emit(std::array{Event(2U), conflicting}),
               forge::CompatibilityError);
  EXPECT_EQ(sender.calls.size(), 2U);
  EXPECT_EQ(state.flush().exported_events, 1U);
}

TEST(NativeOtlpPreflight, SuccessfulPrefixSurvivesLaterFailure) {
  ControlledSender sender;
  sender.fail_at = 3U;
  detail::OtlpDeliveryState state(Options(), sender);
  EXPECT_THROW(state.emit(std::array{Event(), Event(2U)}), forge::IoError);
  EXPECT_EQ(state.flush().exported_events, 1U);
  const auto retry = state.emit(std::array{Event(), Event(2U)});
  EXPECT_EQ(retry.dispositions,
            (std::vector{forge::TelemetryDisposition::duplicate,
                         forge::TelemetryDisposition::accepted}));
  EXPECT_EQ(sender.calls.size(), 5U);
  EXPECT_EQ(state.close().exported_events, 2U);
}

TEST(NativeOtlpPreflight, AuditFailureDoesNotCommitTheMetricEvent) {
  ControlledSender sender;
  sender.fail_at = 2U;
  detail::OtlpDeliveryState state(Options(), sender);
  EXPECT_THROW(state.emit(std::array{Event()}), forge::IoError);
  EXPECT_EQ(state.flush().exported_events, 0U);
  EXPECT_EQ(state.emit(std::array{Event()}).dispositions.front(),
            forge::TelemetryDisposition::accepted);
  EXPECT_EQ(sender.calls,
            (std::vector{ex::OtlpSignal::metrics, ex::OtlpSignal::logs,
                         ex::OtlpSignal::metrics, ex::OtlpSignal::logs}));
}

TEST(NativeOtlpPreflight,
     LiveZeroAndCanonicalRecoveryKeepProjectionConflictsExplicit) {
  auto live = Event();
  auto& point =
      std::get<forge::TelemetryMetricBatch>(live.payload).points.front();
  point.observation = {.status = forge::TelemetryValueStatus::finite,
                       .value = -0.0};
  ASSERT_TRUE(std::signbit(*point.observation.value));
  const auto bytes = forge::encode_telemetry_envelope(live);
  const auto recovered = forge::decode_telemetry_envelope(bytes);
  EXPECT_FALSE(
      std::signbit(*std::get<forge::TelemetryMetricBatch>(recovered.payload)
                        .points.front()
                        .observation.value));
  EXPECT_EQ(forge::telemetry_content_digest(live).hex,
            forge::telemetry_content_digest(recovered).hex);
  const ex::OtlpProjectionEncoder encoder(Options().mapping);
  const auto live_requests = encoder.encode(live);
  const auto recovered_requests = encoder.encode(recovered);
  ASSERT_EQ(live_requests.size(), 2U);
  ASSERT_EQ(recovered_requests.size(), 2U);
  EXPECT_NE(live_requests.front().body, recovered_requests.front().body);
  ControlledSender sender;
  detail::OtlpDeliveryState state(Options(), sender);
  state.emit(std::array{live});
  EXPECT_THROW(state.emit(std::array{recovered}), forge::CompatibilityError);
  EXPECT_EQ(sender.calls.size(), 2U);
  EXPECT_EQ(state.close().exported_events, 1U);
}

TEST(NativeOtlpPreflight, CancellationDeadlineAndClosedStateStayExplicit) {
  ControlledSender sender;
  detail::OtlpDeliveryState state(Options(), sender);
  std::stop_source stopped;
  stopped.request_stop();
  EXPECT_THROW(state.emit(std::array{Event()}, stopped.get_token()),
               forge::Cancelled);
  EXPECT_THROW(state.emit(std::array{Event()}, {},
                          std::chrono::steady_clock::now() -
                              std::chrono::milliseconds(1)),
               forge::DeadlineExceeded);
  EXPECT_TRUE(sender.calls.empty());
  EXPECT_EQ(state.close().exported_events, 0U);
  EXPECT_EQ(state.close(stopped.get_token()).exported_events, 0U);
  EXPECT_THROW(state.emit(std::array{Event()}), forge::NotReady);
}
}  // namespace
