// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

// Draft only: not registered, compiled, or executed during chapter freeze.

#include <jyppx/liuding/experiment/telemetry_exporters.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <gtest/gtest.h>
#include <iterator>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace jy = jyppx::liuding;
namespace ex = jyppx::liuding::experiment;
using Json = nlohmann::json;

class DraftDirectory {
 public:
  DraftDirectory() {
    static std::atomic<unsigned> sequence{0U};
    const auto stamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    path = std::filesystem::temp_directory_path() /
           ("jyppx-draft-telemetry-" + std::to_string(stamp) + "-" +
            std::to_string(sequence.fetch_add(1U)));
    if (!std::filesystem::create_directory(path)) {
      throw std::runtime_error("Cannot acquire unique draft-test directory");
    }
  }
  ~DraftDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
  std::filesystem::path path;
};

jy::TelemetryEnvelope event(std::uint64_t sequence = 1U) {
  return {
      .identity = {.stream_id = "exporter.test/attempt.1",
                   .event_id = "event." + std::to_string(sequence),
                   .idempotency_key = "operation." + std::to_string(sequence),
                   .sequence = sequence},
      .time = {.wall_time_unix_ns = 1700000000123456789LL},
      .correlation = {.project_id = "project.test", .run_id = "run.test"},
      .reliability = jy::TelemetryReliability::required,
      .payload =
          jy::TelemetryLog{jy::TelemetryLogSeverity::info, "already redacted"}};
}

std::string read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

void write_file(const std::filesystem::path& path, std::string_view content) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output << content;
  ASSERT_TRUE(output.good());
}

class FakeTransport final : public ex::TelemetrySidecarTransport {
 public:
  using Handler = std::function<std::string(const Json&)>;
  std::string transact(std::string_view request, std::stop_token stop,
                       jy::TelemetryDeadline deadline) override {
    jy::check_telemetry_operation(stop, deadline);
    ++calls;
    last_request = Json::parse(request);
    return handler ? handler(last_request) : success(last_request).dump();
  }
  Json success(const Json& request) {
    Json result{{"protocol_version", 1U},
                {"request_id", request.at("request_id")},
                {"counters",
                 {{"exported_events", exported},
                  {"sampled_out_events", 0U},
                  {"unsupported_events", 0U}}}};
    if (request.at("operation") == "emit") {
      result["receipts"] = Json::array();
      for (const auto& unused : request.at("events")) {
        static_cast<void>(unused);
        result["receipts"].push_back(1U);
      }
      exported += request.at("events").size();
      result["counters"]["exported_events"] = exported;
    }
    return result;
  }
  Handler handler;
  Json last_request;
  std::uint64_t exported = 0U;
  unsigned calls = 0U;
};

TEST(TelemetryExporterTest, JsonlReopensAndDeduplicatesImmutableOriginals) {
  DraftDirectory directory;
  ex::JsonlTelemetryOptions options{.path = directory.path / "events.jsonl"};
  const std::array batch{event(), event(2U)};
  {
    ex::JsonlTelemetrySink sink(options);
    EXPECT_EQ(sink.descriptor().role, jy::TelemetrySinkRole::observer);
    EXPECT_EQ(sink.emit(batch).dispositions,
              (std::vector{jy::TelemetryDisposition::accepted,
                           jy::TelemetryDisposition::accepted}));
    EXPECT_EQ(sink.close().exported_events, 2U);
    EXPECT_EQ(sink.close().exported_events, 2U);
    EXPECT_THROW(sink.emit(batch), jy::NotReady);
  }
  const auto original = read_file(options.path);
  ex::JsonlTelemetrySink reopened(options);
  EXPECT_EQ(reopened.recovery().complete_records, 2U);
  EXPECT_EQ(reopened.recovery().truncated_tail_bytes, 0U);
  EXPECT_EQ(reopened.emit(batch).dispositions,
            (std::vector{jy::TelemetryDisposition::duplicate,
                         jy::TelemetryDisposition::duplicate}));
  EXPECT_EQ(reopened.close().exported_events, 2U);
  EXPECT_EQ(read_file(options.path), original);
}

TEST(TelemetryExporterTest, JsonlTruncatesOnlyAnIncompleteTail) {
  DraftDirectory directory;
  ex::JsonlTelemetryOptions options{.path = directory.path / "events.jsonl"};
  {
    ex::JsonlTelemetrySink sink(options);
    sink.emit(std::array{event()});
    sink.close();
  }
  const auto original = read_file(options.path);
  constexpr std::string_view tail = "{\"schema_version\":1,\"partial\":";
  write_file(options.path, original + std::string(tail));
  ex::JsonlTelemetrySink reopened(options);
  EXPECT_EQ(reopened.recovery().complete_records, 1U);
  EXPECT_EQ(reopened.recovery().truncated_tail_bytes, tail.size());
  reopened.close();
  EXPECT_EQ(read_file(options.path), original);
  write_file(options.path, original + std::string(tail) + "\n");
  EXPECT_THROW(ex::JsonlTelemetrySink{options}, jy::SerializationError);
  EXPECT_EQ(read_file(options.path), original + std::string(tail) + "\n");
}

TEST(TelemetryExporterTest,
     JsonlRejectsCompleteHashProjectionAndDuplicateKeyCorruption) {
  DraftDirectory directory;
  ex::JsonlTelemetryOptions options{.path = directory.path / "events.jsonl"};
  {
    ex::JsonlTelemetrySink sink(options);
    sink.emit(std::array{event()});
    sink.close();
  }
  const auto original = read_file(options.path);
  auto record = Json::parse(original);
  record["content_digest"] = std::string(64U, '0');
  write_file(options.path, record.dump() + "\n");
  EXPECT_THROW(ex::JsonlTelemetrySink{options}, jy::SerializationError);
  record = Json::parse(original);
  record["event"]["payload"]["body"] = "altered projection";
  write_file(options.path, record.dump() + "\n");
  EXPECT_THROW(ex::JsonlTelemetrySink{options}, jy::SerializationError);
  record = Json::parse(original);
  record["canonical_hex"] =
      record.at("canonical_hex").get<std::string>() + "00";
  write_file(options.path, record.dump() + "\n");
  EXPECT_THROW(ex::JsonlTelemetrySink{options}, jy::SerializationError);
  auto duplicate = original;
  duplicate.insert(1U, "\"schema_version\":1,");
  write_file(options.path, duplicate);
  EXPECT_THROW(ex::JsonlTelemetrySink{options}, jy::SerializationError);
}

TEST(TelemetryExporterTest,
     JsonlPreflightsConflictsAndCapacityBeforeAppending) {
  DraftDirectory directory;
  ex::JsonlTelemetryOptions options{.path = directory.path / "events.jsonl",
                                    .max_records = 1U};
  ex::JsonlTelemetrySink sink(options);
  EXPECT_THROW(sink.emit(std::array{event(), event(2U)}),
               jy::ResourceExhausted);
  EXPECT_TRUE(read_file(options.path).empty());
  sink.emit(std::array{event()});
  const auto original = read_file(options.path);
  auto changed = event();
  std::get<jy::TelemetryLog>(changed.payload).body = "conflict";
  EXPECT_THROW(sink.emit(std::array{changed}), jy::CompatibilityError);
  EXPECT_THROW(sink.emit(std::array{event(2U)}), jy::ResourceExhausted);
  EXPECT_EQ(read_file(options.path), original);
  EXPECT_EQ(sink.close().exported_events, 1U);
}

TEST(TelemetryExporterTest,
     JsonlRejectsLineFileAndRecoveryQuotaWithoutChangingCompleteInput) {
  DraftDirectory directory;
  ex::JsonlTelemetryOptions options{.path = directory.path / "small.jsonl",
                                    .max_line_bytes = 32U};
  {
    ex::JsonlTelemetrySink sink(options);
    EXPECT_THROW(sink.emit(std::array{event()}), jy::ResourceExhausted);
    EXPECT_TRUE(read_file(options.path).empty());
    sink.close();
  }
  options.max_line_bytes = 4U * 1024U * 1024U;
  options.max_file_bytes = 64U;
  {
    ex::JsonlTelemetrySink sink(options);
    EXPECT_THROW(sink.emit(std::array{event()}), jy::ResourceExhausted);
    sink.close();
  }
  options.max_file_bytes = 1ULL << 30U;
  {
    ex::JsonlTelemetrySink sink(options);
    sink.emit(std::array{event(), event(2U)});
    sink.close();
  }
  const auto original = read_file(options.path);
  options.max_records = 1U;
  EXPECT_THROW(ex::JsonlTelemetrySink{options}, jy::ResourceExhausted);
  EXPECT_EQ(read_file(options.path), original);
}

TEST(TelemetryExporterTest,
     ProjectionUsesNullForSymbolicNonfiniteValueAndCompleteDefinition) {
  auto value = event();
  value.payload = jy::TelemetryMetricBatch{{jy::TelemetryMetricPoint{
      .definition = {.metric_id = "validation.loss",
                     .version = 1U,
                     .unit = jy::TelemetryMetricUnit::dimensionless,
                     .aggregation = jy::TelemetryAggregation::mean,
                     .direction = jy::TelemetryMetricDirection::minimize},
      .step = {.kind = jy::TelemetryStepKind::global_step, .value = 7U},
      .observation = {.status = jy::TelemetryValueStatus::positive_infinity}}}};
  const auto json = Json::parse(ex::telemetry_projection_json(value));
  const auto& point = json.at("payload").at("points").at(0U);
  EXPECT_TRUE(point.at("value").is_null());
  EXPECT_EQ(point.at("status"), 3U);
  EXPECT_EQ(point.at("definition").at("metric_id"), "validation.loss");
  EXPECT_EQ(
      point.at("definition").at("definition_digest").get<std::string>().size(),
      64U);
}

TEST(TelemetryExporterTest,
     SidecarSendsOriginalProjectionAndStableBatchIdentity) {
  auto transport = std::make_shared<FakeTransport>();
  ex::OfficialTelemetrySink sink(transport);
  const std::array batch{event(), event(2U)};
  EXPECT_EQ(sink.emit(batch).dispositions.size(), 2U);
  const auto original_request = transport->last_request;
  EXPECT_EQ(original_request.at("backend"), "tensorboard");
  EXPECT_EQ(original_request.at("protocol_version"), 1U);
  EXPECT_EQ(original_request.at("events").at(0U).at("envelope"),
            Json::parse(ex::telemetry_projection_json(batch.front())));
  EXPECT_EQ(original_request.at("events").at(0U).at("content_digest"),
            jy::telemetry_content_digest(batch.front()).hex);
  sink.emit(batch);
  EXPECT_EQ(transport->last_request, original_request);
  EXPECT_EQ(sink.close().exported_events, 4U);
  const auto calls = transport->calls;
  sink.close();
  EXPECT_EQ(transport->calls, calls);
  EXPECT_THROW(sink.emit(batch), jy::NotReady);
}

TEST(TelemetryExporterTest,
     SidecarRejectsMalformedRepliesAndCommitsNoInvalidCounters) {
  auto transport = std::make_shared<FakeTransport>();
  ex::OfficialTelemetrySink sink(transport);
  const std::array batch{event()};
  transport->handler = [&](const Json& request) {
    auto result = transport->success(request);
    result["receipts"] = Json::array();
    result["counters"]["exported_events"] = 100U;
    return result.dump();
  };
  EXPECT_THROW(sink.emit(batch), jy::SerializationError);
  transport->exported = 0U;
  transport->handler = {};
  EXPECT_EQ(sink.emit(batch).dispositions.front(),
            jy::TelemetryDisposition::accepted);
  EXPECT_EQ(sink.flush().exported_events, 1U);
  transport->handler = [&](const Json& request) {
    auto result = transport->success(request);
    result["request_id"] = "wrong";
    return result.dump();
  };
  EXPECT_THROW(sink.emit(batch), jy::SerializationError);
  transport->handler = [](const Json&) { return "{\"protocol_version\":"; };
  EXPECT_THROW(sink.emit(batch), jy::SerializationError);
  transport->handler = [&](const Json& request) {
    auto result = transport->success(request);
    result["receipts"] = Json::array({4294967297ULL});
    return result.dump();
  };
  EXPECT_THROW(sink.emit(batch), jy::SerializationError);
}

TEST(TelemetryExporterTest,
     SidecarDistinguishesUnsupportedErrorsAndIllegalSampling) {
  auto transport = std::make_shared<FakeTransport>();
  ex::OfficialTelemetrySink sink(transport);
  const std::array batch{event()};
  transport->handler = [&](const Json& request) {
    auto reply = transport->success(request);
    reply["receipts"] = Json::array({4U});
    reply["counters"] = {{"exported_events", 0U},
                         {"sampled_out_events", 0U},
                         {"unsupported_events", 1U}};
    return reply.dump();
  };
  EXPECT_EQ(sink.emit(batch).dispositions.front(),
            jy::TelemetryDisposition::unsupported);
  transport->handler = [&](const Json& request) {
    auto reply = transport->success(request);
    reply["receipts"] = Json::array({3U});
    reply["counters"]["unsupported_events"] = 1U;
    return reply.dump();
  };
  EXPECT_THROW(sink.emit(batch), jy::CompatibilityError);
  transport->handler = [](const Json& request) {
    return Json{{"protocol_version", 1U},
                {"request_id", request.at("request_id")},
                {"error", "observer_failed"}}
        .dump();
  };
  EXPECT_THROW(sink.emit(batch), jy::BackendError);
}

TEST(TelemetryExporterTest,
     SidecarEnforcesBoundsCancellationAndDeadlineBeforeTransport) {
  auto transport = std::make_shared<FakeTransport>();
  ex::OfficialTelemetryOptions options;
  options.max_request_bytes = 32U;
  ex::OfficialTelemetrySink bounded(transport, options);
  const std::array batch{event()};
  EXPECT_THROW(bounded.emit(batch), jy::ResourceExhausted);
  EXPECT_EQ(transport->calls, 0U);
  ex::OfficialTelemetrySink sink(transport);
  std::stop_source source;
  source.request_stop();
  EXPECT_THROW(sink.emit(batch, source.get_token()), jy::Cancelled);
  EXPECT_THROW(sink.flush({}, std::chrono::steady_clock::now() -
                                  std::chrono::seconds(1)),
               jy::DeadlineExceeded);
  EXPECT_EQ(transport->calls, 0U);
}

// FakeTransport validates only our framing/receipt adapter. TensorBoard reader
// and actual collector acceptance are separate official-consumer gates.
}  // namespace
