// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/telemetry_native/tensorboard.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <stop_token>
#include <string>
#include <vector>

#ifndef _WIN32
#include <cerrno>
#include <csignal>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {
namespace jy = jyppx::liuding;
namespace ex = jy::experiment;

class NativeTensorBoardTest : public ::testing::Test {
 protected:
  void SetUp() override {
    static std::atomic<unsigned> sequence{0U};
    root = std::filesystem::temp_directory_path() /
           ("liuding-tensorboard-" +
            std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()) +
            "-" + std::to_string(sequence.fetch_add(1U)));
    ASSERT_TRUE(std::filesystem::create_directory(root));
  }
  void TearDown() override {
    std::error_code ignored;
    std::filesystem::remove_all(root, ignored);
  }
  ex::TensorBoardTelemetryOptions options() const {
    return {.directory = root / "events"};
  }
  jy::TelemetryEnvelope event(std::uint64_t sequence = 1U) const {
    jy::TelemetryMetricPoint point;
    point.definition.metric_id = "validation.loss";
    point.step.value = sequence;
    point.observation = jy::make_telemetry_metric_value(0.125);
    return {.identity = {.stream_id = "stream.1",
                         .event_id = "event." + std::to_string(sequence),
                         .idempotency_key = "key." + std::to_string(sequence),
                         .sequence = sequence},
            .time = {.wall_time_unix_ns = 1700000000123456789LL},
            .payload = jy::TelemetryMetricBatch{{point}}};
  }
  std::filesystem::path root;
};

TEST_F(NativeTensorBoardTest, ExclusiveDirectoryPreservesExistingContent) {
  const auto destination = root / "events";
  ASSERT_TRUE(std::filesystem::create_directory(destination));
  const auto sentinel = destination / "do-not-touch.txt";
  {
    std::ofstream output(sentinel);
    output << "keep";
  }
  EXPECT_THROW(static_cast<void>(ex::TensorBoardTelemetrySink(options())),
               jy::IoError);
  EXPECT_EQ(std::filesystem::file_size(sentinel), 4U);
  EXPECT_FALSE(
      std::filesystem::exists(destination / "events.out.tfevents.liuding"));
}

TEST_F(NativeTensorBoardTest, InvalidCapacitiesPrecedeFilesystemEffects) {
  auto invalid = options();
  invalid.max_record_bytes = 0U;
  EXPECT_THROW(static_cast<void>(ex::TensorBoardTelemetrySink(invalid)),
               jy::InvalidArgument);
  EXPECT_FALSE(std::filesystem::exists(invalid.directory));
  invalid = options();
  invalid.max_file_bytes = 1U;
  EXPECT_THROW(static_cast<void>(ex::TensorBoardTelemetrySink(invalid)),
               jy::ResourceExhausted);
  EXPECT_FALSE(std::filesystem::exists(invalid.directory));
}

TEST_F(NativeTensorBoardTest, AcceptedUnsupportedAndReplayHaveExactCounters) {
  ex::TensorBoardTelemetrySink observer(options());
  auto unsupported = event(2U);
  unsupported.payload =
      jy::TelemetryLog{jy::TelemetryLogSeverity::info, "explicit omission"};
  const std::array batch{event(), event(), unsupported, unsupported};
  const auto first = observer.emit(batch);
  EXPECT_EQ(first.dispositions,
            (std::vector{jy::TelemetryDisposition::accepted,
                         jy::TelemetryDisposition::duplicate,
                         jy::TelemetryDisposition::unsupported,
                         jy::TelemetryDisposition::unsupported}));
  const auto bytes = std::filesystem::file_size(observer.event_file());
  EXPECT_EQ(observer.emit(batch).dispositions,
            (std::vector{jy::TelemetryDisposition::duplicate,
                         jy::TelemetryDisposition::duplicate,
                         jy::TelemetryDisposition::unsupported,
                         jy::TelemetryDisposition::unsupported}));
  EXPECT_EQ(std::filesystem::file_size(observer.event_file()), bytes);
  const auto counters = observer.close();
  EXPECT_EQ(counters.exported_events, 1U);
  EXPECT_EQ(counters.unsupported_events, 1U);
  EXPECT_EQ(counters.sampled_out_events, 0U);
  EXPECT_EQ(observer.close().exported_events, 1U);
  EXPECT_EQ(observer.flush().unsupported_events, 1U);
  EXPECT_THROW(observer.emit(batch), jy::NotReady);
}

TEST_F(NativeTensorBoardTest, BatchValidationPrecedesEveryBatchWrite) {
  ex::TensorBoardTelemetrySink observer(options());
  const auto before = std::filesystem::file_size(observer.event_file());
  auto invalid = event(2U);
  invalid.identity.event_id.clear();
  const std::array batch{event(), invalid};
  EXPECT_THROW(observer.emit(batch), jy::InvalidArgument);
  EXPECT_EQ(std::filesystem::file_size(observer.event_file()), before);
  EXPECT_EQ(observer.flush().exported_events, 0U);
  EXPECT_EQ(observer.emit(std::array{event()}).dispositions.front(),
            jy::TelemetryDisposition::accepted);
}

TEST_F(NativeTensorBoardTest,
       CrossBatchOperationEventAndSequenceConflictsAreRejected) {
  ex::TensorBoardTelemetrySink observer(options());
  observer.emit(std::array{event()});
  const auto bytes = std::filesystem::file_size(observer.event_file());
  auto changed = event();
  changed.attributes["changed"] = "yes";
  EXPECT_THROW(observer.emit(std::array{changed}), jy::CompatibilityError);
  auto same_event = event(2U);
  same_event.identity.event_id = "event.1";
  EXPECT_THROW(observer.emit(std::array{same_event}), jy::CompatibilityError);
  auto same_sequence = event(2U);
  same_sequence.identity.sequence = 1U;
  EXPECT_THROW(observer.emit(std::array{same_sequence}),
               jy::CompatibilityError);
  EXPECT_EQ(std::filesystem::file_size(observer.event_file()), bytes);
  EXPECT_EQ(observer.flush().exported_events, 1U);
}

TEST_F(NativeTensorBoardTest,
       PreflightCancellationAndDeadlinePreserveReadyHandle) {
  ex::TensorBoardTelemetrySink observer(options());
  const auto bytes = std::filesystem::file_size(observer.event_file());
  std::stop_source stopped;
  stopped.request_stop();
  EXPECT_THROW(observer.emit(std::array{event()}, stopped.get_token()),
               jy::Cancelled);
  EXPECT_THROW(observer.close(stopped.get_token()), jy::Cancelled);
  EXPECT_THROW(observer.emit(std::array{event()}, {},
                             std::chrono::steady_clock::now() -
                                 std::chrono::milliseconds(1)),
               jy::DeadlineExceeded);
  EXPECT_EQ(std::filesystem::file_size(observer.event_file()), bytes);
  EXPECT_TRUE(observer.emit(std::span<const jy::TelemetryEnvelope>{})
                  .dispositions.empty());
  EXPECT_EQ(observer.emit(std::array{event()}).dispositions.front(),
            jy::TelemetryDisposition::accepted);
}

TEST_F(NativeTensorBoardTest,
       RecordBatchAndFileCapacitiesFailWithoutPartialOutput) {
  for (unsigned mode = 0U; mode < 3U; ++mode) {
    auto configured = options();
    configured.directory = root / ("capacity-" + std::to_string(mode));
    if (mode == 0U) configured.max_record_bytes = 128U;
    if (mode == 1U) configured.max_batch_encoded_bytes = 128U;
    if (mode == 2U) configured.max_file_bytes = 128U;
    ex::TensorBoardTelemetrySink observer(configured);
    const auto before = std::filesystem::file_size(observer.event_file());
    EXPECT_THROW(observer.emit(std::array{event()}), jy::ResourceExhausted);
    EXPECT_EQ(std::filesystem::file_size(observer.event_file()), before);
    EXPECT_EQ(observer.close().exported_events, 0U);
  }
}

TEST_F(NativeTensorBoardTest, BoundedReplayIndexRetainsOldRetries) {
  auto configured = options();
  configured.max_seen = 1U;
  ex::TensorBoardTelemetrySink observer(configured);
  observer.emit(std::array{event()});
  const auto before = std::filesystem::file_size(observer.event_file());
  EXPECT_THROW(observer.emit(std::array{event(2U)}), jy::ResourceExhausted);
  EXPECT_EQ(observer.emit(std::array{event()}).dispositions.front(),
            jy::TelemetryDisposition::duplicate);
  EXPECT_EQ(std::filesystem::file_size(observer.event_file()), before);
  configured.directory = root / "index-bytes";
  configured.max_seen_bytes = 1U;
  ex::TensorBoardTelemetrySink small(configured);
  EXPECT_THROW(small.emit(std::array{event()}), jy::ResourceExhausted);
  EXPECT_EQ(small.close().exported_events, 0U);
}

TEST_F(NativeTensorBoardTest,
       MissingAndUnsignedStepsHaveDeclaredUnsupportedPolicy) {
  ex::TensorBoardTelemetrySink observer(options());
  auto missing = event();
  auto& point =
      std::get<jy::TelemetryMetricBatch>(missing.payload).points.front();
  point.observation =
      jy::make_telemetry_metric_value(std::nullopt, "not measured");
  auto lifecycle = event(2U);
  lifecycle.payload =
      jy::TelemetryLifecycle{.name = "validation.finished", .metrics = {point}};
  auto large = event(3U);
  std::get<jy::TelemetryMetricBatch>(large.payload).points.front().step.value =
      static_cast<std::uint64_t>((std::numeric_limits<std::int64_t>::max)()) +
      1U;
  auto no_clock = event(4U);
  no_clock.time.wall_time_unix_ns.reset();
  EXPECT_EQ(observer.emit(std::array{missing, lifecycle, large, no_clock})
                .dispositions,
            (std::vector{jy::TelemetryDisposition::unsupported,
                         jy::TelemetryDisposition::accepted,
                         jy::TelemetryDisposition::unsupported,
                         jy::TelemetryDisposition::unsupported}));
  EXPECT_EQ(observer.close().unsupported_events, 3U);
}

#ifndef _WIN32
TEST_F(NativeTensorBoardTest, RealPartialFileWritePoisonsTheHandle) {
  ex::TensorBoardTelemetrySink observer(options());
  const auto bytes = std::filesystem::file_size(observer.event_file());
  const auto child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    struct rlimit limit{};
    if (::getrlimit(RLIMIT_FSIZE, &limit) != 0) ::_exit(10);
    limit.rlim_cur = static_cast<rlim_t>(bytes + 8U);
    if (::setrlimit(RLIMIT_FSIZE, &limit) != 0 ||
        std::signal(SIGXFSZ, SIG_IGN) == SIG_ERR)
      ::_exit(11);
    try {
      observer.emit(std::array{event()});
      ::_exit(12);
    } catch (const jy::IoError&) {
    } catch (...) {
      ::_exit(13);
    }
    try {
      observer.emit(std::array{event()});
      ::_exit(14);
    } catch (const jy::NotReady&) {
    } catch (...) {
      ::_exit(15);
    }
    try {
      observer.close();
      ::_exit(16);
    } catch (const jy::NotReady&) {
    } catch (...) {
      ::_exit(17);
    }
    ::_exit(0);
  }
  int status = 0;
  pid_t waited;
  do {
    waited = ::waitpid(child, &status, 0);
  } while (waited < 0 && errno == EINTR);
  ASSERT_EQ(waited, child);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
  EXPECT_EQ(std::filesystem::file_size(observer.event_file()), bytes + 8U);
}
#endif
}  // namespace
