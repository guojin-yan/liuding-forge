// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/training/callback_dispatcher.hpp>
#include <jyppx/liuding/training/callbacks/early_stopping_callback.hpp>
#include <jyppx/liuding/training/event_sink.hpp>
#include <jyppx/liuding/training/telemetry_adapter.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stop_token>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

class MemoryStateStore final : public jyppx::liuding::StateWriter,
                               public jyppx::liuding::StateReader {
 public:
  /** @brief Creates an empty store for one exact participant manifest. */
  explicit MemoryStateStore(jyppx::liuding::StateManifest manifest)
      : manifest_(std::move(manifest)) {}

  /** @brief Copies one declared field into owned test memory. */
  void write_blob(std::string_view name,
                  std::span<const std::byte> data) override {
    blobs_.emplace(std::string(name),
                   std::vector<std::byte>(data.begin(), data.end()));
  }

  /** @brief Returns the immutable participant manifest. */
  const jyppx::liuding::StateManifest& manifest() const noexcept override {
    return manifest_;
  }

  /** @brief Reports whether the named field was written. */
  bool contains(std::string_view name) const override {
    return blobs_.contains(std::string(name));
  }

  /** @brief Returns one stored field under the requested bound. */
  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t max_bytes) const override {
    const auto& data = blobs_.at(std::string(name));
    if (data.size() > max_bytes) {
      throw jyppx::liuding::ResourceExhausted("test state exceeds read limit");
    }
    return data;
  }

  /**
   * @brief Replaces one byte counted from a field's end.
   * @param name Existing field name.
   * @param trailing_offset Zero selects the final byte.
   * @param value Replacement byte.
   */
  void overwrite_from_end(std::string_view name, std::size_t trailing_offset,
                          std::byte value) {
    auto& data = blobs_.at(std::string(name));
    data.at(data.size() - 1U - trailing_offset) = value;
  }

  std::vector<std::byte>& mutable_blob(std::string_view name) {
    return blobs_.at(std::string(name));
  }

 private:
  jyppx::liuding::StateManifest
      manifest_;  ///< Participant-owned field contract.
  std::map<std::string, std::vector<std::byte>>
      blobs_;  ///< Complete blobs keyed by field name.
};

class RecordingSink final : public jyppx::liuding::training::EventSink {
 public:
  /**
   * @brief Creates a deterministic sink with optional injected failures.
   * @param sink_id Stable outbox ordering identity.
   * @param failures Number of publish attempts rejected before acceptance.
   */
  explicit RecordingSink(std::string sink_id, int failures = 0,
                         int close_failures = 0)
      : descriptor_{.schema_version = 1,
                    .type_id = "test.recording-sink/v1",
                    .sink_id = std::move(sink_id),
                    .destination_id = "test.memory",
                    .delivery_guarantee = jyppx::liuding::training::
                        EventDeliveryGuarantee::at_least_once},
        failures_remaining_(failures),
        close_failures_remaining_(close_failures) {}

  /** @brief Returns immutable test sink identity. */
  const jyppx::liuding::training::EventSinkDescriptor& descriptor()
      const noexcept override {
    return descriptor_;
  }

  /** @brief Records the event unless an injected failure remains. */
  void publish(const jyppx::liuding::training::EventEnvelope& envelope,
               std::stop_token stop) override {
    ++publish_attempts_;
    if (stop.stop_requested()) {
      throw jyppx::liuding::Cancelled("test sink cancelled");
    }
    if (failures_remaining_ > 0) {
      --failures_remaining_;
      throw jyppx::liuding::IoError("injected sink failure");
    }
    accepted_indices_.push_back(envelope.event.event_index);
  }

  /** @brief Records a flush operation. */
  void flush(std::stop_token stop) override {
    if (stop.stop_requested()) {
      throw jyppx::liuding::Cancelled("test flush cancelled");
    }
    if (closed_) {
      throw jyppx::liuding::InternalError("flush called after test sink close");
    }
    ++flush_calls_;
  }

  /** @brief Idempotently records explicit close. */
  void close(std::stop_token stop) override {
    if (closed_) {
      return;
    }
    if (close_failures_remaining_ > 0) {
      --close_failures_remaining_;
      throw jyppx::liuding::IoError("injected sink close failure");
    }
    closed_ = true;
    ++close_calls_;
  }

  /** @brief Returns attempted publish count including failures. */
  int publish_attempts() const noexcept { return publish_attempts_; }

  /** @brief Returns successfully accepted event indices. */
  const std::vector<std::uint64_t>& accepted_indices() const noexcept {
    return accepted_indices_;
  }

  /** @brief Returns close transition count. */
  int close_calls() const noexcept { return close_calls_; }

  /** @brief Returns successful flush count. */
  int flush_calls() const noexcept { return flush_calls_; }

 private:
  jyppx::liuding::training::EventSinkDescriptor
      descriptor_;              ///< Stable sink topology.
  int failures_remaining_ = 0;  ///< Rejected attempts before first success.
  int close_failures_remaining_ = 0;  ///< Rejected close attempts.
  int publish_attempts_ = 0;          ///< All publish invocations.
  int flush_calls_ = 0;               ///< Successful flush invocations.
  int close_calls_ = 0;               ///< Successful close transitions.
  bool closed_ = false;               ///< Whether explicit close completed.
  std::vector<std::uint64_t> accepted_indices_;  ///< Accepted event order.
};

jyppx::liuding::training::CallbackEvent run_started() {
  return {.schema_version = 1,
          .kind = jyppx::liuding::training::CallbackEventKind::run_started,
          .phase = jyppx::liuding::training::CallbackPhase::preparing,
          .event_index = 1,
          .run_id = "run.test",
          .trace_id = "trace.test",
          .epoch = 0,
          .global_step = 0,
          .metrics = {}};
}

jyppx::liuding::training::CallbackEvent epoch_finished(std::uint64_t index,
                                                       double loss = 1.0) {
  return {.schema_version = 1,
          .kind = jyppx::liuding::training::CallbackEventKind::epoch_finished,
          .phase = jyppx::liuding::training::CallbackPhase::training,
          .event_index = index,
          .run_id = "run.test",
          .trace_id = "trace.test",
          .epoch = index - 1U,
          .global_step = index - 1U,
          .metrics = {{"validation.loss", loss}}};
}

TEST(EventOutboxTest, PublishesOnlyAfterCallbackTransactionCommits) {
  auto sink = std::make_shared<RecordingSink>("sink.recording");
  jyppx::liuding::training::EventOutbox outbox({sink});
  jyppx::liuding::training::CallbackDispatcher dispatcher;

  const auto result = dispatcher.dispatch(run_started(), outbox);
  EXPECT_TRUE(result.commands.empty());
  EXPECT_TRUE(sink->accepted_indices().empty());
  EXPECT_EQ(outbox.pending_events(), 1U);
  EXPECT_EQ(dispatcher.last_event_index(), 1U);

  EXPECT_TRUE(outbox.drain_one());
  EXPECT_EQ(sink->accepted_indices(), std::vector<std::uint64_t>{1});
  EXPECT_EQ(outbox.pending_events(), 0U);
  EXPECT_EQ(outbox.last_delivered_event_index(), 1U);
}

TEST(EventOutboxTest, QueueExhaustionRollsBackStatefulCallbacksAndCursor) {
  auto sink = std::make_shared<RecordingSink>("sink.recording");
  jyppx::liuding::training::EventOutbox outbox(
      {sink},
      jyppx::liuding::training::EventOutboxOptions{.max_pending_events = 1});
  auto early = std::make_shared<
      jyppx::liuding::training::callbacks::EarlyStoppingCallback>();
  jyppx::liuding::training::CallbackDispatcher dispatcher({early});
  dispatcher.dispatch(run_started(), outbox);

  EXPECT_THROW(dispatcher.dispatch(epoch_finished(2), outbox),
               jyppx::liuding::ResourceExhausted);
  EXPECT_FALSE(early->has_best_value());
  EXPECT_EQ(dispatcher.last_event_index(), 1U);
  EXPECT_EQ(outbox.last_enqueued_event_index(), 1U);
  EXPECT_EQ(outbox.pending_events(), 1U);
}

TEST(EventOutboxTest, RetriesOnlyTheFirstUnacknowledgedSink) {
  auto first = std::make_shared<RecordingSink>("sink.a");
  auto second = std::make_shared<RecordingSink>("sink.b", 1);
  jyppx::liuding::training::EventOutbox outbox({second, first});
  outbox.enqueue(run_started(), {});

  EXPECT_THROW(outbox.drain_one(), jyppx::liuding::IoError);
  EXPECT_EQ(first->accepted_indices(), std::vector<std::uint64_t>{1});
  EXPECT_TRUE(second->accepted_indices().empty());
  EXPECT_EQ(outbox.pending_events(), 1U);

  EXPECT_TRUE(outbox.drain_one());
  EXPECT_EQ(first->publish_attempts(), 1);
  EXPECT_EQ(second->publish_attempts(), 2);
  EXPECT_EQ(second->accepted_indices(), std::vector<std::uint64_t>{1});
  EXPECT_EQ(outbox.pending_events(), 0U);
}

TEST(EventOutboxTest, CheckpointRestoresPendingPerSinkDeliveryCursor) {
  auto source_first = std::make_shared<RecordingSink>("sink.a");
  auto source_second = std::make_shared<RecordingSink>("sink.b", 1);
  jyppx::liuding::training::EventOutbox source({source_second, source_first});
  source.enqueue(run_started(), {});
  EXPECT_THROW(source.drain_one(), jyppx::liuding::IoError);

  MemoryStateStore state(source.describe_state());
  source.save_state(state);

  auto restored_first = std::make_shared<RecordingSink>("sink.a");
  auto restored_second = std::make_shared<RecordingSink>("sink.b");
  jyppx::liuding::training::EventOutbox restored(
      {restored_second, restored_first});
  restored.load_state(state, jyppx::liuding::RestorePolicy::exact_only);

  EXPECT_EQ(restored.pending_events(), 1U);
  EXPECT_TRUE(restored.drain_one());
  EXPECT_TRUE(restored_first->accepted_indices().empty());
  EXPECT_EQ(restored_second->accepted_indices(), std::vector<std::uint64_t>{1});
  EXPECT_EQ(restored.last_delivered_event_index(), 1U);
}

TEST(EventOutboxTest, CheckpointRestoresDeliveredPrefixAndPendingSuffix) {
  auto source_sink = std::make_shared<RecordingSink>("sink.recording");
  jyppx::liuding::training::EventOutbox source({source_sink});
  source.enqueue(run_started(), {});
  ASSERT_TRUE(source.drain_one());
  source.enqueue(epoch_finished(2), {});

  MemoryStateStore state(source.describe_state());
  source.save_state(state);

  auto restored_sink = std::make_shared<RecordingSink>("sink.recording");
  jyppx::liuding::training::EventOutbox restored({restored_sink});
  restored.load_state(state, jyppx::liuding::RestorePolicy::exact_only);

  EXPECT_EQ(restored.last_delivered_event_index(), 1U);
  EXPECT_EQ(restored.last_enqueued_event_index(), 2U);
  ASSERT_EQ(restored.pending_events(), 1U);
  EXPECT_TRUE(restored.drain_one());
  EXPECT_EQ(restored_sink->accepted_indices(), std::vector<std::uint64_t>{2});
  EXPECT_EQ(restored.last_delivered_event_index(), 2U);
}

TEST(EventOutboxTest, CorruptRestoreLeavesLiveQueueUnchanged) {
  auto sink = std::make_shared<RecordingSink>("sink.recording");
  jyppx::liuding::training::EventOutbox source({sink});
  source.enqueue(run_started(), {});
  MemoryStateStore state(source.describe_state());
  source.save_state(state);
  state.overwrite_from_end("control", 0, std::byte{2});

  auto target_sink = std::make_shared<RecordingSink>("sink.recording");
  jyppx::liuding::training::EventOutbox target({target_sink});
  target.enqueue(run_started(), {});
  EXPECT_THROW(
      target.load_state(state, jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::Error);
  EXPECT_EQ(target.pending_events(), 1U);
  EXPECT_EQ(target.last_enqueued_event_index(), 1U);
}

TEST(EventOutboxTest, RejectsInvalidControlFlagsWithoutMutation) {
  auto source_sink = std::make_shared<RecordingSink>("sink.recording");
  jyppx::liuding::training::EventOutbox source({source_sink});
  source.enqueue(run_started(), {});
  MemoryStateStore state(source.describe_state());
  source.save_state(state);

  constexpr std::string_view kControlMagic = "JYPPX-EVENT-OUTBOX-CONTROL-V2";
  auto& control = state.mutable_blob("control");
  control.at(sizeof(std::uint64_t) + kControlMagic.size()) = std::byte{2};

  auto target_sink = std::make_shared<RecordingSink>("sink.recording");
  jyppx::liuding::training::EventOutbox target({target_sink});
  target.enqueue(run_started(), {});
  EXPECT_THROW(
      target.load_state(state, jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::SerializationError);
  EXPECT_EQ(target.pending_events(), 1U);
  EXPECT_EQ(target.last_enqueued_event_index(), 1U);
  EXPECT_EQ(target.last_delivered_event_index(), 0U);
}

TEST(EventOutboxTest, RejectsTruncatedControlAndUnknownPolicy) {
  auto sink = std::make_shared<RecordingSink>("sink.recording");
  jyppx::liuding::training::EventOutbox source({sink});
  source.enqueue(run_started(), {});
  MemoryStateStore state(source.describe_state());
  source.save_state(state);

  auto target_sink = std::make_shared<RecordingSink>("sink.recording");
  jyppx::liuding::training::EventOutbox target({target_sink});
  EXPECT_THROW(
      target.load_state(state, static_cast<jyppx::liuding::RestorePolicy>(255)),
      jyppx::liuding::CompatibilityError);
  EXPECT_EQ(target.pending_events(), 0U);

  auto& control = state.mutable_blob("control");
  control.pop_back();
  EXPECT_THROW(
      target.load_state(state, jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::SerializationError);
  EXPECT_EQ(target.pending_events(), 0U);
  EXPECT_EQ(target.last_enqueued_event_index(), 0U);
}

TEST(EventOutboxTest, CancellationAndCloseHaveExplicitBoundaries) {
  auto sink = std::make_shared<RecordingSink>("sink.recording");
  jyppx::liuding::training::EventOutbox outbox({sink});
  outbox.enqueue(run_started(), {});
  std::stop_source cancellation;
  cancellation.request_stop();

  EXPECT_THROW(outbox.drain(cancellation.get_token()),
               jyppx::liuding::Cancelled);
  EXPECT_EQ(outbox.pending_events(), 1U);
  outbox.close();
  outbox.close();
  EXPECT_TRUE(outbox.closed());
  EXPECT_EQ(sink->close_calls(), 1);
  EXPECT_THROW(outbox.enqueue(epoch_finished(2), {}), jyppx::liuding::NotReady);
}

TEST(EventOutboxTest, CloseRetrySkipsAlreadyClosedSinks) {
  auto first = std::make_shared<RecordingSink>("sink.a");
  auto second = std::make_shared<RecordingSink>("sink.b", 0, 1);
  jyppx::liuding::training::EventOutbox outbox({second, first});

  EXPECT_THROW(outbox.close(), jyppx::liuding::IoError);
  EXPECT_EQ(first->close_calls(), 1);
  EXPECT_EQ(first->flush_calls(), 1);
  EXPECT_EQ(second->close_calls(), 0);
  EXPECT_THROW(
      [&] {
        MemoryStateStore state(outbox.describe_state());
        outbox.save_state(state);
      }(),
      jyppx::liuding::NotReady);

  outbox.close();
  EXPECT_TRUE(outbox.closed());
  EXPECT_EQ(first->close_calls(), 1);
  EXPECT_EQ(first->flush_calls(), 1);
  EXPECT_EQ(second->close_calls(), 1);
  EXPECT_EQ(second->flush_calls(), 1);
}

TEST(EventOutboxTest, RejectsInconsistentCallbackResultWithoutMutation) {
  auto sink = std::make_shared<RecordingSink>("sink.recording");
  jyppx::liuding::training::EventOutbox outbox({sink});
  jyppx::liuding::training::CallbackDispatchResult invalid{
      .commands = {{.kind = jyppx::liuding::training::CallbackCommandKind::stop,
                    .callback_id = "callback.stop",
                    .reason = "done"}},
      .effective_command = std::nullopt};

  EXPECT_THROW(outbox.enqueue(run_started(), invalid),
               jyppx::liuding::CompatibilityError);
  EXPECT_EQ(outbox.pending_events(), 0U);
  EXPECT_EQ(outbox.last_enqueued_event_index(), 0U);
}

TEST(EventSinkTest, ConsoleWritesCanonicalMachineReadableLine) {
  std::ostringstream output;
  auto sink =
      std::make_shared<jyppx::liuding::training::ConsoleEventSink>(output);
  jyppx::liuding::training::EventOutbox outbox({sink});
  outbox.enqueue(run_started(), {});
  outbox.close();

  const auto line = output.str();
  EXPECT_NE(line.find("\"idempotency_key\":\"run.test:1\""), std::string::npos);
  EXPECT_NE(line.find("\"kind\":\"run_started\""), std::string::npos);
  EXPECT_EQ(std::count(line.begin(), line.end(), '\n'), 1);
}

TEST(EventSinkTest, JsonlAppendsWithoutLeavingTestArtifacts) {
  const auto path = std::filesystem::temp_directory_path() /
                    "jyppx-event-sink-contract.jsonl";
  std::error_code ignored;
  std::filesystem::remove(path, ignored);
  {
    auto sink = std::make_shared<jyppx::liuding::training::JsonlEventSink>(
        jyppx::liuding::training::JsonlEventSinkOptions{.path = path});
    jyppx::liuding::training::EventOutbox outbox({sink});
    outbox.enqueue(run_started(), {});
    outbox.close();
  }
  std::ifstream input(path, std::ios::binary);
  const std::string contents((std::istreambuf_iterator<char>(input)),
                             std::istreambuf_iterator<char>());
  input.close();
  std::filesystem::remove(path, ignored);

  EXPECT_NE(contents.find("\"run_id\":\"run.test\""), std::string::npos);
  EXPECT_EQ(std::count(contents.begin(), contents.end(), '\n'), 1);
}

class NeutralRecordingSink final : public jyppx::liuding::TelemetrySink {
 public:
  explicit NeutralRecordingSink(std::string sink_id = "sink.neutral",
                                jyppx::liuding::TelemetrySinkRole role =
                                    jyppx::liuding::TelemetrySinkRole::evidence)
      : descriptor_{.type_id = "test.neutral-sink/v2",
                    .sink_id = std::move(sink_id),
                    .destination_id = "test.memory",
                    .role = role} {}

  const jyppx::liuding::TelemetrySinkDescriptor& descriptor()
      const noexcept override {
    return descriptor_;
  }
  jyppx::liuding::TelemetryEmitResult emit(
      std::span<const jyppx::liuding::TelemetryEnvelope> envelopes,
      std::stop_token stop,
      jyppx::liuding::TelemetryDeadline deadline) override {
    jyppx::liuding::check_telemetry_operation(stop, deadline);
    jyppx::liuding::validate_telemetry_batch(envelopes);
    for (const auto& envelope : envelopes) {
      attempted.push_back(jyppx::liuding::encode_telemetry_envelope(envelope));
    }
    if (failures > 0) {
      --failures;
      throw jyppx::liuding::IoError("injected neutral delivery failure");
    }
    if (incomplete_receipt) return {};
    accepted.insert(accepted.end(), envelopes.begin(), envelopes.end());
    return {std::vector<jyppx::liuding::TelemetryDisposition>(envelopes.size(),
                                                              disposition)};
  }
  jyppx::liuding::TelemetryFlushResult flush(
      std::stop_token stop,
      jyppx::liuding::TelemetryDeadline deadline) override {
    jyppx::liuding::check_telemetry_operation(stop, deadline);
    ++flush_calls;
    return {.exported_events = static_cast<std::uint64_t>(accepted.size())};
  }
  jyppx::liuding::TelemetryFlushResult close(
      std::stop_token stop,
      jyppx::liuding::TelemetryDeadline deadline) override {
    jyppx::liuding::check_telemetry_operation(stop, deadline);
    if (!closed) {
      ++close_calls;
      closed = true;
    }
    return {.exported_events = static_cast<std::uint64_t>(accepted.size())};
  }

  int failures = 0;
  bool incomplete_receipt = false;
  jyppx::liuding::TelemetryDisposition disposition =
      jyppx::liuding::TelemetryDisposition::accepted;
  std::vector<std::vector<std::byte>> attempted;
  std::vector<jyppx::liuding::TelemetryEnvelope> accepted;
  int flush_calls = 0;
  int close_calls = 0;

 private:
  jyppx::liuding::TelemetrySinkDescriptor descriptor_;
  bool closed = false;
};

jyppx::liuding::training::TrainingTelemetryAdapterOptions typed_options() {
  jyppx::liuding::training::TrainingTelemetryAdapterOptions options;
  options.project_id = "project.test";
  options.attempt_id = "attempt.test";
  options.metrics["validation.loss"] = {
      .definition = {.metric_id = "training.validation.loss",
                     .unit = jyppx::liuding::TelemetryMetricUnit::dimensionless,
                     .aggregation = jyppx::liuding::TelemetryAggregation::mean,
                     .direction =
                         jyppx::liuding::TelemetryMetricDirection::minimize,
                     .context_keys = {"split"}},
      .context = {{"split", "validation"}}};
  return options;
}

std::vector<std::byte> fixed_hex(std::string_view text) {
  const auto nibble = [](char c) -> unsigned int {
    return c <= '9' ? static_cast<unsigned int>(c - '0')
                    : static_cast<unsigned int>(c - 'a' + 10);
  };
  std::vector<std::byte> result;
  result.reserve(text.size() / 2U);
  for (std::size_t index = 0U; index < text.size(); index += 2U) {
    result.push_back(static_cast<std::byte>((nibble(text[index]) << 4U) |
                                            nibble(text[index + 1U])));
  }
  return result;
}

// Frozen byte-level schema-1 layout from the released EventOutbox writer.
// These blobs are independent of the current writer and exercise load_state.
MemoryStateStore frozen_v1_state(bool terminal = false) {
  jyppx::liuding::StateManifest manifest{
      .participant_id = "events.outbox",
      .schema_version = 1U,
      .restore_level = jyppx::liuding::RestoreLevel::exact,
      .fields = {{.name = "topology",
                  .media_type =
                      "application/vnd.jyppx.event-outbox-topology-v1",
                  .max_bytes = 1048576U},
                 {.name = "control",
                  .media_type = "application/vnd.jyppx.event-outbox-control-v1",
                  .max_bytes = 68157440U}},
      .degradation_reasons = {}};
  MemoryStateStore state(std::move(manifest));
  const auto topology = fixed_hex(
      "1e000000000000004a595050582d4556454e542d4f5554424f582d544f504f4c4f47592d"
      "56310d000000000000006576"
      "656e74732e6f7574626f7840000000000000000010000000000000001000000000000000"
      "040000000000000010000000"
      "000000000010000000000000000004000000000000100000000000000010040000000002"
      "000000000000000100000016"
      "00000000000000746573742e7265636f7264696e672d73696e6b2f763106000000000000"
      "0073696e6b2e610b00000000"
      "000000746573742e6d656d6f727901010000001600000000000000746573742e7265636f"
      "7264696e672d73696e6b2f76"
      "31060000000000000073696e6b2e620b00000000000000746573742e6d656d6f727901");
  const auto pending =
      terminal ? fixed_hex(
                     "1d000000000000004a595050582d4556454e542d4f5554424f582d434"
                     "f4e54524f4c2d56310101080000000000000072"
                     "756e2e746573740200000000000000010000000000000001000000000"
                     "000000100000000000000000000000000000000"
                     "000000000000006500000000000000010000000000000000000000000"
                     "000006500000000000000550000000000000001"
                     "0000000100000002050200000000000000080000000000000072756e2"
                     "e746573740a0000000000000074726163652e74"
                     "657374010000000000000001000000000000000000000000000000000"
                     "000000000000000")
               : fixed_hex(
                     "1d000000000000004a595050582d4556454e542d4f5554424f582d434"
                     "f4e54524f4c2d56310100080000000000000072"
                     "756e2e746573740200000000000000010000000000000001000000000"
                     "000000100000000000000000000000000000000"
                     "00000000000000c800000000000000010000000000000001000000000"
                     "00000c800000000000000b80000000000000001"
                     "0000000100000006020200000000000000080000000000000072756e2"
                     "e746573740a0000000000000074726163652e74"
                     "6573740100000000000000010000000000000001000000000000000f0"
                     "000000000000076616c69646174696f6e2e6c6f"
                     "7373000000000000f43f0100000000000000030d00000000000000636"
                     "16c6c6261636b2e73746f700400000000000000"
                     "646f6e6501030d0000000000000063616c6c6261636b2e73746f70040"
                     "0000000000000646f6e65");
  state.write_blob("topology", topology);
  state.write_blob("control", pending);
  return state;
}

TEST(EventOutboxTelemetryTest,
     StagesBeforeCallbackCommitAndRestoresExactRetryBytes) {
  auto first = std::make_shared<NeutralRecordingSink>("sink.a");
  auto second = std::make_shared<NeutralRecordingSink>("sink.b");
  second->failures = 1;
  auto options = typed_options();
  jyppx::liuding::training::EventOutbox source({second, first}, options);
  jyppx::liuding::training::CallbackDispatcher dispatcher;
  dispatcher.dispatch(run_started(), source);
  EXPECT_TRUE(first->accepted.empty());
  EXPECT_EQ(source.pending_events(), 1U);
  EXPECT_EQ(dispatcher.last_event_index(), 1U);
  EXPECT_THROW(source.drain(), jyppx::liuding::IoError);
  ASSERT_EQ(first->accepted.size(), 1U);
  ASSERT_EQ(second->attempted.size(), 1U);
  const auto original = second->attempted.front();
  const auto stream = std::string(source.telemetry_stream_id());
  const auto original_time = first->accepted.front().time;
  EXPECT_TRUE(original_time.wall_time_unix_ns.has_value());
  EXPECT_TRUE(original_time.monotonic_time_ns.has_value());
  EXPECT_FALSE(original_time.clock_id.empty());

  MemoryStateStore state(source.describe_state());
  EXPECT_EQ(state.manifest().schema_version, 2U);
  source.save_state(state);
  auto restored_first = std::make_shared<NeutralRecordingSink>("sink.a");
  auto restored_second = std::make_shared<NeutralRecordingSink>("sink.b");
  jyppx::liuding::training::EventOutbox restored(
      {restored_first, restored_second}, options);
  EXPECT_NE(restored.telemetry_stream_id(), stream);
  restored.load_state(state, jyppx::liuding::RestorePolicy::exact_only);
  EXPECT_EQ(restored.telemetry_stream_id(), stream);
  restored.drain();
  EXPECT_TRUE(restored_first->attempted.empty());
  ASSERT_EQ(restored_second->attempted.size(), 1U);
  EXPECT_EQ(restored_second->attempted.front(), original);

  // The caller's mutable binding object cannot alter already-frozen semantics.
  options.metrics.at("validation.loss").definition.unit =
      jyppx::liuding::TelemetryMetricUnit::bytes;
  restored.enqueue(epoch_finished(2U), {});
  restored.drain();
  ASSERT_EQ(restored_second->accepted.size(), 2U);
  const auto& next = restored_second->accepted.back();
  EXPECT_EQ(next.identity.stream_id, stream);
  EXPECT_EQ(next.identity.sequence, 2U);
  EXPECT_NE(next.time.clock_id, original_time.clock_id);
  const auto& metric =
      std::get<jyppx::liuding::TelemetryLifecycle>(next.payload).metrics.at(0U);
  EXPECT_EQ(metric.definition.unit,
            jyppx::liuding::TelemetryMetricUnit::dimensionless);
  EXPECT_EQ(metric.context.at("split"), "validation");
}

TEST(EventOutboxTelemetryTest,
     MissingBindingRollsBackStatefulCallbackAndBothCursors) {
  auto sink = std::make_shared<NeutralRecordingSink>();
  auto options = typed_options();
  options.metrics.clear();
  jyppx::liuding::training::EventOutbox outbox({sink}, options);
  auto early = std::make_shared<
      jyppx::liuding::training::callbacks::EarlyStoppingCallback>();
  jyppx::liuding::training::CallbackDispatcher dispatcher({early});
  dispatcher.dispatch(run_started(), outbox);
  outbox.drain();
  EXPECT_THROW(dispatcher.dispatch(epoch_finished(2U), outbox),
               jyppx::liuding::CompatibilityError);
  EXPECT_FALSE(early->has_best_value());
  EXPECT_EQ(dispatcher.last_event_index(), 1U);
  EXPECT_EQ(outbox.last_enqueued_event_index(), 1U);
  EXPECT_EQ(outbox.last_delivered_event_index(), 1U);
  EXPECT_EQ(outbox.pending_events(), 0U);
}

TEST(EventOutboxTelemetryTest,
     FullyDeliveredCheckpointStillRestoresProducerStreamAndNextSequence) {
  auto source_sink = std::make_shared<NeutralRecordingSink>();
  const auto options = typed_options();
  jyppx::liuding::training::EventOutbox source({source_sink}, options);
  source.enqueue(run_started(), {});
  source.drain();
  source.enqueue(epoch_finished(2U), {});
  source.drain();
  const auto stream = std::string(source.telemetry_stream_id());
  MemoryStateStore checkpoint(source.describe_state());
  source.save_state(checkpoint);
  auto target_sink = std::make_shared<NeutralRecordingSink>();
  jyppx::liuding::training::EventOutbox target({target_sink}, options);
  target.load_state(checkpoint, jyppx::liuding::RestorePolicy::exact_only);
  EXPECT_EQ(target.pending_events(), 0U);
  EXPECT_EQ(target.last_delivered_event_index(), 2U);
  EXPECT_EQ(target.telemetry_stream_id(), stream);
  target.enqueue(epoch_finished(3U), {});
  target.drain();
  ASSERT_EQ(target_sink->accepted.size(), 1U);
  EXPECT_EQ(target_sink->accepted.front().identity.sequence, 3U);
  EXPECT_EQ(target_sink->accepted.front().identity.stream_id, stream);
}

TEST(EventOutboxTelemetryTest,
     RejectsIncompleteAndMissingRequiredEvidenceReceipts) {
  auto sink = std::make_shared<NeutralRecordingSink>();
  jyppx::liuding::training::EventOutbox outbox({sink}, typed_options());
  outbox.enqueue(run_started(), {});
  sink->incomplete_receipt = true;
  EXPECT_THROW(outbox.drain(), jyppx::liuding::CompatibilityError);
  sink->incomplete_receipt = false;
  sink->disposition = jyppx::liuding::TelemetryDisposition::unsupported;
  EXPECT_THROW(outbox.drain(), jyppx::liuding::CompatibilityError);
  sink->disposition = jyppx::liuding::TelemetryDisposition::sampled_out;
  EXPECT_THROW(outbox.drain(), jyppx::liuding::CompatibilityError);
  EXPECT_EQ(outbox.last_delivered_event_index(), 0U);
  EXPECT_EQ(outbox.pending_events(), 1U);
  ASSERT_EQ(sink->attempted.size(), 3U);
  EXPECT_EQ(sink->attempted[0], sink->attempted[1]);
  EXPECT_EQ(sink->attempted[0], sink->attempted[2]);
  sink->disposition = jyppx::liuding::TelemetryDisposition::duplicate;
  outbox.drain();
  EXPECT_EQ(outbox.last_delivered_event_index(), 1U);
  EXPECT_EQ(outbox.pending_bytes(), 0U);
}

TEST(EventOutboxTelemetryTest,
     ObserverReportsUnsupportedWithoutClaimingEvidence) {
  auto sink = std::make_shared<NeutralRecordingSink>(
      "sink.observer", jyppx::liuding::TelemetrySinkRole::observer);
  sink->disposition = jyppx::liuding::TelemetryDisposition::unsupported;
  jyppx::liuding::training::EventOutbox outbox({sink}, typed_options());
  outbox.enqueue(run_started(), {});
  EXPECT_TRUE(outbox.drain_one());
  EXPECT_EQ(sink->attempted.size(), 1U);
  EXPECT_EQ(outbox.last_delivered_event_index(), 1U);
}

TEST(EventOutboxTelemetryTest,
     CancellationAndAbsoluteDeadlineRetainTheStagedNode) {
  auto sink = std::make_shared<NeutralRecordingSink>();
  jyppx::liuding::training::EventOutbox outbox({sink}, typed_options());
  outbox.enqueue(run_started(), {});
  const auto charge = outbox.pending_bytes();
  const auto expired =
      std::chrono::steady_clock::now() - std::chrono::seconds(1);
  EXPECT_THROW(outbox.drain({}, expired), jyppx::liuding::DeadlineExceeded);
  std::stop_source stop;
  stop.request_stop();
  EXPECT_THROW(outbox.drain(stop.get_token()), jyppx::liuding::Cancelled);
  EXPECT_TRUE(sink->attempted.empty());
  EXPECT_EQ(outbox.pending_bytes(), charge);
  outbox.close();
  outbox.close();
  EXPECT_TRUE(outbox.closed());
  EXPECT_EQ(sink->close_calls, 1);
}

TEST(EventOutboxTelemetryTest,
     FrozenTopologyAndExplicitStreamMismatchLeaveLiveQueueUntouched) {
  auto sink = std::make_shared<NeutralRecordingSink>();
  const auto options = typed_options();
  jyppx::liuding::training::EventOutbox source({sink}, options);
  source.enqueue(run_started(), {});
  MemoryStateStore state(source.describe_state());
  source.save_state(state);

  auto changed = options;
  changed.metrics.at("validation.loss").definition.aggregation =
      jyppx::liuding::TelemetryAggregation::last;
  auto target_sink = std::make_shared<NeutralRecordingSink>();
  jyppx::liuding::training::EventOutbox target({target_sink}, changed);
  target.enqueue(run_started(), {});
  const auto previous_charge = target.pending_bytes();
  const auto previous_stream = std::string(target.telemetry_stream_id());
  EXPECT_THROW(
      target.load_state(state, jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::CompatibilityError);
  EXPECT_EQ(target.pending_bytes(), previous_charge);
  EXPECT_EQ(target.telemetry_stream_id(), previous_stream);

  auto explicit_options = options;
  explicit_options.stream_id = "different.stream";
  jyppx::liuding::training::EventOutbox explicit_target({target_sink},
                                                        explicit_options);
  EXPECT_THROW(explicit_target.load_state(
                   state, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::CompatibilityError);
  EXPECT_EQ(explicit_target.last_enqueued_event_index(), 0U);
}

TEST(EventOutboxTelemetryTest,
     CombinedSnapshotAndCanonicalBudgetFailsAtomically) {
  auto sink = std::make_shared<NeutralRecordingSink>();
  jyppx::liuding::training::EventOutbox probe({sink}, typed_options());
  probe.enqueue(run_started(), {});
  jyppx::liuding::training::EventOutboxOptions bounds;
  bounds.max_pending_bytes = probe.pending_bytes() - 1U;
  jyppx::liuding::training::EventOutbox limited({sink}, typed_options(),
                                                bounds);
  EXPECT_THROW(limited.enqueue(run_started(), {}),
               jyppx::liuding::ResourceExhausted);
  EXPECT_EQ(limited.pending_events(), 0U);
  EXPECT_EQ(limited.pending_bytes(), 0U);
  EXPECT_EQ(limited.last_enqueued_event_index(), 0U);
}

TEST(EventOutboxTelemetryTest,
     LoadsActualV1BlobsAndResavesCanonicalSchema2WithSinkCursor) {
  auto state = frozen_v1_state();
  auto first = std::make_shared<NeutralRecordingSink>("sink.a");
  auto second = std::make_shared<NeutralRecordingSink>("sink.b");
  jyppx::liuding::training::EventOutboxOptions bounds;
  bounds.v1_sink_type_migrations = {
      {"test.recording-sink/v1", "test.neutral-sink/v2"}};
  jyppx::liuding::training::EventOutbox migrated({second, first},
                                                 typed_options(), bounds);
  migrated.load_state(state, jyppx::liuding::RestorePolicy::exact_only);
  EXPECT_EQ(migrated.last_delivered_event_index(), 1U);
  EXPECT_EQ(migrated.last_enqueued_event_index(), 2U);
  EXPECT_EQ(migrated.pending_events(), 1U);
  const auto stream = std::string(migrated.telemetry_stream_id());
  MemoryStateStore current(migrated.describe_state());
  migrated.save_state(current);
  EXPECT_EQ(current.manifest().schema_version, 2U);

  jyppx::liuding::training::EventOutbox restored({first, second},
                                                 typed_options(), bounds);
  restored.load_state(current, jyppx::liuding::RestorePolicy::exact_only);
  restored.drain();
  EXPECT_TRUE(first->attempted.empty());
  ASSERT_EQ(second->accepted.size(), 1U);
  const auto& old = second->accepted.front();
  EXPECT_EQ(old.identity.sequence, 2U);
  EXPECT_EQ(old.identity.stream_id, stream);
  EXPECT_FALSE(old.time.wall_time_unix_ns.has_value());
  EXPECT_FALSE(old.time.monotonic_time_ns.has_value());
  EXPECT_EQ(old.attributes.at("migration.time"), "unavailable");
  const auto& lifecycle =
      std::get<jyppx::liuding::TelemetryLifecycle>(old.payload);
  ASSERT_EQ(lifecycle.decisions.size(), 1U);
  EXPECT_EQ(lifecycle.decisions.front().actor_id, "callback.stop");
  EXPECT_TRUE(lifecycle.decisions.front().effective);
  EXPECT_EQ(lifecycle.metrics.at(0U).definition.unit,
            jyppx::liuding::TelemetryMetricUnit::unspecified);

  restored.enqueue(epoch_finished(3U), {});
  restored.drain();
  ASSERT_EQ(second->accepted.size(), 2U);
  EXPECT_EQ(second->accepted.back().identity.sequence, 3U);
  EXPECT_EQ(second->accepted.back().identity.stream_id, stream);
  EXPECT_TRUE(second->accepted.back().time.wall_time_unix_ns.has_value());
  EXPECT_EQ(std::get<jyppx::liuding::TelemetryLifecycle>(
                second->accepted.back().payload)
                .metrics.at(0U)
                .definition.unit,
            jyppx::liuding::TelemetryMetricUnit::dimensionless);
}

TEST(EventOutboxTelemetryTest,
     V1TopologyAndCorruptionCannotResetLiveDeliveryState) {
  auto state = frozen_v1_state();
  auto first = std::make_shared<NeutralRecordingSink>("sink.a");
  auto second = std::make_shared<NeutralRecordingSink>("sink.b");
  jyppx::liuding::training::EventOutbox target({first, second},
                                               typed_options());
  target.enqueue(run_started(), {});
  const auto stream = std::string(target.telemetry_stream_id());
  const auto charge = target.pending_bytes();
  EXPECT_THROW(
      target.load_state(state, jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::CompatibilityError);
  EXPECT_EQ(target.pending_bytes(), charge);
  EXPECT_EQ(target.telemetry_stream_id(), stream);

  jyppx::liuding::training::EventOutboxOptions bounds;
  bounds.v1_sink_type_migrations = {
      {"test.recording-sink/v1", "test.neutral-sink/v2"}};
  jyppx::liuding::training::EventOutbox mapped({first, second}, typed_options(),
                                               bounds);
  mapped.enqueue(run_started(), {});
  auto& control = state.mutable_blob("control");
  control.push_back(std::byte{0});
  EXPECT_THROW(
      mapped.load_state(state, jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::SerializationError);
  EXPECT_EQ(mapped.last_enqueued_event_index(), 1U);
  EXPECT_EQ(mapped.pending_events(), 1U);
  EXPECT_EQ(mapped.last_delivered_event_index(), 0U);
}

TEST(EventOutboxTelemetryTest,
     V1TerminalClosureSurvivesRealBlobMigrationAndRestore) {
  auto state = frozen_v1_state(true);
  auto first = std::make_shared<RecordingSink>("sink.a");
  auto second = std::make_shared<RecordingSink>("sink.b");
  jyppx::liuding::training::EventOutbox migrated({first, second});
  migrated.load_state(state, jyppx::liuding::RestorePolicy::exact_only);
  EXPECT_THROW(migrated.enqueue(epoch_finished(3U), {}),
               jyppx::liuding::InvalidArgument);
  MemoryStateStore current(migrated.describe_state());
  migrated.save_state(current);
  jyppx::liuding::training::EventOutbox restored({first, second});
  restored.load_state(current, jyppx::liuding::RestorePolicy::exact_only);
  restored.drain();
  EXPECT_EQ(first->accepted_indices(), std::vector<std::uint64_t>{2U});
  EXPECT_EQ(second->accepted_indices(), std::vector<std::uint64_t>{2U});
  EXPECT_THROW(restored.enqueue(epoch_finished(3U), {}),
               jyppx::liuding::InvalidArgument);
}

}  // namespace
