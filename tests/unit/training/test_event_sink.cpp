// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/training/callback_dispatcher.hpp>
#include <jyppx/training/callbacks/early_stopping_callback.hpp>
#include <jyppx/training/event_sink.hpp>

#include <algorithm>
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

class MemoryStateStore final : public jyppx::StateWriter,
                               public jyppx::StateReader {
 public:
  /** @brief Creates an empty store for one exact participant manifest. */
  explicit MemoryStateStore(jyppx::StateManifest manifest)
      : manifest_(std::move(manifest)) {}

  /** @brief Copies one declared field into owned test memory. */
  void write_blob(std::string_view name,
                  std::span<const std::byte> data) override {
    blobs_.emplace(std::string(name),
                   std::vector<std::byte>(data.begin(), data.end()));
  }

  /** @brief Returns the immutable participant manifest. */
  const jyppx::StateManifest& manifest() const noexcept override {
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
      throw jyppx::ResourceExhausted("test state exceeds read limit");
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

 private:
  jyppx::StateManifest manifest_;  ///< Participant-owned field contract.
  std::map<std::string, std::vector<std::byte>>
      blobs_;  ///< Complete blobs keyed by field name.
};

class RecordingSink final : public jyppx::training::EventSink {
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
                    .delivery_guarantee =
                        jyppx::training::EventDeliveryGuarantee::at_least_once},
        failures_remaining_(failures),
        close_failures_remaining_(close_failures) {}

  /** @brief Returns immutable test sink identity. */
  const jyppx::training::EventSinkDescriptor& descriptor()
      const noexcept override {
    return descriptor_;
  }

  /** @brief Records the event unless an injected failure remains. */
  void publish(const jyppx::training::EventEnvelope& envelope,
               std::stop_token stop) override {
    ++publish_attempts_;
    if (stop.stop_requested()) {
      throw jyppx::Cancelled("test sink cancelled");
    }
    if (failures_remaining_ > 0) {
      --failures_remaining_;
      throw jyppx::IoError("injected sink failure");
    }
    accepted_indices_.push_back(envelope.event.event_index);
  }

  /** @brief Records a flush operation. */
  void flush(std::stop_token stop) override {
    if (stop.stop_requested()) {
      throw jyppx::Cancelled("test flush cancelled");
    }
    if (closed_) {
      throw jyppx::InternalError("flush called after test sink close");
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
      throw jyppx::IoError("injected sink close failure");
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
  jyppx::training::EventSinkDescriptor descriptor_;  ///< Stable sink topology.
  int failures_remaining_ = 0;  ///< Rejected attempts before first success.
  int close_failures_remaining_ = 0;  ///< Rejected close attempts.
  int publish_attempts_ = 0;          ///< All publish invocations.
  int flush_calls_ = 0;               ///< Successful flush invocations.
  int close_calls_ = 0;               ///< Successful close transitions.
  bool closed_ = false;               ///< Whether explicit close completed.
  std::vector<std::uint64_t> accepted_indices_;  ///< Accepted event order.
};

jyppx::training::CallbackEvent run_started() {
  return {.schema_version = 1,
          .kind = jyppx::training::CallbackEventKind::run_started,
          .phase = jyppx::training::CallbackPhase::preparing,
          .event_index = 1,
          .run_id = "run.test",
          .trace_id = "trace.test",
          .epoch = 0,
          .global_step = 0,
          .metrics = {}};
}

jyppx::training::CallbackEvent epoch_finished(std::uint64_t index,
                                              double loss = 1.0) {
  return {.schema_version = 1,
          .kind = jyppx::training::CallbackEventKind::epoch_finished,
          .phase = jyppx::training::CallbackPhase::training,
          .event_index = index,
          .run_id = "run.test",
          .trace_id = "trace.test",
          .epoch = index - 1U,
          .global_step = index - 1U,
          .metrics = {{"val_loss", loss}}};
}

TEST(EventOutboxTest, PublishesOnlyAfterCallbackTransactionCommits) {
  auto sink = std::make_shared<RecordingSink>("sink.recording");
  jyppx::training::EventOutbox outbox({sink});
  jyppx::training::CallbackDispatcher dispatcher;

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
  jyppx::training::EventOutbox outbox(
      {sink}, jyppx::training::EventOutboxOptions{.max_pending_events = 1});
  auto early =
      std::make_shared<jyppx::training::callbacks::EarlyStoppingCallback>();
  jyppx::training::CallbackDispatcher dispatcher({early});
  dispatcher.dispatch(run_started(), outbox);

  EXPECT_THROW(dispatcher.dispatch(epoch_finished(2), outbox),
               jyppx::ResourceExhausted);
  EXPECT_FALSE(early->has_best_value());
  EXPECT_EQ(dispatcher.last_event_index(), 1U);
  EXPECT_EQ(outbox.last_enqueued_event_index(), 1U);
  EXPECT_EQ(outbox.pending_events(), 1U);
}

TEST(EventOutboxTest, RetriesOnlyTheFirstUnacknowledgedSink) {
  auto first = std::make_shared<RecordingSink>("sink.a");
  auto second = std::make_shared<RecordingSink>("sink.b", 1);
  jyppx::training::EventOutbox outbox({second, first});
  outbox.enqueue(run_started(), {});

  EXPECT_THROW(outbox.drain_one(), jyppx::IoError);
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
  jyppx::training::EventOutbox source({source_second, source_first});
  source.enqueue(run_started(), {});
  EXPECT_THROW(source.drain_one(), jyppx::IoError);

  MemoryStateStore state(source.describe_state());
  source.save_state(state);

  auto restored_first = std::make_shared<RecordingSink>("sink.a");
  auto restored_second = std::make_shared<RecordingSink>("sink.b");
  jyppx::training::EventOutbox restored({restored_second, restored_first});
  restored.load_state(state, jyppx::RestorePolicy::exact_only);

  EXPECT_EQ(restored.pending_events(), 1U);
  EXPECT_TRUE(restored.drain_one());
  EXPECT_TRUE(restored_first->accepted_indices().empty());
  EXPECT_EQ(restored_second->accepted_indices(), std::vector<std::uint64_t>{1});
  EXPECT_EQ(restored.last_delivered_event_index(), 1U);
}

TEST(EventOutboxTest, CheckpointRestoresDeliveredPrefixAndPendingSuffix) {
  auto source_sink = std::make_shared<RecordingSink>("sink.recording");
  jyppx::training::EventOutbox source({source_sink});
  source.enqueue(run_started(), {});
  ASSERT_TRUE(source.drain_one());
  source.enqueue(epoch_finished(2), {});

  MemoryStateStore state(source.describe_state());
  source.save_state(state);

  auto restored_sink = std::make_shared<RecordingSink>("sink.recording");
  jyppx::training::EventOutbox restored({restored_sink});
  restored.load_state(state, jyppx::RestorePolicy::exact_only);

  EXPECT_EQ(restored.last_delivered_event_index(), 1U);
  EXPECT_EQ(restored.last_enqueued_event_index(), 2U);
  ASSERT_EQ(restored.pending_events(), 1U);
  EXPECT_TRUE(restored.drain_one());
  EXPECT_EQ(restored_sink->accepted_indices(), std::vector<std::uint64_t>{2});
  EXPECT_EQ(restored.last_delivered_event_index(), 2U);
}

TEST(EventOutboxTest, CorruptRestoreLeavesLiveQueueUnchanged) {
  auto sink = std::make_shared<RecordingSink>("sink.recording");
  jyppx::training::EventOutbox source({sink});
  source.enqueue(run_started(), {});
  MemoryStateStore state(source.describe_state());
  source.save_state(state);
  state.overwrite_from_end("control", 0, std::byte{2});

  auto target_sink = std::make_shared<RecordingSink>("sink.recording");
  jyppx::training::EventOutbox target({target_sink});
  target.enqueue(run_started(), {});
  EXPECT_THROW(target.load_state(state, jyppx::RestorePolicy::exact_only),
               jyppx::Error);
  EXPECT_EQ(target.pending_events(), 1U);
  EXPECT_EQ(target.last_enqueued_event_index(), 1U);
}

TEST(EventOutboxTest, CancellationAndCloseHaveExplicitBoundaries) {
  auto sink = std::make_shared<RecordingSink>("sink.recording");
  jyppx::training::EventOutbox outbox({sink});
  outbox.enqueue(run_started(), {});
  std::stop_source cancellation;
  cancellation.request_stop();

  EXPECT_THROW(outbox.drain(cancellation.get_token()), jyppx::Cancelled);
  EXPECT_EQ(outbox.pending_events(), 1U);
  outbox.close();
  outbox.close();
  EXPECT_TRUE(outbox.closed());
  EXPECT_EQ(sink->close_calls(), 1);
  EXPECT_THROW(outbox.enqueue(epoch_finished(2), {}), jyppx::NotReady);
}

TEST(EventOutboxTest, CloseRetrySkipsAlreadyClosedSinks) {
  auto first = std::make_shared<RecordingSink>("sink.a");
  auto second = std::make_shared<RecordingSink>("sink.b", 0, 1);
  jyppx::training::EventOutbox outbox({second, first});

  EXPECT_THROW(outbox.close(), jyppx::IoError);
  EXPECT_EQ(first->close_calls(), 1);
  EXPECT_EQ(first->flush_calls(), 1);
  EXPECT_EQ(second->close_calls(), 0);
  EXPECT_THROW(
      [&] {
        MemoryStateStore state(outbox.describe_state());
        outbox.save_state(state);
      }(),
      jyppx::NotReady);

  outbox.close();
  EXPECT_TRUE(outbox.closed());
  EXPECT_EQ(first->close_calls(), 1);
  EXPECT_EQ(first->flush_calls(), 1);
  EXPECT_EQ(second->close_calls(), 1);
  EXPECT_EQ(second->flush_calls(), 1);
}

TEST(EventOutboxTest, RejectsInconsistentCallbackResultWithoutMutation) {
  auto sink = std::make_shared<RecordingSink>("sink.recording");
  jyppx::training::EventOutbox outbox({sink});
  jyppx::training::CallbackDispatchResult invalid{
      .commands = {{.kind = jyppx::training::CallbackCommandKind::stop,
                    .callback_id = "callback.stop",
                    .reason = "done"}},
      .effective_command = std::nullopt};

  EXPECT_THROW(outbox.enqueue(run_started(), invalid),
               jyppx::CompatibilityError);
  EXPECT_EQ(outbox.pending_events(), 0U);
  EXPECT_EQ(outbox.last_enqueued_event_index(), 0U);
}

TEST(EventSinkTest, ConsoleWritesCanonicalMachineReadableLine) {
  std::ostringstream output;
  auto sink = std::make_shared<jyppx::training::ConsoleEventSink>(output);
  jyppx::training::EventOutbox outbox({sink});
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
    auto sink = std::make_shared<jyppx::training::JsonlEventSink>(
        jyppx::training::JsonlEventSinkOptions{.path = path});
    jyppx::training::EventOutbox outbox({sink});
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

}  // namespace
