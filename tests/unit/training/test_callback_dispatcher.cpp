// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/training/callback_dispatcher.hpp>
#include <jyppx/training/callbacks/early_stopping_callback.hpp>
#include <jyppx/training/checkpoint_coordinator.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

class MemoryStateStore final : public jyppx::StateWriter,
                               public jyppx::StateReader {
 public:
  /**
   * @brief Creates a bounded in-memory participant store.
   * @param manifest Exact manifest enforced by reads and writes.
   */
  explicit MemoryStateStore(jyppx::StateManifest manifest)
      : manifest_(std::move(manifest)) {
    jyppx::validate_state_manifest(manifest_);
  }

  /** @brief Stores one declared field exactly once. */
  void write_blob(std::string_view name,
                  std::span<const std::byte> data) override {
    const auto field =
        std::find_if(manifest_.fields.begin(), manifest_.fields.end(),
                     [name](const jyppx::StateFieldDescriptor& candidate) {
                       return candidate.name == name;
                     });
    if (field == manifest_.fields.end() || data.size() > field->max_bytes ||
        !blobs_
             .emplace(std::string(name),
                      std::vector<std::byte>(data.begin(), data.end()))
             .second) {
      throw jyppx::SerializationError("invalid callback test write");
    }
  }

  /** @brief Returns the exact manifest supplied at construction. */
  const jyppx::StateManifest& manifest() const noexcept override {
    return manifest_;
  }

  /** @brief Reports whether a named field has been written. */
  bool contains(std::string_view name) const override {
    return blobs_.contains(std::string(name));
  }

  /** @brief Reads one stored field under the caller's byte limit. */
  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t max_bytes) const override {
    const auto field = blobs_.find(std::string(name));
    if (field == blobs_.end() || field->second.size() > max_bytes) {
      throw jyppx::SerializationError("invalid callback test read");
    }
    return field->second;
  }

  /**
   * @brief Replaces one byte near the end of a stored field.
   * @param name Existing field name.
   * @param trailing_offset Zero selects the final byte.
   * @param value Replacement byte value.
   */
  void overwrite_from_end(std::string_view name, std::size_t trailing_offset,
                          std::byte value) {
    auto& bytes = blobs_.at(std::string(name));
    bytes.at(bytes.size() - 1U - trailing_offset) = value;
  }

 private:
  jyppx::StateManifest manifest_;  ///< Exact participant manifest.
  std::map<std::string, std::vector<std::byte>>
      blobs_;  ///< Encoded fields indexed by stable field name.
};

class RecordingCallback final : public jyppx::training::CallbackHandler {
 public:
  /**
   * @brief Creates a stateless callback with one deterministic response.
   * @param descriptor Frozen callback capabilities.
   * @param request Response returned for every subscribed event.
   * @param calls Shared execution-order recorder.
   */
  RecordingCallback(jyppx::training::CallbackDescriptor descriptor,
                    jyppx::training::CallbackCommandRequest request,
                    std::vector<std::string>* calls)
      : descriptor_(std::move(descriptor)),
        request_(std::move(request)),
        calls_(calls) {}

  /** @brief Returns the frozen mock descriptor. */
  const jyppx::training::CallbackDescriptor& descriptor()
      const noexcept override {
    return descriptor_;
  }

  /** @brief Records invocation and returns the configured response. */
  jyppx::training::CallbackCommandRequest handle(
      const jyppx::training::CallbackEvent&) override {
    calls_->push_back(descriptor_.callback_id);
    return request_;
  }

 private:
  jyppx::training::CallbackDescriptor descriptor_;   ///< Mock capabilities.
  jyppx::training::CallbackCommandRequest request_;  ///< Fixed response.
  std::vector<std::string>* calls_;  ///< Borrowed execution-order recorder.
};

class RollbackFailingCallback final : public jyppx::training::CallbackHandler,
                                      public jyppx::CheckpointParticipant {
 public:
  /** @brief Creates a stateful callback that issues an undeclared command. */
  RollbackFailingCallback() = default;

  /** @brief Returns the fixed stateful test descriptor. */
  const jyppx::training::CallbackDescriptor& descriptor()
      const noexcept override {
    return descriptor_;
  }

  /** @brief Mutates state before returning an intentionally invalid command. */
  jyppx::training::CallbackCommandRequest handle(
      const jyppx::training::CallbackEvent&) override {
    ++value_;
    return {.kind = jyppx::training::CallbackCommandKind::request_checkpoint,
            .reason = "undeclared"};
  }

  /** @brief Returns the callback identity used by the test manifest. */
  std::string participant_id() const override {
    return descriptor_.callback_id;
  }

  /** @brief Describes one byte of exact callback state. */
  jyppx::StateManifest describe_state() const override {
    return {.participant_id = participant_id(),
            .schema_version = 1,
            .restore_level = jyppx::RestoreLevel::exact,
            .fields = {{.name = "value", .max_bytes = 1}}};
  }

  /** @brief Saves the one-byte mutable test value. */
  void save_state(jyppx::StateWriter& writer) const override {
    const std::array bytes{static_cast<std::byte>(value_)};
    writer.write_blob("value", bytes);
  }

  /** @brief Fails deliberately to exercise indeterminate rollback reporting. */
  void load_state(const jyppx::StateReader&, jyppx::RestorePolicy) override {
    throw jyppx::SerializationError("deliberate callback rollback failure");
  }

  /** @brief Returns the mutable value for post-failure assertions. */
  std::uint8_t value() const noexcept { return value_; }

 private:
  jyppx::training::CallbackDescriptor descriptor_{
      .schema_version = 1,
      .type_id = "test.rollback_failure.v1",
      .callback_id = "callback.rollback_failure",
      .priority = 0,
      .subscriptions = {jyppx::training::CallbackEventKind::epoch_finished},
      .allowed_commands = {},
      .stateful = true};    ///< Fixed state identity and capabilities.
  std::uint8_t value_ = 0;  ///< Mutable value changed before failure.
};

jyppx::training::CallbackDescriptor descriptor(
    std::string callback_id, std::int32_t priority,
    jyppx::training::CallbackCommandKind allowed) {
  std::vector<jyppx::training::CallbackCommandKind> commands;
  if (allowed != jyppx::training::CallbackCommandKind::none) {
    commands.push_back(allowed);
  }
  return {.schema_version = 1,
          .type_id = "test.recording.v1",
          .callback_id = std::move(callback_id),
          .priority = priority,
          .subscriptions = {jyppx::training::CallbackEventKind::epoch_finished},
          .allowed_commands = std::move(commands),
          .stateful = false};
}

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

jyppx::training::CallbackEvent epoch_finished(
    std::uint64_t event_index, std::uint64_t epoch,
    std::map<std::string, double> metrics = {}) {
  return {.schema_version = 1,
          .kind = jyppx::training::CallbackEventKind::epoch_finished,
          .phase = jyppx::training::CallbackPhase::training,
          .event_index = event_index,
          .run_id = "run.test",
          .trace_id = "trace.test",
          .epoch = epoch,
          .global_step = epoch,
          .metrics = std::move(metrics)};
}

TEST(CallbackDispatcherTest, UsesStableExecutionAndFixedCommandPriority) {
  using jyppx::training::CallbackCommandKind;
  std::vector<std::string> calls;
  auto checkpoint = std::make_shared<RecordingCallback>(
      descriptor("callback.a", 10, CallbackCommandKind::request_checkpoint),
      jyppx::training::CallbackCommandRequest{
          .kind = CallbackCommandKind::request_checkpoint,
          .reason = "periodic"},
      &calls);
  auto skip = std::make_shared<RecordingCallback>(
      descriptor("callback.z", 100, CallbackCommandKind::skip_validation),
      jyppx::training::CallbackCommandRequest{
          .kind = CallbackCommandKind::skip_validation, .reason = "not due"},
      &calls);
  auto stop = std::make_shared<RecordingCallback>(
      descriptor("callback.b", 10, CallbackCommandKind::stop),
      jyppx::training::CallbackCommandRequest{.kind = CallbackCommandKind::stop,
                                              .reason = "limit"},
      &calls);
  jyppx::training::CallbackDispatcher dispatcher({stop, checkpoint, skip});

  EXPECT_TRUE(dispatcher.dispatch(run_started()).commands.empty());
  const auto result = dispatcher.dispatch(epoch_finished(2, 1));

  EXPECT_EQ(calls, (std::vector<std::string>{"callback.z", "callback.a",
                                             "callback.b"}));
  ASSERT_EQ(result.commands.size(), 3U);
  ASSERT_TRUE(result.effective_command.has_value());
  EXPECT_EQ(result.effective_command->kind, CallbackCommandKind::stop);
  EXPECT_EQ(result.effective_command->callback_id, "callback.b");
  EXPECT_EQ(dispatcher.last_event_index(), 2U);
}

TEST(CallbackDispatcherTest, RejectsInvalidAndDuplicateEventStreams) {
  jyppx::training::CallbackDispatcher dispatcher;
  auto wrong_start = epoch_finished(1, 0);
  EXPECT_THROW(dispatcher.dispatch(wrong_start), jyppx::InvalidArgument);

  dispatcher.dispatch(run_started());
  auto wrong_phase = epoch_finished(2, 1);
  wrong_phase.phase = jyppx::training::CallbackPhase::validation;
  EXPECT_THROW(dispatcher.dispatch(wrong_phase), jyppx::InvalidArgument);
  EXPECT_EQ(dispatcher.last_event_index(), 1U);

  dispatcher.dispatch(epoch_finished(2, 1));
  EXPECT_THROW(dispatcher.dispatch(epoch_finished(2, 1)),
               jyppx::InvalidArgument);
  EXPECT_THROW(dispatcher.dispatch(epoch_finished(4, 2)),
               jyppx::InvalidArgument);

  auto terminal = epoch_finished(3, 2);
  terminal.kind = jyppx::training::CallbackEventKind::run_finished;
  terminal.phase = jyppx::training::CallbackPhase::completed;
  dispatcher.dispatch(terminal);
  EXPECT_THROW(dispatcher.dispatch(epoch_finished(4, 3)),
               jyppx::InvalidArgument);
}

TEST(CallbackDispatcherTest, RejectsUndeclaredCommandCapability) {
  using jyppx::training::CallbackCommandKind;
  std::vector<std::string> calls;
  auto callback = std::make_shared<RecordingCallback>(
      descriptor("callback.invalid", 0, CallbackCommandKind::stop),
      jyppx::training::CallbackCommandRequest{
          .kind = CallbackCommandKind::request_checkpoint,
          .reason = "undeclared"},
      &calls);
  jyppx::training::CallbackDispatcher dispatcher({callback});
  dispatcher.dispatch(run_started());

  EXPECT_THROW(dispatcher.dispatch(epoch_finished(2, 1)),
               jyppx::CompatibilityError);
  EXPECT_EQ(dispatcher.last_event_index(), 1U);
}

TEST(CallbackDispatcherTest, RejectsDuplicateAndFalseStatefulDescriptors) {
  std::vector<std::string> calls;
  auto first = std::make_shared<RecordingCallback>(
      descriptor("callback.same", 0,
                 jyppx::training::CallbackCommandKind::none),
      jyppx::training::CallbackCommandRequest{}, &calls);
  auto second = std::make_shared<RecordingCallback>(
      descriptor("callback.same", 1,
                 jyppx::training::CallbackCommandKind::none),
      jyppx::training::CallbackCommandRequest{}, &calls);
  EXPECT_THROW(jyppx::training::CallbackDispatcher({first, second}),
               jyppx::InvalidArgument);

  auto stateful_descriptor = descriptor(
      "callback.false_state", 0, jyppx::training::CallbackCommandKind::none);
  stateful_descriptor.stateful = true;
  auto false_stateful = std::make_shared<RecordingCallback>(
      std::move(stateful_descriptor), jyppx::training::CallbackCommandRequest{},
      &calls);
  EXPECT_THROW(jyppx::training::CallbackDispatcher({false_stateful}),
               jyppx::InvalidArgument);
}

TEST(EarlyStoppingCallbackTest, AppliesExplicitPatienceAndDeltaSemantics) {
  auto early =
      std::make_shared<jyppx::training::callbacks::EarlyStoppingCallback>(
          jyppx::training::callbacks::EarlyStoppingOptions{
              .callback_id = "callback.early",
              .monitor = "val_loss",
              .patience = 1,
              .min_delta = 0.1});
  jyppx::training::CallbackDispatcher dispatcher({early});
  dispatcher.dispatch(run_started());

  EXPECT_FALSE(dispatcher.dispatch(epoch_finished(2, 1, {{"val_loss", 1.0}}))
                   .effective_command.has_value());
  EXPECT_FALSE(dispatcher.dispatch(epoch_finished(3, 2, {{"val_loss", 0.95}}))
                   .effective_command.has_value());
  EXPECT_EQ(early->non_improving_observations(), 1U);
  const auto stopped =
      dispatcher.dispatch(epoch_finished(4, 3, {{"val_loss", 0.94}}));
  ASSERT_TRUE(stopped.effective_command.has_value());
  EXPECT_EQ(stopped.effective_command->kind,
            jyppx::training::CallbackCommandKind::stop);
  EXPECT_TRUE(early->should_stop());
  EXPECT_DOUBLE_EQ(early->best_value(), 1.0);
}

TEST(EarlyStoppingCallbackTest, RejectsMissingMetricWithoutStateMutation) {
  auto early =
      std::make_shared<jyppx::training::callbacks::EarlyStoppingCallback>();
  jyppx::training::CallbackDispatcher dispatcher({early});
  dispatcher.dispatch(run_started());

  try {
    (void)dispatcher.dispatch(epoch_finished(2, 1));
    FAIL() << "callback failure should terminate dispatch";
  } catch (const jyppx::InternalError& error) {
    EXPECT_EQ(error.details().actual,
              "callback.early_stopping at epoch_finished");
    EXPECT_THROW(std::rethrow_if_nested(error), jyppx::InvalidArgument);
  }
  EXPECT_FALSE(early->has_best_value());
  EXPECT_EQ(early->non_improving_observations(), 0U);
  EXPECT_EQ(dispatcher.last_event_index(), 1U);
}

TEST(CallbackDispatcherTest, LaterInvalidCommandRollsBackEarlierStatefulWork) {
  using jyppx::training::CallbackCommandKind;
  std::vector<std::string> calls;
  auto early =
      std::make_shared<jyppx::training::callbacks::EarlyStoppingCallback>(
          jyppx::training::callbacks::EarlyStoppingOptions{
              .callback_id = "callback.early", .priority = 10});
  auto invalid = std::make_shared<RecordingCallback>(
      descriptor("callback.invalid", 0, CallbackCommandKind::stop),
      jyppx::training::CallbackCommandRequest{
          .kind = CallbackCommandKind::request_checkpoint,
          .reason = "undeclared"},
      &calls);
  jyppx::training::CallbackDispatcher dispatcher({invalid, early});
  dispatcher.dispatch(run_started());

  EXPECT_THROW(dispatcher.dispatch(epoch_finished(2, 1, {{"val_loss", 1.0}})),
               jyppx::CompatibilityError);
  EXPECT_FALSE(early->has_best_value());
  EXPECT_EQ(early->non_improving_observations(), 0U);
  EXPECT_EQ(dispatcher.last_event_index(), 1U);
}

TEST(CallbackDispatcherTest, RegistersCompleteCallbackStateDomainAtomically) {
  using jyppx::training::callbacks::EarlyStoppingCallback;
  using jyppx::training::callbacks::EarlyStoppingOptions;
  auto existing = std::make_shared<EarlyStoppingCallback>(
      EarlyStoppingOptions{.callback_id = "callback.shared"});
  jyppx::training::CheckpointParticipantRegistry registry;
  registry.register_participant(existing);

  auto duplicate = std::make_shared<EarlyStoppingCallback>(
      EarlyStoppingOptions{.callback_id = "callback.shared"});
  auto rejected_dispatcher =
      std::make_shared<jyppx::training::CallbackDispatcher>(
          std::vector<std::shared_ptr<jyppx::training::CallbackHandler>>{
              duplicate},
          jyppx::training::CallbackDispatcherOptions{.dispatcher_id =
                                                         "callbacks.rejected"});
  EXPECT_THROW(jyppx::training::register_callback_checkpoint_participants(
                   registry, rejected_dispatcher),
               jyppx::InvalidArgument);
  EXPECT_EQ(registry.participant_ids(),
            std::vector<std::string>{"callback.shared"});

  auto accepted = std::make_shared<EarlyStoppingCallback>(
      EarlyStoppingOptions{.callback_id = "callback.accepted"});
  auto accepted_dispatcher =
      std::make_shared<jyppx::training::CallbackDispatcher>(
          std::vector<std::shared_ptr<jyppx::training::CallbackHandler>>{
              accepted},
          jyppx::training::CallbackDispatcherOptions{.dispatcher_id =
                                                         "callbacks.accepted"});
  jyppx::training::register_callback_checkpoint_participants(
      registry, accepted_dispatcher);
  EXPECT_EQ(registry.participant_ids(),
            (std::vector<std::string>{"callback.accepted", "callback.shared",
                                      "callbacks.accepted"}));
}

TEST(CallbackDispatcherTest, EnforcesRollbackBudgetBeforeHandlerMutation) {
  auto early =
      std::make_shared<jyppx::training::callbacks::EarlyStoppingCallback>();
  jyppx::training::CallbackDispatcher dispatcher(
      {early}, jyppx::training::CallbackDispatcherOptions{
                   .max_dispatch_rollback_bytes = 1});
  dispatcher.dispatch(run_started());

  EXPECT_THROW(dispatcher.dispatch(epoch_finished(2, 1, {{"val_loss", 1.0}})),
               jyppx::ResourceExhausted);
  EXPECT_FALSE(early->has_best_value());
  EXPECT_EQ(dispatcher.last_event_index(), 1U);
}

TEST(CallbackDispatcherTest, ReportsIncompleteRollbackWithOriginalCause) {
  auto callback = std::make_shared<RollbackFailingCallback>();
  jyppx::training::CallbackDispatcher dispatcher({callback});
  dispatcher.dispatch(run_started());

  try {
    (void)dispatcher.dispatch(epoch_finished(2, 1));
    FAIL() << "incomplete rollback should be terminal";
  } catch (const jyppx::InternalError& error) {
    EXPECT_EQ(error.details().operation, "dispatch.rollback");
    EXPECT_EQ(error.details().actual, "callback.rollback_failure");
    EXPECT_THROW(std::rethrow_if_nested(error), jyppx::CompatibilityError);
  }
  EXPECT_EQ(callback->value(), 1U);
  EXPECT_EQ(dispatcher.last_event_index(), 1U);
}

TEST(CallbackDispatcherTest, RestoresDispatcherAndStatefulCallbackExactly) {
  using jyppx::training::callbacks::EarlyStoppingCallback;
  const auto options = jyppx::training::callbacks::EarlyStoppingOptions{
      .callback_id = "callback.resume",
      .monitor = "score",
      .patience = 1,
      .min_delta = 0.0,
      .mode = jyppx::training::callbacks::EarlyStoppingMode::maximize};
  auto source_callback = std::make_shared<EarlyStoppingCallback>(options);
  jyppx::training::CallbackDispatcher source({source_callback});
  source.dispatch(run_started());
  source.dispatch(epoch_finished(2, 1, {{"score", 0.8}}));
  source.dispatch(epoch_finished(3, 2, {{"score", 0.7}}));

  MemoryStateStore dispatcher_store(source.describe_state());
  source.save_state(dispatcher_store);
  MemoryStateStore callback_store(source_callback->describe_state());
  source_callback->save_state(callback_store);

  auto restored_callback = std::make_shared<EarlyStoppingCallback>(options);
  jyppx::training::CallbackDispatcher restored({restored_callback});
  restored_callback->load_state(callback_store,
                                jyppx::RestorePolicy::exact_only);
  restored.load_state(dispatcher_store, jyppx::RestorePolicy::exact_only);

  const auto participants = restored.checkpoint_participants();
  ASSERT_EQ(participants.size(), 1U);
  EXPECT_EQ(participants.front()->participant_id(), "callback.resume");
  EXPECT_EQ(restored.last_event_index(), 3U);
  EXPECT_EQ(restored_callback->non_improving_observations(), 1U);
  const auto result = restored.dispatch(epoch_finished(4, 3, {{"score", 0.6}}));
  ASSERT_TRUE(result.effective_command.has_value());
  EXPECT_EQ(result.effective_command->kind,
            jyppx::training::CallbackCommandKind::stop);
}

TEST(EarlyStoppingCallbackTest, RejectsCorruptStateTransactionally) {
  using jyppx::training::callbacks::EarlyStoppingCallback;
  EarlyStoppingCallback source;
  source.handle(epoch_finished(2, 1, {{"val_loss", 1.0}}));
  MemoryStateStore store(source.describe_state());
  source.save_state(store);
  store.overwrite_from_end("state", 0, std::byte{1});

  EarlyStoppingCallback restored;
  restored.handle(epoch_finished(2, 1, {{"val_loss", 2.0}}));
  EXPECT_THROW(restored.load_state(store, jyppx::RestorePolicy::exact_only),
               jyppx::SerializationError);
  EXPECT_TRUE(restored.has_best_value());
  EXPECT_DOUBLE_EQ(restored.best_value(), 2.0);
  EXPECT_FALSE(restored.should_stop());
}

}  // namespace
