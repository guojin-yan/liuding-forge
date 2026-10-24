// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/training/callback_dispatcher.hpp>
#include <jyppx/liuding/training/callbacks/early_stopping_callback.hpp>
#include <jyppx/liuding/training/checkpoint_coordinator.hpp>

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

class MemoryStateStore final : public jyppx::liuding::StateWriter,
                               public jyppx::liuding::StateReader {
 public:
  /**
   * @brief Creates a bounded in-memory participant store.
   * @param manifest Exact manifest enforced by reads and writes.
   */
  explicit MemoryStateStore(jyppx::liuding::StateManifest manifest)
      : manifest_(std::move(manifest)) {
    jyppx::liuding::validate_state_manifest(manifest_);
  }

  /** @brief Stores one declared field exactly once. */
  void write_blob(std::string_view name,
                  std::span<const std::byte> data) override {
    const auto field =
        std::find_if(manifest_.fields.begin(), manifest_.fields.end(),
                     [name](const jyppx::liuding::StateFieldDescriptor& candidate) {
                       return candidate.name == name;
                     });
    if (field == manifest_.fields.end() || data.size() > field->max_bytes ||
        !blobs_
             .emplace(std::string(name),
                      std::vector<std::byte>(data.begin(), data.end()))
             .second) {
      throw jyppx::liuding::SerializationError("invalid callback test write");
    }
  }

  /** @brief Returns the exact manifest supplied at construction. */
  const jyppx::liuding::StateManifest& manifest() const noexcept override {
    return manifest_;
  }

  /** @brief Changes metadata independently to exercise strict restore checks.
   */
  jyppx::liuding::StateManifest& mutable_manifest() noexcept { return manifest_; }

  /** @brief Reports whether a named field has been written. */
  bool contains(std::string_view name) const override {
    return blobs_.contains(std::string(name));
  }

  /** @brief Reads one stored field under the caller's byte limit. */
  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t max_bytes) const override {
    const auto field = blobs_.find(std::string(name));
    if (field == blobs_.end() || field->second.size() > max_bytes) {
      throw jyppx::liuding::SerializationError("invalid callback test read");
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
  jyppx::liuding::StateManifest manifest_;  ///< Exact participant manifest.
  std::map<std::string, std::vector<std::byte>>
      blobs_;  ///< Encoded fields indexed by stable field name.
};

class RecordingCallback final : public jyppx::liuding::training::CallbackHandler {
 public:
  /**
   * @brief Creates a stateless callback with one deterministic response.
   * @param descriptor Frozen callback capabilities.
   * @param request Response returned for every subscribed event.
   * @param calls Shared execution-order recorder.
   */
  RecordingCallback(jyppx::liuding::training::CallbackDescriptor descriptor,
                    jyppx::liuding::training::CallbackCommandRequest request,
                    std::vector<std::string>* calls)
      : descriptor_(std::move(descriptor)),
        request_(std::move(request)),
        calls_(calls) {}

  /** @brief Returns the frozen mock descriptor. */
  const jyppx::liuding::training::CallbackDescriptor& descriptor()
      const noexcept override {
    return descriptor_;
  }

  /** @brief Records invocation and returns the configured response. */
  jyppx::liuding::training::CallbackCommandRequest handle(
      const jyppx::liuding::training::CallbackEvent&) override {
    calls_->push_back(descriptor_.callback_id);
    return request_;
  }

 private:
  jyppx::liuding::training::CallbackDescriptor descriptor_;   ///< Mock capabilities.
  jyppx::liuding::training::CallbackCommandRequest request_;  ///< Fixed response.
  std::vector<std::string>* calls_;  ///< Borrowed execution-order recorder.
};

class RollbackFailingCallback final : public jyppx::liuding::training::CallbackHandler,
                                      public jyppx::liuding::CheckpointParticipant {
 public:
  /** @brief Creates a stateful callback that issues an undeclared command. */
  RollbackFailingCallback() = default;

  /** @brief Returns the fixed stateful test descriptor. */
  const jyppx::liuding::training::CallbackDescriptor& descriptor()
      const noexcept override {
    return descriptor_;
  }

  /** @brief Mutates state before returning an intentionally invalid command. */
  jyppx::liuding::training::CallbackCommandRequest handle(
      const jyppx::liuding::training::CallbackEvent&) override {
    ++value_;
    return {.kind = jyppx::liuding::training::CallbackCommandKind::request_checkpoint,
            .reason = "undeclared"};
  }

  /** @brief Returns the callback identity used by the test manifest. */
  std::string participant_id() const override {
    return descriptor_.callback_id;
  }

  /** @brief Describes one byte of exact callback state. */
  jyppx::liuding::StateManifest describe_state() const override {
    return {.participant_id = participant_id(),
            .schema_version = 1,
            .restore_level = jyppx::liuding::RestoreLevel::exact,
            .fields = {{.name = "value", .max_bytes = 1}}};
  }

  /** @brief Saves the one-byte mutable test value. */
  void save_state(jyppx::liuding::StateWriter& writer) const override {
    const std::array bytes{static_cast<std::byte>(value_)};
    writer.write_blob("value", bytes);
  }

  /** @brief Fails deliberately to exercise indeterminate rollback reporting. */
  void load_state(const jyppx::liuding::StateReader&, jyppx::liuding::RestorePolicy) override {
    throw jyppx::liuding::SerializationError("deliberate callback rollback failure");
  }

  /** @brief Returns the mutable value for post-failure assertions. */
  std::uint8_t value() const noexcept { return value_; }

 private:
  jyppx::liuding::training::CallbackDescriptor descriptor_{
      .schema_version = 1,
      .type_id = "test.rollback_failure.v1",
      .callback_id = "callback.rollback_failure",
      .priority = 0,
      .subscriptions = {jyppx::liuding::training::CallbackEventKind::epoch_finished},
      .allowed_commands = {},
      .stateful = true};    ///< Fixed state identity and capabilities.
  std::uint8_t value_ = 0;  ///< Mutable value changed before failure.
};

jyppx::liuding::training::CallbackDescriptor descriptor(
    std::string callback_id, std::int32_t priority,
    jyppx::liuding::training::CallbackCommandKind allowed) {
  std::vector<jyppx::liuding::training::CallbackCommandKind> commands;
  if (allowed != jyppx::liuding::training::CallbackCommandKind::none) {
    commands.push_back(allowed);
  }
  return {.schema_version = 1,
          .type_id = "test.recording.v1",
          .callback_id = std::move(callback_id),
          .priority = priority,
          .subscriptions = {jyppx::liuding::training::CallbackEventKind::epoch_finished},
          .allowed_commands = std::move(commands),
          .stateful = false};
}

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

jyppx::liuding::training::CallbackEvent epoch_finished(
    std::uint64_t event_index, std::uint64_t epoch,
    std::map<std::string, double> metrics = {}) {
  return {.schema_version = 1,
          .kind = jyppx::liuding::training::CallbackEventKind::epoch_finished,
          .phase = jyppx::liuding::training::CallbackPhase::training,
          .event_index = event_index,
          .run_id = "run.test",
          .trace_id = "trace.test",
          .epoch = epoch,
          .global_step = epoch,
          .metrics = std::move(metrics)};
}

TEST(CallbackDispatcherTest, UsesStableExecutionAndFixedCommandPriority) {
  using jyppx::liuding::training::CallbackCommandKind;
  std::vector<std::string> calls;
  auto checkpoint = std::make_shared<RecordingCallback>(
      descriptor("callback.a", 10, CallbackCommandKind::request_checkpoint),
      jyppx::liuding::training::CallbackCommandRequest{
          .kind = CallbackCommandKind::request_checkpoint,
          .reason = "periodic"},
      &calls);
  auto skip = std::make_shared<RecordingCallback>(
      descriptor("callback.z", 100, CallbackCommandKind::skip_validation),
      jyppx::liuding::training::CallbackCommandRequest{
          .kind = CallbackCommandKind::skip_validation, .reason = "not due"},
      &calls);
  auto stop = std::make_shared<RecordingCallback>(
      descriptor("callback.b", 10, CallbackCommandKind::stop),
      jyppx::liuding::training::CallbackCommandRequest{.kind = CallbackCommandKind::stop,
                                              .reason = "limit"},
      &calls);
  jyppx::liuding::training::CallbackDispatcher dispatcher({stop, checkpoint, skip});

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
  jyppx::liuding::training::CallbackDispatcher dispatcher;
  auto wrong_start = epoch_finished(1, 0);
  EXPECT_THROW(dispatcher.dispatch(wrong_start), jyppx::liuding::InvalidArgument);

  dispatcher.dispatch(run_started());
  auto wrong_phase = epoch_finished(2, 1);
  wrong_phase.phase = jyppx::liuding::training::CallbackPhase::validation;
  EXPECT_THROW(dispatcher.dispatch(wrong_phase), jyppx::liuding::InvalidArgument);
  EXPECT_EQ(dispatcher.last_event_index(), 1U);

  dispatcher.dispatch(epoch_finished(2, 1));
  EXPECT_THROW(dispatcher.dispatch(epoch_finished(2, 1)),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(dispatcher.dispatch(epoch_finished(4, 2)),
               jyppx::liuding::InvalidArgument);

  auto terminal = epoch_finished(3, 2);
  terminal.kind = jyppx::liuding::training::CallbackEventKind::run_finished;
  terminal.phase = jyppx::liuding::training::CallbackPhase::completed;
  dispatcher.dispatch(terminal);
  EXPECT_THROW(dispatcher.dispatch(epoch_finished(4, 3)),
               jyppx::liuding::InvalidArgument);
}

TEST(CallbackDispatcherTest, RejectsUndeclaredCommandCapability) {
  using jyppx::liuding::training::CallbackCommandKind;
  std::vector<std::string> calls;
  auto callback = std::make_shared<RecordingCallback>(
      descriptor("callback.invalid", 0, CallbackCommandKind::stop),
      jyppx::liuding::training::CallbackCommandRequest{
          .kind = CallbackCommandKind::request_checkpoint,
          .reason = "undeclared"},
      &calls);
  jyppx::liuding::training::CallbackDispatcher dispatcher({callback});
  dispatcher.dispatch(run_started());

  EXPECT_THROW(dispatcher.dispatch(epoch_finished(2, 1)),
               jyppx::liuding::CompatibilityError);
  EXPECT_EQ(dispatcher.last_event_index(), 1U);
}

TEST(CallbackDispatcherTest, RejectsDuplicateAndFalseStatefulDescriptors) {
  std::vector<std::string> calls;
  auto first = std::make_shared<RecordingCallback>(
      descriptor("callback.same", 0,
                 jyppx::liuding::training::CallbackCommandKind::none),
      jyppx::liuding::training::CallbackCommandRequest{}, &calls);
  auto second = std::make_shared<RecordingCallback>(
      descriptor("callback.same", 1,
                 jyppx::liuding::training::CallbackCommandKind::none),
      jyppx::liuding::training::CallbackCommandRequest{}, &calls);
  EXPECT_THROW(jyppx::liuding::training::CallbackDispatcher({first, second}),
               jyppx::liuding::InvalidArgument);

  auto stateful_descriptor = descriptor(
      "callback.false_state", 0, jyppx::liuding::training::CallbackCommandKind::none);
  stateful_descriptor.stateful = true;
  auto false_stateful = std::make_shared<RecordingCallback>(
      std::move(stateful_descriptor), jyppx::liuding::training::CallbackCommandRequest{},
      &calls);
  EXPECT_THROW(jyppx::liuding::training::CallbackDispatcher({false_stateful}),
               jyppx::liuding::InvalidArgument);
}

TEST(EarlyStoppingCallbackTest, AppliesExplicitPatienceAndDeltaSemantics) {
  auto early =
      std::make_shared<jyppx::liuding::training::callbacks::EarlyStoppingCallback>(
          jyppx::liuding::training::callbacks::EarlyStoppingOptions{
              .callback_id = "callback.early",
              .monitor = "val_loss",
              .patience = 1,
              .min_delta = 0.1});
  jyppx::liuding::training::CallbackDispatcher dispatcher({early});
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
            jyppx::liuding::training::CallbackCommandKind::stop);
  EXPECT_TRUE(early->should_stop());
  EXPECT_DOUBLE_EQ(early->best_value(), 1.0);
}

TEST(EarlyStoppingCallbackTest, RejectsMissingMetricWithoutStateMutation) {
  auto early =
      std::make_shared<jyppx::liuding::training::callbacks::EarlyStoppingCallback>();
  jyppx::liuding::training::CallbackDispatcher dispatcher({early});
  dispatcher.dispatch(run_started());

  try {
    (void)dispatcher.dispatch(epoch_finished(2, 1));
    FAIL() << "callback failure should terminate dispatch";
  } catch (const jyppx::liuding::InternalError& error) {
    EXPECT_EQ(error.details().actual,
              "callback.early_stopping at epoch_finished");
    EXPECT_THROW(std::rethrow_if_nested(error), jyppx::liuding::InvalidArgument);
  }
  EXPECT_FALSE(early->has_best_value());
  EXPECT_EQ(early->non_improving_observations(), 0U);
  EXPECT_EQ(dispatcher.last_event_index(), 1U);
}

TEST(CallbackDispatcherTest, LaterInvalidCommandRollsBackEarlierStatefulWork) {
  using jyppx::liuding::training::CallbackCommandKind;
  std::vector<std::string> calls;
  auto early =
      std::make_shared<jyppx::liuding::training::callbacks::EarlyStoppingCallback>(
          jyppx::liuding::training::callbacks::EarlyStoppingOptions{
              .callback_id = "callback.early", .priority = 10});
  auto invalid = std::make_shared<RecordingCallback>(
      descriptor("callback.invalid", 0, CallbackCommandKind::stop),
      jyppx::liuding::training::CallbackCommandRequest{
          .kind = CallbackCommandKind::request_checkpoint,
          .reason = "undeclared"},
      &calls);
  jyppx::liuding::training::CallbackDispatcher dispatcher({invalid, early});
  dispatcher.dispatch(run_started());

  EXPECT_THROW(
      dispatcher.dispatch(epoch_finished(2, 1, {{"validation.loss", 1.0}})),
      jyppx::liuding::CompatibilityError);
  EXPECT_FALSE(early->has_best_value());
  EXPECT_EQ(early->non_improving_observations(), 0U);
  EXPECT_EQ(dispatcher.last_event_index(), 1U);
}

TEST(CallbackDispatcherTest, RegistersCompleteCallbackStateDomainAtomically) {
  using jyppx::liuding::training::callbacks::EarlyStoppingCallback;
  using jyppx::liuding::training::callbacks::EarlyStoppingOptions;
  auto existing = std::make_shared<EarlyStoppingCallback>(
      EarlyStoppingOptions{.callback_id = "callback.shared"});
  jyppx::liuding::training::CheckpointParticipantRegistry registry;
  registry.register_participant(existing);

  auto duplicate = std::make_shared<EarlyStoppingCallback>(
      EarlyStoppingOptions{.callback_id = "callback.shared"});
  auto rejected_dispatcher =
      std::make_shared<jyppx::liuding::training::CallbackDispatcher>(
          std::vector<std::shared_ptr<jyppx::liuding::training::CallbackHandler>>{
              duplicate},
          jyppx::liuding::training::CallbackDispatcherOptions{.dispatcher_id =
                                                         "callbacks.rejected"});
  EXPECT_THROW(jyppx::liuding::training::register_callback_checkpoint_participants(
                   registry, rejected_dispatcher),
               jyppx::liuding::InvalidArgument);
  EXPECT_EQ(registry.participant_ids(),
            std::vector<std::string>{"callback.shared"});

  auto accepted = std::make_shared<EarlyStoppingCallback>(
      EarlyStoppingOptions{.callback_id = "callback.accepted"});
  auto accepted_dispatcher =
      std::make_shared<jyppx::liuding::training::CallbackDispatcher>(
          std::vector<std::shared_ptr<jyppx::liuding::training::CallbackHandler>>{
              accepted},
          jyppx::liuding::training::CallbackDispatcherOptions{.dispatcher_id =
                                                         "callbacks.accepted"});
  jyppx::liuding::training::register_callback_checkpoint_participants(
      registry, accepted_dispatcher);
  EXPECT_EQ(registry.participant_ids(),
            (std::vector<std::string>{"callback.accepted", "callback.shared",
                                      "callbacks.accepted"}));
}

TEST(CallbackDispatcherTest, EnforcesRollbackBudgetBeforeHandlerMutation) {
  auto early =
      std::make_shared<jyppx::liuding::training::callbacks::EarlyStoppingCallback>();
  jyppx::liuding::training::CallbackDispatcher dispatcher(
      {early}, jyppx::liuding::training::CallbackDispatcherOptions{
                   .max_dispatch_rollback_bytes = 1});
  dispatcher.dispatch(run_started());

  EXPECT_THROW(dispatcher.dispatch(epoch_finished(2, 1, {{"val_loss", 1.0}})),
               jyppx::liuding::ResourceExhausted);
  EXPECT_FALSE(early->has_best_value());
  EXPECT_EQ(dispatcher.last_event_index(), 1U);
}

TEST(CallbackDispatcherTest, ReportsIncompleteRollbackWithOriginalCause) {
  auto callback = std::make_shared<RollbackFailingCallback>();
  jyppx::liuding::training::CallbackDispatcher dispatcher({callback});
  dispatcher.dispatch(run_started());

  try {
    (void)dispatcher.dispatch(epoch_finished(2, 1));
    FAIL() << "incomplete rollback should be terminal";
  } catch (const jyppx::liuding::InternalError& error) {
    EXPECT_EQ(error.details().operation, "dispatch.rollback");
    EXPECT_EQ(error.details().actual, "callback.rollback_failure");
    EXPECT_THROW(std::rethrow_if_nested(error), jyppx::liuding::CompatibilityError);
  }
  EXPECT_EQ(callback->value(), 1U);
  EXPECT_EQ(dispatcher.last_event_index(), 1U);
}

TEST(CallbackDispatcherTest,
     RestoresDispatcherAndStatefulCallbackWithReorderedFields) {
  using jyppx::liuding::training::callbacks::EarlyStoppingCallback;
  const auto options = jyppx::liuding::training::callbacks::EarlyStoppingOptions{
      .callback_id = "callback.resume",
      .monitor = "score",
      .patience = 1,
      .min_delta = 0.0,
      .mode = jyppx::liuding::training::callbacks::EarlyStoppingMode::maximize};
  auto source_callback = std::make_shared<EarlyStoppingCallback>(options);
  jyppx::liuding::training::CallbackDispatcher source({source_callback});
  source.dispatch(run_started());
  source.dispatch(epoch_finished(2, 1, {{"score", 0.8}}));
  source.dispatch(epoch_finished(3, 2, {{"score", 0.7}}));

  auto dispatcher_manifest = source.describe_state();
  std::reverse(dispatcher_manifest.fields.begin(),
               dispatcher_manifest.fields.end());
  MemoryStateStore dispatcher_store(std::move(dispatcher_manifest));
  source.save_state(dispatcher_store);
  auto callback_manifest = source_callback->describe_state();
  std::reverse(callback_manifest.fields.begin(),
               callback_manifest.fields.end());
  MemoryStateStore callback_store(std::move(callback_manifest));
  source_callback->save_state(callback_store);

  auto restored_callback = std::make_shared<EarlyStoppingCallback>(options);
  jyppx::liuding::training::CallbackDispatcher restored({restored_callback});
  restored_callback->load_state(callback_store,
                                jyppx::liuding::RestorePolicy::exact_only);
  restored.load_state(dispatcher_store, jyppx::liuding::RestorePolicy::exact_only);

  const auto participants = restored.checkpoint_participants();
  ASSERT_EQ(participants.size(), 1U);
  EXPECT_EQ(participants.front()->participant_id(), "callback.resume");
  EXPECT_EQ(restored.last_event_index(), 3U);
  EXPECT_EQ(restored_callback->non_improving_observations(), 1U);
  const auto result = restored.dispatch(epoch_finished(4, 3, {{"score", 0.6}}));
  ASSERT_TRUE(result.effective_command.has_value());
  EXPECT_EQ(result.effective_command->kind,
            jyppx::liuding::training::CallbackCommandKind::stop);
}

TEST(EarlyStoppingCallbackTest, ReorderedFieldsStillRejectChangedContracts) {
  using jyppx::liuding::training::callbacks::EarlyStoppingCallback;
  EarlyStoppingCallback source;
  source.handle(epoch_finished(2, 1, {{"validation.loss", 1.0}}));
  MemoryStateStore store(source.describe_state());
  source.save_state(store);
  std::reverse(store.mutable_manifest().fields.begin(),
               store.mutable_manifest().fields.end());
  const auto valid = store.manifest();

  EarlyStoppingCallback restored;
  restored.handle(epoch_finished(2, 1, {{"validation.loss", 2.0}}));
  auto& fields = store.mutable_manifest().fields;
  fields.pop_back();
  EXPECT_THROW(restored.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::CompatibilityError);
  store.mutable_manifest() = valid;
  fields.push_back({.name = "unknown", .max_bytes = 1});
  EXPECT_THROW(restored.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::CompatibilityError);
  store.mutable_manifest() = valid;
  fields.front().name = "unknown";
  EXPECT_THROW(restored.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::CompatibilityError);
  store.mutable_manifest() = valid;
  fields.front() = fields.back();
  EXPECT_THROW(restored.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::InvalidArgument);
  store.mutable_manifest() = valid;
  fields.front().media_type = "application/x-other";
  EXPECT_THROW(restored.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::CompatibilityError);
  store.mutable_manifest() = valid;
  fields.front().required = false;
  EXPECT_THROW(restored.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::CompatibilityError);
  EXPECT_DOUBLE_EQ(restored.best_value(), 2.0);

  store.mutable_manifest() = valid;
  restored.load_state(store, jyppx::liuding::RestorePolicy::exact_only);
  EXPECT_DOUBLE_EQ(restored.best_value(), 1.0);
}

TEST(EarlyStoppingCallbackTest, RejectsCorruptStateTransactionally) {
  using jyppx::liuding::training::callbacks::EarlyStoppingCallback;
  EarlyStoppingCallback source;
  source.handle(epoch_finished(2, 1, {{"validation.loss", 1.0}}));
  MemoryStateStore store(source.describe_state());
  source.save_state(store);
  store.overwrite_from_end("state", 0, std::byte{1});

  EarlyStoppingCallback restored;
  restored.handle(epoch_finished(2, 1, {{"validation.loss", 2.0}}));
  EXPECT_THROW(restored.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::SerializationError);
  EXPECT_TRUE(restored.has_best_value());
  EXPECT_DOUBLE_EQ(restored.best_value(), 2.0);
  EXPECT_FALSE(restored.should_stop());
}

}  // namespace
