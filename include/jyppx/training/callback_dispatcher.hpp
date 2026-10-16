// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_TRAINING_CALLBACK_DISPATCHER_HPP
#define JYPPX_TRAINING_CALLBACK_DISPATCHER_HPP

/**
 * @file callback_dispatcher.hpp
 * @brief Defines immutable callback events and deterministic command dispatch.
 */

#include <jyppx/foundation/checkpoint.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace jyppx::training {

/** @brief Registry receiving callback checkpoint participants atomically. */
class CheckpointParticipantRegistry;

/**
 * @brief Identifies a stable training lifecycle event.
 *
 * Numeric values are part of the callback protocol and must remain stable.
 */
enum class CallbackEventKind : std::uint8_t {
  run_started = 1,             ///< A run entered its preparation boundary.
  run_finished = 2,            ///< A run completed successfully.
  run_failed = 3,              ///< A run terminated because of an error.
  run_cancelled = 4,           ///< A run honored a cancellation request.
  epoch_started = 5,           ///< A training epoch started.
  epoch_finished = 6,          ///< A training epoch reached its safe boundary.
  batch_started = 7,           ///< A training or validation batch started.
  batch_finished = 8,          ///< A training or validation batch completed.
  optimizer_step_started = 9,  ///< An optimizer update started.
  optimizer_step_finished = 10,  ///< An optimizer update committed.
  validation_started = 11,       ///< A validation pass started.
  validation_finished = 12,      ///< A validation pass completed.
  checkpoint_started = 13,       ///< A checkpoint transaction started.
  checkpoint_committed = 14,     ///< A checkpoint transaction committed.
  checkpoint_failed = 15,        ///< A checkpoint transaction rolled back.
};

/**
 * @brief Returns the stable lowercase name of a callback event kind.
 * @param kind Event kind to render.
 * @return A process-lifetime name, or `"unknown"` for an invalid value.
 */
std::string_view callback_event_kind_name(CallbackEventKind kind) noexcept;

/**
 * @brief Identifies the lifecycle phase that owns an event.
 *
 * Numeric values are part of the callback protocol and must remain stable.
 */
enum class CallbackPhase : std::uint8_t {
  preparing = 1,      ///< Run setup before training work is accepted.
  training = 2,       ///< Training data, epochs, batches, and optimizer work.
  validation = 3,     ///< Evaluation work performed during a training run.
  checkpointing = 4,  ///< Checkpoint preparation, publication, or rollback.
  completed = 5,      ///< Successful terminal phase.
  cancelled = 6,      ///< User-requested terminal phase.
  failed = 7,         ///< Error terminal phase.
};

/**
 * @brief Returns the stable lowercase name of a callback phase.
 * @param phase Phase to render.
 * @return A process-lifetime name, or `"unknown"` for an invalid value.
 */
std::string_view callback_phase_name(CallbackPhase phase) noexcept;

/**
 * @brief Selects a bounded control request returned by a callback.
 *
 * Conflict priority is fixed as `stop > request_checkpoint > skip_validation`.
 * Numeric values are part of the callback protocol and must remain stable.
 */
enum class CallbackCommandKind : std::uint8_t {
  none = 0,             ///< Observe the event without requesting control.
  skip_validation = 1,  ///< Skip the next validation pass at a safe boundary.
  request_checkpoint = 2,  ///< Request checkpointing at a safe boundary.
  stop = 3,                ///< Stop the run at the next safe boundary.
};

/**
 * @brief Returns the stable lowercase name of a callback command kind.
 * @param kind Command kind to render.
 * @return A process-lifetime name, or `"unknown"` for an invalid value.
 */
std::string_view callback_command_kind_name(CallbackCommandKind kind) noexcept;

/** @brief Carries one immutable event snapshot to subscribed callbacks. */
struct CallbackEvent {
  std::uint32_t schema_version = 1;  ///< Positive event schema version.
  CallbackEventKind kind =
      CallbackEventKind::run_started;  ///< Stable lifecycle event identity.
  CallbackPhase phase =
      CallbackPhase::preparing;   ///< Phase that must be valid for `kind`.
  std::uint64_t event_index = 0;  ///< One-based, gap-free run event sequence.
  std::string run_id;             ///< Non-empty stable identity of the run.
  std::string trace_id;  ///< Non-empty trace context propagated to observers.
  std::uint64_t epoch = 0;        ///< Zero-based epoch cursor at this event.
  std::uint64_t global_step = 0;  ///< Committed optimizer update count.
  std::map<std::string, double>
      metrics;  ///< Finite named observations in deterministic key order.
};

/** @brief Describes one bounded command requested by a callback handler. */
struct CallbackCommandRequest {
  CallbackCommandKind kind =
      CallbackCommandKind::none;  ///< Requested action, or `none`.
  std::string reason;  ///< Non-empty diagnostic reason for a real command.
};

/** @brief Records one validated command together with its callback owner. */
struct CallbackCommand {
  CallbackCommandKind kind = CallbackCommandKind::none;  ///< Requested action.
  std::string callback_id;  ///< Stable identity of the issuing callback.
  std::string reason;       ///< Human-readable command rationale.
};

/** @brief Freezes one callback's identity, order, and allowed capabilities. */
struct CallbackDescriptor {
  std::uint32_t schema_version = 1;  ///< Positive descriptor schema version.
  std::string type_id;  ///< Stable implementation and semantic schema identity.
  std::string callback_id;    ///< Stable callback and participant identity.
  std::int32_t priority = 0;  ///< Higher values execute before lower values.
  std::vector<CallbackEventKind>
      subscriptions;  ///< Unique event kinds observed by this callback.
  std::vector<CallbackCommandKind>
      allowed_commands;   ///< Unique non-`none` commands this callback may
                          ///< issue.
  bool stateful = false;  ///< Whether the callback is a checkpoint participant.
};

/**
 * @brief Implements one immutable-event callback without writable Trainer
 * access.
 *
 * Stateful implementations must also derive from `CheckpointParticipant`.
 * Instances are invoked serially and need not provide internal synchronization.
 */
class CallbackHandler {
 public:
  /** @brief Enables destruction through the callback interface. */
  virtual ~CallbackHandler() = default;

  /**
   * @brief Returns immutable callback identity and capabilities.
   * @return A descriptor reference valid for the handler lifetime.
   */
  virtual const CallbackDescriptor& descriptor() const noexcept = 0;

  /**
   * @brief Observes one subscribed event and optionally requests a command.
   * @param event Fully validated immutable event snapshot.
   * @return A bounded command request validated by the dispatcher.
   */
  virtual CallbackCommandRequest handle(const CallbackEvent& event) = 0;
};

/** @brief Reports every issued command and the deterministic winning command.
 */
struct CallbackDispatchResult {
  std::vector<CallbackCommand>
      commands;  ///< Issued commands in deterministic callback execution order.
  std::optional<CallbackCommand>
      effective_command;  ///< Highest fixed-priority command, if one was
                          ///< issued.
};

/** @brief Configures callback and checkpoint resource limits. */
struct CallbackDispatcherOptions {
  std::string dispatcher_id =
      "callbacks.dispatcher";        ///< Stable control participant identity.
  std::size_t max_callbacks = 1024;  ///< Maximum retained callback count.
  std::size_t max_metrics_per_event =
      4096;  ///< Maximum finite metric observations on one event.
  std::size_t max_string_bytes =
      4096;  ///< Maximum bytes in an identity, trace, metric, or reason string.
  std::uint64_t max_topology_bytes =
      1024ULL * 1024ULL;  ///< Maximum canonical descriptor topology size.
  std::uint64_t max_control_bytes =
      64ULL * 1024ULL;  ///< Maximum event cursor state size.
  std::uint64_t max_dispatch_rollback_bytes =
      64ULL * 1024ULL *
      1024ULL;  ///< Maximum declared state snapshotted before one event.
};

/**
 * @brief Dispatches callbacks in stable order and checkpoints the event cursor.
 *
 * Handlers execute by descending descriptor priority and then ascending
 * callback ID. All valid requests are retained for audit. The effective request
 * uses the fixed command priority documented by `CallbackCommandKind`; callback
 * execution priority cannot override a stronger command. The dispatcher owns
 * only stream control state. Each stateful callback remains an independent
 * participant and is returned by `checkpoint_participants()` for registry
 * insertion. Before invoking an event, subscribed stateful callbacks are
 * snapshotted under an explicit byte budget. A callback exception or invalid
 * command restores every attempted stateful callback in reverse order before
 * the failure is rethrown. Stateless external effects cannot be rolled back;
 * handlers producing such effects must be idempotent or publish through the
 * later event-sink boundary.
 *
 * Instances are not internally synchronized. The training loop must serialize
 * dispatch and checkpoint I/O at a consistency boundary.
 */
class CallbackDispatcher final : public CheckpointParticipant {
 public:
  /**
   * @brief Creates a deterministic callback event boundary.
   * @param callbacks Callback handlers retained by shared ownership.
   * @param options Stable identity and positive resource limits.
   * @throws InvalidArgument If an option or descriptor is invalid or duplicate.
   * @throws ResourceExhausted If callback topology exceeds configured limits.
   */
  explicit CallbackDispatcher(
      std::vector<std::shared_ptr<CallbackHandler>> callbacks = {},
      CallbackDispatcherOptions options = {});

  /** @brief Prevents copying a live event cursor and mutable callback tree. */
  CallbackDispatcher(const CallbackDispatcher&) = delete;

  /** @brief Prevents assignment over a live dispatcher registry. */
  CallbackDispatcher& operator=(const CallbackDispatcher&) = delete;

  /** @brief Prevents moving an object retained by a participant registry. */
  CallbackDispatcher(CallbackDispatcher&&) = delete;

  /** @brief Prevents move assignment of checkpointed dispatch state. */
  CallbackDispatcher& operator=(CallbackDispatcher&&) = delete;

  /**
   * @brief Delivers one gap-free event and resolves callback commands.
   * @param event Immutable event snapshot for the current run.
   * @return Audit list and deterministic effective command.
   * @throws InvalidArgument If event order, identity, phase, or data is
   * invalid.
   * @throws CompatibilityError If a callback returns an undeclared command.
   * @throws ResourceExhausted If an event or command exceeds configured limits.
   * @throws InternalError If a handler fails or callback rollback is
   * incomplete. The original failure is retained as a nested exception.
   */
  CallbackDispatchResult dispatch(const CallbackEvent& event);

  /**
   * @brief Returns independently registered stateful callback participants.
   * @return Shared participant owners in deterministic callback order.
   */
  std::vector<std::shared_ptr<CheckpointParticipant>> checkpoint_participants()
      const;

  /**
   * @brief Returns the last successfully dispatched event index.
   * @return Zero before `run_started`, otherwise the committed event cursor.
   */
  std::uint64_t last_event_index() const noexcept;

  /**
   * @brief Returns the stable dispatcher checkpoint identity.
   * @return An owned identifier matching `describe_state()`.
   */
  std::string participant_id() const override;

  /**
   * @brief Describes canonical callback topology and event cursor fields.
   * @return A validated current-only exact participant manifest.
   */
  StateManifest describe_state() const override;

  /**
   * @brief Writes canonical callback topology and committed event cursor.
   * @param writer Storage-neutral checkpoint destination.
   */
  void save_state(StateWriter& writer) const override;

  /**
   * @brief Restores a compatible event cursor after complete preflight.
   * @param reader Storage-neutral checkpoint source.
   * @param policy Restore policy that must accept exact state.
   * @throws CompatibilityError If manifest or callback topology differs.
   * @throws SerializationError If cursor encoding is malformed.
   * @throws ResourceExhausted If configured read limits are exceeded.
   */
  void load_state(const StateReader& reader, RestorePolicy policy) override;

 private:
  CallbackDispatcherOptions options_;  ///< Immutable identity and limits.
  std::vector<std::shared_ptr<CallbackHandler>>
      callbacks_;             ///< Handlers in deterministic execution order.
  bool initialized_ = false;  ///< Whether `run_started` has committed.
  bool terminal_ = false;     ///< Whether a terminal run event has committed.
  std::string run_id_;        ///< Active run identity after initialization.
  std::uint64_t last_event_index_ = 0;  ///< Last committed global event index.
  std::uint64_t last_epoch_ = 0;        ///< Last committed epoch cursor.
  std::uint64_t last_global_step_ =
      0;  ///< Last committed optimizer step count.
};

/**
 * @brief Atomically registers a dispatcher and all stateful callback children.
 * @param registry Registry receiving the complete callback state domain.
 * @param dispatcher Non-null shared dispatcher retained by the registry.
 * @throws InvalidArgument If the dispatcher is null or any participant ID
 * conflicts with the existing registry.
 *
 * Registration has a strong guarantee: validation or duplicate failure leaves
 * the registry unchanged. The dispatcher control cursor and each stateful
 * callback are registered as independent participants in one operation.
 */
void register_callback_checkpoint_participants(
    CheckpointParticipantRegistry& registry,
    const std::shared_ptr<CallbackDispatcher>& dispatcher);

}  // namespace jyppx::training

#endif  // JYPPX_TRAINING_CALLBACK_DISPATCHER_HPP
