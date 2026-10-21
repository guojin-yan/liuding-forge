// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "detail/memory_participant_state.hpp"

#include <jyppx/foundation/error.hpp>
#include <jyppx/training/callback_dispatcher.hpp>
#include <jyppx/training/checkpoint_coordinator.hpp>
#include <jyppx/training/event_sink.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace jyppx::training {
namespace {

constexpr char kComponent[] = "training.callback_dispatcher";
constexpr char kTopologyField[] = "topology";
constexpr char kControlField[] = "control";
constexpr char kTopologyMediaType[] =
    "application/vnd.jyppx.callback-topology-v1";
constexpr char kControlMediaType[] =
    "application/vnd.jyppx.callback-control-v1";
constexpr std::string_view kTopologyMagic = "JYPPX-CALLBACK-TOPOLOGY-V1";
constexpr std::string_view kControlMagic = "JYPPX-CALLBACK-CONTROL-V1";

[[noreturn]] void throw_invalid(std::string operation, std::string expected,
                                std::string actual, std::string hint) {
  throw InvalidArgument("Invalid callback dispatch contract",
                        ErrorDetails{.component = kComponent,
                                     .operation = std::move(operation),
                                     .expected = std::move(expected),
                                     .actual = std::move(actual),
                                     .hint = std::move(hint)});
}

[[noreturn]] void throw_incompatible(std::string operation,
                                     std::string expected, std::string actual,
                                     std::string hint) {
  throw CompatibilityError("Callback checkpoint is incompatible",
                           ErrorDetails{.component = kComponent,
                                        .operation = std::move(operation),
                                        .expected = std::move(expected),
                                        .actual = std::move(actual),
                                        .hint = std::move(hint)});
}

[[noreturn]] void throw_serialization(std::string operation,
                                      std::string expected, std::string actual,
                                      std::string hint) {
  throw SerializationError("Malformed callback checkpoint state",
                           ErrorDetails{.component = kComponent,
                                        .operation = std::move(operation),
                                        .expected = std::move(expected),
                                        .actual = std::move(actual),
                                        .hint = std::move(hint)});
}

[[noreturn]] void throw_exhausted(std::string operation, std::string expected,
                                  std::string actual, std::string hint) {
  throw ResourceExhausted("Callback resource limit exceeded",
                          ErrorDetails{.component = kComponent,
                                       .operation = std::move(operation),
                                       .expected = std::move(expected),
                                       .actual = std::move(actual),
                                       .hint = std::move(hint)});
}

bool is_identifier(std::string_view value) {
  if (value.empty()) {
    return false;
  }
  return std::all_of(value.begin(), value.end(), [](char character) {
    const auto byte = static_cast<unsigned char>(character);
    return (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') ||
           (byte >= '0' && byte <= '9') || byte == '.' || byte == '_' ||
           byte == '-';
  });
}

bool valid_event_kind(CallbackEventKind kind) {
  return callback_event_kind_name(kind) != "unknown";
}

bool valid_phase(CallbackPhase phase) {
  return callback_phase_name(phase) != "unknown";
}

bool valid_command_kind(CallbackCommandKind kind) {
  return callback_command_kind_name(kind) != "unknown";
}

bool is_terminal_event(CallbackEventKind kind) {
  return kind == CallbackEventKind::run_finished ||
         kind == CallbackEventKind::run_failed ||
         kind == CallbackEventKind::run_cancelled;
}

bool phase_matches_event(CallbackEventKind kind, CallbackPhase phase) {
  switch (kind) {
    case CallbackEventKind::run_started:
      return phase == CallbackPhase::preparing;
    case CallbackEventKind::run_finished:
      return phase == CallbackPhase::completed;
    case CallbackEventKind::run_failed:
      return phase == CallbackPhase::failed;
    case CallbackEventKind::run_cancelled:
      return phase == CallbackPhase::cancelled;
    case CallbackEventKind::epoch_started:
    case CallbackEventKind::epoch_finished:
    case CallbackEventKind::optimizer_step_started:
    case CallbackEventKind::optimizer_step_finished:
      return phase == CallbackPhase::training;
    case CallbackEventKind::batch_started:
    case CallbackEventKind::batch_finished:
      return phase == CallbackPhase::training ||
             phase == CallbackPhase::validation;
    case CallbackEventKind::validation_started:
    case CallbackEventKind::validation_finished:
      return phase == CallbackPhase::validation;
    case CallbackEventKind::checkpoint_started:
    case CallbackEventKind::checkpoint_committed:
    case CallbackEventKind::checkpoint_failed:
      return phase == CallbackPhase::checkpointing;
  }
  return false;
}

void append_u8(std::vector<std::byte>& output, std::uint8_t value) {
  output.push_back(static_cast<std::byte>(value));
}

void append_u32(std::vector<std::byte>& output, std::uint32_t value) {
  for (std::size_t index = 0; index < sizeof(value); ++index) {
    output.push_back(static_cast<std::byte>((value >> (index * 8U)) & 0xffU));
  }
}

void append_u64(std::vector<std::byte>& output, std::uint64_t value) {
  for (std::size_t index = 0; index < sizeof(value); ++index) {
    output.push_back(static_cast<std::byte>((value >> (index * 8U)) & 0xffU));
  }
}

void append_i32(std::vector<std::byte>& output, std::int32_t value) {
  append_u32(output, std::bit_cast<std::uint32_t>(value));
}

void append_string(std::vector<std::byte>& output, std::string_view value) {
  append_u64(output, static_cast<std::uint64_t>(value.size()));
  for (const auto character : value) {
    output.push_back(
        static_cast<std::byte>(static_cast<unsigned char>(character)));
  }
}

class ByteReader final {
 public:
  explicit ByteReader(std::span<const std::byte> bytes) : bytes_(bytes) {}

  std::uint8_t read_u8(std::string_view field) {
    require(sizeof(std::uint8_t), field);
    return std::to_integer<std::uint8_t>(bytes_[offset_++]);
  }

  std::uint64_t read_u64(std::string_view field) {
    require(sizeof(std::uint64_t), field);
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < sizeof(value); ++index) {
      value |= static_cast<std::uint64_t>(
                   std::to_integer<std::uint8_t>(bytes_[offset_ + index]))
               << (index * 8U);
    }
    offset_ += sizeof(value);
    return value;
  }

  std::string read_string(std::string_view field, std::size_t max_bytes) {
    const auto encoded_size = read_u64(field);
    if (encoded_size > max_bytes ||
        encoded_size > static_cast<std::uint64_t>(bytes_.size() - offset_)) {
      throw_serialization(std::string(field),
                          "a bounded length-prefixed string",
                          std::to_string(encoded_size),
                          "Reject truncated or oversized callback state.");
    }
    const auto size = static_cast<std::size_t>(encoded_size);
    std::string value;
    value.reserve(size);
    for (std::size_t index = 0; index < size; ++index) {
      value.push_back(static_cast<char>(
          std::to_integer<unsigned char>(bytes_[offset_ + index])));
    }
    offset_ += size;
    return value;
  }

  void require_finished(std::string_view field) const {
    if (offset_ != bytes_.size()) {
      throw_serialization(
          std::string(field), "no trailing bytes",
          std::to_string(bytes_.size() - offset_) + " trailing bytes",
          "Reject state from an unknown schema.");
    }
  }

 private:
  void require(std::size_t count, std::string_view field) const {
    if (count > bytes_.size() - offset_) {
      throw_serialization(std::string(field), "a complete encoded value",
                          "truncated",
                          "Reject incomplete callback checkpoint state.");
    }
  }

  std::span<const std::byte> bytes_;  ///< Immutable encoded checkpoint bytes.
  std::size_t offset_ = 0;            ///< Next unread byte offset.
};

void validate_descriptor(const CallbackDescriptor& descriptor,
                         const CallbackDispatcherOptions& options) {
  if (descriptor.schema_version != 1) {
    throw_invalid("descriptor.schema_version", "1",
                  std::to_string(descriptor.schema_version),
                  "Use the current callback descriptor schema.");
  }
  if (!is_identifier(descriptor.type_id) ||
      descriptor.type_id.size() > options.max_string_bytes) {
    throw_invalid("descriptor.type_id", "a bounded stable identifier",
                  descriptor.type_id,
                  "Use ASCII letters, digits, dot, underscore, or hyphen.");
  }
  if (!is_identifier(descriptor.callback_id) ||
      descriptor.callback_id.size() > options.max_string_bytes) {
    throw_invalid("descriptor.callback_id", "a bounded stable identifier",
                  descriptor.callback_id,
                  "Use a unique callback registry identity.");
  }
  if (descriptor.subscriptions.empty()) {
    throw_invalid("descriptor.subscriptions", "at least one unique event",
                  "empty", "Declare every lifecycle event the callback uses.");
  }
  std::set<CallbackEventKind> subscriptions;
  for (const auto kind : descriptor.subscriptions) {
    if (!valid_event_kind(kind) || !subscriptions.insert(kind).second) {
      throw_invalid("descriptor.subscriptions", "unique known event kinds",
                    std::string(callback_event_kind_name(kind)),
                    "Remove unknown or duplicate subscriptions.");
    }
  }
  std::set<CallbackCommandKind> commands;
  for (const auto kind : descriptor.allowed_commands) {
    if (!valid_command_kind(kind) || kind == CallbackCommandKind::none ||
        !commands.insert(kind).second) {
      throw_invalid("descriptor.allowed_commands",
                    "unique known non-none commands",
                    std::string(callback_command_kind_name(kind)),
                    "Remove unknown, none, or duplicate capabilities.");
    }
  }
}

std::vector<std::byte> encode_topology(
    const CallbackDispatcherOptions& options,
    const std::vector<std::shared_ptr<CallbackHandler>>& callbacks) {
  std::vector<std::byte> output;
  append_string(output, kTopologyMagic);
  append_string(output, options.dispatcher_id);
  append_u64(output, static_cast<std::uint64_t>(callbacks.size()));
  for (const auto& callback : callbacks) {
    const auto& descriptor = callback->descriptor();
    append_u32(output, descriptor.schema_version);
    append_string(output, descriptor.type_id);
    append_string(output, descriptor.callback_id);
    append_i32(output, descriptor.priority);
    append_u8(output, descriptor.stateful ? 1U : 0U);
    append_u64(output,
               static_cast<std::uint64_t>(descriptor.subscriptions.size()));
    for (const auto kind : descriptor.subscriptions) {
      append_u8(output, static_cast<std::uint8_t>(kind));
    }
    append_u64(output,
               static_cast<std::uint64_t>(descriptor.allowed_commands.size()));
    for (const auto kind : descriptor.allowed_commands) {
      append_u8(output, static_cast<std::uint8_t>(kind));
    }
  }
  return output;
}

std::vector<std::byte> encode_control(bool initialized, bool terminal,
                                      std::string_view run_id,
                                      std::uint64_t event_index,
                                      std::uint64_t epoch,
                                      std::uint64_t global_step) {
  std::vector<std::byte> output;
  append_string(output, kControlMagic);
  append_u8(output, initialized ? 1U : 0U);
  append_u8(output, terminal ? 1U : 0U);
  append_u64(output, event_index);
  append_u64(output, epoch);
  append_u64(output, global_step);
  append_string(output, run_id);
  return output;
}

struct DecodedControl {
  bool initialized = false;       ///< Whether a run has started.
  bool terminal = false;          ///< Whether the run is terminal.
  std::string run_id;             ///< Restored active run identity.
  std::uint64_t event_index = 0;  ///< Restored event sequence cursor.
  std::uint64_t epoch = 0;        ///< Restored epoch cursor.
  std::uint64_t global_step = 0;  ///< Restored optimizer-step cursor.
};

DecodedControl decode_control(std::span<const std::byte> encoded,
                              std::size_t max_string_bytes) {
  ByteReader reader(encoded);
  if (reader.read_string("control.magic", kControlMagic.size()) !=
      kControlMagic) {
    throw_serialization("control.magic", std::string(kControlMagic),
                        "different magic",
                        "Restore only current callback control state.");
  }
  const auto initialized = reader.read_u8("control.initialized");
  const auto terminal = reader.read_u8("control.terminal");
  if (initialized > 1U || terminal > 1U) {
    throw_serialization("control.flags", "boolean flag values", "out of range",
                        "Reject malformed callback control flags.");
  }
  DecodedControl result{.initialized = initialized == 1U,
                        .terminal = terminal == 1U,
                        .run_id = {},
                        .event_index = reader.read_u64("control.event_index"),
                        .epoch = reader.read_u64("control.epoch"),
                        .global_step = reader.read_u64("control.global_step")};
  result.run_id = reader.read_string("control.run_id", max_string_bytes);
  reader.require_finished("control");
  if (!result.initialized) {
    if (result.terminal || !result.run_id.empty() || result.event_index != 0U ||
        result.epoch != 0U || result.global_step != 0U) {
      throw_serialization(
          "control.uninitialized", "zero cursor and empty run identity",
          "non-zero state", "Reject internally inconsistent control state.");
    }
  } else if (!is_identifier(result.run_id) || result.event_index == 0U) {
    throw_serialization(
        "control.initialized", "a stable run ID and positive event cursor",
        result.run_id, "Reject internally inconsistent control state.");
  }
  return result;
}

void validate_manifest_shape(const StateManifest& stored,
                             const StateManifest& expected,
                             RestorePolicy policy) {
  validate_state_manifest(stored);
  if (!restore_policy_accepts(policy, stored.restore_level)) {
    throw_incompatible("restore.policy", "a policy accepting exact state",
                       std::string(restore_policy_name(policy)),
                       "Use a policy that accepts exact callback state.");
  }
  if (stored.participant_id != expected.participant_id ||
      stored.schema_version != expected.schema_version ||
      stored.restore_level != RestoreLevel::exact ||
      !stored.degradation_reasons.empty() ||
      stored.fields.size() != expected.fields.size()) {
    throw_incompatible("restore.manifest", expected.participant_id,
                       stored.participant_id,
                       "Restore current exact state for this dispatcher.");
  }
  for (const auto& expected_field : expected.fields) {
    const auto stored_field =
        std::find_if(stored.fields.begin(), stored.fields.end(),
                     [&](const StateFieldDescriptor& candidate) {
                       return candidate.name == expected_field.name;
                     });
    if (stored_field == stored.fields.end() ||
        stored_field->media_type != expected_field.media_type ||
        stored_field->required != expected_field.required) {
      throw_incompatible(
          "restore.fields", expected_field.name,
          stored_field == stored.fields.end() ? "missing" : stored_field->name,
          "Restore the current dispatcher field schema.");
    }
  }
}

bool is_subscribed(const CallbackDescriptor& descriptor,
                   CallbackEventKind kind) {
  return std::find(descriptor.subscriptions.begin(),
                   descriptor.subscriptions.end(),
                   kind) != descriptor.subscriptions.end();
}

bool command_is_allowed(const CallbackDescriptor& descriptor,
                        CallbackCommandKind kind) {
  return std::find(descriptor.allowed_commands.begin(),
                   descriptor.allowed_commands.end(),
                   kind) != descriptor.allowed_commands.end();
}

int command_priority(CallbackCommandKind kind) {
  switch (kind) {
    case CallbackCommandKind::none:
      return 0;
    case CallbackCommandKind::skip_validation:
      return 1;
    case CallbackCommandKind::request_checkpoint:
      return 2;
    case CallbackCommandKind::stop:
      return 3;
  }
  return -1;
}

struct CallbackStateSnapshot {
  std::string callback_id;
  std::shared_ptr<CheckpointParticipant> participant;
  std::unique_ptr<detail::MemoryParticipantState> state;
};

std::vector<CallbackStateSnapshot> capture_callback_state(
    const CallbackEvent& event,
    const std::vector<std::shared_ptr<CallbackHandler>>& callbacks,
    std::uint64_t max_rollback_bytes) {
  std::vector<CallbackStateSnapshot> snapshots;
  std::uint64_t declared_bytes = 0;
  for (const auto& callback : callbacks) {
    const auto& descriptor = callback->descriptor();
    if (!descriptor.stateful || !is_subscribed(descriptor, event.kind)) {
      continue;
    }
    auto participant =
        std::dynamic_pointer_cast<CheckpointParticipant>(callback);
    if (!participant) {
      throw InternalError(
          "Stateful callback lost its checkpoint participant interface",
          ErrorDetails{.component = kComponent,
                       .operation = "dispatch.snapshot",
                       .expected = descriptor.callback_id,
                       .actual = "non-participant handler",
                       .hint = "Keep callback descriptor identity immutable."});
    }
    validate_checkpoint_participant(*participant);
    auto manifest = participant->describe_state();
    for (const auto& field : manifest.fields) {
      if (field.max_bytes > max_rollback_bytes - declared_bytes) {
        throw_exhausted(
            "dispatch.rollback_budget", std::to_string(max_rollback_bytes),
            "more than configured",
            "Reduce callback state bounds or raise the explicit budget.");
      }
      declared_bytes += field.max_bytes;
    }
    auto state = std::make_unique<detail::MemoryParticipantState>(
        std::move(manifest), kComponent, "dispatch.snapshot");
    participant->save_state(*state);
    state->finalize();
    snapshots.push_back({.callback_id = descriptor.callback_id,
                         .participant = std::move(participant),
                         .state = std::move(state)});
  }
  return snapshots;
}

[[noreturn]] void rethrow_callback_failure(const CallbackDescriptor& descriptor,
                                           const CallbackEvent& event) {
  try {
    throw;
  } catch (...) {
    std::throw_with_nested(InternalError(
        "Callback handler failed",
        ErrorDetails{
            .component = kComponent,
            .operation = "dispatch.handle",
            .expected = "successful handling or one declared command",
            .actual = descriptor.callback_id + " at " +
                      std::string(callback_event_kind_name(event.kind)),
            .hint = "Inspect the nested callback failure and abort the run."}));
  }
}

[[noreturn]] void rollback_and_rethrow(
    const std::vector<CallbackStateSnapshot>& snapshots,
    std::size_t attempted_stateful_callbacks) {
  const auto dispatch_failure = std::current_exception();
  std::string rollback_failure_id;
  for (std::size_t offset = 0; offset < attempted_stateful_callbacks;
       ++offset) {
    const auto index = attempted_stateful_callbacks - offset - 1U;
    try {
      snapshots[index].participant->load_state(
          *snapshots[index].state, RestorePolicy::allow_best_effort);
    } catch (...) {
      if (rollback_failure_id.empty()) {
        rollback_failure_id = snapshots[index].callback_id;
      }
    }
  }
  if (!rollback_failure_id.empty()) {
    try {
      std::rethrow_exception(dispatch_failure);
    } catch (...) {
      std::throw_with_nested(InternalError(
          "Callback dispatch failed and rollback was incomplete",
          ErrorDetails{.component = kComponent,
                       .operation = "dispatch.rollback",
                       .expected = "all attempted stateful callbacks restored",
                       .actual = rollback_failure_id,
                       .hint = "Stop training because callback state is "
                               "indeterminate."}));
    }
  }
  std::rethrow_exception(dispatch_failure);
}

}  // namespace

std::string_view callback_event_kind_name(CallbackEventKind kind) noexcept {
  switch (kind) {
    case CallbackEventKind::run_started:
      return "run_started";
    case CallbackEventKind::run_finished:
      return "run_finished";
    case CallbackEventKind::run_failed:
      return "run_failed";
    case CallbackEventKind::run_cancelled:
      return "run_cancelled";
    case CallbackEventKind::epoch_started:
      return "epoch_started";
    case CallbackEventKind::epoch_finished:
      return "epoch_finished";
    case CallbackEventKind::batch_started:
      return "batch_started";
    case CallbackEventKind::batch_finished:
      return "batch_finished";
    case CallbackEventKind::optimizer_step_started:
      return "optimizer_step_started";
    case CallbackEventKind::optimizer_step_finished:
      return "optimizer_step_finished";
    case CallbackEventKind::validation_started:
      return "validation_started";
    case CallbackEventKind::validation_finished:
      return "validation_finished";
    case CallbackEventKind::checkpoint_started:
      return "checkpoint_started";
    case CallbackEventKind::checkpoint_committed:
      return "checkpoint_committed";
    case CallbackEventKind::checkpoint_failed:
      return "checkpoint_failed";
  }
  return "unknown";
}

std::string_view callback_phase_name(CallbackPhase phase) noexcept {
  switch (phase) {
    case CallbackPhase::preparing:
      return "preparing";
    case CallbackPhase::training:
      return "training";
    case CallbackPhase::validation:
      return "validation";
    case CallbackPhase::checkpointing:
      return "checkpointing";
    case CallbackPhase::completed:
      return "completed";
    case CallbackPhase::cancelled:
      return "cancelled";
    case CallbackPhase::failed:
      return "failed";
  }
  return "unknown";
}

std::string_view callback_command_kind_name(CallbackCommandKind kind) noexcept {
  switch (kind) {
    case CallbackCommandKind::none:
      return "none";
    case CallbackCommandKind::skip_validation:
      return "skip_validation";
    case CallbackCommandKind::request_checkpoint:
      return "request_checkpoint";
    case CallbackCommandKind::stop:
      return "stop";
  }
  return "unknown";
}

CallbackDispatcher::CallbackDispatcher(
    std::vector<std::shared_ptr<CallbackHandler>> callbacks,
    CallbackDispatcherOptions options)
    : options_(std::move(options)), callbacks_(std::move(callbacks)) {
  if (!is_identifier(options_.dispatcher_id) ||
      options_.dispatcher_id.size() > options_.max_string_bytes) {
    throw_invalid("options.dispatcher_id", "a bounded stable identifier",
                  options_.dispatcher_id,
                  "Use a unique callback control participant ID.");
  }
  if (options_.max_callbacks == 0U || options_.max_metrics_per_event == 0U ||
      options_.max_string_bytes == 0U || options_.max_topology_bytes == 0U ||
      options_.max_control_bytes == 0U ||
      options_.max_dispatch_rollback_bytes == 0U) {
    throw_invalid("options.limits", "positive resource limits", "zero",
                  "Configure explicit positive callback limits.");
  }
  if (callbacks_.size() > options_.max_callbacks) {
    throw_exhausted("callbacks", std::to_string(options_.max_callbacks),
                    std::to_string(callbacks_.size()),
                    "Reduce callback count or raise the explicit limit.");
  }

  std::set<std::string> callback_ids;
  for (const auto& callback : callbacks_) {
    if (!callback) {
      throw_invalid("callbacks", "non-null callback handlers", "null",
                    "Remove null entries before dispatcher construction.");
    }
    const auto& descriptor = callback->descriptor();
    validate_descriptor(descriptor, options_);
    if (!callback_ids.insert(descriptor.callback_id).second) {
      throw_invalid("callbacks.callback_id", "unique callback identities",
                    descriptor.callback_id,
                    "Assign a stable unique ID to every callback instance.");
    }
    const auto participant =
        std::dynamic_pointer_cast<CheckpointParticipant>(callback);
    if (descriptor.stateful != static_cast<bool>(participant)) {
      throw_invalid("callbacks.stateful",
                    "stateful iff the handler is a checkpoint participant",
                    descriptor.callback_id,
                    "Make state ownership explicit in the descriptor.");
    }
    if (participant) {
      validate_checkpoint_participant(*participant);
      if (participant->participant_id() != descriptor.callback_id) {
        throw_invalid("callbacks.participant_id", descriptor.callback_id,
                      participant->participant_id(),
                      "Use one stable identity for callback and state.");
      }
    }
  }
  std::sort(callbacks_.begin(), callbacks_.end(),
            [](const auto& left, const auto& right) {
              const auto& left_descriptor = left->descriptor();
              const auto& right_descriptor = right->descriptor();
              if (left_descriptor.priority != right_descriptor.priority) {
                return left_descriptor.priority > right_descriptor.priority;
              }
              return left_descriptor.callback_id < right_descriptor.callback_id;
            });
  const auto topology = encode_topology(options_, callbacks_);
  if (topology.size() > options_.max_topology_bytes) {
    throw_exhausted("topology", std::to_string(options_.max_topology_bytes),
                    std::to_string(topology.size()),
                    "Reduce descriptor size or raise the explicit limit.");
  }
}

CallbackDispatchResult CallbackDispatcher::dispatch(
    const CallbackEvent& event) {
  return dispatch_impl(event, nullptr);
}

CallbackDispatchResult CallbackDispatcher::dispatch(const CallbackEvent& event,
                                                    EventOutbox& outbox) {
  return dispatch_impl(event, &outbox);
}

CallbackDispatchResult CallbackDispatcher::dispatch_impl(
    const CallbackEvent& event, EventOutbox* outbox) {
  if (event.schema_version != 1U || !valid_event_kind(event.kind) ||
      !valid_phase(event.phase)) {
    throw_invalid("event.schema", "current known event schema", "unknown",
                  "Publish schema v1 with known event and phase enums.");
  }
  if (!phase_matches_event(event.kind, event.phase)) {
    throw_invalid("event.phase",
                  "a phase compatible with " +
                      std::string(callback_event_kind_name(event.kind)),
                  std::string(callback_phase_name(event.phase)),
                  "Publish lifecycle events from their owning phase.");
  }
  if (!is_identifier(event.run_id) ||
      event.run_id.size() > options_.max_string_bytes) {
    throw_invalid("event.run_id", "a bounded stable identifier", event.run_id,
                  "Assign one stable run identity before publishing events.");
  }
  if (event.trace_id.empty() ||
      event.trace_id.size() > options_.max_string_bytes) {
    throw_invalid("event.trace_id", "a non-empty bounded trace identity",
                  event.trace_id,
                  "Propagate trace context with every callback event.");
  }
  if (event.metrics.size() > options_.max_metrics_per_event) {
    throw_exhausted("event.metrics",
                    std::to_string(options_.max_metrics_per_event),
                    std::to_string(event.metrics.size()),
                    "Reduce event observations or raise the explicit limit.");
  }
  for (const auto& [name, value] : event.metrics) {
    if (!is_identifier(name) || name.size() > options_.max_string_bytes ||
        !std::isfinite(value)) {
      throw_invalid("event.metrics", "bounded names with finite values", name,
                    "Validate metric observations before event publication.");
    }
  }
  if (!initialized_) {
    if (event.kind != CallbackEventKind::run_started ||
        event.event_index != 1U) {
      throw_invalid("event.start", "run_started with event_index 1",
                    std::string(callback_event_kind_name(event.kind)) + "@" +
                        std::to_string(event.event_index),
                    "Begin each dispatcher stream with one run_started event.");
    }
  } else {
    if (terminal_) {
      throw_invalid("event.terminal", "no event after a terminal event",
                    std::string(callback_event_kind_name(event.kind)),
                    "Create a new dispatcher for a new run.");
    }
    if (event.run_id != run_id_) {
      throw_invalid("event.run_id", run_id_, event.run_id,
                    "Do not mix multiple runs in one dispatcher stream.");
    }
    if (last_event_index_ == std::numeric_limits<std::uint64_t>::max()) {
      throw_exhausted("event.event_index", "a sequence below uint64 maximum",
                      "uint64 maximum", "Create a new run identity.");
    }
    const auto expected_index = last_event_index_ + 1U;
    if (event.event_index != expected_index) {
      throw_invalid("event.event_index", std::to_string(expected_index),
                    std::to_string(event.event_index),
                    "Reject duplicate or skipped callback events.");
    }
    if (event.epoch < last_epoch_ || event.global_step < last_global_step_) {
      throw_invalid(
          "event.cursor", "non-decreasing epoch and global step",
          std::to_string(event.epoch) + "/" + std::to_string(event.global_step),
          "Publish events only after cursor ownership is committed.");
    }
  }

  auto next_run_id = event.run_id;
  const auto snapshots = capture_callback_state(
      event, callbacks_, options_.max_dispatch_rollback_bytes);
  std::size_t attempted_stateful_callbacks = 0;
  CallbackDispatchResult result;
  try {
    for (const auto& callback : callbacks_) {
      const auto& descriptor = callback->descriptor();
      if (!is_subscribed(descriptor, event.kind)) {
        continue;
      }
      if (descriptor.stateful) {
        if (attempted_stateful_callbacks >= snapshots.size() ||
            snapshots[attempted_stateful_callbacks].callback_id !=
                descriptor.callback_id) {
          throw InternalError(
              "Callback snapshot order changed during dispatch",
              ErrorDetails{.component = kComponent,
                           .operation = "dispatch.snapshot_order",
                           .expected = descriptor.callback_id,
                           .actual = "different callback topology",
                           .hint = "Keep descriptors immutable after "
                                   "dispatcher construction."});
        }
        ++attempted_stateful_callbacks;
      }
      CallbackCommandRequest request;
      try {
        request = callback->handle(event);
      } catch (...) {
        rethrow_callback_failure(descriptor, event);
      }
      if (!valid_command_kind(request.kind)) {
        throw_incompatible("command.kind", "a known declared command",
                           "unknown",
                           "Return only commands from the stable protocol.");
      }
      if (request.kind == CallbackCommandKind::none) {
        if (!request.reason.empty()) {
          throw_incompatible("command.reason", "empty reason for none",
                             request.reason,
                             "Attach reasons only to real command requests.");
        }
        continue;
      }
      if (!command_is_allowed(descriptor, request.kind)) {
        throw_incompatible(
            "command.capability",
            "a command declared by " + descriptor.callback_id,
            std::string(callback_command_kind_name(request.kind)),
            "Freeze callback capabilities in its descriptor.");
      }
      if (request.reason.empty() ||
          request.reason.size() > options_.max_string_bytes) {
        throw_incompatible("command.reason", "a non-empty bounded reason",
                           request.reason,
                           "Return an auditable reason for each command.");
      }
      CallbackCommand command{.kind = request.kind,
                              .callback_id = descriptor.callback_id,
                              .reason = std::move(request.reason)};
      if (!result.effective_command.has_value() ||
          command_priority(command.kind) >
              command_priority(result.effective_command->kind)) {
        result.effective_command = command;
      }
      result.commands.push_back(std::move(command));
    }
    if (outbox != nullptr) {
      outbox->enqueue(event, result);
    }
  } catch (...) {
    rollback_and_rethrow(snapshots, attempted_stateful_callbacks);
  }

  initialized_ = true;
  terminal_ = is_terminal_event(event.kind);
  run_id_ = std::move(next_run_id);
  last_event_index_ = event.event_index;
  last_epoch_ = event.epoch;
  last_global_step_ = event.global_step;
  return result;
}

std::vector<std::shared_ptr<CheckpointParticipant>>
CallbackDispatcher::checkpoint_participants() const {
  std::vector<std::shared_ptr<CheckpointParticipant>> result;
  for (const auto& callback : callbacks_) {
    if (callback->descriptor().stateful) {
      result.push_back(
          std::dynamic_pointer_cast<CheckpointParticipant>(callback));
    }
  }
  return result;
}

std::uint64_t CallbackDispatcher::last_event_index() const noexcept {
  return last_event_index_;
}

std::string CallbackDispatcher::participant_id() const {
  return options_.dispatcher_id;
}

StateManifest CallbackDispatcher::describe_state() const {
  return {.participant_id = options_.dispatcher_id,
          .schema_version = 1,
          .restore_level = RestoreLevel::exact,
          .fields = {{.name = kTopologyField,
                      .media_type = kTopologyMediaType,
                      .max_bytes = options_.max_topology_bytes},
                     {.name = kControlField,
                      .media_type = kControlMediaType,
                      .max_bytes = options_.max_control_bytes}},
          .degradation_reasons = {}};
}

void CallbackDispatcher::save_state(StateWriter& writer) const {
  const auto topology = encode_topology(options_, callbacks_);
  const auto control =
      encode_control(initialized_, terminal_, run_id_, last_event_index_,
                     last_epoch_, last_global_step_);
  if (topology.size() > options_.max_topology_bytes ||
      control.size() > options_.max_control_bytes) {
    throw_exhausted("save_state", "state within configured limits", "oversized",
                    "Raise explicit limits before dispatcher construction.");
  }
  writer.write_blob(kTopologyField, std::span<const std::byte>(
                                        topology.data(), topology.size()));
  writer.write_blob(kControlField,
                    std::span<const std::byte>(control.data(), control.size()));
}

void CallbackDispatcher::load_state(const StateReader& reader,
                                    RestorePolicy policy) {
  const auto expected_manifest = describe_state();
  validate_manifest_shape(reader.manifest(), expected_manifest, policy);
  const auto stored_topology =
      reader.read_blob(kTopologyField, options_.max_topology_bytes);
  const auto expected_topology = encode_topology(options_, callbacks_);
  if (stored_topology.size() != expected_topology.size() ||
      !std::equal(stored_topology.begin(), stored_topology.end(),
                  expected_topology.begin())) {
    throw_incompatible("restore.topology", "identical callback descriptors",
                       "different canonical topology",
                       "Restore with the same ordered callback contracts.");
  }
  const auto candidate = decode_control(
      reader.read_blob(kControlField, options_.max_control_bytes),
      options_.max_string_bytes);
  initialized_ = candidate.initialized;
  terminal_ = candidate.terminal;
  run_id_ = candidate.run_id;
  last_event_index_ = candidate.event_index;
  last_epoch_ = candidate.epoch;
  last_global_step_ = candidate.global_step;
}

void register_callback_checkpoint_participants(
    CheckpointParticipantRegistry& registry,
    const std::shared_ptr<CallbackDispatcher>& dispatcher) {
  if (!dispatcher) {
    throw_invalid("registry.dispatcher", "a non-null shared dispatcher", "null",
                  "Construct the callback state domain first.");
  }
  std::vector<std::shared_ptr<CheckpointParticipant>> participants;
  auto children = dispatcher->checkpoint_participants();
  participants.reserve(children.size() + 1U);
  participants.push_back(
      std::static_pointer_cast<CheckpointParticipant>(dispatcher));
  participants.insert(participants.end(), children.begin(), children.end());
  registry.register_participants(std::move(participants));
}

}  // namespace jyppx::training
