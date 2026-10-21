// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/training/event_sink.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace jyppx::training {
namespace {

constexpr char kComponent[] = "training.event_outbox";
constexpr char kTopologyField[] = "topology";
constexpr char kControlField[] = "control";
constexpr char kTopologyMediaType[] =
    "application/vnd.jyppx.event-outbox-topology-v1";
constexpr char kControlMediaType[] =
    "application/vnd.jyppx.event-outbox-control-v1";
constexpr std::string_view kTopologyMagic = "JYPPX-EVENT-OUTBOX-TOPOLOGY-V1";
constexpr std::string_view kControlMagic = "JYPPX-EVENT-OUTBOX-CONTROL-V1";

[[noreturn]] void throw_invalid(std::string operation, std::string expected,
                                std::string actual, std::string hint) {
  throw InvalidArgument("Invalid event delivery contract",
                        ErrorDetails{.component = kComponent,
                                     .operation = std::move(operation),
                                     .expected = std::move(expected),
                                     .actual = std::move(actual),
                                     .hint = std::move(hint)});
}

[[noreturn]] void throw_incompatible(std::string operation,
                                     std::string expected, std::string actual,
                                     std::string hint) {
  throw CompatibilityError("Event outbox state is incompatible",
                           ErrorDetails{.component = kComponent,
                                        .operation = std::move(operation),
                                        .expected = std::move(expected),
                                        .actual = std::move(actual),
                                        .hint = std::move(hint)});
}

[[noreturn]] void throw_serialization(std::string operation,
                                      std::string expected, std::string actual,
                                      std::string hint) {
  throw SerializationError("Malformed event outbox state",
                           ErrorDetails{.component = kComponent,
                                        .operation = std::move(operation),
                                        .expected = std::move(expected),
                                        .actual = std::move(actual),
                                        .hint = std::move(hint)});
}

[[noreturn]] void throw_exhausted(std::string operation, std::string expected,
                                  std::string actual, std::string hint) {
  throw ResourceExhausted("Event outbox resource limit exceeded",
                          ErrorDetails{.component = kComponent,
                                       .operation = std::move(operation),
                                       .expected = std::move(expected),
                                       .actual = std::move(actual),
                                       .hint = std::move(hint)});
}

[[noreturn]] void throw_cancelled(std::string operation) {
  throw Cancelled("Event delivery was cancelled",
                  ErrorDetails{.component = kComponent,
                               .operation = std::move(operation),
                               .expected = "operation allowed to continue",
                               .actual = "stop requested",
                               .hint = "Retry or close the outbox from a safe "
                                       "lifecycle boundary."});
}

void check_stop(const std::stop_token& stop, std::string operation) {
  if (stop.stop_requested()) {
    throw_cancelled(std::move(operation));
  }
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

bool is_valid_utf8(std::string_view value) {
  std::size_t index = 0;
  while (index < value.size()) {
    const auto first = static_cast<unsigned char>(value[index]);
    if (first <= 0x7fU) {
      ++index;
      continue;
    }
    std::size_t trailing = 0;
    std::uint32_t code_point = 0;
    if ((first & 0xe0U) == 0xc0U) {
      trailing = 1;
      code_point = first & 0x1fU;
    } else if ((first & 0xf0U) == 0xe0U) {
      trailing = 2;
      code_point = first & 0x0fU;
    } else if ((first & 0xf8U) == 0xf0U) {
      trailing = 3;
      code_point = first & 0x07U;
    } else {
      return false;
    }
    if (trailing > value.size() - index - 1U) {
      return false;
    }
    for (std::size_t offset = 1; offset <= trailing; ++offset) {
      const auto byte = static_cast<unsigned char>(value[index + offset]);
      if ((byte & 0xc0U) != 0x80U) {
        return false;
      }
      code_point = (code_point << 6U) | (byte & 0x3fU);
    }
    const auto minimum = trailing == 1U   ? 0x80U
                         : trailing == 2U ? 0x800U
                                          : 0x10000U;
    if (code_point < minimum || code_point > 0x10ffffU ||
        (code_point >= 0xd800U && code_point <= 0xdfffU)) {
      return false;
    }
    index += trailing + 1U;
  }
  return true;
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

bool is_terminal_event(CallbackEventKind kind) {
  return kind == CallbackEventKind::run_finished ||
         kind == CallbackEventKind::run_failed ||
         kind == CallbackEventKind::run_cancelled;
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

bool same_command(const CallbackCommand& left, const CallbackCommand& right) {
  return left.kind == right.kind && left.callback_id == right.callback_id &&
         left.reason == right.reason;
}

void validate_options(const EventOutboxOptions& options) {
  if (!is_identifier(options.outbox_id) || options.max_sinks == 0 ||
      options.max_pending_events == 0 || options.max_metrics_per_event == 0 ||
      options.max_commands_per_event == 0 || options.max_string_bytes == 0 ||
      options.max_event_bytes == 0 || options.max_pending_bytes == 0 ||
      options.max_topology_bytes == 0 || options.max_control_bytes == 0) {
    throw_invalid("options", "a stable ID and positive resource limits",
                  options.outbox_id,
                  "Configure explicit nonzero event delivery bounds.");
  }
  if (options.outbox_id.size() > options.max_string_bytes) {
    throw_exhausted("options.outbox_id",
                    std::to_string(options.max_string_bytes),
                    std::to_string(options.outbox_id.size()),
                    "Use a shorter stable outbox identity.");
  }
  constexpr std::uint64_t kControlBaseOverhead = 4096U;
  constexpr std::uint64_t kControlNodeOverheadBeyondQueueCharge = 8U;
  if (options.max_pending_events >
      (std::numeric_limits<std::uint64_t>::max() - kControlBaseOverhead) /
          kControlNodeOverheadBeyondQueueCharge) {
    throw_invalid("options.max_pending_events", "a representable state bound",
                  std::to_string(options.max_pending_events),
                  "Reduce the explicit pending-event limit.");
  }
  const auto control_overhead =
      kControlBaseOverhead +
      static_cast<std::uint64_t>(options.max_pending_events) *
          kControlNodeOverheadBeyondQueueCharge;
  if (options.max_pending_bytes >
          std::numeric_limits<std::uint64_t>::max() - control_overhead ||
      options.max_control_bytes <
          options.max_pending_bytes + control_overhead) {
    throw_invalid("options.max_control_bytes",
                  "pending bytes plus bounded node and control metadata",
                  std::to_string(options.max_control_bytes),
                  "Reserve checkpoint metadata in addition to event bytes.");
  }
}

void validate_sink_descriptor(const EventSinkDescriptor& descriptor,
                              const EventOutboxOptions& options) {
  if (descriptor.schema_version != 1U ||
      event_delivery_guarantee_name(descriptor.delivery_guarantee) ==
          "unknown" ||
      !is_identifier(descriptor.sink_id) || descriptor.type_id.empty() ||
      descriptor.destination_id.empty()) {
    throw_invalid("sink.descriptor", "current complete sink descriptor",
                  descriptor.sink_id,
                  "Declare stable sink, type, destination, and delivery IDs.");
  }
  if (descriptor.sink_id.size() > options.max_string_bytes ||
      descriptor.type_id.size() > options.max_string_bytes ||
      descriptor.destination_id.size() > options.max_string_bytes) {
    throw_exhausted("sink.descriptor", std::to_string(options.max_string_bytes),
                    "oversized string",
                    "Use bounded sink descriptor identities.");
  }
}

void validate_event_shape(const CallbackEvent& event,
                          const EventOutboxOptions& options) {
  if (event.schema_version != 1U ||
      callback_event_kind_name(event.kind) == "unknown" ||
      callback_phase_name(event.phase) == "unknown" ||
      !phase_matches_event(event.kind, event.phase)) {
    throw_invalid("event.schema", "current event schema and owning phase",
                  std::string(callback_event_kind_name(event.kind)),
                  "Publish a callback-validated lifecycle event.");
  }
  if (!is_identifier(event.run_id) || event.trace_id.empty() ||
      !is_valid_utf8(event.trace_id) ||
      event.run_id.size() > options.max_string_bytes ||
      event.trace_id.size() > options.max_string_bytes) {
    throw_invalid("event.identity", "bounded run and trace identities",
                  event.run_id,
                  "Propagate the callback event identity unchanged.");
  }
  if (event.metrics.size() > options.max_metrics_per_event) {
    throw_exhausted("event.metrics",
                    std::to_string(options.max_metrics_per_event),
                    std::to_string(event.metrics.size()),
                    "Reduce event observations or raise the explicit limit.");
  }
  for (const auto& [name, value] : event.metrics) {
    if (!is_identifier(name) || name.size() > options.max_string_bytes ||
        !std::isfinite(value)) {
      throw_invalid("event.metrics", "bounded names with finite values", name,
                    "Validate observations before event publication.");
    }
  }
}

void validate_callback_result(const CallbackDispatchResult& result,
                              const EventOutboxOptions& options) {
  if (result.commands.size() > options.max_commands_per_event) {
    throw_exhausted("event.commands",
                    std::to_string(options.max_commands_per_event),
                    std::to_string(result.commands.size()),
                    "Reduce callback count or raise the explicit limit.");
  }
  std::set<std::string> callback_ids;
  const CallbackCommand* expected_effective = nullptr;
  for (const auto& command : result.commands) {
    if (command.kind == CallbackCommandKind::none ||
        callback_command_kind_name(command.kind) == "unknown" ||
        !is_identifier(command.callback_id) || command.reason.empty() ||
        command.callback_id.size() > options.max_string_bytes ||
        command.reason.size() > options.max_string_bytes ||
        !is_valid_utf8(command.reason) ||
        !callback_ids.insert(command.callback_id).second) {
      throw_incompatible(
          "event.commands", "unique bounded non-none callback commands",
          command.callback_id,
          "Forward the exact dispatcher result without modification.");
    }
    if (expected_effective == nullptr ||
        command_priority(command.kind) >
            command_priority(expected_effective->kind)) {
      expected_effective = &command;
    }
  }
  if ((expected_effective == nullptr) !=
      (!result.effective_command.has_value())) {
    throw_incompatible("event.effective_command",
                       "presence matching issued commands", "mismatch",
                       "Forward the exact dispatcher result.");
  }
  if (expected_effective != nullptr &&
      !same_command(*expected_effective, *result.effective_command)) {
    throw_incompatible("event.effective_command",
                       "the first strongest issued command",
                       result.effective_command->callback_id,
                       "Do not recompute callback command priority.");
  }
}

struct StreamState {
  bool initialized = false;
  bool terminal = false;
  std::string run_id;
  std::uint64_t last_event_index = 0;
  std::uint64_t last_epoch = 0;
  std::uint64_t last_global_step = 0;
};

StreamState validate_transition(const CallbackEvent& event,
                                const StreamState& current) {
  auto next = current;
  if (!current.initialized) {
    if (event.kind != CallbackEventKind::run_started ||
        event.event_index != 1U) {
      throw_invalid("event.start", "run_started with event_index 1",
                    std::string(callback_event_kind_name(event.kind)) + "@" +
                        std::to_string(event.event_index),
                    "Begin each outbox stream with one run_started event.");
    }
  } else {
    if (current.terminal) {
      throw_invalid("event.terminal", "no event after a terminal event",
                    std::string(callback_event_kind_name(event.kind)),
                    "Create a new outbox for a new run.");
    }
    if (event.run_id != current.run_id) {
      throw_invalid("event.run_id", current.run_id, event.run_id,
                    "Do not mix runs in one event outbox.");
    }
    if (current.last_event_index == std::numeric_limits<std::uint64_t>::max()) {
      throw_exhausted("event.event_index", "below uint64 maximum",
                      "uint64 maximum", "Create a new run and outbox.");
    }
    const auto expected = current.last_event_index + 1U;
    if (event.event_index != expected) {
      throw_invalid("event.event_index", std::to_string(expected),
                    std::to_string(event.event_index),
                    "Reject duplicate or skipped event delivery.");
    }
    if (event.epoch < current.last_epoch ||
        event.global_step < current.last_global_step) {
      throw_invalid(
          "event.cursor", "non-decreasing epoch and global step",
          std::to_string(event.epoch) + "/" + std::to_string(event.global_step),
          "Enqueue events only after training state commits.");
    }
  }
  next.initialized = true;
  next.terminal = is_terminal_event(event.kind);
  next.run_id = event.run_id;
  next.last_event_index = event.event_index;
  next.last_epoch = event.epoch;
  next.last_global_step = event.global_step;
  return next;
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

void append_double(std::vector<std::byte>& output, double value) {
  append_u64(output, std::bit_cast<std::uint64_t>(value));
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

  std::uint32_t read_u32(std::string_view field) {
    require(sizeof(std::uint32_t), field);
    std::uint32_t value = 0;
    for (std::size_t index = 0; index < sizeof(value); ++index) {
      value |= static_cast<std::uint32_t>(
                   std::to_integer<std::uint8_t>(bytes_[offset_ + index]))
               << (index * 8U);
    }
    offset_ += sizeof(value);
    return value;
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

  double read_double(std::string_view field) {
    return std::bit_cast<double>(read_u64(field));
  }

  std::string read_string(std::string_view field, std::size_t max_bytes) {
    const auto encoded_size = read_u64(field);
    if (encoded_size > max_bytes ||
        encoded_size > static_cast<std::uint64_t>(bytes_.size() - offset_)) {
      throw_serialization(std::string(field),
                          "a bounded length-prefixed string",
                          std::to_string(encoded_size),
                          "Reject truncated or oversized outbox state.");
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
                          "truncated", "Reject incomplete outbox state.");
    }
  }

  std::span<const std::byte> bytes_;  ///< Borrowed complete field bytes.
  std::size_t offset_ = 0;            ///< Next unread byte position.
};

void encode_command(std::vector<std::byte>& output,
                    const CallbackCommand& command) {
  append_u8(output, static_cast<std::uint8_t>(command.kind));
  append_string(output, command.callback_id);
  append_string(output, command.reason);
}

CallbackCommand decode_command(ByteReader& reader,
                               const EventOutboxOptions& options) {
  return {
      .kind = static_cast<CallbackCommandKind>(reader.read_u8("command.kind")),
      .callback_id =
          reader.read_string("command.callback_id", options.max_string_bytes),
      .reason = reader.read_string("command.reason", options.max_string_bytes)};
}

std::vector<std::byte> encode_envelope(const EventEnvelope& envelope) {
  std::vector<std::byte> output;
  append_u32(output, envelope.schema_version);
  append_u32(output, envelope.event.schema_version);
  append_u8(output, static_cast<std::uint8_t>(envelope.event.kind));
  append_u8(output, static_cast<std::uint8_t>(envelope.event.phase));
  append_u64(output, envelope.event.event_index);
  append_string(output, envelope.event.run_id);
  append_string(output, envelope.event.trace_id);
  append_u64(output, envelope.event.epoch);
  append_u64(output, envelope.event.global_step);
  append_u64(output, static_cast<std::uint64_t>(envelope.event.metrics.size()));
  for (const auto& [name, value] : envelope.event.metrics) {
    append_string(output, name);
    append_double(output, value);
  }
  append_u64(output, static_cast<std::uint64_t>(
                         envelope.callback_result.commands.size()));
  for (const auto& command : envelope.callback_result.commands) {
    encode_command(output, command);
  }
  append_u8(output,
            envelope.callback_result.effective_command.has_value() ? 1U : 0U);
  if (envelope.callback_result.effective_command.has_value()) {
    encode_command(output, *envelope.callback_result.effective_command);
  }
  return output;
}

EventEnvelope decode_envelope(std::span<const std::byte> bytes,
                              const EventOutboxOptions& options) {
  ByteReader reader(bytes);
  EventEnvelope envelope;
  envelope.schema_version = reader.read_u32("envelope.schema_version");
  envelope.event.schema_version = reader.read_u32("event.schema_version");
  envelope.event.kind =
      static_cast<CallbackEventKind>(reader.read_u8("event.kind"));
  envelope.event.phase =
      static_cast<CallbackPhase>(reader.read_u8("event.phase"));
  envelope.event.event_index = reader.read_u64("event.event_index");
  envelope.event.run_id =
      reader.read_string("event.run_id", options.max_string_bytes);
  envelope.event.trace_id =
      reader.read_string("event.trace_id", options.max_string_bytes);
  envelope.event.epoch = reader.read_u64("event.epoch");
  envelope.event.global_step = reader.read_u64("event.global_step");
  const auto metric_count = reader.read_u64("event.metric_count");
  if (metric_count > options.max_metrics_per_event) {
    throw_exhausted("state.metric_count",
                    std::to_string(options.max_metrics_per_event),
                    std::to_string(metric_count),
                    "Reject checkpoint state above configured limits.");
  }
  for (std::uint64_t index = 0; index < metric_count; ++index) {
    auto name =
        reader.read_string("event.metric_name", options.max_string_bytes);
    const auto value = reader.read_double("event.metric_value");
    if (!envelope.event.metrics.emplace(std::move(name), value).second) {
      throw_serialization("event.metrics", "unique metric names", "duplicate",
                          "Reject ambiguous event observations.");
    }
  }
  const auto command_count = reader.read_u64("event.command_count");
  if (command_count > options.max_commands_per_event) {
    throw_exhausted("state.command_count",
                    std::to_string(options.max_commands_per_event),
                    std::to_string(command_count),
                    "Reject checkpoint state above configured limits.");
  }
  envelope.callback_result.commands.reserve(
      static_cast<std::size_t>(command_count));
  for (std::uint64_t index = 0; index < command_count; ++index) {
    envelope.callback_result.commands.push_back(
        decode_command(reader, options));
  }
  const auto has_effective = reader.read_u8("event.has_effective_command");
  if (has_effective > 1U) {
    throw_serialization("event.has_effective_command", "zero or one",
                        std::to_string(has_effective),
                        "Reject invalid optional-command encoding.");
  }
  if (has_effective == 1U) {
    envelope.callback_result.effective_command =
        decode_command(reader, options);
  }
  reader.require_finished("event.envelope");
  if (envelope.schema_version != 1U) {
    throw_incompatible("event.envelope_schema", "1",
                       std::to_string(envelope.schema_version),
                       "Migrate event state before restoring it.");
  }
  validate_event_shape(envelope.event, options);
  validate_callback_result(envelope.callback_result, options);
  return envelope;
}

std::vector<std::byte> encode_topology(
    const EventOutboxOptions& options,
    const std::vector<std::shared_ptr<EventSink>>& sinks) {
  std::vector<std::byte> output;
  append_string(output, kTopologyMagic);
  append_string(output, options.outbox_id);
  append_u64(output, options.max_sinks);
  append_u64(output, options.max_pending_events);
  append_u64(output, options.max_metrics_per_event);
  append_u64(output, options.max_commands_per_event);
  append_u64(output, options.max_string_bytes);
  append_u64(output, options.max_event_bytes);
  append_u64(output, options.max_pending_bytes);
  append_u64(output, options.max_topology_bytes);
  append_u64(output, options.max_control_bytes);
  append_u64(output, static_cast<std::uint64_t>(sinks.size()));
  for (const auto& sink : sinks) {
    const auto& descriptor = sink->descriptor();
    append_u32(output, descriptor.schema_version);
    append_string(output, descriptor.type_id);
    append_string(output, descriptor.sink_id);
    append_string(output, descriptor.destination_id);
    append_u8(output, static_cast<std::uint8_t>(descriptor.delivery_guarantee));
  }
  return output;
}

void validate_manifest(const StateManifest& stored,
                       const StateManifest& expected, RestorePolicy policy) {
  validate_state_manifest(stored);
  if (!restore_policy_accepts(policy, stored.restore_level)) {
    throw_incompatible("manifest.restore_level",
                       std::string(restore_policy_name(policy)),
                       std::string(restore_level_name(stored.restore_level)),
                       "Use exact restore policy for event delivery state.");
  }
  if (stored.participant_id != expected.participant_id ||
      stored.schema_version != expected.schema_version ||
      stored.restore_level != RestoreLevel::exact ||
      stored.fields.size() != expected.fields.size()) {
    throw_incompatible("manifest", expected.participant_id,
                       stored.participant_id,
                       "Restore only current state for the same outbox.");
  }
  for (const auto& right : expected.fields) {
    const auto left = std::find_if(stored.fields.begin(), stored.fields.end(),
                                   [&](const StateFieldDescriptor& candidate) {
                                     return candidate.name == right.name;
                                   });
    if (left == stored.fields.end() || left->media_type != right.media_type ||
        left->required != right.required || left->max_bytes > right.max_bytes) {
      throw_incompatible("manifest.fields", right.name,
                         left == stored.fields.end() ? "missing" : left->name,
                         "Reject a changed outbox field contract.");
    }
  }
}

std::string json_string(std::string_view value) {
  std::ostringstream output;
  output << '"';
  constexpr char kHex[] = "0123456789abcdef";
  for (const auto character : value) {
    const auto byte = static_cast<unsigned char>(character);
    switch (byte) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (byte < 0x20U) {
          output << "\\u00" << kHex[(byte >> 4U) & 0x0fU] << kHex[byte & 0x0fU];
        } else {
          output << static_cast<char>(byte);
        }
    }
  }
  output << '"';
  return output.str();
}

void append_command_json(std::ostringstream& output,
                         const CallbackCommand& command) {
  output << "{\"kind\":"
         << json_string(callback_command_kind_name(command.kind))
         << ",\"callback_id\":" << json_string(command.callback_id)
         << ",\"reason\":" << json_string(command.reason) << '}';
}

std::string render_json(const EventEnvelope& envelope) {
  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << std::setprecision(std::numeric_limits<double>::max_digits10);
  output << "{\"schema_version\":" << envelope.schema_version
         << ",\"idempotency_key\":"
         << json_string(envelope.event.run_id + ":" +
                        std::to_string(envelope.event.event_index))
         << ",\"kind\":"
         << json_string(callback_event_kind_name(envelope.event.kind))
         << ",\"phase\":"
         << json_string(callback_phase_name(envelope.event.phase))
         << ",\"event_index\":" << envelope.event.event_index
         << ",\"run_id\":" << json_string(envelope.event.run_id)
         << ",\"trace_id\":" << json_string(envelope.event.trace_id)
         << ",\"epoch\":" << envelope.event.epoch
         << ",\"global_step\":" << envelope.event.global_step
         << ",\"metrics\":{";
  bool first = true;
  for (const auto& [name, value] : envelope.event.metrics) {
    if (!first) {
      output << ',';
    }
    first = false;
    output << json_string(name) << ':' << value;
  }
  output << "},\"commands\":[";
  first = true;
  for (const auto& command : envelope.callback_result.commands) {
    if (!first) {
      output << ',';
    }
    first = false;
    append_command_json(output, command);
  }
  output << "],\"effective_command\":";
  if (envelope.callback_result.effective_command.has_value()) {
    append_command_json(output, *envelope.callback_result.effective_command);
  } else {
    output << "null";
  }
  output << '}';
  return output.str();
}

void validate_local_sink_identity(const EventSinkDescriptor& descriptor) {
  EventOutboxOptions limits;
  validate_sink_descriptor(descriptor, limits);
}

}  // namespace

std::string_view event_delivery_guarantee_name(
    EventDeliveryGuarantee guarantee) noexcept {
  switch (guarantee) {
    case EventDeliveryGuarantee::at_least_once:
      return "at_least_once";
  }
  return "unknown";
}

struct EventOutbox::Impl {
  struct PendingEvent {
    EventEnvelope envelope;           ///< Complete committed event.
    std::size_t next_sink_index = 0;  ///< First sink not yet acknowledged.
    std::uint64_t encoded_bytes = 0;  ///< Queue-budget charge for this node.
  };

  EventOutboxOptions options;  ///< Immutable participant limits and identity.
  std::vector<std::shared_ptr<EventSink>> sinks;  ///< Stable sink order.
  std::vector<PendingEvent> pending;  ///< Ordered bounded delivery queue.
  StreamState stream;                 ///< Last accepted event stream cursor.
  std::uint64_t pending_bytes = 0;    ///< Charged encoded queue bytes.
  std::uint64_t last_delivered_event_index = 0;  ///< Fully delivered cursor.
  std::uint64_t last_delivered_epoch = 0;        ///< Epoch at delivered cursor.
  std::uint64_t last_delivered_global_step = 0;  ///< Step at delivered cursor.
  std::size_t next_close_sink_index = 0;         ///< First sink not yet closed.
  bool close_prepared = false;  ///< Whether drain and flush already succeeded.
  bool closed = false;          ///< Whether explicit finalization completed.
};

EventOutbox::EventOutbox(std::vector<std::shared_ptr<EventSink>> sinks,
                         EventOutboxOptions options)
    : impl_(std::make_unique<Impl>()) {
  if constexpr (std::endian::native != std::endian::little) {
    throw NotImplemented(
        "Event outbox checkpoint byte order is unsupported",
        ErrorDetails{.component = kComponent,
                     .operation = "platform.endian",
                     .expected = "little endian",
                     .actual = "non-little endian",
                     .hint = "Add an explicit byte-swap codec."});
  }
  validate_options(options);
  if (sinks.empty() || sinks.size() > options.max_sinks) {
    throw_invalid("sinks", "one to " + std::to_string(options.max_sinks),
                  std::to_string(sinks.size()),
                  "Attach at least one bounded event destination.");
  }
  std::set<std::string> sink_ids;
  for (const auto& sink : sinks) {
    if (!sink) {
      throw_invalid("sinks", "non-null sink owners", "null",
                    "Construct each sink before the outbox.");
    }
    validate_sink_descriptor(sink->descriptor(), options);
    if (!sink_ids.insert(sink->descriptor().sink_id).second) {
      throw_invalid("sinks.sink_id", "unique stable sink IDs",
                    sink->descriptor().sink_id,
                    "Rename the duplicate sink descriptor.");
    }
  }
  std::sort(sinks.begin(), sinks.end(),
            [](const auto& left, const auto& right) {
              return left->descriptor().sink_id < right->descriptor().sink_id;
            });
  const auto topology = encode_topology(options, sinks);
  if (topology.size() > options.max_topology_bytes) {
    throw_exhausted("topology", std::to_string(options.max_topology_bytes),
                    std::to_string(topology.size()),
                    "Reduce sink descriptors or raise the explicit limit.");
  }
  impl_->options = std::move(options);
  impl_->sinks = std::move(sinks);
  impl_->pending.reserve(impl_->options.max_pending_events);
}

EventOutbox::~EventOutbox() = default;

void EventOutbox::enqueue(const CallbackEvent& event,
                          const CallbackDispatchResult& callback_result) {
  if (impl_->closed || impl_->close_prepared) {
    throw NotReady(
        "Event outbox is already closed",
        ErrorDetails{.component = kComponent,
                     .operation = "enqueue",
                     .expected = "an open outbox",
                     .actual = impl_->closed ? "closed" : "close in progress",
                     .hint = "Create a new outbox for a new run."});
  }
  validate_event_shape(event, impl_->options);
  validate_callback_result(callback_result, impl_->options);
  auto next_stream = validate_transition(event, impl_->stream);
  if (impl_->pending.size() >= impl_->options.max_pending_events) {
    throw_exhausted("queue.events",
                    std::to_string(impl_->options.max_pending_events),
                    std::to_string(impl_->pending.size() + 1U),
                    "Drain pending events before accepting more work.");
  }

  EventEnvelope candidate{
      .schema_version = 1, .event = event, .callback_result = callback_result};
  const auto encoded = encode_envelope(candidate);
  if (encoded.size() > impl_->options.max_event_bytes) {
    throw_exhausted("queue.event_bytes",
                    std::to_string(impl_->options.max_event_bytes),
                    std::to_string(encoded.size()),
                    "Reduce event fields or raise the explicit limit.");
  }
  constexpr std::uint64_t kNodeOverhead = 16U;
  const auto charged = static_cast<std::uint64_t>(encoded.size()) +
                       static_cast<std::uint64_t>(kNodeOverhead);
  if (charged > impl_->options.max_pending_bytes - impl_->pending_bytes) {
    throw_exhausted("queue.pending_bytes",
                    std::to_string(impl_->options.max_pending_bytes),
                    std::to_string(impl_->pending_bytes + charged),
                    "Drain pending events before accepting more work.");
  }

  impl_->pending.push_back({.envelope = std::move(candidate),
                            .next_sink_index = 0,
                            .encoded_bytes = charged});
  impl_->pending_bytes += charged;
  impl_->stream = std::move(next_stream);
}

bool EventOutbox::drain_one(std::stop_token stop) {
  if (impl_->pending.empty()) {
    return false;
  }
  auto& pending = impl_->pending.front();
  while (pending.next_sink_index < impl_->sinks.size()) {
    check_stop(stop, "drain.publish");
    impl_->sinks[pending.next_sink_index]->publish(pending.envelope, stop);
    ++pending.next_sink_index;
  }
  impl_->last_delivered_event_index = pending.envelope.event.event_index;
  impl_->last_delivered_epoch = pending.envelope.event.epoch;
  impl_->last_delivered_global_step = pending.envelope.event.global_step;
  impl_->pending_bytes -= pending.encoded_bytes;
  impl_->pending.erase(impl_->pending.begin());
  return true;
}

void EventOutbox::drain(std::stop_token stop) {
  while (drain_one(stop)) {
  }
}

void EventOutbox::flush(std::stop_token stop) {
  if (impl_->closed) {
    return;
  }
  if (impl_->close_prepared && !impl_->closed) {
    throw NotReady(
        "Event outbox close is already in progress",
        ErrorDetails{
            .component = kComponent,
            .operation = "flush",
            .expected = "close() retry",
            .actual = "partial sink finalization",
            .hint = "Retry close() so already closed sinks are skipped."});
  }
  drain(stop);
  for (const auto& sink : impl_->sinks) {
    check_stop(stop, "flush");
    sink->flush(stop);
  }
}

void EventOutbox::close(std::stop_token stop) {
  if (impl_->closed) {
    return;
  }
  if (!impl_->close_prepared) {
    flush(stop);
    impl_->close_prepared = true;
  }
  while (impl_->next_close_sink_index < impl_->sinks.size()) {
    check_stop(stop, "close");
    impl_->sinks[impl_->next_close_sink_index]->close(stop);
    ++impl_->next_close_sink_index;
  }
  impl_->closed = true;
}

bool EventOutbox::closed() const noexcept { return impl_->closed; }

std::size_t EventOutbox::pending_events() const noexcept {
  return impl_->pending.size();
}

std::uint64_t EventOutbox::pending_bytes() const noexcept {
  return impl_->pending_bytes;
}

std::uint64_t EventOutbox::last_enqueued_event_index() const noexcept {
  return impl_->stream.last_event_index;
}

std::uint64_t EventOutbox::last_delivered_event_index() const noexcept {
  return impl_->last_delivered_event_index;
}

const EventOutboxOptions& EventOutbox::options() const noexcept {
  return impl_->options;
}

std::string EventOutbox::participant_id() const {
  return impl_->options.outbox_id;
}

StateManifest EventOutbox::describe_state() const {
  return {.participant_id = impl_->options.outbox_id,
          .schema_version = 1,
          .restore_level = RestoreLevel::exact,
          .fields = {{.name = kTopologyField,
                      .media_type = kTopologyMediaType,
                      .max_bytes = impl_->options.max_topology_bytes},
                     {.name = kControlField,
                      .media_type = kControlMediaType,
                      .max_bytes = impl_->options.max_control_bytes}},
          .degradation_reasons = {}};
}

void EventOutbox::save_state(StateWriter& writer) const {
  if (impl_->closed || impl_->close_prepared) {
    throw NotReady(
        "Cannot checkpoint an event outbox during or after close",
        ErrorDetails{
            .component = kComponent,
            .operation = "save_state",
            .expected = "an open outbox before sink finalization",
            .actual = impl_->closed ? "closed" : "close in progress",
            .hint = "Checkpoint pending delivery before close starts."});
  }
  const auto topology = encode_topology(impl_->options, impl_->sinks);
  std::vector<std::byte> control;
  append_string(control, kControlMagic);
  append_u8(control, impl_->stream.initialized ? 1U : 0U);
  append_u8(control, impl_->stream.terminal ? 1U : 0U);
  append_string(control, impl_->stream.run_id);
  append_u64(control, impl_->stream.last_event_index);
  append_u64(control, impl_->stream.last_epoch);
  append_u64(control, impl_->stream.last_global_step);
  append_u64(control, impl_->last_delivered_event_index);
  append_u64(control, impl_->last_delivered_epoch);
  append_u64(control, impl_->last_delivered_global_step);
  append_u64(control, impl_->pending_bytes);
  append_u64(control, static_cast<std::uint64_t>(impl_->pending.size()));
  for (const auto& pending : impl_->pending) {
    const auto encoded = encode_envelope(pending.envelope);
    append_u64(control, static_cast<std::uint64_t>(pending.next_sink_index));
    append_u64(control, pending.encoded_bytes);
    append_u64(control, static_cast<std::uint64_t>(encoded.size()));
    control.insert(control.end(), encoded.begin(), encoded.end());
  }
  if (topology.size() > impl_->options.max_topology_bytes ||
      control.size() > impl_->options.max_control_bytes) {
    throw_exhausted("save_state", "state within configured limits",
                    std::to_string(topology.size() + control.size()),
                    "Drain events or raise checkpoint limits.");
  }
  writer.write_blob(kTopologyField, std::span<const std::byte>(
                                        topology.data(), topology.size()));
  writer.write_blob(kControlField,
                    std::span<const std::byte>(control.data(), control.size()));
}

void EventOutbox::load_state(const StateReader& reader, RestorePolicy policy) {
  if (impl_->closed || impl_->close_prepared) {
    throw NotReady(
        "Cannot restore a closed event outbox",
        ErrorDetails{.component = kComponent,
                     .operation = "load_state",
                     .expected = "an open outbox",
                     .actual = impl_->closed ? "closed" : "close in progress",
                     .hint = "Restore into a newly constructed "
                             "outbox with matching sinks."});
  }
  const auto expected_manifest = describe_state();
  validate_manifest(reader.manifest(), expected_manifest, policy);
  const auto stored_topology =
      reader.read_blob(kTopologyField, impl_->options.max_topology_bytes);
  const auto expected_topology = encode_topology(impl_->options, impl_->sinks);
  if (stored_topology.size() != expected_topology.size() ||
      !std::equal(stored_topology.begin(), stored_topology.end(),
                  expected_topology.begin())) {
    throw_incompatible("topology", "identical sink topology and limits",
                       "different",
                       "Restore with the same ordered logical destinations.");
  }

  const auto stored_control =
      reader.read_blob(kControlField, impl_->options.max_control_bytes);
  ByteReader control(stored_control);
  if (control.read_string("control.magic", kControlMagic.size()) !=
      kControlMagic) {
    throw_serialization("control.magic", std::string(kControlMagic),
                        "different", "Reject unknown outbox state encoding.");
  }
  const auto initialized = control.read_u8("control.initialized");
  const auto terminal = control.read_u8("control.terminal");
  if (initialized > 1U || terminal > 1U) {
    throw_serialization("control.flags", "zero or one", "invalid",
                        "Reject invalid boolean state encoding.");
  }
  StreamState candidate_stream{
      .initialized = initialized == 1U,
      .terminal = terminal == 1U,
      .run_id = control.read_string("control.run_id",
                                    impl_->options.max_string_bytes),
      .last_event_index = control.read_u64("control.last_event_index"),
      .last_epoch = control.read_u64("control.last_epoch"),
      .last_global_step = control.read_u64("control.last_global_step")};
  const auto candidate_delivered =
      control.read_u64("control.last_delivered_event_index");
  const auto candidate_delivered_epoch =
      control.read_u64("control.last_delivered_epoch");
  const auto candidate_delivered_global_step =
      control.read_u64("control.last_delivered_global_step");
  const auto stored_pending_bytes = control.read_u64("control.pending_bytes");
  const auto pending_count = control.read_u64("control.pending_count");
  if (pending_count > impl_->options.max_pending_events) {
    throw_exhausted("state.pending_count",
                    std::to_string(impl_->options.max_pending_events),
                    std::to_string(pending_count),
                    "Reject checkpoint state above configured limits.");
  }

  std::vector<Impl::PendingEvent> candidate_pending;
  candidate_pending.reserve(static_cast<std::size_t>(pending_count));
  std::uint64_t candidate_pending_bytes = 0;
  StreamState replay_stream{
      .initialized = candidate_delivered != 0U,
      .terminal = false,
      .run_id =
          candidate_delivered == 0U ? std::string{} : candidate_stream.run_id,
      .last_event_index = candidate_delivered,
      .last_epoch = candidate_delivered_epoch,
      .last_global_step = candidate_delivered_global_step};
  std::uint64_t expected_pending_index = candidate_delivered;
  for (std::uint64_t index = 0; index < pending_count; ++index) {
    const auto next_sink = control.read_u64("pending.next_sink_index");
    const auto charged = control.read_u64("pending.encoded_bytes");
    const auto encoded_size = control.read_u64("pending.envelope_size");
    if (next_sink > impl_->sinks.size() ||
        encoded_size > impl_->options.max_event_bytes ||
        encoded_size > stored_control.size()) {
      throw_serialization("pending.node", "bounded sink and event sizes",
                          "out of range",
                          "Reject corrupt pending event metadata.");
    }
    std::vector<std::byte> encoded;
    encoded.reserve(static_cast<std::size_t>(encoded_size));
    for (std::uint64_t byte = 0; byte < encoded_size; ++byte) {
      encoded.push_back(
          static_cast<std::byte>(control.read_u8("pending.envelope")));
    }
    auto envelope = decode_envelope(encoded, impl_->options);
    if (expected_pending_index == std::numeric_limits<std::uint64_t>::max() ||
        envelope.event.event_index != expected_pending_index + 1U) {
      throw_serialization("pending.event_index", "gap-free delivery order",
                          std::to_string(envelope.event.event_index),
                          "Reject duplicate or skipped pending events.");
    }
    expected_pending_index = envelope.event.event_index;
    replay_stream = validate_transition(envelope.event, replay_stream);
    constexpr std::uint64_t kNodeOverhead = 16U;
    const auto expected_charged = encoded_size + kNodeOverhead;
    if (charged != expected_charged ||
        charged > impl_->options.max_pending_bytes - candidate_pending_bytes) {
      throw_serialization("pending.encoded_bytes",
                          std::to_string(expected_charged),
                          std::to_string(charged),
                          "Reject inconsistent queue resource accounting.");
    }
    candidate_pending_bytes += charged;
    candidate_pending.push_back(
        {.envelope = std::move(envelope),
         .next_sink_index = static_cast<std::size_t>(next_sink),
         .encoded_bytes = charged});
  }
  control.require_finished("control");

  if (stored_pending_bytes != candidate_pending_bytes ||
      candidate_delivered > candidate_stream.last_event_index ||
      candidate_delivered_epoch > candidate_stream.last_epoch ||
      candidate_delivered_global_step > candidate_stream.last_global_step ||
      (candidate_delivered == 0U && (candidate_delivered_epoch != 0U ||
                                     candidate_delivered_global_step != 0U)) ||
      (!candidate_stream.initialized &&
       (candidate_stream.terminal || !candidate_stream.run_id.empty() ||
        candidate_stream.last_event_index != 0U || candidate_delivered != 0U ||
        !candidate_pending.empty())) ||
      (candidate_stream.initialized &&
       (!is_identifier(candidate_stream.run_id) ||
        candidate_stream.run_id.size() > impl_->options.max_string_bytes)) ||
      (candidate_pending.empty() &&
       (candidate_delivered != candidate_stream.last_event_index ||
        candidate_delivered_epoch != candidate_stream.last_epoch ||
        candidate_delivered_global_step !=
            candidate_stream.last_global_step)) ||
      (!candidate_pending.empty() &&
       (candidate_pending.front().envelope.event.event_index !=
            candidate_delivered + 1U ||
        candidate_pending.back().envelope.event.event_index !=
            candidate_stream.last_event_index ||
        replay_stream.run_id != candidate_stream.run_id ||
        replay_stream.last_epoch != candidate_stream.last_epoch ||
        replay_stream.last_global_step != candidate_stream.last_global_step ||
        replay_stream.terminal != candidate_stream.terminal))) {
    throw_serialization("control", "internally consistent stream and queue",
                        "inconsistent",
                        "Reject partial or reordered event outbox state.");
  }

  impl_->pending.swap(candidate_pending);
  impl_->stream = std::move(candidate_stream);
  impl_->pending_bytes = candidate_pending_bytes;
  impl_->last_delivered_event_index = candidate_delivered;
  impl_->last_delivered_epoch = candidate_delivered_epoch;
  impl_->last_delivered_global_step = candidate_delivered_global_step;
}

struct JsonlEventSink::Impl {
  EventSinkDescriptor descriptor;  ///< Immutable logical destination.
  std::filesystem::path path;      ///< Owned append-only file path.
  std::ofstream output;            ///< Owned append-mode output stream.
  bool closed = false;             ///< Whether explicit close completed.
};

JsonlEventSink::JsonlEventSink(JsonlEventSinkOptions options)
    : impl_(std::make_unique<Impl>()) {
  impl_->descriptor = {
      .schema_version = 1,
      .type_id = "jyppx.event-sink.jsonl/v1",
      .sink_id = std::move(options.sink_id),
      .destination_id = std::move(options.destination_id),
      .delivery_guarantee = EventDeliveryGuarantee::at_least_once};
  validate_local_sink_identity(impl_->descriptor);
  if (options.path.empty()) {
    throw_invalid("jsonl.path", "a nonempty output path", "empty",
                  "Choose a local JSONL destination.");
  }
  impl_->path = std::move(options.path);
  impl_->output.open(impl_->path,
                     std::ios::out | std::ios::app | std::ios::binary);
  if (!impl_->output.is_open()) {
    throw IoError("Could not open JSONL event sink",
                  ErrorDetails{.component = "training.event_sink.jsonl",
                               .operation = "open",
                               .expected = "writable append-only file",
                               .actual = impl_->path.string(),
                               .hint = "Create the parent directory and verify "
                                       "write permissions."});
  }
}

JsonlEventSink::~JsonlEventSink() = default;

const EventSinkDescriptor& JsonlEventSink::descriptor() const noexcept {
  return impl_->descriptor;
}

void JsonlEventSink::publish(const EventEnvelope& envelope,
                             std::stop_token stop) {
  check_stop(stop, "jsonl.publish");
  if (impl_->closed) {
    throw NotReady("JSONL event sink is closed",
                   ErrorDetails{.component = "training.event_sink.jsonl",
                                .operation = "publish",
                                .expected = "an open stream",
                                .actual = "closed",
                                .hint = "Create a new sink for another run."});
  }
  const auto line = render_json(envelope);
  impl_->output << line << '\n';
  impl_->output.flush();
  if (!impl_->output) {
    throw IoError("Could not append JSONL event",
                  ErrorDetails{.component = "training.event_sink.jsonl",
                               .operation = "publish",
                               .expected = "complete flushed JSON line",
                               .actual = impl_->path.string(),
                               .hint = "Check storage capacity and stream "
                                       "permissions before retrying."});
  }
}

void JsonlEventSink::flush(std::stop_token stop) {
  check_stop(stop, "jsonl.flush");
  if (impl_->closed) {
    return;
  }
  impl_->output.flush();
  if (!impl_->output) {
    throw IoError(
        "Could not flush JSONL event sink",
        ErrorDetails{.component = "training.event_sink.jsonl",
                     .operation = "flush",
                     .expected = "successful stream flush",
                     .actual = impl_->path.string(),
                     .hint = "Check storage health before retrying."});
  }
}

void JsonlEventSink::close(std::stop_token stop) {
  if (impl_->closed) {
    return;
  }
  flush(stop);
  impl_->output.close();
  if (impl_->output.fail()) {
    throw IoError("Could not close JSONL event sink",
                  ErrorDetails{.component = "training.event_sink.jsonl",
                               .operation = "close",
                               .expected = "successful stream close",
                               .actual = impl_->path.string(),
                               .hint = "Inspect the filesystem and retry "
                                       "explicit finalization."});
  }
  impl_->closed = true;
}

ConsoleEventSink::ConsoleEventSink(std::ostream& output,
                                   ConsoleEventSinkOptions options)
    : descriptor_{.schema_version = 1,
                  .type_id = "jyppx.event-sink.console/v1",
                  .sink_id = std::move(options.sink_id),
                  .destination_id = std::move(options.destination_id),
                  .delivery_guarantee = EventDeliveryGuarantee::at_least_once},
      output_(&output) {
  validate_local_sink_identity(descriptor_);
}

const EventSinkDescriptor& ConsoleEventSink::descriptor() const noexcept {
  return descriptor_;
}

void ConsoleEventSink::publish(const EventEnvelope& envelope,
                               std::stop_token stop) {
  check_stop(stop, "console.publish");
  if (closed_) {
    throw NotReady("Console event sink is closed",
                   ErrorDetails{.component = "training.event_sink.console",
                                .operation = "publish",
                                .expected = "an open stream",
                                .actual = "closed",
                                .hint = "Create a new sink for another run."});
  }
  *output_ << render_json(envelope) << '\n';
  output_->flush();
  if (!*output_) {
    throw IoError("Could not write console event",
                  ErrorDetails{.component = "training.event_sink.console",
                               .operation = "publish",
                               .expected = "complete flushed JSON line",
                               .actual = descriptor_.destination_id,
                               .hint = "Repair the output stream before "
                                       "retrying delivery."});
  }
}

void ConsoleEventSink::flush(std::stop_token stop) {
  check_stop(stop, "console.flush");
  if (closed_) {
    return;
  }
  output_->flush();
  if (!*output_) {
    throw IoError("Could not flush console event sink",
                  ErrorDetails{.component = "training.event_sink.console",
                               .operation = "flush",
                               .expected = "successful stream flush",
                               .actual = descriptor_.destination_id,
                               .hint = "Repair the output stream before "
                                       "retrying finalization."});
  }
}

void ConsoleEventSink::close(std::stop_token stop) {
  if (closed_) {
    return;
  }
  flush(stop);
  closed_ = true;
}

}  // namespace jyppx::training
