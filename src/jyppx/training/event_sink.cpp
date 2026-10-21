// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/training/event_sink.hpp>
#include <jyppx/training/telemetry_adapter.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <random>
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
    "application/vnd.jyppx.event-outbox-topology-v2";
constexpr char kControlMediaType[] =
    "application/vnd.jyppx.event-outbox-control-v2";
constexpr std::string_view kTopologyMagic = "JYPPX-EVENT-OUTBOX-TOPOLOGY-V2";
constexpr std::string_view kControlMagic = "JYPPX-EVENT-OUTBOX-CONTROL-V2";
constexpr std::string_view kTopologyMagicV1 = "JYPPX-EVENT-OUTBOX-TOPOLOGY-V1";
constexpr std::string_view kControlMagicV1 = "JYPPX-EVENT-OUTBOX-CONTROL-V1";

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
  constexpr std::uint64_t kControlNodeOverheadBeyondQueueCharge = 17U;
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

  std::span<const std::byte> read_bytes(std::string_view field,
                                        std::uint64_t max_bytes) {
    const auto count = read_u64(field);
    if (count > max_bytes || count > bytes_.size() - offset_) {
      throw_serialization(std::string(field), "bounded complete byte string",
                          std::to_string(count),
                          "Reject corrupt outbox bytes.");
    }
    const auto result =
        bytes_.subspan(offset_, static_cast<std::size_t>(count));
    offset_ += static_cast<std::size_t>(count);
    return result;
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

void append_bytes(std::vector<std::byte>& output,
                  std::span<const std::byte> bytes) {
  append_u64(output, static_cast<std::uint64_t>(bytes.size()));
  output.insert(output.end(), bytes.begin(), bytes.end());
}

std::string new_identity(std::string_view prefix) {
  std::random_device random;
  constexpr char hex[] = "0123456789abcdef";
  std::string result(prefix);
  for (std::size_t index = 0U; index < 16U; ++index) {
    const auto value = static_cast<unsigned int>(random()) & 0xffU;
    result.push_back(hex[value >> 4U]);
    result.push_back(hex[value & 15U]);
  }
  return result;
}

TelemetryLimits compatibility_limits(const EventOutboxOptions& options) {
  TelemetryLimits result;
  result.max_identifier_bytes =
      std::max<std::size_t>(256U, options.max_string_bytes);
  result.max_text_bytes =
      std::max<std::size_t>(4096U, options.max_string_bytes);
  result.max_metrics_per_envelope = options.max_metrics_per_event;
  result.max_decisions_per_envelope = options.max_commands_per_event;
  result.max_envelope_bytes = options.max_event_bytes;
  result.max_batch_bytes = options.max_event_bytes;
  return result;
}

void encode_options(std::vector<std::byte>& output,
                    const EventOutboxOptions& options) {
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
}

std::vector<std::byte> encode_topology(
    const EventOutboxOptions& options,
    const std::vector<TelemetrySinkDescriptor>& sinks,
    const TrainingTelemetryAdapterOptions& telemetry, bool compatibility) {
  std::vector<std::byte> output;
  append_string(output, kTopologyMagic);
  encode_options(output, options);
  append_u8(output, compatibility ? 1U : 0U);
  append_u64(output, static_cast<std::uint64_t>(sinks.size()));
  for (const auto& descriptor : sinks) {
    append_u32(output, descriptor.schema_version);
    append_string(output, descriptor.type_id);
    append_string(output, descriptor.sink_id);
    append_string(output, descriptor.destination_id);
    append_u8(output, static_cast<std::uint8_t>(descriptor.role));
  }
  append_string(output, telemetry.project_id);
  append_string(output, telemetry.attempt_id);
  append_u64(output, static_cast<std::uint64_t>(telemetry.metrics.size()));
  for (const auto& [name, binding] : telemetry.metrics) {
    append_string(output, name);
    append_bytes(output, encode_telemetry_metric_definition(binding.definition,
                                                            telemetry.limits));
    append_u64(output, static_cast<std::uint64_t>(binding.context.size()));
    for (const auto& [key, value] : binding.context) {
      append_string(output, key);
      append_string(output, value);
    }
  }
  const auto& limits = telemetry.limits;
  append_u64(output, limits.max_identifier_bytes);
  append_u64(output, limits.max_text_bytes);
  append_u64(output, limits.max_attributes);
  append_u64(output, limits.max_context_fields);
  append_u64(output, limits.max_metrics_per_envelope);
  append_u64(output, limits.max_decisions_per_envelope);
  append_u64(output, limits.max_batch_events);
  append_u64(output, limits.max_envelope_bytes);
  append_u64(output, limits.max_batch_bytes);
  append_u64(output, limits.max_artifact_reference_bytes);
  return output;
}

EventOutboxOptions decode_v1_topology(
    std::span<const std::byte> bytes, const EventOutboxOptions& options,
    const std::vector<TelemetrySinkDescriptor>& current) {
  ByteReader reader(bytes);
  if (reader.read_string("topology.magic", kTopologyMagicV1.size()) !=
          kTopologyMagicV1 ||
      reader.read_string("topology.outbox_id", options.max_string_bytes) !=
          options.outbox_id) {
    throw_incompatible("topology.v1", "recognized same-outbox N-1 topology",
                       "different", "Keep the original legacy checkpoint.");
  }
  const std::uint64_t bounds[] = {options.max_sinks,
                                  options.max_pending_events,
                                  options.max_metrics_per_event,
                                  options.max_commands_per_event,
                                  options.max_string_bytes,
                                  options.max_event_bytes,
                                  options.max_pending_bytes,
                                  options.max_topology_bytes,
                                  options.max_control_bytes};
  std::uint64_t old_bounds[9]{};
  for (std::size_t index = 0U; index < 9U; ++index) {
    const auto bound = bounds[index];
    const auto previous = reader.read_u64("topology.limit");
    if (previous == 0U || previous > bound) {
      throw_incompatible("topology.v1.limit", "no smaller explicit new limits",
                         std::to_string(previous),
                         "Retain or raise all legacy decode bounds.");
    }
    old_bounds[index] = previous;
  }
  EventOutboxOptions legacy = options;
  legacy.max_sinks = static_cast<std::size_t>(old_bounds[0]);
  legacy.max_pending_events = static_cast<std::size_t>(old_bounds[1]);
  legacy.max_metrics_per_event = static_cast<std::size_t>(old_bounds[2]);
  legacy.max_commands_per_event = static_cast<std::size_t>(old_bounds[3]);
  legacy.max_string_bytes = static_cast<std::size_t>(old_bounds[4]);
  legacy.max_event_bytes = old_bounds[5];
  legacy.max_pending_bytes = old_bounds[6];
  legacy.max_topology_bytes = old_bounds[7];
  legacy.max_control_bytes = old_bounds[8];
  if (bytes.size() > legacy.max_topology_bytes) {
    throw_serialization("topology.v1.size", "within original topology bound",
                        std::to_string(bytes.size()),
                        "Reject state outside its own legacy contract.");
  }
  const auto count = reader.read_u64("topology.sink_count");
  if (count != current.size() || count > legacy.max_sinks) {
    throw_incompatible("topology.v1.sinks", "same logical destinations",
                       std::to_string(count),
                       "Do not reset old delivery cursors.");
  }
  std::vector<EventSinkDescriptor> result;
  result.reserve(current.size());
  for (std::size_t index = 0U; index < current.size(); ++index) {
    EventSinkDescriptor descriptor;
    descriptor.schema_version = reader.read_u32("sink.schema");
    descriptor.type_id =
        reader.read_string("sink.type", legacy.max_string_bytes);
    descriptor.sink_id = reader.read_string("sink.id", legacy.max_string_bytes);
    descriptor.destination_id =
        reader.read_string("sink.destination", legacy.max_string_bytes);
    descriptor.delivery_guarantee =
        static_cast<EventDeliveryGuarantee>(reader.read_u8("sink.guarantee"));
    validate_sink_descriptor(descriptor, legacy);
    if (!result.empty() && result.back().sink_id >= descriptor.sink_id) {
      throw_serialization(
          "topology.v1.order", "unique ordered legacy sinks",
          descriptor.sink_id,
          "Reject duplicate or reordered logical destinations.");
    }
    const auto& target = current[index];
    const auto mapping = std::make_pair(descriptor.type_id, target.type_id);
    if (descriptor.sink_id != target.sink_id ||
        descriptor.destination_id != target.destination_id ||
        (descriptor.type_id != target.type_id &&
         std::find(options.v1_sink_type_migrations.begin(),
                   options.v1_sink_type_migrations.end(),
                   mapping) == options.v1_sink_type_migrations.end())) {
      throw_incompatible(
          "topology.v1.destination",
          "same IDs and explicitly mapped implementation", descriptor.sink_id,
          "Never replace delivery state with a new destination.");
    }
    result.push_back(std::move(descriptor));
  }
  reader.require_finished("topology.v1");
  return legacy;
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
    EventEnvelope snapshot;      ///< Necessary exact callback audit snapshot.
    TelemetryEnvelope envelope;  ///< Immutable neutral observation.
    std::vector<std::byte> snapshot_bytes;  ///< Canonical callback snapshot.
    std::vector<std::byte>
        canonical;             ///< Canonical neutral bytes, never rebuilt.
    std::uint8_t format = 1U;  ///< Live typed, live compatibility, or N-1.
    std::size_t next_sink_index = 0U;
    std::uint64_t encoded_bytes = 0U;
  };

  EventOutboxOptions options;
  TrainingTelemetryAdapterOptions telemetry;
  bool compatibility = false;
  bool explicit_stream = false;
  std::string clock_id;
  std::vector<std::shared_ptr<EventSink>> legacy_sinks;
  std::vector<std::shared_ptr<TelemetrySink>> neutral_sinks;
  std::vector<TelemetrySinkDescriptor> descriptors;
  std::vector<std::byte> topology;
  std::vector<PendingEvent> pending;
  StreamState stream;
  std::uint64_t pending_bytes = 0U;
  std::uint64_t last_delivered_event_index = 0U;
  std::uint64_t last_delivered_epoch = 0U;
  std::uint64_t last_delivered_global_step = 0U;
  std::size_t next_close_sink_index = 0U;
  bool close_prepared = false;
  bool closed = false;

  void prepare() {
    validate_options(options);
    validate_telemetry_limits(telemetry.limits);
    explicit_stream = !telemetry.stream_id.empty();
    if (!explicit_stream)
      telemetry.stream_id = new_identity("training.stream/");
    clock_id = new_identity("training.clock/");
    if (descriptors.empty() || descriptors.size() > options.max_sinks ||
        telemetry.metrics.size() > options.max_metrics_per_event ||
        options.v1_sink_type_migrations.size() > options.max_sinks) {
      throw_invalid(
          "topology", "bounded nonempty destinations and metric bindings",
          "invalid size", "Use explicit finite destination/binding limits.");
    }
    for (std::size_t index = 0U; index < descriptors.size(); ++index) {
      if (!compatibility) {
        validate_telemetry_sink_descriptor(descriptors[index],
                                           telemetry.limits);
      }
      if (index != 0U &&
          descriptors[index - 1U].sink_id >= descriptors[index].sink_id) {
        throw_invalid("sinks.sink_id", "unique sorted logical sink IDs",
                      descriptors[index].sink_id,
                      "Remove duplicate destinations.");
      }
    }
    for (const auto& [old_type, new_type] : options.v1_sink_type_migrations) {
      if (old_type.empty() || new_type.empty() ||
          old_type.size() > options.max_string_bytes ||
          new_type.size() > options.max_string_bytes) {
        throw_invalid("migration.types", "bounded explicit implementation IDs",
                      "invalid",
                      "Declare only supported N-1 implementation pairs.");
      }
    }
    EventEnvelope probe;
    probe.event = {.kind = CallbackEventKind::run_started,
                   .phase = CallbackPhase::preparing,
                   .event_index = 1U,
                   .run_id = "run.validation",
                   .trace_id = "trace.validation",
                   .epoch = 0U,
                   .global_step = 0U,
                   .metrics = {}};
    for (const auto& [name, binding] : telemetry.metrics) {
      static_cast<void>(binding);
      probe.event.metrics.emplace(name, 0.0);
    }
    const TelemetryTime probe_time{
        .wall_time_unix_ns = 0, .monotonic_time_ns = 0U, .clock_id = clock_id};
    if (compatibility) {
      adapt_training_compatibility_event(probe, telemetry.stream_id, probe_time,
                                         telemetry.limits);
    } else {
      validate_event_shape(probe.event, options);
      adapt_training_event(probe, telemetry, probe_time);
    }
    topology = encode_topology(options, descriptors, telemetry, compatibility);
    if (topology.size() > options.max_topology_bytes) {
      throw_exhausted("topology", std::to_string(options.max_topology_bytes),
                      std::to_string(topology.size()),
                      "Reduce frozen bindings or raise the explicit limit.");
    }
    pending.reserve(options.max_pending_events);
  }

  TelemetryTime capture_time() const {
    const auto wall = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();
    const auto monotonic =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count();
    if (monotonic < 0) {
      throw_incompatible("clock", "nonnegative monotonic clock", "negative",
                         "Use a supported steady clock.");
    }
    return {.wall_time_unix_ns = static_cast<std::int64_t>(wall),
            .monotonic_time_ns = static_cast<std::uint64_t>(monotonic),
            .clock_id = clock_id};
  }

  TelemetryEnvelope adapt(const EventEnvelope& source, std::uint8_t format,
                          const TelemetryTime& time,
                          const TrainingTelemetryAdapterOptions& config) const {
    if (format == 3U)
      return migrate_training_envelope_v1(source, config.limits);
    if (format == 2U && compatibility) {
      return adapt_training_compatibility_event(source, config.stream_id, time,
                                                config.limits);
    }
    if (format == 1U && !compatibility) {
      return adapt_training_event(source, config, time);
    }
    throw_serialization("pending.format", "matching live or N-1 snapshot",
                        std::to_string(format),
                        "Reject changed producer semantics.");
  }

  std::uint64_t charge(std::size_t snapshot_size,
                       std::size_t canonical_size) const {
    if (snapshot_size > options.max_event_bytes ||
        canonical_size > options.max_event_bytes) {
      throw_exhausted("queue.event_bytes",
                      std::to_string(options.max_event_bytes),
                      std::to_string(std::max(snapshot_size, canonical_size)),
                      "Reduce one event or raise its explicit bound.");
    }
    const auto snapshot = static_cast<std::uint64_t>(snapshot_size);
    const auto canonical = static_cast<std::uint64_t>(canonical_size);
    constexpr std::uint64_t overhead = 16U;
    if (snapshot > std::numeric_limits<std::uint64_t>::max() - overhead ||
        canonical >
            std::numeric_limits<std::uint64_t>::max() - overhead - snapshot) {
      throw_exhausted("queue.pending_bytes", "representable node bytes",
                      "overflow", "Reduce event bounds.");
    }
    return snapshot + canonical + overhead;
  }

  void check_control_budget(std::uint64_t charged, std::uint64_t count,
                            std::string_view run_id,
                            std::string_view stream_id) const {
    constexpr std::uint64_t fixed =
        8U + kControlMagic.size() + 2U + 8U + 8U * 8U + 8U;
    const auto maximum = options.max_control_bytes;
    std::uint64_t remaining = maximum;
    const auto take = [&](std::uint64_t size) {
      if (size > remaining) {
        throw_exhausted("queue.control_bytes", std::to_string(maximum),
                        "checkpoint metadata exceeds budget",
                        "Drain events or raise the explicit control bound.");
      }
      remaining -= size;
    };
    take(fixed);
    take(static_cast<std::uint64_t>(run_id.size()));
    take(static_cast<std::uint64_t>(stream_id.size()));
    if (count > remaining / 17U) {
      throw_exhausted("queue.control_bytes", std::to_string(maximum),
                      "too many node headers",
                      "Raise the explicit control bound.");
    }
    take(count * 17U);
    take(charged);
  }
};

EventOutbox::EventOutbox(std::vector<std::shared_ptr<EventSink>> sinks,
                         EventOutboxOptions options)
    : impl_(std::make_unique<Impl>()) {
  validate_options(options);
  if (sinks.empty() || sinks.size() > options.max_sinks) {
    throw_invalid("sinks", "bounded nonempty sinks",
                  std::to_string(sinks.size()),
                  "Attach at least one bounded event destination.");
  }
  for (const auto& sink : sinks) {
    if (!sink)
      throw_invalid("sinks", "non-null owners", "null", "Construct each sink.");
    validate_sink_descriptor(sink->descriptor(), options);
  }
  std::sort(sinks.begin(), sinks.end(),
            [](const auto& left, const auto& right) {
              return left->descriptor().sink_id < right->descriptor().sink_id;
            });
  impl_->compatibility = true;
  impl_->options = std::move(options);
  impl_->telemetry.limits = compatibility_limits(impl_->options);
  for (const auto& sink : sinks) {
    const auto& source = sink->descriptor();
    impl_->descriptors.push_back({.schema_version = source.schema_version,
                                  .type_id = source.type_id,
                                  .sink_id = source.sink_id,
                                  .destination_id = source.destination_id,
                                  .role = TelemetrySinkRole::observer});
  }
  impl_->legacy_sinks = std::move(sinks);
  impl_->prepare();
}

EventOutbox::EventOutbox(std::vector<std::shared_ptr<TelemetrySink>> sinks,
                         const TrainingTelemetryAdapterOptions& telemetry,
                         EventOutboxOptions options)
    : impl_(std::make_unique<Impl>()) {
  validate_options(options);
  validate_telemetry_limits(telemetry.limits);
  if (sinks.empty() || sinks.size() > options.max_sinks) {
    throw_invalid("sinks", "bounded nonempty sinks",
                  std::to_string(sinks.size()),
                  "Attach at least one neutral destination.");
  }
  for (const auto& sink : sinks) {
    if (!sink)
      throw_invalid("sinks", "non-null owners", "null", "Construct each sink.");
    validate_telemetry_sink_descriptor(sink->descriptor(), telemetry.limits);
  }
  std::sort(sinks.begin(), sinks.end(),
            [](const auto& left, const auto& right) {
              return left->descriptor().sink_id < right->descriptor().sink_id;
            });
  impl_->options = std::move(options);
  impl_->telemetry = telemetry;
  for (const auto& sink : sinks)
    impl_->descriptors.push_back(sink->descriptor());
  impl_->neutral_sinks = std::move(sinks);
  impl_->prepare();
}

EventOutbox::~EventOutbox() = default;

void EventOutbox::enqueue(const CallbackEvent& event,
                          const CallbackDispatchResult& callback_result) {
  if (impl_->closed || impl_->close_prepared) {
    throw NotReady("Event outbox is closed or finalizing");
  }
  validate_event_shape(event, impl_->options);
  validate_callback_result(callback_result, impl_->options);
  auto next_stream = validate_transition(event, impl_->stream);
  if (impl_->pending.size() >= impl_->options.max_pending_events) {
    throw_exhausted("queue.events",
                    std::to_string(impl_->options.max_pending_events),
                    std::to_string(impl_->pending.size() + 1U),
                    "Drain pending evidence before accepting more work.");
  }
  Impl::PendingEvent candidate;
  candidate.snapshot = {
      .schema_version = 1U, .event = event, .callback_result = callback_result};
  candidate.format = impl_->compatibility ? 2U : 1U;
  candidate.snapshot_bytes = encode_envelope(candidate.snapshot);
  candidate.envelope = impl_->adapt(candidate.snapshot, candidate.format,
                                    impl_->capture_time(), impl_->telemetry);
  candidate.canonical =
      encode_telemetry_envelope(candidate.envelope, impl_->telemetry.limits);
  candidate.encoded_bytes = impl_->charge(candidate.snapshot_bytes.size(),
                                          candidate.canonical.size());
  if (candidate.encoded_bytes >
      impl_->options.max_pending_bytes - impl_->pending_bytes) {
    throw_exhausted("queue.pending_bytes",
                    std::to_string(impl_->options.max_pending_bytes),
                    std::to_string(candidate.encoded_bytes),
                    "Drain pending events before accepting more work.");
  }
  const auto charged = candidate.encoded_bytes;
  impl_->check_control_budget(
      impl_->pending_bytes + charged,
      static_cast<std::uint64_t>(impl_->pending.size()) + 1U,
      next_stream.run_id, impl_->telemetry.stream_id);
  impl_->pending.push_back(std::move(candidate));
  impl_->pending_bytes += charged;
  impl_->stream = std::move(next_stream);
}

bool EventOutbox::drain_one(std::stop_token stop, TelemetryDeadline deadline) {
  if (impl_->pending.empty()) return false;
  auto& pending = impl_->pending.front();
  while (pending.next_sink_index < impl_->descriptors.size()) {
    check_telemetry_operation(stop, deadline);
    const auto index = pending.next_sink_index;
    if (impl_->compatibility) {
      impl_->legacy_sinks[index]->publish(pending.snapshot, stop);
    } else {
      const std::span<const TelemetryEnvelope> batch(&pending.envelope, 1U);
      const auto receipt =
          impl_->neutral_sinks[index]->emit(batch, stop, deadline);
      validate_telemetry_emit_result(impl_->descriptors[index], batch, receipt,
                                     impl_->telemetry.limits);
    }
    ++pending.next_sink_index;
  }
  impl_->last_delivered_event_index = pending.snapshot.event.event_index;
  impl_->last_delivered_epoch = pending.snapshot.event.epoch;
  impl_->last_delivered_global_step = pending.snapshot.event.global_step;
  impl_->pending_bytes -= pending.encoded_bytes;
  impl_->pending.erase(impl_->pending.begin());
  return true;
}

void EventOutbox::drain(std::stop_token stop, TelemetryDeadline deadline) {
  while (drain_one(stop, deadline)) {
  }
}

void EventOutbox::flush(std::stop_token stop, TelemetryDeadline deadline) {
  if (impl_->closed) return;
  if (impl_->close_prepared)
    throw NotReady("Retry outbox close after partial finalization");
  drain(stop, deadline);
  for (std::size_t index = 0U; index < impl_->descriptors.size(); ++index) {
    check_telemetry_operation(stop, deadline);
    if (impl_->compatibility)
      impl_->legacy_sinks[index]->flush(stop);
    else
      impl_->neutral_sinks[index]->flush(stop, deadline);
  }
}

void EventOutbox::close(std::stop_token stop, TelemetryDeadline deadline) {
  if (impl_->closed) return;
  if (!impl_->close_prepared) {
    flush(stop, deadline);
    impl_->close_prepared = true;
  }
  while (impl_->next_close_sink_index < impl_->descriptors.size()) {
    check_telemetry_operation(stop, deadline);
    const auto index = impl_->next_close_sink_index;
    if (impl_->compatibility)
      impl_->legacy_sinks[index]->close(stop);
    else
      impl_->neutral_sinks[index]->close(stop, deadline);
    ++impl_->next_close_sink_index;
  }
  impl_->closed = true;
}

bool EventOutbox::closed() const noexcept { return impl_->closed; }
std::string_view EventOutbox::telemetry_stream_id() const noexcept {
  return impl_->telemetry.stream_id;
}
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
          .schema_version = 2U,
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
    throw NotReady("Cannot checkpoint an outbox during or after close");
  }
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
  append_string(control, impl_->telemetry.stream_id);
  for (const auto& pending : impl_->pending) {
    append_u8(control, pending.format);
    append_u64(control, static_cast<std::uint64_t>(pending.next_sink_index));
    append_u64(control, pending.encoded_bytes);
    append_bytes(control, pending.snapshot_bytes);
    append_bytes(control, pending.canonical);
  }
  if (control.size() > impl_->options.max_control_bytes) {
    throw_exhausted("save_state",
                    std::to_string(impl_->options.max_control_bytes),
                    std::to_string(control.size()),
                    "Drain events or raise checkpoint limits.");
  }
  writer.write_blob(kTopologyField, impl_->topology);
  writer.write_blob(kControlField, control);
}

void EventOutbox::load_state(const StateReader& reader, RestorePolicy policy) {
  if (impl_->closed || impl_->close_prepared)
    throw NotReady("Cannot restore a closed outbox");
  auto expected_manifest = describe_state();
  const auto version = reader.manifest().schema_version;
  const bool old = version == 1U;
  if (old) {
    expected_manifest.schema_version = 1U;
    expected_manifest.fields[0].media_type =
        "application/vnd.jyppx.event-outbox-topology-v1";
    expected_manifest.fields[1].media_type =
        "application/vnd.jyppx.event-outbox-control-v1";
  }
  validate_manifest(reader.manifest(), expected_manifest, policy);
  if (!reader.contains(kTopologyField) || !reader.contains(kControlField)) {
    throw_serialization(
        "state.fields", "both complete required blobs", "missing",
        "Keep the original checkpoint and reject partial state.");
  }
  const auto topology =
      reader.read_blob(kTopologyField, impl_->options.max_topology_bytes);
  auto source_options = impl_->options;
  if (old) {
    source_options =
        decode_v1_topology(topology, impl_->options, impl_->descriptors);
  } else if (topology.size() != impl_->topology.size() ||
             !std::equal(topology.begin(), topology.end(),
                         impl_->topology.begin())) {
    throw_incompatible(
        "topology", "identical frozen sinks, bindings, and limits", "different",
        "Restore with the saved observation semantics.");
  }
  const auto bytes =
      reader.read_blob(kControlField, source_options.max_control_bytes);
  ByteReader control(bytes);
  const auto magic = old ? kControlMagicV1 : kControlMagic;
  if (control.read_string("control.magic", magic.size()) != magic) {
    throw_serialization("control.magic", std::string(magic), "different",
                        "Reject an unsupported outbox encoding.");
  }
  const auto initialized = control.read_u8("control.initialized");
  const auto terminal = control.read_u8("control.terminal");
  if (initialized > 1U || terminal > 1U) {
    throw_serialization("control.flags", "zero or one", "invalid",
                        "Reject invalid boolean state encoding.");
  }
  StreamState stream{
      .initialized = initialized == 1U,
      .terminal = terminal == 1U,
      .run_id = control.read_string("control.run_id",
                                    source_options.max_string_bytes),
      .last_event_index = control.read_u64("control.last_event_index"),
      .last_epoch = control.read_u64("control.last_epoch"),
      .last_global_step = control.read_u64("control.last_global_step")};
  const auto delivered = control.read_u64("control.last_delivered_event_index");
  const auto delivered_epoch = control.read_u64("control.last_delivered_epoch");
  const auto delivered_step =
      control.read_u64("control.last_delivered_global_step");
  const auto stored_charge = control.read_u64("control.pending_bytes");
  const auto count = control.read_u64("control.pending_count");
  if (count > source_options.max_pending_events ||
      stored_charge > source_options.max_pending_bytes) {
    throw_exhausted("state.pending", "configured queue bounds", "out of range",
                    "Reject an oversized checkpoint before allocation.");
  }
  auto config = impl_->telemetry;
  if (!old) {
    config.stream_id = control.read_string("control.stream_id",
                                           config.limits.max_identifier_bytes);
  } else if (stream.initialized) {
    EventEnvelope seed;
    seed.event = {.kind = CallbackEventKind::run_started,
                  .phase = CallbackPhase::preparing,
                  .event_index = 1U,
                  .run_id = stream.run_id,
                  .trace_id = "migration.unavailable",
                  .epoch = 0U,
                  .global_step = 0U,
                  .metrics = {}};
    config.stream_id =
        migrate_training_envelope_v1(seed, config.limits).identity.stream_id;
  }
  if (impl_->explicit_stream &&
      config.stream_id != impl_->telemetry.stream_id) {
    throw_incompatible(
        "control.stream_id", impl_->telemetry.stream_id, config.stream_id,
        "Resume the saved producer stream; never reset its sequence.");
  }
  // Validate even an empty-state stream before accepting it into live state.
  EventEnvelope identity_probe;
  identity_probe.event = {.kind = CallbackEventKind::run_started,
                          .phase = CallbackPhase::preparing,
                          .event_index = 1U,
                          .run_id = "run.validation",
                          .trace_id = "trace.validation",
                          .epoch = 0U,
                          .global_step = 0U,
                          .metrics = {}};
  impl_->adapt(identity_probe, impl_->compatibility ? 2U : 1U, {}, config);
  std::vector<Impl::PendingEvent> pending;
  pending.reserve(static_cast<std::size_t>(count));
  StreamState replay{.initialized = delivered != 0U,
                     .terminal = false,
                     .run_id = delivered == 0U ? std::string{} : stream.run_id,
                     .last_event_index = delivered,
                     .last_epoch = delivered_epoch,
                     .last_global_step = delivered_step};
  std::uint64_t charged_total = 0U;
  std::uint64_t old_total = 0U;
  for (std::uint64_t index = 0U; index < count; ++index) {
    Impl::PendingEvent node;
    node.format = old ? 3U : control.read_u8("pending.format");
    const auto sink = control.read_u64("pending.next_sink_index");
    const auto charged = control.read_u64("pending.encoded_bytes");
    if (sink > impl_->descriptors.size() || (index != 0U && sink != 0U)) {
      throw_serialization(
          "pending.sink_cursor", "only the front may be partly delivered",
          std::to_string(sink), "Reject impossible delivery cursors.");
    }
    node.next_sink_index = static_cast<std::size_t>(sink);
    const auto snapshot =
        control.read_bytes("pending.snapshot", source_options.max_event_bytes);
    node.snapshot = decode_envelope(snapshot, source_options);
    node.snapshot_bytes.assign(snapshot.begin(), snapshot.end());
    if (encode_envelope(node.snapshot) != node.snapshot_bytes) {
      throw_serialization("pending.snapshot", "canonical callback snapshot",
                          "different",
                          "Reject noncanonical callback ordering.");
    }
    if (old) {
      const auto expected_old =
          static_cast<std::uint64_t>(snapshot.size()) + 16U;
      if (charged != expected_old ||
          charged > source_options.max_pending_bytes - old_total) {
        throw_serialization("pending.v1.charge", "exact legacy node accounting",
                            "different",
                            "Keep the unmodified legacy checkpoint.");
      }
      old_total += charged;
      node.envelope =
          migrate_training_envelope_v1(node.snapshot, config.limits);
      node.canonical = encode_telemetry_envelope(node.envelope, config.limits);
    } else {
      const auto canonical = control.read_bytes(
          "pending.canonical", std::min(impl_->options.max_event_bytes,
                                        config.limits.max_envelope_bytes));
      node.envelope = decode_telemetry_envelope(canonical, config.limits);
      node.canonical.assign(canonical.begin(), canonical.end());
      const auto expected =
          impl_->adapt(node.snapshot, node.format, node.envelope.time, config);
      if (encode_telemetry_envelope(expected, config.limits) !=
          node.canonical) {
        throw_serialization(
            "pending.canonical", "exact staged snapshot and telemetry",
            "different",
            "Never recapture time or change saved metric semantics.");
      }
    }
    if (node.envelope.identity.stream_id != config.stream_id) {
      throw_serialization(
          "pending.stream_id", config.stream_id, "different",
          "Keep every pending operation in its saved producer stream.");
    }
    replay = validate_transition(node.snapshot.event, replay);
    node.encoded_bytes =
        impl_->charge(node.snapshot_bytes.size(), node.canonical.size());
    if ((!old && charged != node.encoded_bytes) ||
        node.encoded_bytes > impl_->options.max_pending_bytes - charged_total) {
      throw_exhausted("pending.charge", "exact bounded migrated node bytes",
                      "different",
                      "Reject inconsistent accounting or increase explicit "
                      "migration capacity.");
    }
    charged_total += node.encoded_bytes;
    pending.push_back(std::move(node));
  }
  control.require_finished("control");
  impl_->check_control_budget(charged_total, count, stream.run_id,
                              config.stream_id);
  if (stored_charge != (old ? old_total : charged_total) ||
      delivered > stream.last_event_index ||
      delivered_epoch > stream.last_epoch ||
      delivered_step > stream.last_global_step ||
      (delivered == 0U && (delivered_epoch != 0U || delivered_step != 0U)) ||
      (!stream.initialized &&
       (stream.terminal || !stream.run_id.empty() ||
        stream.last_event_index != 0U || stream.last_epoch != 0U ||
        stream.last_global_step != 0U || delivered != 0U ||
        !pending.empty())) ||
      (stream.initialized &&
       (stream.last_event_index == 0U || !is_identifier(stream.run_id) ||
        stream.run_id.size() > impl_->options.max_string_bytes)) ||
      (pending.empty() && (delivered != stream.last_event_index ||
                           delivered_epoch != stream.last_epoch ||
                           delivered_step != stream.last_global_step)) ||
      (!pending.empty() &&
       (replay.run_id != stream.run_id ||
        replay.last_event_index != stream.last_event_index ||
        replay.last_epoch != stream.last_epoch ||
        replay.last_global_step != stream.last_global_step ||
        replay.terminal != stream.terminal))) {
    throw_serialization(
        "control", "internally consistent stream and pending suffix",
        "inconsistent", "Reject partial or reordered outbox state.");
  }
  // All allocations, conversion, topology and receipt-cursor checks precede
  // commit.
  impl_->pending.swap(pending);
  impl_->telemetry = std::move(config);
  impl_->stream = std::move(stream);
  impl_->pending_bytes = charged_total;
  impl_->last_delivered_event_index = delivered;
  impl_->last_delivered_epoch = delivered_epoch;
  impl_->last_delivered_global_step = delivered_step;
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
