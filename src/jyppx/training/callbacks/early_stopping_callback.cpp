// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/training/callbacks/early_stopping_callback.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace jyppx::training::callbacks {
namespace {

constexpr char kComponent[] = "training.callbacks.early_stopping";
constexpr char kTopologyField[] = "topology";
constexpr char kStateField[] = "state";
constexpr char kTopologyMediaType[] =
    "application/vnd.jyppx.early-stopping-topology-v1";
constexpr char kStateMediaType[] =
    "application/vnd.jyppx.early-stopping-state-v1";
constexpr char kTypeId[] = "jyppx.training.callbacks.early_stopping.v1";
constexpr std::string_view kTopologyMagic = "JYPPX-EARLY-STOPPING-TOPOLOGY-V1";
constexpr std::string_view kStateMagic = "JYPPX-EARLY-STOPPING-STATE-V1";

[[noreturn]] void throw_invalid(std::string operation, std::string expected,
                                std::string actual, std::string hint) {
  throw InvalidArgument("Invalid early-stopping callback contract",
                        ErrorDetails{.component = kComponent,
                                     .operation = std::move(operation),
                                     .expected = std::move(expected),
                                     .actual = std::move(actual),
                                     .hint = std::move(hint)});
}

[[noreturn]] void throw_incompatible(std::string operation,
                                     std::string expected, std::string actual,
                                     std::string hint) {
  throw CompatibilityError("Early-stopping checkpoint is incompatible",
                           ErrorDetails{.component = kComponent,
                                        .operation = std::move(operation),
                                        .expected = std::move(expected),
                                        .actual = std::move(actual),
                                        .hint = std::move(hint)});
}

[[noreturn]] void throw_serialization(std::string operation,
                                      std::string expected, std::string actual,
                                      std::string hint) {
  throw SerializationError("Malformed early-stopping checkpoint state",
                           ErrorDetails{.component = kComponent,
                                        .operation = std::move(operation),
                                        .expected = std::move(expected),
                                        .actual = std::move(actual),
                                        .hint = std::move(hint)});
}

[[noreturn]] void throw_exhausted(std::string operation, std::string expected,
                                  std::string actual, std::string hint) {
  throw ResourceExhausted("Early-stopping resource limit exceeded",
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
                          "Reject incomplete early-stopping state.");
    }
  }

  std::span<const std::byte> bytes_;  ///< Immutable encoded checkpoint bytes.
  std::size_t offset_ = 0;            ///< Next unread byte offset.
};

std::vector<std::byte> encode_topology(const EarlyStoppingOptions& options) {
  std::vector<std::byte> output;
  append_string(output, kTopologyMagic);
  append_u32(output, 1U);
  append_string(output, kTypeId);
  append_string(output, options.callback_id);
  append_string(output, options.monitor);
  append_u64(output, options.patience);
  append_double(output, options.min_delta);
  append_u8(output, static_cast<std::uint8_t>(options.mode));
  append_i32(output, options.priority);
  return output;
}

std::vector<std::byte> encode_state(bool has_best, double best_value,
                                    std::uint64_t non_improving,
                                    bool should_stop) {
  std::vector<std::byte> output;
  append_string(output, kStateMagic);
  append_u8(output, has_best ? 1U : 0U);
  append_double(output, best_value);
  append_u64(output, non_improving);
  append_u8(output, should_stop ? 1U : 0U);
  return output;
}

struct DecodedState {
  bool has_best = false;            ///< Whether a best value was observed.
  double best_value = 0.0;          ///< Restored best finite metric value.
  std::uint64_t non_improving = 0;  ///< Restored patience counter.
  bool should_stop = false;         ///< Restored stop decision.
};

DecodedState decode_state(std::span<const std::byte> encoded,
                          std::uint64_t patience) {
  ByteReader reader(encoded);
  if (reader.read_string("state.magic", kStateMagic.size()) != kStateMagic) {
    throw_serialization("state.magic", std::string(kStateMagic),
                        "different magic",
                        "Restore only current early-stopping state.");
  }
  const auto has_best = reader.read_u8("state.has_best");
  const auto best_value = reader.read_double("state.best_value");
  const auto non_improving = reader.read_u64("state.non_improving");
  const auto should_stop = reader.read_u8("state.should_stop");
  reader.require_finished("state");
  if (has_best > 1U || should_stop > 1U) {
    throw_serialization("state.flags", "boolean flag values", "out of range",
                        "Reject malformed early-stopping flags.");
  }
  DecodedState result{.has_best = has_best == 1U,
                      .best_value = best_value,
                      .non_improving = non_improving,
                      .should_stop = should_stop == 1U};
  if (!result.has_best) {
    if (result.best_value != 0.0 || result.non_improving != 0U ||
        result.should_stop) {
      throw_serialization("state.uninitialized", "canonical zero state",
                          "non-zero state",
                          "Reject internally inconsistent callback state.");
    }
  } else if (!std::isfinite(result.best_value)) {
    throw_serialization("state.best_value", "a finite value", "non-finite",
                        "Reject corrupt early-stopping state.");
  }
  if (result.should_stop != (result.non_improving > patience)) {
    throw_serialization("state.should_stop",
                        "a decision matching the patience counter",
                        result.should_stop ? "true" : "false",
                        "Reject internally inconsistent callback state.");
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
                       "Restore current exact state for this callback.");
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
          "Restore the current callback field schema.");
    }
  }
}

}  // namespace

std::string_view early_stopping_mode_name(EarlyStoppingMode mode) noexcept {
  switch (mode) {
    case EarlyStoppingMode::minimize:
      return "minimize";
    case EarlyStoppingMode::maximize:
      return "maximize";
  }
  return "unknown";
}

EarlyStoppingCallback::EarlyStoppingCallback(
    EarlyStoppingOptions options,
    EarlyStoppingCheckpointOptions checkpoint_options)
    : options_(std::move(options)),
      checkpoint_options_(checkpoint_options),
      descriptor_{.schema_version = 1,
                  .type_id = kTypeId,
                  .callback_id = options_.callback_id,
                  .priority = options_.priority,
                  .subscriptions = {CallbackEventKind::epoch_finished},
                  .allowed_commands = {CallbackCommandKind::stop},
                  .stateful = true} {
  if (!is_identifier(options_.callback_id)) {
    throw_invalid("options.callback_id", "a stable identifier",
                  options_.callback_id,
                  "Use ASCII letters, digits, dot, underscore, or hyphen.");
  }
  if (!is_identifier(options_.monitor)) {
    throw_invalid("options.monitor", "a stable metric identifier",
                  options_.monitor,
                  "Use the exact finite metric key published by the event.");
  }
  if (!std::isfinite(options_.min_delta) || options_.min_delta < 0.0) {
    throw_invalid("options.min_delta", "a finite non-negative threshold",
                  std::to_string(options_.min_delta),
                  "Use an absolute finite improvement threshold.");
  }
  if (early_stopping_mode_name(options_.mode) == "unknown") {
    throw_invalid("options.mode", "minimize or maximize", "unknown",
                  "Select a stable early-stopping comparison mode.");
  }
  if (checkpoint_options_.max_topology_bytes == 0U ||
      checkpoint_options_.max_state_bytes == 0U) {
    throw_invalid("checkpoint_options", "positive resource limits", "zero",
                  "Configure explicit positive checkpoint limits.");
  }
  const auto topology = encode_topology(options_);
  if (topology.size() > checkpoint_options_.max_topology_bytes) {
    throw_exhausted("topology", "topology within configured limit",
                    std::to_string(topology.size()),
                    "Reduce identifiers or raise the explicit limit.");
  }
}

const CallbackDescriptor& EarlyStoppingCallback::descriptor() const noexcept {
  return descriptor_;
}

CallbackCommandRequest EarlyStoppingCallback::handle(
    const CallbackEvent& event) {
  if (event.kind != CallbackEventKind::epoch_finished) {
    throw_invalid("handle.event", "epoch_finished",
                  std::string(callback_event_kind_name(event.kind)),
                  "Invoke only for the declared subscription.");
  }
  if (should_stop_) {
    return {.kind = CallbackCommandKind::stop,
            .reason = "early-stopping patience already exceeded for " +
                      options_.monitor};
  }
  const auto metric = event.metrics.find(options_.monitor);
  if (metric == event.metrics.end() || !std::isfinite(metric->second)) {
    throw_invalid("handle.metric", "one finite " + options_.monitor,
                  metric == event.metrics.end() ? "missing" : "non-finite",
                  "Publish the monitored epoch metric before dispatch.");
  }

  const auto current = metric->second;
  bool improved = !has_best_;
  if (has_best_) {
    const auto delta = options_.mode == EarlyStoppingMode::minimize
                           ? static_cast<long double>(best_value_) - current
                           : static_cast<long double>(current) - best_value_;
    improved = delta > static_cast<long double>(options_.min_delta);
  }
  if (improved) {
    has_best_ = true;
    best_value_ = current;
    non_improving_ = 0;
    return {};
  }
  if (non_improving_ == std::numeric_limits<std::uint64_t>::max()) {
    throw_exhausted("handle.non_improving", "a count below uint64 maximum",
                    "uint64 maximum", "Reset or start a new callback run.");
  }
  ++non_improving_;
  should_stop_ = non_improving_ > options_.patience;
  if (!should_stop_) {
    return {};
  }
  return {.kind = CallbackCommandKind::stop,
          .reason = "early-stopping patience exceeded for " + options_.monitor};
}

void EarlyStoppingCallback::reset() noexcept {
  has_best_ = false;
  best_value_ = 0.0;
  non_improving_ = 0;
  should_stop_ = false;
}

bool EarlyStoppingCallback::should_stop() const noexcept {
  return should_stop_;
}

std::uint64_t EarlyStoppingCallback::non_improving_observations()
    const noexcept {
  return non_improving_;
}

bool EarlyStoppingCallback::has_best_value() const noexcept {
  return has_best_;
}

double EarlyStoppingCallback::best_value() const noexcept {
  return best_value_;
}

std::string EarlyStoppingCallback::participant_id() const {
  return options_.callback_id;
}

StateManifest EarlyStoppingCallback::describe_state() const {
  return {.participant_id = options_.callback_id,
          .schema_version = 1,
          .restore_level = RestoreLevel::exact,
          .fields = {{.name = kTopologyField,
                      .media_type = kTopologyMediaType,
                      .max_bytes = checkpoint_options_.max_topology_bytes},
                     {.name = kStateField,
                      .media_type = kStateMediaType,
                      .max_bytes = checkpoint_options_.max_state_bytes}},
          .degradation_reasons = {}};
}

void EarlyStoppingCallback::save_state(StateWriter& writer) const {
  const auto topology = encode_topology(options_);
  const auto state =
      encode_state(has_best_, best_value_, non_improving_, should_stop_);
  if (topology.size() > checkpoint_options_.max_topology_bytes ||
      state.size() > checkpoint_options_.max_state_bytes) {
    throw_exhausted("save_state", "state within configured limits", "oversized",
                    "Raise limits before callback construction.");
  }
  writer.write_blob(kTopologyField, std::span<const std::byte>(
                                        topology.data(), topology.size()));
  writer.write_blob(kStateField,
                    std::span<const std::byte>(state.data(), state.size()));
}

void EarlyStoppingCallback::load_state(const StateReader& reader,
                                       RestorePolicy policy) {
  const auto expected_manifest = describe_state();
  validate_manifest_shape(reader.manifest(), expected_manifest, policy);
  const auto stored_topology =
      reader.read_blob(kTopologyField, checkpoint_options_.max_topology_bytes);
  const auto expected_topology = encode_topology(options_);
  if (stored_topology.size() != expected_topology.size() ||
      !std::equal(stored_topology.begin(), stored_topology.end(),
                  expected_topology.begin())) {
    throw_incompatible("restore.topology", "identical early-stopping semantics",
                       "different canonical topology",
                       "Restore with the same callback identity and options.");
  }
  const auto candidate = decode_state(
      reader.read_blob(kStateField, checkpoint_options_.max_state_bytes),
      options_.patience);
  has_best_ = candidate.has_best;
  best_value_ = candidate.best_value;
  non_improving_ = candidate.non_improving;
  should_stop_ = candidate.should_stop;
}

}  // namespace jyppx::training::callbacks
