// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/optim/step_lr_scheduler.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <iomanip>
#include <limits>
#include <new>
#include <set>
#include <span>
#include <sstream>
#include <string_view>
#include <type_traits>
#include <utility>

namespace jyppx::optim {
namespace {

constexpr char kComponent[] = "optim.lr_scheduler";
constexpr char kTypeId[] = "jyppx.scheduler.step_lr/v1";
constexpr std::uint32_t kDescriptorSchemaVersion = 1;
constexpr std::uint32_t kParticipantSchemaVersion = 3;
constexpr std::uint32_t kTopologyVersion = 1;
constexpr std::uint32_t kControlVersion = 3;
constexpr std::size_t kConstructionMaximumIdentifierBytes = 4096;
constexpr std::array<std::byte, 8> kTopologyMagic = {
    std::byte{'J'}, std::byte{'Y'}, std::byte{'P'}, std::byte{'P'},
    std::byte{'X'}, std::byte{'S'}, std::byte{'L'}, std::byte{'D'}};
constexpr std::array<std::byte, 8> kControlMagic = {
    std::byte{'J'}, std::byte{'Y'}, std::byte{'P'}, std::byte{'P'},
    std::byte{'X'}, std::byte{'S'}, std::byte{'L'}, std::byte{'C'}};

struct SchedulerSnapshot {
  std::vector<std::byte> topology;  ///< Immutable canonical descriptor bytes.
  std::vector<std::byte> control;   ///< Mutable counter and learning rates.
  StateManifest manifest;           ///< Complete exact state contract.
};

struct DecodedControl {
  detail::SchedulerState state;        ///< Restored scheduler state tree.
  std::vector<double> learning_rates;  ///< Restored rates in stable order.
};

[[noreturn]] void throw_invalid(std::string operation, std::string expected,
                                std::string actual, std::string hint) {
  throw InvalidArgument("Invalid learning-rate scheduler state",
                        ErrorDetails{.component = kComponent,
                                     .operation = std::move(operation),
                                     .expected = std::move(expected),
                                     .actual = std::move(actual),
                                     .hint = std::move(hint)});
}

[[noreturn]] void throw_serialization(std::string operation,
                                      std::string expected, std::string actual,
                                      std::string hint) {
  throw SerializationError("Malformed scheduler checkpoint state",
                           ErrorDetails{.component = kComponent,
                                        .operation = std::move(operation),
                                        .expected = std::move(expected),
                                        .actual = std::move(actual),
                                        .hint = std::move(hint)});
}

[[noreturn]] void throw_compatibility(std::string operation,
                                      std::string expected, std::string actual,
                                      std::string hint) {
  throw CompatibilityError(
      "Scheduler checkpoint is incompatible with the current scheduler",
      ErrorDetails{.component = kComponent,
                   .operation = std::move(operation),
                   .expected = std::move(expected),
                   .actual = std::move(actual),
                   .hint = std::move(hint)});
}

[[noreturn]] void throw_resource(std::string operation, std::uint64_t expected,
                                 std::uint64_t actual, std::string hint) {
  throw ResourceExhausted("Scheduler resource limit exceeded",
                          ErrorDetails{.component = kComponent,
                                       .operation = std::move(operation),
                                       .expected = std::to_string(expected),
                                       .actual = std::to_string(actual),
                                       .hint = std::move(hint)});
}

[[noreturn]] void translate_bad_alloc(std::string_view operation) {
  throw ResourceExhausted(
      "Unable to allocate scheduler state",
      ErrorDetails{.component = kComponent,
                   .operation = std::string(operation),
                   .expected = "memory within configured scheduler limits",
                   .actual = "allocation failed",
                   .hint = "Reduce descriptor or checkpoint state size."});
}

void validate_identifier(std::string_view identifier, std::size_t maximum,
                         std::string_view operation) {
  if (identifier.empty() || identifier.size() > maximum ||
      identifier.find('\0') != std::string_view::npos) {
    throw_invalid(std::string(operation),
                  "a nonempty non-NUL stable ID within " +
                      std::to_string(maximum) + " bytes",
                  std::string(identifier),
                  "Use a bounded logical identifier instead of a path.");
  }
}

void validate_rates(std::span<const double> rates, std::size_t group_count,
                    std::string_view operation) {
  if (rates.size() != group_count) {
    throw_invalid(std::string(operation) + ".count",
                  std::to_string(group_count), std::to_string(rates.size()),
                  "Expose one learning rate per stable parameter group.");
  }
  for (const auto rate : rates) {
    if (!std::isfinite(rate) || rate < 0.0) {
      throw_invalid(std::string(operation), "finite non-negative rates",
                    std::to_string(rate),
                    "Validate optimizer rates before scheduling.");
    }
  }
}

void append_u8(std::vector<std::byte>& output, std::uint8_t value) {
  output.push_back(static_cast<std::byte>(value));
}

void append_u32(std::vector<std::byte>& output, std::uint32_t value) {
  for (std::size_t shift = 0; shift < 32; shift += 8) {
    output.push_back(static_cast<std::byte>((value >> shift) & 0xffU));
  }
}

void append_u64(std::vector<std::byte>& output, std::uint64_t value) {
  for (std::size_t shift = 0; shift < 64; shift += 8) {
    output.push_back(static_cast<std::byte>((value >> shift) & 0xffULL));
  }
}

void append_string(std::vector<std::byte>& output, std::string_view value) {
  if (value.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw_invalid("descriptor.string", "at most uint32 maximum bytes",
                  std::to_string(value.size()), "Use bounded descriptor IDs.");
  }
  append_u32(output, static_cast<std::uint32_t>(value.size()));
  for (const auto character : value) {
    output.push_back(static_cast<std::byte>(character));
  }
}

void append_double(std::vector<std::byte>& output, double value) {
  append_u64(output, std::bit_cast<std::uint64_t>(value));
}

void append_config_value(std::vector<std::byte>& output,
                         const SchedulerConfigValue& value) {
  append_u8(output, static_cast<std::uint8_t>(value.index() + 1));
  std::visit(
      [&output](const auto& typed) {
        using Value = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<Value, bool>) {
          append_u8(output, typed ? 1 : 0);
        } else if constexpr (std::is_same_v<Value, std::int64_t>) {
          append_u64(output, std::bit_cast<std::uint64_t>(typed));
        } else if constexpr (std::is_same_v<Value, double>) {
          append_double(output, typed);
        } else if constexpr (std::is_same_v<Value, std::string>) {
          append_string(output, typed);
        } else {
          if (typed.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw_invalid("descriptor.array", "at most uint32 elements",
                          std::to_string(typed.size()),
                          "Use a bounded typed configuration array.");
          }
          append_u32(output, static_cast<std::uint32_t>(typed.size()));
          for (const auto& element : typed) {
            if constexpr (std::is_same_v<typename Value::value_type,
                                         std::int64_t>) {
              append_u64(output, std::bit_cast<std::uint64_t>(element));
            } else if constexpr (std::is_same_v<typename Value::value_type,
                                                double>) {
              append_double(output, element);
            } else {
              append_string(output, element);
            }
          }
        }
      },
      value);
}

void append_descriptor(std::vector<std::byte>& output,
                       const SchedulerDescriptor& descriptor) {
  append_u32(output, descriptor.schema_version);
  append_string(output, descriptor.type_id);
  append_u8(output, static_cast<std::uint8_t>(descriptor.step_mode));
  if (descriptor.config.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw_invalid("descriptor.config", "at most uint32 entries",
                  std::to_string(descriptor.config.size()),
                  "Use a bounded typed scheduler configuration.");
  }
  append_u32(output, static_cast<std::uint32_t>(descriptor.config.size()));
  for (const auto& [key, value] : descriptor.config) {
    append_string(output, key);
    append_config_value(output, value);
  }
  append_string(output, descriptor.optimizer_id);
  if (descriptor.optimizer_group_ids.size() >
      std::numeric_limits<std::uint32_t>::max()) {
    throw_invalid("descriptor.optimizer_groups", "at most uint32 groups",
                  std::to_string(descriptor.optimizer_group_ids.size()),
                  "Use a bounded optimizer topology.");
  }
  append_u32(output,
             static_cast<std::uint32_t>(descriptor.optimizer_group_ids.size()));
  for (const auto& group_id : descriptor.optimizer_group_ids) {
    append_string(output, group_id);
  }
  append_string(output, descriptor.optimizer_group_signature);
  append_u8(output, descriptor.callable_id.has_value() ? 1 : 0);
  if (descriptor.callable_id.has_value()) {
    append_string(output, *descriptor.callable_id);
  }
  if (descriptor.children.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw_invalid("descriptor.children", "at most uint32 children",
                  std::to_string(descriptor.children.size()),
                  "Use a bounded composite scheduler descriptor.");
  }
  append_u32(output, static_cast<std::uint32_t>(descriptor.children.size()));
  for (const auto& child : descriptor.children) {
    append_descriptor(output, child);
    append_string(output, child.fingerprint);
  }
}

std::string fnv1a_hex(std::span<const std::byte> bytes) {
  std::uint64_t value = 14695981039346656037ULL;
  for (const auto byte : bytes) {
    value ^= std::to_integer<std::uint8_t>(byte);
    value *= 1099511628211ULL;
  }
  std::ostringstream stream;
  stream << std::hex << std::setfill('0') << std::setw(16) << value;
  return stream.str();
}

std::string descriptor_fingerprint(const SchedulerDescriptor& descriptor) {
  std::vector<std::byte> canonical;
  append_descriptor(canonical, descriptor);
  return fnv1a_hex(canonical);
}

std::string group_signature(std::string_view optimizer_id,
                            const std::vector<std::string>& group_ids) {
  std::vector<std::byte> canonical;
  append_string(canonical, optimizer_id);
  append_u32(canonical, static_cast<std::uint32_t>(group_ids.size()));
  for (const auto& group_id : group_ids) {
    append_string(canonical, group_id);
  }
  return fnv1a_hex(canonical);
}

class Cursor {
 public:
  explicit Cursor(std::span<const std::byte> bytes) : bytes_(bytes) {}

  std::uint8_t read_u8(std::string_view operation) {
    require(1, operation);
    return std::to_integer<std::uint8_t>(bytes_[offset_++]);
  }

  std::uint32_t read_u32(std::string_view operation) {
    require(4, operation);
    std::uint32_t value = 0;
    for (std::size_t shift = 0; shift < 32; shift += 8) {
      value |= static_cast<std::uint32_t>(
                   std::to_integer<std::uint8_t>(bytes_[offset_++]))
               << shift;
    }
    return value;
  }

  std::uint64_t read_u64(std::string_view operation) {
    require(8, operation);
    std::uint64_t value = 0;
    for (std::size_t shift = 0; shift < 64; shift += 8) {
      value |= static_cast<std::uint64_t>(
                   std::to_integer<std::uint8_t>(bytes_[offset_++]))
               << shift;
    }
    return value;
  }

  std::string read_string(std::size_t maximum, std::string_view operation,
                          bool allow_empty = false) {
    const auto size = read_u32(operation);
    if ((!allow_empty && size == 0) || size > maximum) {
      throw_serialization(
          std::string(operation),
          std::string(allow_empty ? "a string within "
                                  : "a nonempty string within ") +
              std::to_string(maximum) + " bytes",
          std::to_string(size),
          "Reject unbounded or empty descriptor identities.");
    }
    require(size, operation);
    std::string value;
    value.reserve(size);
    for (std::uint32_t index = 0; index < size; ++index) {
      const auto character =
          static_cast<char>(std::to_integer<unsigned char>(bytes_[offset_++]));
      if (character == '\0') {
        throw_serialization(std::string(operation), "a non-NUL string", "NUL",
                            "Reject ambiguous descriptor identities.");
      }
      value.push_back(character);
    }
    return value;
  }

  void expect_magic(std::span<const std::byte> magic,
                    std::string_view operation) {
    require(magic.size(), operation);
    if (!std::equal(magic.begin(), magic.end(), bytes_.begin() + offset_)) {
      throw_serialization(std::string(operation), "the current format magic",
                          "different magic",
                          "Do not load legacy or unrelated scheduler state.");
    }
    offset_ += magic.size();
  }

  bool empty() const noexcept { return offset_ == bytes_.size(); }

 private:
  void require(std::size_t count, std::string_view operation) const {
    if (count > bytes_.size() - offset_) {
      throw_serialization(std::string(operation), "complete bounded metadata",
                          "truncated",
                          "Reject incomplete scheduler checkpoint fields.");
    }
  }

  std::span<const std::byte> bytes_;  ///< Immutable field bytes being decoded.
  std::size_t offset_ = 0;            ///< Next unread byte position.
};

SchedulerConfigValue read_config_value(
    Cursor& cursor, const LrSchedulerCheckpointOptions& limits) {
  const auto tag = cursor.read_u8("topology.config.type");
  switch (tag) {
    case 1: {
      const auto value = cursor.read_u8("topology.config.bool");
      if (value > 1) {
        throw_serialization("topology.config.bool", "zero or one",
                            std::to_string(value),
                            "Reject invalid boolean configuration.");
      }
      return value == 1;
    }
    case 2:
      return std::bit_cast<std::int64_t>(
          cursor.read_u64("topology.config.int64"));
    case 3: {
      const auto value =
          std::bit_cast<double>(cursor.read_u64("topology.config.double"));
      if (!std::isfinite(value)) {
        throw_serialization("topology.config.double", "a finite value",
                            "non-finite",
                            "Reject non-finite scheduler configuration.");
      }
      return value;
    }
    case 4:
      return cursor.read_string(limits.max_identifier_bytes,
                                "topology.config.string", true);
    case 5: {
      const auto count = cursor.read_u32("topology.config.int64_array.count");
      if (count > limits.max_config_array_elements) {
        throw_resource("topology.config.int64_array.count",
                       limits.max_config_array_elements, count,
                       "Reduce the scheduler configuration array.");
      }
      std::vector<std::int64_t> values;
      values.reserve(count);
      for (std::uint32_t index = 0; index < count; ++index) {
        values.push_back(std::bit_cast<std::int64_t>(
            cursor.read_u64("topology.config.int64_array.value")));
      }
      return values;
    }
    case 6: {
      const auto count = cursor.read_u32("topology.config.double_array.count");
      if (count > limits.max_config_array_elements) {
        throw_resource("topology.config.double_array.count",
                       limits.max_config_array_elements, count,
                       "Reduce the scheduler configuration array.");
      }
      std::vector<double> values;
      values.reserve(count);
      for (std::uint32_t index = 0; index < count; ++index) {
        const auto value = std::bit_cast<double>(
            cursor.read_u64("topology.config.double_array.value"));
        if (!std::isfinite(value)) {
          throw_serialization("topology.config.double_array.value",
                              "finite values", "non-finite",
                              "Reject non-finite scheduler configuration.");
        }
        values.push_back(value);
      }
      return values;
    }
    case 7: {
      const auto count = cursor.read_u32("topology.config.string_array.count");
      if (count > limits.max_config_array_elements) {
        throw_resource("topology.config.string_array.count",
                       limits.max_config_array_elements, count,
                       "Reduce the scheduler configuration array.");
      }
      std::vector<std::string> values;
      values.reserve(count);
      for (std::uint32_t index = 0; index < count; ++index) {
        values.push_back(
            cursor.read_string(limits.max_identifier_bytes,
                               "topology.config.string_array.value", true));
      }
      return values;
    }
    default:
      throw_serialization(
          "topology.config.type", "a current typed value tag",
          std::to_string(tag),
          "Do not infer configuration types from unknown tags.");
  }
}

SchedulerDescriptor read_descriptor(Cursor& cursor,
                                    const LrSchedulerCheckpointOptions& limits,
                                    std::size_t depth) {
  if (depth > limits.max_descriptor_depth) {
    throw_resource("topology.descriptor_depth", limits.max_descriptor_depth,
                   depth, "Reduce composite scheduler nesting.");
  }
  SchedulerDescriptor descriptor;
  descriptor.schema_version = cursor.read_u32("topology.schema_version");
  if (descriptor.schema_version == 0) {
    throw_serialization("topology.schema_version", "a positive version", "0",
                        "Reject an unversioned scheduler descriptor.");
  }
  descriptor.type_id =
      cursor.read_string(limits.max_identifier_bytes, "topology.type_id");
  descriptor.step_mode =
      static_cast<SchedulerStepMode>(cursor.read_u8("topology.step_mode"));
  if (scheduler_step_mode_name(descriptor.step_mode) == "unknown") {
    throw_serialization("topology.step_mode", "a declared SchedulerStepMode",
                        "unknown", "Reject unknown scheduler step semantics.");
  }
  const auto config_count = cursor.read_u32("topology.config.count");
  if (config_count > limits.max_config_entries) {
    throw_resource("topology.config.count", limits.max_config_entries,
                   config_count, "Reduce scheduler descriptor entries.");
  }
  for (std::uint32_t index = 0; index < config_count; ++index) {
    auto key =
        cursor.read_string(limits.max_identifier_bytes, "topology.config.key");
    if (!descriptor.config
             .emplace(std::move(key), read_config_value(cursor, limits))
             .second) {
      throw_serialization("topology.config.key", "unique keys", "duplicate",
                          "Reject ambiguous scheduler configuration.");
    }
  }
  descriptor.optimizer_id =
      cursor.read_string(limits.max_identifier_bytes, "topology.optimizer_id");
  const auto group_count = cursor.read_u32("topology.optimizer_groups.count");
  if (group_count == 0 || group_count > limits.max_groups) {
    throw_resource("topology.optimizer_groups.count", limits.max_groups,
                   group_count, "Use a bounded nonempty optimizer topology.");
  }
  std::set<std::string> group_ids;
  descriptor.optimizer_group_ids.reserve(group_count);
  for (std::uint32_t index = 0; index < group_count; ++index) {
    auto group_id = cursor.read_string(limits.max_identifier_bytes,
                                       "topology.optimizer_group_id");
    if (!group_ids.insert(group_id).second) {
      throw_serialization("topology.optimizer_group_id", "unique group IDs",
                          group_id, "Reject ambiguous optimizer topology.");
    }
    descriptor.optimizer_group_ids.push_back(std::move(group_id));
  }
  descriptor.optimizer_group_signature = cursor.read_string(
      limits.max_identifier_bytes, "topology.optimizer_group_signature");
  const auto expected_group_signature =
      group_signature(descriptor.optimizer_id, descriptor.optimizer_group_ids);
  if (descriptor.optimizer_group_signature != expected_group_signature) {
    throw_serialization("topology.optimizer_group_signature",
                        expected_group_signature,
                        descriptor.optimizer_group_signature,
                        "Reject corrupted optimizer group topology.");
  }
  const auto has_callable = cursor.read_u8("topology.callable.presence");
  if (has_callable > 1) {
    throw_serialization("topology.callable.presence", "zero or one",
                        std::to_string(has_callable),
                        "Reject invalid callable presence metadata.");
  }
  if (has_callable == 1) {
    descriptor.callable_id =
        cursor.read_string(limits.max_identifier_bytes, "topology.callable_id");
  }
  const auto child_count = cursor.read_u32("topology.children.count");
  if (child_count > limits.max_config_entries) {
    throw_resource("topology.children.count", limits.max_config_entries,
                   child_count, "Reduce composite scheduler fanout.");
  }
  descriptor.children.reserve(child_count);
  for (std::uint32_t index = 0; index < child_count; ++index) {
    auto child = read_descriptor(cursor, limits, depth + 1);
    child.fingerprint = cursor.read_string(limits.max_identifier_bytes,
                                           "topology.child.fingerprint");
    if (child.fingerprint != descriptor_fingerprint(child)) {
      throw_serialization("topology.child.fingerprint",
                          "the canonical child descriptor fingerprint",
                          child.fingerprint,
                          "Reject corrupted composite topology.");
    }
    descriptor.children.push_back(std::move(child));
  }
  return descriptor;
}

std::vector<std::byte> encode_topology(std::string_view scheduler_id,
                                       const SchedulerDescriptor& descriptor) {
  std::vector<std::byte> output(kTopologyMagic.begin(), kTopologyMagic.end());
  append_u32(output, kTopologyVersion);
  append_string(output, scheduler_id);
  append_descriptor(output, descriptor);
  append_string(output, descriptor.fingerprint);
  return output;
}

void validate_topology_blob(std::span<const std::byte> bytes,
                            const LrSchedulerCheckpointOptions& limits) {
  Cursor cursor(bytes);
  cursor.expect_magic(kTopologyMagic, "topology.magic");
  const auto version = cursor.read_u32("topology.version");
  if (version != kTopologyVersion) {
    throw_serialization("topology.version", std::to_string(kTopologyVersion),
                        std::to_string(version),
                        "Migrate scheduler topology explicitly.");
  }
  static_cast<void>(
      cursor.read_string(limits.max_identifier_bytes, "topology.scheduler_id"));
  auto descriptor = read_descriptor(cursor, limits, 1);
  descriptor.fingerprint =
      cursor.read_string(limits.max_identifier_bytes, "topology.fingerprint");
  if (descriptor.fingerprint != descriptor_fingerprint(descriptor)) {
    throw_serialization("topology.fingerprint",
                        "the canonical scheduler descriptor fingerprint",
                        descriptor.fingerprint,
                        "Reject corrupted scheduler topology.");
  }
  if (!cursor.empty()) {
    throw_serialization("topology.trailing_bytes", "no trailing metadata",
                        "present", "Reject ambiguous topology extensions.");
  }
}

void append_scheduler_state(std::vector<std::byte>& output,
                            const detail::SchedulerState& state) {
  if (state.control.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw_invalid("control.values", "at most uint32 mutable values",
                  std::to_string(state.control.size()),
                  "Use a bounded scheduler control state.");
  }
  if (state.children.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw_invalid("control.children", "at most uint32 child states",
                  std::to_string(state.children.size()),
                  "Use a bounded composite scheduler tree.");
  }
  append_u64(output, state.completed_steps);
  append_u32(output, static_cast<std::uint32_t>(state.control.size()));
  for (const auto& [key, value] : state.control) {
    append_string(output, key);
    append_config_value(output, value);
  }
  append_u32(output, static_cast<std::uint32_t>(state.children.size()));
  for (const auto& child : state.children) {
    append_scheduler_state(output, child);
  }
}

detail::SchedulerState read_scheduler_state(
    Cursor& cursor, const SchedulerDescriptor& descriptor,
    const LrSchedulerCheckpointOptions& limits, std::size_t depth) {
  if (depth > limits.max_descriptor_depth) {
    throw_resource("control.state_depth", limits.max_descriptor_depth, depth,
                   "Reduce composite scheduler nesting.");
  }
  detail::SchedulerState state{
      .completed_steps = cursor.read_u64("control.completed_steps"),
      .control = {},
      .children = {},
  };
  const auto control_count = cursor.read_u32("control.values.count");
  if (control_count > limits.max_config_entries) {
    throw_resource("control.values.count", limits.max_config_entries,
                   control_count, "Reduce scheduler mutable control values.");
  }
  for (std::uint32_t index = 0; index < control_count; ++index) {
    auto key =
        cursor.read_string(limits.max_identifier_bytes, "control.value.key");
    auto value = read_config_value(cursor, limits);
    if (!state.control.emplace(std::move(key), std::move(value)).second) {
      throw_serialization("control.value.key", "unique mutable state keys",
                          "duplicate",
                          "Reject ambiguous scheduler control state.");
    }
  }
  const auto child_count = cursor.read_u32("control.children.count");
  if (child_count != descriptor.children.size()) {
    throw_compatibility(
        "control.children.count", std::to_string(descriptor.children.size()),
        std::to_string(child_count), "Restore the exact scheduler tree shape.");
  }
  if (child_count > limits.max_config_entries) {
    throw_resource("control.children.count", limits.max_config_entries,
                   child_count, "Reduce composite scheduler fanout.");
  }
  state.children.reserve(child_count);
  for (std::uint32_t index = 0; index < child_count; ++index) {
    state.children.push_back(read_scheduler_state(
        cursor, descriptor.children[index], limits, depth + 1));
  }
  return state;
}

std::vector<std::byte> encode_control(const detail::SchedulerState& state,
                                      std::span<const double> learning_rates) {
  std::vector<std::byte> output(kControlMagic.begin(), kControlMagic.end());
  append_u32(output, kControlVersion);
  append_scheduler_state(output, state);
  append_u32(output, static_cast<std::uint32_t>(learning_rates.size()));
  for (const auto rate : learning_rates) {
    append_double(output, rate);
  }
  return output;
}

DecodedControl decode_control(std::span<const std::byte> bytes,
                              const SchedulerDescriptor& descriptor,
                              const LrSchedulerCheckpointOptions& limits) {
  Cursor cursor(bytes);
  cursor.expect_magic(kControlMagic, "control.magic");
  const auto version = cursor.read_u32("control.version");
  if (version != kControlVersion) {
    throw_serialization("control.version", std::to_string(kControlVersion),
                        std::to_string(version),
                        "Migrate scheduler control state explicitly.");
  }
  DecodedControl decoded{
      .state = read_scheduler_state(cursor, descriptor, limits, 1),
      .learning_rates = {},
  };
  const auto group_count = cursor.read_u32("control.group_count");
  if (group_count != descriptor.optimizer_group_ids.size()) {
    throw_compatibility("control.group_count",
                        std::to_string(descriptor.optimizer_group_ids.size()),
                        std::to_string(group_count),
                        "Restore into the same optimizer group topology.");
  }
  if (group_count > limits.max_groups) {
    throw_resource("control.group_count", limits.max_groups, group_count,
                   "Reduce optimizer groups or raise max_groups.");
  }
  decoded.learning_rates.reserve(group_count);
  for (std::uint32_t index = 0; index < group_count; ++index) {
    const auto rate =
        std::bit_cast<double>(cursor.read_u64("control.learning_rate"));
    if (!std::isfinite(rate) || rate < 0.0) {
      throw_serialization("control.learning_rate", "a finite non-negative rate",
                          std::to_string(rate),
                          "Reject invalid scheduler control state.");
    }
    decoded.learning_rates.push_back(rate);
  }
  if (!cursor.empty()) {
    throw_serialization("control.trailing_bytes", "no trailing metadata",
                        "present", "Reject ambiguous control extensions.");
  }
  return decoded;
}

bool field_equal(const StateFieldDescriptor& left,
                 const StateFieldDescriptor& right) {
  return left.name == right.name && left.media_type == right.media_type &&
         left.max_bytes == right.max_bytes && left.required == right.required;
}

bool manifest_equal(const StateManifest& left, const StateManifest& right) {
  return left.participant_id == right.participant_id &&
         left.schema_version == right.schema_version &&
         left.restore_level == right.restore_level &&
         left.degradation_reasons == right.degradation_reasons &&
         left.fields.size() == right.fields.size() &&
         std::equal(left.fields.begin(), left.fields.end(),
                    right.fields.begin(), field_equal);
}

void validate_checkpoint_options(const LrSchedulerCheckpointOptions& options,
                                 std::string_view participant_id) {
  if (options.max_groups == 0 || options.max_identifier_bytes == 0 ||
      options.max_config_entries == 0 ||
      options.max_config_array_elements == 0 ||
      options.max_descriptor_depth == 0 || options.max_topology_bytes == 0 ||
      options.max_control_bytes == 0 || options.max_total_bytes == 0) {
    throw_invalid("checkpoint.options", "positive resource limits", "zero",
                  "Set every scheduler checkpoint limit above zero.");
  }
  if (options.max_groups > std::numeric_limits<std::uint32_t>::max() ||
      options.max_identifier_bytes >
          std::numeric_limits<std::uint32_t>::max() ||
      options.max_config_entries > std::numeric_limits<std::uint32_t>::max() ||
      options.max_config_array_elements >
          std::numeric_limits<std::uint32_t>::max()) {
    throw_invalid("checkpoint.options", "limits within uint32 range",
                  "excessive", "Use bounded scheduler checkpoint limits.");
  }
  validate_identifier(participant_id, options.max_identifier_bytes,
                      "checkpoint.participant_id");
  const StateManifest probe{
      .participant_id = std::string(participant_id),
      .schema_version = kParticipantSchemaVersion,
      .restore_level = RestoreLevel::exact,
      .fields = {{.name = "topology", .max_bytes = 1}},
      .degradation_reasons = {},
  };
  validate_state_manifest(probe);
}

void validate_config_value_limits(const SchedulerConfigValue& value,
                                  const LrSchedulerCheckpointOptions& limits) {
  std::visit(
      [&limits](const auto& typed) {
        using Value = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<Value, double>) {
          if (!std::isfinite(typed)) {
            throw_invalid("descriptor.config.double", "a finite value",
                          "non-finite",
                          "Use deterministic finite scheduler options.");
          }
        } else if constexpr (std::is_same_v<Value, std::string>) {
          if (typed.size() > limits.max_identifier_bytes ||
              typed.find('\0') != std::string::npos) {
            throw_invalid("descriptor.config.string",
                          "a bounded non-NUL string",
                          std::to_string(typed.size()),
                          "Reduce typed scheduler configuration metadata.");
          }
        } else if constexpr (!std::is_same_v<Value, bool> &&
                             !std::is_same_v<Value, std::int64_t>) {
          if (typed.size() > limits.max_config_array_elements) {
            throw_resource("descriptor.config.array",
                           limits.max_config_array_elements, typed.size(),
                           "Reduce the typed scheduler configuration array.");
          }
          for (const auto& element : typed) {
            using Element = typename Value::value_type;
            if constexpr (std::is_same_v<Element, double>) {
              if (!std::isfinite(element)) {
                throw_invalid("descriptor.config.double_array", "finite values",
                              "non-finite",
                              "Use deterministic finite scheduler options.");
              }
            } else if constexpr (std::is_same_v<Element, std::string>) {
              if (element.size() > limits.max_identifier_bytes ||
                  element.find('\0') != std::string::npos) {
                throw_invalid("descriptor.config.string_array",
                              "bounded non-NUL strings",
                              std::to_string(element.size()),
                              "Reduce typed scheduler metadata.");
              }
            }
          }
        }
      },
      value);
}

void validate_descriptor_limits(const SchedulerDescriptor& descriptor,
                                const LrSchedulerCheckpointOptions& limits,
                                std::size_t depth) {
  if (depth > limits.max_descriptor_depth) {
    throw_resource("descriptor.depth", limits.max_descriptor_depth, depth,
                   "Reduce composite scheduler nesting.");
  }
  if (descriptor.schema_version == 0 ||
      scheduler_step_mode_name(descriptor.step_mode) == "unknown") {
    throw_invalid("descriptor.schema", "a versioned declared step mode",
                  "invalid", "Construct a supported scheduler descriptor.");
  }
  validate_identifier(descriptor.type_id, limits.max_identifier_bytes,
                      "descriptor.type_id");
  validate_identifier(descriptor.optimizer_id, limits.max_identifier_bytes,
                      "descriptor.optimizer_id");
  if (descriptor.config.size() > limits.max_config_entries) {
    throw_resource("descriptor.config", limits.max_config_entries,
                   descriptor.config.size(),
                   "Reduce typed scheduler configuration entries.");
  }
  for (const auto& [key, value] : descriptor.config) {
    validate_identifier(key, limits.max_identifier_bytes,
                        "descriptor.config.key");
    validate_config_value_limits(value, limits);
  }
  if (descriptor.optimizer_group_ids.empty() ||
      descriptor.optimizer_group_ids.size() > limits.max_groups) {
    throw_resource("descriptor.optimizer_groups", limits.max_groups,
                   descriptor.optimizer_group_ids.size(),
                   "Use a bounded nonempty optimizer topology.");
  }
  std::set<std::string> group_ids;
  for (const auto& group_id : descriptor.optimizer_group_ids) {
    validate_identifier(group_id, limits.max_identifier_bytes,
                        "descriptor.optimizer_group_id");
    if (!group_ids.insert(group_id).second) {
      throw_invalid("descriptor.optimizer_group_id", "unique group IDs",
                    group_id, "Reject ambiguous optimizer topology.");
    }
  }
  validate_identifier(descriptor.optimizer_group_signature,
                      limits.max_identifier_bytes,
                      "descriptor.optimizer_group_signature");
  if (descriptor.optimizer_group_signature !=
      group_signature(descriptor.optimizer_id,
                      descriptor.optimizer_group_ids)) {
    throw_invalid("descriptor.optimizer_group_signature",
                  "the canonical optimizer group signature", "mismatch",
                  "Rebuild the scheduler from stable optimizer groups.");
  }
  if (descriptor.callable_id.has_value()) {
    validate_identifier(*descriptor.callable_id, limits.max_identifier_bytes,
                        "descriptor.callable_id");
  }
  if (descriptor.children.size() > limits.max_config_entries) {
    throw_resource("descriptor.children", limits.max_config_entries,
                   descriptor.children.size(),
                   "Reduce composite scheduler fanout.");
  }
  for (const auto& child : descriptor.children) {
    validate_descriptor_limits(child, limits, depth + 1);
  }
  validate_identifier(descriptor.fingerprint, limits.max_identifier_bytes,
                      "descriptor.fingerprint");
  if (descriptor.fingerprint != descriptor_fingerprint(descriptor)) {
    throw_invalid("descriptor.fingerprint",
                  "the canonical descriptor fingerprint", "mismatch",
                  "Rebuild the scheduler descriptor.");
  }
}

void checked_total(std::uint64_t left, std::uint64_t right,
                   std::uint64_t maximum) {
  if (right > maximum || left > maximum - right) {
    throw_resource("state.total_bytes", maximum,
                   right > maximum ? right : left + right,
                   "Reduce scheduler state or raise max_total_bytes.");
  }
}

SchedulerSnapshot inspect_scheduler(
    const LrScheduler& scheduler, const LrSchedulerCheckpointOptions& limits) {
  const auto& descriptor = scheduler.descriptor();
  validate_descriptor_limits(descriptor, limits, 1);
  const auto rates = scheduler.learning_rates();
  const auto state = detail::SchedulerStateAccess::capture_state(scheduler);
  detail::SchedulerStateAccess::validate_state(scheduler, state);
  auto topology = encode_topology(scheduler.scheduler_id(), descriptor);
  auto control = encode_control(state, rates);
  if (topology.size() > limits.max_topology_bytes) {
    throw_resource("state.topology_bytes", limits.max_topology_bytes,
                   topology.size(), "Raise the topology limit explicitly.");
  }
  if (control.size() > limits.max_control_bytes) {
    throw_resource("state.control_bytes", limits.max_control_bytes,
                   control.size(), "Raise the control limit explicitly.");
  }
  checked_total(topology.size(), control.size(), limits.max_total_bytes);
  StateManifest manifest{
      .participant_id = scheduler.scheduler_id(),
      .schema_version = kParticipantSchemaVersion,
      .restore_level = RestoreLevel::exact,
      .fields = {{.name = "topology",
                  .media_type = "application/vnd.jyppx.scheduler-descriptor-v1",
                  .max_bytes = limits.max_topology_bytes},
                 {.name = "control",
                  .media_type = "application/vnd.jyppx.scheduler-state-tree-v2",
                  .max_bytes = limits.max_control_bytes}},
      .degradation_reasons = {},
  };
  validate_state_manifest(manifest);
  return {.topology = std::move(topology),
          .control = std::move(control),
          .manifest = std::move(manifest)};
}

}  // namespace

struct LrScheduler::Impl {
  std::shared_ptr<OptimizerRateControl>
      optimizer;  ///< Retained atomic optimizer rate-control surface.
  std::string scheduler_id;  ///< Stable checkpoint participant identity.
  std::vector<double> base_learning_rates;  ///< Construction-time group rates.
  SchedulerDescriptor descriptor;  ///< Immutable canonical scheduler topology.
};

std::string_view scheduler_step_mode_name(SchedulerStepMode mode) noexcept {
  switch (mode) {
    case SchedulerStepMode::batch:
      return "batch";
    case SchedulerStepMode::epoch:
      return "epoch";
    case SchedulerStepMode::metric:
      return "metric";
    case SchedulerStepMode::manual:
      return "manual";
  }
  return "unknown";
}

LrScheduler::LrScheduler(std::shared_ptr<OptimizerRateControl> optimizer,
                         std::string scheduler_id)
    : impl_(std::make_unique<Impl>()) {
  try {
    if (!optimizer) {
      throw_invalid("optimizer", "a non-null OptimizerRateControl", "null",
                    "Create the optimizer before its scheduler.");
    }
    validate_identifier(scheduler_id, kConstructionMaximumIdentifierBytes,
                        "scheduler_id");
    validate_identifier(optimizer->optimizer_id(),
                        kConstructionMaximumIdentifierBytes, "optimizer_id");
    const auto& group_ids = optimizer->parameter_group_ids();
    if (group_ids.empty() ||
        group_ids.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw_invalid("optimizer.group_ids", "a bounded nonempty topology",
                    std::to_string(group_ids.size()),
                    "Define stable optimizer parameter groups.");
    }
    std::set<std::string> unique_group_ids;
    for (const auto& group_id : group_ids) {
      validate_identifier(group_id, kConstructionMaximumIdentifierBytes,
                          "optimizer.group_id");
      if (!unique_group_ids.insert(group_id).second) {
        throw_invalid("optimizer.group_id", "unique stable group IDs", group_id,
                      "Rename the duplicate optimizer group.");
      }
    }
    auto base_rates = optimizer->learning_rates();
    validate_rates(base_rates, group_ids.size(), "optimizer.learning_rates");
    impl_->optimizer = std::move(optimizer);
    impl_->scheduler_id = std::move(scheduler_id);
    impl_->base_learning_rates = std::move(base_rates);
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    translate_bad_alloc("construct");
  }
}

LrScheduler::~LrScheduler() = default;

void LrScheduler::configure_descriptor(
    std::string type_id, SchedulerStepMode step_mode,
    std::map<std::string, SchedulerConfigValue> config,
    std::vector<SchedulerDescriptor> children,
    std::optional<std::string> callable_id) {
  try {
    if (!impl_->descriptor.type_id.empty()) {
      throw_invalid("descriptor", "configured exactly once", "already set",
                    "Configure a descriptor only during construction.");
    }
    validate_identifier(type_id, kConstructionMaximumIdentifierBytes,
                        "descriptor.type_id");
    if (scheduler_step_mode_name(step_mode) == "unknown") {
      throw_invalid("descriptor.step_mode", "a declared mode", "unknown",
                    "Use a stable SchedulerStepMode value.");
    }
    for (const auto& child : children) {
      if (child.optimizer_id != impl_->optimizer->optimizer_id() ||
          child.optimizer_group_ids !=
              impl_->optimizer->parameter_group_ids()) {
        throw_invalid("descriptor.children.optimizer",
                      "the parent optimizer identity and ordered groups",
                      child.optimizer_id,
                      "Construct every child from the same optimizer.");
      }
    }
    SchedulerDescriptor descriptor{
        .schema_version = kDescriptorSchemaVersion,
        .type_id = std::move(type_id),
        .step_mode = step_mode,
        .config = std::move(config),
        .optimizer_id = impl_->optimizer->optimizer_id(),
        .optimizer_group_ids = impl_->optimizer->parameter_group_ids(),
        .optimizer_group_signature =
            group_signature(impl_->optimizer->optimizer_id(),
                            impl_->optimizer->parameter_group_ids()),
        .callable_id = std::move(callable_id),
        .children = std::move(children),
        .fingerprint = {},
    };
    descriptor.fingerprint = descriptor_fingerprint(descriptor);
    validate_descriptor_limits(descriptor, LrSchedulerCheckpointOptions{}, 1);
    impl_->descriptor = std::move(descriptor);
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    translate_bad_alloc("configure descriptor");
  }
}

void LrScheduler::initialize_learning_rates(
    std::span<const double> learning_rates) {
  try {
    validate_rates(learning_rates, impl_->base_learning_rates.size(),
                   "initialize.learning_rates");
    impl_->optimizer->set_learning_rates(
        std::vector<double>(learning_rates.begin(), learning_rates.end()));
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    translate_bad_alloc("initialize learning rates");
  }
}

std::vector<double> LrScheduler::learning_rates() const {
  try {
    if (impl_->optimizer->optimizer_id() != impl_->descriptor.optimizer_id) {
      throw_compatibility("optimizer_id", impl_->descriptor.optimizer_id,
                          impl_->optimizer->optimizer_id(),
                          "Keep scheduler and optimizer identities immutable.");
    }
    if (impl_->optimizer->parameter_group_ids() !=
        impl_->descriptor.optimizer_group_ids) {
      throw_compatibility("optimizer_group_ids",
                          "the captured ordered group IDs", "topology changed",
                          "Do not add, remove, reorder, or rename groups.");
    }
    auto rates = impl_->optimizer->learning_rates();
    validate_rates(rates, impl_->descriptor.optimizer_group_ids.size(),
                   "optimizer.learning_rates");
    return rates;
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    translate_bad_alloc("learning_rates");
  }
}

void LrScheduler::step() {
  try {
    if (impl_->descriptor.type_id.empty()) {
      throw_invalid("descriptor", "a configured scheduler descriptor",
                    "unconfigured",
                    "Complete derived scheduler construction before use.");
    }
    if (impl_->descriptor.step_mode == SchedulerStepMode::metric) {
      throw_invalid("step.event", "a supplied finite metric",
                    "metric-less event", "Call step(metric) instead.");
    }
    const auto rates = learning_rates();
    auto transition = detail::SchedulerStateAccess::preview_step(*this, rates);
    validate_rates(transition.learning_rates,
                   impl_->descriptor.optimizer_group_ids.size(),
                   "step.learning_rates");
    detail::SchedulerStateAccess::validate_state(*this, transition.state);
    impl_->optimizer->set_learning_rates(transition.learning_rates);
    detail::SchedulerStateAccess::apply_state(*this, transition.state);
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    translate_bad_alloc("step");
  }
}

void LrScheduler::step(double metric) {
  try {
    if (impl_->descriptor.type_id.empty()) {
      throw_invalid("descriptor", "a configured scheduler descriptor",
                    "unconfigured",
                    "Complete derived scheduler construction before use.");
    }
    if (impl_->descriptor.step_mode != SchedulerStepMode::metric) {
      throw_invalid("step.event", "the scheduler's declared event mode",
                    "metric event", "Call the metric-less step() overload.");
    }
    if (!std::isfinite(metric)) {
      throw_invalid("step.metric", "a finite monitored value",
                    std::to_string(metric),
                    "Filter invalid validation metrics before scheduling.");
    }
    const auto rates = learning_rates();
    auto transition =
        detail::SchedulerStateAccess::preview_metric_step(*this, rates, metric);
    validate_rates(transition.learning_rates,
                   impl_->descriptor.optimizer_group_ids.size(),
                   "step.learning_rates");
    detail::SchedulerStateAccess::validate_state(*this, transition.state);
    impl_->optimizer->set_learning_rates(transition.learning_rates);
    detail::SchedulerStateAccess::apply_state(*this, transition.state);
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    translate_bad_alloc("metric step");
  }
}

detail::SchedulerTransition LrScheduler::preview_metric_step_impl(
    std::span<const double> learning_rates, double metric) const {
  static_cast<void>(learning_rates);
  static_cast<void>(metric);
  throw_invalid("step.event", "a scheduler implementing metric events",
                "unsupported metric event",
                "Use metric input only with a metric-driven scheduler.");
}

const std::string& LrScheduler::scheduler_id() const noexcept {
  return impl_->scheduler_id;
}

std::uint64_t LrScheduler::completed_steps() const noexcept {
  return completed_steps_impl();
}

const std::vector<double>& LrScheduler::base_learning_rates() const noexcept {
  return impl_->base_learning_rates;
}

const SchedulerDescriptor& LrScheduler::descriptor() const noexcept {
  return impl_->descriptor;
}

namespace detail {

SchedulerTransition SchedulerStateAccess::preview_step(
    const LrScheduler& scheduler, std::span<const double> learning_rates) {
  validate_rates(learning_rates,
                 scheduler.descriptor().optimizer_group_ids.size(),
                 "preview.input_learning_rates");
  auto transition = scheduler.preview_step_impl(learning_rates);
  validate_rates(transition.learning_rates,
                 scheduler.descriptor().optimizer_group_ids.size(),
                 "preview.output_learning_rates");
  scheduler.validate_state_impl(transition.state);
  return transition;
}

SchedulerTransition SchedulerStateAccess::preview_metric_step(
    const LrScheduler& scheduler, std::span<const double> learning_rates,
    double metric) {
  validate_rates(learning_rates,
                 scheduler.descriptor().optimizer_group_ids.size(),
                 "preview.input_learning_rates");
  if (!std::isfinite(metric)) {
    throw_invalid("preview.metric", "a finite monitored value",
                  std::to_string(metric),
                  "Filter invalid validation metrics before scheduling.");
  }
  auto transition = scheduler.preview_metric_step_impl(learning_rates, metric);
  validate_rates(transition.learning_rates,
                 scheduler.descriptor().optimizer_group_ids.size(),
                 "preview.output_learning_rates");
  scheduler.validate_state_impl(transition.state);
  return transition;
}

SchedulerState SchedulerStateAccess::capture_state(
    const LrScheduler& scheduler) {
  return scheduler.capture_state_impl();
}

void SchedulerStateAccess::validate_state(const LrScheduler& scheduler,
                                          const SchedulerState& state) {
  scheduler.validate_state_impl(state);
}

void SchedulerStateAccess::apply_state(LrScheduler& scheduler,
                                       const SchedulerState& state) noexcept {
  scheduler.apply_state_impl(state);
}

const std::shared_ptr<OptimizerRateControl>& SchedulerStateAccess::optimizer(
    const LrScheduler& scheduler) noexcept {
  return scheduler.impl_->optimizer;
}

}  // namespace detail

StepLrScheduler::StepLrScheduler(
    std::shared_ptr<OptimizerRateControl> optimizer,
    StepLrSchedulerOptions options)
    : LrScheduler(optimizer, options.scheduler_id),
      options_(std::move(options)) {
  if (options_.step_size == 0 ||
      options_.step_size > static_cast<std::uint64_t>(
                               std::numeric_limits<std::int64_t>::max())) {
    throw_invalid("step_size", "a positive int64-compatible value",
                  std::to_string(options_.step_size),
                  "Choose a bounded positive decay interval.");
  }
  if (!std::isfinite(options_.gamma) || options_.gamma < 0.0) {
    throw_invalid("gamma", "a finite non-negative multiplier",
                  std::to_string(options_.gamma),
                  "Choose a valid decay multiplier.");
  }
  configure_descriptor(
      kTypeId, SchedulerStepMode::epoch,
      {{"base_learning_rates", base_learning_rates()},
       {"gamma", options_.gamma},
       {"step_size", static_cast<std::int64_t>(options_.step_size)}});
}

StepLrScheduler::~StepLrScheduler() = default;

detail::SchedulerTransition StepLrScheduler::preview_step_impl(
    std::span<const double> learning_rates) const {
  if (completed_steps_ == std::numeric_limits<std::uint64_t>::max()) {
    throw ResourceExhausted(
        "StepLR scheduler counter exhausted",
        ErrorDetails{.component = kComponent,
                     .operation = "step.completed_steps",
                     .expected = "a count below uint64 maximum",
                     .actual = "uint64 maximum",
                     .hint = "Start a new training run identity."});
  }
  std::vector<double> candidate(learning_rates.begin(), learning_rates.end());
  const auto next_step = completed_steps_ + 1;
  if (next_step % options_.step_size == 0) {
    for (auto& rate : candidate) {
      rate *= options_.gamma;
      if (!std::isfinite(rate) || rate < 0.0) {
        throw NumericalError(
            "StepLR decay produced an invalid learning rate",
            ErrorDetails{.component = kComponent,
                         .operation = "step.learning_rate",
                         .expected = "a finite non-negative rate",
                         .actual = std::to_string(rate),
                         .hint = "Reduce gamma or the current rate."});
      }
    }
  }
  return {
      .state = {.completed_steps = next_step, .control = {}, .children = {}},
      .learning_rates = std::move(candidate)};
}

detail::SchedulerState StepLrScheduler::capture_state_impl() const {
  return {.completed_steps = completed_steps_, .control = {}, .children = {}};
}

std::uint64_t StepLrScheduler::completed_steps_impl() const noexcept {
  return completed_steps_;
}

void StepLrScheduler::validate_state_impl(
    const detail::SchedulerState& state) const {
  if (!state.control.empty() || !state.children.empty()) {
    throw_compatibility("state.shape", "a StepLR counter-only leaf",
                        "control values or children present",
                        "Restore state from the same scheduler type.");
  }
}

void StepLrScheduler::apply_state_impl(
    const detail::SchedulerState& state) noexcept {
  completed_steps_ = state.completed_steps;
}

LrSchedulerCheckpointParticipant::LrSchedulerCheckpointParticipant(
    std::shared_ptr<LrScheduler> scheduler,
    LrSchedulerCheckpointOptions options)
    : scheduler_(std::move(scheduler)), options_(options) {
  if (!scheduler_) {
    throw_invalid("scheduler", "a non-null shared LrScheduler", "null",
                  "Create the scheduler before its checkpoint participant.");
  }
  if constexpr (std::endian::native != std::endian::little) {
    throw NotImplemented(
        "Scheduler checkpoint byte order is unsupported",
        ErrorDetails{.component = kComponent,
                     .operation = "platform.endian",
                     .expected = "little endian",
                     .actual = "non-little endian",
                     .hint = "Add an explicit scheduler byte-swap codec."});
  }
  validate_checkpoint_options(options_, scheduler_->scheduler_id());
  static_cast<void>(describe_state());
}

std::string LrSchedulerCheckpointParticipant::participant_id() const {
  return scheduler_->scheduler_id();
}

StateManifest LrSchedulerCheckpointParticipant::describe_state() const {
  try {
    return inspect_scheduler(*scheduler_, options_).manifest;
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    translate_bad_alloc("describe");
  }
}

void LrSchedulerCheckpointParticipant::save_state(StateWriter& writer) const {
  try {
    const auto snapshot = inspect_scheduler(*scheduler_, options_);
    writer.write_blob("topology", snapshot.topology);
    writer.write_blob("control", snapshot.control);
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    translate_bad_alloc("save");
  }
}

void LrSchedulerCheckpointParticipant::load_state(const StateReader& reader,
                                                  RestorePolicy policy) {
  try {
    if (!restore_policy_accepts(policy, RestoreLevel::exact)) {
      throw_compatibility("restore.policy", "a policy accepting exact state",
                          std::string(restore_policy_name(policy)),
                          "Use a declared RestorePolicy value.");
    }
    const auto current = inspect_scheduler(*scheduler_, options_);
    const auto& stored_manifest = reader.manifest();
    validate_state_manifest(stored_manifest);
    if (!manifest_equal(stored_manifest, current.manifest)) {
      throw_compatibility("restore.manifest", scheduler_->scheduler_id(),
                          stored_manifest.participant_id,
                          "Use state produced by the same scheduler schema.");
    }
    for (const auto& field : current.manifest.fields) {
      if (field.required && !reader.contains(field.name)) {
        throw_serialization("restore.field", field.name, "missing",
                            "Provide every required scheduler state field.");
      }
    }
    const auto topology =
        reader.read_blob("topology", current.manifest.fields.front().max_bytes);
    validate_topology_blob(topology, options_);
    if (topology.size() != current.topology.size() ||
        !std::equal(topology.begin(), topology.end(),
                    current.topology.begin())) {
      throw_compatibility(
          "restore.topology",
          "identical scheduler type, typed options, optimizer ID, and ordered "
          "group IDs",
          "topology differs",
          "Apply an explicit scheduler state migration before restore.");
    }
    const auto control =
        reader.read_blob("control", current.manifest.fields[1].max_bytes);
    checked_total(topology.size(), control.size(), options_.max_total_bytes);
    const auto decoded =
        decode_control(control, scheduler_->descriptor(), options_);
    detail::SchedulerStateAccess::validate_state(*scheduler_, decoded.state);

    // OptimizerRateControl requires this operation to be atomic. Commit the
    // state tree only after every externally owned rate has been replaced.
    detail::SchedulerStateAccess::optimizer(*scheduler_)
        ->set_learning_rates(decoded.learning_rates);
    detail::SchedulerStateAccess::apply_state(*scheduler_, decoded.state);
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    translate_bad_alloc("restore");
  }
}

LrScheduler& LrSchedulerCheckpointParticipant::scheduler() noexcept {
  return *scheduler_;
}

const LrScheduler& LrSchedulerCheckpointParticipant::scheduler()
    const noexcept {
  return *scheduler_;
}

const LrSchedulerCheckpointOptions& LrSchedulerCheckpointParticipant::options()
    const noexcept {
  return options_;
}

}  // namespace jyppx::optim
