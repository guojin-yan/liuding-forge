// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/training/amp/grad_scaler_checkpoint_participant.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <new>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace jyppx::training::amp {
namespace {

constexpr std::uint32_t kSchemaVersion = 1;
constexpr std::uint32_t kTopologyMagic = 0x5343474AU;
constexpr std::uint32_t kControlMagic = 0x4343474AU;
constexpr std::uint64_t kControlBytes = 24;
constexpr std::string_view kComponent = "training.grad_scaler.checkpoint";

[[noreturn]] void throw_invalid(std::string operation, std::string expected,
                                std::string actual, std::string hint) {
  throw InvalidArgument("Invalid GradScaler checkpoint configuration",
                        ErrorDetails{.component = std::string(kComponent),
                                     .operation = std::move(operation),
                                     .expected = std::move(expected),
                                     .actual = std::move(actual),
                                     .hint = std::move(hint)});
}

[[noreturn]] void throw_resource(std::string operation, std::uint64_t limit,
                                 std::uint64_t actual, std::string hint) {
  throw ResourceExhausted(
      "GradScaler checkpoint resource limit exceeded",
      ErrorDetails{.component = std::string(kComponent),
                   .operation = std::move(operation),
                   .expected = "at most " + std::to_string(limit) + " bytes",
                   .actual = std::to_string(actual) + " bytes",
                   .hint = std::move(hint)});
}

[[noreturn]] void throw_serialization(std::string operation,
                                      std::string expected, std::string actual,
                                      std::string hint) {
  throw SerializationError("Malformed GradScaler checkpoint state",
                           ErrorDetails{.component = std::string(kComponent),
                                        .operation = std::move(operation),
                                        .expected = std::move(expected),
                                        .actual = std::move(actual),
                                        .hint = std::move(hint)});
}

[[noreturn]] void throw_compatibility(std::string operation,
                                      std::string expected, std::string actual,
                                      std::string hint) {
  throw CompatibilityError("Incompatible GradScaler checkpoint state",
                           ErrorDetails{.component = std::string(kComponent),
                                        .operation = std::move(operation),
                                        .expected = std::move(expected),
                                        .actual = std::move(actual),
                                        .hint = std::move(hint)});
}

void append_u8(std::vector<std::byte>& bytes, std::uint8_t value) {
  bytes.push_back(static_cast<std::byte>(value));
}

void append_u32(std::vector<std::byte>& bytes, std::uint32_t value) {
  for (unsigned int shift = 0; shift < 32; shift += 8) {
    append_u8(bytes, static_cast<std::uint8_t>(value >> shift));
  }
}

void append_u64(std::vector<std::byte>& bytes, std::uint64_t value) {
  for (unsigned int shift = 0; shift < 64; shift += 8) {
    append_u8(bytes, static_cast<std::uint8_t>(value >> shift));
  }
}

void append_double(std::vector<std::byte>& bytes, double value) {
  append_u64(bytes, std::bit_cast<std::uint64_t>(value));
}

class Cursor {
 public:
  explicit Cursor(std::span<const std::byte> bytes) : bytes_(bytes) {}

  std::uint32_t read_u32(std::string_view field) {
    return static_cast<std::uint32_t>(read_integer(4, field));
  }

  std::uint64_t read_u64(std::string_view field) {
    return read_integer(8, field);
  }

  double read_double(std::string_view field) {
    return std::bit_cast<double>(read_u64(field));
  }

  bool empty() const noexcept { return offset_ == bytes_.size(); }

 private:
  std::uint64_t read_integer(std::size_t width, std::string_view field) {
    if (width > bytes_.size() - offset_) {
      throw_serialization(std::string(field), std::to_string(width) + " bytes",
                          std::to_string(bytes_.size() - offset_) + " bytes",
                          "Reject truncated scaler control state.");
    }
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < width; ++index) {
      value |= static_cast<std::uint64_t>(
                   std::to_integer<std::uint8_t>(bytes_[offset_ + index]))
               << (index * 8U);
    }
    offset_ += width;
    return value;
  }

  std::span<const std::byte> bytes_;  ///< Borrowed immutable encoded bytes.
  std::size_t offset_ = 0;  ///< Number of bytes consumed from `bytes_`.
};

void validate_checkpoint_options(const GradScalerCheckpointOptions& options,
                                 const GradScalerOptions& scaler_options) {
  if (options.max_identifier_bytes == 0 || options.max_topology_bytes == 0 ||
      options.max_control_bytes < kControlBytes ||
      options.max_total_bytes == 0) {
    throw_invalid("limits", "positive limits and control capacity >= 24",
                  "zero or undersized limit",
                  "Configure positive bounded participant resources.");
  }
  if (scaler_options.scaler_id.size() > options.max_identifier_bytes) {
    throw_resource("scaler_id", options.max_identifier_bytes,
                   scaler_options.scaler_id.size(),
                   "Use a shorter stable scaler identity.");
  }
}

std::vector<std::byte> encode_topology(
    const GradScalerOptions& options, const GradScalerCheckpointOptions& caps) {
  if (options.scaler_id.size() >
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw_resource("scaler_id", std::numeric_limits<std::uint32_t>::max(),
                   options.scaler_id.size(),
                   "Use an identity representable by schema version 1.");
  }
  std::vector<std::byte> bytes;
  bytes.reserve(52 + options.scaler_id.size());
  append_u32(bytes, kTopologyMagic);
  append_u32(bytes, kSchemaVersion);
  append_u32(bytes, static_cast<std::uint32_t>(options.scaler_id.size()));
  for (const char value : options.scaler_id) {
    append_u8(bytes, static_cast<std::uint8_t>(value));
  }
  append_u8(bytes, options.enabled ? 1U : 0U);
  for (int index = 0; index < 7; ++index) {
    append_u8(bytes, 0U);
  }
  append_double(bytes, options.init_scale);
  append_double(bytes, options.growth_factor);
  append_double(bytes, options.backoff_factor);
  append_u64(bytes, static_cast<std::uint64_t>(options.growth_interval));
  if (bytes.size() > caps.max_topology_bytes) {
    throw_resource("topology", caps.max_topology_bytes, bytes.size(),
                   "Raise max_topology_bytes for this validated identity.");
  }
  return bytes;
}

std::vector<std::byte> encode_control(const GradScalerState& state) {
  std::vector<std::byte> bytes;
  bytes.reserve(kControlBytes);
  append_u32(bytes, kControlMagic);
  append_u32(bytes, kSchemaVersion);
  append_double(bytes, state.scale);
  append_u64(bytes, state.growth_tracker);
  return bytes;
}

GradScalerState decode_control(std::span<const std::byte> bytes,
                               const GradScaler& scaler) {
  if (bytes.size() != kControlBytes) {
    throw_serialization("control.size", std::to_string(kControlBytes),
                        std::to_string(bytes.size()),
                        "Use a complete schema-version-1 control blob.");
  }
  Cursor cursor(bytes);
  if (cursor.read_u32("control.magic") != kControlMagic ||
      cursor.read_u32("control.schema") != kSchemaVersion) {
    throw_serialization("control.header", "schema-version-1 header",
                        "unknown header",
                        "Apply an explicit scaler state migration.");
  }
  auto state = scaler.capture_state();
  state.scale = cursor.read_double("control.scale");
  state.growth_tracker = cursor.read_u64("control.growth_tracker");
  if (!cursor.empty()) {
    throw_serialization("control.trailing_bytes", "no trailing bytes",
                        "present", "Reject ambiguous schema extensions.");
  }
  if (!std::isfinite(state.scale) || state.scale <= 0.0 ||
      (!state.enabled && state.scale != 1.0) ||
      state.growth_tracker >=
          static_cast<std::uint64_t>(state.growth_interval)) {
    throw_serialization(
        "control.values", "positive compatible scaler state",
        "invalid scale or growth tracker",
        "Restore state produced at a completed update boundary.");
  }
  return state;
}

bool manifest_equal(const StateManifest& left, const StateManifest& right) {
  return left.participant_id == right.participant_id &&
         left.schema_version == right.schema_version &&
         left.restore_level == right.restore_level &&
         std::is_permutation(left.degradation_reasons.begin(),
                             left.degradation_reasons.end(),
                             right.degradation_reasons.begin(),
                             right.degradation_reasons.end()) &&
         state_fields_equal(left.fields, right.fields);
}

StateManifest make_manifest(const GradScaler& scaler,
                            const GradScalerCheckpointOptions& options) {
  const auto topology = encode_topology(scaler.options(), options);
  if (topology.size() > options.max_total_bytes ||
      kControlBytes > options.max_total_bytes - topology.size()) {
    throw_resource("total", options.max_total_bytes,
                   topology.size() + kControlBytes,
                   "Raise max_total_bytes for the validated scaler schema.");
  }
  StateManifest manifest{
      .participant_id = scaler.options().scaler_id,
      .schema_version = kSchemaVersion,
      .restore_level = RestoreLevel::exact,
      .fields = {{.name = "topology",
                  .media_type = "application/vnd.jyppx.grad-scaler-topology-v1",
                  .max_bytes = topology.size()},
                 {.name = "control",
                  .media_type = "application/vnd.jyppx.grad-scaler-control-v1",
                  .max_bytes = kControlBytes}},
      .degradation_reasons = {}};
  validate_state_manifest(manifest);
  return manifest;
}

[[noreturn]] void translate_bad_alloc(std::string operation) {
  throw ResourceExhausted(
      "Unable to allocate GradScaler checkpoint state",
      ErrorDetails{.component = std::string(kComponent),
                   .operation = std::move(operation),
                   .expected = "memory within configured scaler limits",
                   .actual = "allocation failed",
                   .hint = "Reduce state limits or the active workload."});
}

}  // namespace

GradScalerCheckpointParticipant::GradScalerCheckpointParticipant(
    std::shared_ptr<GradScaler> scaler, GradScalerCheckpointOptions options)
    : scaler_(std::move(scaler)), options_(options) {
  if (!scaler_) {
    throw_invalid("scaler", "a non-null shared GradScaler", "null",
                  "Create the scaler before its checkpoint participant.");
  }
  if constexpr (std::endian::native != std::endian::little) {
    throw NotImplemented(
        "GradScaler checkpoint byte order is unsupported",
        ErrorDetails{.component = std::string(kComponent),
                     .operation = "platform.endian",
                     .expected = "little endian",
                     .actual = "non-little endian",
                     .hint = "Add an explicit scalar byte-swap codec."});
  }
  validate_checkpoint_options(options_, scaler_->options());
  static_cast<void>(describe_state());
}

std::string GradScalerCheckpointParticipant::participant_id() const {
  return scaler_->options().scaler_id;
}

StateManifest GradScalerCheckpointParticipant::describe_state() const {
  try {
    return make_manifest(*scaler_, options_);
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    translate_bad_alloc("describe");
  }
}

void GradScalerCheckpointParticipant::save_state(StateWriter& writer) const {
  try {
    const auto state = scaler_->capture_state();
    const auto topology = encode_topology(scaler_->options(), options_);
    const auto control = encode_control(state);
    static_cast<void>(make_manifest(*scaler_, options_));
    writer.write_blob("topology", topology);
    writer.write_blob("control", control);
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    translate_bad_alloc("save");
  }
}

void GradScalerCheckpointParticipant::load_state(const StateReader& reader,
                                                 RestorePolicy policy) {
  try {
    static_cast<void>(scaler_->capture_state());
    if (!restore_policy_accepts(policy, RestoreLevel::exact)) {
      throw_compatibility("restore.policy", "a policy accepting exact state",
                          std::string(restore_policy_name(policy)),
                          "Use a declared RestorePolicy value.");
    }
    const auto current_manifest = make_manifest(*scaler_, options_);
    const auto& stored_manifest = reader.manifest();
    validate_state_manifest(stored_manifest);
    if (!manifest_equal(stored_manifest, current_manifest)) {
      throw_compatibility("restore.manifest", participant_id(),
                          stored_manifest.participant_id,
                          "Use state produced by the same scaler schema.");
    }
    for (const auto& field : current_manifest.fields) {
      if (field.required && !reader.contains(field.name)) {
        throw_serialization("restore.field", field.name, "missing",
                            "Provide every required scaler state field.");
      }
    }

    const auto expected_topology =
        encode_topology(scaler_->options(), options_);
    const auto topology =
        reader.read_blob("topology", current_manifest.fields.front().max_bytes);
    if (topology.size() != expected_topology.size() ||
        !std::equal(topology.begin(), topology.end(),
                    expected_topology.begin())) {
      throw_compatibility(
          "restore.topology", "identical identity and immutable options",
          "topology differs",
          "Construct the scaler with the checkpoint's frozen configuration.");
    }

    const auto control =
        reader.read_blob("control", current_manifest.fields[1].max_bytes);
    const auto candidate = decode_control(control, *scaler_);
    scaler_->restore_state(candidate);
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    translate_bad_alloc("restore");
  }
}

GradScaler& GradScalerCheckpointParticipant::scaler() noexcept {
  return *scaler_;
}

const GradScaler& GradScalerCheckpointParticipant::scaler() const noexcept {
  return *scaler_;
}

const GradScalerCheckpointOptions& GradScalerCheckpointParticipant::options()
    const noexcept {
  return options_;
}

}  // namespace jyppx::training::amp
