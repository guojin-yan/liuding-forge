// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/foundation/random_checkpoint_participant.hpp>

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <new>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace jyppx::core {
namespace {

constexpr std::uint32_t kSchemaVersion = 1;
constexpr std::uint32_t kTopologyMagic = 0x4E52474AU;
constexpr std::string_view kComponent = "foundation.rng.checkpoint";

[[noreturn]] void throw_invalid(std::string operation, std::string expected,
                                std::string actual, std::string hint) {
  throw InvalidArgument("Invalid RNG checkpoint configuration",
                        ErrorDetails{.component = std::string(kComponent),
                                     .operation = std::move(operation),
                                     .expected = std::move(expected),
                                     .actual = std::move(actual),
                                     .hint = std::move(hint)});
}

[[noreturn]] void throw_resource(std::string operation, std::uint64_t limit,
                                 std::uint64_t actual, std::string hint) {
  throw ResourceExhausted(
      "RNG checkpoint resource limit exceeded",
      ErrorDetails{.component = std::string(kComponent),
                   .operation = std::move(operation),
                   .expected = "at most " + std::to_string(limit) + " bytes",
                   .actual = std::to_string(actual) + " bytes",
                   .hint = std::move(hint)});
}

[[noreturn]] void throw_serialization(std::string operation,
                                      std::string expected, std::string actual,
                                      std::string hint) {
  throw SerializationError("Malformed RNG checkpoint state",
                           ErrorDetails{.component = std::string(kComponent),
                                        .operation = std::move(operation),
                                        .expected = std::move(expected),
                                        .actual = std::move(actual),
                                        .hint = std::move(hint)});
}

[[noreturn]] void throw_compatibility(std::string operation,
                                      std::string expected, std::string actual,
                                      std::string hint) {
  throw CompatibilityError("Incompatible RNG checkpoint state",
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

std::string cuda_field_name(std::size_t index) {
  return "cuda." + std::to_string(index);
}

std::uint64_t checked_add(std::uint64_t total, std::uint64_t value,
                          std::uint64_t limit, std::string_view operation) {
  if (value > limit || total > limit - value) {
    throw_resource(std::string(operation), limit,
                   value > std::numeric_limits<std::uint64_t>::max() - total
                       ? std::numeric_limits<std::uint64_t>::max()
                       : total + value,
                   "Reduce selected generator state or raise the bound.");
  }
  return total + value;
}

std::uint64_t tensor_bytes(const Tensor& tensor) {
  return static_cast<std::uint64_t>(tensor.numel());
}

void validate_options(const RngCheckpointParticipantOptions& options) {
  if (options.participant_id.empty() ||
      options.participant_id.find('\0') != std::string::npos) {
    throw_invalid("participant_id", "a nonempty non-NUL stable identity",
                  "empty or contains NUL",
                  "Assign one deterministic process RNG identity.");
  }
  if (options.max_identifier_bytes == 0 || options.max_cuda_devices == 0 ||
      options.max_generator_state_bytes == 0 ||
      options.max_topology_bytes == 0 || options.max_total_bytes == 0) {
    throw_invalid("limits", "positive resource limits", "zero limit",
                  "Configure bounded positive RNG checkpoint resources.");
  }
  if (options.participant_id.size() > options.max_identifier_bytes) {
    throw_resource("participant_id", options.max_identifier_bytes,
                   options.participant_id.size(),
                   "Use a shorter stable process RNG identity.");
  }
}

std::vector<std::byte> encode_topology(
    const RngState& state, const RngCheckpointParticipantOptions& options) {
  if (state.cuda.size() >
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw_resource("cuda.count", std::numeric_limits<std::uint32_t>::max(),
                   state.cuda.size(),
                   "Use a supported visible CUDA device topology.");
  }
  std::vector<std::byte> bytes;
  bytes.reserve(24 + state.cuda.size() * 8);
  append_u32(bytes, kTopologyMagic);
  append_u32(bytes, kSchemaVersion);
  append_u8(bytes, options.include_cuda ? 1U : 0U);
  append_u8(bytes, 0U);
  append_u8(bytes, 0U);
  append_u8(bytes, 0U);
  append_u32(bytes, static_cast<std::uint32_t>(state.cuda.size()));
  append_u64(bytes, tensor_bytes(state.cpu));
  for (const auto& cuda_state : state.cuda) {
    append_u64(bytes, tensor_bytes(cuda_state));
  }
  if (bytes.size() > options.max_topology_bytes) {
    throw_resource("topology", options.max_topology_bytes, bytes.size(),
                   "Reduce device count or raise max_topology_bytes.");
  }
  return bytes;
}

struct Snapshot {
  RngState state;                   ///< Owned selected generator state.
  std::vector<std::byte> topology;  ///< Canonical immutable descriptor.
  StateManifest manifest;           ///< Storage-neutral field contract.
};

Snapshot inspect_rng(const RngState& state,
                     const RngCheckpointParticipantOptions& options) {
  validate_rng_state(state, options.include_cuda);
  if (state.cuda.size() > options.max_cuda_devices) {
    throw_resource("cuda.count", options.max_cuda_devices, state.cuda.size(),
                   "Reduce visible devices or raise max_cuda_devices.");
  }
  const auto cpu_bytes = tensor_bytes(state.cpu);
  if (cpu_bytes > options.max_generator_state_bytes) {
    throw_resource("cpu", options.max_generator_state_bytes, cpu_bytes,
                   "Raise the per-generator state bound for this runtime.");
  }
  auto topology = encode_topology(state, options);
  std::uint64_t total = topology.size();
  total = checked_add(total, cpu_bytes, options.max_total_bytes, "total");
  for (std::size_t index = 0; index < state.cuda.size(); ++index) {
    const auto bytes = tensor_bytes(state.cuda[index]);
    if (bytes > options.max_generator_state_bytes) {
      throw_resource(cuda_field_name(index), options.max_generator_state_bytes,
                     bytes,
                     "Raise the per-generator state bound for this runtime.");
    }
    total = checked_add(total, bytes, options.max_total_bytes, "total");
  }

  StateManifest manifest{
      .participant_id = options.participant_id,
      .schema_version = kSchemaVersion,
      .restore_level = RestoreLevel::exact,
      .fields = {{.name = "topology",
                  .media_type = "application/vnd.jyppx.rng-topology-v1",
                  .max_bytes = topology.size()},
                 {.name = "cpu",
                  .media_type = "application/vnd.jyppx.rng-state-raw-v1",
                  .max_bytes = cpu_bytes}},
      .degradation_reasons = {}};
  manifest.fields.reserve(state.cuda.size() + 2);
  for (std::size_t index = 0; index < state.cuda.size(); ++index) {
    manifest.fields.push_back(
        {.name = cuda_field_name(index),
         .media_type = "application/vnd.jyppx.rng-state-raw-v1",
         .max_bytes = tensor_bytes(state.cuda[index])});
  }
  validate_state_manifest(manifest);
  return {.state = state,
          .topology = std::move(topology),
          .manifest = std::move(manifest)};
}

Snapshot inspect_current(const RngCheckpointParticipantOptions& options) {
  return inspect_rng(get_rng_state(options.include_cuda), options);
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

Tensor tensor_from_bytes(std::span<const std::byte> bytes) {
  auto tensor = torch::empty(
      {static_cast<std::int64_t>(bytes.size())},
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
  if (!bytes.empty()) {
    std::memcpy(tensor.data_ptr(), bytes.data(), bytes.size());
  }
  return tensor;
}

[[noreturn]] void translate_bad_alloc(std::string operation) {
  throw ResourceExhausted(
      "Unable to allocate RNG checkpoint state",
      ErrorDetails{.component = std::string(kComponent),
                   .operation = std::move(operation),
                   .expected = "memory within configured RNG limits",
                   .actual = "allocation failed",
                   .hint = "Reduce selected state or the active workload."});
}

}  // namespace

RngCheckpointParticipant::RngCheckpointParticipant(
    RngCheckpointParticipantOptions options)
    : options_(std::move(options)) {
  if constexpr (std::endian::native != std::endian::little) {
    throw NotImplemented(
        "RNG checkpoint byte order is unsupported",
        ErrorDetails{.component = std::string(kComponent),
                     .operation = "platform.endian",
                     .expected = "little endian",
                     .actual = "non-little endian",
                     .hint = "Add an explicit topology byte-swap codec."});
  }
  validate_options(options_);
  static_cast<void>(describe_state());
}

std::string RngCheckpointParticipant::participant_id() const {
  return options_.participant_id;
}

StateManifest RngCheckpointParticipant::describe_state() const {
  try {
    return inspect_current(options_).manifest;
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    translate_bad_alloc("describe");
  }
}

void RngCheckpointParticipant::save_state(StateWriter& writer) const {
  try {
    const auto snapshot = inspect_current(options_);
    writer.write_blob("topology", snapshot.topology);
    const auto* cpu =
        static_cast<const std::byte*>(snapshot.state.cpu.data_ptr());
    writer.write_blob("cpu", std::span<const std::byte>(
                                 cpu, tensor_bytes(snapshot.state.cpu)));
    for (std::size_t index = 0; index < snapshot.state.cuda.size(); ++index) {
      const auto& state = snapshot.state.cuda[index];
      const auto* data = static_cast<const std::byte*>(state.data_ptr());
      writer.write_blob(cuda_field_name(index),
                        std::span<const std::byte>(data, tensor_bytes(state)));
    }
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    translate_bad_alloc("save");
  }
}

void RngCheckpointParticipant::load_state(const StateReader& reader,
                                          RestorePolicy policy) {
  try {
    if (!restore_policy_accepts(policy, RestoreLevel::exact)) {
      throw_compatibility("restore.policy", "a policy accepting exact state",
                          std::string(restore_policy_name(policy)),
                          "Use a declared RestorePolicy value.");
    }
    const auto current = inspect_current(options_);
    const auto& stored_manifest = reader.manifest();
    validate_state_manifest(stored_manifest);
    if (!manifest_equal(stored_manifest, current.manifest)) {
      throw_compatibility("restore.manifest", options_.participant_id,
                          stored_manifest.participant_id,
                          "Use state from the same selected RNG topology.");
    }
    for (const auto& field : current.manifest.fields) {
      if (field.required && !reader.contains(field.name)) {
        throw_serialization("restore.field", field.name, "missing",
                            "Provide every required generator state field.");
      }
    }

    const auto topology =
        reader.read_blob("topology", current.manifest.fields[0].max_bytes);
    if (topology.size() != current.topology.size() ||
        !std::equal(topology.begin(), topology.end(),
                    current.topology.begin())) {
      throw_compatibility(
          "restore.topology",
          "identical selected scope, device order, and state lengths",
          "topology differs",
          "Restore under the same runtime and visible-device mapping.");
    }

    std::vector<std::span<const std::byte>> encoded;
    encoded.reserve(current.manifest.fields.size() - 1);
    for (std::size_t index = 1; index < current.manifest.fields.size();
         ++index) {
      const auto& field = current.manifest.fields[index];
      const auto bytes = reader.read_blob(field.name, field.max_bytes);
      if (bytes.size() != field.max_bytes) {
        throw_serialization("restore." + field.name,
                            std::to_string(field.max_bytes),
                            std::to_string(bytes.size()),
                            "Reject truncated or extended generator state.");
      }
      encoded.push_back(bytes);
    }

    RngState candidate;
    candidate.cpu = tensor_from_bytes(encoded.front());
    candidate.cuda.reserve(encoded.size() - 1);
    for (std::size_t index = 1; index < encoded.size(); ++index) {
      candidate.cuda.push_back(tensor_from_bytes(encoded[index]));
    }
    validate_rng_state(candidate, options_.include_cuda);
    set_rng_state(candidate, options_.include_cuda);
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    translate_bad_alloc("restore");
  }
}

const RngCheckpointParticipantOptions& RngCheckpointParticipant::options()
    const noexcept {
  return options_;
}

}  // namespace jyppx::core
