// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/nn/model_checkpoint_participant.hpp>

#include <torch/torch.h>

#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <array>
#include <bit>
#include <c10/core/ScalarType.h>
#include <c10/core/Storage.h>
#include <cstring>
#include <exception>
#include <iomanip>
#include <limits>
#include <map>
#include <new>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace jyppx::liuding::nn {
namespace {

constexpr char kComponent[] = "nn.model_checkpoint";
constexpr std::uint32_t kSchemaVersion = 1;
constexpr std::uint32_t kTopologyVersion = 1;
constexpr std::size_t kMaximumRank = 64;
constexpr std::array<std::byte, 8> kTopologyMagic = {
    std::byte{'J'}, std::byte{'Y'}, std::byte{'P'}, std::byte{'P'},
    std::byte{'X'}, std::byte{'M'}, std::byte{'D'}, std::byte{'L'}};

enum class TensorRole : std::uint8_t {
  parameter = 1,
  buffer = 2,
};

struct TensorEntry {
  TensorRole role;
  std::string key;
  Tensor tensor;
  std::uint8_t dtype_code;
  std::uint32_t storage_group;
  std::int64_t storage_offset;
  std::vector<std::int64_t> sizes;
  std::vector<std::int64_t> strides;
  std::uint64_t byte_count;
  bool requires_grad;
};

struct ModelSnapshot {
  std::vector<TensorEntry> entries;
  std::vector<std::byte> topology;
  StateManifest manifest;
};

[[noreturn]] void throw_invalid(std::string operation, std::string expected,
                                std::string actual, std::string hint) {
  throw InvalidArgument("Invalid model checkpoint participant",
                        ErrorDetails{.component = kComponent,
                                     .operation = std::move(operation),
                                     .expected = std::move(expected),
                                     .actual = std::move(actual),
                                     .hint = std::move(hint)});
}

[[noreturn]] void throw_serialization(std::string operation,
                                      std::string expected, std::string actual,
                                      std::string hint) {
  throw SerializationError("Malformed model checkpoint state",
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
      "Model checkpoint is incompatible with the current model",
      ErrorDetails{.component = kComponent,
                   .operation = std::move(operation),
                   .expected = std::move(expected),
                   .actual = std::move(actual),
                   .hint = std::move(hint)});
}

[[noreturn]] void throw_resource(std::string operation, std::uint64_t expected,
                                 std::uint64_t actual, std::string hint) {
  throw ResourceExhausted("Model checkpoint resource limit exceeded",
                          ErrorDetails{.component = kComponent,
                                       .operation = std::move(operation),
                                       .expected = std::to_string(expected),
                                       .actual = std::to_string(actual),
                                       .hint = std::move(hint)});
}

void validate_options(const ModelCheckpointParticipantOptions& options) {
  if (options.max_tensors == 0 || options.max_state_key_bytes == 0 ||
      options.max_topology_bytes == 0 || options.max_tensor_bytes == 0 ||
      options.max_total_bytes == 0) {
    throw_invalid("options", "positive resource limits", "zero",
                  "Set every model checkpoint limit explicitly above zero.");
  }
  if (options.max_tensors >
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw_invalid("max_tensors", "at most uint32 maximum",
                  std::to_string(options.max_tensors),
                  "Use a bounded model state tensor count.");
  }
  if (options.max_state_key_bytes >
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw_invalid("max_state_key_bytes", "at most uint32 maximum",
                  std::to_string(options.max_state_key_bytes),
                  "Use bounded model state keys.");
  }
  StateManifest probe{
      .participant_id = options.participant_id,
      .schema_version = kSchemaVersion,
      .restore_level = RestoreLevel::exact,
      .fields = {{.name = "topology", .max_bytes = 1}},
      .degradation_reasons = {},
  };
  validate_state_manifest(probe);
}

std::uint8_t dtype_code(c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::Byte:
      return 1;
    case c10::ScalarType::Char:
      return 2;
    case c10::ScalarType::Short:
      return 3;
    case c10::ScalarType::Int:
      return 4;
    case c10::ScalarType::Long:
      return 5;
    case c10::ScalarType::Half:
      return 6;
    case c10::ScalarType::Float:
      return 7;
    case c10::ScalarType::Double:
      return 8;
    case c10::ScalarType::ComplexHalf:
      return 9;
    case c10::ScalarType::ComplexFloat:
      return 10;
    case c10::ScalarType::ComplexDouble:
      return 11;
    case c10::ScalarType::Bool:
      return 12;
    case c10::ScalarType::BFloat16:
      return 13;
    default:
      throw_invalid("tensor.dtype", "a supported dense training scalar type",
                    std::string(c10::toString(dtype)),
                    "Convert unsupported model state before checkpointing.");
  }
}

c10::ScalarType scalar_type(std::uint8_t code) {
  switch (code) {
    case 1:
      return c10::ScalarType::Byte;
    case 2:
      return c10::ScalarType::Char;
    case 3:
      return c10::ScalarType::Short;
    case 4:
      return c10::ScalarType::Int;
    case 5:
      return c10::ScalarType::Long;
    case 6:
      return c10::ScalarType::Half;
    case 7:
      return c10::ScalarType::Float;
    case 8:
      return c10::ScalarType::Double;
    case 9:
      return c10::ScalarType::ComplexHalf;
    case 10:
      return c10::ScalarType::ComplexFloat;
    case 11:
      return c10::ScalarType::ComplexDouble;
    case 12:
      return c10::ScalarType::Bool;
    case 13:
      return c10::ScalarType::BFloat16;
    default:
      throw_serialization("topology.dtype", "a schema-version-1 dtype code",
                          std::to_string(code),
                          "Reject unknown tensor encodings before allocation.");
  }
}

std::uint64_t element_size(std::uint8_t code) {
  switch (code) {
    case 1:
    case 2:
    case 12:
      return 1;
    case 3:
    case 6:
    case 13:
      return 2;
    case 4:
    case 7:
    case 9:
      return 4;
    case 5:
    case 8:
    case 10:
      return 8;
    case 11:
      return 16;
    default:
      throw_serialization("topology.dtype", "a schema-version-1 dtype code",
                          std::to_string(code),
                          "Reject unknown tensor encodings before allocation.");
  }
}

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right,
                          std::uint64_t limit, std::string_view operation) {
  if (right > limit || left > limit - right) {
    throw_resource(std::string(operation), limit, right,
                   "Reduce model state or raise the explicit limit.");
  }
  return left + right;
}

std::uint64_t checked_tensor_bytes(const Tensor& tensor, std::uint8_t code) {
  const auto count = tensor.numel();
  if (count < 0) {
    throw_invalid("tensor.numel", "a non-negative element count",
                  std::to_string(count),
                  "Materialize symbolic model state before checkpointing.");
  }
  const auto unsigned_count = static_cast<std::uint64_t>(count);
  const auto width = element_size(code);
  if (unsigned_count > std::numeric_limits<std::uint64_t>::max() / width) {
    throw_resource("tensor.byte_count",
                   std::numeric_limits<std::uint64_t>::max(), unsigned_count,
                   "Reduce tensor dimensions before checkpointing.");
  }
  return unsigned_count * width;
}

std::string tensor_field_name(std::size_t index) {
  std::ostringstream output;
  output << "tensor." << std::setw(10) << std::setfill('0') << index;
  return output.str();
}

void append_u8(std::vector<std::byte>& output, std::uint8_t value) {
  output.push_back(static_cast<std::byte>(value));
}

void append_u32(std::vector<std::byte>& output, std::uint32_t value) {
  for (std::size_t shift = 0; shift < 32; shift += 8) {
    append_u8(output, static_cast<std::uint8_t>(value >> shift));
  }
}

void append_u64(std::vector<std::byte>& output, std::uint64_t value) {
  for (std::size_t shift = 0; shift < 64; shift += 8) {
    append_u8(output, static_cast<std::uint8_t>(value >> shift));
  }
}

void append_i64(std::vector<std::byte>& output, std::int64_t value) {
  append_u64(output, static_cast<std::uint64_t>(value));
}

std::vector<std::byte> encode_topology(
    const std::vector<TensorEntry>& entries,
    const ModelCheckpointParticipantOptions& options) {
  std::vector<std::byte> output(kTopologyMagic.begin(), kTopologyMagic.end());
  append_u32(output, kTopologyVersion);
  append_u32(output, static_cast<std::uint32_t>(entries.size()));
  for (const auto& entry : entries) {
    append_u8(output, static_cast<std::uint8_t>(entry.role));
    append_u8(output, entry.dtype_code);
    append_u8(output, entry.requires_grad ? 1 : 0);
    append_u8(output, 0);
    append_u32(output, static_cast<std::uint32_t>(entry.key.size()));
    append_u32(output, static_cast<std::uint32_t>(entry.sizes.size()));
    append_u32(output, entry.storage_group);
    append_i64(output, entry.storage_offset);
    append_u64(output, entry.byte_count);
    output.insert(output.end(),
                  reinterpret_cast<const std::byte*>(entry.key.data()),
                  reinterpret_cast<const std::byte*>(entry.key.data()) +
                      entry.key.size());
    for (const auto dimension : entry.sizes) {
      append_i64(output, dimension);
    }
    for (const auto stride : entry.strides) {
      append_i64(output, stride);
    }
    if (output.size() > options.max_topology_bytes) {
      throw_resource("topology.bytes", options.max_topology_bytes,
                     output.size(),
                     "Reduce model topology or raise max_topology_bytes.");
    }
  }
  return output;
}

class ByteCursor final {
 public:
  explicit ByteCursor(std::span<const std::byte> bytes) : bytes_(bytes) {}

  std::uint8_t read_u8(std::string_view field) {
    require(1, field);
    return std::to_integer<std::uint8_t>(bytes_[position_++]);
  }

  std::uint32_t read_u32(std::string_view field) {
    std::uint32_t value = 0;
    for (std::size_t shift = 0; shift < 32; shift += 8) {
      value |= static_cast<std::uint32_t>(read_u8(field)) << shift;
    }
    return value;
  }

  std::uint64_t read_u64(std::string_view field) {
    std::uint64_t value = 0;
    for (std::size_t shift = 0; shift < 64; shift += 8) {
      value |= static_cast<std::uint64_t>(read_u8(field)) << shift;
    }
    return value;
  }

  std::int64_t read_i64(std::string_view field) {
    return std::bit_cast<std::int64_t>(read_u64(field));
  }

  std::string read_string(std::size_t length, std::string_view field) {
    require(length, field);
    const auto* begin =
        reinterpret_cast<const char*>(bytes_.data() + position_);
    position_ += length;
    return std::string(begin, length);
  }

  bool empty() const noexcept { return position_ == bytes_.size(); }

 private:
  void require(std::size_t count, std::string_view field) const {
    if (count > bytes_.size() - position_) {
      throw_serialization(std::string(field), "complete topology bytes",
                          "truncated",
                          "Reject incomplete metadata before model mutation.");
    }
  }

  std::span<const std::byte> bytes_;
  std::size_t position_ = 0;
};

void validate_topology_blob(std::span<const std::byte> bytes,
                            const ModelCheckpointParticipantOptions& options) {
  if (bytes.size() > options.max_topology_bytes) {
    throw_resource("topology.bytes", options.max_topology_bytes, bytes.size(),
                   "Reject oversized topology before parsing.");
  }
  ByteCursor cursor(bytes);
  for (const auto expected : kTopologyMagic) {
    if (cursor.read_u8("topology.magic") !=
        std::to_integer<std::uint8_t>(expected)) {
      throw_serialization("topology.magic", "JYPPXMDL", "mismatch",
                          "Use a model topology produced by this schema.");
    }
  }
  const auto version = cursor.read_u32("topology.version");
  if (version != kTopologyVersion) {
    throw_compatibility("topology.version", std::to_string(kTopologyVersion),
                        std::to_string(version),
                        "Migrate model topology metadata explicitly.");
  }
  const auto count = cursor.read_u32("topology.count");
  if (count == 0) {
    throw_serialization("topology.count", "at least one state tensor", "0",
                        "Do not restore a stateless model participant.");
  }
  if (count > options.max_tensors) {
    throw_resource("topology.count", options.max_tensors, count,
                   "Reject excessive tensor counts before allocation.");
  }

  std::set<std::string> keys;
  std::uint64_t total = bytes.size();
  for (std::uint32_t index = 0; index < count; ++index) {
    const auto role = cursor.read_u8("topology.role");
    if (role != static_cast<std::uint8_t>(TensorRole::parameter) &&
        role != static_cast<std::uint8_t>(TensorRole::buffer)) {
      throw_serialization("topology.role", "parameter or buffer",
                          std::to_string(role),
                          "Reject unknown model state roles.");
    }
    const auto code = cursor.read_u8("topology.dtype");
    static_cast<void>(scalar_type(code));
    const auto requires_grad = cursor.read_u8("topology.requires_grad");
    if (requires_grad > 1 ||
        (role == static_cast<std::uint8_t>(TensorRole::buffer) &&
         requires_grad != 0)) {
      throw_serialization("topology.requires_grad",
                          "a boolean set only for parameters",
                          std::to_string(requires_grad),
                          "Reject inconsistent model state metadata.");
    }
    if (cursor.read_u8("topology.reserved") != 0) {
      throw_serialization("topology.reserved", "zero", "non-zero",
                          "Reject metadata using unknown schema flags.");
    }
    const auto key_length = cursor.read_u32("topology.key_length");
    const auto rank = cursor.read_u32("topology.rank");
    const auto storage_group = cursor.read_u32("topology.storage_group");
    const auto storage_offset = cursor.read_i64("topology.storage_offset");
    const auto byte_count = cursor.read_u64("topology.byte_count");
    if (key_length == 0 || key_length > options.max_state_key_bytes) {
      throw_resource("topology.key_length", options.max_state_key_bytes,
                     key_length, "Reject empty or oversized model state keys.");
    }
    if (rank > kMaximumRank) {
      throw_resource("topology.rank", kMaximumRank, rank,
                     "Reject excessive tensor ranks before allocation.");
    }
    if (storage_group > index || storage_offset < 0) {
      throw_serialization(
          "topology.storage", "a prior alias group and non-negative offset",
          std::to_string(storage_group) + ":" + std::to_string(storage_offset),
          "Reject invalid storage topology.");
    }
    if (byte_count > options.max_tensor_bytes) {
      throw_resource("topology.byte_count", options.max_tensor_bytes,
                     byte_count, "Reject oversized tensors before allocation.");
    }
    const auto key = cursor.read_string(key_length, "topology.key");
    if (key.find('\0') != std::string::npos || !keys.insert(key).second) {
      throw_serialization("topology.key", "unique non-NUL state keys", key,
                          "Reject ambiguous model state mappings.");
    }

    std::uint64_t numel = 1;
    bool has_zero_dimension = false;
    for (std::uint32_t dimension_index = 0; dimension_index < rank;
         ++dimension_index) {
      const auto dimension = cursor.read_i64("topology.size");
      if (dimension < 0) {
        throw_serialization("topology.size", "non-negative dimensions",
                            std::to_string(dimension),
                            "Reject symbolic or malformed stored shapes.");
      }
      if (dimension == 0) {
        has_zero_dimension = true;
      } else if (!has_zero_dimension) {
        const auto value = static_cast<std::uint64_t>(dimension);
        if (numel > std::numeric_limits<std::uint64_t>::max() / value) {
          throw_resource("topology.numel",
                         std::numeric_limits<std::uint64_t>::max(), value,
                         "Reject shapes whose element count overflows.");
        }
        numel *= value;
      }
    }
    for (std::uint32_t dimension_index = 0; dimension_index < rank;
         ++dimension_index) {
      static_cast<void>(cursor.read_i64("topology.stride"));
    }
    if (has_zero_dimension) {
      numel = 0;
    }
    const auto width = element_size(code);
    if (numel > std::numeric_limits<std::uint64_t>::max() / width ||
        numel * width != byte_count) {
      throw_serialization(
          "topology.byte_count", "shape multiplied by scalar width",
          std::to_string(byte_count), "Reject inconsistent tensor metadata.");
    }
    total = checked_add(total, std::max<std::uint64_t>(1, byte_count),
                        options.max_total_bytes, "state.total_bytes");
  }
  if (!cursor.empty()) {
    throw_serialization("topology.trailing_bytes", "no trailing metadata",
                        "present", "Reject ambiguous topology extensions.");
  }
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

ModelSnapshot inspect_model(const ModelStateView& state,
                            const ModelCheckpointParticipantOptions& options) {
  std::vector<TensorEntry> entries;
  const auto parameters = state.named_parameters();
  const auto buffers = state.named_buffers();
  if (parameters.size() > options.max_tensors ||
      buffers.size() > options.max_tensors - parameters.size()) {
    throw_resource("state.tensor_count", options.max_tensors,
                   parameters.size() + buffers.size(),
                   "Reduce model state or raise max_tensors.");
  }
  entries.reserve(parameters.size() + buffers.size());

  std::map<const void*, std::uint32_t> storage_groups;
  std::set<std::string> keys;
  std::uint64_t total_tensor_bytes = 0;
  auto append = [&](TensorRole role, const std::string& key,
                    const Tensor& tensor) {
    if (key.empty() || key.size() > options.max_state_key_bytes ||
        key.find('\0') != std::string::npos || !keys.insert(key).second) {
      throw_invalid("state.key", "unique non-NUL keys within the limit", key,
                    "Use stable, bounded parameter and buffer names.");
    }
    if (!tensor.defined() || tensor.is_meta() || tensor.is_quantized() ||
        tensor.layout() != c10::Layout::Strided ||
        !tensor.is_non_overlapping_and_dense()) {
      throw_invalid(
          "state.tensor", "defined dense strided non-overlapping storage", key,
          "Materialize unsupported model state before checkpointing.");
    }
    if (tensor.dim() > static_cast<std::int64_t>(kMaximumRank)) {
      throw_resource("state.rank", kMaximumRank, tensor.dim(),
                     "Reduce tensor rank before checkpointing.");
    }
    const auto code = dtype_code(tensor.scalar_type());
    const auto bytes = checked_tensor_bytes(tensor, code);
    if (bytes > options.max_tensor_bytes) {
      throw_resource("state.tensor_bytes", options.max_tensor_bytes, bytes,
                     "Reduce the tensor or raise max_tensor_bytes.");
    }
    total_tensor_bytes =
        checked_add(total_tensor_bytes, std::max<std::uint64_t>(1, bytes),
                    options.max_total_bytes, "state.total_bytes");

    const void* storage = tensor.storage().unsafeGetStorageImpl();
    const auto [group, inserted] = storage_groups.emplace(
        storage, static_cast<std::uint32_t>(storage_groups.size()));
    static_cast<void>(inserted);
    entries.push_back(TensorEntry{.role = role,
                                  .key = key,
                                  .tensor = tensor,
                                  .dtype_code = code,
                                  .storage_group = group->second,
                                  .storage_offset = tensor.storage_offset(),
                                  .sizes = tensor.sizes().vec(),
                                  .strides = tensor.strides().vec(),
                                  .byte_count = bytes,
                                  .requires_grad = role == TensorRole::parameter
                                                       ? tensor.requires_grad()
                                                       : false});
  };

  for (const auto& item : parameters) {
    append(TensorRole::parameter, item.key(), item.value());
  }
  for (const auto& item : buffers) {
    append(TensorRole::buffer, item.key(), item.value());
  }
  if (entries.empty()) {
    throw_invalid("state", "at least one parameter or buffer", "empty",
                  "Do not register stateless modules as participants.");
  }

  auto topology = encode_topology(entries, options);
  static_cast<void>(checked_add(total_tensor_bytes, topology.size(),
                                options.max_total_bytes, "state.total_bytes"));

  StateManifest manifest{
      .participant_id = options.participant_id,
      .schema_version = kSchemaVersion,
      .restore_level = RestoreLevel::exact,
      .fields = {{.name = "topology",
                  .media_type = "application/vnd.jyppx.model-topology-v1",
                  .max_bytes = topology.size()}},
      .degradation_reasons = {},
  };
  manifest.fields.reserve(entries.size() + 1);
  for (std::size_t index = 0; index < entries.size(); ++index) {
    manifest.fields.push_back(
        {.name = tensor_field_name(index),
         .media_type = "application/vnd.jyppx.tensor-raw-v1",
         .max_bytes = std::max<std::uint64_t>(1, entries[index].byte_count)});
  }
  validate_state_manifest(manifest);
  return {.entries = std::move(entries),
          .topology = std::move(topology),
          .manifest = std::move(manifest)};
}

[[noreturn]] void translate_bad_alloc(std::string_view operation) {
  throw ResourceExhausted(
      "Unable to allocate model checkpoint state",
      ErrorDetails{.component = kComponent,
                   .operation = std::string(operation),
                   .expected = "memory within configured model limits",
                   .actual = "allocation failed",
                   .hint = "Reduce model state or the active workload."});
}

[[noreturn]] void translate_backend(const c10::Error& cause,
                                    std::string operation) {
  detail::rethrow_backend_error(
      cause, detail::BackendErrorContext{
                 .component = kComponent,
                 .operation = std::move(operation),
                 .failure_kind = detail::BackendFailureKind::unknown,
                 .expected = "a supported dense model state operation",
                 .hint = "Inspect the nested LibTorch diagnostic and device."});
}

}  // namespace

ModelCheckpointParticipant::ModelCheckpointParticipant(
    std::shared_ptr<ModelStateView> state,
    ModelCheckpointParticipantOptions options)
    : state_(std::move(state)), options_(std::move(options)) {
  if (!state_) {
    throw_invalid("state", "a non-null shared ModelStateView", "null",
                  "Create a model state view before its participant.");
  }
  if constexpr (std::endian::native != std::endian::little) {
    throw NotImplemented(
        "Model checkpoint byte order is unsupported",
        ErrorDetails{.component = kComponent,
                     .operation = "platform.endian",
                     .expected = "little endian",
                     .actual = "non-little endian",
                     .hint = "Add an explicit raw tensor byte swap codec."});
  }
  validate_options(options_);
  static_cast<void>(describe_state());
}

ModelCheckpointParticipant::ModelCheckpointParticipant(
    const ModelAdapter& adapter, ModelCheckpointParticipantOptions options)
    : ModelCheckpointParticipant(
          std::make_shared<ModelStateView>(adapter.state().shared_module()),
          std::move(options)) {}

std::string ModelCheckpointParticipant::participant_id() const {
  return options_.participant_id;
}

StateManifest ModelCheckpointParticipant::describe_state() const {
  try {
    return inspect_model(*state_, options_).manifest;
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    translate_bad_alloc("describe");
  } catch (const c10::Error& cause) {
    translate_backend(cause, "describe model state");
  }
}

void ModelCheckpointParticipant::save_state(StateWriter& writer) const {
  try {
    const auto snapshot = inspect_model(*state_, options_);
    writer.write_blob("topology", snapshot.topology);
    for (std::size_t index = 0; index < snapshot.entries.size(); ++index) {
      const auto& entry = snapshot.entries[index];
      auto cpu = entry.tensor.detach().to(torch::kCPU).contiguous();
      const auto actual_bytes = checked_tensor_bytes(cpu, entry.dtype_code);
      if (actual_bytes != entry.byte_count) {
        throw SerializationError(
            "Model tensor changed while checkpointing",
            ErrorDetails{.component = kComponent,
                         .operation = entry.key,
                         .expected = std::to_string(entry.byte_count),
                         .actual = std::to_string(actual_bytes),
                         .hint = "Quiesce model mutation during save."});
      }
      const auto* data = static_cast<const std::byte*>(cpu.data_ptr());
      writer.write_blob(tensor_field_name(index),
                        std::span<const std::byte>(data, actual_bytes));
    }
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    translate_bad_alloc("save");
  } catch (const c10::Error& cause) {
    translate_backend(cause, "save model state");
  }
}

void ModelCheckpointParticipant::load_state(const StateReader& reader,
                                            RestorePolicy policy) {
  try {
    if (!restore_policy_accepts(policy, RestoreLevel::exact)) {
      throw_compatibility("restore.policy", "a policy accepting exact state",
                          std::string(restore_policy_name(policy)),
                          "Use a declared RestorePolicy value.");
    }
    const auto current = inspect_model(*state_, options_);
    const auto& stored_manifest = reader.manifest();
    validate_state_manifest(stored_manifest);
    if (!manifest_equal(stored_manifest, current.manifest)) {
      throw_compatibility("restore.manifest", options_.participant_id,
                          stored_manifest.participant_id,
                          "Use state produced by the same model schema.");
    }
    for (const auto& field : current.manifest.fields) {
      if (field.required && !reader.contains(field.name)) {
        throw_serialization("restore.field", field.name, "missing",
                            "Provide every required model state field.");
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
          "identical state keys, roles, types, shapes, "
          "strides, aliases, and gradient flags",
          "topology differs",
          "Apply an explicit model state migration before restore.");
    }

    std::vector<Tensor> candidates;
    candidates.reserve(current.entries.size());
    for (std::size_t index = 0; index < current.entries.size(); ++index) {
      const auto& entry = current.entries[index];
      const auto bytes =
          reader.read_blob(tensor_field_name(index),
                           std::max<std::uint64_t>(1, entry.byte_count));
      if (bytes.size() != entry.byte_count) {
        throw_serialization("restore.tensor_bytes",
                            std::to_string(entry.byte_count),
                            std::to_string(bytes.size()),
                            "Reject truncated or extended tensor state.");
      }
      if (entry.dtype_code == 12 &&
          std::any_of(bytes.begin(), bytes.end(), [](std::byte value) {
            return value != std::byte{0} && value != std::byte{1};
          })) {
        throw_serialization("restore.bool_tensor", "only 0 or 1 bytes",
                            "non-boolean byte",
                            "Reject malformed boolean tensor state.");
      }
      auto cpu =
          torch::empty(entry.sizes, torch::TensorOptions()
                                        .dtype(scalar_type(entry.dtype_code))
                                        .device(torch::kCPU));
      if (!bytes.empty()) {
        std::memcpy(cpu.data_ptr(), bytes.data(), bytes.size());
      }
      candidates.push_back(cpu.to(entry.tensor.device(),
                                  entry.tensor.scalar_type(), false, true));
    }

    std::vector<Tensor> rollback;
    rollback.reserve(current.entries.size());
    for (const auto& entry : current.entries) {
      rollback.push_back(entry.tensor.detach().clone());
    }

    at::NoGradGuard no_grad;
    std::size_t attempted = 0;
    try {
      for (; attempted < current.entries.size(); ++attempted) {
        current.entries[attempted].tensor.copy_(candidates[attempted]);
      }
    } catch (...) {
      const auto restore_failure = std::current_exception();
      std::string rollback_failure_key;
      const auto rollback_count =
          std::min(attempted + 1, current.entries.size());
      for (std::size_t offset = 0; offset < rollback_count; ++offset) {
        const auto index = rollback_count - offset - 1;
        try {
          current.entries[index].tensor.copy_(rollback[index]);
        } catch (...) {
          if (rollback_failure_key.empty()) {
            rollback_failure_key = current.entries[index].key;
          }
        }
      }
      if (!rollback_failure_key.empty()) {
        try {
          std::rethrow_exception(restore_failure);
        } catch (...) {
          std::throw_with_nested(InternalError(
              "Model restore failed and rollback was incomplete",
              ErrorDetails{
                  .component = kComponent,
                  .operation = "restore.rollback",
                  .expected = "all attempted tensors restored",
                  .actual = rollback_failure_key,
                  .hint = "Stop training; model state is indeterminate."}));
        }
      }
      std::rethrow_exception(restore_failure);
    }
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    translate_bad_alloc("restore");
  } catch (const c10::Error& cause) {
    translate_backend(cause, "restore model state");
  }
}

ModelStateView& ModelCheckpointParticipant::state() noexcept { return *state_; }

const ModelStateView& ModelCheckpointParticipant::state() const noexcept {
  return *state_;
}

const ModelCheckpointParticipantOptions& ModelCheckpointParticipant::options()
    const noexcept {
  return options_;
}

}  // namespace jyppx::liuding::nn
