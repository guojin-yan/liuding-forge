// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/foundation/sha256.hpp>
#include <jyppx/interop/safetensors.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <new>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace jyppx::interop {
namespace {

using Json = nlohmann::json;

constexpr std::string_view kComponent = "interop.safetensors";
constexpr std::string_view kMetadataKey = "__metadata__";
constexpr std::uint64_t kMaximumHeaderBytes = 100000000U;
constexpr std::array<SafeTensorDtype, 13> kDtypes{
    SafeTensorDtype::boolean, SafeTensorDtype::u8,   SafeTensorDtype::i8,
    SafeTensorDtype::u16,     SafeTensorDtype::i16,  SafeTensorDtype::u32,
    SafeTensorDtype::i32,     SafeTensorDtype::u64,  SafeTensorDtype::i64,
    SafeTensorDtype::f16,     SafeTensorDtype::bf16, SafeTensorDtype::f32,
    SafeTensorDtype::f64};

ErrorDetails Details(std::string_view operation, std::string expected,
                     std::string actual) {
  return {.component = std::string(kComponent),
          .operation = std::string(operation),
          .expected = std::move(expected),
          .actual = std::move(actual),
          .hint =
              "Use a complete supported safetensors artifact and explicit "
              "resource bounds."};
}

[[noreturn]] void Malformed(std::string_view operation,
                            std::string_view actual) {
  throw SerializationError("Malformed safetensors artifact",
                           Details(operation, "valid bounded safetensors data",
                                   std::string(actual)));
}

[[noreturn]] void Invalid(std::string_view operation, std::string_view actual) {
  throw InvalidArgument("Invalid safetensors encoding request",
                        Details(operation, "valid supported tensor sources",
                                std::string(actual)));
}

[[noreturn]] void Unsupported(std::string_view operation,
                              std::string_view actual) {
  throw CompatibilityError(
      "Unsupported safetensors encoding",
      Details(operation, "a supported byte-addressable little-endian dtype",
              std::string(actual)));
}

[[noreturn]] void Resource(std::string_view operation, std::uint64_t maximum) {
  throw ResourceExhausted(
      "Safetensors resource limit exceeded",
      Details(operation, "at most " + std::to_string(maximum),
              "more than the configured bound"));
}

[[noreturn]] void BadAllocation() {
  throw ResourceExhausted(
      "Safetensors memory allocation failed",
      Details("memory", "sufficient memory for bounded metadata/output",
              "allocation failed"));
}

void ValidateLimits(const SafeTensorsLimits& limits) {
  if (limits.max_file_bytes == 0U || limits.max_payload_bytes == 0U ||
      limits.max_tensor_bytes == 0U || limits.max_tensor_elements == 0U ||
      limits.max_dimension == 0U || limits.max_header_bytes == 0U ||
      limits.max_tensors == 0U || limits.max_rank == 0U ||
      limits.max_metadata_entries == 0U || limits.max_string_bytes == 0U ||
      limits.max_total_string_bytes == 0U || limits.max_json_nodes == 0U ||
      limits.max_json_depth == 0U ||
      limits.max_header_bytes > kMaximumHeaderBytes ||
      limits.max_file_bytes > (std::numeric_limits<std::size_t>::max)() ||
      limits.max_file_bytes >
          static_cast<std::uint64_t>(
              (std::numeric_limits<std::ptrdiff_t>::max)())) {
    Invalid("limits", "zero, platform-incompatible, or excessive header limit");
  }
}

void AddLimited(std::uint64_t& total, std::uint64_t addition,
                std::uint64_t maximum, std::string_view operation) {
  if (total > maximum || addition > maximum - total)
    Resource(operation, maximum);
  total += addition;
}

bool IsContinuation(unsigned char value) noexcept {
  return (value & 0xc0U) == 0x80U;
}

bool IsValidUtf8(std::string_view text) noexcept {
  for (std::size_t offset = 0; offset < text.size();) {
    const auto first = static_cast<unsigned char>(text[offset]);
    if (first <= 0x7fU) {
      ++offset;
    } else if (first >= 0xc2U && first <= 0xdfU) {
      if (text.size() - offset < 2U ||
          !IsContinuation(static_cast<unsigned char>(text[offset + 1U]))) {
        return false;
      }
      offset += 2U;
    } else if (first >= 0xe0U && first <= 0xefU) {
      if (text.size() - offset < 3U) return false;
      const auto second = static_cast<unsigned char>(text[offset + 1U]);
      if (!IsContinuation(second) ||
          !IsContinuation(static_cast<unsigned char>(text[offset + 2U])) ||
          (first == 0xe0U && second < 0xa0U) ||
          (first == 0xedU && second >= 0xa0U))
        return false;
      offset += 3U;
    } else if (first >= 0xf0U && first <= 0xf4U) {
      if (text.size() - offset < 4U) return false;
      const auto second = static_cast<unsigned char>(text[offset + 1U]);
      if (!IsContinuation(second) ||
          !IsContinuation(static_cast<unsigned char>(text[offset + 2U])) ||
          !IsContinuation(static_cast<unsigned char>(text[offset + 3U])) ||
          (first == 0xf0U && second < 0x90U) ||
          (first == 0xf4U && second >= 0x90U))
        return false;
      offset += 4U;
    } else {
      return false;
    }
  }
  return true;
}

void CountText(std::string_view value, const SafeTensorsLimits& limits,
               std::uint64_t& total, bool caller) {
  if (!IsValidUtf8(value)) {
    if (caller) Invalid("strings.utf8", "invalid UTF-8");
    Malformed("strings.utf8", "invalid UTF-8");
  }
  if (value.size() > limits.max_string_bytes) {
    Resource("strings.one", limits.max_string_bytes);
  }
  AddLimited(total, value.size(), limits.max_total_string_bytes,
             "strings.total");
}

std::uint64_t ReadInteger(const Json& value, std::string_view operation) {
  if (value.is_number_unsigned()) return value.get<std::uint64_t>();
  if (value.is_number_integer()) {
    const auto signed_value = value.get<std::int64_t>();
    if (signed_value >= 0) return static_cast<std::uint64_t>(signed_value);
  }
  Malformed(operation, "expected a non-negative uint64 integer");
}

std::uint64_t TensorBytes(std::span<const std::uint64_t> shape,
                          SafeTensorDtype dtype,
                          const SafeTensorsLimits& limits, bool caller) {
  if (shape.size() > limits.max_rank) Resource("shape.rank", limits.max_rank);
  const auto width = safetensors_dtype_size(dtype);
  if (width == 0U) Unsupported("dtype", safetensors_dtype_name(dtype));
  std::uint64_t elements = 1U;
  for (const auto dimension : shape) {
    if (dimension > limits.max_dimension)
      Resource("shape.dimension", limits.max_dimension);
    if (dimension != 0U &&
        elements > (std::numeric_limits<std::uint64_t>::max)() / dimension) {
      if (caller) Invalid("shape.product", "uint64 multiplication overflow");
      Malformed("shape.product", "uint64 multiplication overflow");
    }
    elements *= dimension;
    if (elements > limits.max_tensor_elements) {
      Resource("shape.elements", limits.max_tensor_elements);
    }
  }
  if (elements > (std::numeric_limits<std::uint64_t>::max)() / width) {
    if (caller) Invalid("shape.bytes", "uint64 multiplication overflow");
    Malformed("shape.bytes", "uint64 multiplication overflow");
  }
  const auto bytes = elements * width;
  if (bytes > limits.max_tensor_bytes)
    Resource("tensor.bytes", limits.max_tensor_bytes);
  return bytes;
}

Json ParseJson(std::string_view header, const SafeTensorsLimits& limits) {
  if (!IsValidUtf8(header)) Malformed("header.utf8", "invalid UTF-8");
  if (header.empty() || header.front() != '{') {
    Malformed("header.start", "header must start with '{'");
  }
  // The format permits ASCII space padding, not another trailing JSON value,
  // a BOM, or arbitrary post-object whitespace.
  const auto last = header.find_last_not_of(' ');
  if (last == std::string_view::npos || header[last] != '}') {
    Malformed("header.padding",
              "header must end with an object and ASCII spaces");
  }
  std::vector<std::unordered_set<std::string>> object_keys;
  std::uint64_t nodes = 0;
  std::uint64_t strings = 0;
  auto callback = [&](int depth, Json::parse_event_t event, Json& parsed) {
    if ((event == Json::parse_event_t::object_start ||
         event == Json::parse_event_t::array_start) &&
        static_cast<std::size_t>(depth + 1) > limits.max_json_depth) {
      Resource("json.depth", limits.max_json_depth);
    }
    if (event == Json::parse_event_t::object_start ||
        event == Json::parse_event_t::array_start ||
        event == Json::parse_event_t::key ||
        event == Json::parse_event_t::value) {
      AddLimited(nodes, 1U, limits.max_json_nodes, "json.nodes");
    }
    if (event == Json::parse_event_t::object_start) {
      object_keys.emplace_back();
    } else if (event == Json::parse_event_t::key) {
      const auto& key = parsed.get_ref<const std::string&>();
      if (object_keys.empty() || !object_keys.back().insert(key).second) {
        Malformed("json.key", "duplicate object key");
      }
    } else if (event == Json::parse_event_t::object_end) {
      if (!object_keys.empty()) object_keys.pop_back();
    }
    if ((event == Json::parse_event_t::key ||
         event == Json::parse_event_t::value) &&
        parsed.is_string()) {
      CountText(parsed.get_ref<const std::string&>(), limits, strings, false);
    }
    return true;
  };
  return Json::parse(header.begin(), header.end(), callback, true, false);
}

SafeTensorsDirectory ParseDirectory(std::span<const std::byte> bytes,
                                    const SafeTensorsLimits& limits) {
  ValidateLimits(limits);
  if (bytes.size() > limits.max_file_bytes)
    Resource("file.bytes", limits.max_file_bytes);
  if (bytes.size() < 8U) Malformed("header.prefix", "truncated length prefix");
  std::uint64_t header_size = 0U;
  for (unsigned index = 0U; index < 8U; ++index) {
    header_size |= static_cast<std::uint64_t>(bytes[index]) << (index * 8U);
  }
  if (header_size > limits.max_header_bytes)
    Resource("header.bytes", limits.max_header_bytes);
  if (header_size > bytes.size() - 8U)
    Malformed("header.bytes", "truncated header");
  const auto payload_offset = 8U + header_size;
  const auto payload_size = bytes.size() - payload_offset;
  if (payload_size > limits.max_payload_bytes)
    Resource("payload.bytes", limits.max_payload_bytes);
  const auto root = ParseJson({reinterpret_cast<const char*>(bytes.data() + 8U),
                               static_cast<std::size_t>(header_size)},
                              limits);
  if (!root.is_object()) Malformed("header.root", "expected an object");
  SafeTensorsDirectory result{.metadata = {},
                              .tensors = {},
                              .payload_offset = payload_offset,
                              .payload_size = payload_size};
  for (const auto& [name, value] : root.items()) {
    if (name == kMetadataKey) {
      if (!value.is_object())
        Malformed("metadata", "expected a string-to-string object");
      if (value.size() > limits.max_metadata_entries)
        Resource("metadata.entries", limits.max_metadata_entries);
      for (const auto& [key, item] : value.items()) {
        if (!item.is_string())
          Malformed("metadata.value", "metadata values must be strings");
        result.metadata.emplace(key, item.get<std::string>());
      }
      continue;
    }
    if (result.tensors.size() == limits.max_tensors)
      Resource("tensors.count", limits.max_tensors);
    if (!value.is_object() || value.size() != 3U || !value.contains("dtype") ||
        !value.contains("shape") || !value.contains("data_offsets")) {
      Malformed("tensor.fields",
                "expected exactly dtype, shape, and data_offsets");
    }
    if (!value.at("dtype").is_string())
      Malformed("tensor.dtype", "expected a string");
    if (!value.at("shape").is_array())
      Malformed("tensor.shape", "expected an array");
    if (value.at("shape").size() > limits.max_rank)
      Resource("shape.rank", limits.max_rank);
    const auto& offsets = value.at("data_offsets");
    if (!offsets.is_array() || offsets.size() != 2U)
      Malformed("tensor.offsets", "expected two integers");
    SafeTensorInfo tensor{.name = name,
                          .dtype = safetensors_dtype_from_name(
                              value.at("dtype").get_ref<const std::string&>()),
                          .shape = {},
                          .data_begin = ReadInteger(offsets[0], "tensor.begin"),
                          .data_end = ReadInteger(offsets[1], "tensor.end")};
    tensor.shape.reserve(value.at("shape").size());
    for (const auto& dimension : value.at("shape")) {
      tensor.shape.push_back(ReadInteger(dimension, "tensor.dimension"));
    }
    const auto expected =
        TensorBytes(tensor.shape, tensor.dtype, limits, false);
    if (tensor.data_end < tensor.data_begin || tensor.data_end > payload_size ||
        tensor.data_end - tensor.data_begin != expected) {
      Malformed("tensor.extent",
                "shape/dtype byte count differs from bounded offsets");
    }
    result.tensors.push_back(std::move(tensor));
  }
  std::vector<const SafeTensorInfo*> physical;
  physical.reserve(result.tensors.size());
  for (const auto& tensor : result.tensors) physical.push_back(&tensor);
  std::sort(physical.begin(), physical.end(),
            [](const auto* left, const auto* right) {
              if (left->data_begin != right->data_begin)
                return left->data_begin < right->data_begin;
              if (left->data_end != right->data_end)
                return left->data_end < right->data_end;
              return left->name < right->name;
            });
  std::uint64_t cursor = 0U;
  for (const auto* tensor : physical) {
    if (tensor->data_begin != cursor)
      Malformed("payload.coverage", "overlapping tensors or a hole");
    cursor = tensor->data_end;
  }
  if (cursor != payload_size)
    Malformed("payload.coverage", "trailing unindexed payload");
  // nlohmann::json object iteration is sorted by key, independent of the
  // original header order. Physical offsets were separately validated above.
  return result;
}

std::vector<std::byte> Encode(
    std::span<const SafeTensorSource> tensors,
    const std::map<std::string, std::string>& metadata,
    const SafeTensorsLimits& limits) {
  ValidateLimits(limits);
  if (tensors.size() > limits.max_tensors)
    Resource("tensors.count", limits.max_tensors);
  if (metadata.size() > limits.max_metadata_entries)
    Resource("metadata.entries", limits.max_metadata_entries);
  if (limits.max_json_depth <
      (tensors.empty() ? (metadata.empty() ? 1U : 2U) : 3U)) {
    Resource("json.depth", limits.max_json_depth);
  }
  std::uint64_t strings = 0U;
  std::uint64_t nodes = 1U;
  std::uint64_t payload_bytes = 0U;
  std::unordered_set<std::string_view> names;
  names.reserve(tensors.size());
  std::vector<const SafeTensorSource*> physical;
  physical.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    CountText(tensor.name, limits, strings, true);
    if (tensor.name == kMetadataKey || !names.insert(tensor.name).second) {
      Invalid("tensor.name", "reserved or duplicate tensor name");
    }
    if (tensor.byte_order != SafeTensorByteOrder::little) {
      Unsupported("tensor.endianness",
                  "source bytes are not explicitly little-endian");
    }
    const auto size = TensorBytes(tensor.shape, tensor.dtype, limits, true);
    if (size != tensor.data.size())
      Invalid("tensor.bytes", "shape/dtype does not match supplied bytes");
    AddLimited(payload_bytes, size, limits.max_payload_bytes, "payload.bytes");
    CountText("dtype", limits, strings, true);
    CountText("shape", limits, strings, true);
    CountText("data_offsets", limits, strings, true);
    CountText(safetensors_dtype_name(tensor.dtype), limits, strings, true);
    AddLimited(nodes, 10U, limits.max_json_nodes, "json.nodes");
    AddLimited(nodes, tensor.shape.size(), limits.max_json_nodes, "json.nodes");
    physical.push_back(&tensor);
  }
  if (!metadata.empty()) {
    CountText(kMetadataKey, limits, strings, true);
    AddLimited(nodes, 2U, limits.max_json_nodes, "json.nodes");
    for (const auto& [key, value] : metadata) {
      CountText(key, limits, strings, true);
      CountText(value, limits, strings, true);
      AddLimited(nodes, 2U, limits.max_json_nodes, "json.nodes");
    }
  }
  std::sort(physical.begin(), physical.end(),
            [](const auto* left, const auto* right) {
              const auto left_width = safetensors_dtype_size(left->dtype);
              const auto right_width = safetensors_dtype_size(right->dtype);
              return left_width != right_width ? left_width > right_width
                                               : left->name < right->name;
            });
  Json root = Json::object();
  if (!metadata.empty()) root[std::string(kMetadataKey)] = metadata;
  std::uint64_t offset = 0U;
  for (const auto* tensor : physical) {
    const auto end = offset + tensor->data.size();
    root[tensor->name] = Json{{"dtype", safetensors_dtype_name(tensor->dtype)},
                              {"shape", tensor->shape},
                              {"data_offsets", Json::array({offset, end})}};
    offset = end;
  }
  auto header = root.dump(-1, ' ', false, Json::error_handler_t::strict);
  const auto padding = (8U - header.size() % 8U) % 8U;
  if (header.size() > limits.max_header_bytes ||
      padding > limits.max_header_bytes - header.size())
    Resource("header.bytes", limits.max_header_bytes);
  header.append(padding, ' ');
  std::uint64_t file_size = 8U;
  AddLimited(file_size, header.size(), limits.max_file_bytes, "file.bytes");
  AddLimited(file_size, payload_bytes, limits.max_file_bytes, "file.bytes");
  std::vector<std::byte> encoded;
  encoded.reserve(static_cast<std::size_t>(file_size));
  const auto header_size = static_cast<std::uint64_t>(header.size());
  for (unsigned index = 0U; index < 8U; ++index) {
    encoded.push_back(static_cast<std::byte>(header_size >> (index * 8U)));
  }
  const auto* header_data = reinterpret_cast<const std::byte*>(header.data());
  encoded.insert(encoded.end(), header_data, header_data + header.size());
  for (const auto* tensor : physical) {
    if (!tensor->data.empty()) {
      encoded.insert(encoded.end(), tensor->data.begin(), tensor->data.end());
    }
  }
  return encoded;
}

}  // namespace

std::string_view safetensors_dtype_name(SafeTensorDtype dtype) noexcept {
  switch (dtype) {
    case SafeTensorDtype::boolean:
      return "BOOL";
    case SafeTensorDtype::u8:
      return "U8";
    case SafeTensorDtype::i8:
      return "I8";
    case SafeTensorDtype::u16:
      return "U16";
    case SafeTensorDtype::i16:
      return "I16";
    case SafeTensorDtype::u32:
      return "U32";
    case SafeTensorDtype::i32:
      return "I32";
    case SafeTensorDtype::u64:
      return "U64";
    case SafeTensorDtype::i64:
      return "I64";
    case SafeTensorDtype::f16:
      return "F16";
    case SafeTensorDtype::bf16:
      return "BF16";
    case SafeTensorDtype::f32:
      return "F32";
    case SafeTensorDtype::f64:
      return "F64";
  }
  return "unknown";
}

std::size_t safetensors_dtype_size(SafeTensorDtype dtype) noexcept {
  switch (dtype) {
    case SafeTensorDtype::boolean:
    case SafeTensorDtype::u8:
    case SafeTensorDtype::i8:
      return 1U;
    case SafeTensorDtype::u16:
    case SafeTensorDtype::i16:
    case SafeTensorDtype::f16:
    case SafeTensorDtype::bf16:
      return 2U;
    case SafeTensorDtype::u32:
    case SafeTensorDtype::i32:
    case SafeTensorDtype::f32:
      return 4U;
    case SafeTensorDtype::u64:
    case SafeTensorDtype::i64:
    case SafeTensorDtype::f64:
      return 8U;
  }
  return 0U;
}

SafeTensorDtype safetensors_dtype_from_name(std::string_view name) {
  for (const auto dtype : kDtypes) {
    if (safetensors_dtype_name(dtype) == name) return dtype;
  }
  Unsupported("dtype", name);
}

SafeTensorsView::SafeTensorsView(std::span<const std::byte> bytes,
                                 SafeTensorsLimits limits)
    : bytes_(bytes) {
  try {
    directory_ = ParseDirectory(bytes_, limits);
  } catch (const Error&) {
    throw;
  } catch (const nlohmann::json::exception& error) {
    Malformed("header.json", "JSON parser error " + std::to_string(error.id));
  } catch (const std::bad_alloc&) {
    BadAllocation();
  } catch (const std::length_error&) {
    BadAllocation();
  }
}

const SafeTensorsDirectory& SafeTensorsView::directory() const noexcept {
  return directory_;
}

std::span<const std::byte> SafeTensorsView::tensor_bytes(
    std::string_view name) const {
  const auto found = std::lower_bound(
      directory_.tensors.begin(), directory_.tensors.end(), name,
      [](const SafeTensorInfo& tensor, std::string_view value) {
        return tensor.name < value;
      });
  if (found == directory_.tensors.end() || found->name != name) {
    Invalid("tensor.lookup", "tensor name is absent");
  }
  return bytes_.subspan(
      static_cast<std::size_t>(directory_.payload_offset + found->data_begin),
      static_cast<std::size_t>(found->data_end - found->data_begin));
}

ArtifactDigest SafeTensorsView::digest() const {
  try {
    return {.algorithm = ArtifactDigestAlgorithm::sha256,
            .hex = sha256_hex(bytes_)};
  } catch (const std::bad_alloc&) {
    BadAllocation();
  } catch (const std::length_error&) {
    BadAllocation();
  }
}

std::vector<std::byte> serialize_safetensors(
    std::span<const SafeTensorSource> tensors,
    const std::map<std::string, std::string>& metadata,
    SafeTensorsLimits limits) {
  try {
    return Encode(tensors, metadata, limits);
  } catch (const Error&) {
    throw;
  } catch (const nlohmann::json::exception& error) {
    Invalid("header.json", "JSON encoder error " + std::to_string(error.id));
  } catch (const std::bad_alloc&) {
    BadAllocation();
  } catch (const std::length_error&) {
    BadAllocation();
  }
}

}  // namespace jyppx::interop
