// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/interop/safetensors.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(JYPPX_SAFETENSORS_PYTHON_INTEROP)
#include <cstdlib>
#include <filesystem>
#include <fstream>
#endif

namespace {

using jyppx::liuding::interop::SafeTensorByteOrder;
using jyppx::liuding::interop::SafeTensorDtype;
using jyppx::liuding::interop::SafeTensorsLimits;
using jyppx::liuding::interop::SafeTensorSource;
using jyppx::liuding::interop::SafeTensorsView;
using jyppx::liuding::interop::serialize_safetensors;

constexpr std::array<SafeTensorDtype, 13> kDtypes{
    SafeTensorDtype::boolean, SafeTensorDtype::u8,   SafeTensorDtype::i8,
    SafeTensorDtype::u16,     SafeTensorDtype::i16,  SafeTensorDtype::u32,
    SafeTensorDtype::i32,     SafeTensorDtype::u64,  SafeTensorDtype::i64,
    SafeTensorDtype::f16,     SafeTensorDtype::bf16, SafeTensorDtype::f32,
    SafeTensorDtype::f64};

/** @brief Builds deliberately noncanonical wire inputs independently of writer.
 */
std::vector<std::byte> Wire(std::string_view header,
                            std::size_t payload_size = 0U) {
  std::vector<std::byte> bytes;
  const auto size = static_cast<std::uint64_t>(header.size());
  for (unsigned index = 0; index < 8U; ++index) {
    bytes.push_back(static_cast<std::byte>(size >> (index * 8U)));
  }
  const auto* data = reinterpret_cast<const std::byte*>(header.data());
  bytes.insert(bytes.end(), data, data + header.size());
  bytes.resize(bytes.size() + payload_size);
  return bytes;
}

/** @brief Replaces the framing length without changing actual file bytes. */
void HeaderLength(std::vector<std::byte>& bytes, std::uint64_t length) {
  for (unsigned index = 0; index < 8U; ++index) {
    bytes[index] = static_cast<std::byte>(length >> (index * 8U));
  }
}

/** @brief Owns deterministic payloads shared with the official Python fixture.
 */
struct Matrix {
  std::vector<std::vector<std::byte>> payloads;
  std::vector<SafeTensorSource> tensors;

  Matrix() {
    payloads.reserve(kDtypes.size() + 2U);
    tensors.reserve(kDtypes.size() + 2U);
    for (std::size_t index = 0; index < kDtypes.size(); ++index) {
      const auto dtype = kDtypes[index];
      auto& payload = payloads.emplace_back(
          4U * jyppx::liuding::interop::safetensors_dtype_size(dtype));
      for (std::size_t byte = 0; byte < payload.size(); ++byte) {
        payload[byte] =
            static_cast<std::byte>(dtype == SafeTensorDtype::boolean
                                       ? byte % 2U
                                       : (index * 19U + byte * 7U) % 256U);
      }
      tensors.push_back(
          {.name = "tensor." +
                   std::string(
                       jyppx::liuding::interop::safetensors_dtype_name(dtype)),
           .dtype = dtype,
           .shape = {2U, 2U},
           .data = payload,
           .byte_order = SafeTensorByteOrder::little});
    }
    auto& scalar = payloads.emplace_back(std::vector<std::byte>{
        std::byte{0x00}, std::byte{0x00}, std::byte{0x80}, std::byte{0x3f}});
    tensors.push_back({.name = "scalar",
                       .dtype = SafeTensorDtype::f32,
                       .shape = {},
                       .data = scalar,
                       .byte_order = SafeTensorByteOrder::little});
    tensors.push_back({.name = "empty",
                       .dtype = SafeTensorDtype::bf16,
                       .shape = {2U, 0U, 3U},
                       .data = {},
                       .byte_order = SafeTensorByteOrder::little});
  }
};

TEST(SafeTensorsTest, RoundTripsEverySupportedDtypeScalarEmptyAndUtf8Metadata) {
  Matrix matrix;
  const std::map<std::string, std::string> metadata{
      {"producer", "jyppx"}, {"description", "\xe4\xb8\xad\xe6\x96\x87"}};
  const auto encoded = serialize_safetensors(matrix.tensors, metadata);
  const SafeTensorsView view(encoded);
  EXPECT_EQ(view.directory().metadata, metadata);
  ASSERT_EQ(view.directory().tensors.size(), matrix.tensors.size());
  EXPECT_EQ(view.directory().payload_offset % 8U, 0U);
  for (const auto& tensor : matrix.tensors) {
    SCOPED_TRACE(tensor.name);
    const auto found = std::find_if(
        view.directory().tensors.begin(), view.directory().tensors.end(),
        [&](const auto& item) { return item.name == tensor.name; });
    ASSERT_NE(found, view.directory().tensors.end());
    EXPECT_EQ(found->dtype, tensor.dtype);
    EXPECT_EQ(found->shape, tensor.shape);
    const auto actual = view.tensor_bytes(tensor.name);
    EXPECT_EQ(std::vector<std::byte>(actual.begin(), actual.end()),
              std::vector<std::byte>(tensor.data.begin(), tensor.data.end()));
    EXPECT_EQ(actual.data(), encoded.data() + view.directory().payload_offset +
                                 found->data_begin);
  }
  EXPECT_EQ(view.digest().hex, jyppx::liuding::sha256_hex(encoded));
  EXPECT_THROW(view.tensor_bytes("unknown"), jyppx::liuding::InvalidArgument);
  std::reverse(matrix.tensors.begin(), matrix.tensors.end());
  EXPECT_EQ(serialize_safetensors(matrix.tensors, metadata), encoded);
}

TEST(SafeTensorsTest, DtypeTokensAreExplicitAndUnknownFutureTypesAreRejected) {
  for (const auto dtype : kDtypes) {
    const auto name = jyppx::liuding::interop::safetensors_dtype_name(dtype);
    EXPECT_EQ(jyppx::liuding::interop::safetensors_dtype_from_name(name),
              dtype);
    EXPECT_GT(jyppx::liuding::interop::safetensors_dtype_size(dtype), 0U);
  }
  const auto unknown = static_cast<SafeTensorDtype>(255);
  EXPECT_EQ(jyppx::liuding::interop::safetensors_dtype_size(unknown), 0U);
  EXPECT_EQ(jyppx::liuding::interop::safetensors_dtype_name(unknown),
            "unknown");
  for (const auto name :
       {"F8_E4M3", "F8_E5M2", "F4", "C64", "F128", "f32", ""}) {
    EXPECT_THROW(jyppx::liuding::interop::safetensors_dtype_from_name(name),
                 jyppx::liuding::CompatibilityError);
    const auto bytes = Wire("{\"x\":{\"dtype\":\"" + std::string(name) +
                            "\",\"shape\":[],\"data_offsets\":[0,0]}}");
    EXPECT_THROW({ SafeTensorsView invalid(bytes); },
                 jyppx::liuding::CompatibilityError);
  }
}

TEST(SafeTensorsTest,
     ReadsIndependentWireOrderUnalignedDataAndEmptyBoundaries) {
  // Directory order, dtype alignment, and physical order are independent.
  // Empty intervals may share a boundary with another empty or nonempty one.
  const auto bytes = Wire(
      R"({"z":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"F64","shape":[],"data_offsets":[1,9]},"a":{"dtype":"U8","shape":[0],"data_offsets":[1,1]},"c":{"dtype":"I8","shape":[0],"data_offsets":[1,1]}})",
      9U);
  const SafeTensorsView view(bytes);
  EXPECT_EQ(view.directory().tensors.front().name, "a");
  EXPECT_EQ(view.tensor_bytes("b").size(), 8U);
  EXPECT_TRUE(view.tensor_bytes("a").empty());
  const auto empty_file = Wire("{}");
  EXPECT_NO_THROW({ SafeTensorsView empty(empty_file); });
  const auto metadata_only = Wire(R"({"__metadata__":{"k":"v"}})");
  EXPECT_NO_THROW({ SafeTensorsView empty(metadata_only); });
  EXPECT_THROW({ SafeTensorsView invalid(Wire("{}", 1U)); },
               jyppx::liuding::SerializationError);
  EXPECT_NO_THROW(serialize_safetensors({}));
}

TEST(SafeTensorsTest,
     RejectsDuplicateKeysAtEveryObjectLevelIncludingEscapedNames) {
  for (
      const auto header :
      {R"({"x":{"dtype":"U8","shape":[0],"data_offsets":[0,0]},"x":{"dtype":"U8","shape":[0],"data_offsets":[0,0]}})",
       R"({"x":{"dtype":"U8","dtype":"U8","shape":[0],"data_offsets":[0,0]}})",
       R"({"x":{"dtype":"U8","shape":[0],"data_offsets":[0,0]},"\u0078":{"dtype":"U8","shape":[0],"data_offsets":[0,0]}})",
       R"({"__metadata__":{"producer":"a","producer":"b"}})",
       R"({"__metadata__":{},"__metadata__":{}})"}) {
    const auto bytes = Wire(header);
    EXPECT_THROW({ SafeTensorsView invalid(bytes); },
                 jyppx::liuding::SerializationError);
  }
}

TEST(SafeTensorsTest,
     RejectsWrongJsonTypesUnknownTensorFieldsAndNonStringMetadata) {
  for (
      const auto header :
      {R"([])", R"({"__metadata__":[]})", R"({"__metadata__":{"x":1}})",
       R"({"x":{"dtype":"U8","shape":[0],"data_offsets":[0,0],"extra":0}})",
       R"({"x":{"dtype":"U8","data_offsets":[0,0]}})",
       R"({"x":{"dtype":1,"shape":[0],"data_offsets":[0,0]}})",
       R"({"x":{"dtype":"U8","shape":{},"data_offsets":[0,0]}})",
       R"({"x":{"dtype":"U8","shape":[-1],"data_offsets":[0,0]}})",
       R"({"x":{"dtype":"U8","shape":[true],"data_offsets":[0,0]}})",
       R"({"x":{"dtype":"U8","shape":[1.0],"data_offsets":[0,1]}})",
       R"({"x":{"dtype":"U8","shape":[0],"data_offsets":[0]}})",
       R"({"x":{"dtype":"U8","shape":[0],"data_offsets":[0,0,0]}})",
       R"({"x":{"dtype":"U8","shape":[0],"data_offsets":[-1,0]}})",
       R"({"x":{"dtype":"U8","shape":[0],"data_offsets":[0,0.0]}})",
       R"({"x":{"dtype":"U8","shape":[0],"data_offsets":[0,18446744073709551616]}})",
       R"({"x":{"dtype":"U8","shape":[0],"data_offsets":[0,0]}} garbage)"}) {
    const auto bytes = Wire(header);
    EXPECT_THROW({ SafeTensorsView invalid(bytes); },
                 jyppx::liuding::SerializationError);
  }
}

TEST(SafeTensorsTest,
     RejectsOverlapHolesTruncationAndUnindexedTrailingPayload) {
  for (
      const auto header :
      {R"({"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U8","shape":[2],"data_offsets":[1,3]}})",
       R"({"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"U8","shape":[1],"data_offsets":[2,3]}})",
       R"({"a":{"dtype":"U8","shape":[3],"data_offsets":[1,4]}})",
       R"({"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}})",
       R"({"a":{"dtype":"U8","shape":[3],"data_offsets":[3,0]}})",
       R"({"a":{"dtype":"F64","shape":[1],"data_offsets":[0,3]}})",
       R"({"a":{"dtype":"U8","shape":[3],"data_offsets":[0,4]}})",
       R"({"a":{"dtype":"U8","shape":[3],"data_offsets":[0,3]},"e":{"dtype":"U8","shape":[0],"data_offsets":[1,1]}})"}) {
    const auto bytes = Wire(header, 3U);
    EXPECT_THROW({ SafeTensorsView invalid(bytes); },
                 jyppx::liuding::SerializationError);
  }
  for (std::size_t size = 0; size < 8U; ++size) {
    const std::vector<std::byte> short_prefix(size);
    EXPECT_THROW({ SafeTensorsView invalid(short_prefix); },
                 jyppx::liuding::SerializationError);
  }
  auto bytes = Wire("{}");
  HeaderLength(bytes, 4U);
  EXPECT_THROW({ SafeTensorsView invalid(bytes); },
               jyppx::liuding::SerializationError);
  HeaderLength(bytes, (std::numeric_limits<std::uint64_t>::max)());
  EXPECT_THROW({ SafeTensorsView invalid(bytes); },
               jyppx::liuding::ResourceExhausted);
}

TEST(SafeTensorsTest, RejectsBadUtf8InvalidEscapesAndNonSpaceHeaderPadding) {
  for (const auto suffix :
       {std::string("\xc0\xaf"), std::string("\xed\xa0\x80"),
        std::string("\xf4\x90\x80\x80"), std::string("\x80")}) {
    const auto bytes = Wire("{\"__metadata__\":{\"x\":\"" + suffix + "\"}}");
    EXPECT_THROW({ SafeTensorsView invalid(bytes); },
                 jyppx::liuding::SerializationError);
  }
  for (const auto header : {" {}", "\xef\xbb\xbf{}", "{}\n", "{}\t",
                            R"({"__metadata__":{"x":"\uD800"}})"}) {
    const auto bytes = Wire(header);
    EXPECT_THROW({ SafeTensorsView invalid(bytes); },
                 jyppx::liuding::SerializationError);
  }
  const auto space_padded = Wire("{}   ");
  EXPECT_NO_THROW({ SafeTensorsView valid(space_padded); });
}

TEST(SafeTensorsTest, BoundsHeaderNodesDepthStringsTensorCountRankAndBytes) {
  const auto valid = Wire(
      R"({"x":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"__metadata__":{"a":"b","c":"d"}})",
      2U);
  const auto constrained = [&](const SafeTensorsLimits& limits) {
    EXPECT_THROW({ SafeTensorsView invalid(valid, limits); },
                 jyppx::liuding::ResourceExhausted);
  };
  SafeTensorsLimits limits;
  limits.max_file_bytes = valid.size() - 1U;
  constrained(limits);
  limits = {};
  limits.max_header_bytes = 2U;
  constrained(limits);
  limits = {};
  limits.max_payload_bytes = 1U;
  constrained(limits);
  limits = {};
  limits.max_tensor_bytes = 1U;
  constrained(limits);
  limits = {};
  limits.max_tensor_elements = 1U;
  constrained(limits);
  limits = {};
  limits.max_dimension = 1U;
  constrained(limits);
  limits = {};
  limits.max_metadata_entries = 1U;
  constrained(limits);
  limits = {};
  limits.max_string_bytes = 2U;
  constrained(limits);
  limits = {};
  limits.max_total_string_bytes = 8U;
  constrained(limits);
  limits = {};
  limits.max_json_nodes = 2U;
  constrained(limits);
  limits = {};
  limits.max_json_depth = 2U;
  constrained(limits);
  limits = {};
  limits.max_tensors = 1U;
  const auto two = Wire(
      R"({"a":{"dtype":"U8","shape":[0],"data_offsets":[0,0]},"b":{"dtype":"U8","shape":[0],"data_offsets":[0,0]}})");
  EXPECT_THROW({ SafeTensorsView invalid(two, limits); },
               jyppx::liuding::ResourceExhausted);
  limits = {};
  limits.max_rank = 1U;
  const auto rank =
      Wire(R"({"a":{"dtype":"U8","shape":[0,0],"data_offsets":[0,0]}})");
  EXPECT_THROW({ SafeTensorsView invalid(rank, limits); },
               jyppx::liuding::ResourceExhausted);
  const auto nested = Wire(R"({"__metadata__":{"x":[[[[]]]]}})");
  limits.max_json_depth = 3U;
  EXPECT_THROW({ SafeTensorsView invalid(nested, limits); },
               jyppx::liuding::ResourceExhausted);
}

TEST(SafeTensorsTest, DetectsShapeAndDtypeByteMultiplicationOverflow) {
  SafeTensorsLimits limits;
  limits.max_dimension = (std::numeric_limits<std::uint64_t>::max)();
  limits.max_tensor_elements = (std::numeric_limits<std::uint64_t>::max)();
  limits.max_tensor_bytes = (std::numeric_limits<std::uint64_t>::max)();
  for (
      const auto header :
      {R"({"x":{"dtype":"U8","shape":[18446744073709551615,2],"data_offsets":[0,0]}})",
       R"({"x":{"dtype":"F64","shape":[18446744073709551615],"data_offsets":[0,0]}})"}) {
    const auto bytes = Wire(header);
    EXPECT_THROW({ SafeTensorsView invalid(bytes, limits); },
                 jyppx::liuding::SerializationError);
  }
}

TEST(SafeTensorsTest,
     WriterRejectsDuplicatesWrongSizesByteOrderAndInvalidLimits) {
  const std::array<std::byte, 4> payload{};
  SafeTensorSource tensor{.name = "x",
                          .dtype = SafeTensorDtype::f32,
                          .shape = {},
                          .data = payload,
                          .byte_order = SafeTensorByteOrder::little};
  std::array<SafeTensorSource, 2> duplicate{tensor, tensor};
  EXPECT_THROW(serialize_safetensors(duplicate),
               jyppx::liuding::InvalidArgument);
  auto source = std::array{tensor};
  source[0].name = "__metadata__";
  EXPECT_THROW(serialize_safetensors(source), jyppx::liuding::InvalidArgument);
  source[0] = tensor;
  source[0].shape = {2U};
  EXPECT_THROW(serialize_safetensors(source), jyppx::liuding::InvalidArgument);
  source[0] = tensor;
  source[0].byte_order = SafeTensorByteOrder::big;
  EXPECT_THROW(serialize_safetensors(source),
               jyppx::liuding::CompatibilityError);
  source[0] = tensor;
  source[0].dtype = static_cast<SafeTensorDtype>(255);
  EXPECT_THROW(serialize_safetensors(source),
               jyppx::liuding::CompatibilityError);
  source[0] = tensor;
  source[0].name = std::string("\xc0\xaf");
  EXPECT_THROW(serialize_safetensors(source), jyppx::liuding::InvalidArgument);
  source[0] = tensor;
  SafeTensorsLimits limits;
  limits.max_file_bytes = 12U;
  EXPECT_THROW(serialize_safetensors(source, {}, limits),
               jyppx::liuding::ResourceExhausted);
  limits = {};
  limits.max_header_bytes = 2U;
  EXPECT_THROW(serialize_safetensors(source, {}, limits),
               jyppx::liuding::ResourceExhausted);
  limits = {};
  limits.max_json_nodes = 1U;
  EXPECT_THROW(serialize_safetensors(source, {}, limits),
               jyppx::liuding::ResourceExhausted);
  limits = {};
  limits.max_rank = 0U;
  EXPECT_THROW(serialize_safetensors(source, {}, limits),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW({ SafeTensorsView invalid(Wire("{}"), limits); },
               jyppx::liuding::InvalidArgument);
  limits = {};
  limits.max_header_bytes = 100000001U;
  EXPECT_THROW(serialize_safetensors(source, {}, limits),
               jyppx::liuding::InvalidArgument);
}

#if defined(JYPPX_SAFETENSORS_PYTHON_INTEROP)
TEST(SafeTensorsTest, OfficialPythonBidirectionalInterchange) {
  const auto* directory = std::getenv("JYPPX_SAFETENSORS_INTEROP_DIR");
  ASSERT_NE(directory, nullptr);
  const auto root = std::filesystem::path(directory);
  std::ifstream input(root / "python.safetensors",
                      std::ios::binary | std::ios::ate);
  ASSERT_TRUE(input);
  const auto size = input.tellg();
  ASSERT_GE(size, 0);
  ASSERT_LE(size, 16U * 1024U * 1024U);
  std::vector<std::byte> bytes(static_cast<std::size_t>(size));
  input.seekg(0);
  input.read(reinterpret_cast<char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  ASSERT_TRUE(input);
  const SafeTensorsView view(bytes);
  Matrix expected;
  ASSERT_EQ(view.directory().tensors.size(), expected.tensors.size());
  for (const auto& tensor : expected.tensors) {
    const auto actual = view.tensor_bytes(tensor.name);
    EXPECT_EQ(std::vector<std::byte>(actual.begin(), actual.end()),
              std::vector<std::byte>(tensor.data.begin(), tensor.data.end()));
    const auto found = std::find_if(
        view.directory().tensors.begin(), view.directory().tensors.end(),
        [&](const auto& item) { return item.name == tensor.name; });
    ASSERT_NE(found, view.directory().tensors.end());
    EXPECT_EQ(found->dtype, tensor.dtype);
    EXPECT_EQ(found->shape, tensor.shape);
  }
  EXPECT_EQ(view.directory().metadata.at("producer"), "official-python");
  const auto encoded =
      serialize_safetensors(expected.tensors, view.directory().metadata);
  std::ofstream output(root / "cpp.safetensors",
                       std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(encoded.data()),
               static_cast<std::streamsize>(encoded.size()));
  output.close();
  ASSERT_TRUE(output);
}
#endif

}  // namespace
