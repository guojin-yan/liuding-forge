// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_INTEROP_SAFETENSORS_HPP
#define JYPPX_INTEROP_SAFETENSORS_HPP

/**
 * @file safetensors.hpp
 * @brief Encodes and inspects bounded, storage-neutral safetensors bytes.
 */

#include <jyppx/foundation/artifact.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace jyppx::interop {

/**
 * @brief Supported byte-addressable safetensors scalar encodings.
 *
 * Values are stable JYPPX API identities; the file uses official string tokens.
 * Floating-point payloads are opaque IEEE/BF16 bits, including NaN and
 * infinity. Sub-byte, FP8, complex, and future encodings are explicitly
 * unsupported.
 */
enum class SafeTensorDtype : std::uint8_t {
  boolean =
      1,      ///< BOOL, one byte per element, preserved without normalization.
  u8 = 2,     ///< U8, unsigned 8-bit integer.
  i8 = 3,     ///< I8, signed 8-bit integer.
  u16 = 4,    ///< U16, unsigned 16-bit integer.
  i16 = 5,    ///< I16, signed 16-bit integer.
  u32 = 6,    ///< U32, unsigned 32-bit integer.
  i32 = 7,    ///< I32, signed 32-bit integer.
  u64 = 8,    ///< U64, unsigned 64-bit integer.
  i64 = 9,    ///< I64, signed 64-bit integer.
  f16 = 10,   ///< F16, IEEE binary16.
  bf16 = 11,  ///< BF16, brain floating-point with 8 exponent bits.
  f32 = 12,   ///< F32, IEEE binary32.
  f64 = 13,   ///< F64, IEEE binary64.
};

/** @brief Declares the byte order of the caller's packed tensor payload. */
enum class SafeTensorByteOrder : std::uint8_t {
  little = 1,  ///< Canonical little-endian bytes required by safetensors.
  big = 2,     ///< Explicitly rejected; this codec does not convert values.
};

/**
 * @brief Returns the official safetensors token for a scalar encoding.
 * @param dtype Stable encoding identity to inspect.
 * @return Official token, or `unknown` for an unrecognized enum value.
 */
std::string_view safetensors_dtype_name(SafeTensorDtype dtype) noexcept;

/**
 * @brief Returns the number of bytes occupied by one supported scalar.
 * @param dtype Stable encoding identity to inspect.
 * @return One, two, four, or eight bytes; zero for an unknown enum value.
 */
std::size_t safetensors_dtype_size(SafeTensorDtype dtype) noexcept;

/**
 * @brief Converts an official dtype token into a supported stable identity.
 * @param name Exact case-sensitive safetensors token such as `F32`.
 * @return Supported scalar encoding.
 * @throws CompatibilityError If the token is unknown or unsupported.
 */
SafeTensorDtype safetensors_dtype_from_name(std::string_view name);

/**
 * @brief Explicit resource bounds for metadata, shapes, and artifact bytes.
 *
 * These are encoded-size and object-count limits, not a process RSS guarantee.
 * The borrowing reader retains no tensor copies. The writer allocates one full
 * encoded file in addition to caller-owned payloads and bounded JSON metadata.
 */
struct SafeTensorsLimits {
  std::uint64_t max_file_bytes = 1ULL << 34U;  ///< Includes prefix and header.
  std::uint64_t max_payload_bytes = 1ULL << 34U;  ///< Aggregate tensor bytes.
  std::uint64_t max_tensor_bytes = 1ULL << 32U;   ///< One tensor's raw bytes.
  std::uint64_t max_tensor_elements = 1ULL
                                      << 32U;  ///< Includes scalar count 1.
  std::uint64_t max_dimension = 1ULL << 32U;   ///< Largest single dimension.
  std::size_t max_header_bytes = 16U * 1024U * 1024U;  ///< At most 100,000,000.
  std::size_t max_tensors = 65536U;  ///< Zero tensors are a valid file.
  std::size_t max_rank = 32U;  ///< Maximum dimensions; scalar rank is zero.
  std::size_t max_metadata_entries = 4096U;  ///< String pairs in __metadata__.
  std::size_t max_string_bytes = 1024U * 1024U;  ///< One decoded key or value.
  std::size_t max_total_string_bytes = 16U * 1024U * 1024U;  ///< All JSON text.
  std::size_t max_json_nodes =
      4194304U;                     ///< Objects, arrays, keys, and values.
  std::size_t max_json_depth = 8U;  ///< Nested JSON container depth.
};

/** @brief Describes one C-order tensor without depending on a tensor runtime.
 */
struct SafeTensorInfo {
  std::string name;  ///< UTF-8 logical name; never interpreted as a path.
  SafeTensorDtype dtype = SafeTensorDtype::u8;  ///< Official scalar encoding.
  std::vector<std::uint64_t> shape;  ///< Dimensions; empty means one scalar.
  std::uint64_t data_begin = 0;  ///< Inclusive byte offset within the payload.
  std::uint64_t data_end = 0;    ///< Exclusive byte offset within the payload.
};

/** @brief Validated tensor directory and free-form string metadata. */
struct SafeTensorsDirectory {
  std::map<std::string, std::string> metadata;  ///< Sorted UTF-8 string pairs.
  std::vector<SafeTensorInfo> tensors;  ///< Sorted by unique tensor name.
  std::uint64_t payload_offset = 0;  ///< Absolute file byte offset of payload.
  std::uint64_t payload_size = 0;  ///< Exactly covered contiguous payload size.
};

/**
 * @brief Borrows a fully validated safetensors artifact without tensor copies.
 *
 * The caller must keep the supplied bytes alive and unchanged for the complete
 * view and returned-span lifetimes. Construction validates the complete header,
 * all tensor extents, and full payload coverage before exposing any tensor.
 * Payload values are opaque and the format has no embedded checksum; use the
 * complete-artifact digest with an independently trusted ArtifactRef.
 * Concurrent const operations are safe while the underlying bytes stay fixed.
 */
class SafeTensorsView final {
 public:
  /**
   * @brief Validates and borrows one complete safetensors file byte range.
   * @param bytes Stable complete file bytes including the length prefix.
   * @param limits Positive parser and resource bounds.
   * @throws InvalidArgument If limits are invalid or cannot fit this platform.
   * @throws SerializationError If JSON, shapes, or payload extents are invalid.
   * @throws CompatibilityError If a tensor dtype is unsupported.
   * @throws ResourceExhausted If a configured bound or memory limit is
   * exceeded.
   */
  explicit SafeTensorsView(std::span<const std::byte> bytes,
                           SafeTensorsLimits limits = {});

  /**
   * @brief Returns immutable, validated header and tensor-directory metadata.
   * @return Metadata owned by this view and valid for the view lifetime.
   */
  const SafeTensorsDirectory& directory() const noexcept;

  /**
   * @brief Retrieves the raw bytes of one declared tensor by logical name.
   * @param name Exact UTF-8 tensor identity; no path interpretation occurs.
   * @return Borrowed, packed little-endian bytes in C element order.
   * @throws InvalidArgument If the name is absent from the validated directory.
   */
  std::span<const std::byte> tensor_bytes(std::string_view name) const;

  /**
   * @brief Hashes the complete original file using foundation SHA-256.
   * @return Content identity for prefix, original header padding, and payload.
   * @throws ResourceExhausted If digest allocation fails.
   */
  ArtifactDigest digest() const;

 private:
  std::span<const std::byte>
      bytes_;                       ///< Stable caller-owned complete artifact.
  SafeTensorsDirectory directory_;  ///< Owned, fully validated metadata.
};

/**
 * @brief Supplies one packed tensor for deterministic safetensors encoding.
 *
 * The caller owns every byte and must quiesce mutation until encoding returns.
 * Strides, aliases, devices, and framework tensor objects belong to adapters.
 */
struct SafeTensorSource {
  std::string
      name;  ///< Unique UTF-8 name other than the reserved __metadata__.
  SafeTensorDtype dtype = SafeTensorDtype::u8;  ///< Supported scalar encoding.
  std::vector<std::uint64_t>
      shape;  ///< C-order shape; zero dimensions allowed.
  std::span<const std::byte>
      data;  ///< Exact product(shape) * scalar byte width.
  SafeTensorByteOrder byte_order =
      SafeTensorByteOrder::little;  ///< Explicit LE.
};

/**
 * @brief Encodes tensors and metadata into one official safetensors byte range.
 * @param tensors Unique tensor sources; the input order does not affect output.
 * @param metadata UTF-8 string-to-string metadata stored under __metadata__.
 * @param limits Positive bounds applied before output allocation.
 * @return Complete file with an LE length prefix, JSON, and contiguous payload.
 * @throws InvalidArgument If sources, byte counts, names, or options are
 * invalid.
 * @throws CompatibilityError If a dtype or non-little byte order is
 * unsupported.
 * @throws ResourceExhausted If encoded sizes, metadata, or memory exceed
 * bounds.
 *
 * Tensors are laid out by descending scalar width and then by name. The JSON
 * keys are sorted and its header is space-padded to eight bytes. No conversion,
 * compression, sharding, filesystem I/O, fsync, or atomic publication occurs.
 */
std::vector<std::byte> serialize_safetensors(
    std::span<const SafeTensorSource> tensors,
    const std::map<std::string, std::string>& metadata = {},
    SafeTensorsLimits limits = {});

}  // namespace jyppx::interop

#endif  // JYPPX_INTEROP_SAFETENSORS_HPP
