// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "image_batch_detail.hpp"

#include <jyppx/liuding/data_image/batch_identity.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <string_view>
#include <vector>

namespace jyppx::liuding::data::image {
namespace {
using detail::BatchLimit;
using detail::InvalidBatch;
using Value = c10::IValue;

void CheckStop(std::stop_token stop) {
  if (stop.stop_requested())
    throw Cancelled("Image batch identity cancelled",
                    ErrorDetails{.component = "data.image.batch_identity",
                                 .operation = "batch_identity.cancel"});
}

std::string_view DtypeName(torch::ScalarType type) {
  switch (type) {
    case torch::kUInt8:
      return "uint8";
    case torch::kUInt16:
      return "uint16";
    case torch::kFloat16:
      return "float16";
    case torch::kBFloat16:
      return "bfloat16";
    case torch::kFloat32:
      return "float32";
    case torch::kFloat64:
      return "float64";
    case torch::kInt64:
      return "int64";
    case torch::kBool:
      return "bool";
    default:
      InvalidBatch("batch_identity.dtype");
  }
}

class CanonicalHash final {
 public:
  CanonicalHash(const ImageBatchLimits& limits, std::stop_token stop)
      : limits_(limits), stop_(stop) {}

  void Byte(std::uint8_t value) {
    const auto byte = static_cast<std::byte>(value);
    hash_.update(std::span(&byte, 1));
  }

  void Integer(std::uint64_t value) {
    std::array<std::byte, 8> encoded{};
    for (std::size_t index = 0; index < encoded.size(); ++index)
      encoded[index] = static_cast<std::byte>((value >> (index * 8U)) & 0xffU);
    hash_.update(encoded);
  }

  void Bytes(std::span<const std::byte> bytes) {
    Integer(bytes.size());
    hash_.update(bytes);
  }

  void String(std::string_view value) {
    Byte(0x04);
    Bytes(std::as_bytes(std::span(value.data(), value.size())));
  }

  void ValueOf(const Value& value, std::size_t depth = 0) {
    CheckStop(stop_);
    if (depth > 64 || ++nodes_ > limits_.max_metadata_nodes)
      BatchLimit("batch_identity.metadata");

    if (value.isNone()) {
      Byte(0x00);
    } else if (value.isBool()) {
      Byte(0x01);
      Byte(value.toBool() ? 1 : 0);
    } else if (value.isInt()) {
      Byte(0x02);
      Integer(std::bit_cast<std::uint64_t>(value.toInt()));
    } else if (value.isDouble()) {
      Byte(0x03);
      const double number = value.toDouble();
      if (!std::isfinite(number)) InvalidBatch("batch_identity.number");
      Integer(std::bit_cast<std::uint64_t>(number == 0 ? 0.0 : number));
    } else if (value.isString()) {
      String(value.toStringRef());
    } else if (value.isTensor()) {
      TensorValue(value.toTensor());
    } else if (value.isTuple()) {
      const auto& elements = value.toTupleRef().elements();
      Byte(0x05);
      Integer(elements.size());
      for (const auto& element : elements) ValueOf(element, depth + 1);
    } else if (value.isList()) {
      const auto elements = value.toListRef();
      Byte(0x06);
      Integer(elements.size());
      for (const auto& element : elements) ValueOf(element, depth + 1);
    } else {
      InvalidBatch("batch_identity.value_kind");
    }
  }

  std::string Finish() const { return sha256_digest_hex(hash_.finish()); }

 private:
  void TensorValue(const Tensor& tensor) {
    if (!tensor.device().is_cpu() || tensor.layout() != torch::kStrided ||
        !tensor.is_contiguous())
      InvalidBatch("batch_identity.tensor_layout");
    Byte(0x07);
    String(DtypeName(tensor.scalar_type()));
    Integer(static_cast<std::uint64_t>(tensor.dim()));
    for (const auto size : tensor.sizes()) {
      if (size < 0) InvalidBatch("batch_identity.tensor_shape");
      Integer(static_cast<std::uint64_t>(size));
    }
    const auto count = static_cast<std::uint64_t>(tensor.numel());
    const auto element_size = tensor.element_size();
    if (element_size == 0 ||
        count > std::numeric_limits<std::uint64_t>::max() / element_size)
      BatchLimit("batch_identity.tensor_bytes");
    const auto byte_count = count * element_size;
    if (byte_count > limits_.max_output_bytes)
      BatchLimit("batch_identity.tensor_bytes");
    Integer(element_size);
    Integer(byte_count);
    if (byte_count == 0) return;

    const auto* data = static_cast<const std::byte*>(tensor.const_data_ptr());
    constexpr std::size_t kChunkBytes = 1U << 20U;
    if constexpr (std::endian::native == std::endian::little) {
      for (std::uint64_t offset = 0; offset < byte_count;) {
        CheckStop(stop_);
        const auto chunk = static_cast<std::size_t>(
            std::min<std::uint64_t>(kChunkBytes, byte_count - offset));
        hash_.update(std::span(data + offset, chunk));
        offset += chunk;
      }
    } else if constexpr (std::endian::native == std::endian::big) {
      const auto elements_per_chunk =
          std::max<std::size_t>(1, kChunkBytes / element_size);
      std::vector<std::byte> encoded;
      encoded.reserve(elements_per_chunk * element_size);
      std::uint64_t offset = 0;
      while (offset < count) {
        CheckStop(stop_);
        const auto elements = static_cast<std::size_t>(
            std::min<std::uint64_t>(elements_per_chunk, count - offset));
        encoded.resize(elements * element_size);
        for (std::size_t element = 0; element < elements; ++element) {
          const auto* source = data + (offset + element) * element_size;
          auto* destination = encoded.data() + element * element_size;
          std::reverse_copy(source, source + element_size, destination);
        }
        hash_.update(encoded);
        offset += elements;
      }
    } else {
      InvalidBatch("batch_identity.endianness");
    }
  }

  const ImageBatchLimits& limits_;
  std::stop_token stop_;
  std::size_t nodes_ = 0;
  Sha256 hash_;
};

}  // namespace

std::string image_batch_content_identity(const ImageBatch& batch,
                                         const ImageBatchLimits& limits,
                                         std::stop_token stop) {
  detail::CheckImageBatch(batch, limits, stop);
  CheckStop(stop);
  const auto& first = batch.layout == ImageBatchLayout::ragged
                          ? batch.image_list.front()
                          : batch.images;
  if (!first.device().is_cpu()) InvalidBatch("batch_identity.cpu_only");

  try {
    const auto value = image_batch_value(batch, limits);
    CanonicalHash hash(limits, stop);
    constexpr std::string_view domain = "liuding.image-batch-content.v1";
    hash.Bytes(std::as_bytes(std::span(domain.data(), domain.size())));
    const auto dictionary = value.toGenericDict();
    constexpr std::array<std::string_view, 11> keys{
        "format",        "images", "items", "label_offsets",
        "labels",        "layout", "masks", "paired",
        "paired_pixels", "pixels", "valid"};
    if (dictionary.size() != keys.size())
      InvalidBatch("batch_identity.dictionary_size");
    for (const auto key : keys) {
      CheckStop(stop);
      if (!dictionary.contains(std::string(key)))
        InvalidBatch("batch_identity.dictionary_key");
      hash.String(key);
      hash.ValueOf(dictionary.at(std::string(key)));
    }
    CheckStop(stop);
    return hash.Finish();
  } catch (const std::bad_alloc&) {
    BatchLimit("batch_identity.allocate");
  } catch (const c10::OutOfMemoryError&) {
    BatchLimit("batch_identity.allocate");
  } catch (const c10::Error&) {
    throw DeviceError("Image batch identity failed");
  }
}

}  // namespace jyppx::liuding::data::image
