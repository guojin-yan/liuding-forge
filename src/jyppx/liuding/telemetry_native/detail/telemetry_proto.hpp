// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_PRIVATE_TELEMETRY_PROTO_HPP
#define JYPPX_LIUDING_PRIVATE_TELEMETRY_PROTO_HPP

#include "nanopb_prefix.h"

#include <jyppx/liuding/foundation/error.hpp>

#include <cstddef>
#include <pb_decode.h>
#include <pb_encode.h>
#include <string>
#include <string_view>

namespace jyppx::liuding::experiment::detail::telemetry_proto {

// These borrowed inputs must outlive both official sizing and encoding passes.
// The callbacks only use generated field iterators and mature runtime APIs;
// protobuf field numbers, keys, lengths and scalar bytes are never hand-coded.
struct Bytes {
  std::string_view value;
};

struct Message {
  const pb_msgdesc_t* descriptor;
  const void* value;
};

inline bool encode_bytes(pb_ostream_t* stream, const pb_field_t* field,
                         void* const* argument) noexcept {
  const auto& bytes = *static_cast<const Bytes*>(*argument);
  return pb_encode_tag_for_field(stream, field) &&
         pb_encode_string(
             stream, reinterpret_cast<const pb_byte_t*>(bytes.value.data()),
             bytes.value.size());
}

inline bool encode_message(pb_ostream_t* stream, const pb_field_t* field,
                           void* const* argument) noexcept {
  const auto& message = *static_cast<const Message*>(*argument);
  return pb_encode_tag_for_field(stream, field) &&
         pb_encode_submessage(stream, message.descriptor, message.value);
}

inline void bind_bytes(pb_callback_t& field, const Bytes& bytes) noexcept {
  field.funcs.encode = &encode_bytes;
  field.arg = const_cast<Bytes*>(&bytes);
}

inline void bind_message(pb_callback_t& field,
                         const Message& message) noexcept {
  field.funcs.encode = &encode_message;
  field.arg = const_cast<Message*>(&message);
}

inline std::string encode(const pb_msgdesc_t* descriptor, const void* value,
                          std::size_t maximum) {
  std::size_t size = 0U;
  if (!pb_get_encoded_size(&size, descriptor, value))
    throw InternalError("Native telemetry schema sizing failed");
  if (size > maximum)
    throw ResourceExhausted("Native telemetry protobuf capacity exceeded");
  std::string bytes(size, '\0');
  auto stream = pb_ostream_from_buffer(
      reinterpret_cast<pb_byte_t*>(bytes.data()), bytes.size());
  if (!pb_encode(&stream, descriptor, value) || stream.bytes_written != size)
    throw InternalError("Native telemetry schema encoding failed");
  return bytes;
}

}  // namespace jyppx::liuding::experiment::detail::telemetry_proto
#endif
