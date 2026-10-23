// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data/batch_schema.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <iomanip>
#include <sstream>
#include <string_view>
#include <utility>

namespace jyppx::liuding::data {
namespace {

constexpr char kComponent[] = "data.batch_schema";
constexpr std::size_t kMaxSchemaIdBytes = 4096;
constexpr std::string_view kCanonicalMagic = "JYPPX_BATCH_SCHEMA_V1";

void append_u32(std::string& output, std::uint32_t value) {
  for (unsigned int shift = 0; shift < 32; shift += 8) {
    output.push_back(
        static_cast<char>(static_cast<std::uint8_t>(value >> shift)));
  }
}

void append_u64(std::string& output, std::uint64_t value) {
  for (unsigned int shift = 0; shift < 64; shift += 8) {
    output.push_back(
        static_cast<char>(static_cast<std::uint8_t>(value >> shift)));
  }
}

void append_string(std::string& output, std::string_view value) {
  append_u64(output, static_cast<std::uint64_t>(value.size()));
  output.append(value);
}

void validate_batch_identity(const BatchSchema& schema) {
  if (schema.id.empty() || schema.version == 0) {
    throw InvalidArgument(
        "Batch schema identity and version are required",
        ErrorDetails{
            .component = kComponent,
            .operation = "validate",
            .expected = "non-empty ID and positive version",
            .actual = schema.id.empty() ? "empty ID" : "version zero",
            .hint = "Assign a stable ID and positive schema version."});
  }
  if (schema.id.size() > kMaxSchemaIdBytes) {
    throw ResourceExhausted(
        "Batch schema identity exceeds its bound",
        ErrorDetails{.component = kComponent,
                     .operation = "validate.id",
                     .expected = "at most 4096 bytes",
                     .actual = std::to_string(schema.id.size()) + " bytes",
                     .hint = "Use a compact stable schema identity."});
  }
}

}  // namespace

BatchSchema BatchSchema::any(std::string id, std::uint32_t version) {
  return {.id = std::move(id),
          .version = version,
          .value = ValueSchema::any("batch")};
}

BatchSchema BatchSchema::tensor(std::string id, TensorSchema constraints,
                                std::uint32_t version,
                                std::string semantic_tag) {
  return {.id = std::move(id),
          .version = version,
          .value = ValueSchema::tensor(std::move(constraints), "batch",
                                       std::move(semantic_tag))};
}

void validate_batch_schema(const BatchSchema& schema,
                           const ValueSchemaLimits& limits) {
  validate_batch_identity(schema);
  validate_value_schema(schema.value, "batch", limits);
}

void validate_batch_value(const c10::IValue& value, const BatchSchema& schema,
                          const ValueSchemaLimits& limits) {
  validate_batch_identity(schema);
  validate_value(value, schema.value, "batch", limits);
}

std::string canonical_batch_schema(const BatchSchema& schema,
                                   const ValueSchemaLimits& limits) {
  validate_batch_schema(schema, limits);
  std::string output;
  append_string(output, kCanonicalMagic);
  append_string(output, schema.id);
  append_u32(output, schema.version);
  append_string(output, canonical_value_schema(schema.value, limits));
  return output;
}

std::string batch_schema_fingerprint(const BatchSchema& schema,
                                     const ValueSchemaLimits& limits) {
  const auto canonical = canonical_batch_schema(schema, limits);
  std::uint64_t hash = 14695981039346656037ULL;
  for (const auto byte : canonical) {
    hash ^= static_cast<std::uint8_t>(byte);
    hash *= 1099511628211ULL;
  }
  std::ostringstream output;
  output << std::hex << std::setfill('0') << std::setw(16) << hash;
  return output.str();
}

}  // namespace jyppx::liuding::data
