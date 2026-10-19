// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_DATA_BATCH_SCHEMA_HPP
#define JYPPX_DATA_BATCH_SCHEMA_HPP

/**
 * @file batch_schema.hpp
 * @brief Defines versioned schemas for consumer-visible data batches.
 */

#include <jyppx/core/value_schema.hpp>

#include <ATen/core/ivalue.h>
#include <cstdint>
#include <string>

namespace jyppx::data {

/**
 * @brief Binds a stable producer identity and version to a value schema.
 *
 * The identity describes the producer contract, while `value` describes the
 * complete consumer-visible IValue after optional memory staging. Any
 * structural or semantic change must either update `version` or use a new ID.
 */
struct BatchSchema {
  std::string id;             ///< Stable non-empty producer schema identity.
  std::uint32_t version = 1;  ///< Positive producer-controlled schema version.
  ValueSchema value;          ///< Recursive consumer-visible value contract.

  /**
   * @brief Creates an identity-bound unconstrained non-null batch schema.
   * @param id Stable producer schema identity.
   * @param version Positive producer schema version.
   * @return Batch schema accepting every non-null IValue category.
   */
  static BatchSchema any(std::string id, std::uint32_t version = 1);

  /**
   * @brief Creates an identity-bound tensor batch schema.
   * @param id Stable producer schema identity.
   * @param constraints Consumer-visible tensor constraints.
   * @param version Positive producer schema version.
   * @param semantic_tag Stable optional domain meaning for the tensor.
   * @return Batch schema accepting a matching defined tensor.
   */
  static BatchSchema tensor(std::string id, TensorSchema constraints = {},
                            std::uint32_t version = 1,
                            std::string semantic_tag = {});
};

/**
 * @brief Validates batch identity, version, and recursive value constraints.
 * @param schema Batch schema to inspect.
 * @param limits Resource limits applied to the recursive value schema.
 * @throws InvalidArgument If identity, version, or structure is invalid.
 * @throws ResourceExhausted If identity or nested schema exceeds its bound.
 */
void validate_batch_schema(const BatchSchema& schema,
                           const ValueSchemaLimits& limits = {});

/**
 * @brief Validates a consumer-visible batch before pipeline commit.
 * @param value Staged or original batch value.
 * @param schema Expected producer schema.
 * @param limits Runtime traversal and schema limits.
 * @throws InvalidArgument If the batch does not satisfy the schema.
 * @throws ResourceExhausted If validation traversal exceeds its bound.
 */
void validate_batch_value(const c10::IValue& value, const BatchSchema& schema,
                          const ValueSchemaLimits& limits = {});

/**
 * @brief Produces the complete deterministic binary batch schema descriptor.
 * @param schema Valid identity-bound schema to encode.
 * @param limits Resource limits applied during schema validation.
 * @return Binary-safe string containing ID, version, and value descriptor.
 */
std::string canonical_batch_schema(const BatchSchema& schema,
                                   const ValueSchemaLimits& limits = {});

/**
 * @brief Produces a compact diagnostic fingerprint for a batch schema.
 * @param schema Valid identity-bound schema to fingerprint.
 * @param limits Resource limits applied during schema validation.
 * @return Lowercase 64-bit FNV-1a hexadecimal diagnostic fingerprint.
 *
 * Restore compatibility compares the complete canonical descriptor, not this
 * compact diagnostic value.
 */
std::string batch_schema_fingerprint(const BatchSchema& schema,
                                     const ValueSchemaLimits& limits = {});

}  // namespace jyppx::data

#endif  // JYPPX_DATA_BATCH_SCHEMA_HPP
