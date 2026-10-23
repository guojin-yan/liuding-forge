// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_CORE_VALUE_SCHEMA_HPP
#define JYPPX_LIUDING_CORE_VALUE_SCHEMA_HPP
/**
 * @file value_schema.hpp
 * @brief Defines bounded schemas for LibTorch-backed structured values.
 */

#include <jyppx/liuding/core/types.hpp>

#include <ATen/core/ivalue.h>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace jyppx::liuding {

/**
 * @brief Identifies one runtime category accepted by a value schema.
 */
enum class ValueKind : std::uint8_t {
  any = 0,         ///< Accepts every non-null IValue category.
  tensor = 1,      ///< Accepts a defined tensor.
  tuple = 2,       ///< Accepts a fixed-length heterogeneous tuple.
  list = 3,        ///< Accepts a homogeneous generic list.
  dictionary = 4,  ///< Accepts a string-keyed generic dictionary.
  integer = 5,     ///< Accepts a signed integer scalar.
  floating = 6,    ///< Accepts a double-precision scalar.
  boolean = 7,     ///< Accepts a Boolean scalar.
  string = 8,      ///< Accepts an owned string value.
  none = 9,        ///< Accepts only the IValue null sentinel.
  device = 10,     ///< Accepts a LibTorch device descriptor.
};

/**
 * @brief Returns the stable lowercase name of a value category.
 * @param kind Value category to render.
 * @return A process-lifetime name, or `"unknown"` for an invalid value.
 */
std::string_view value_kind_name(ValueKind kind) noexcept;

/**
 * @brief Describes optional constraints for a tensor-valued field.
 *
 * An unset property is unconstrained. When `shape` is present, an empty vector
 * requires a scalar tensor and an empty dimension entry accepts any extent.
 */
struct TensorSchema {
  std::optional<Dtype> dtype;  ///< Exact scalar type required from the tensor.
  std::optional<Device> device;      ///< Exact tensor placement requirement.
  std::optional<std::int64_t> rank;  ///< Required number of dimensions.
  std::optional<std::vector<std::optional<std::int64_t>>>
      shape;  ///< Per-dimension extents; null entries are dynamic.
};

/**
 * @brief Bounds schema metadata and runtime validation traversal.
 */
struct ValueSchemaLimits {
  std::size_t max_depth = 32;           ///< Maximum nested container depth.
  std::size_t max_schema_nodes = 4096;  ///< Maximum declared schema nodes.
  std::size_t max_metadata_bytes =
      64U * 1024U;  ///< Maximum names, tags, keys, and device text bytes.
  std::size_t max_runtime_nodes =
      65536;  ///< Maximum values visited during one validation.
};

/**
 * @brief Recursively describes one structured framework value.
 *
 * `elements` contains tuple fields in order, or exactly one homogeneous list
 * element schema. `fields` contains dictionary entries. `required` controls
 * whether a named dictionary or invocation field may be omitted. `nullable`
 * permits an explicit IValue null in place of the declared non-null kind.
 * `semantic_tag` records domain meaning such as `features`, `target`, or
 * `sample_weight` without changing runtime type validation.
 */
struct ValueSchema {
  std::string name;          ///< Diagnostic name for this value or field.
  std::string semantic_tag;  ///< Stable optional domain meaning.
  ValueKind kind = ValueKind::any;  ///< Runtime category when present.
  bool required = true;             ///< Whether a named field must be present.
  bool nullable = false;  ///< Whether an explicit null value is accepted.
  std::optional<TensorSchema>
      tensor_constraints;  ///< Constraints for `ValueKind::tensor`.
  std::vector<ValueSchema>
      elements;  ///< Tuple entries or one homogeneous list entry.
  std::map<std::string, ValueSchema>
      fields;  ///< Declared string-keyed dictionary entries.
  bool allow_extra_fields =
      false;  ///< Whether undeclared dictionary keys are accepted.

  /**
   * @brief Creates an unconstrained non-null value schema.
   * @param name Diagnostic name for the value.
   * @param semantic_tag Optional stable domain meaning.
   * @return A schema accepting every non-null IValue category.
   */
  static ValueSchema any(std::string name = {}, std::string semantic_tag = {});

  /**
   * @brief Creates a tensor value schema.
   * @param constraints Optional dtype, device, rank, and shape constraints.
   * @param name Diagnostic name for the tensor.
   * @param semantic_tag Optional stable domain meaning.
   * @return A tensor schema with the supplied constraints.
   */
  static ValueSchema tensor(TensorSchema constraints = {},
                            std::string name = {},
                            std::string semantic_tag = {});

  /**
   * @brief Creates a fixed-length tuple schema.
   * @param elements Ordered schemas for tuple elements.
   * @param name Diagnostic name for the tuple.
   * @param semantic_tag Optional stable domain meaning.
   * @return A tuple schema owning the element schemas.
   */
  static ValueSchema tuple(std::vector<ValueSchema> elements,
                           std::string name = {},
                           std::string semantic_tag = {});

  /**
   * @brief Creates a homogeneous generic-list schema.
   * @param element Schema applied to every list element.
   * @param name Diagnostic name for the list.
   * @param semantic_tag Optional stable domain meaning.
   * @return A list schema owning the element schema.
   */
  static ValueSchema list(ValueSchema element, std::string name = {},
                          std::string semantic_tag = {});

  /**
   * @brief Creates a string-keyed dictionary schema.
   * @param fields Declared field schemas indexed by key.
   * @param allow_extra_fields Whether undeclared keys are accepted.
   * @param name Diagnostic name for the dictionary.
   * @param semantic_tag Optional stable domain meaning.
   * @return A dictionary schema owning all field schemas.
   */
  static ValueSchema dictionary(std::map<std::string, ValueSchema> fields,
                                bool allow_extra_fields = false,
                                std::string name = {},
                                std::string semantic_tag = {});
};

/**
 * @brief Validates that a value schema is internally consistent and bounded.
 * @param schema Schema to inspect recursively.
 * @param path Diagnostic path used for malformed schema fields.
 * @param limits Resource limits applied before accepting the schema.
 * @throws InvalidArgument If incompatible constraints are combined.
 * @throws ResourceExhausted If schema depth, nodes, or metadata exceed limits.
 */
void validate_value_schema(const ValueSchema& schema,
                           std::string_view path = "value",
                           const ValueSchemaLimits& limits = {});

/**
 * @brief Validates one runtime value against a declared bounded schema.
 * @param value Runtime IValue to inspect.
 * @param schema Expected value contract.
 * @param path Diagnostic path for a failed check.
 * @param limits Schema and runtime traversal limits.
 * @throws InvalidArgument If the value does not satisfy the contract.
 * @throws ResourceExhausted If runtime traversal exceeds its declared bound.
 */
void validate_value(const c10::IValue& value, const ValueSchema& schema,
                    std::string_view path = "value",
                    const ValueSchemaLimits& limits = {});

/**
 * @brief Produces the complete deterministic binary schema descriptor.
 * @param schema Valid schema to encode.
 * @param limits Resource limits applied while validating the schema.
 * @return Binary-safe string containing every schema field in canonical order.
 */
std::string canonical_value_schema(const ValueSchema& schema,
                                   const ValueSchemaLimits& limits = {});

/**
 * @brief Produces a compact diagnostic fingerprint of the complete schema.
 * @param schema Valid schema to fingerprint.
 * @param limits Resource limits applied while validating the schema.
 * @return Lowercase 64-bit FNV-1a hexadecimal diagnostic fingerprint.
 *
 * Compatibility checks must compare the full canonical descriptor and must
 * not rely on this compact fingerprint alone.
 */
std::string value_schema_fingerprint(const ValueSchema& schema,
                                     const ValueSchemaLimits& limits = {});

}  // namespace jyppx::liuding

#endif  // JYPPX_LIUDING_CORE_VALUE_SCHEMA_HPP