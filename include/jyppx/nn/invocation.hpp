// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_NN_INVOCATION_HPP
#define JYPPX_NN_INVOCATION_HPP

/**
 * @file invocation.hpp
 * @brief Defines schema-checked, type-erased model invocation values.
 */

#include <jyppx/core/types.hpp>

#include <ATen/core/ivalue.h>

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace jyppx::nn {

/**
 * @brief Identifies the runtime value category accepted by a value schema.
 */
enum class ValueKind : std::uint8_t {
  any = 0,       ///< Accepts every non-null IValue category.
  tensor = 1,    ///< Accepts a defined tensor.
  tuple = 2,     ///< Accepts a fixed-length heterogeneous tuple.
  list = 3,      ///< Accepts a homogeneous generic list.
  dictionary = 4,  ///< Accepts a string-keyed generic dictionary.
  integer = 5,     ///< Accepts a signed integer scalar.
  floating = 6,    ///< Accepts a double-precision scalar.
  boolean = 7,     ///< Accepts a Boolean scalar.
  string = 8,      ///< Accepts an owned string value.
  none = 9,        ///< Accepts only the IValue null sentinel.
};

/**
 * @brief Returns the stable lowercase name of a value category.
 * @param kind Value category to render.
 * @return A process-lifetime name, or `"unknown"` for an invalid numeric value.
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
  std::optional<Device> device;  ///< Exact device required from the tensor.
  std::optional<std::int64_t> rank;  ///< Required number of tensor dimensions.
  std::optional<std::vector<std::optional<std::int64_t>>>
      shape;  ///< Per-dimension extents; null entries are dynamic.
};

/**
 * @brief Recursively describes one input or output value.
 *
 * `elements` contains tuple fields in order, or exactly one homogeneous list
 * element schema. `fields` contains dictionary entries. `required` controls
 * whether a named keyword or dictionary field may be omitted; positional and
 * tuple entries always have fixed cardinality. `nullable` permits an explicit
 * IValue null in place of the declared non-null kind.
 */
struct ValueSchema {
  std::string name;  ///< Diagnostic name for this value or field.
  ValueKind kind = ValueKind::any;  ///< Runtime category required when present.
  bool required = true;  ///< Whether a named field must be present.
  bool nullable = false;  ///< Whether an explicit null value is accepted.
  std::optional<TensorSchema>
      tensor_constraints;  ///< Tensor constraints for `ValueKind::tensor`.
  std::vector<ValueSchema>
      elements;  ///< Ordered tuple entries or one homogeneous list entry.
  std::map<std::string, ValueSchema>
      fields;  ///< Declared string-keyed dictionary entries.
  bool allow_extra_fields =
      false;  ///< Whether undeclared dictionary keys are accepted.

  /**
   * @brief Creates an unconstrained value schema.
   * @param name Diagnostic name for the value.
   * @return A schema accepting every non-null IValue category.
   */
  static ValueSchema any(std::string name = {});

  /**
   * @brief Creates a tensor value schema.
   * @param constraints Optional dtype, device, rank, and shape constraints.
   * @param name Diagnostic name for the tensor.
   * @return A tensor schema with the supplied constraints.
   */
  static ValueSchema tensor(TensorSchema constraints = {},
                            std::string name = {});

  /**
   * @brief Creates a fixed-length tuple schema.
   * @param elements Ordered schemas for tuple elements.
   * @param name Diagnostic name for the tuple.
   * @return A tuple schema owning the element schemas.
   */
  static ValueSchema tuple(std::vector<ValueSchema> elements,
                           std::string name = {});

  /**
   * @brief Creates a homogeneous generic-list schema.
   * @param element Schema applied to every list element.
   * @param name Diagnostic name for the list.
   * @return A list schema owning the element schema.
   */
  static ValueSchema list(ValueSchema element, std::string name = {});

  /**
   * @brief Creates a string-keyed dictionary schema.
   * @param fields Declared field schemas indexed by key.
   * @param allow_extra_fields Whether undeclared keys are accepted.
   * @param name Diagnostic name for the dictionary.
   * @return A dictionary schema owning all field schemas.
   */
  static ValueSchema dictionary(std::map<std::string, ValueSchema> fields,
                                bool allow_extra_fields = false,
                                std::string name = {});
};

/**
 * @brief Describes the positional and keyword boundary of a model call.
 */
struct InvocationSchema {
  std::vector<ValueSchema>
      positional;  ///< Fixed ordered schemas for positional arguments.
  std::map<std::string, ValueSchema>
      keyword;  ///< Declared schemas for named arguments.
  bool allow_extra_keywords =
      false;  ///< Whether undeclared keyword arguments are accepted.
};

/**
 * @brief Carries one in-process model invocation.
 *
 * IValue supports tensors, scalars, tuples, lists, and dictionaries without
 * imposing a process-stable serialization format. Persisted protocols must use
 * a separately versioned schema and codec.
 */
struct Invocation {
  std::vector<c10::IValue> positional;  ///< Ordered positional arguments.
  std::map<std::string, c10::IValue> keyword;  ///< Named call arguments.
};

/**
 * @brief Carries a validated model result and the schema that validated it.
 */
struct InvocationResult {
  c10::IValue value;  ///< Type-erased result returned by the model invoker.
  ValueSchema schema;  ///< Declared output schema satisfied by `value`.
};

/**
 * @brief Validates that a value schema is internally consistent.
 * @param schema Schema to inspect recursively.
 * @param path Diagnostic path used when reporting a malformed schema.
 * @throws InvalidArgument If incompatible constraints are combined.
 */
void validate_value_schema(const ValueSchema& schema,
                           std::string_view path = "value");

/**
 * @brief Validates that an invocation schema is internally consistent.
 * @param schema Invocation schema to inspect recursively.
 * @throws InvalidArgument If a keyword name or nested value schema is invalid.
 */
void validate_invocation_schema(const InvocationSchema& schema);

/**
 * @brief Validates one runtime value against a declared schema.
 * @param value Runtime IValue to inspect.
 * @param schema Expected value contract.
 * @param path Diagnostic path for a failed check.
 * @throws InvalidArgument If the value does not satisfy the contract.
 */
void validate_value(const c10::IValue& value, const ValueSchema& schema,
                    std::string_view path = "value");

/**
 * @brief Validates positional and keyword arguments before model execution.
 * @param invocation Runtime arguments to inspect.
 * @param schema Expected invocation contract.
 * @throws InvalidArgument If cardinality, names, or nested values mismatch.
 */
void validate_invocation(const Invocation& invocation,
                         const InvocationSchema& schema);

}  // namespace jyppx::nn

#endif  // JYPPX_NN_INVOCATION_HPP
