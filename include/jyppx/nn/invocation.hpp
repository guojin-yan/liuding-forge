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

#include <jyppx/core/value_schema.hpp>

#include <ATen/core/ivalue.h>
#include <map>
#include <string>
#include <vector>

namespace jyppx::nn {

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
  c10::IValue value;   ///< Type-erased result returned by the model invoker.
  ValueSchema schema;  ///< Declared output schema satisfied by `value`.
};

/**
 * @brief Validates that an invocation schema is internally consistent.
 * @param schema Invocation schema to inspect recursively.
 * @throws InvalidArgument If a keyword name or nested value schema is invalid.
 */
void validate_invocation_schema(const InvocationSchema& schema);

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
