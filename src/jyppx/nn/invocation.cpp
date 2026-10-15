// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/nn/invocation.hpp>

#include <c10/core/ScalarType.h>

#include <sstream>
#include <utility>

namespace jyppx::nn {
namespace {

constexpr char kComponent[] = "nn.invocation";

[[noreturn]] void throw_schema_error(std::string operation,
                                     std::string expected,
                                     std::string actual, std::string hint) {
  throw InvalidArgument(
      "Invocation schema check failed",
      ErrorDetails{.component = kComponent,
                   .operation = std::move(operation),
                   .expected = std::move(expected),
                   .actual = std::move(actual),
                   .hint = std::move(hint)});
}

std::string display_name(const ValueSchema& schema, std::string_view path) {
  if (schema.name.empty()) {
    return std::string(path);
  }
  return std::string(path) + " (" + schema.name + ")";
}

void reject_unused_children(const ValueSchema& schema, std::string_view path) {
  if (schema.tensor_constraints || !schema.elements.empty() ||
      !schema.fields.empty() || schema.allow_extra_fields) {
    throw_schema_error(std::string(path), "no nested constraints",
                       "nested constraints are present",
                       "Remove constraints that do not belong to this value kind.");
  }
}

bool value_matches_kind(const c10::IValue& value, ValueKind kind) {
  switch (kind) {
    case ValueKind::any:
      return !value.isNone();
    case ValueKind::tensor:
      return value.isTensor();
    case ValueKind::tuple:
      return value.isTuple();
    case ValueKind::list:
      return value.isList();
    case ValueKind::dictionary:
      return value.isGenericDict();
    case ValueKind::integer:
      return value.isInt();
    case ValueKind::floating:
      return value.isDouble();
    case ValueKind::boolean:
      return value.isBool();
    case ValueKind::string:
      return value.isString();
    case ValueKind::none:
      return value.isNone();
  }
  return false;
}

std::string render_shape(const Tensor& tensor) {
  std::ostringstream output;
  output << tensor.sizes();
  return output.str();
}

void validate_tensor(const Tensor& tensor, const TensorSchema& schema,
                     std::string_view path) {
  if (!tensor.defined()) {
    throw_schema_error(std::string(path), "a defined tensor", "undefined tensor",
                       "Return or pass a defined tensor handle.");
  }
  if (schema.dtype && tensor.scalar_type() != *schema.dtype) {
    throw_schema_error(std::string(path) + ".dtype",
                       c10::toString(*schema.dtype),
                       c10::toString(tensor.scalar_type()),
                       "Cast the tensor to the declared scalar type.");
  }
  if (schema.device && tensor.device() != *schema.device) {
    throw_schema_error(std::string(path) + ".device", schema.device->str(),
                       tensor.device().str(),
                       "Move the tensor to the declared device.");
  }
  if (schema.rank && tensor.dim() != *schema.rank) {
    throw_schema_error(std::string(path) + ".rank",
                       std::to_string(*schema.rank),
                       std::to_string(tensor.dim()),
                       "Reshape the tensor to the declared rank.");
  }
  if (!schema.shape) {
    return;
  }
  if (tensor.dim() != static_cast<std::int64_t>(schema.shape->size())) {
    throw_schema_error(std::string(path) + ".shape",
                       std::to_string(schema.shape->size()) + " dimensions",
                       render_shape(tensor),
                       "Match the declared tensor shape.");
  }
  for (std::size_t index = 0; index < schema.shape->size(); ++index) {
    const auto& extent = schema.shape->at(index);
    if (extent && tensor.size(static_cast<std::int64_t>(index)) != *extent) {
      throw_schema_error(std::string(path) + ".shape[" +
                             std::to_string(index) + "]",
                         std::to_string(*extent),
                         std::to_string(
                             tensor.size(static_cast<std::int64_t>(index))),
                         "Match the declared fixed dimension.");
    }
  }
}

}  // namespace

std::string_view value_kind_name(ValueKind kind) noexcept {
  switch (kind) {
    case ValueKind::any:
      return "any";
    case ValueKind::tensor:
      return "tensor";
    case ValueKind::tuple:
      return "tuple";
    case ValueKind::list:
      return "list";
    case ValueKind::dictionary:
      return "dictionary";
    case ValueKind::integer:
      return "integer";
    case ValueKind::floating:
      return "floating";
    case ValueKind::boolean:
      return "boolean";
    case ValueKind::string:
      return "string";
    case ValueKind::none:
      return "none";
  }
  return "unknown";
}

ValueSchema ValueSchema::any(std::string name) {
  ValueSchema schema;
  schema.name = std::move(name);
  return schema;
}

ValueSchema ValueSchema::tensor(TensorSchema constraints, std::string name) {
  ValueSchema schema;
  schema.name = std::move(name);
  schema.kind = ValueKind::tensor;
  schema.tensor_constraints = std::move(constraints);
  return schema;
}

ValueSchema ValueSchema::tuple(std::vector<ValueSchema> elements,
                               std::string name) {
  ValueSchema schema;
  schema.name = std::move(name);
  schema.kind = ValueKind::tuple;
  schema.elements = std::move(elements);
  return schema;
}

ValueSchema ValueSchema::list(ValueSchema element, std::string name) {
  ValueSchema schema;
  schema.name = std::move(name);
  schema.kind = ValueKind::list;
  schema.elements.push_back(std::move(element));
  return schema;
}

ValueSchema ValueSchema::dictionary(
    std::map<std::string, ValueSchema> fields, bool allow_extra_fields,
    std::string name) {
  ValueSchema schema;
  schema.name = std::move(name);
  schema.kind = ValueKind::dictionary;
  schema.fields = std::move(fields);
  schema.allow_extra_fields = allow_extra_fields;
  return schema;
}

void validate_value_schema(const ValueSchema& schema, std::string_view path) {
  switch (schema.kind) {
    case ValueKind::tensor: {
      if (!schema.elements.empty() || !schema.fields.empty() ||
          schema.allow_extra_fields) {
        throw_schema_error(std::string(path), "tensor constraints only",
                           "container constraints are present",
                           "Remove tuple, list, or dictionary constraints.");
      }
      if (!schema.tensor_constraints) {
        return;
      }
      const auto& tensor = *schema.tensor_constraints;
      if (tensor.rank && *tensor.rank < 0) {
        throw_schema_error(std::string(path) + ".rank", "zero or greater",
                           std::to_string(*tensor.rank),
                           "Use a non-negative tensor rank.");
      }
      if (tensor.shape) {
        if (tensor.rank &&
            *tensor.rank != static_cast<std::int64_t>(tensor.shape->size())) {
          throw_schema_error(std::string(path) + ".shape",
                             std::to_string(*tensor.rank) + " dimensions",
                             std::to_string(tensor.shape->size()) +
                                 " dimensions",
                             "Keep rank and shape cardinality consistent.");
        }
        for (std::size_t index = 0; index < tensor.shape->size(); ++index) {
          const auto& extent = tensor.shape->at(index);
          if (extent && *extent < 0) {
            throw_schema_error(std::string(path) + ".shape[" +
                                   std::to_string(index) + "]",
                               "zero or greater", std::to_string(*extent),
                               "Use null for a dynamic dimension.");
          }
        }
      }
      return;
    }
    case ValueKind::tuple:
      if (schema.tensor_constraints || !schema.fields.empty() ||
          schema.allow_extra_fields) {
        throw_schema_error(std::string(path), "tuple element schemas only",
                           "non-tuple constraints are present",
                           "Keep only ordered tuple elements.");
      }
      for (std::size_t index = 0; index < schema.elements.size(); ++index) {
        validate_value_schema(schema.elements[index],
                              std::string(path) + "[" +
                                  std::to_string(index) + "]");
      }
      return;
    case ValueKind::list:
      if (schema.tensor_constraints || schema.elements.size() != 1 ||
          !schema.fields.empty() || schema.allow_extra_fields) {
        throw_schema_error(std::string(path),
                           "exactly one homogeneous list element schema",
                           std::to_string(schema.elements.size()) +
                               " element schemas",
                           "Declare one schema shared by every list element.");
      }
      validate_value_schema(schema.elements.front(),
                            std::string(path) + "[]");
      return;
    case ValueKind::dictionary:
      if (schema.tensor_constraints || !schema.elements.empty()) {
        throw_schema_error(std::string(path), "dictionary field schemas only",
                           "non-dictionary constraints are present",
                           "Keep only named dictionary fields.");
      }
      for (const auto& [name, field] : schema.fields) {
        if (name.empty()) {
          throw_schema_error(std::string(path), "non-empty dictionary keys",
                             "empty key",
                             "Assign a stable name to every dictionary field.");
        }
        validate_value_schema(field, std::string(path) + "." + name);
      }
      return;
    case ValueKind::any:
    case ValueKind::integer:
    case ValueKind::floating:
    case ValueKind::boolean:
    case ValueKind::string:
    case ValueKind::none:
      reject_unused_children(schema, path);
      return;
  }
  throw_schema_error(std::string(path), "a known value kind", "unknown",
                     "Use a declared ValueKind enumerator.");
}

void validate_invocation_schema(const InvocationSchema& schema) {
  for (std::size_t index = 0; index < schema.positional.size(); ++index) {
    validate_value_schema(schema.positional[index],
                          "input.positional[" + std::to_string(index) + "]");
  }
  for (const auto& [name, field] : schema.keyword) {
    if (name.empty()) {
      throw_schema_error("input.keyword", "non-empty keyword names",
                         "empty name", "Assign a stable keyword name.");
    }
    validate_value_schema(field, "input.keyword." + name);
  }
}

void validate_value(const c10::IValue& value, const ValueSchema& schema,
                    std::string_view path) {
  const std::string diagnostic_path = display_name(schema, path);
  if (value.isNone() && schema.nullable) {
    return;
  }
  if (!value_matches_kind(value, schema.kind)) {
    throw_schema_error(diagnostic_path, std::string(value_kind_name(schema.kind)),
                       value.tagKind(),
                       "Match the declared model invocation schema.");
  }

  switch (schema.kind) {
    case ValueKind::tensor:
      validate_tensor(value.toTensor(),
                      schema.tensor_constraints.value_or(TensorSchema{}),
                      diagnostic_path);
      return;
    case ValueKind::tuple: {
      const auto& elements = value.toTupleRef().elements();
      if (elements.size() != schema.elements.size()) {
        throw_schema_error(diagnostic_path,
                           std::to_string(schema.elements.size()) +
                               " tuple elements",
                           std::to_string(elements.size()) + " tuple elements",
                           "Return or pass the declared tuple cardinality.");
      }
      for (std::size_t index = 0; index < elements.size(); ++index) {
        validate_value(elements[index], schema.elements[index],
                       std::string(path) + "[" + std::to_string(index) + "]");
      }
      return;
    }
    case ValueKind::list: {
      const auto elements = value.toListRef();
      for (std::size_t index = 0; index < elements.size(); ++index) {
        validate_value(elements[index], schema.elements.front(),
                       std::string(path) + "[" + std::to_string(index) + "]");
      }
      return;
    }
    case ValueKind::dictionary: {
      const auto dictionary = value.toGenericDict();
      std::map<std::string, c10::IValue> actual_fields;
      for (const auto& entry : dictionary) {
        if (!entry.key().isString()) {
          throw_schema_error(diagnostic_path, "string dictionary keys",
                             entry.key().tagKind(),
                             "Use string keys at the model schema boundary.");
        }
        const std::string key(entry.key().toStringRef());
        if (!schema.allow_extra_fields && !schema.fields.contains(key)) {
          throw_schema_error(diagnostic_path + "." + key,
                             "a declared dictionary field", "unexpected field",
                             "Remove the field or declare it in the schema.");
        }
        actual_fields.emplace(key, entry.value());
      }
      for (const auto& [name, field_schema] : schema.fields) {
        const auto found = actual_fields.find(name);
        if (found == actual_fields.end()) {
          if (field_schema.required) {
            throw_schema_error(diagnostic_path + "." + name,
                               "a required dictionary field", "missing field",
                               "Provide every required model field.");
          }
          continue;
        }
        validate_value(found->second, field_schema,
                       std::string(path) + "." + name);
      }
      return;
    }
    case ValueKind::any:
    case ValueKind::integer:
    case ValueKind::floating:
    case ValueKind::boolean:
    case ValueKind::string:
    case ValueKind::none:
      return;
  }
}

void validate_invocation(const Invocation& invocation,
                         const InvocationSchema& schema) {
  if (invocation.positional.size() != schema.positional.size()) {
    throw_schema_error("input.positional",
                       std::to_string(schema.positional.size()) + " arguments",
                       std::to_string(invocation.positional.size()) +
                           " arguments",
                       "Match the declared positional argument count.");
  }
  for (std::size_t index = 0; index < invocation.positional.size(); ++index) {
    validate_value(invocation.positional[index], schema.positional[index],
                   "input.positional[" + std::to_string(index) + "]");
  }
  for (const auto& [name, value] : invocation.keyword) {
    const auto expected = schema.keyword.find(name);
    if (expected == schema.keyword.end()) {
      if (!schema.allow_extra_keywords) {
        throw_schema_error("input.keyword." + name, "a declared keyword",
                           "unexpected keyword",
                           "Remove the keyword or declare it in the schema.");
      }
      continue;
    }
    validate_value(value, expected->second, "input.keyword." + name);
  }
  for (const auto& [name, expected] : schema.keyword) {
    if (expected.required && !invocation.keyword.contains(name)) {
      throw_schema_error("input.keyword." + name, "a required keyword",
                         "missing keyword",
                         "Provide every required model keyword.");
    }
  }
}

}  // namespace jyppx::nn
