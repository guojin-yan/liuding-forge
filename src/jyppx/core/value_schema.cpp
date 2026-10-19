// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/core/value_schema.hpp>
#include <jyppx/foundation/error.hpp>

#include <c10/core/ScalarType.h>
#include <iomanip>
#include <sstream>
#include <utility>

namespace jyppx {
namespace {

constexpr char kComponent[] = "core.value_schema";
constexpr std::string_view kCanonicalMagic = "JYPPX_VALUE_SCHEMA_V1";

[[noreturn]] void throw_schema_error(std::string operation,
                                     std::string expected, std::string actual,
                                     std::string hint) {
  throw InvalidArgument("Value schema check failed",
                        ErrorDetails{.component = kComponent,
                                     .operation = std::move(operation),
                                     .expected = std::move(expected),
                                     .actual = std::move(actual),
                                     .hint = std::move(hint)});
}

[[noreturn]] void throw_limit_error(std::string operation, std::string expected,
                                    std::string actual) {
  throw ResourceExhausted(
      "Value schema resource limit exceeded",
      ErrorDetails{.component = kComponent,
                   .operation = std::move(operation),
                   .expected = std::move(expected),
                   .actual = std::move(actual),
                   .hint = "Reduce schema or runtime container complexity."});
}

void validate_limits(const ValueSchemaLimits& limits) {
  if (limits.max_depth == 0 || limits.max_schema_nodes == 0 ||
      limits.max_metadata_bytes == 0 || limits.max_runtime_nodes == 0) {
    throw_schema_error("limits", "strictly positive resource limits", "zero",
                       "Set every ValueSchema limit to a positive value.");
  }
}

void checked_add(std::size_t& current, std::size_t addition,
                 std::size_t maximum, std::string_view operation) {
  if (addition > maximum || current > maximum - addition) {
    throw_limit_error(std::string(operation),
                      "at most " + std::to_string(maximum),
                      "more than the declared bound");
  }
  current += addition;
}

std::string display_name(const ValueSchema& schema, std::string_view path) {
  return schema.name.empty() ? std::string(path)
                             : std::string(path) + " (" + schema.name + ")";
}

void reject_unused_children(const ValueSchema& schema, std::string_view path) {
  if (schema.tensor_constraints || !schema.elements.empty() ||
      !schema.fields.empty() || schema.allow_extra_fields) {
    throw_schema_error(
        std::string(path), "no nested constraints",
        "nested constraints are present",
        "Remove constraints that do not belong to this value kind.");
  }
}

struct SchemaBudget {
  const ValueSchemaLimits& limits;
  std::size_t nodes = 0;
  std::size_t metadata_bytes = 0;
};

void validate_schema_node(const ValueSchema& schema, std::string_view path,
                          std::size_t depth, SchemaBudget& budget) {
  if (depth > budget.limits.max_depth) {
    throw_limit_error(std::string(path) + ".depth",
                      "at most " + std::to_string(budget.limits.max_depth),
                      std::to_string(depth));
  }
  checked_add(budget.nodes, 1, budget.limits.max_schema_nodes,
              std::string(path) + ".nodes");
  checked_add(budget.metadata_bytes, schema.name.size(),
              budget.limits.max_metadata_bytes, std::string(path) + ".name");
  checked_add(budget.metadata_bytes, schema.semantic_tag.size(),
              budget.limits.max_metadata_bytes,
              std::string(path) + ".semantic_tag");

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
      if (tensor.device) {
        checked_add(budget.metadata_bytes, tensor.device->str().size(),
                    budget.limits.max_metadata_bytes,
                    std::string(path) + ".device");
      }
      if (!tensor.shape) {
        return;
      }
      checked_add(budget.nodes, tensor.shape->size(),
                  budget.limits.max_schema_nodes,
                  std::string(path) + ".shape_dimensions");
      if (tensor.rank &&
          *tensor.rank != static_cast<std::int64_t>(tensor.shape->size())) {
        throw_schema_error(std::string(path) + ".shape",
                           std::to_string(*tensor.rank) + " dimensions",
                           std::to_string(tensor.shape->size()) + " dimensions",
                           "Keep rank and shape cardinality consistent.");
      }
      for (std::size_t index = 0; index < tensor.shape->size(); ++index) {
        const auto& extent = tensor.shape->at(index);
        if (extent && *extent < 0) {
          throw_schema_error(
              std::string(path) + ".shape[" + std::to_string(index) + "]",
              "zero or greater", std::to_string(*extent),
              "Use null for a dynamic dimension.");
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
        const auto child_path =
            std::string(path) + "[" + std::to_string(index) + "]";
        validate_schema_node(schema.elements[index], child_path, depth + 1,
                             budget);
      }
      return;
    case ValueKind::list:
      if (schema.tensor_constraints || schema.elements.size() != 1 ||
          !schema.fields.empty() || schema.allow_extra_fields) {
        throw_schema_error(
            std::string(path), "exactly one homogeneous list element schema",
            std::to_string(schema.elements.size()) + " element schemas",
            "Declare one schema shared by every list element.");
      }
      validate_schema_node(schema.elements.front(), std::string(path) + "[]",
                           depth + 1, budget);
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
                             "Assign a stable name to every field.");
        }
        checked_add(budget.metadata_bytes, name.size(),
                    budget.limits.max_metadata_bytes,
                    std::string(path) + ".field_name");
        validate_schema_node(field, std::string(path) + "." + name, depth + 1,
                             budget);
      }
      return;
    case ValueKind::any:
    case ValueKind::integer:
    case ValueKind::floating:
    case ValueKind::boolean:
    case ValueKind::string:
    case ValueKind::none:
    case ValueKind::device:
      reject_unused_children(schema, path);
      return;
  }
  throw_schema_error(std::string(path), "a known value kind", "unknown",
                     "Use a declared ValueKind enumerator.");
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
    case ValueKind::device:
      return value.isDevice();
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
    throw_schema_error(std::string(path), "a defined tensor", "undefined",
                       "Provide a defined tensor handle.");
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
      throw_schema_error(
          std::string(path) + ".shape[" + std::to_string(index) + "]",
          std::to_string(*extent),
          std::to_string(tensor.size(static_cast<std::int64_t>(index))),
          "Match the declared fixed dimension.");
    }
  }
}

struct RuntimeBudget {
  const ValueSchemaLimits& limits;
  std::size_t nodes = 0;
  std::size_t metadata_bytes = 0;
};

void charge_runtime(RuntimeBudget& budget, std::string_view path,
                    std::size_t depth) {
  if (depth > budget.limits.max_depth) {
    throw_limit_error(std::string(path) + ".depth",
                      "at most " + std::to_string(budget.limits.max_depth),
                      std::to_string(depth));
  }
  checked_add(budget.nodes, 1, budget.limits.max_runtime_nodes,
              std::string(path) + ".runtime_nodes");
}

void charge_runtime_metadata(RuntimeBudget& budget, std::string_view path,
                             std::size_t bytes) {
  checked_add(budget.metadata_bytes, bytes, budget.limits.max_metadata_bytes,
              std::string(path) + ".runtime_metadata_bytes");
}

void scan_any(const c10::IValue& value, std::string_view path,
              std::size_t depth, RuntimeBudget& budget) {
  charge_runtime(budget, path, depth);
  if (value.isTuple()) {
    const auto& values = value.toTupleRef().elements();
    for (std::size_t index = 0; index < values.size(); ++index) {
      scan_any(values[index],
               std::string(path) + "[" + std::to_string(index) + "]", depth + 1,
               budget);
    }
  } else if (value.isList()) {
    const auto values = value.toListRef();
    for (std::size_t index = 0; index < values.size(); ++index) {
      scan_any(values[index],
               std::string(path) + "[" + std::to_string(index) + "]", depth + 1,
               budget);
    }
  } else if (value.isGenericDict()) {
    std::size_t index = 0;
    for (const auto& entry : value.toGenericDict()) {
      if (entry.key().isString()) {
        charge_runtime_metadata(
            budget, std::string(path) + ".key[" + std::to_string(index) + "]",
            entry.key().toStringRef().size());
      }
      scan_any(entry.key(),
               std::string(path) + ".key[" + std::to_string(index) + "]",
               depth + 1, budget);
      scan_any(entry.value(),
               std::string(path) + ".value[" + std::to_string(index) + "]",
               depth + 1, budget);
      ++index;
    }
  }
}

void validate_runtime(const c10::IValue& value, const ValueSchema& schema,
                      std::string_view path, std::size_t depth,
                      RuntimeBudget& budget) {
  const auto diagnostic_path = display_name(schema, path);
  if (value.isNone() && schema.nullable) {
    charge_runtime(budget, diagnostic_path, depth);
    return;
  }
  if (!value_matches_kind(value, schema.kind)) {
    throw_schema_error(
        diagnostic_path, std::string(value_kind_name(schema.kind)),
        value.tagKind(), "Match the declared structured value schema.");
  }
  if (schema.kind == ValueKind::any) {
    scan_any(value, path, depth, budget);
    return;
  }
  charge_runtime(budget, diagnostic_path, depth);

  switch (schema.kind) {
    case ValueKind::tensor:
      validate_tensor(value.toTensor(),
                      schema.tensor_constraints.value_or(TensorSchema{}),
                      diagnostic_path);
      return;
    case ValueKind::tuple: {
      const auto& values = value.toTupleRef().elements();
      if (values.size() != schema.elements.size()) {
        throw_schema_error(
            diagnostic_path,
            std::to_string(schema.elements.size()) + " tuple elements",
            std::to_string(values.size()) + " tuple elements",
            "Produce the declared tuple cardinality.");
      }
      for (std::size_t index = 0; index < values.size(); ++index) {
        validate_runtime(values[index], schema.elements[index],
                         std::string(path) + "[" + std::to_string(index) + "]",
                         depth + 1, budget);
      }
      return;
    }
    case ValueKind::list: {
      const auto values = value.toListRef();
      for (std::size_t index = 0; index < values.size(); ++index) {
        validate_runtime(values[index], schema.elements.front(),
                         std::string(path) + "[" + std::to_string(index) + "]",
                         depth + 1, budget);
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
                             "Use string keys at the schema boundary.");
        }
        const auto key_view = entry.key().toStringRef();
        charge_runtime_metadata(budget, diagnostic_path, key_view.size());
        const std::string key(key_view);
        if (!schema.allow_extra_fields && !schema.fields.contains(key)) {
          throw_schema_error(diagnostic_path + "." + key,
                             "a declared dictionary field", "unexpected",
                             "Remove or declare the field.");
        }
        actual_fields.emplace(key, entry.value());
      }
      for (const auto& [name, field_schema] : schema.fields) {
        const auto found = actual_fields.find(name);
        if (found == actual_fields.end()) {
          if (field_schema.required) {
            throw_schema_error(diagnostic_path + "." + name,
                               "a required dictionary field", "missing",
                               "Provide every required field.");
          }
          continue;
        }
        validate_runtime(found->second, field_schema,
                         std::string(path) + "." + name, depth + 1, budget);
        actual_fields.erase(found);
      }
      if (schema.allow_extra_fields) {
        for (const auto& [name, extra] : actual_fields) {
          scan_any(extra, std::string(path) + "." + name, depth + 1, budget);
        }
      }
      return;
    }
    case ValueKind::any:
    case ValueKind::integer:
    case ValueKind::floating:
    case ValueKind::boolean:
    case ValueKind::string:
    case ValueKind::none:
    case ValueKind::device:
      return;
  }
}

void append_u8(std::string& output, std::uint8_t value) {
  output.push_back(static_cast<char>(value));
}

void append_u64(std::string& output, std::uint64_t value) {
  for (unsigned int shift = 0; shift < 64; shift += 8) {
    append_u8(output, static_cast<std::uint8_t>(value >> shift));
  }
}

void append_string(std::string& output, std::string_view value) {
  append_u64(output, static_cast<std::uint64_t>(value.size()));
  output.append(value);
}

template <typename T, typename AppendValue>
void append_optional(std::string& output, const std::optional<T>& value,
                     AppendValue append_value) {
  append_u8(output, value.has_value() ? 1U : 0U);
  if (value) {
    append_value(output, *value);
  }
}

void append_schema(std::string& output, const ValueSchema& schema) {
  append_u8(output, static_cast<std::uint8_t>(schema.kind));
  append_u8(output, schema.required ? 1U : 0U);
  append_u8(output, schema.nullable ? 1U : 0U);
  append_u8(output, schema.allow_extra_fields ? 1U : 0U);
  append_string(output, schema.name);
  append_string(output, schema.semantic_tag);
  append_u8(output, schema.tensor_constraints.has_value() ? 1U : 0U);
  if (schema.tensor_constraints) {
    const auto& tensor = *schema.tensor_constraints;
    append_optional<Dtype>(
        output, tensor.dtype, [](std::string& target, Dtype dtype) {
          append_u8(target, static_cast<std::uint8_t>(dtype));
        });
    append_optional<Device>(output, tensor.device,
                            [](std::string& target, const Device& device) {
                              append_string(target, device.str());
                            });
    append_optional<std::int64_t>(
        output, tensor.rank, [](std::string& target, std::int64_t rank) {
          append_u64(target, static_cast<std::uint64_t>(rank));
        });
    append_u8(output, tensor.shape.has_value() ? 1U : 0U);
    if (tensor.shape) {
      append_u64(output, static_cast<std::uint64_t>(tensor.shape->size()));
      for (const auto& extent : *tensor.shape) {
        append_optional<std::int64_t>(
            output, extent, [](std::string& target, std::int64_t value) {
              append_u64(target, static_cast<std::uint64_t>(value));
            });
      }
    }
  }
  append_u64(output, static_cast<std::uint64_t>(schema.elements.size()));
  for (const auto& element : schema.elements) {
    append_schema(output, element);
  }
  append_u64(output, static_cast<std::uint64_t>(schema.fields.size()));
  for (const auto& [name, field] : schema.fields) {
    append_string(output, name);
    append_schema(output, field);
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
    case ValueKind::device:
      return "device";
  }
  return "unknown";
}

ValueSchema ValueSchema::any(std::string name, std::string semantic_tag) {
  ValueSchema schema;
  schema.name = std::move(name);
  schema.semantic_tag = std::move(semantic_tag);
  return schema;
}

ValueSchema ValueSchema::tensor(TensorSchema constraints, std::string name,
                                std::string semantic_tag) {
  ValueSchema schema;
  schema.name = std::move(name);
  schema.semantic_tag = std::move(semantic_tag);
  schema.kind = ValueKind::tensor;
  schema.tensor_constraints = std::move(constraints);
  return schema;
}

ValueSchema ValueSchema::tuple(std::vector<ValueSchema> elements,
                               std::string name, std::string semantic_tag) {
  ValueSchema schema;
  schema.name = std::move(name);
  schema.semantic_tag = std::move(semantic_tag);
  schema.kind = ValueKind::tuple;
  schema.elements = std::move(elements);
  return schema;
}

ValueSchema ValueSchema::list(ValueSchema element, std::string name,
                              std::string semantic_tag) {
  ValueSchema schema;
  schema.name = std::move(name);
  schema.semantic_tag = std::move(semantic_tag);
  schema.kind = ValueKind::list;
  schema.elements.push_back(std::move(element));
  return schema;
}

ValueSchema ValueSchema::dictionary(std::map<std::string, ValueSchema> fields,
                                    bool allow_extra_fields, std::string name,
                                    std::string semantic_tag) {
  ValueSchema schema;
  schema.name = std::move(name);
  schema.semantic_tag = std::move(semantic_tag);
  schema.kind = ValueKind::dictionary;
  schema.fields = std::move(fields);
  schema.allow_extra_fields = allow_extra_fields;
  return schema;
}

void validate_value_schema(const ValueSchema& schema, std::string_view path,
                           const ValueSchemaLimits& limits) {
  validate_limits(limits);
  SchemaBudget budget{limits};
  validate_schema_node(schema, path, 1, budget);
}

void validate_value(const c10::IValue& value, const ValueSchema& schema,
                    std::string_view path, const ValueSchemaLimits& limits) {
  validate_value_schema(schema, path, limits);
  RuntimeBudget budget{limits};
  validate_runtime(value, schema, path, 1, budget);
}

std::string canonical_value_schema(const ValueSchema& schema,
                                   const ValueSchemaLimits& limits) {
  validate_value_schema(schema, "value", limits);
  std::string output;
  append_string(output, kCanonicalMagic);
  append_schema(output, schema);
  return output;
}

std::string value_schema_fingerprint(const ValueSchema& schema,
                                     const ValueSchemaLimits& limits) {
  const auto canonical = canonical_value_schema(schema, limits);
  std::uint64_t hash = 14695981039346656037ULL;
  for (const auto byte : canonical) {
    hash ^= static_cast<std::uint8_t>(byte);
    hash *= 1099511628211ULL;
  }
  std::ostringstream output;
  output << std::hex << std::setfill('0') << std::setw(16) << hash;
  return output.str();
}

}  // namespace jyppx
