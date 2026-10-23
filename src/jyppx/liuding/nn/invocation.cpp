// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/nn/invocation.hpp>

#include <utility>

namespace jyppx::liuding::nn {
namespace {

constexpr char kComponent[] = "nn.invocation";

[[noreturn]] void throw_invocation_error(std::string operation,
                                         std::string expected,
                                         std::string actual, std::string hint) {
  throw InvalidArgument("Invocation schema check failed",
                        ErrorDetails{.component = kComponent,
                                     .operation = std::move(operation),
                                     .expected = std::move(expected),
                                     .actual = std::move(actual),
                                     .hint = std::move(hint)});
}

}  // namespace

void validate_invocation_schema(const InvocationSchema& schema) {
  for (std::size_t index = 0; index < schema.positional.size(); ++index) {
    validate_value_schema(schema.positional[index],
                          "input.positional[" + std::to_string(index) + "]");
  }
  for (const auto& [name, field] : schema.keyword) {
    if (name.empty()) {
      throw_invocation_error("input.keyword", "non-empty keyword names",
                             "empty name", "Assign a stable keyword name.");
    }
    validate_value_schema(field, "input.keyword." + name);
  }
}

void validate_invocation(const Invocation& invocation,
                         const InvocationSchema& schema) {
  if (invocation.positional.size() != schema.positional.size()) {
    throw_invocation_error(
        "input.positional",
        std::to_string(schema.positional.size()) + " arguments",
        std::to_string(invocation.positional.size()) + " arguments",
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
        throw_invocation_error(
            "input.keyword." + name, "a declared keyword", "unexpected keyword",
            "Remove the keyword or declare it in the schema.");
      }
      continue;
    }
    validate_value(value, expected->second, "input.keyword." + name);
  }
  for (const auto& [name, expected] : schema.keyword) {
    if (expected.required && !invocation.keyword.contains(name)) {
      throw_invocation_error("input.keyword." + name, "a required keyword",
                             "missing keyword",
                             "Provide every required model keyword.");
    }
  }
}

}  // namespace jyppx::liuding::nn
