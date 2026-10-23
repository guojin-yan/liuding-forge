// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/experiment/config.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>
#include <system_error>

namespace jyppx::liuding::experiment::config {
namespace {

using Json = nlohmann::json;

constexpr char kComponent[] = "experiment.config";
constexpr std::string_view kRedacted = "<redacted>";

[[noreturn]] void ThrowInvalid(std::string operation, std::string expected,
                               std::string actual, std::string hint) {
  throw InvalidArgument("Config validation failed",
                        ErrorDetails{.component = kComponent,
                                     .operation = std::move(operation),
                                     .expected = std::move(expected),
                                     .actual = std::move(actual),
                                     .hint = std::move(hint)});
}

[[noreturn]] void ThrowSerialization(std::string operation,
                                     std::string expected, std::string actual,
                                     std::string hint) {
  throw SerializationError("Config JSON is invalid",
                           ErrorDetails{.component = kComponent,
                                        .operation = std::move(operation),
                                        .expected = std::move(expected),
                                        .actual = std::move(actual),
                                        .hint = std::move(hint)});
}

[[noreturn]] void ThrowLimit(std::string operation, std::string expected,
                             std::string actual) {
  throw ResourceExhausted(
      "Config resource limit exceeded",
      ErrorDetails{.component = kComponent,
                   .operation = std::move(operation),
                   .expected = std::move(expected),
                   .actual = std::move(actual),
                   .hint = "Reduce the config source or raise an explicit "
                           "trusted ConfigLimits bound."});
}

[[noreturn]] void ThrowIo(std::string operation, std::string actual) {
  throw IoError("Config file I/O failed",
                ErrorDetails{.component = kComponent,
                             .operation = std::move(operation),
                             .expected = "a readable regular JSON file",
                             .actual = std::move(actual),
                             .hint = "Check the path, permissions, and file "
                                     "size before resolving config."});
}

std::optional<std::string> ReadProcessEnvironmentVariable(
    const std::string& name) {
#ifdef _WIN32
  char* raw_value = nullptr;
  std::size_t length = 0;
  const errno_t status = _dupenv_s(&raw_value, &length, name.c_str());
  std::unique_ptr<char, decltype(&std::free)> value(raw_value, &std::free);
  if (status != 0) {
    throw IoError(
        "Process environment access failed",
        ErrorDetails{.component = kComponent,
                     .operation = "environment[" + name + "]",
                     .expected = "a readable process environment snapshot",
                     .actual = "runtime error " + std::to_string(status),
                     .hint = "Inject an explicit environment snapshot with "
                             "add_environment()."});
  }
  if (value == nullptr) {
    return std::nullopt;
  }
  return std::string(value.get(), length == 0 ? 0 : length - 1);
#else
  const char* value = std::getenv(name.c_str());
  if (value == nullptr) {
    return std::nullopt;
  }
  return std::string(value);
#endif
}

void CheckedAdd(std::size_t& value, std::size_t addition, std::size_t maximum,
                std::string_view operation) {
  if (addition > maximum || value > maximum - addition) {
    ThrowLimit(std::string(operation), "at most " + std::to_string(maximum),
               "more than the declared bound");
  }
  value += addition;
}

bool IsContinuation(unsigned char byte) { return (byte & 0xC0U) == 0x80U; }

bool IsValidUtf8(std::string_view text) {
  for (std::size_t index = 0; index < text.size();) {
    const auto first = static_cast<unsigned char>(text[index]);
    if (first <= 0x7FU) {
      ++index;
      continue;
    }
    if (first >= 0xC2U && first <= 0xDFU) {
      if (index + 1 >= text.size() ||
          !IsContinuation(static_cast<unsigned char>(text[index + 1]))) {
        return false;
      }
      index += 2;
      continue;
    }
    if (first >= 0xE0U && first <= 0xEFU) {
      if (index + 2 >= text.size()) return false;
      const auto second = static_cast<unsigned char>(text[index + 1]);
      const auto third = static_cast<unsigned char>(text[index + 2]);
      if (!IsContinuation(second) || !IsContinuation(third) ||
          (first == 0xE0U && second < 0xA0U) ||
          (first == 0xEDU && second >= 0xA0U)) {
        return false;
      }
      index += 3;
      continue;
    }
    if (first >= 0xF0U && first <= 0xF4U) {
      if (index + 3 >= text.size()) return false;
      const auto second = static_cast<unsigned char>(text[index + 1]);
      const auto third = static_cast<unsigned char>(text[index + 2]);
      const auto fourth = static_cast<unsigned char>(text[index + 3]);
      if (!IsContinuation(second) || !IsContinuation(third) ||
          !IsContinuation(fourth) || (first == 0xF0U && second < 0x90U) ||
          (first == 0xF4U && second >= 0x90U)) {
        return false;
      }
      index += 4;
      continue;
    }
    return false;
  }
  return true;
}

void ValidateUtf8(std::string_view text, std::string_view operation) {
  if (!IsValidUtf8(text)) {
    ThrowInvalid(std::string(operation), "well-formed UTF-8",
                 "invalid UTF-8 byte sequence",
                 "Encode config text and metadata as UTF-8 without "
                 "surrogate or overlong sequences.");
  }
}

bool IsIdentifier(std::string_view text) {
  if (text.empty() || text.size() > 128U) return false;
  return std::all_of(text.begin(), text.end(), [](char value) {
    const auto byte = static_cast<unsigned char>(value);
    return (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') ||
           (byte >= '0' && byte <= '9') || value == '.' || value == '_' ||
           value == '-' || value == '/';
  });
}

void ValidateLimits(const ConfigLimits& limits) {
  if (limits.max_file_bytes == 0 || limits.max_depth == 0 ||
      limits.max_nodes == 0 || limits.max_string_bytes == 0 ||
      limits.max_total_string_bytes == 0 || limits.max_json_files == 0 ||
      limits.max_cli_overrides == 0) {
    ThrowInvalid("limits", "strictly positive config limits", "zero",
                 "Set every ConfigLimits field to a positive value.");
  }
}

std::string PathText(const std::filesystem::path& path) {
  const auto utf8 = path.generic_u8string();
  return std::string(reinterpret_cast<const char*>(utf8.data()), utf8.size());
}

std::string EscapePointerToken(std::string_view token) {
  std::string output;
  output.reserve(token.size());
  for (const char value : token) {
    if (value == '~') {
      output += "~0";
    } else if (value == '/') {
      output += "~1";
    } else {
      output.push_back(value);
    }
  }
  return output;
}

std::string ChildPath(std::string_view path, std::string_view token) {
  return std::string(path) + "/" + EscapePointerToken(token);
}

std::vector<std::string> DecodePointer(std::string_view path) {
  if (path.empty()) return {};
  if (path.front() != '/') {
    ThrowInvalid(std::string(path), "an empty path or RFC 6901 JSON Pointer",
                 "path does not start with '/'",
                 "Use a path such as /trainer/learning_rate.");
  }
  std::vector<std::string> tokens;
  std::size_t start = 1;
  while (start <= path.size()) {
    const auto end = path.find('/', start);
    const auto encoded =
        path.substr(start, end == std::string_view::npos ? path.size() - start
                                                         : end - start);
    std::string token;
    token.reserve(encoded.size());
    for (std::size_t index = 0; index < encoded.size(); ++index) {
      if (encoded[index] != '~') {
        token.push_back(encoded[index]);
        continue;
      }
      if (index + 1 >= encoded.size() ||
          (encoded[index + 1] != '0' && encoded[index + 1] != '1')) {
        ThrowInvalid(std::string(path), "valid ~0 and ~1 pointer escapes",
                     "invalid '~' escape", "Escape '~' as ~0 and '/' as ~1.");
      }
      token.push_back(encoded[++index] == '0' ? '~' : '/');
    }
    if (token.empty()) {
      ThrowInvalid(std::string(path), "non-empty declared field tokens",
                   "empty JSON Pointer token", "Select a named schema field.");
    }
    tokens.push_back(std::move(token));
    if (end == std::string_view::npos) break;
    start = end + 1;
  }
  return tokens;
}

std::string CanonicalPointer(const std::vector<std::string>& tokens) {
  std::string output;
  for (const auto& token : tokens) output = ChildPath(output, token);
  return output;
}

std::size_t ParseArrayIndex(std::string_view token, std::string_view path) {
  if (token.empty() || (token.size() > 1 && token.front() == '0')) {
    ThrowInvalid(std::string(path), "a canonical non-negative array index",
                 std::string(token), "Use 0 or a decimal index without zeros.");
  }
  std::size_t value = 0;
  const auto result =
      std::from_chars(token.data(), token.data() + token.size(), value);
  if (result.ec != std::errc{} || result.ptr != token.data() + token.size()) {
    ThrowInvalid(std::string(path), "a non-negative array index",
                 std::string(token), "Select a valid list or tuple element.");
  }
  return value;
}

const ValueSchema& SchemaAtTokens(const ValueSchema& root,
                                  const std::vector<std::string>& tokens,
                                  std::string_view original_path) {
  const ValueSchema* current = &root;
  for (const auto& token : tokens) {
    if (current->kind == ValueKind::dictionary) {
      const auto iterator = current->fields.find(token);
      if (iterator == current->fields.end()) {
        ThrowInvalid(std::string(original_path), "a declared config field",
                     token,
                     "Remove the unknown field or add it to the schema.");
      }
      current = &iterator->second;
      continue;
    }
    if (current->kind == ValueKind::list) {
      (void)ParseArrayIndex(token, original_path);
      current = &current->elements.front();
      continue;
    }
    if (current->kind == ValueKind::tuple) {
      const auto index = ParseArrayIndex(token, original_path);
      if (index >= current->elements.size()) {
        ThrowInvalid(
            std::string(original_path),
            "a tuple index below " + std::to_string(current->elements.size()),
            std::to_string(index), "Select a declared tuple element.");
      }
      current = &current->elements[index];
      continue;
    }
    ThrowInvalid(std::string(original_path), "a container path segment", token,
                 "Stop the pointer at the scalar field.");
  }
  return *current;
}

const ValueSchema& SchemaAtPointer(const ValueSchema& root,
                                   std::string_view path) {
  return SchemaAtTokens(root, DecodePointer(path), path);
}

void ValidateSchemaNode(const ValueSchema& schema, std::string_view path) {
  if (schema.allow_extra_fields) {
    ThrowInvalid(std::string(path) + ".allow_extra_fields", "false", "true",
                 "The first config slice rejects every unknown field.");
  }
  switch (schema.kind) {
    case ValueKind::dictionary:
      for (const auto& [name, child] : schema.fields) {
        ValidateUtf8(name, std::string(path) + ".field");
        ValidateSchemaNode(child, ChildPath(path, name));
      }
      return;
    case ValueKind::list:
      ValidateSchemaNode(schema.elements.front(), std::string(path) + "/*");
      return;
    case ValueKind::tuple:
      for (std::size_t index = 0; index < schema.elements.size(); ++index) {
        ValidateSchemaNode(schema.elements[index],
                           ChildPath(path, std::to_string(index)));
      }
      return;
    case ValueKind::integer:
    case ValueKind::floating:
    case ValueKind::boolean:
    case ValueKind::string:
    case ValueKind::none:
      return;
    case ValueKind::any:
    case ValueKind::tensor:
    case ValueKind::device:
      ThrowInvalid(std::string(path),
                   "dictionary, list, tuple, scalar, string, or none schema",
                   std::string(value_kind_name(schema.kind)),
                   "Use the existing ValueSchema config-compatible kinds.");
  }
  ThrowInvalid(std::string(path), "a known ValueKind", "unknown",
               "Use a declared ValueKind enumerator.");
}

void ValidatePartialValue(const c10::IValue& value, const ValueSchema& schema,
                          std::string_view path,
                          const ValueSchemaLimits& limits) {
  if (value.isNone()) {
    if (schema.nullable || schema.kind == ValueKind::none) return;
    ThrowInvalid(std::string(path), std::string(value_kind_name(schema.kind)),
                 "null", "Remove null or declare the field nullable.");
  }
  if (schema.kind != ValueKind::dictionary) {
    validate_value(value, schema, path, limits);
    return;
  }
  if (!value.isGenericDict()) {
    ThrowInvalid(std::string(path), "dictionary", "another value kind",
                 "Match defaults to the declared ValueSchema kind.");
  }
  for (const auto& entry : value.toGenericDict()) {
    if (!entry.key().isString()) {
      ThrowInvalid(std::string(path), "string dictionary keys",
                   "non-string key", "Use JSON-compatible string keys.");
    }
    const std::string key(entry.key().toStringRef());
    const auto iterator = schema.fields.find(key);
    if (iterator == schema.fields.end()) {
      ThrowInvalid(ChildPath(path, key), "a declared schema field", key,
                   "Remove the unknown default field.");
    }
    ValidatePartialValue(entry.value(), iterator->second, ChildPath(path, key),
                         limits);
  }
}

struct JsonBudget {
  const ConfigLimits& limits;
  std::size_t nodes = 0;
  std::size_t string_bytes = 0;
};

void ValidateJsonBudget(const Json& value, std::string_view path,
                        std::size_t depth, JsonBudget& budget) {
  if (depth > budget.limits.max_depth) {
    ThrowLimit(std::string(path) + ".depth",
               "at most " + std::to_string(budget.limits.max_depth),
               std::to_string(depth));
  }
  CheckedAdd(budget.nodes, 1, budget.limits.max_nodes,
             std::string(path) + ".nodes");
  if (value.is_string()) {
    const auto& text = value.get_ref<const std::string&>();
    if (text.size() > budget.limits.max_string_bytes) {
      ThrowLimit(std::string(path) + ".string",
                 "at most " + std::to_string(budget.limits.max_string_bytes) +
                     " bytes",
                 std::to_string(text.size()) + " bytes");
    }
    ValidateUtf8(text, path);
    CheckedAdd(budget.string_bytes, text.size(),
               budget.limits.max_total_string_bytes,
               std::string(path) + ".string_bytes");
  } else if (value.is_object()) {
    for (const auto& [key, child] : value.items()) {
      if (key.size() > budget.limits.max_string_bytes) {
        ThrowLimit(std::string(path) + ".key",
                   "at most " + std::to_string(budget.limits.max_string_bytes) +
                       " bytes",
                   std::to_string(key.size()) + " bytes");
      }
      ValidateUtf8(key, std::string(path) + ".key");
      CheckedAdd(budget.string_bytes, key.size(),
                 budget.limits.max_total_string_bytes,
                 std::string(path) + ".string_bytes");
      ValidateJsonBudget(child, ChildPath(path, key), depth + 1, budget);
    }
  } else if (value.is_array()) {
    for (std::size_t index = 0; index < value.size(); ++index) {
      ValidateJsonBudget(value[index], ChildPath(path, std::to_string(index)),
                         depth + 1, budget);
    }
  }
  if (value.is_number_float() && !std::isfinite(value.get<double>())) {
    ThrowInvalid(std::string(path), "a finite JSON number", "NaN or infinity",
                 "Use a finite configuration value.");
  }
}

void ValidateJsonBudget(const Json& value, const ConfigLimits& limits,
                        std::string_view path = "config") {
  JsonBudget budget{limits};
  ValidateJsonBudget(value, path, 0, budget);
}

Json ParseJson(std::string_view text, std::string_view source,
               const ConfigLimits& limits) {
  if (text.size() > limits.max_file_bytes) {
    ThrowLimit(std::string(source) + ".bytes",
               "at most " + std::to_string(limits.max_file_bytes),
               std::to_string(text.size()));
  }
  ValidateUtf8(text, source);
  std::vector<std::set<std::string>> object_keys;
  auto callback = [&object_keys, source](int, Json::parse_event_t event,
                                         Json& parsed) {
    if (event == Json::parse_event_t::object_start) {
      object_keys.emplace_back();
    } else if (event == Json::parse_event_t::key) {
      const auto key = parsed.get<std::string>();
      if (object_keys.empty() || !object_keys.back().insert(key).second) {
        ThrowSerialization(std::string(source), "unique object keys", key,
                           "Remove the duplicate JSON field.");
      }
    } else if (event == Json::parse_event_t::object_end) {
      if (!object_keys.empty()) object_keys.pop_back();
    }
    return true;
  };
  try {
    Json parsed = Json::parse(text.begin(), text.end(), callback, true, false);
    ValidateJsonBudget(parsed, limits, source);
    return parsed;
  } catch (const Error&) {
    throw;
  } catch (const nlohmann::json::parse_error& error) {
    ThrowSerialization(
        std::string(source), "strict RFC 8259 JSON",
        "parser error " + std::to_string(error.id) + " at byte " +
            std::to_string(error.byte),
        "Fix JSON syntax, encoding, number range, or duplicates.");
  } catch (const nlohmann::json::exception& error) {
    ThrowSerialization(std::string(source), "strict RFC 8259 JSON",
                       "parser error " + std::to_string(error.id),
                       "Fix JSON syntax, encoding, number range, or type.");
  }
}

Json ReadJsonFile(const std::filesystem::path& path,
                  const ConfigLimits& limits) {
  const auto display_path = PathText(path);
  std::error_code status_error;
  if (!std::filesystem::is_regular_file(path, status_error) || status_error) {
    ThrowIo(display_path,
            status_error ? status_error.message() : "not a regular file");
  }
  const auto file_size = std::filesystem::file_size(path, status_error);
  if (status_error) ThrowIo(display_path, status_error.message());
  if (file_size > limits.max_file_bytes) {
    ThrowLimit(display_path + ".bytes",
               "at most " + std::to_string(limits.max_file_bytes),
               std::to_string(file_size));
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) ThrowIo(display_path, "open failed");
  std::string content(static_cast<std::size_t>(file_size), '\0');
  if (!content.empty()) {
    input.read(content.data(), static_cast<std::streamsize>(content.size()));
  }
  if (!input || input.peek() != std::ifstream::traits_type::eof()) {
    ThrowIo(display_path, "short read or file changed during read");
  }
  return ParseJson(content, display_path, limits);
}

c10::IValue JsonToValue(const Json& value, const ValueSchema& schema,
                        std::string_view path) {
  if (value.is_null()) {
    if (schema.nullable || schema.kind == ValueKind::none) return c10::IValue();
    ThrowInvalid(std::string(path), std::string(value_kind_name(schema.kind)),
                 "null", "Remove null or declare the field nullable.");
  }
  switch (schema.kind) {
    case ValueKind::integer: {
      if (!value.is_number_integer() && !value.is_number_unsigned()) {
        ThrowInvalid(std::string(path), "integer", value.type_name(),
                     "Use an integral JSON number.");
      }
      if (value.is_number_unsigned()) {
        const auto number = value.get<std::uint64_t>();
        if (number > static_cast<std::uint64_t>(
                         std::numeric_limits<std::int64_t>::max())) {
          ThrowInvalid(std::string(path), "signed 64-bit integer",
                       std::to_string(number), "Reduce the integer value.");
        }
        return c10::IValue(static_cast<std::int64_t>(number));
      }
      return c10::IValue(value.get<std::int64_t>());
    }
    case ValueKind::floating:
      if (!value.is_number_float()) {
        ThrowInvalid(std::string(path), "floating-point JSON number",
                     value.type_name(), "Use a number containing a decimal.");
      }
      return c10::IValue(value.get<double>());
    case ValueKind::boolean:
      if (!value.is_boolean()) {
        ThrowInvalid(std::string(path), "boolean", value.type_name(),
                     "Use true or false without quotes.");
      }
      return c10::IValue(value.get<bool>());
    case ValueKind::string:
      if (!value.is_string()) {
        ThrowInvalid(std::string(path), "string", value.type_name(),
                     "Use a JSON string value.");
      }
      return c10::IValue(value.get<std::string>());
    case ValueKind::none:
      ThrowInvalid(std::string(path), "null", value.type_name(),
                   "Use JSON null for a none field.");
    case ValueKind::list: {
      if (!value.is_array()) {
        ThrowInvalid(std::string(path), "array", value.type_name(),
                     "Use a JSON array.");
      }
      c10::impl::GenericList output(c10::AnyType::get());
      output.reserve(value.size());
      for (std::size_t index = 0; index < value.size(); ++index) {
        output.push_back(JsonToValue(value[index], schema.elements.front(),
                                     ChildPath(path, std::to_string(index))));
      }
      return c10::IValue(output);
    }
    case ValueKind::tuple: {
      if (!value.is_array() || value.size() != schema.elements.size()) {
        ThrowInvalid(std::string(path),
                     "array with " + std::to_string(schema.elements.size()) +
                         " elements",
                     value.is_array()
                         ? std::to_string(value.size()) + " elements"
                         : value.type_name(),
                     "Match the declared tuple length.");
      }
      std::vector<c10::IValue> output;
      output.reserve(value.size());
      for (std::size_t index = 0; index < value.size(); ++index) {
        output.push_back(JsonToValue(value[index], schema.elements[index],
                                     ChildPath(path, std::to_string(index))));
      }
      return c10::IValue(c10::ivalue::Tuple::create(std::move(output)));
    }
    case ValueKind::dictionary: {
      if (!value.is_object()) {
        ThrowInvalid(std::string(path), "object", value.type_name(),
                     "Use a JSON object.");
      }
      c10::impl::GenericDict output(c10::StringType::get(),
                                    c10::AnyType::get());
      for (const auto& [key, child] : value.items()) {
        const auto iterator = schema.fields.find(key);
        if (iterator == schema.fields.end()) {
          ThrowInvalid(ChildPath(path, key), "a declared config field", key,
                       "Remove the unknown field or add it to the schema.");
        }
        output.insert(
            key, JsonToValue(child, iterator->second, ChildPath(path, key)));
      }
      return c10::IValue(output);
    }
    case ValueKind::any:
    case ValueKind::tensor:
    case ValueKind::device:
      break;
  }
  ThrowInvalid(std::string(path), "a config-compatible schema kind",
               std::string(value_kind_name(schema.kind)),
               "Use the schema kinds allowed by validate_config_schema().");
}

Json ValueToJson(const c10::IValue& value, const ValueSchema& schema,
                 std::string_view path) {
  if (value.isNone()) return nullptr;
  switch (schema.kind) {
    case ValueKind::integer:
      return value.toInt();
    case ValueKind::floating:
      return value.toDouble();
    case ValueKind::boolean:
      return value.toBool();
    case ValueKind::string:
      return std::string(value.toStringRef());
    case ValueKind::none:
      return nullptr;
    case ValueKind::list: {
      Json output = Json::array();
      const auto values = value.toListRef();
      for (std::size_t index = 0; index < values.size(); ++index) {
        output.push_back(ValueToJson(values[index], schema.elements.front(),
                                     ChildPath(path, std::to_string(index))));
      }
      return output;
    }
    case ValueKind::tuple: {
      Json output = Json::array();
      const auto& values = value.toTupleRef().elements();
      for (std::size_t index = 0; index < values.size(); ++index) {
        output.push_back(ValueToJson(values[index], schema.elements[index],
                                     ChildPath(path, std::to_string(index))));
      }
      return output;
    }
    case ValueKind::dictionary: {
      Json output = Json::object();
      for (const auto& entry : value.toGenericDict()) {
        const std::string key(entry.key().toStringRef());
        const auto iterator = schema.fields.find(key);
        if (iterator == schema.fields.end()) {
          ThrowInvalid(ChildPath(path, key), "a declared config field", key,
                       "Remove the unknown field.");
        }
        output[key] =
            ValueToJson(entry.value(), iterator->second, ChildPath(path, key));
      }
      return output;
    }
    case ValueKind::any:
    case ValueKind::tensor:
    case ValueKind::device:
      break;
  }
  ThrowInvalid(std::string(path), "a config-compatible typed value",
               std::string(value_kind_name(schema.kind)),
               "Use the validated config schema.");
}

const Json& JsonAtTokens(const Json& root,
                         const std::vector<std::string>& tokens,
                         std::string_view original_path) {
  const Json* current = &root;
  for (const auto& token : tokens) {
    if (current->is_object()) {
      const auto iterator = current->find(token);
      if (iterator == current->end()) {
        ThrowInvalid(std::string(original_path), "an existing config path",
                     token, "Resolve or default the selected field first.");
      }
      current = &*iterator;
    } else if (current->is_array()) {
      const auto index = ParseArrayIndex(token, original_path);
      if (index >= current->size()) {
        ThrowInvalid(std::string(original_path),
                     "an array index below " + std::to_string(current->size()),
                     std::to_string(index), "Select an existing element.");
      }
      current = &(*current)[index];
    } else {
      ThrowInvalid(std::string(original_path), "a container path segment",
                   token, "Stop the pointer at the scalar value.");
    }
  }
  return *current;
}

const Json* TryJsonAtTokens(const Json& root,
                            const std::vector<std::string>& tokens) {
  const Json* current = &root;
  for (const auto& token : tokens) {
    if (current->is_object()) {
      const auto iterator = current->find(token);
      if (iterator == current->end()) return nullptr;
      current = &*iterator;
    } else if (current->is_array()) {
      std::size_t index = 0;
      const auto result =
          std::from_chars(token.data(), token.data() + token.size(), index);
      if (result.ec != std::errc{} ||
          result.ptr != token.data() + token.size() ||
          index >= current->size()) {
        return nullptr;
      }
      current = &(*current)[index];
    } else {
      return nullptr;
    }
  }
  return current;
}

void SetJsonAtTokens(Json& current, const ValueSchema& schema,
                     const std::vector<std::string>& tokens,
                     std::size_t position, Json value,
                     std::string_view original_path) {
  if (position == tokens.size()) {
    current = std::move(value);
    return;
  }
  const auto& token = tokens[position];
  if (schema.kind == ValueKind::dictionary) {
    if (!current.is_object()) {
      if (current.is_null()) {
        current = Json::object();
      } else {
        ThrowInvalid(std::string(original_path), "an object parent",
                     current.type_name(), "Supply a compatible parent value.");
      }
    }
    const auto iterator = schema.fields.find(token);
    if (iterator == schema.fields.end()) {
      ThrowInvalid(std::string(original_path), "a declared config field", token,
                   "Remove the unknown override.");
    }
    if (position + 1 == tokens.size()) {
      current[token] = std::move(value);
      return;
    }
    if (!current.contains(token)) {
      if (iterator->second.kind == ValueKind::dictionary) {
        current[token] = Json::object();
      } else {
        ThrowInvalid(std::string(original_path), "an existing container parent",
                     token,
                     "Default or supply the list/tuple before overriding an "
                     "element.");
      }
    }
    SetJsonAtTokens(current[token], iterator->second, tokens, position + 1,
                    std::move(value), original_path);
    return;
  }
  if (schema.kind == ValueKind::list || schema.kind == ValueKind::tuple) {
    if (!current.is_array()) {
      ThrowInvalid(std::string(original_path), "an existing array parent",
                   current.type_name(), "Supply the complete array first.");
    }
    const auto index = ParseArrayIndex(token, original_path);
    if (index >= current.size()) {
      ThrowInvalid(std::string(original_path),
                   "an array index below " + std::to_string(current.size()),
                   std::to_string(index), "Select an existing array element.");
    }
    const auto& child_schema = schema.kind == ValueKind::list
                                   ? schema.elements.front()
                                   : schema.elements[index];
    SetJsonAtTokens(current[index], child_schema, tokens, position + 1,
                    std::move(value), original_path);
    return;
  }
  ThrowInvalid(std::string(original_path), "a container parent", token,
               "Stop the override path at the scalar field.");
}

void EraseOriginDescendants(
    std::map<std::string, std::vector<ConfigOrigin>>& origins,
    std::string_view path) {
  const std::string prefix = std::string(path) + "/";
  auto iterator = origins.lower_bound(prefix);
  while (iterator != origins.end() && iterator->first.starts_with(prefix)) {
    iterator = origins.erase(iterator);
  }
}

void RecordOriginTree(
    const Json& value, std::string_view path, const ConfigOrigin& origin,
    std::map<std::string, std::vector<ConfigOrigin>>& origins) {
  origins[std::string(path)].push_back(origin);
  if (value.is_object()) {
    for (const auto& [key, child] : value.items()) {
      RecordOriginTree(child, ChildPath(path, key), origin, origins);
    }
  } else if (value.is_array()) {
    for (std::size_t index = 0; index < value.size(); ++index) {
      RecordOriginTree(value[index], ChildPath(path, std::to_string(index)),
                       origin, origins);
    }
  }
}

void MergeJson(const Json& overlay, const ValueSchema& schema,
               std::string_view path, const ConfigOrigin& origin, Json& target,
               std::map<std::string, std::vector<ConfigOrigin>>& origins) {
  if (overlay.is_object() && schema.kind == ValueKind::dictionary) {
    if (!target.is_object()) target = Json::object();
    origins[std::string(path)].push_back(origin);
    for (const auto& [key, child] : overlay.items()) {
      const auto schema_iterator = schema.fields.find(key);
      if (schema_iterator == schema.fields.end()) {
        ThrowInvalid(ChildPath(path, key), "a declared config field", key,
                     "Remove the unknown JSON field.");
      }
      const auto child_path = ChildPath(path, key);
      if (child.is_object() &&
          schema_iterator->second.kind == ValueKind::dictionary) {
        if (!target.contains(key)) target[key] = Json::object();
        MergeJson(child, schema_iterator->second, child_path, origin,
                  target[key], origins);
      } else {
        target[key] = child;
        EraseOriginDescendants(origins, child_path);
        RecordOriginTree(child, child_path, origin, origins);
      }
    }
    return;
  }
  target = overlay;
  EraseOriginDescendants(origins, path);
  RecordOriginTree(target, path, origin, origins);
}

Json ParseEnvironmentValue(std::string_view text, const ValueSchema& schema,
                           std::string_view source,
                           const ConfigLimits& limits) {
  if (text.size() > limits.max_string_bytes) {
    ThrowLimit(std::string(source),
               "at most " + std::to_string(limits.max_string_bytes) + " bytes",
               std::to_string(text.size()) + " bytes");
  }
  ValidateUtf8(text, source);
  if (schema.kind == ValueKind::string) {
    Json value = std::string(text);
    ValidateJsonBudget(value, limits, source);
    return value;
  }
  return ParseJson(text, source, limits);
}

void ValidateSecrets(const Json& value, const ConfigSchema& schema) {
  for (const auto& path : schema.secret_paths) {
    const auto tokens = DecodePointer(path);
    const Json* selected = TryJsonAtTokens(value, tokens);
    if (selected == nullptr) continue;
    if (!selected->is_string() ||
        !is_secret_reference(selected->get_ref<const std::string&>())) {
      ThrowInvalid(path, "secret://provider/scope/name?version=... reference",
                   selected->is_string() ? "non-reference secret text"
                                         : selected->type_name(),
                   "Store only a secret reference; resolve the value at its "
                   "point of use.");
    }
  }
}

bool IsSecretPath(const ConfigSchema& schema, std::string_view path) {
  return std::find(schema.secret_paths.begin(), schema.secret_paths.end(),
                   path) != schema.secret_paths.end();
}

bool ContainsPresentSecret(const Json& value, const ConfigSchema& schema,
                           std::string_view path) {
  const std::string descendant_prefix =
      path.empty() ? "/" : std::string(path) + "/";
  return std::any_of(schema.secret_paths.begin(), schema.secret_paths.end(),
                     [&](const std::string& secret_path) {
                       return (secret_path == path ||
                               secret_path.starts_with(descendant_prefix)) &&
                              TryJsonAtTokens(
                                  value, DecodePointer(secret_path)) != nullptr;
                     });
}

Json RedactedJson(Json value, const ConfigSchema& schema) {
  for (const auto& path : schema.secret_paths) {
    const auto tokens = DecodePointer(path);
    if (TryJsonAtTokens(value, tokens) != nullptr) {
      SetJsonAtTokens(value, schema.root, tokens, 0, Json(kRedacted), path);
    }
  }
  return value;
}

std::string CompactJson(const Json& value) {
  return value.dump(-1, ' ', false, Json::error_handler_t::strict);
}

}  // namespace

std::string_view config_source_kind_name(ConfigSourceKind kind) noexcept {
  switch (kind) {
    case ConfigSourceKind::schema_default:
      return "schema_default";
    case ConfigSourceKind::json_file:
      return "json_file";
    case ConfigSourceKind::environment:
      return "environment";
    case ConfigSourceKind::command_line:
      return "command_line";
  }
  return "unknown";
}

bool is_secret_reference(std::string_view reference) noexcept {
  constexpr std::string_view kPrefix = "secret://";
  if (!reference.starts_with(kPrefix) || reference.size() > 4096U) return false;
  reference.remove_prefix(kPrefix.size());
  const auto query = reference.find('?');
  const auto path = reference.substr(0, query);
  const auto first = path.find('/');
  const auto second = first == std::string_view::npos
                          ? std::string_view::npos
                          : path.find('/', first + 1);
  if (first == std::string_view::npos || second == std::string_view::npos ||
      path.find('/', second + 1) != std::string_view::npos || first == 0 ||
      second == first + 1 || second + 1 == path.size()) {
    return false;
  }
  auto valid_token = [](std::string_view token) {
    return std::all_of(token.begin(), token.end(), [](char value) {
      const auto byte = static_cast<unsigned char>(value);
      return (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') ||
             (byte >= '0' && byte <= '9') || value == '.' || value == '_' ||
             value == '-';
    });
  };
  if (!valid_token(path.substr(0, first)) ||
      !valid_token(path.substr(first + 1, second - first - 1)) ||
      !valid_token(path.substr(second + 1))) {
    return false;
  }
  if (query == std::string_view::npos) return true;
  constexpr std::string_view kVersion = "?version=";
  const auto suffix = reference.substr(query);
  return suffix.starts_with(kVersion) && suffix.size() > kVersion.size() &&
         valid_token(suffix.substr(kVersion.size()));
}

void validate_config_schema(const ConfigSchema& schema,
                            const ConfigLimits& limits) {
  ValidateLimits(limits);
  if (!IsIdentifier(schema.schema_id)) {
    ThrowInvalid("schema_id", "1-128 ASCII identifier characters",
                 schema.schema_id,
                 "Use letters, digits, '.', '_', '-', or '/'.");
  }
  if (schema.version == 0) {
    ThrowInvalid("schema.version", "a positive version", "0",
                 "Start config schemas at version 1.");
  }
  validate_value_schema(schema.root, "config", limits.value_schema_limits);
  if (schema.root.kind != ValueKind::dictionary) {
    ThrowInvalid("config", "a dictionary root",
                 std::string(value_kind_name(schema.root.kind)),
                 "Declare the config root as a closed dictionary.");
  }
  ValidateSchemaNode(schema.root, "config");
  if (!schema.defaults.isNone()) {
    ValidatePartialValue(schema.defaults, schema.root, "defaults",
                         limits.value_schema_limits);
    const Json defaults = ValueToJson(schema.defaults, schema.root, "defaults");
    ValidateJsonBudget(defaults, limits, "defaults");
    ValidateSecrets(defaults, schema);
  }

  std::set<std::string> variables;
  for (const auto& [path, variable] : schema.environment_bindings) {
    const auto canonical = CanonicalPointer(DecodePointer(path));
    if (canonical != path || path.empty()) {
      ThrowInvalid("environment_bindings", "canonical non-root JSON Pointers",
                   path, "Use an exact declared field path.");
    }
    (void)SchemaAtPointer(schema.root, path);
    if (!IsIdentifier(variable) || !variables.insert(variable).second) {
      ThrowInvalid(path, "a unique allowlisted environment variable", variable,
                   "Use one stable variable name per field.");
    }
  }

  std::set<std::string> secret_paths;
  for (const auto& path : schema.secret_paths) {
    const auto canonical = CanonicalPointer(DecodePointer(path));
    const auto& selected = SchemaAtPointer(schema.root, path);
    if (canonical != path || path.empty() ||
        !secret_paths.insert(path).second ||
        selected.kind != ValueKind::string) {
      ThrowInvalid("secret_paths", "unique string-field JSON Pointers", path,
                   "Mark each secret reference string field exactly once.");
    }
  }
}

CapabilityDescriptor make_config_capability(const ConfigSchema& schema,
                                            const ConfigLimits& limits) {
  validate_config_schema(schema, limits);
  CapabilityDescriptor descriptor{
      .feature_id = std::string(kConfigCapabilityFeatureId),
      .scope_id = schema.schema_id,
      .feature_version = 1U,
      .status = CapabilityStatus::experimental,
      .implementation = CapabilityImplementation::native,
      .source = CapabilitySource::configured_instance,
      .backend = "nlohmann-json",
      .devices = {},
      .dtypes = {},
      .schemas = {schema.schema_id + "/v" + std::to_string(schema.version)},
      .operations = {"canonicalize", "explain", "resolve_cli", "resolve_env",
                     "resolve_json"},
      .limits =
          {{.name = "max_cli_overrides",
            .value = limits.max_cli_overrides,
            .unit = "count"},
           {.name = "max_depth", .value = limits.max_depth, .unit = "count"},
           {.name = "max_file_bytes",
            .value = limits.max_file_bytes,
            .unit = "bytes"},
           {.name = "max_json_files",
            .value = limits.max_json_files,
            .unit = "count"},
           {.name = "max_nodes", .value = limits.max_nodes, .unit = "count"},
           {.name = "max_string_bytes",
            .value = limits.max_string_bytes,
            .unit = "bytes"},
           {.name = "max_total_string_bytes",
            .value = limits.max_total_string_bytes,
            .unit = "bytes"}},
      .limitations = {
          "json-only first slice; no config migration or secret resolution"}};
  validate_capability_descriptor(descriptor);
  return descriptor;
}

ResolvedConfig::ResolvedConfig(
    ConfigSchema schema, c10::IValue value,
    std::map<std::string, std::vector<ConfigOrigin>> origins,
    ConfigLimits limits)
    : schema_(std::move(schema)),
      value_(std::move(value)),
      origins_(std::move(origins)),
      limits_(limits) {}

c10::IValue ResolvedConfig::value() const {
  const Json json = ValueToJson(value_, schema_.root, "config");
  return JsonToValue(json, schema_.root, "config");
}

c10::IValue ResolvedConfig::at(std::string_view path) const {
  const auto tokens = DecodePointer(path);
  const auto canonical = CanonicalPointer(tokens);
  const auto& selected_schema = SchemaAtTokens(schema_.root, tokens, path);
  const Json root = ValueToJson(value_, schema_.root, "config");
  const auto& selected = JsonAtTokens(root, tokens, path);
  return JsonToValue(selected, selected_schema,
                     canonical.empty() ? "config" : canonical);
}

std::int64_t ResolvedConfig::integer(std::string_view path) const {
  const auto selected = at(path);
  if (!selected.isInt()) {
    ThrowInvalid(std::string(path), "integer", "another value kind",
                 "Use the accessor matching the field schema.");
  }
  return selected.toInt();
}

double ResolvedConfig::floating(std::string_view path) const {
  const auto selected = at(path);
  if (!selected.isDouble()) {
    ThrowInvalid(std::string(path), "floating", "another value kind",
                 "Use the accessor matching the field schema.");
  }
  return selected.toDouble();
}

bool ResolvedConfig::boolean(std::string_view path) const {
  const auto selected = at(path);
  if (!selected.isBool()) {
    ThrowInvalid(std::string(path), "boolean", "another value kind",
                 "Use the accessor matching the field schema.");
  }
  return selected.toBool();
}

std::string ResolvedConfig::string(std::string_view path) const {
  const auto selected = at(path);
  if (!selected.isString()) {
    ThrowInvalid(std::string(path), "string", "another value kind",
                 "Use the accessor matching the field schema.");
  }
  return std::string(selected.toStringRef());
}

ConfigSchema ResolvedConfig::schema() const {
  ConfigSchema copy = schema_;
  if (!copy.defaults.isNone()) {
    const Json defaults = ValueToJson(copy.defaults, copy.root, "defaults");
    copy.defaults = JsonToValue(defaults, copy.root, "defaults");
  }
  return copy;
}

std::vector<ConfigOrigin> ResolvedConfig::origins(std::string_view path) const {
  const auto tokens = DecodePointer(path);
  const auto canonical = CanonicalPointer(tokens);
  (void)SchemaAtTokens(schema_.root, tokens, path);
  const Json root = ValueToJson(value_, schema_.root, "config");
  (void)JsonAtTokens(root, tokens, path);
  const auto iterator = origins_.find(canonical);
  if (iterator == origins_.end()) return {};
  return iterator->second;
}

ConfigExplanation ResolvedConfig::explain(std::string_view path) const {
  const auto tokens = DecodePointer(path);
  const auto canonical = CanonicalPointer(tokens);
  const auto& selected_schema = SchemaAtTokens(schema_.root, tokens, path);
  const Json root = ValueToJson(value_, schema_.root, "config");
  const auto& selected = JsonAtTokens(root, tokens, path);
  const bool exact_secret = IsSecretPath(schema_, canonical);
  const bool redacted = ContainsPresentSecret(root, schema_, canonical);
  Json redacted_root;
  const Json* display = &selected;
  if (redacted && !exact_secret) {
    redacted_root = RedactedJson(root, schema_);
    display = &JsonAtTokens(redacted_root, tokens, path);
  }
  return {.path = canonical,
          .type = std::string(value_kind_name(selected_schema.kind)),
          .display_value =
              exact_secret ? std::string(kRedacted) : CompactJson(*display),
          .redacted = redacted,
          .origins = origins(canonical)};
}

std::string ResolvedConfig::explain_json(std::string_view path) const {
  const auto explanation = explain(path);
  Json origins = Json::array();
  for (const auto& origin : explanation.origins) {
    origins.push_back({{"kind", config_source_kind_name(origin.kind)},
                       {"source", origin.source}});
  }
  const Json output{{"origins", std::move(origins)},
                    {"path", explanation.path},
                    {"redacted", explanation.redacted},
                    {"type", explanation.type},
                    {"value", explanation.display_value}};
  return CompactJson(output);
}

std::string ResolvedConfig::canonical_json() const {
  Json output = ValueToJson(value_, schema_.root, "config");
  output = RedactedJson(std::move(output), schema_);
  ValidateJsonBudget(output, limits_, "config");
  return CompactJson(output);
}

ConfigResolver::ConfigResolver(ConfigSchema schema, ConfigLimits limits)
    : schema_(std::move(schema)), limits_(limits) {
  validate_config_schema(schema_, limits_);
  if (!schema_.defaults.isNone()) {
    const Json defaults =
        ValueToJson(schema_.defaults, schema_.root, "defaults");
    schema_.defaults = JsonToValue(defaults, schema_.root, "defaults");
  }
}

void ConfigResolver::add_json_file(std::filesystem::path path) {
  if (path.empty()) {
    ThrowInvalid("json_file", "a non-empty path", "empty path",
                 "Select a local JSON file.");
  }
  if (json_files_.size() >= limits_.max_json_files) {
    ThrowLimit("json_files",
               "at most " + std::to_string(limits_.max_json_files),
               std::to_string(json_files_.size() + 1));
  }
  json_files_.push_back(std::move(path));
}

void ConfigResolver::add_environment(
    std::map<std::string, std::string> values) {
  if (environment_) {
    ThrowInvalid("environment", "one immutable environment snapshot",
                 "multiple snapshots",
                 "Capture or inject environment variables exactly once.");
  }
  std::size_t total_bytes = 0;
  for (const auto& [name, value] : values) {
    ValidateUtf8(name, "environment.name");
    ValidateUtf8(value, name);
    if (name.size() > limits_.max_string_bytes ||
        value.size() > limits_.max_string_bytes) {
      ThrowLimit(name,
                 "name and value at most " +
                     std::to_string(limits_.max_string_bytes) + " bytes",
                 std::to_string(name.size() + value.size()) + " bytes");
    }
    CheckedAdd(total_bytes, name.size(), limits_.max_total_string_bytes,
               "environment.string_bytes");
    CheckedAdd(total_bytes, value.size(), limits_.max_total_string_bytes,
               "environment.string_bytes");
  }
  environment_ = std::move(values);
}

void ConfigResolver::add_process_environment() {
  if (environment_) {
    ThrowInvalid("environment", "one immutable environment snapshot",
                 "multiple snapshots",
                 "Capture or inject environment variables exactly once.");
  }
  std::map<std::string, std::string> captured;
  for (const auto& [path, variable] : schema_.environment_bindings) {
    (void)path;
    if (auto value = ReadProcessEnvironmentVariable(variable); value) {
      captured.emplace(variable, std::move(*value));
    }
  }
  add_environment(std::move(captured));
}

void ConfigResolver::add_cli(std::span<const std::string_view> arguments) {
  for (std::size_t index = 0; index < arguments.size(); ++index) {
    std::string_view assignment;
    if (arguments[index] == "--set") {
      if (++index >= arguments.size()) {
        ThrowInvalid("argv", "--set followed by /path=<json>", "missing value",
                     "Provide the override assignment as the next argv token.");
      }
      assignment = arguments[index];
    } else if (arguments[index].starts_with("--set=")) {
      assignment = arguments[index].substr(6);
    } else {
      ThrowInvalid("argv[" + std::to_string(index) + "]",
                   "--set /path=<json> or --set=/path=<json>",
                   std::string(arguments[index]),
                   "Pass only structured config override arguments.");
    }
    const auto separator = assignment.find('=');
    if (separator == std::string_view::npos || separator == 0 ||
        separator + 1 == assignment.size()) {
      ThrowInvalid("argv[" + std::to_string(index) + "]", "/path=<json>",
                   std::string(assignment),
                   "Separate the JSON Pointer and JSON value with '='.");
    }
    add_cli_override(std::string(assignment.substr(0, separator)),
                     std::string(assignment.substr(separator + 1)));
  }
}

void ConfigResolver::add_cli_override(std::string path,
                                      std::string json_value) {
  if (cli_overrides_.size() >= limits_.max_cli_overrides) {
    ThrowLimit("cli_overrides",
               "at most " + std::to_string(limits_.max_cli_overrides),
               std::to_string(cli_overrides_.size() + 1));
  }
  if (path.empty()) {
    ThrowInvalid("cli_override", "a non-root JSON Pointer", "empty path",
                 "Override a declared field rather than the entire config.");
  }
  const auto canonical = CanonicalPointer(DecodePointer(path));
  if (canonical != path) {
    ThrowInvalid("cli_override", "a canonical JSON Pointer", path,
                 "Use RFC 6901 escapes and canonical array indices.");
  }
  (void)SchemaAtPointer(schema_.root, path);
  if (json_value.size() > limits_.max_file_bytes) {
    ThrowLimit(path,
               "at most " + std::to_string(limits_.max_file_bytes) + " bytes",
               std::to_string(json_value.size()) + " bytes");
  }
  ValidateUtf8(json_value, path);
  cli_overrides_.emplace_back(std::move(path), std::move(json_value));
}

ResolvedConfig ConfigResolver::resolve() const {
  Json resolved = Json::object();
  std::map<std::string, std::vector<ConfigOrigin>> origins;
  if (!schema_.defaults.isNone()) {
    resolved = ValueToJson(schema_.defaults, schema_.root, "defaults");
    RecordOriginTree(
        resolved, "",
        {.kind = ConfigSourceKind::schema_default,
         .source = schema_.schema_id + "/v" + std::to_string(schema_.version)},
        origins);
  }

  for (const auto& path : json_files_) {
    const auto display_path = PathText(path);
    const Json source = ReadJsonFile(path, limits_);
    const auto typed = JsonToValue(source, schema_.root, display_path);
    (void)typed;
    MergeJson(source, schema_.root, "",
              {.kind = ConfigSourceKind::json_file, .source = display_path},
              resolved, origins);
  }

  if (environment_) {
    for (const auto& [path, variable] : schema_.environment_bindings) {
      const auto iterator = environment_->find(variable);
      if (iterator == environment_->end()) continue;
      const auto tokens = DecodePointer(path);
      const auto& selected_schema = SchemaAtTokens(schema_.root, tokens, path);
      Json parsed = ParseEnvironmentValue(iterator->second, selected_schema,
                                          variable, limits_);
      (void)JsonToValue(parsed, selected_schema, path);
      SetJsonAtTokens(resolved, schema_.root, tokens, 0, parsed, path);
      EraseOriginDescendants(origins, path);
      RecordOriginTree(
          parsed, path,
          {.kind = ConfigSourceKind::environment, .source = variable}, origins);
    }
  }

  for (std::size_t index = 0; index < cli_overrides_.size(); ++index) {
    const auto& [path, text] = cli_overrides_[index];
    const auto tokens = DecodePointer(path);
    const auto& selected_schema = SchemaAtTokens(schema_.root, tokens, path);
    Json parsed =
        ParseJson(text, "argv[" + std::to_string(index) + "]", limits_);
    (void)JsonToValue(parsed, selected_schema, path);
    SetJsonAtTokens(resolved, schema_.root, tokens, 0, parsed, path);
    EraseOriginDescendants(origins, path);
    RecordOriginTree(parsed, path,
                     {.kind = ConfigSourceKind::command_line,
                      .source = "argv[" + std::to_string(index) + "]"},
                     origins);
  }

  ValidateJsonBudget(resolved, limits_, "config");
  ValidateSecrets(resolved, schema_);
  auto typed = JsonToValue(resolved, schema_.root, "config");
  validate_value(typed, schema_.root, "config", limits_.value_schema_limits);
  return ResolvedConfig(schema_, std::move(typed), std::move(origins), limits_);
}

}  // namespace jyppx::liuding::experiment::config
