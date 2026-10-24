// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "config_migration.hpp"

#include "migration_files.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <utility>

namespace jyppx::liuding::cli {
namespace {
namespace fs = std::filesystem;
using experiment::config::ConfigSchema;
using migration_files::kInputBytes;
using migration_files::sha256;
constexpr std::uint32_t kMaxVersion = std::numeric_limits<std::uint32_t>::max();

void tree_budget(const Json& value, std::size_t depth, std::size_t& nodes,
                 std::size_t& strings) {
  if (++nodes > 4096U || depth > 32U)
    throw ResourceExhausted("Migration config node/depth budget exceeded");
  if (value.is_string()) {
    const auto size = value.get_ref<const std::string&>().size();
    if (size > 65536U || size > kInputBytes - strings)
      throw ResourceExhausted("Migration config string budget exceeded");
    strings += size;
  }
  if (value.is_object() || value.is_array()) {
    for (auto iterator = value.begin(); iterator != value.end(); ++iterator) {
      if (value.is_object()) {
        if (++nodes > 4096U || iterator.key().size() > 65536U ||
            iterator.key().size() > kInputBytes - strings)
          throw ResourceExhausted("Migration config key budget exceeded");
        strings += iterator.key().size();
      }
      tree_budget(iterator.value(), depth + 1U, nodes, strings);
    }
  }
}

void tree_budget(const Json& value) {
  std::size_t nodes = 0U;
  std::size_t strings = 0U;
  tree_budget(value, 0U, nodes, strings);
}

void identifier(const Json& value) {
  if (!value.is_string())
    throw SerializationError("Migration identity must be text");
  const auto& name = value.get_ref<const std::string&>();
  if (name.empty() || name.size() > 128U ||
      !std::all_of(name.begin(), name.end(), [](unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
               (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
      }))
    throw InvalidArgument("Use bounded ASCII migration identifiers");
}

std::uint32_t version(const Json& value) {
  const auto result = unsigned_number(value, kMaxVersion);
  if (result == 0U) throw InvalidArgument("Migration version must be positive");
  return static_cast<std::uint32_t>(result);
}

std::int64_t integer(const Json& value) {
  if (value.is_number_unsigned()) {
    return static_cast<std::int64_t>(unsigned_number(
        value,
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())));
  }
  if (!value.is_number_integer())
    throw InvalidArgument("Exact unit conversion requires int64 values");
  return value.get<std::int64_t>();
}

std::vector<std::string> pointer(const Json& value) {
  if (!value.is_string())
    throw SerializationError("Migration path must be a JSON Pointer");
  const auto path = value.get<std::string>();
  if (path.empty() || path.front() != '/' || path.size() > 1024U)
    throw InvalidArgument("Use a bounded non-root JSON Pointer");
  std::vector<std::string> tokens;
  std::string token;
  for (std::size_t index = 1U; index <= path.size(); ++index) {
    if (index == path.size() || path[index] == '/') {
      if (token.empty() || tokens.size() == 32U)
        throw InvalidArgument("Migration pointer has an empty or deep token");
      tokens.push_back(std::move(token));
      token.clear();
    } else if (path[index] == '~') {
      if (++index >= path.size() || (path[index] != '0' && path[index] != '1'))
        throw InvalidArgument("Use canonical RFC6901 pointer escapes");
      token += path[index] == '0' ? '~' : '/';
    } else {
      const auto byte = static_cast<unsigned char>(path[index]);
      if (byte < 0x20U || byte == 0x7fU)
        throw InvalidArgument("Migration pointer must not contain controls");
      token += path[index];
    }
  }
  return tokens;
}

Json& parent(Json& value, const std::vector<std::string>& tokens) {
  auto* current = &value;
  for (std::size_t index = 0U; index + 1U < tokens.size(); ++index) {
    if (!current->is_object() || !current->contains(tokens[index]))
      throw CompatibilityError("Migration dictionary parent is missing");
    current = &current->at(tokens[index]);
  }
  if (!current->is_object())
    throw CompatibilityError("Migration operations require dictionary parents");
  return *current;
}

bool overlap(const std::vector<std::string>& left,
             const std::vector<std::string>& right) {
  return std::equal(left.begin(),
                    left.begin() + static_cast<std::ptrdiff_t>(
                                       std::min(left.size(), right.size())),
                    right.begin());
}

std::int64_t scaled(std::int64_t value, std::int64_t numerator,
                    std::int64_t denominator) {
  if (denominator <= 0)
    throw InvalidArgument("Unit conversion denominator must be positive");
  const auto magnitude = [](std::int64_t number) {
    return number < 0 ? static_cast<std::uint64_t>(-(number + 1)) + 1U
                      : static_cast<std::uint64_t>(number);
  };
  auto left = magnitude(value);
  auto right = magnitude(numerator);
  auto divisor = static_cast<std::uint64_t>(denominator);
  auto common = std::gcd(left, divisor);
  left /= common;
  divisor /= common;
  common = std::gcd(right, divisor);
  right /= common;
  divisor /= common;
  if (divisor != 1U)
    throw CompatibilityError("Unit conversion would round an integer");
  const bool negative = (value < 0) != (numerator < 0);
  const auto maximum =
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) +
      (negative ? 1U : 0U);
  if (left != 0U && right > maximum / left)
    throw ResourceExhausted("Unit conversion overflows int64");
  const auto product = left * right;
  if (!negative) return static_cast<std::int64_t>(product);
  if (product == maximum &&
      maximum !=
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    return std::numeric_limits<std::int64_t>::min();
  return -static_cast<std::int64_t>(product);
}

void validate_operation(const Json& operation) {
  if (!operation.is_object() || !operation.contains("kind"))
    throw SerializationError("Migration operation requires kind");
  const auto kind = operation.at("kind").get<std::string>();
  if (kind == "rename" || kind == "copy") {
    fields(operation, {"kind", "from", "path"});
    if (overlap(pointer(operation.at("from")), pointer(operation.at("path"))))
      throw InvalidArgument("Migration copy/rename paths must not overlap");
  } else if (kind == "set-default") {
    fields(operation, {"kind", "path", "value"});
  } else if (kind == "scale-int") {
    fields(operation, {"kind", "path", "numerator", "denominator"});
    static_cast<void>(integer(operation.at("numerator")));
    if (integer(operation.at("denominator")) <= 0)
      throw InvalidArgument("Unit conversion denominator must be positive");
  } else if (kind == "replace") {
    fields(operation, {"kind", "path", "expected", "value"});
  } else if (kind == "remove-checked") {
    fields(operation, {"kind", "path", "expected"});
  } else {
    throw CompatibilityError("Unsupported migration operation");
  }
  static_cast<void>(pointer(operation.at("path")));
}

void apply_operation(Json& value, const Json& operation) {
  const auto kind = operation.at("kind").get<std::string>();
  const auto target = pointer(operation.at("path"));
  auto& destination = parent(value, target);
  const auto& name = target.back();
  if (kind == "rename" || kind == "copy") {
    const auto source_tokens = pointer(operation.at("from"));
    auto& source = parent(value, source_tokens);
    if (!source.contains(source_tokens.back()) || destination.contains(name))
      throw CompatibilityError("Migration rename/copy precondition failed");
    Json copied = source.at(source_tokens.back());
    destination[name] = std::move(copied);
    if (kind == "rename") source.erase(source_tokens.back());
  } else if (kind == "set-default") {
    if (!destination.contains(name)) destination[name] = operation.at("value");
  } else if (kind == "scale-int") {
    if (!destination.contains(name))
      throw CompatibilityError("Migration conversion source is missing");
    destination[name] = scaled(integer(destination.at(name)),
                               integer(operation.at("numerator")),
                               integer(operation.at("denominator")));
  } else {
    if (!destination.contains(name) ||
        destination.at(name) != operation.at("expected"))
      throw CompatibilityError("Migration replacement precondition failed");
    if (kind == "remove-checked")
      destination.erase(name);
    else
      destination[name] = operation.at("value");
  }
}

struct Plan {
  Json document;
  std::map<std::uint32_t, ConfigSchema> schemas;
  std::map<std::pair<std::uint32_t, std::uint32_t>, Json> steps;
};

Plan read_plan(std::string_view bytes) {
  Plan plan{parse_json(bytes), {}, {}};
  const auto& root = plan.document;
  fields(root, {"schema_version", "plan_id", "feature_id", "schema_id",
                "schemas", "steps"});
  if (unsigned_number(root.at("schema_version"), 1U) != 1U)
    throw CompatibilityError("Unsupported migration plan schema");
  for (const auto field : {"plan_id", "feature_id", "schema_id"})
    identifier(root.at(field));
  if (!root.at("schemas").is_array() || root.at("schemas").empty() ||
      root.at("schemas").size() > 16U || !root.at("steps").is_array() ||
      root.at("steps").size() > 64U)
    throw ResourceExhausted("Migration schema/step count exceeds bounds");
  for (const auto& encoded : root.at("schemas")) {
    auto schema = decode_config_schema(encoded);
    if (schema.schema_id != root.at("schema_id").get<std::string>() ||
        !schema.defaults.isNone() || !schema.environment_bindings.empty() ||
        schema.root.fields.size() > 256U)
      throw CompatibilityError(
          "Migration schemas require one identity and explicit "
          "defaults/environment");
    if (!plan.schemas.emplace(schema.version, std::move(schema)).second)
      throw SerializationError("Duplicate migration schema version");
  }
  std::size_t count = 0U;
  for (const auto& step : root.at("steps")) {
    fields(step,
           {"from_version", "to_version", "breaking_change", "operations"});
    const auto from = version(step.at("from_version"));
    const auto to = version(step.at("to_version"));
    identifier(step.at("breaking_change"));
    if (!plan.schemas.contains(from) || !plan.schemas.contains(to) ||
        (from > to ? from - to : to - from) != 1U)
      throw CompatibilityError(
          "Migration steps require adjacent declared versions");
    const auto& operations = step.at("operations");
    if (!operations.is_array() || operations.size() > 256U - count)
      throw ResourceExhausted("Migration operations exceed 256");
    count += operations.size();
    for (const auto& operation : operations) validate_operation(operation);
    if (!plan.steps.emplace(std::pair{from, to}, step).second)
      throw SerializationError("Duplicate directed migration step");
  }
  return plan;
}

void validate_document(const Json& document, const Plan& plan) {
  fields(document, {"schema_version", "schema_id", "version", "values"});
  if (unsigned_number(document.at("schema_version"), 1U) != 1U ||
      document.at("schema_id") != plan.document.at("schema_id"))
    throw CompatibilityError("Config document identity does not match plan");
  const auto selected = plan.schemas.find(version(document.at("version")));
  if (selected == plan.schemas.end())
    throw CompatibilityError("Config document version has no declared schema");
  tree_budget(document.at("values"));
  validate_migration_values(document.at("values"), selected->second);
}

Json migrate(Json input, const Plan& plan, std::uint32_t target,
             Json& changes) {
  validate_document(input, plan);
  if (!plan.schemas.contains(target))
    throw CompatibilityError("Target config version has no declared schema");
  auto current = version(input.at("version"));
  while (current != target) {
    const auto next = current < target ? current + 1U : current - 1U;
    const auto found = plan.steps.find({current, next});
    if (found == plan.steps.end())
      throw CompatibilityError("No explicit directed migration step");
    const auto& step = found->second;
    for (const auto& operation : step.at("operations")) {
      apply_operation(input.at("values"), operation);
      tree_budget(input.at("values"));
      Json change{{"kind", operation.at("kind")},
                  {"path", operation.at("path")},
                  {"from_version", current},
                  {"to_version", next},
                  {"breaking_change", step.at("breaking_change")}};
      if (operation.contains("from")) change["from"] = operation.at("from");
      changes.push_back(std::move(change));
    }
    input["version"] = next;
    // Re-apply the shared codec budget before each intermediate typed schema.
    input = parse_json(input.dump());
    validate_document(input, plan);
    current = next;
  }
  return input;
}

std::string encoded(const Json& value) { return value.dump() + "\n"; }

Result command(const Arguments& arguments) {
  const auto mode = arguments.optional("mode").value_or("preview");
  if (mode == "rollback") {
    arguments.allow({"mode", "receipt", "plan", "output"});
    const auto folder = migration_files::checked_path(
        migration_files::path(arguments.require("receipt")));
    const auto output_directory =
        fs::absolute(migration_files::path(arguments.require("output")))
            .lexically_normal();
    if (output_directory == folder ||
        std::mismatch(folder.begin(), folder.end(), output_directory.begin(),
                      output_directory.end())
                .first == folder.end())
      throw InvalidArgument(
          "Rollback output must be outside the receipt directory");
    std::set<std::string> actual;
    for (const auto& entry : fs::directory_iterator(folder))
      actual.insert(entry.path().filename().string());
    if (actual !=
        std::set<std::string>{"original.json", "migrated.json", "receipt.json"})
      throw SerializationError(
          "Rollback receipt directory contains unknown/missing files");
    const auto receipt =
        parse_json(migration_files::read(folder / "receipt.json"));
    fields(receipt, {"schema_version", "kind", "plan_id", "feature_id",
                     "schema_id", "plan_sha256", "from_version", "to_version",
                     "source_sha256", "output_sha256", "changes"});
    const auto plan_bytes =
        migration_files::read(migration_files::path(arguments.require("plan")));
    const auto plan = read_plan(plan_bytes);
    if (unsigned_number(receipt.at("schema_version"), 1U) != 1U ||
        receipt.at("kind") != "config-migration" ||
        receipt.at("plan_sha256") != sha256(plan_bytes))
      throw CompatibilityError(
          "Rollback receipt does not match migration plan");
    for (const auto key : {"plan_id", "feature_id", "schema_id"})
      if (receipt.at(key) != plan.document.at(key))
        throw CompatibilityError("Rollback identity does not match plan");
    const auto original = migration_files::read(folder / "original.json");
    const auto migrated = migration_files::read(folder / "migrated.json");
    if (receipt.at("source_sha256") != sha256(original) ||
        receipt.at("output_sha256") != sha256(migrated))
      throw SerializationError("Rollback backup/output digest mismatch");
    const auto document = parse_json(original);
    if (document.at("version") != receipt.at("from_version"))
      throw CompatibilityError("Rollback source version mismatch");
    Json changes = Json::array();
    const auto replayed =
        migrate(document, plan, version(receipt.at("to_version")), changes);
    if (encoded(replayed) != migrated || changes != receipt.at("changes"))
      throw CompatibilityError("Rollback receipt cannot reproduce migration");
    Json result{{"restored", true},
                {"source_untouched", true},
                {"schema_id", receipt.at("schema_id")},
                {"version", receipt.at("from_version")},
                {"restored_sha256", sha256(original)}};
    migration_files::publish(
        output_directory,
        {{"restored.json", original}, {"rollback.json", encoded(result)}});
    return {result, 0};
  }
  arguments.allow({"mode", "input", "plan", "to-version", "output"});
  if (mode != "preview" && mode != "apply")
    usage("config.migrate", "Use preview, apply or rollback.");
  if (mode == "preview" && arguments.optional("output"))
    usage("config.migrate",
          "Only apply/rollback writes a new output directory.");
  const auto input_bytes =
      migration_files::read(migration_files::path(arguments.require("input")));
  const auto plan_bytes =
      migration_files::read(migration_files::path(arguments.require("plan")));
  const auto plan = read_plan(plan_bytes);
  const auto input = parse_json(input_bytes);
  const auto target = arguments.number("to-version", 0U, kMaxVersion);
  if (target == 0U)
    usage("config.migrate", "Supply a positive explicit --to-version.");
  Json changes = Json::array();
  const auto output =
      migrate(input, plan, static_cast<std::uint32_t>(target), changes);
  const auto output_bytes = encoded(output);
  if (output_bytes.size() > kInputBytes)
    throw ResourceExhausted("Migrated config exceeds 1 MiB");
  Json receipt{{"schema_version", 1U},
               {"kind", "config-migration"},
               {"plan_id", plan.document.at("plan_id")},
               {"feature_id", plan.document.at("feature_id")},
               {"schema_id", plan.document.at("schema_id")},
               {"plan_sha256", sha256(plan_bytes)},
               {"from_version", input.at("version")},
               {"to_version", target},
               {"source_sha256", sha256(input_bytes)},
               {"output_sha256", sha256(output_bytes)},
               {"changes", changes}};
  const auto receipt_bytes = encoded(receipt);
  if (receipt_bytes.size() > kInputBytes)
    throw ResourceExhausted("Migration receipt exceeds 1 MiB");
  if (mode == "apply") {
    migration_files::publish(migration_files::path(arguments.require("output")),
                             {{"original.json", input_bytes},
                              {"migrated.json", output_bytes},
                              {"receipt.json", receipt_bytes}});
  }
  receipt["mode"] = mode;
  receipt["source_untouched"] = true;
  receipt["already_at_target"] = input.at("version") == target;
  return {receipt, 0};
}
}  // namespace

Result config_migration_command(const Arguments& arguments) {
  try {
    return command(arguments);
  } catch (const Error& error) {
    // ConfigResolver diagnostics can include rejected input text. Preserve the
    // typed category but never surface values or secret references in reports.
    throw Error(error.code(),
                "Config migration rejected the input or operation",
                "Check the explicit schema chain, preconditions, limits and "
                "new output paths.");
  } catch (const Json::exception&) {
    throw SerializationError("Config migration has malformed fields or values");
  } catch (const fs::filesystem_error&) {
    throw IoError("Config migration filesystem operation failed");
  }
}
}  // namespace jyppx::liuding::cli
