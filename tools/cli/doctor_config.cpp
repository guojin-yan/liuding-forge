// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "cli.hpp"

#include <jyppx/core/value_schema.hpp>
#include <jyppx/experiment/config.hpp>
#include <jyppx/foundation/capability.hpp>
#include <jyppx/foundation/sha256.hpp>
#include <jyppx/foundation/version.hpp>

#include <torch/serialize.h>
#include <torch/torch.h>

#include <ATen/Context.h>
#include <cmath>
#include <limits>
#include <sstream>

namespace jyppx::cli {
namespace {

ValueSchema schema_node(const Json& value, std::size_t depth = 0U) {
  if (depth > 24U)
    throw ResourceExhausted("CLI config schema depth exceeds 24");
  fields(value, {"kind"},
         {"name", "semantic_tag", "required", "nullable", "fields", "element",
          "elements"});
  ValueSchema schema;
  const auto kind = value.at("kind").get<std::string>();
  if (kind == "dictionary")
    schema.kind = ValueKind::dictionary;
  else if (kind == "list")
    schema.kind = ValueKind::list;
  else if (kind == "tuple")
    schema.kind = ValueKind::tuple;
  else if (kind == "integer")
    schema.kind = ValueKind::integer;
  else if (kind == "floating")
    schema.kind = ValueKind::floating;
  else if (kind == "boolean")
    schema.kind = ValueKind::boolean;
  else if (kind == "string")
    schema.kind = ValueKind::string;
  else
    usage("config.schema.kind",
          "Use dictionary/list/tuple/integer/floating/boolean/string.");
  schema.name = value.value("name", std::string{});
  schema.semantic_tag = value.value("semantic_tag", std::string{});
  schema.required = value.value("required", true);
  schema.nullable = value.value("nullable", false);
  if (schema.kind == ValueKind::dictionary) {
    if (!value.contains("fields") || !value.at("fields").is_object() ||
        value.contains("element") || value.contains("elements")) {
      usage("config.schema",
            "Dictionary requires fields and has no element/elements.");
    }
    for (const auto& item : value.at("fields").items()) {
      schema.fields.emplace(item.key(), schema_node(item.value(), depth + 1U));
    }
  } else if (schema.kind == ValueKind::list) {
    if (!value.contains("element") || value.contains("fields") ||
        value.contains("elements")) {
      usage("config.schema", "List requires exactly one element schema.");
    }
    schema.elements.push_back(schema_node(value.at("element"), depth + 1U));
  } else if (schema.kind == ValueKind::tuple) {
    if (!value.contains("elements") || !value.at("elements").is_array() ||
        value.contains("fields") || value.contains("element")) {
      usage("config.schema", "Tuple requires an ordered elements array.");
    }
    for (const auto& element : value.at("elements"))
      schema.elements.push_back(schema_node(element, depth + 1U));
  } else if (value.contains("fields") || value.contains("element") ||
             value.contains("elements")) {
    usage("config.schema", "Scalar schemas cannot contain child schemas.");
  }
  return schema;
}

c10::IValue config_value(const Json& value, const ValueSchema& schema,
                         std::size_t depth = 0U) {
  if (depth > 24U)
    throw ResourceExhausted("CLI config defaults depth exceeds 24");
  if (value.is_null()) return {};
  if (value.is_boolean()) return value.get<bool>();
  if (value.is_number_unsigned()) {
    if (schema.kind == ValueKind::floating) return value.get<double>();
    return static_cast<std::int64_t>(unsigned_number(
        value,
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())));
  }
  if (value.is_number_integer()) {
    if (schema.kind == ValueKind::floating) return value.get<double>();
    return value.get<std::int64_t>();
  }
  if (value.is_number_float()) {
    const auto number = value.get<double>();
    if (!std::isfinite(number))
      throw SerializationError("Config defaults must contain finite numbers");
    return number;
  }
  if (value.is_string()) return value.get<std::string>();
  if (value.is_array()) {
    if (schema.kind == ValueKind::tuple) {
      if (value.size() != schema.elements.size()) {
        throw SerializationError("Config tuple default has the wrong size");
      }
      std::vector<c10::IValue> elements;
      elements.reserve(value.size());
      for (std::size_t index = 0U; index < value.size(); ++index) {
        elements.push_back(
            config_value(value[index], schema.elements[index], depth + 1U));
      }
      return c10::ivalue::Tuple::create(std::move(elements));
    }
    if (schema.kind != ValueKind::list || schema.elements.size() != 1U) {
      throw SerializationError(
          "Config array default requires a list or tuple schema");
    }
    c10::impl::GenericList result(c10::AnyType::get());
    for (const auto& element : value)
      result.push_back(
          config_value(element, schema.elements.front(), depth + 1U));
    return result;
  }
  if (value.is_object()) {
    if (schema.kind != ValueKind::dictionary) {
      throw SerializationError(
          "Config object default requires a dictionary schema");
    }
    c10::impl::GenericDict result(c10::StringType::get(), c10::AnyType::get());
    for (const auto& item : value.items()) {
      const auto child = schema.fields.find(item.key());
      if (child == schema.fields.end())
        throw SerializationError("Unknown config default field");
      result.insert(item.key(),
                    config_value(item.value(), child->second, depth + 1U));
    }
    return result;
  }
  throw SerializationError("Unsupported config default value");
}
}  // namespace

Result doctor(const Arguments& arguments) {
  arguments.allow({"device", "expected-libtorch"});
  const auto device_name = arguments.optional("device").value_or("cpu");
  if (device_name != "cpu" && device_name != "cuda")
    usage("doctor.device", "Choose cpu or cuda.");
  const auto build = current_build_fingerprint();
  const bool cuda_available =
      at::globalContext().hasCUDA() && at::getNumGPUs() > 0;
  const bool requested_available = device_name == "cpu" || cuda_available;
  Json data{{"build", parse_json(build_fingerprint_json(build))},
            {"capabilities", parse_json(current_capability_manifest_json())},
            {"cuda_available", cuda_available},
            {"device", device_name},
            {"libtorch_matches_expected", true},
            {"checks", Json::array()}};
  int exit_code = 0;
  if (const auto expected = arguments.optional("expected-libtorch")) {
    data["expected_libtorch"] = *expected;
    data["libtorch_matches_expected"] = *expected == build.libtorch_version;
    if (*expected != build.libtorch_version) {
      exit_code = 1;
      data["checks"].push_back({{"check", "libtorch_version"},
                                {"status", "fail"},
                                {"hint",
                                 "Select the LibTorch distribution declared by "
                                 "this consumer's compatibility lane."}});
    }
  }
  if (!requested_available) {
    data["checks"].push_back(
        {{"check", "device"},
         {"status", "fail"},
         {"hint",
          "Install a compatible CUDA driver/runtime or select --device cpu."}});
    return {data, 1};
  }
  try {
    const auto options = torch::TensorOptions()
                             .dtype(torch::kFloat64)
                             .device(torch::Device(device_name));
    auto tensor = torch::ones({2, 3}, options.requires_grad(true));
    (tensor * tensor).sum().backward();
    const bool autograd =
        torch::allclose(tensor.grad(), torch::full_like(tensor, 2.0));
    std::ostringstream output(std::ios::binary | std::ios::out);
    torch::save(tensor.detach(), output);
    std::istringstream input(output.str(), std::ios::binary | std::ios::in);
    torch::Tensor restored;
    torch::load(restored, input);
    const bool serialization = torch::equal(tensor.detach(), restored);
    data["checks"].push_back(
        {{"check", "tensor_autograd"}, {"status", autograd ? "pass" : "fail"}});
    data["checks"].push_back({{"check", "serialization"},
                              {"status", serialization ? "pass" : "fail"}});
    if (!autograd || !serialization) exit_code = 1;
  } catch (const c10::Error&) {
    data["checks"].push_back({{"check", "tensor_autograd_serialization"},
                              {"status", "fail"},
                              {"hint",
                               "Verify LibTorch ABI, runtime library search "
                               "paths, and the selected CUDA device."}});
    exit_code = 1;
  }
  return {data, exit_code};
}

Result config_command(const Arguments& arguments) {
  using namespace experiment::config;
  arguments.allow({"schema", "file", "set", "path"});
  const auto action = arguments.words()[1];
  if (action != "lint" && action != "render" && action != "explain")
    usage("config", "Choose lint, render, or explain.");
  if (action != "explain" && arguments.optional("path"))
    usage("config", "Only explain accepts --path.");
  const auto encoded =
      parse_json(read_text(arguments.require("schema"), 1024U * 1024U));
  fields(encoded, {"schema_version", "schema_id", "version", "root"},
         {"defaults", "environment_bindings", "secret_paths"});
  if (unsigned_number(encoded.at("schema_version"), 1U) != 1U)
    throw CompatibilityError("Unsupported CLI config schema");
  ConfigSchema schema;
  schema.schema_id = encoded.at("schema_id").get<std::string>();
  schema.version = static_cast<std::uint32_t>(unsigned_number(
      encoded.at("version"), std::numeric_limits<std::uint32_t>::max()));
  schema.root = schema_node(encoded.at("root"));
  if (encoded.contains("defaults"))
    schema.defaults = config_value(encoded.at("defaults"), schema.root);
  if (encoded.contains("environment_bindings"))
    schema.environment_bindings =
        encoded.at("environment_bindings")
            .get<std::map<std::string, std::string>>();
  if (encoded.contains("secret_paths"))
    schema.secret_paths =
        encoded.at("secret_paths").get<std::vector<std::string>>();
  ConfigResolver resolver(std::move(schema));
  for (const auto& path : arguments.repeated("file"))
    resolver.add_json_file(std::filesystem::u8path(path));
  resolver.add_process_environment();
  for (const auto& assignment : arguments.repeated("set")) {
    const auto equal = assignment.find('=');
    if (equal == std::string::npos)
      usage("config.set", "Use --set /json/pointer=JSON.");
    resolver.add_cli_override(assignment.substr(0U, equal),
                              assignment.substr(equal + 1U));
  }
  const auto result = resolver.resolve();
  const auto canonical = result.canonical_json();
  Json data{{"schema_id", result.schema().schema_id},
            {"schema_version", result.schema().version},
            {"redacted_config_sha256",
             sha256_hex(std::as_bytes(std::span(canonical)))}};
  if (action == "explain")
    data["explanation"] = parse_json(
        result.explain_json(arguments.optional("path").value_or("")));
  else if (action == "render")
    data["config"] = parse_json(canonical);
  else
    data["valid"] = true;
  return {data, 0};
}
}  // namespace jyppx::cli
