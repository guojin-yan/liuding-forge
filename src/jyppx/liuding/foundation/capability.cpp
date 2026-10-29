// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "jyppx/liuding/foundation/capability.hpp"

#include "jyppx/liuding/foundation/error.hpp"

#include <algorithm>
#include <array>
#include <iomanip>
#include <new>
#include <set>
#include <sstream>
#include <utility>

namespace jyppx::liuding {
namespace {

constexpr char kComponent[] = "foundation.capability";

[[noreturn]] void throw_invalid(std::string operation, std::string expected,
                                std::string actual, std::string hint) {
  throw InvalidArgument("Invalid capability manifest",
                        ErrorDetails{.component = kComponent,
                                     .operation = std::move(operation),
                                     .expected = std::move(expected),
                                     .actual = std::move(actual),
                                     .hint = std::move(hint)});
}

[[noreturn]] void throw_resource(std::string operation, std::string expected,
                                 std::string actual) {
  throw ResourceExhausted(
      "Capability manifest exceeds its resource bound",
      ErrorDetails{.component = kComponent,
                   .operation = std::move(operation),
                   .expected = std::move(expected),
                   .actual = std::move(actual),
                   .hint = "Reduce capability metadata before publication."});
}

bool valid_identifier(std::string_view value) noexcept {
  if (value.empty()) {
    return false;
  }
  const auto first = static_cast<unsigned char>(value.front());
  if (!((first >= 'a' && first <= 'z') || (first >= '0' && first <= '9'))) {
    return false;
  }
  return std::all_of(value.begin() + 1, value.end(), [](char character) {
    const auto byte = static_cast<unsigned char>(character);
    return (byte >= 'a' && byte <= 'z') || (byte >= '0' && byte <= '9') ||
           byte == '.' || byte == '_' || byte == '-' || byte == '/';
  });
}

bool valid_utf8(std::string_view value) noexcept {
  std::size_t index = 0;
  while (index < value.size()) {
    const auto first = static_cast<unsigned char>(value[index]);
    if (first <= 0x7FU) {
      ++index;
      continue;
    }

    std::size_t trailing = 0;
    std::uint32_t code_point = 0;
    if ((first & 0xE0U) == 0xC0U) {
      trailing = 1;
      code_point = first & 0x1FU;
    } else if ((first & 0xF0U) == 0xE0U) {
      trailing = 2;
      code_point = first & 0x0FU;
    } else if ((first & 0xF8U) == 0xF0U) {
      trailing = 3;
      code_point = first & 0x07U;
    } else {
      return false;
    }
    if (trailing > value.size() - index - 1U) {
      return false;
    }
    for (std::size_t offset = 1; offset <= trailing; ++offset) {
      const auto byte = static_cast<unsigned char>(value[index + offset]);
      if ((byte & 0xC0U) != 0x80U) {
        return false;
      }
      code_point = (code_point << 6U) | (byte & 0x3FU);
    }
    const auto minimum = trailing == 1U   ? 0x80U
                         : trailing == 2U ? 0x800U
                                          : 0x10000U;
    if (code_point < minimum || code_point > 0x10FFFFU ||
        (code_point >= 0xD800U && code_point <= 0xDFFFU)) {
      return false;
    }
    index += trailing + 1U;
  }
  return true;
}

bool printable_text(std::string_view value) noexcept {
  return !value.empty() && valid_utf8(value) &&
         std::none_of(value.begin(), value.end(), [](char character) {
           const auto value = static_cast<unsigned char>(character);
           return value < 0x20U || value == 0x7FU;
         });
}

void validate_string(std::string_view value, std::string operation,
                     bool identifier) {
  if (value.size() > kMaxCapabilityStringBytes) {
    throw_resource(
        std::move(operation),
        "at most " + std::to_string(kMaxCapabilityStringBytes) + " bytes",
        std::to_string(value.size()) + " bytes");
  }
  const auto valid =
      identifier ? valid_identifier(value) : printable_text(value);
  if (!valid) {
    throw_invalid(std::move(operation),
                  identifier ? "a nonempty lowercase identifier"
                             : "nonempty printable UTF-8 text",
                  std::string(value),
                  "Publish bounded machine tokens and keep prose in "
                  "limitations.");
  }
}

void validate_build(const BuildFingerprint& build) {
  const std::pair<std::string_view, std::string_view> fields[] = {
      {"build.jyppx_version", build.jyppx_version},
      {"build.git_commit", build.git_commit},
      {"build.compiler", build.compiler},
      {"build.cxx_standard", build.cxx_standard},
      {"build.os_arch", build.os_arch},
      {"build.build_type", build.build_type},
      {"build.libtorch_version", build.libtorch_version},
      {"build.libtorch_build", build.libtorch_build},
      {"build.accelerator_runtime", build.accelerator_runtime},
      {"build.dependency_lock_digest", build.dependency_lock_digest},
  };
  for (const auto& [name, value] : fields) {
    validate_string(value, std::string(name), false);
  }
}

void validate_values(const std::vector<std::string>& values,
                     std::string_view field, bool identifiers) {
  if (values.size() > kMaxCapabilityValues) {
    throw_resource(
        std::string(field),
        "at most " + std::to_string(kMaxCapabilityValues) + " values",
        std::to_string(values.size()) + " values");
  }
  std::set<std::string_view> unique;
  for (const auto& value : values) {
    validate_string(value, std::string(field), identifiers);
    if (!unique.emplace(value).second) {
      throw_invalid(std::string(field), "unique values", value,
                    "Remove the duplicate capability value.");
    }
  }
}

std::string json_escape(std::string_view value) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string output;
  output.reserve(value.size());
  for (const auto character : value) {
    const auto byte = static_cast<unsigned char>(character);
    switch (character) {
      case '\\':
        output += "\\\\";
        break;
      case '"':
        output += "\\\"";
        break;
      case '\b':
        output += "\\b";
        break;
      case '\f':
        output += "\\f";
        break;
      case '\n':
        output += "\\n";
        break;
      case '\r':
        output += "\\r";
        break;
      case '\t':
        output += "\\t";
        break;
      default:
        if (byte < 0x20U) {
          output += "\\u00";
          output.push_back(kHex[(byte >> 4U) & 0x0FU]);
          output.push_back(kHex[byte & 0x0FU]);
        } else {
          output.push_back(character);
        }
        break;
    }
  }
  return output;
}

void append_string_array(std::ostringstream& output,
                         const std::vector<std::string>& values) {
  output << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0U) {
      output << ',';
    }
    output << '"' << json_escape(values[index]) << '"';
  }
  output << ']';
}

void canonicalize_values(std::vector<std::string>& values) {
  std::sort(values.begin(), values.end());
}

CapabilityManifest canonicalized(CapabilityManifest manifest) {
  validate_capability_manifest(manifest);
  for (auto& feature : manifest.features) {
    canonicalize_values(feature.devices);
    canonicalize_values(feature.dtypes);
    canonicalize_values(feature.schemas);
    canonicalize_values(feature.operations);
    canonicalize_values(feature.limitations);
    std::sort(feature.limits.begin(), feature.limits.end(),
              [](const CapabilityLimit& left, const CapabilityLimit& right) {
                return left.name < right.name;
              });
  }
  std::sort(
      manifest.features.begin(), manifest.features.end(),
      [](const CapabilityDescriptor& left, const CapabilityDescriptor& right) {
        if (left.feature_id != right.feature_id) {
          return left.feature_id < right.feature_id;
        }
        return left.scope_id < right.scope_id;
      });
  return manifest;
}

CapabilityDescriptor component_descriptor(std::string component, bool enabled,
                                          bool preview = false) {
  return CapabilityDescriptor{
      .feature_id = "jyppx.component." + std::move(component),
      .scope_id = {},
      .feature_version = 1U,
      .status = enabled ? (preview ? CapabilityStatus::experimental
                                   : CapabilityStatus::available)
                        : CapabilityStatus::unsupported,
      .implementation = enabled ? CapabilityImplementation::native
                                : CapabilityImplementation::unsupported,
      .source = CapabilitySource::build,
      .backend = "jyppx",
      .devices = {},
      .dtypes = {},
      .schemas = {"component/v1"},
      .operations = {},
      .limits = {},
      .limitations =
          enabled
              ? (preview
                     ? std::vector<
                           std::string>{"Preview component; capability and "
                                        "compatibility limits remain explicit"}
                     : std::vector<std::string>{})
              : std::vector<
                    std::string>{"component disabled at configure time"},
  };
}

}  // namespace

std::string_view capability_status_name(CapabilityStatus status) noexcept {
  switch (status) {
    case CapabilityStatus::available:
      return "available";
    case CapabilityStatus::experimental:
      return "experimental";
    case CapabilityStatus::unsupported:
      return "unsupported";
  }
  return "unknown";
}

std::string_view capability_implementation_name(
    CapabilityImplementation implementation) noexcept {
  switch (implementation) {
    case CapabilityImplementation::native:
      return "native";
    case CapabilityImplementation::delegated:
      return "delegated";
    case CapabilityImplementation::adapted:
      return "adapted";
    case CapabilityImplementation::unsupported:
      return "unsupported";
  }
  return "unknown";
}

std::string_view capability_source_name(CapabilitySource source) noexcept {
  switch (source) {
    case CapabilitySource::build:
      return "build";
    case CapabilitySource::runtime:
      return "runtime";
    case CapabilitySource::configured_instance:
      return "configured-instance";
  }
  return "unknown";
}

void validate_capability_descriptor(const CapabilityDescriptor& descriptor) {
  try {
    validate_string(descriptor.feature_id, "features.feature_id", true);
    if (descriptor.feature_version == 0U) {
      throw_invalid("features.feature_version", "a positive version", "0",
                    "Increment the feature descriptor version from one.");
    }
    if (capability_status_name(descriptor.status) == "unknown") {
      throw_invalid(
          "features.status", "a schema-v1 status",
          std::to_string(static_cast<std::uint8_t>(descriptor.status)),
          "Use available, experimental, or unsupported.");
    }
    if (capability_implementation_name(descriptor.implementation) ==
        "unknown") {
      throw_invalid(
          "features.implementation", "a schema-v1 implementation",
          std::to_string(static_cast<std::uint8_t>(descriptor.implementation)),
          "Use native, delegated, adapted, or unsupported.");
    }
    if (capability_source_name(descriptor.source) == "unknown") {
      throw_invalid(
          "features.source", "a schema-v1 source",
          std::to_string(static_cast<std::uint8_t>(descriptor.source)),
          "Use build, runtime, or configured-instance.");
    }
    if (descriptor.source == CapabilitySource::configured_instance) {
      validate_string(descriptor.scope_id, "features.scope_id", true);
    } else if (!descriptor.scope_id.empty()) {
      throw_invalid("features.scope_id",
                    "empty for build and runtime descriptors",
                    descriptor.scope_id,
                    "Use scope_id only for a validated configured instance.");
    }
    validate_string(descriptor.backend, "features.backend", true);
    validate_values(descriptor.devices, "features.devices", false);
    validate_values(descriptor.dtypes, "features.dtypes", false);
    validate_values(descriptor.schemas, "features.schemas", false);
    validate_values(descriptor.operations, "features.operations", true);
    validate_values(descriptor.limitations, "features.limitations", false);
    if (descriptor.limits.size() > kMaxCapabilityValues) {
      throw_resource(
          "features.limits",
          "at most " + std::to_string(kMaxCapabilityValues) + " limits",
          std::to_string(descriptor.limits.size()) + " limits");
    }
    std::set<std::string_view> limit_names;
    for (const auto& limit : descriptor.limits) {
      validate_string(limit.name, "features.limits.name", true);
      validate_string(limit.unit, "features.limits.unit", true);
      if (!limit_names.emplace(limit.name).second) {
        throw_invalid("features.limits.name", "unique limit names", limit.name,
                      "Merge or rename the duplicate numeric bound.");
      }
    }

    const auto unsupported_status =
        descriptor.status == CapabilityStatus::unsupported;
    const auto unsupported_implementation =
        descriptor.implementation == CapabilityImplementation::unsupported;
    if (unsupported_status != unsupported_implementation) {
      throw_invalid(
          "features.status_implementation",
          "unsupported status and implementation to agree",
          std::string(capability_status_name(descriptor.status)) + "/" +
              std::string(
                  capability_implementation_name(descriptor.implementation)),
          "Do not advertise an executable implementation for an "
          "unsupported feature.");
    }
    if (unsupported_status && descriptor.limitations.empty()) {
      throw_invalid("features.limitations", "at least one unsupported reason",
                    "empty", "Explain why the feature is unavailable.");
    }
    if (unsupported_status && !descriptor.operations.empty()) {
      throw_invalid(
          "features.operations",
          "no executable operations for an unsupported feature",
          std::to_string(descriptor.operations.size()) + " operations",
          "Move intended operations to documentation until the "
          "feature is implemented.");
    }
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    throw_resource("features", "bounded validation memory",
                   "allocation failed");
  }
}

void validate_capability_manifest(const CapabilityManifest& manifest) {
  try {
    if (manifest.schema_version != CapabilityManifest::format_version) {
      throw_invalid("schema_version",
                    std::to_string(CapabilityManifest::format_version),
                    std::to_string(manifest.schema_version),
                    "Use an explicit migration before reading a different "
                    "manifest schema.");
    }
    validate_build(manifest.build);
    if (manifest.features.size() > kMaxCapabilityFeatures) {
      throw_resource(
          "features",
          "at most " + std::to_string(kMaxCapabilityFeatures) + " entries",
          std::to_string(manifest.features.size()) + " entries");
    }
    std::set<std::pair<std::string_view, std::string_view>> feature_keys;
    for (const auto& feature : manifest.features) {
      validate_capability_descriptor(feature);
      if (!feature_keys.emplace(feature.feature_id, feature.scope_id).second) {
        throw_invalid(
            "features.identity", "unique (feature_id, scope_id) pairs",
            feature.feature_id + "/" + feature.scope_id,
            "Assign every configured instance a stable unique scope.");
      }
    }
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    throw_resource("manifest", "bounded validation memory",
                   "allocation failed");
  }
}

CapabilityManifest make_capability_manifest(
    std::vector<CapabilityDescriptor> features, BuildFingerprint build) {
  try {
    return canonicalized(CapabilityManifest{
        .schema_version = CapabilityManifest::format_version,
        .build = std::move(build),
        .features = std::move(features),
    });
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    throw_resource("manifest", "bounded construction memory",
                   "allocation failed");
  }
}

CapabilityManifest make_current_capability_manifest(
    std::vector<CapabilityDescriptor> features) {
  return make_capability_manifest(std::move(features),
                                  current_build_fingerprint());
}

CapabilityManifest current_capability_manifest() {
  const std::array<std::pair<const char*, bool>, 15U> components{{
      {"foundation", true},
      {"core", JYPPX_LIUDING_COMPONENT_CORE_ENABLED != 0},
      {"interop", JYPPX_LIUDING_COMPONENT_INTEROP_ENABLED != 0},
      {"data", JYPPX_LIUDING_COMPONENT_DATA_ENABLED != 0},
      {"data_image", JYPPX_LIUDING_COMPONENT_DATA_IMAGE_ENABLED != 0},
      {"vision_render", JYPPX_LIUDING_COMPONENT_VISION_RENDER_ENABLED != 0},
      {"nn", JYPPX_LIUDING_COMPONENT_NN_ENABLED != 0},
      {"optim", JYPPX_LIUDING_COMPONENT_OPTIM_ENABLED != 0},
      {"training", JYPPX_LIUDING_COMPONENT_TRAINING_ENABLED != 0},
      {"experiment", JYPPX_LIUDING_COMPONENT_EXPERIMENT_ENABLED != 0},
      {"models", JYPPX_LIUDING_COMPONENT_MODELS_ENABLED != 0},
      {"onnx_export", JYPPX_LIUDING_COMPONENT_ONNX_EXPORT_ENABLED != 0},
      {"onnx_runtime", JYPPX_LIUDING_COMPONENT_ONNX_RUNTIME_ENABLED != 0},
      {"quantization", JYPPX_LIUDING_COMPONENT_QUANTIZATION_ENABLED != 0},
      {"serving", JYPPX_LIUDING_COMPONENT_SERVING_ENABLED != 0},
  }};
  std::vector<CapabilityDescriptor> features;
  features.reserve(components.size());
  for (const auto& [component, enabled] : components) {
    const auto name = std::string_view(component);
    const bool preview = name == "models" || name == "quantization" ||
                         name == "serving" || name == "data_image" ||
                         name == "vision_render" || name == "onnx_export" ||
                         name == "onnx_runtime";
    features.push_back(component_descriptor(component, enabled, preview));
  }
  return make_current_capability_manifest(std::move(features));
}

std::string capability_manifest_json(const CapabilityManifest& manifest) {
  try {
    const auto value = canonicalized(manifest);
    std::ostringstream output;
    output << '{' << "\"schema_version\":" << value.schema_version << ','
           << "\"build\":" << build_fingerprint_json(value.build) << ','
           << "\"features\":[";
    for (std::size_t feature_index = 0; feature_index < value.features.size();
         ++feature_index) {
      if (feature_index != 0U) {
        output << ',';
      }
      const auto& feature = value.features[feature_index];
      output << '{' << "\"feature_id\":\"" << json_escape(feature.feature_id)
             << "\",\"scope_id\":\"" << json_escape(feature.scope_id) << "\","
             << "\"feature_version\":" << feature.feature_version << ','
             << "\"status\":\"" << capability_status_name(feature.status)
             << "\",\"implementation\":\""
             << capability_implementation_name(feature.implementation)
             << "\",\"source\":\"" << capability_source_name(feature.source)
             << "\",\"backend\":\"" << json_escape(feature.backend)
             << "\",\"devices\":";
      append_string_array(output, feature.devices);
      output << ",\"dtypes\":";
      append_string_array(output, feature.dtypes);
      output << ",\"schemas\":";
      append_string_array(output, feature.schemas);
      output << ",\"operations\":";
      append_string_array(output, feature.operations);
      output << ",\"limits\":[";
      for (std::size_t limit_index = 0; limit_index < feature.limits.size();
           ++limit_index) {
        if (limit_index != 0U) {
          output << ',';
        }
        const auto& limit = feature.limits[limit_index];
        output << "{\"name\":\"" << json_escape(limit.name)
               << "\",\"value\":" << limit.value << ",\"unit\":\""
               << json_escape(limit.unit) << "\"}";
      }
      output << "],\"limitations\":";
      append_string_array(output, feature.limitations);
      output << '}';
    }
    output << "]}";
    return output.str();
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    throw_resource("json", "bounded serialization memory", "allocation failed");
  }
}

std::string current_capability_manifest_json() {
  return capability_manifest_json(current_capability_manifest());
}

}  // namespace jyppx::liuding
