// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_EXPERIMENT_CONFIG_HPP
#define JYPPX_LIUDING_EXPERIMENT_CONFIG_HPP
/**
 * @file config.hpp
 * @brief Defines bounded, schema-driven experiment configuration resolution.
 */

#include <jyppx/liuding/core/value_schema.hpp>
#include <jyppx/liuding/foundation/capability.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace jyppx::liuding::experiment::config {

/** @brief Stable capability feature ID for the config resolver. */
inline constexpr std::string_view kConfigCapabilityFeatureId =
    "experiment.config";

/**
 * @brief Identifies the precedence layer that supplied a configuration value.
 *
 * Numeric values are stable machine-readable identifiers and must not be
 * renumbered.
 */
enum class ConfigSourceKind : std::uint8_t {
  schema_default = 1,  ///< Value declared by the schema defaults object.
  json_file = 2,       ///< Value read from a bounded JSON file.
  environment = 3,     ///< Value read from one allowlisted environment key.
  command_line = 4,    ///< Value supplied by a structured CLI override.
};

/**
 * @brief Returns the stable lowercase name of a config source kind.
 * @param kind Source kind to render.
 * @return A process-lifetime name, or `"unknown"` for an invalid value.
 */
std::string_view config_source_kind_name(ConfigSourceKind kind) noexcept;

/** @brief Records one layer that contributed to a resolved value. */
struct ConfigOrigin {
  ConfigSourceKind kind =
      ConfigSourceKind::schema_default;  ///< Precedence layer identity.
  std::string source;  ///< File, variable, schema, or argv identity.
};

/** @brief Bounds all untrusted config parsing and resolution work. */
struct ConfigLimits {
  std::size_t max_file_bytes =
      1024U * 1024U;              ///< Maximum bytes read from one JSON file.
  std::size_t max_depth = 32U;    ///< Maximum JSON/config nesting depth.
  std::size_t max_nodes = 4096U;  ///< Maximum nodes in one parsed source.
  std::size_t max_string_bytes =
      64U * 1024U;  ///< Maximum UTF-8 bytes in one key or string.
  std::size_t max_total_string_bytes =
      1024U * 1024U;  ///< Maximum aggregate key and string bytes per source.
  std::size_t max_json_files = 16U;      ///< Maximum ordered JSON file sources.
  std::size_t max_cli_overrides = 256U;  ///< Maximum CLI `--set` entries.
  ValueSchemaLimits value_schema_limits{};  ///< Reused schema/runtime bounds.
};

/**
 * @brief Defines one versioned typed config schema and its safe bindings.
 *
 * `root` reuses the core `ValueSchema` type system and must be a closed
 * dictionary tree. `defaults` is a partial value tree; omitted required fields
 * may be supplied by higher-precedence sources. Environment bindings map JSON
 * Pointer paths to explicitly allowlisted process variable names. Secret paths
 * must identify string fields containing `secret://` references.
 */
struct ConfigSchema {
  std::string schema_id;       ///< Stable schema identity and capability scope.
  std::uint32_t version = 1U;  ///< Positive config schema version.
  ValueSchema root;      ///< Closed typed root dictionary shared with core.
  c10::IValue defaults;  ///< Partial typed defaults, or none for no defaults.
  std::map<std::string, std::string>
      environment_bindings;  ///< JSON Pointer to allowlisted variable name.
  std::vector<std::string>
      secret_paths;  ///< JSON Pointer paths containing secret references.
};

/** @brief Describes one resolved value without exposing sensitive content. */
struct ConfigExplanation {
  std::string path;  ///< Canonical JSON Pointer, empty for the root.
  std::string type;  ///< Stable `ValueKind` name from the reused schema.
  std::string display_value;  ///< Canonical JSON or `"<redacted>"`.
  bool redacted = false;      ///< Whether a secret field value was suppressed.
  std::vector<ConfigOrigin>
      origins;  ///< Ordered low-to-high precedence contribution chain.
};

/**
 * @brief Validates a config schema, its defaults, bindings, and resource
 * bounds.
 * @param schema Schema to validate without modifying it.
 * @param limits Resource limits applied to schema metadata and defaults.
 * @throws InvalidArgument If the schema, binding, or default type is invalid.
 * @throws ResourceExhausted If schema metadata or defaults exceed a bound.
 */
void validate_config_schema(const ConfigSchema& schema,
                            const ConfigLimits& limits = {});

/**
 * @brief Reports whether text is a syntactically valid secret reference.
 * @param reference Candidate `secret://provider/scope/name?version=...` text.
 * @return True only for the bounded, nonempty reference grammar.
 */
bool is_secret_reference(std::string_view reference) noexcept;

/**
 * @brief Maps a validated config schema into capability schema v1.
 * @param schema Config schema represented by the configured-instance entry.
 * @param limits Resolver resource bounds advertised as structured limits.
 * @return Experimental configured-instance descriptor scoped by `schema_id`.
 * @throws Error If the schema or resulting capability descriptor is invalid.
 */
CapabilityDescriptor make_config_capability(const ConfigSchema& schema,
                                            const ConfigLimits& limits = {});

/** @brief Owns one immutable validated config and its origin chains. */
class ResolvedConfig {
 public:
  /**
   * @brief Returns a detached copy of the complete typed config value.
   * @return Deep copy whose containers cannot mutate this resolved config.
   */
  c10::IValue value() const;

  /**
   * @brief Returns a detached typed value at a JSON Pointer.
   * @param path Empty for the root or an RFC 6901 JSON Pointer.
   * @return Deep copy of the selected value.
   * @throws InvalidArgument If the path is malformed or absent.
   */
  c10::IValue at(std::string_view path) const;

  /**
   * @brief Returns one integer config value.
   * @param path JSON Pointer identifying an integer schema field.
   * @return Signed 64-bit configured value.
   * @throws InvalidArgument If the path is absent or has another type.
   */
  std::int64_t integer(std::string_view path) const;

  /**
   * @brief Returns one floating-point config value.
   * @param path JSON Pointer identifying a floating schema field.
   * @return Finite double configured value.
   * @throws InvalidArgument If the path is absent or has another type.
   */
  double floating(std::string_view path) const;

  /**
   * @brief Returns one Boolean config value.
   * @param path JSON Pointer identifying a Boolean schema field.
   * @return Configured Boolean value.
   * @throws InvalidArgument If the path is absent or has another type.
   */
  bool boolean(std::string_view path) const;

  /**
   * @brief Returns one string config value, including a secret reference.
   * @param path JSON Pointer identifying a string schema field.
   * @return Owned configured UTF-8 string.
   * @throws InvalidArgument If the path is absent or has another type.
   */
  std::string string(std::string_view path) const;

  /**
   * @brief Returns a detached copy of the validated resolution schema.
   * @return Schema copy whose defaults cannot mutate this resolved config.
   */
  ConfigSchema schema() const;

  /**
   * @brief Returns the complete origin chain for one config path.
   * @param path Empty for the root or an RFC 6901 JSON Pointer.
   * @return Low-to-high precedence contributions for the selected value.
   * @throws InvalidArgument If the path is malformed or absent.
   */
  std::vector<ConfigOrigin> origins(std::string_view path) const;

  /**
   * @brief Explains one resolved value with automatic secret redaction.
   * @param path Empty for the root or an RFC 6901 JSON Pointer.
   * @return Typed display value and complete origin chain.
   * @throws InvalidArgument If the path is malformed or absent.
   */
  ConfigExplanation explain(std::string_view path) const;

  /**
   * @brief Serializes one explanation as deterministic compact UTF-8 JSON.
   * @param path Empty for the root or an RFC 6901 JSON Pointer.
   * @return Canonical object with path, type, redaction, value, and origins.
   * @throws InvalidArgument If the path is malformed or absent.
   */
  std::string explain_json(std::string_view path) const;

  /**
   * @brief Serializes the complete config as deterministic redacted JSON.
   * @return Compact UTF-8 JSON with sorted object keys and redacted secrets.
   */
  std::string canonical_json() const;

 private:
  friend class ConfigResolver;

  /**
   * @brief Creates a validated immutable result from resolver-owned state.
   * @param schema Validated schema copied into the result.
   * @param value Fully validated resolved typed value.
   * @param origins Origin chains keyed by canonical JSON Pointer.
   * @param limits Resource bounds retained for detached conversions.
   */
  ResolvedConfig(ConfigSchema schema, c10::IValue value,
                 std::map<std::string, std::vector<ConfigOrigin>> origins,
                 ConfigLimits limits);

  ConfigSchema schema_;  ///< Owned schema and binding metadata.
  c10::IValue value_;  ///< Fully validated typed value; never exposed by alias.
  std::map<std::string, std::vector<ConfigOrigin>>
      origins_;          ///< Contribution chains keyed by JSON Pointer.
  ConfigLimits limits_;  ///< Bounds used by safe copy and serialization paths.
};

/**
 * @brief Resolves schema defaults, JSON files, environment, and CLI overrides.
 *
 * Resolution order is fixed regardless of method call order:
 * `schema defaults < JSON files < environment < command line`. Later JSON
 * files and later CLI overrides win within their layer. Objects deep-merge;
 * lists and scalar values replace the complete prior value.
 */
class ConfigResolver {
 public:
  /**
   * @brief Creates a resolver for one validated schema.
   * @param schema Versioned typed config contract and partial defaults.
   * @param limits Resource limits for every source and final value.
   * @throws Error If the schema, defaults, bindings, or limits are invalid.
   */
  explicit ConfigResolver(ConfigSchema schema, ConfigLimits limits = {});

  /**
   * @brief Adds one bounded JSON file source.
   * @param path File read during `resolve()`; later files have higher priority.
   * @throws ResourceExhausted If the source count exceeds its limit.
   */
  void add_json_file(std::filesystem::path path);

  /**
   * @brief Supplies a deterministic environment snapshot for allowlisted keys.
   * @param values Variable names and raw UTF-8 values available to bindings.
   * @throws InvalidArgument If an environment snapshot was already supplied.
   * @throws ResourceExhausted If a value exceeds a string resource bound.
   */
  void add_environment(std::map<std::string, std::string> values);

  /**
   * @brief Captures only schema-allowlisted variables from the current process.
   * @throws InvalidArgument If an environment snapshot was already supplied.
   * @throws ResourceExhausted If a value exceeds a string resource bound.
   */
  void add_process_environment();

  /**
   * @brief Adds structured `--set /json/pointer=<json>` CLI arguments.
   *
   * Both `--set` followed by one assignment and `--set=<assignment>` are
   * accepted. No shell expansion or arbitrary option parsing is performed.
   *
   * @param arguments Exact argv tokens excluding the executable name.
   * @throws InvalidArgument If a token or assignment is malformed.
   * @throws ResourceExhausted If override count or text exceeds a bound.
   */
  void add_cli(std::span<const std::string_view> arguments);

  /**
   * @brief Adds one highest-precedence typed JSON override.
   * @param path JSON Pointer to a declared schema field.
   * @param json_value JSON scalar, list, tuple, or object text.
   * @throws InvalidArgument If the path or value type is invalid.
   * @throws ResourceExhausted If override count or text exceeds a bound.
   */
  void add_cli_override(std::string path, std::string json_value);

  /**
   * @brief Resolves, validates, and freezes all configured sources.
   * @return Immutable typed config with origin and redaction metadata.
   * @throws Error If I/O, JSON, UTF-8, schema, secret, or resource checks fail.
   */
  ResolvedConfig resolve() const;

 private:
  ConfigSchema schema_;  ///< Validated schema owned by this resolver.
  ConfigLimits limits_;  ///< Bounds enforced for every resolution.
  std::vector<std::filesystem::path>
      json_files_;  ///< Ordered low-to-high precedence JSON files.
  std::optional<std::map<std::string, std::string>>
      environment_;  ///< Explicit allowlisted environment snapshot.
  std::vector<std::pair<std::string, std::string>>
      cli_overrides_;  ///< Ordered JSON Pointer and JSON value pairs.
};

}  // namespace jyppx::liuding::experiment::config

#endif  // JYPPX_LIUDING_EXPERIMENT_CONFIG_HPP