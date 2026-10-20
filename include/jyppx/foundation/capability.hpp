// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_FOUNDATION_CAPABILITY_HPP
#define JYPPX_FOUNDATION_CAPABILITY_HPP

/**
 * @file capability.hpp
 * @brief Defines the bounded cross-domain capability manifest contract.
 */

#include "jyppx/foundation/version.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace jyppx {

/** @brief Maximum feature entries accepted by one capability manifest. */
inline constexpr std::size_t kMaxCapabilityFeatures = 1024U;

/** @brief Maximum values accepted by one descriptor collection. */
inline constexpr std::size_t kMaxCapabilityValues = 256U;

/** @brief Maximum UTF-8 bytes accepted by one capability string. */
inline constexpr std::size_t kMaxCapabilityStringBytes = 4096U;

/**
 * @brief Identifies the evidence maturity of one advertised feature.
 *
 * Numeric values are part of the schema-v1 machine contract and must not be
 * renumbered.
 */
enum class CapabilityStatus : std::uint8_t {
  available = 1,     ///< Verified for the descriptor's declared scope.
  experimental = 2,  ///< Implemented but missing a complete support gate.
  unsupported = 3,   ///< Explicitly unavailable in the current scope.
};

/**
 * @brief Returns the stable lowercase capability-status name.
 * @param status Status to render.
 * @return A process-lifetime name, or `"unknown"` for an invalid value.
 */
std::string_view capability_status_name(CapabilityStatus status) noexcept;

/**
 * @brief Identifies how JYPPX provides one feature.
 *
 * Numeric values are part of the schema-v1 machine contract and must not be
 * renumbered.
 */
enum class CapabilityImplementation : std::uint8_t {
  native = 1,       ///< Implemented directly by JYPPX.
  delegated = 2,    ///< Delegated to the declared backend without emulation.
  adapted = 3,      ///< Adapted across a declared semantic boundary.
  unsupported = 4,  ///< No executable implementation is advertised.
};

/**
 * @brief Returns the stable lowercase implementation name.
 * @param implementation Implementation kind to render.
 * @return A process-lifetime name, or `"unknown"` for an invalid value.
 */
std::string_view capability_implementation_name(
    CapabilityImplementation implementation) noexcept;

/**
 * @brief Identifies where one capability observation originated.
 *
 * Numeric values are part of the schema-v1 machine contract and must not be
 * renumbered.
 */
enum class CapabilitySource : std::uint8_t {
  build = 1,                ///< Derived from immutable build configuration.
  runtime = 2,              ///< Observed from the current backend runtime.
  configured_instance = 3,  ///< Declared by one validated configured object.
};

/**
 * @brief Returns the stable lowercase capability-source name.
 * @param source Source kind to render.
 * @return A process-lifetime name, or `"unknown"` for an invalid value.
 */
std::string_view capability_source_name(CapabilitySource source) noexcept;

/** @brief Describes one finite numeric capability bound. */
struct CapabilityLimit {
  std::string name;         ///< Stable lowercase bound identity.
  std::uint64_t value = 0;  ///< Non-negative bound encoded as a JSON number.
  std::string unit;  ///< Stable lowercase unit such as `bytes` or `count`.
};

/**
 * @brief Describes one feature/backend capability in structured form.
 *
 * Collections contain machine tokens, not prose. Human-readable constraints
 * belong in `limitations`. Manifest identity is the `(feature_id, scope_id)`
 * pair. Build/runtime descriptors use an empty scope; each configured-instance
 * descriptor uses the stable identity of the represented object.
 */
struct CapabilityDescriptor {
  std::string feature_id;  ///< Stable lowercase cross-domain feature identity.
  std::string scope_id;    ///< Stable configured-instance identity, or empty.
  std::uint32_t feature_version = 1;  ///< Positive semantic descriptor version.
  CapabilityStatus status =
      CapabilityStatus::experimental;  ///< Evidence maturity for this scope.
  CapabilityImplementation implementation =
      CapabilityImplementation::native;  ///< Native/delegated/adapted origin.
  CapabilitySource source =
      CapabilitySource::build;  ///< Build, runtime, or configured observation.
  std::string backend;          ///< Stable backend identity such as `libtorch`.
  std::vector<std::string> devices;  ///< Supported or configured device tokens.
  std::vector<std::string> dtypes;   ///< Supported or configured dtype tokens.
  std::vector<std::string> schemas;  ///< Versioned schema identities/digests.
  std::vector<std::string> operations;  ///< Executable operation tokens.
  std::vector<CapabilityLimit> limits;  ///< Finite structured numeric bounds.
  std::vector<std::string>
      limitations;  ///< Bounded human-readable restrictions or gaps.
};

/**
 * @brief Binds canonical feature capabilities to one exact build fingerprint.
 */
struct CapabilityManifest {
  static constexpr std::uint32_t format_version =
      1U;  ///< Current machine-readable manifest schema.

  std::uint32_t schema_version =
      format_version;      ///< Serialized capability manifest schema version.
  BuildFingerprint build;  ///< Exact build and runtime identity for entries.
  std::vector<CapabilityDescriptor>
      features;  ///< Features keyed by unique `(feature_id, scope_id)` pairs.
};

/**
 * @brief Validates one capability descriptor without modifying it.
 * @param descriptor Descriptor to validate.
 * @throws InvalidArgument If enums, identifiers, duplicates, or status and
 * implementation semantics are invalid.
 * @throws ResourceExhausted If a collection or string exceeds its bound.
 */
void validate_capability_descriptor(const CapabilityDescriptor& descriptor);

/**
 * @brief Validates one bounded schema-v1 capability manifest.
 * @param manifest Manifest to validate without canonicalizing it.
 * @throws InvalidArgument If schema, build identity, feature identity, or a
 * nested descriptor is invalid.
 * @throws ResourceExhausted If the manifest exceeds a resource bound.
 */
void validate_capability_manifest(const CapabilityManifest& manifest);

/**
 * @brief Creates a validated canonical manifest for an explicit build.
 * @param features Cross-domain descriptors in arbitrary order.
 * @param build Exact build fingerprint associated with every descriptor.
 * @return Manifest with features and nested collections in stable order.
 * @throws Error If validation or runtime allocation fails.
 */
CapabilityManifest make_capability_manifest(
    std::vector<CapabilityDescriptor> features, BuildFingerprint build);

/**
 * @brief Creates a validated canonical manifest for the current build.
 * @param features Cross-domain descriptors in arbitrary order.
 * @return Manifest bound to `current_build_fingerprint()` in stable order.
 * @throws Error If validation, runtime discovery, or allocation fails.
 */
CapabilityManifest make_current_capability_manifest(
    std::vector<CapabilityDescriptor> features);

/**
 * @brief Serializes a validated canonical capability manifest as JSON.
 * @param manifest Manifest to validate and serialize without modifying it.
 * @return Compact deterministic UTF-8 JSON with typed numeric limits.
 * @throws Error If the manifest is invalid or allocation fails.
 */
std::string capability_manifest_json(const CapabilityManifest& manifest);

}  // namespace jyppx

#endif  // JYPPX_FOUNDATION_CAPABILITY_HPP
