// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_INTEROP_CHECKPOINT_MANIFEST_HPP
#define JYPPX_LIUDING_INTEROP_CHECKPOINT_MANIFEST_HPP
/**
 * @file checkpoint_manifest.hpp
 * @brief Defines the bounded checkpoint artifact manifest codec.
 */

#include "jyppx/liuding/foundation/artifact.hpp"
#include "jyppx/liuding/foundation/capability.hpp"
#include "jyppx/liuding/foundation/checkpoint.hpp"
#include "jyppx/liuding/foundation/version.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace jyppx::liuding::interop {

/** @brief Describes one participant state blob in a checkpoint artifact. */
struct CheckpointBlobEntry {
  std::string field_name;        ///< Declared logical state-field name.
  std::string relative_path;     ///< Portable artifact-relative blob path.
  std::uint64_t size_bytes = 0;  ///< Exact stored blob size before decoding.
  ArtifactDigest digest;         ///< Digest of the stored blob bytes.
};

/** @brief Binds one storage-neutral state contract to its concrete blobs. */
struct CheckpointParticipantEntry {
  StateManifest state;  ///< Versioned participant state contract.
  std::vector<CheckpointBlobEntry>
      blobs;  ///< Present fields keyed by unique logical field names.
};

/**
 * @brief Describes one versioned checkpoint artifact without filesystem state.
 */
struct CheckpointArtifactManifest {
  /** @brief Current checkpoint artifact manifest schema version. */
  static constexpr std::uint32_t format_version = 1U;

  std::uint32_t schema_version =
      format_version;      ///< Serialized artifact manifest schema version.
  BuildFingerprint build;  ///< Exact framework build that produced the state.
  std::vector<CheckpointParticipantEntry>
      participants;  ///< Participants keyed by stable participant identity.
};

/** @brief Bounds all untrusted checkpoint manifest parsing and validation. */
struct CheckpointManifestLimits {
  std::size_t max_manifest_bytes =
      1024U * 1024U;               ///< Maximum encoded JSON manifest bytes.
  std::size_t max_depth = 32U;     ///< Maximum JSON container nesting depth.
  std::size_t max_nodes = 32768U;  ///< Maximum JSON values and object keys.
  std::size_t max_string_bytes =
      4096U;  ///< Maximum UTF-8 bytes in one string or object key.
  std::size_t max_total_string_bytes =
      1024U * 1024U;  ///< Maximum aggregate UTF-8 bytes in one manifest.
  std::size_t max_participants =
      1024U;  ///< Maximum participant entries in one checkpoint.
  std::size_t max_blobs = 16384U;  ///< Maximum blobs across all participants.
  std::size_t max_path_bytes =
      512U;  ///< Maximum portable relative-path bytes per blob.
  std::uint64_t max_total_blob_bytes =
      1ULL << 40U;  ///< Maximum declared bytes across all blobs.
};

/**
 * @brief Validates a checkpoint artifact manifest and all nested state.
 * @param manifest Manifest to inspect without modifying it.
 * @param limits Resource limits applied to metadata and declared blobs.
 * @throws InvalidArgument If metadata, paths, digests, or state are invalid.
 * @throws CompatibilityError If the artifact schema is unsupported.
 * @throws ResourceExhausted If an entry count, string, or byte total exceeds a
 * configured limit.
 */
void validate_checkpoint_artifact_manifest(
    const CheckpointArtifactManifest& manifest,
    const CheckpointManifestLimits& limits = {});

/**
 * @brief Parses strict bounded JSON into a validated artifact manifest.
 * @param json UTF-8 JSON text with unique keys and no unknown fields.
 * @param limits Resource limits enforced before returning any state.
 * @return Fully validated manifest using foundation checkpoint contracts.
 * @throws SerializationError If JSON syntax, fields, or types are malformed.
 * @throws CompatibilityError If the artifact schema is unsupported.
 * @throws ResourceExhausted If parsing or declared state exceeds a limit.
 */
CheckpointArtifactManifest parse_checkpoint_artifact_manifest_json(
    std::string_view json, const CheckpointManifestLimits& limits = {});

/**
 * @brief Serializes a validated manifest as deterministic compact JSON.
 * @param manifest Artifact manifest to validate and serialize.
 * @param limits Resource limits applied before and after encoding.
 * @return Canonical UTF-8 JSON with sorted participants, fields, and blobs.
 * @throws Error If validation or bounded serialization fails.
 */
std::string checkpoint_artifact_manifest_json(
    const CheckpointArtifactManifest& manifest,
    const CheckpointManifestLimits& limits = {});

/**
 * @brief Describes the checkpoint artifact manifest codec as a capability.
 * @param limits Resource bounds advertised by the codec instance.
 * @return Experimental build capability for schema-v1 parse/serialize/validate.
 * @throws Error If limits or the resulting capability are invalid.
 */
CapabilityDescriptor make_checkpoint_manifest_capability(
    const CheckpointManifestLimits& limits = {});

}  // namespace jyppx::liuding::interop

#endif  // JYPPX_LIUDING_INTEROP_CHECKPOINT_MANIFEST_HPP