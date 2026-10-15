// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_FOUNDATION_CHECKPOINT_HPP
#define JYPPX_FOUNDATION_CHECKPOINT_HPP

/**
 * @file checkpoint.hpp
 * @brief Defines storage-neutral contracts for recoverable component state.
 */

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace jyppx {

/**
 * @brief Describes the reproducibility guarantee offered by saved state.
 *
 * Numeric values are persisted in checkpoint metadata and must remain stable.
 */
enum class RestoreLevel : std::uint8_t {
  exact = 1,  ///< Reproduces the continuation under the supported matrix.
  bounded = 2,  ///< Reproduces behavior within documented numeric/data bounds.
  best_effort = 3,  ///< Restores listed state without equivalence guarantees.
};

/**
 * @brief Returns the stable lowercase name of a restore level.
 * @param level Restore guarantee to render.
 * @return A process-lifetime name, or `"unknown"` for an invalid value.
 */
std::string_view restore_level_name(RestoreLevel level) noexcept;

/**
 * @brief Selects the weakest restore guarantee a caller is willing to accept.
 *
 * Numeric values are persisted in restore requests and must remain stable.
 */
enum class RestorePolicy : std::uint8_t {
  exact_only = 1,  ///< Accepts only exact participants.
  allow_bounded = 2,  ///< Accepts exact or bounded participants.
  allow_best_effort = 3,  ///< Accepts every declared restore level.
};

/**
 * @brief Returns the stable lowercase name of a restore policy.
 * @param policy Restore policy to render.
 * @return A process-lifetime name, or `"unknown"` for an invalid value.
 */
std::string_view restore_policy_name(RestorePolicy policy) noexcept;

/**
 * @brief Tests whether a restore policy accepts a participant guarantee.
 * @param policy Weakest guarantee allowed by the caller.
 * @param level Guarantee declared by the participant.
 * @return `true` when `level` is at least as strong as `policy` requires.
 */
bool restore_policy_accepts(RestorePolicy policy, RestoreLevel level) noexcept;

/**
 * @brief Describes one named, bounded state blob owned by a participant.
 *
 * Names are logical identifiers rather than filesystem paths. Storage codecs
 * must not interpret them as paths without a separate escaping step.
 */
struct StateFieldDescriptor {
  std::string name;  ///< Stable ASCII identifier unique within a participant.
  std::string media_type =
      "application/octet-stream";  ///< Blob media type used for diagnostics.
  std::uint64_t max_bytes = 0;  ///< Positive upper bound enforced when reading.
  bool required = true;  ///< Whether absence makes the participant incomplete.
};

/**
 * @brief Describes the state contract published by one checkpoint participant.
 *
 * This descriptor does not contain actual blob lengths or digests. Those are
 * properties of a concrete checkpoint artifact and belong to its interop
 * manifest.
 */
struct StateManifest {
  std::string participant_id;  ///< Stable registry identifier for the owner.
  std::uint32_t schema_version =
      1;  ///< Positive participant-owned state schema version.
  RestoreLevel restore_level =
      RestoreLevel::exact;  ///< Strongest guarantee supported by this state.
  std::vector<StateFieldDescriptor>
      fields;  ///< Complete set of blobs understood by this schema version.
  std::vector<std::string>
      degradation_reasons;  ///< Non-empty reasons for a non-exact guarantee.
};

/**
 * @brief Validates a participant state descriptor before registration or I/O.
 * @param manifest Descriptor to validate without modifying it.
 * @throws InvalidArgument If an enum, identifier, version, field, size bound,
 * media type, duplicate, or degradation declaration is invalid.
 */
void validate_state_manifest(const StateManifest& manifest);

/**
 * @brief Receives participant-owned blobs without exposing a storage format.
 *
 * An implementation must copy or fully consume `data` before `write_blob`
 * returns. It must reject undeclared, duplicate, or oversized fields according
 * to the descriptor supplied by its coordinator.
 */
class StateWriter {
 public:
  /** @brief Enables destruction through the storage-neutral interface. */
  virtual ~StateWriter() = default;

  /**
   * @brief Writes one complete named blob.
   * @param name Logical field name declared by the participant manifest.
   * @param data Blob bytes valid for the duration of this call.
   * @throws SerializationError If the field is invalid or cannot be stored.
   * @throws ResourceExhausted If a declared or global size limit is exceeded.
   */
  virtual void write_blob(std::string_view name,
                          std::span<const std::byte> data) = 0;
};

/**
 * @brief Exposes validated participant blobs without revealing their storage.
 *
 * Implementations must enforce the caller-provided maximum before returning a
 * view. A returned span remains valid until the reader is destroyed.
 */
class StateReader {
 public:
  /** @brief Enables destruction through the storage-neutral interface. */
  virtual ~StateReader() = default;

  /**
   * @brief Returns the validated descriptor stored with the checkpoint state.
   * @return A reference valid for the lifetime of this reader.
   *
   * Participants use this descriptor to select schema migrations. Restore
   * policy must be evaluated against this stored descriptor rather than the
   * capabilities of the currently constructed participant.
   */
  virtual const StateManifest& manifest() const noexcept = 0;

  /**
   * @brief Reports whether a named blob is available.
   * @param name Logical field name to query.
   * @return `true` when the reader contains the complete field.
   */
  virtual bool contains(std::string_view name) const = 0;

  /**
   * @brief Reads one complete named blob under an explicit resource limit.
   * @param name Logical field name to read.
   * @param max_bytes Maximum accepted blob size before exposing any data.
   * @return A view valid for the lifetime of this reader.
   * @throws SerializationError If the field is missing or malformed.
   * @throws ResourceExhausted If the blob exceeds `max_bytes`.
   */
  virtual std::span<const std::byte> read_blob(
      std::string_view name, std::uint64_t max_bytes) const = 0;
};

/**
 * @brief Supplies versioned state to a checkpoint coordinator.
 *
 * Implementations own their schema and blob encoding, but not the checkpoint
 * container, filesystem path, hashing, publication, or participant registry.
 * `load_state` must validate the stored manifest, every participant-owned
 * field, and the requested policy before mutation; any exception must leave
 * the participant unchanged.
 */
class CheckpointParticipant {
 public:
  /** @brief Enables destruction through the participant interface. */
  virtual ~CheckpointParticipant() = default;

  /**
   * @brief Returns the stable registry identifier for this participant.
   * @return An owned identifier matching `describe_state().participant_id`.
   */
  virtual std::string participant_id() const = 0;

  /**
   * @brief Describes the current participant state contract.
   * @return A complete descriptor suitable for validation and registration.
   */
  virtual StateManifest describe_state() const = 0;

  /**
   * @brief Writes every required field for the current state snapshot.
   * @param writer Storage-neutral destination owned by the coordinator.
   */
  virtual void save_state(StateWriter& writer) const = 0;

  /**
   * @brief Restores state atomically from a validated reader.
   * @param reader Storage-neutral source owned by the coordinator.
   * @param policy Weakest restore guarantee accepted by the caller.
   * @throws CompatibilityError If the policy or schema is incompatible.
   * @throws SerializationError If a required state field is malformed.
   */
  virtual void load_state(const StateReader& reader, RestorePolicy policy) = 0;
};

/**
 * @brief Validates a participant identifier and its published manifest.
 * @param participant Participant to inspect without saving or loading state.
 * @throws InvalidArgument If the identifier and manifest disagree or the
 * manifest is invalid.
 */
void validate_checkpoint_participant(
    const CheckpointParticipant& participant);

}  // namespace jyppx

#endif  // JYPPX_FOUNDATION_CHECKPOINT_HPP
