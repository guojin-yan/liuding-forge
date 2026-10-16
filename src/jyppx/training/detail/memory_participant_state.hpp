// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_TRAINING_DETAIL_MEMORY_PARTICIPANT_STATE_HPP
#define JYPPX_TRAINING_DETAIL_MEMORY_PARTICIPANT_STATE_HPP

#include <jyppx/foundation/checkpoint.hpp>
#include <jyppx/foundation/error.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <new>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace jyppx::training::detail {

/**
 * @brief Stores one bounded participant snapshot entirely in memory.
 *
 * This private adapter implements both checkpoint directions so transaction
 * owners can capture state before mutation and restore it after failure. It
 * accepts only fields declared by the participant manifest and owns copied
 * bytes; no executable archive or storage backend is involved.
 */
class MemoryParticipantState final : public StateWriter, public StateReader {
 public:
  /**
   * @brief Creates an empty snapshot governed by one participant manifest.
   * @param manifest Valid manifest declaring every accepted field and bound.
   * @param component Error component reported by snapshot operations.
   * @param operation_prefix Prefix used to distinguish the owning transaction.
   * @throws JyppxError If the supplied state manifest is invalid.
   */
  MemoryParticipantState(StateManifest manifest, std::string component,
                         std::string operation_prefix)
      : manifest_(std::move(manifest)),
        component_(std::move(component)),
        operation_prefix_(std::move(operation_prefix)) {
    validate_state_manifest(manifest_);
  }

  /**
   * @brief Copies one declared field into the rollback snapshot.
   * @param name Declared field name written exactly once.
   * @param data Complete field bytes, bounded by the manifest declaration.
   * @throws SerializationError If the field is undeclared or duplicated.
   * @throws ResourceExhausted If the field exceeds its bound or allocation
   * fails.
   */
  void write_blob(std::string_view name,
                  std::span<const std::byte> data) override {
    const auto field =
        std::find_if(manifest_.fields.begin(), manifest_.fields.end(),
                     [name](const StateFieldDescriptor& candidate) {
                       return candidate.name == name;
                     });
    if (field == manifest_.fields.end()) {
      throw SerializationError(
          "Participant wrote an undeclared rollback field",
          ErrorDetails{.component = component_,
                       .operation = operation_prefix_ + ".write_blob",
                       .expected = "a field declared by the participant",
                       .actual = std::string(name),
                       .hint = "Keep save_state and describe_state aligned."});
    }
    if (data.size() > field->max_bytes) {
      throw ResourceExhausted(
          "Rollback field exceeds its declared size limit",
          ErrorDetails{
              .component = component_,
              .operation = operation_prefix_ + ".write_blob",
              .expected = std::to_string(field->max_bytes),
              .actual = std::to_string(data.size()),
              .hint = "Increase the declared field bound deliberately."});
    }
    if (blobs_.contains(std::string(name))) {
      throw SerializationError(
          "Participant wrote a rollback field more than once",
          ErrorDetails{.component = component_,
                       .operation = operation_prefix_ + ".write_blob",
                       .expected = "one complete write per field",
                       .actual = std::string(name),
                       .hint = "Encode each logical field before writing it."});
    }
    try {
      blobs_.emplace(std::string(name),
                     std::vector<std::byte>(data.begin(), data.end()));
    } catch (const std::bad_alloc&) {
      throw ResourceExhausted(
          "Unable to allocate a participant rollback snapshot",
          ErrorDetails{.component = component_,
                       .operation = operation_prefix_ + ".write_blob",
                       .expected = "memory within the configured budget",
                       .actual = std::to_string(data.size()),
                       .hint = "Reduce checkpoint state or rollback budget."});
    }
  }

  /** @brief Returns the immutable manifest governing this snapshot. */
  const StateManifest& manifest() const noexcept override { return manifest_; }

  /**
   * @brief Reports whether a field was captured.
   * @param name Field name to query.
   * @return True when the snapshot owns bytes for the requested field.
   */
  bool contains(std::string_view name) const override {
    return blobs_.contains(std::string(name));
  }

  /**
   * @brief Returns a borrowed view of one captured field.
   * @param name Captured field name.
   * @param max_bytes Maximum number of bytes accepted by the caller.
   * @return View valid for the lifetime of this snapshot.
   * @throws SerializationError If the requested field is absent.
   * @throws ResourceExhausted If the captured bytes exceed the reader limit.
   */
  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t max_bytes) const override {
    const auto blob = blobs_.find(std::string(name));
    if (blob == blobs_.end()) {
      throw SerializationError(
          "Rollback snapshot field is missing",
          ErrorDetails{.component = component_,
                       .operation = operation_prefix_ + ".read_blob",
                       .expected = std::string(name),
                       .actual = "missing",
                       .hint = "Treat incomplete snapshots as invalid."});
    }
    if (blob->second.size() > max_bytes) {
      throw ResourceExhausted(
          "Rollback snapshot field exceeds the reader limit",
          ErrorDetails{
              .component = component_,
              .operation = operation_prefix_ + ".read_blob",
              .expected = std::to_string(max_bytes),
              .actual = std::to_string(blob->second.size()),
              .hint =
                  "Keep participant read limits aligned with its manifest."});
    }
    return blob->second;
  }

  /**
   * @brief Verifies that every required manifest field was captured.
   * @throws SerializationError If any required field is missing.
   */
  void finalize() const {
    for (const auto& field : manifest_.fields) {
      if (field.required && !blobs_.contains(field.name)) {
        throw SerializationError(
            "Participant omitted a required rollback field",
            ErrorDetails{
                .component = component_,
                .operation = operation_prefix_ + ".finalize",
                .expected = field.name,
                .actual = "missing",
                .hint =
                    "Write every required field before save_state returns."});
      }
    }
  }

 private:
  StateManifest
      manifest_;           ///< Immutable field schema and participant identity.
  std::string component_;  ///< Stable component attached to snapshot errors.
  std::string operation_prefix_;  ///< Owning transaction operation prefix.
  std::map<std::string, std::vector<std::byte>> blobs_;  ///< Owned field bytes.
};

}  // namespace jyppx::training::detail

#endif  // JYPPX_TRAINING_DETAIL_MEMORY_PARTICIPANT_STATE_HPP
