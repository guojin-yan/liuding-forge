// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/training/checkpoint_coordinator.hpp>

#include <algorithm>
#include <exception>
#include <limits>
#include <map>
#include <new>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace jyppx::training {
namespace {

[[noreturn]] void throw_descriptor_error(std::string operation,
                                         std::string expected,
                                         std::string actual, std::string hint) {
  throw InvalidArgument("Invalid coordinated checkpoint descriptor",
                        ErrorDetails{.component = "training.checkpoint",
                                     .operation = std::move(operation),
                                     .expected = std::move(expected),
                                     .actual = std::move(actual),
                                     .hint = std::move(hint)});
}

[[noreturn]] void throw_compatibility_error(std::string operation,
                                            std::string expected,
                                            std::string actual,
                                            std::string hint) {
  throw CompatibilityError(
      "Checkpoint is incompatible with the participant registry",
      ErrorDetails{.component = "training.checkpoint",
                   .operation = std::move(operation),
                   .expected = std::move(expected),
                   .actual = std::move(actual),
                   .hint = std::move(hint)});
}

int restore_level_rank(RestoreLevel level) {
  switch (level) {
    case RestoreLevel::exact:
      return 1;
    case RestoreLevel::bounded:
      return 2;
    case RestoreLevel::best_effort:
      return 3;
  }
  return std::numeric_limits<int>::max();
}

RestoreLevel aggregate_restore_level(
    const std::vector<StateManifest>& manifests) {
  auto aggregate = RestoreLevel::exact;
  for (const auto& manifest : manifests) {
    if (restore_level_rank(manifest.restore_level) >
        restore_level_rank(aggregate)) {
      aggregate = manifest.restore_level;
    }
  }
  return aggregate;
}

bool state_field_equal(const StateFieldDescriptor& left,
                       const StateFieldDescriptor& right) {
  return left.name == right.name && left.media_type == right.media_type &&
         left.max_bytes == right.max_bytes && left.required == right.required;
}

bool state_manifest_equal(const StateManifest& left,
                          const StateManifest& right) {
  return left.participant_id == right.participant_id &&
         left.schema_version == right.schema_version &&
         left.restore_level == right.restore_level &&
         left.degradation_reasons == right.degradation_reasons &&
         left.fields.size() == right.fields.size() &&
         std::equal(left.fields.begin(), left.fields.end(),
                    right.fields.begin(), state_field_equal);
}

const StateFieldDescriptor* find_field(const StateManifest& manifest,
                                       std::string_view name) {
  const auto field =
      std::find_if(manifest.fields.begin(), manifest.fields.end(),
                   [name](const StateFieldDescriptor& candidate) {
                     return candidate.name == name;
                   });
  return field == manifest.fields.end() ? nullptr : &*field;
}

class MemoryParticipantState final : public StateWriter, public StateReader {
 public:
  explicit MemoryParticipantState(StateManifest manifest)
      : manifest_(std::move(manifest)) {
    validate_state_manifest(manifest_);
  }

  void write_blob(std::string_view name,
                  std::span<const std::byte> data) override {
    const auto* field = find_field(manifest_, name);
    if (field == nullptr) {
      throw SerializationError(
          "Participant wrote an undeclared rollback field",
          ErrorDetails{.component = "training.checkpoint",
                       .operation = "snapshot.write_blob",
                       .expected = "a field declared by the participant",
                       .actual = std::string(name),
                       .hint = "Keep save_state and describe_state aligned."});
    }
    if (data.size() > field->max_bytes) {
      throw ResourceExhausted(
          "Rollback field exceeds its declared size limit",
          ErrorDetails{
              .component = "training.checkpoint",
              .operation = "snapshot.write_blob",
              .expected = std::to_string(field->max_bytes),
              .actual = std::to_string(data.size()),
              .hint = "Increase the declared field bound deliberately."});
    }
    if (blobs_.contains(std::string(name))) {
      throw SerializationError(
          "Participant wrote a rollback field more than once",
          ErrorDetails{.component = "training.checkpoint",
                       .operation = "snapshot.write_blob",
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
          ErrorDetails{.component = "training.checkpoint",
                       .operation = "snapshot.write_blob",
                       .expected = "memory within the configured budget",
                       .actual = std::to_string(data.size()),
                       .hint = "Reduce checkpoint state or rollback budget."});
    }
  }

  const StateManifest& manifest() const noexcept override { return manifest_; }

  bool contains(std::string_view name) const override {
    return blobs_.contains(std::string(name));
  }

  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t max_bytes) const override {
    const auto blob = blobs_.find(std::string(name));
    if (blob == blobs_.end()) {
      throw SerializationError(
          "Rollback snapshot field is missing",
          ErrorDetails{.component = "training.checkpoint",
                       .operation = "snapshot.read_blob",
                       .expected = std::string(name),
                       .actual = "missing",
                       .hint = "Treat incomplete snapshots as invalid."});
    }
    if (blob->second.size() > max_bytes) {
      throw ResourceExhausted(
          "Rollback snapshot field exceeds the reader limit",
          ErrorDetails{
              .component = "training.checkpoint",
              .operation = "snapshot.read_blob",
              .expected = std::to_string(max_bytes),
              .actual = std::to_string(blob->second.size()),
              .hint =
                  "Keep participant read limits aligned with its manifest."});
    }
    return blob->second;
  }

  void finalize() const {
    for (const auto& field : manifest_.fields) {
      if (field.required && !blobs_.contains(field.name)) {
        throw SerializationError(
            "Participant omitted a required rollback field",
            ErrorDetails{
                .component = "training.checkpoint",
                .operation = "snapshot.finalize",
                .expected = field.name,
                .actual = "missing",
                .hint =
                    "Write every required field before save_state returns."});
      }
    }
  }

 private:
  StateManifest manifest_;
  std::map<std::string, std::vector<std::byte>> blobs_;
};

void validate_registry_size(std::size_t size,
                            const CheckpointCoordinatorOptions& options) {
  if (size == 0 || size > options.max_participants) {
    throw_descriptor_error(
        "participants",
        "between 1 and " + std::to_string(options.max_participants),
        std::to_string(size), "Register a bounded, non-empty participant set.");
  }
}

void validate_rollback_budget(const CheckpointDescriptor& descriptor,
                              const CheckpointCoordinatorOptions& options) {
  std::uint64_t total = 0;
  for (const auto& manifest : descriptor.participants) {
    for (const auto& field : manifest.fields) {
      if (field.max_bytes > options.max_rollback_bytes - total) {
        throw ResourceExhausted(
            "Declared rollback snapshot exceeds the coordinator budget",
            ErrorDetails{
                .component = "training.checkpoint",
                .operation = "restore.snapshot_budget",
                .expected = std::to_string(options.max_rollback_bytes),
                .actual = "more than configured",
                .hint =
                    "Reduce participant bounds or raise the explicit budget."});
      }
      total += field.max_bytes;
    }
  }
}

}  // namespace

void validate_checkpoint_descriptor(const CheckpointDescriptor& descriptor) {
  if (descriptor.participants.empty()) {
    throw_descriptor_error("participants", "at least one participant", "empty",
                           "Do not publish empty checkpoints.");
  }

  std::string previous_id;
  for (const auto& manifest : descriptor.participants) {
    validate_state_manifest(manifest);
    if (!previous_id.empty() && previous_id >= manifest.participant_id) {
      throw_descriptor_error(
          "participants", "strictly ascending unique participant IDs",
          previous_id + " then " + manifest.participant_id,
          "Canonicalize the complete participant set before publication.");
    }
    previous_id = manifest.participant_id;
  }

  const auto aggregate = aggregate_restore_level(descriptor.participants);
  if (descriptor.restore_level != aggregate) {
    throw_descriptor_error(
        "restore_level", std::string(restore_level_name(aggregate)),
        std::string(restore_level_name(descriptor.restore_level)),
        "Report the weakest guarantee among all participants.");
  }
}

void CheckpointParticipantRegistry::register_participant(
    std::shared_ptr<CheckpointParticipant> participant) {
  if (participant == nullptr) {
    throw InvalidArgument(
        "Cannot register a null checkpoint participant",
        ErrorDetails{.component = "training.checkpoint",
                     .operation = "registry.register_participant",
                     .expected = "a non-null shared participant",
                     .actual = "null",
                     .hint = "Construct the participant before registration."});
  }
  validate_checkpoint_participant(*participant);
  const auto identifier = participant->participant_id();
  if (!participants_.emplace(identifier, std::move(participant)).second) {
    throw InvalidArgument(
        "Checkpoint participant ID is already registered",
        ErrorDetails{
            .component = "training.checkpoint",
            .operation = "registry.register_participant",
            .expected = "a unique participant ID",
            .actual = identifier,
            .hint = "Assign one owner to each recoverable state domain."});
  }
}

bool CheckpointParticipantRegistry::contains(
    std::string_view participant_id) const {
  return participants_.contains(std::string(participant_id));
}

std::size_t CheckpointParticipantRegistry::size() const noexcept {
  return participants_.size();
}

std::vector<std::string> CheckpointParticipantRegistry::participant_ids()
    const {
  std::vector<std::string> identifiers;
  identifiers.reserve(participants_.size());
  for (const auto& [identifier, participant] : participants_) {
    (void)participant;
    identifiers.push_back(identifier);
  }
  return identifiers;
}

CheckpointCoordinator::CheckpointCoordinator(
    CheckpointParticipantRegistry& registry,
    CheckpointCoordinatorOptions options)
    : registry_(registry), options_(options) {
  if (options_.max_participants == 0 || options_.max_rollback_bytes == 0) {
    throw InvalidArgument(
        "Checkpoint coordinator limits must be positive",
        ErrorDetails{.component = "training.checkpoint",
                     .operation = "coordinator.options",
                     .expected = "positive participant and byte limits",
                     .actual = "zero",
                     .hint = "Set explicit non-zero resource limits."});
  }
}

CheckpointDescriptor CheckpointCoordinator::describe() const {
  validate_registry_size(registry_.participants_.size(), options_);

  CheckpointDescriptor descriptor;
  descriptor.participants.reserve(registry_.participants_.size());
  for (const auto& [registered_id, participant] : registry_.participants_) {
    validate_checkpoint_participant(*participant);
    auto manifest = participant->describe_state();
    if (registered_id != manifest.participant_id) {
      throw_descriptor_error(
          "participants", registered_id, manifest.participant_id,
          "Participant identifiers must remain stable after registration.");
    }
    descriptor.participants.push_back(std::move(manifest));
  }
  descriptor.restore_level = aggregate_restore_level(descriptor.participants);
  validate_checkpoint_descriptor(descriptor);
  return descriptor;
}

CheckpointDescriptor CheckpointCoordinator::save(
    CheckpointStateSink& sink) const {
  auto descriptor = describe();
  bool transaction_attempted = false;
  try {
    transaction_attempted = true;
    sink.begin(descriptor);
    for (const auto& manifest : descriptor.participants) {
      auto& writer = sink.writer_for(manifest.participant_id);
      registry_.participants_.at(manifest.participant_id)->save_state(writer);
    }
    sink.commit();
  } catch (...) {
    if (transaction_attempted) {
      sink.rollback();
    }
    throw;
  }
  return descriptor;
}

CheckpointDescriptor CheckpointCoordinator::restore(
    const CheckpointStateSource& source, RestorePolicy policy) {
  if (restore_policy_name(policy) == "unknown") {
    throw InvalidArgument(
        "Unknown checkpoint restore policy",
        ErrorDetails{
            .component = "training.checkpoint",
            .operation = "restore.policy",
            .expected = "a declared RestorePolicy value",
            .actual = std::to_string(static_cast<std::uint8_t>(policy)),
            .hint = "Reject unrecognized persisted policy values."});
  }

  const auto current_descriptor = describe();
  validate_rollback_budget(current_descriptor, options_);

  const auto& stored_descriptor = source.descriptor();
  validate_registry_size(stored_descriptor.participants.size(), options_);
  validate_checkpoint_descriptor(stored_descriptor);
  if (stored_descriptor.participants.size() !=
      current_descriptor.participants.size()) {
    throw_compatibility_error(
        "restore.participants", "the complete registered participant set",
        std::to_string(stored_descriptor.participants.size()) + " stored vs " +
            std::to_string(current_descriptor.participants.size()) + " current",
        "Migrate checkpoint topology explicitly before restore.");
  }
  if (!restore_policy_accepts(policy, stored_descriptor.restore_level)) {
    throw_compatibility_error(
        "restore.policy", std::string(restore_policy_name(policy)),
        std::string(restore_level_name(stored_descriptor.restore_level)),
        "Choose a checkpoint with a stronger guarantee or relax policy "
        "explicitly.");
  }

  std::vector<const StateReader*> readers;
  readers.reserve(stored_descriptor.participants.size());
  for (std::size_t index = 0; index < stored_descriptor.participants.size();
       ++index) {
    const auto& stored_manifest = stored_descriptor.participants[index];
    const auto& current_manifest = current_descriptor.participants[index];
    if (stored_manifest.participant_id != current_manifest.participant_id) {
      throw_compatibility_error(
          "restore.participants", current_manifest.participant_id,
          stored_manifest.participant_id,
          "Restore requires the exact registered participant ID set.");
    }
    if (!restore_policy_accepts(policy, stored_manifest.restore_level)) {
      throw_compatibility_error(
          "restore.policy", std::string(restore_policy_name(policy)),
          stored_manifest.participant_id + ":" +
              std::string(restore_level_name(stored_manifest.restore_level)),
          "Resolve or explicitly accept the participant degradation.");
    }
    const auto& reader = source.reader_for(stored_manifest.participant_id);
    validate_state_manifest(reader.manifest());
    if (!state_manifest_equal(stored_manifest, reader.manifest())) {
      throw_compatibility_error(
          "restore.reader_manifest", stored_manifest.participant_id,
          reader.manifest().participant_id,
          "Reject sources whose logical descriptor and reader disagree.");
    }
    readers.push_back(&reader);
  }

  std::vector<std::unique_ptr<MemoryParticipantState>> snapshots;
  snapshots.reserve(current_descriptor.participants.size());
  for (const auto& manifest : current_descriptor.participants) {
    auto snapshot = std::make_unique<MemoryParticipantState>(manifest);
    registry_.participants_.at(manifest.participant_id)->save_state(*snapshot);
    snapshot->finalize();
    snapshots.push_back(std::move(snapshot));
  }

  std::size_t attempted = 0;
  try {
    for (; attempted < stored_descriptor.participants.size(); ++attempted) {
      const auto& identifier =
          stored_descriptor.participants[attempted].participant_id;
      registry_.participants_.at(identifier)
          ->load_state(*readers[attempted], policy);
    }
  } catch (...) {
    const auto restore_failure = std::current_exception();
    std::string rollback_failure_id;
    const auto rollback_count = std::min(attempted + 1, snapshots.size());
    for (std::size_t offset = 0; offset < rollback_count; ++offset) {
      const auto index = rollback_count - offset - 1;
      const auto& identifier =
          current_descriptor.participants[index].participant_id;
      try {
        registry_.participants_.at(identifier)
            ->load_state(*snapshots[index], RestorePolicy::allow_best_effort);
      } catch (...) {
        if (rollback_failure_id.empty()) {
          rollback_failure_id = identifier;
        }
      }
    }
    if (!rollback_failure_id.empty()) {
      try {
        std::rethrow_exception(restore_failure);
      } catch (...) {
        std::throw_with_nested(InternalError(
            "Checkpoint restore failed and rollback was incomplete",
            ErrorDetails{.component = "training.checkpoint",
                         .operation = "restore.rollback",
                         .expected = "all attempted participants restored",
                         .actual = rollback_failure_id,
                         .hint =
                             "Stop training and preserve diagnostics; state "
                             "is indeterminate."}));
      }
    }
    std::rethrow_exception(restore_failure);
  }

  return stored_descriptor;
}

}  // namespace jyppx::training
