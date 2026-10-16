// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "detail/memory_participant_state.hpp"

#include <jyppx/foundation/error.hpp>
#include <jyppx/training/checkpoint_coordinator.hpp>

#include <algorithm>
#include <exception>
#include <limits>
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
  std::vector<std::shared_ptr<CheckpointParticipant>> group;
  group.push_back(std::move(participant));
  register_participants(std::move(group));
}

void CheckpointParticipantRegistry::register_participants(
    std::vector<std::shared_ptr<CheckpointParticipant>> participants) {
  if (participants.empty()) {
    throw InvalidArgument(
        "Cannot register an empty checkpoint participant group",
        ErrorDetails{.component = "training.checkpoint",
                     .operation = "registry.register_participants",
                     .expected = "at least one participant",
                     .actual = "empty",
                     .hint = "Register one complete recoverable state group."});
  }

  auto candidate = participants_;
  for (auto& participant : participants) {
    if (participant == nullptr) {
      throw InvalidArgument(
          "Cannot register a null checkpoint participant",
          ErrorDetails{
              .component = "training.checkpoint",
              .operation = "registry.register_participants",
              .expected = "non-null shared participants",
              .actual = "null",
              .hint = "Construct every participant before registration."});
    }
    validate_checkpoint_participant(*participant);
    const auto identifier = participant->participant_id();
    if (!candidate.emplace(identifier, std::move(participant)).second) {
      throw InvalidArgument(
          "Checkpoint participant ID is already registered",
          ErrorDetails{
              .component = "training.checkpoint",
              .operation = "registry.register_participants",
              .expected = "unique participant IDs across the whole registry",
              .actual = identifier,
              .hint = "Assign one owner to each recoverable state domain."});
    }
  }
  participants_.swap(candidate);
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

  std::vector<std::unique_ptr<detail::MemoryParticipantState>> snapshots;
  snapshots.reserve(current_descriptor.participants.size());
  for (const auto& manifest : current_descriptor.participants) {
    auto snapshot = std::make_unique<detail::MemoryParticipantState>(
        manifest, "training.checkpoint", "snapshot");
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
