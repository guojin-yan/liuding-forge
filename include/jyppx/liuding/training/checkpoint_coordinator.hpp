// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_TRAINING_CHECKPOINT_COORDINATOR_HPP
#define JYPPX_LIUDING_TRAINING_CHECKPOINT_COORDINATOR_HPP
/**
 * @file checkpoint_coordinator.hpp
 * @brief Coordinates transactional state capture and restore across checkpoint
 * participants.
 */

#include <jyppx/liuding/foundation/checkpoint.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace jyppx::liuding::training {

/**
 * @brief Describes one coordinated checkpoint without defining its container.
 *
 * Participant descriptors are stored in ascending participant-ID order. Blob
 * lengths, digests, compression, and physical paths belong to the interop
 * artifact manifest rather than this training-level descriptor.
 */
struct CheckpointDescriptor {
  RestoreLevel restore_level =
      RestoreLevel::exact;  ///< Weakest guarantee among all participants.
  std::vector<StateManifest>
      participants;  ///< Canonically ordered participant state descriptors.
};

/**
 * @brief Validates a coordinated checkpoint descriptor.
 * @param descriptor Descriptor to validate without modifying it.
 * @throws InvalidArgument If the participant set is empty, unsorted,
 * duplicated, invalid, or inconsistent with the aggregate restore level.
 */
void validate_checkpoint_descriptor(const CheckpointDescriptor& descriptor);

/**
 * @brief Owns checkpoint participants under stable, unique identifiers.
 *
 * Registration order never affects checkpoint order; traversal is canonical
 * by participant ID. The registry owns participants through shared pointers.
 * It is not thread-safe and must not be mutated during coordinator operations.
 */
class CheckpointParticipantRegistry final {
 public:
  /** @brief Constructs an empty participant registry. */
  CheckpointParticipantRegistry() = default;

  /**
   * @brief Registers one validated participant.
   * @param participant Non-null shared participant with a stable descriptor.
   * @throws InvalidArgument If the participant is null, invalid, or duplicates
   * an existing participant ID.
   */
  void register_participant(std::shared_ptr<CheckpointParticipant> participant);

  /**
   * @brief Registers a complete participant group with a strong guarantee.
   * @param participants Non-empty group of non-null shared participants.
   * @throws InvalidArgument If the group is empty, contains an invalid or
   * duplicate participant, or conflicts with an existing participant ID.
   *
   * Every participant is validated against a candidate registry before the
   * live registry changes. Failure therefore leaves the original registry
   * unchanged and never exposes a partially registered state domain.
   */
  void register_participants(
      std::vector<std::shared_ptr<CheckpointParticipant>> participants);

  /**
   * @brief Reports whether a participant ID is registered.
   * @param participant_id Stable logical participant identifier.
   * @return `true` when the registry contains that identifier.
   */
  bool contains(std::string_view participant_id) const;

  /**
   * @brief Returns the number of registered participants.
   * @return Current participant count.
   */
  std::size_t size() const noexcept;

  /**
   * @brief Returns participant IDs in deterministic checkpoint order.
   * @return Ascending stable identifiers owned by the returned vector.
   */
  std::vector<std::string> participant_ids() const;

 private:
  friend class CheckpointCoordinator;

  std::map<std::string, std::shared_ptr<CheckpointParticipant>>
      participants_;  ///< Participants keyed in canonical checkpoint order.
};

/**
 * @brief Receives a coordinated checkpoint through an explicit transaction.
 *
 * `begin()` must copy the descriptor. Writers returned by `writer_for()` stay
 * valid until `commit()` or `rollback()`. `commit()` must reject missing
 * required fields. `rollback()` must be safe after any failed transaction
 * stage and must not throw.
 */
class CheckpointStateSink {
 public:
  /** @brief Enables destruction through the checkpoint sink interface. */
  virtual ~CheckpointStateSink() = default;

  /**
   * @brief Starts a new checkpoint transaction.
   * @param descriptor Validated descriptor for the complete participant set.
   */
  virtual void begin(const CheckpointDescriptor& descriptor) = 0;

  /**
   * @brief Returns the writer assigned to one declared participant.
   * @param participant_id Identifier present in the transaction descriptor.
   * @return A writer valid until transaction completion or rollback.
   */
  virtual StateWriter& writer_for(std::string_view participant_id) = 0;

  /**
   * @brief Atomically publishes every participant written in the transaction.
   */
  virtual void commit() = 0;

  /**
   * @brief Discards all unpublished transaction state.
   *
   * This function must be idempotent, must be safe after a partial `begin()`,
   * and must not throw.
   */
  virtual void rollback() noexcept = 0;
};

/**
 * @brief Supplies all participant states from one immutable checkpoint.
 *
 * The descriptor and readers must remain stable for the source lifetime. The
 * source validates physical integrity, actual blob lengths, and hashes before
 * exposing a reader; the coordinator validates the logical participant set.
 */
class CheckpointStateSource {
 public:
  /** @brief Enables destruction through the checkpoint source interface. */
  virtual ~CheckpointStateSource() = default;

  /**
   * @brief Returns the descriptor stored with this checkpoint.
   * @return An immutable reference valid for the source lifetime.
   */
  virtual const CheckpointDescriptor& descriptor() const noexcept = 0;

  /**
   * @brief Returns one immutable participant reader.
   * @param participant_id Identifier declared by `descriptor()`.
   * @return A reader whose manifest matches the declared participant.
   * @throws SerializationError If the participant state is unavailable.
   */
  virtual const StateReader& reader_for(
      std::string_view participant_id) const = 0;
};

/**
 * @brief Configures coordinator resource and topology limits.
 */
struct CheckpointCoordinatorOptions {
  std::size_t max_participants =
      1024;  ///< Positive maximum participant count per checkpoint.
  std::uint64_t max_rollback_bytes =
      1024ULL * 1024ULL *
      1024ULL;  ///< Maximum declared bytes snapshotted before restore.
};

/**
 * @brief Coordinates deterministic, transactional multi-participant state.
 *
 * The coordinator borrows a registry that must outlive it. Operations are not
 * thread-safe, and callers must quiesce participant mutation while saving or
 * restoring. A restore validates every stored descriptor and captures every
 * current participant before the first mutation. If a participant load fails,
 * all attempted participants are restored in reverse order.
 */
class CheckpointCoordinator final {
 public:
  /**
   * @brief Creates a coordinator over a borrowed participant registry.
   * @param registry Registry that must outlive this coordinator.
   * @param options Positive topology and rollback resource limits.
   * @throws InvalidArgument If an option is zero.
   */
  explicit CheckpointCoordinator(CheckpointParticipantRegistry& registry,
                                 CheckpointCoordinatorOptions options = {});

  /**
   * @brief Describes the current participant set in canonical order.
   * @return Validated descriptor with the weakest aggregate restore level.
   * @throws InvalidArgument If the registry is empty, too large, or changed to
   * contain an invalid participant.
   */
  CheckpointDescriptor describe() const;

  /**
   * @brief Saves every participant through one sink transaction.
   * @param sink Storage adapter receiving the coordinated state.
   * @return The descriptor committed to the sink.
   * @throws Error If validation, participant save, or sink commit fails.
   *
   * Any failure after transaction start invokes `sink.rollback()`.
   */
  CheckpointDescriptor save(CheckpointStateSink& sink) const;

  /**
   * @brief Restores every registered participant transactionally.
   * @param source Immutable checkpoint source with a complete participant set.
   * @param policy Weakest restore guarantee accepted by the caller.
   * @return The stored descriptor after all participants restore successfully.
   * @throws InvalidArgument If the caller policy or coordinator state is
   * invalid.
   * @throws CompatibilityError If topology or restore guarantees conflict.
   * @throws ResourceExhausted If the rollback snapshot budget is exceeded.
   * @throws InternalError If rollback itself fails after a restore error.
   */
  CheckpointDescriptor restore(
      const CheckpointStateSource& source,
      RestorePolicy policy = RestorePolicy::exact_only);

 private:
  CheckpointParticipantRegistry&
      registry_;  ///< Borrowed participant registry coordinated by this object.
  CheckpointCoordinatorOptions
      options_;  ///< Immutable topology and rollback resource limits.
};

}  // namespace jyppx::liuding::training

#endif  // JYPPX_LIUDING_TRAINING_CHECKPOINT_COORDINATOR_HPP