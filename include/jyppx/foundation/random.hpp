#ifndef JYPPX_FOUNDATION_RANDOM_HPP
#define JYPPX_FOUNDATION_RANDOM_HPP

/**
 * @file random.hpp
 * @brief Provides versioned process RNG snapshots, deterministic seed
 * derivation, and scoped state restoration.
 */

#include "jyppx/core/types.hpp"

#include <cstdint>
#include <vector>

namespace jyppx::core {

/**
 * @brief Current serialized layout version for `RngState`.
 *
 * The value changes only when restore compatibility is intentionally broken.
 */
inline constexpr std::uint32_t kRngStateFormatVersion = 1;

/**
 * @brief Owns a snapshot of the process-default CPU and CUDA generators.
 *
 * Tensors are one-dimensional CPU UInt8 buffers owned by this object. CUDA
 * entries are ordered by visible device index. The type is movable and
 * copyable through LibTorch tensor handles; callers must clone tensors before
 * mutating their storage. Access to process-global generators is not
 * synchronized by this structure.
 */
struct RngState {
  Tensor cpu;  ///< CPU generator state as a one-dimensional CPU UInt8 tensor.
  std::vector<Tensor> cuda;  ///< CUDA generator states ordered by device index.
  std::uint32_t format_version =
      kRngStateFormatVersion;  ///< Serialized layout version for compatibility
                               ///< checks.
};

/**
 * @brief Snapshots the process-default random number generators.
 * @param include_cuda Whether to include every visible CUDA device generator.
 * @return An owned state whose tensor buffers do not alias generator storage.
 * @throws DeviceError If LibTorch cannot read a requested generator.
 * @note Concurrent generator use must be externally synchronized when an exact
 * snapshot boundary is required.
 */
RngState get_rng_state(bool include_cuda = true);

/**
 * @brief Validates an RNG snapshot without mutating any generator.
 * @param state Candidate state to validate.
 * @param include_cuda Whether CUDA topology and state buffers are required.
 * @throws CompatibilityError If the format version or CUDA topology differs.
 * @throws InvalidArgument If a tensor representation is invalid or CUDA state
 * is supplied while CUDA restore is disabled.
 * @throws DeviceError If the active generator topology cannot be inspected.
 */
void validate_rng_state(const RngState& state, bool include_cuda = true);

/**
 * @brief Replaces the process-default generator state from a validated
 * snapshot.
 * @param state Snapshot produced by `get_rng_state` on a compatible runtime.
 * @param include_cuda Whether to restore every visible CUDA generator.
 * @throws CompatibilityError If the format version or CUDA topology differs.
 * @throws InvalidArgument If a state tensor has an invalid representation.
 * @throws DeviceError If a backend restore operation fails.
 * @note The function validates before mutation and attempts to restore the
 * pre-call snapshot if LibTorch rejects an update. A second backend failure
 * during rollback cannot provide the strong exception guarantee.
 */
void set_rng_state(const RngState& state, bool include_cuda = true);

/**
 * @brief Seeds the process-default CPU and optional CUDA generators.
 * @param seed Unsigned 64-bit seed applied identically to each selected device.
 * @param include_cuda Whether to seed every visible CUDA generator.
 * @throws DeviceError If a backend generator cannot accept the seed.
 * @note This mutates process-global state and requires external synchronization
 * with concurrent random-number consumers.
 */
void manual_seed(std::uint64_t seed, bool include_cuda = true);

/**
 * @brief Derives a stable independent seed for an epoch, worker, and user
 * stream.
 * @param base_seed Root seed for the training run.
 * @param stream_id Stable caller-defined stream identifier.
 * @param epoch Zero-based epoch index.
 * @param worker_id Zero-based worker identifier within the stream.
 * @return A deterministic 64-bit seed derived with SplitMix64 mixing.
 * @note This function does not access or mutate process-global RNG state.
 */
std::uint64_t derive_seed(std::uint64_t base_seed, std::uint64_t stream_id,
                          std::uint64_t epoch = 0,
                          std::uint64_t worker_id = 0) noexcept;

/**
 * @brief Restores process-default RNG state when leaving a lexical scope.
 *
 * Construction snapshots the selected generators and then seeds them. The
 * guard owns the snapshot and is neither copyable nor movable. Destruction is
 * noexcept; a backend failure during restoration is suppressed because an
 * exception cannot safely escape a destructor. This class is not thread-safe
 * because the guarded generators are process-global.
 */
class RngGuard {
 public:
  /**
   * @brief Snapshots the selected generators and applies a temporary seed.
   * @param seed Unsigned 64-bit seed for the guarded scope.
   * @param include_cuda Whether all visible CUDA generators are guarded.
   * @throws DeviceError If snapshotting or seeding a generator fails.
   */
  explicit RngGuard(std::uint64_t seed, bool include_cuda = true);

  /**
   * @brief Attempts to restore the exact state captured at construction.
   *
   * Backend failures are suppressed to preserve the noexcept destructor
   * contract; code requiring observable recovery must call `set_rng_state`
   * explicitly instead of relying on this guard.
   */
  ~RngGuard() noexcept;

  /** @brief Copy construction is disabled because the guard uniquely restores
   * state. */
  RngGuard(const RngGuard&) = delete;

  /** @brief Copy assignment is disabled because the guard uniquely restores
   * state. */
  RngGuard& operator=(const RngGuard&) = delete;

  /** @brief Move construction is disabled to keep the restore scope fixed. */
  RngGuard(RngGuard&&) = delete;

  /** @brief Move assignment is disabled to keep the restore scope fixed. */
  RngGuard& operator=(RngGuard&&) = delete;

 private:
  RngState
      previous_;  ///< Owned pre-construction snapshot restored on destruction.
  bool include_cuda_;  ///< Whether `previous_` includes CUDA generator states.
};

}  // namespace jyppx::core

#endif  // JYPPX_FOUNDATION_RANDOM_HPP
