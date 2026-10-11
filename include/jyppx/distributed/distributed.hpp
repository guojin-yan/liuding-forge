#ifndef JYPPX_DISTRIBUTED_DISTRIBUTED_HPP
#define JYPPX_DISTRIBUTED_DISTRIBUTED_HPP

#include <jyppx/core/types.hpp>

#include <string>

namespace jyppx {
namespace distributed {

/**
 * @brief Process-group state for distributed training
 *
 * This is a single-process reference implementation: it accepts
 * `init_process_group(backend, 0, 1)` so that distributed-aware code paths
 * (DistributedSampler, DDP gradient hooks) run unchanged on one process.
 * Cross-process collectives require a c10d backend build; requesting any
 * other backend or world size throws std::runtime_error.
 */
void init_process_group(const std::string& backend = "single-process",
                        int rank = 0, int world_size = 1);

/** @brief Whether a process group has been initialized. */
bool is_initialized();

/** @brief Rank of this process (0 before initialization). */
int get_rank();

/** @brief Number of participating processes (1 before initialization). */
int get_world_size();

/** @brief Synchronization point (no-op in the single-process group). */
void barrier();

/**
 * @brief Averages one gradient tensor across the process group
 *
 * With world_size == 1 the gradient is returned unchanged. Larger worlds
 * require a c10d backend and throw in this build; the hook-based DDP wrapper
 * routes every parameter gradient through here.
 */
Tensor average_gradients(Tensor gradient);

} // namespace distributed
} // namespace jyppx

#endif // JYPPX_DISTRIBUTED_DISTRIBUTED_HPP
