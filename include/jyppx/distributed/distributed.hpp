#ifndef JYPPX_DISTRIBUTED_DISTRIBUTED_HPP
#define JYPPX_DISTRIBUTED_DISTRIBUTED_HPP

#include <jyppx/core/types.hpp>
#include <jyppx/foundation/error.hpp>

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
 * other backend raises `NotImplemented`, while an invalid rank/world-size
 * pair raises `InvalidArgument`.
 */
void init_process_group(const std::string& backend = "single-process",
                        int rank = 0, int world_size = 1);

/**
 * @brief Reports whether a process group has been initialized.
 * @return `true` after a successful call to `init_process_group`.
 */
bool is_initialized();

/**
 * @brief Returns the rank of this process.
 * @return Rank zero before initialization and for the single-process group.
 */
int get_rank();

/**
 * @brief Returns the number of participating processes.
 * @return World size one before initialization and for the single-process
 * group.
 */
int get_world_size();

/**
 * @brief Synchronizes the current process group.
 *
 * The single-process implementation is a no-op. A future c10d backend owns
 * the actual collective synchronization.
 */
void barrier();

/**
 * @brief Averages one gradient tensor across the process group
 *
 * With world_size == 1 the gradient is returned unchanged. Larger worlds
 * require a c10d backend and raise `NotImplemented` in this build; the
 * hook-based DDP wrapper routes every parameter gradient through here.
 *
 * @param gradient Gradient tensor produced by autograd.
 * @return The unchanged gradient for the single-process group.
 * @throws NotImplemented If a multi-process reduction is requested without a
 * c10d backend.
 */
Tensor average_gradients(Tensor gradient);

}  // namespace distributed
}  // namespace jyppx

#endif  // JYPPX_DISTRIBUTED_DISTRIBUTED_HPP
