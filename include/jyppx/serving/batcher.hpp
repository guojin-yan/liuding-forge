// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
#ifndef JYPPX_SERVING_BATCHER_HPP
#define JYPPX_SERVING_BATCHER_HPP
/** @file batcher.hpp @brief Bounded single-tensor dynamic batching. */
#include <jyppx/serving/runtime.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <stop_token>
namespace jyppx::serving {
/** @brief Hard scheduling and residency bounds, not backend RSS guarantees. */
struct BatchOptions {
  std::size_t max_queue_size = 64U;  ///< Reserved submitters and queued work.
  std::uint64_t max_resident_tensor_bytes = 256ULL << 20U;  ///< Queue + active.
  std::size_t max_batch_size =
      8U;  ///< Maximum compatible request rows per forward.
  std::chrono::milliseconds max_wait{5};  ///< At most 1 second.
  std::size_t max_oom_retries =
      4U;  ///< Global per-original-batch retries, <=8.
};
/** @brief Scheduler counters, with failed including cancelled/expired work. */
struct BatchStatistics {
  std::uint64_t accepted = 0;   ///< Requests admitted by this scheduler.
  std::uint64_t completed = 0;  ///< Requests resolved with independent results.
  std::uint64_t failed =
      0;  ///< Failed, cancelled or expired admitted requests.
  std::uint64_t oom_retries =
      0;                   ///< Confirmed retry-safe split forwards retried.
  std::size_t queued = 0;  ///< Requests waiting for compatible batching.
  std::size_t active = 0;  ///< Requests in the current worker execution.
  std::uint64_t resident_tensor_bytes =
      0;                  ///< Charged queued plus active tensor bytes.
  bool accepting = true;  ///< New submissions may reserve admission.
  bool stopped = false;   ///< Worker completed shutdown.
};
/** @brief One worker batches a declared pure single-tensor inference boundary.
 * @note Every request must be one positional tensor with leading dimension 1,
 * no kwargs, and one tensor result with matching leading batch dimension.
 * Input/output schemas must declare rank/shape and a dynamic leading dimension.
 * Keys include version, pipeline, dtype/device, rank/trailing shape, priority,
 * deadline class, project and principal. Distinct security scopes never mix.
 * max_wait bounds coalescing after the oldest compatible request, not time
 * behind active compute. Cancellation/deadline cannot preempt a running kernel.
 * Requests own input snapshots; responses own independent output snapshots.
 * Callers must keep input storage quiescent while submit snapshots it.
 */
class DynamicBatcher final {
 public:
  /** @brief Validates batching declarations and starts one owned worker. */
  explicit DynamicBatcher(std::shared_ptr<Predictor> predictor,
                          BatchOptions options = {});
  /** @brief Cancels pending work and joins the active worker safely. */
  ~DynamicBatcher();
  /** @brief Copying queue/worker ownership is forbidden. */
  DynamicBatcher(const DynamicBatcher&) = delete;
  /** @brief Replacing queue/worker ownership by copy is forbidden. */
  DynamicBatcher& operator=(const DynamicBatcher&) = delete;
  /** @brief Reserves capacity before copying; rejection is a typed exception.
   * The future resolves exactly once, independently of its peers.
   */
  std::future<PredictResponse> submit(const PredictRequest& request,
                                      std::stop_token stop = {});
  /** @brief Returns an owning synchronized counter and residency snapshot. */
  BatchStatistics statistics() const;
  /** @brief Immutable owner identity for a protocol adapter. */
  std::shared_ptr<Predictor> predictor() const;
  /** @brief Refuses new work, cancels queued work and waits for active compute.
   * False leaves the worker draining with ownership retained. The predictor
   * remains caller-owned; close it after successful scheduler shutdown.
   */
  bool close(Deadline deadline = {}, std::stop_token stop = {});

 private:
  /** @brief Private bounded queue, worker and retry state. */
  class Impl;
  std::unique_ptr<Impl> impl_;  ///< Exclusive scheduler/worker state ownership.
};
}  // namespace jyppx::serving
#endif
