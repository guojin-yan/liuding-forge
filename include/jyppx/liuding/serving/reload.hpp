// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
#ifndef JYPPX_LIUDING_SERVING_RELOAD_HPP
#define JYPPX_LIUDING_SERVING_RELOAD_HPP
/** @file reload.hpp @brief Bounded same-runtime version publication and leases.
 */
#include <jyppx/liuding/serving/runtime.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
namespace jyppx::liuding::serving {
namespace detail {
/** @brief Private shared ownership retained by version leases. */
class ReloadState;
}  // namespace detail
/** @brief Logical instance reservations, never a process RSS or VRAM limit. */
struct ReloadPolicy {
  std::uint64_t initial_reservation_bytes =
      0;  ///< Owner-declared initial budget.
  std::uint64_t max_resident_bytes =
      0;  ///< Shared old/new/staging reservation cap.
  std::size_t max_active_leases =
      32;  ///< Includes decoding and encoding owners.
};
/** @brief Cooperative candidate and previous-version drain time budgets. */
struct ReloadOptions {
  std::chrono::milliseconds candidate_timeout{
      30000};  ///< Positive, at most 24 h.
  std::chrono::milliseconds drain_timeout{
      5000};  ///< Zero permits a polling drain.
};
/** @brief A successful publication can leave an old leased version resident. */
struct ReloadResult {
  std::string previous_version;   ///< Version retired by this publication.
  std::string current_version;    ///< Version bound by new admissions.
  bool previous_drained = false;  ///< False requires a later drain/close call.
};
/** @brief Synchronized logical ownership and reservation counters. */
struct ReloadStatistics {
  bool accepting = false;  ///< New leases can be admitted.
  bool loading = false;    ///< A single candidate owns a reserved slot.
  std::size_t resident_instances =
      0;                          ///< At most two, including the candidate.
  std::size_t active_leases = 0;  ///< All current/retired admission leases.
  std::uint64_t reserved_bytes =
      0;  ///< Declared reservations, not measured RSS.
  std::uint64_t peak_reserved_bytes = 0;  ///< Reservation high-water mark.
  std::uint64_t publications = 0;  ///< Successfully committed version changes.
};
/** @brief Move-only request ownership that pins one immutable model version.
 * @note Keep the lease through decode/predict/encode. Calls on one lease must
 * be externally serialized. An unreturned lease can indefinitely retain its
 * model. Owner destruction stops admission while existing leases safely retain
 * their version; release leases explicitly for deterministic resource recovery.
 */
class PredictionLease final {
 public:
  /** @brief Releases this admission without unloading executing work. */
  ~PredictionLease();
  /** @brief Transfers the pinned version and its admission reservation. */
  PredictionLease(PredictionLease&&) noexcept;
  /** @brief Releases the old lease and adopts another pinned admission.
   * @return This lease after adopting the source admission.
   */
  PredictionLease& operator=(PredictionLease&&) noexcept;
  PredictionLease(const PredictionLease&) = delete;
  PredictionLease& operator=(const PredictionLease&) = delete;
  /** @brief Executes only on this lease's pinned version.
   * @param request An empty expected version is bound to this lease.
   * @param stop Cooperative cancellation.
   * @return Independent result identifying the pinned model version.
   */
  PredictResponse predict(const PredictRequest& request,
                          std::stop_token stop = {}) const;
  /** @brief Returns the pinned version, even after a new version is published.
   * @return Owning runtime identity and counters.
   */
  ModelRuntimeInfo info() const;

 private:
  /** @brief Retains a single admission and its immutable version slot. */
  class State;
  std::unique_ptr<State>
      state_;  ///< Retained owner and immutable version slot.
  /** @brief Adopts one admitted version slot.
   * @param state Owning admitted state.
   */
  explicit PredictionLease(std::unique_ptr<State> state);
  /** @brief The versioned owner is the sole lease admission authority. */
  friend class ReloadablePredictor;
};
/** @brief Two-slot, same-runtime/schema publication over verified backends.
 * @note Transfer exclusive lifecycle ownership of the initial predictor. Do
 * not call it or close it through retained aliases; backends must return fresh
 * predictors. No worker is detached. Deadlines surround trusted load/warmup/
 * release code and cannot preempt it. Destruction stops admission and transfers
 * remaining lifetime to leases; exclude concurrent member calls and
 * destruction.
 */
class ReloadablePredictor final {
 public:
  /** @brief Adopts one ready, unleased predictor and an explicit reservation.
   * @param initial Fully loaded and warmed initial version.
   * @param policy Logical byte and admission budgets.
   */
  ReloadablePredictor(std::shared_ptr<Predictor> initial, ReloadPolicy policy);
  /** @brief Stops admission; existing leases retain sessions until release. */
  ~ReloadablePredictor();
  ReloadablePredictor(const ReloadablePredictor&) = delete;
  ReloadablePredictor& operator=(const ReloadablePredictor&) = delete;
  /** @brief Pins the current version before request decoding begins.
   * @param expected_version Empty selects current; otherwise must match it.
   * @return Move-only version lease retaining one admission slot.
   */
  PredictionLease acquire(std::string_view expected_version = {});
  /** @brief Acquires one lease and performs a synchronous prediction.
   * @param request Invocation and optional exact expected version.
   * @param stop Cooperative request cancellation.
   * @return Independent result from one fixed version.
   */
  PredictResponse predict(const PredictRequest& request,
                          std::stop_token stop = {});
  /** @brief Loads/warms/checks a candidate off the admission path, then
   * publishes.
   * @param backend Exact same execution descriptor as the initial version.
   * @param directory Candidate data-only bundle.
   * @param verify Bundle integrity/signature policy.
   * @param warmup Mandatory independent known-vector cases.
   * @param load Candidate resource bounds and cancellation/deadline.
   * @param options Cooperative finite candidate/drain budgets.
   * @return Successful publication and whether the previous slot was released.
   * @note Reserves 2 * max_state_bytes + max_snapshot_bytes before loading.
   * Reload control is serialized through the end of this call, including drain.
   * These cover declared state/import/snapshot payloads, not backend
   * workspaces, callback allocations or RSS. A retained previous slot blocks
   * another reload. Failures before publication preserve the previous ready
   * version.
   */
  ReloadResult reload(const RuntimeBackend& backend,
                      const std::filesystem::path& directory,
                      const interop::ModelBundleVerifyOptions& verify,
                      std::span<const WarmupCase> warmup,
                      const LoadOptions& load = {},
                      const ReloadOptions& options = {});
  /** @brief Waits for old leases, then releases the previous version.
   * @param deadline Optional cooperative absolute wait deadline.
   * @param stop Cooperative wait cancellation.
   * @return False keeps the old slot and reservation intact.
   */
  bool drain_previous(Deadline deadline = {}, std::stop_token stop = {});
  /** @brief Returns the current model; after close its identity remains
   * readable.
   * @return Owning current model identity/counters and lifecycle.
   */
  ModelRuntimeInfo info() const;
  /** @brief Returns synchronized logical reservations and lease counts.
   * @return Current ownership counters, without process memory claims.
   */
  ReloadStatistics statistics() const;
  /** @brief Stops admission, waits for candidate/leases and releases both
   * slots.
   * @param deadline Optional cooperative absolute close deadline.
   * @param stop Cooperative wait cancellation.
   * @return False leaves owners retained in draining state.
   */
  bool close(Deadline deadline = {}, std::stop_token stop = {});

 private:
  std::shared_ptr<detail::ReloadState>
      state_;  ///< Shared lease/lifecycle owner.
};
}  // namespace jyppx::liuding::serving
#endif
