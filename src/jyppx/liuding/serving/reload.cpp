// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
#include "detail.hpp"

#include <jyppx/liuding/serving/reload.hpp>

#include <algorithm>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <utility>
namespace jyppx::liuding::serving {
namespace {
using Clock = std::chrono::steady_clock;
bool Expired(const Deadline& deadline, std::stop_token stop) {
  return stop.stop_requested() || (deadline && Clock::now() >= *deadline);
}
bool SameRuntime(const RuntimeDescriptor& a, const RuntimeDescriptor& b) {
  if (a.backend_id != b.backend_id || a.backend_version != b.backend_version ||
      a.format != b.format || a.environment_lock != b.environment_lock ||
      a.device != b.device || a.dynamic_shapes != b.dynamic_shapes ||
      a.tensor_batching != b.tensor_batching ||
      a.max_batch_size != b.max_batch_size ||
      a.oom_retry_safe != b.oom_retry_safe)
    return false;
  auto ad = a.dtypes, bd = b.dtypes;
  std::sort(ad.begin(), ad.end());
  std::sort(bd.begin(), bd.end());
  auto ap = a.dependencies, bp = b.dependencies;
  const auto order = [](const auto& x, const auto& y) {
    return x.name < y.name;
  };
  std::sort(ap.begin(), ap.end(), order);
  std::sort(bp.begin(), bp.end(), order);
  if (ad != bd || ap.size() != bp.size()) return false;
  for (std::size_t i = 0; i < ap.size(); ++i)
    if (ap[i].name != bp[i].name || ap[i].version != bp[i].version)
      return false;
  return true;
}
bool SameInput(const nn::InvocationSchema& a, const nn::InvocationSchema& b) {
  if (a.allow_extra_keywords != b.allow_extra_keywords ||
      a.positional.size() != b.positional.size() ||
      a.keyword.size() != b.keyword.size())
    return false;
  for (std::size_t i = 0; i < a.positional.size(); ++i)
    if (canonical_value_schema(a.positional[i]) !=
        canonical_value_schema(b.positional[i]))
      return false;
  auto x = a.keyword.begin(), y = b.keyword.begin();
  for (; x != a.keyword.end(); ++x, ++y)
    if (x->first != y->first ||
        canonical_value_schema(x->second) != canonical_value_schema(y->second))
      return false;
  return true;
}
std::uint64_t Reservation(const LoadOptions& options) {
  constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
  if (!options.max_state_bytes || !options.max_snapshot_bytes ||
      options.max_state_bytes > (maximum - options.max_snapshot_bytes) / 2)
    detail::Resource("reload.reservation");
  return options.max_state_bytes * 2 + options.max_snapshot_bytes;
}
}  // namespace
namespace detail {
class ReloadState {
 public:
  struct Entry {
    std::shared_ptr<Predictor> predictor;
    std::uint64_t reservation = 0;
    std::size_t leases = 0;
    ~Entry() {
      try {
        if (predictor) predictor->close();
      } catch (...) {
      }
    }
  };
  mutable std::mutex mutex;
  std::condition_variable cv;
  std::shared_ptr<Entry> current, previous;
  RuntimeDescriptor runtime;
  nn::InvocationSchema input;
  ValueSchema output;
  ModelRuntimeInfo last;
  ReloadPolicy policy;
  bool accepting = true, loading = false, reloading = false,
       maintenance = false;
  std::uint64_t candidate_bytes = 0, peak_bytes = 0, publications = 0;
  std::uint64_t Reserved() const {
    return (current ? current->reservation : 0) +
           (previous ? previous->reservation : 0) + candidate_bytes;
  }
  std::size_t Leases() const {
    return (current ? current->leases : 0) + (previous ? previous->leases : 0);
  }
};
}  // namespace detail
class PredictionLease::State {
 public:
  std::shared_ptr<detail::ReloadState> owner;
  std::shared_ptr<detail::ReloadState::Entry> entry;
  ~State() {
    if (!owner) return;
    std::lock_guard lock(owner->mutex);
    --entry->leases;
    owner->cv.notify_all();
  }
};
PredictionLease::PredictionLease(std::unique_ptr<State> state)
    : state_(std::move(state)) {}
PredictionLease::~PredictionLease() = default;
PredictionLease::PredictionLease(PredictionLease&&) noexcept = default;
PredictionLease& PredictionLease::operator=(PredictionLease&&) noexcept =
    default;
ModelRuntimeInfo PredictionLease::info() const {
  if (!state_) throw NotReady("Prediction lease was moved");
  return state_->entry->predictor->info();
}
PredictResponse PredictionLease::predict(const PredictRequest& request,
                                         std::stop_token stop) const {
  if (!state_) throw NotReady("Prediction lease was moved");
  auto bound = request;
  const auto version = info().model_version;
  if (bound.context.model_version.empty())
    bound.context.model_version = version;
  detail::Context(bound.context, version);
  return state_->entry->predictor->predict(bound, stop);
}
ReloadablePredictor::ReloadablePredictor(std::shared_ptr<Predictor> initial,
                                         ReloadPolicy policy)
    : state_(std::make_shared<detail::ReloadState>()) {
  if (!initial || !policy.initial_reservation_bytes ||
      !policy.max_resident_bytes ||
      policy.initial_reservation_bytes > policy.max_resident_bytes ||
      !policy.max_active_leases)
    detail::Invalid("reload.policy",
                    "ready owner, positive reservations and lease bound");
  const auto info = initial->info();
  if (info.health != HealthStatus::ready || info.active)
    detail::Invalid("reload.initial",
                    "ready exclusive unleased initial predictor");
  state_->runtime = info.runtime;
  state_->input = initial->input_schema();
  state_->output = initial->output_schema();
  state_->last = info;
  state_->policy = policy;
  state_->peak_bytes = policy.initial_reservation_bytes;
  state_->current = std::make_shared<detail::ReloadState::Entry>();
  state_->current->predictor = std::move(initial);
  state_->current->reservation = policy.initial_reservation_bytes;
}
ReloadablePredictor::~ReloadablePredictor() {
  std::lock_guard lock(state_->mutex);
  state_->accepting = false;
  state_->cv.notify_all();
  // Leases retain ReloadState. Its entries close only after the last request
  // owner releases, so destruction never waits for a lease on this same thread.
}
PredictionLease ReloadablePredictor::acquire(
    std::string_view expected_version) {
  detail::Text(expected_version);
  auto lease = std::make_unique<PredictionLease::State>();
  std::lock_guard lock(state_->mutex);
  if (!state_->accepting || !state_->current ||
      state_->current->predictor->info().health != HealthStatus::ready)
    throw NotReady("Versioned inference admission is closed");
  const auto version = state_->current->predictor->info().model_version;
  if (!expected_version.empty() && expected_version != version)
    detail::Invalid("reload.version", "the current published model version");
  if (state_->Leases() >= state_->policy.max_active_leases)
    detail::Resource("reload.leases");
  lease->entry = state_->current;
  ++lease->entry->leases;
  lease->owner = state_;
  return PredictionLease(std::move(lease));
}
PredictResponse ReloadablePredictor::predict(const PredictRequest& request,
                                             std::stop_token stop) {
  detail::Boundary(stop, request.deadline);
  return acquire(request.context.model_version).predict(request, stop);
}
ReloadResult ReloadablePredictor::reload(
    const RuntimeBackend& backend, const std::filesystem::path& directory,
    const interop::ModelBundleVerifyOptions& verify,
    std::span<const WarmupCase> warmup, const LoadOptions& load,
    const ReloadOptions& options) {
  const auto maximum_timeout = std::chrono::hours(24);
  if (options.candidate_timeout <= std::chrono::milliseconds::zero() ||
      options.candidate_timeout > maximum_timeout ||
      options.drain_timeout < std::chrono::milliseconds::zero() ||
      options.drain_timeout > maximum_timeout)
    detail::Invalid(
        "reload.timeout",
        "positive candidate and nonnegative drain budgets, at most 24 h");
  auto bounded = load;
  const auto candidate_deadline = Clock::now() + options.candidate_timeout;
  if (!bounded.deadline || candidate_deadline < *bounded.deadline)
    bounded.deadline = candidate_deadline;
  detail::Boundary(bounded.stop, bounded.deadline);
  if (!SameRuntime(state_->runtime, backend.descriptor()))
    throw CompatibilityError(
        "Reload requires the same exact runtime and device");
  const auto reservation = Reservation(bounded);
  {
    std::lock_guard lock(state_->mutex);
    if (!state_->accepting || !state_->current)
      throw NotReady("Versioned inference admission is closed");
    if (state_->reloading || state_->previous)
      throw NotReady(
          "A candidate or previous version still owns the second slot");
    if (reservation > state_->policy.max_resident_bytes - state_->Reserved())
      detail::Resource("reload.resident_bytes");
    state_->loading = true;
    state_->reloading = true;
    state_->candidate_bytes = reservation;
    state_->peak_bytes = std::max(state_->peak_bytes, state_->Reserved());
  }
  struct LoadingGuard {
    std::shared_ptr<detail::ReloadState> state;
    bool armed = true;
    void Finish() {
      if (!armed) return;
      std::lock_guard lock(state->mutex);
      state->loading = false;
      state->reloading = false;
      state->candidate_bytes = 0;
      state->cv.notify_all();
      armed = false;
    }
    ~LoadingGuard() { Finish(); }
  } loading{state_};
  // Destruction order retains the candidate reservation through failed cleanup.
  auto candidate = backend.load(directory, verify, warmup, bounded);
  detail::Boundary(bounded.stop, bounded.deadline);
  if (!candidate)
    detail::Invalid("reload.candidate", "a fresh ready predictor");
  const auto metadata = candidate->info();
  if (metadata.health != HealthStatus::ready || metadata.active ||
      !SameRuntime(state_->runtime, metadata.runtime) ||
      !SameInput(state_->input, candidate->input_schema()) ||
      canonical_value_schema(state_->output) !=
          canonical_value_schema(candidate->output_schema()))
    throw CompatibilityError(
        "Reload candidate runtime or invocation schema is incompatible");
  auto entry = std::make_shared<detail::ReloadState::Entry>();
  entry->reservation = reservation;
  ReloadResult result;
  {
    std::lock_guard lock(state_->mutex);
    detail::Boundary(bounded.stop, bounded.deadline);
    if (!state_->accepting || !state_->current)
      throw NotReady("Versioned inference owner closed during candidate load");
    if (candidate == state_->current->predictor ||
        metadata.model_version ==
            state_->current->predictor->info().model_version)
      detail::Invalid("reload.candidate",
                      "a fresh predictor with a different model version");
    result.previous_version = state_->current->predictor->info().model_version;
    result.current_version = metadata.model_version;
    entry->predictor = std::move(candidate);
    state_->previous = std::move(state_->current);
    state_->current = std::move(entry);
    ++state_->publications;
    state_->loading = false;
    state_->candidate_bytes = 0;
    state_->cv.notify_all();
    // Do not copy owning metadata or allocate after the publication point.
  }
  // Publication is final. A drain failure must never report a failed reload.
  try {
    result.previous_drained =
        drain_previous(Clock::now() + options.drain_timeout, bounded.stop);
  } catch (...) {
    result.previous_drained = false;
  }
  return result;
}
bool ReloadablePredictor::drain_previous(Deadline deadline,
                                         std::stop_token stop) {
  std::unique_lock lock(state_->mutex);
  while (state_->previous &&
         (state_->previous->leases || state_->maintenance)) {
    if (Expired(deadline, stop)) return false;
    state_->cv.wait_for(lock, std::chrono::milliseconds(1));
  }
  if (!state_->previous) return true;
  auto previous = state_->previous;
  state_->maintenance = true;
  lock.unlock();
  bool drained = false;
  try {
    drained = previous->predictor->close(deadline, stop);
  } catch (...) {
    lock.lock();
    state_->maintenance = false;
    state_->cv.notify_all();
    throw;
  }
  lock.lock();
  if (drained) state_->previous.reset();
  state_->maintenance = false;
  state_->cv.notify_all();
  return drained;
}
ModelRuntimeInfo ReloadablePredictor::info() const {
  std::lock_guard lock(state_->mutex);
  auto result =
      state_->current ? state_->current->predictor->info() : state_->last;
  if (!state_->accepting && result.health == HealthStatus::ready)
    result.health = HealthStatus::draining;
  return result;
}
ReloadStatistics ReloadablePredictor::statistics() const {
  std::lock_guard lock(state_->mutex);
  return {.accepting = state_->accepting,
          .loading = state_->loading,
          .resident_instances = static_cast<std::size_t>(!!state_->current) +
                                static_cast<std::size_t>(!!state_->previous) +
                                static_cast<std::size_t>(state_->loading),
          .active_leases = state_->Leases(),
          .reserved_bytes = state_->Reserved(),
          .peak_reserved_bytes = state_->peak_bytes,
          .publications = state_->publications};
}
bool ReloadablePredictor::close(Deadline deadline, std::stop_token stop) {
  std::unique_lock lock(state_->mutex);
  state_->accepting = false;
  while (state_->reloading || state_->maintenance || state_->Leases()) {
    if (Expired(deadline, stop)) return false;
    state_->cv.wait_for(lock, std::chrono::milliseconds(1));
  }
  if (!state_->current && !state_->previous) return true;
  auto current = state_->current, previous = state_->previous;
  state_->maintenance = true;
  lock.unlock();
  bool closed = false;
  ModelRuntimeInfo last;
  try {
    closed = (!previous || previous->predictor->close(deadline, stop)) &&
             (!current || current->predictor->close(deadline, stop));
    if (closed && current) last = current->predictor->info();
  } catch (...) {
    lock.lock();
    state_->maintenance = false;
    state_->cv.notify_all();
    throw;
  }
  lock.lock();
  if (closed) {
    if (current) state_->last = std::move(last);
    state_->current.reset();
    state_->previous.reset();
  }
  state_->maintenance = false;
  state_->cv.notify_all();
  return closed;
}
}  // namespace jyppx::liuding::serving
