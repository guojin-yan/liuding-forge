// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
#include "detail.hpp"

#include <jyppx/liuding/serving/batcher.hpp>

#include <torch/cuda.h>

#include <algorithm>
#include <c10/core/InferenceMode.h>
#include <condition_variable>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>
namespace jyppx::liuding::serving {
namespace {
using Clock = std::chrono::steady_clock;
void BatchSchema(const ValueSchema& schema) {
  if (schema.kind != ValueKind::tensor || schema.nullable ||
      !schema.tensor_constraints)
    detail::Invalid("batch.schema", "nonnullable tensor schema");
  const auto& tensor = *schema.tensor_constraints;
  if (!tensor.rank || *tensor.rank < 1 || !tensor.shape ||
      tensor.shape->size() != static_cast<std::size_t>(*tensor.rank) ||
      tensor.shape->front())
    detail::Invalid("batch.schema",
                    "fixed rank and shape with dynamic leading dimension");
}
Tensor RequestTensor(const PredictRequest& request) {
  if (request.input.positional.size() != 1U || !request.input.keyword.empty() ||
      !request.input.positional.front().isTensor())
    detail::Invalid("batch.request", "one positional tensor and no keywords");
  auto tensor = request.input.positional.front().toTensor();
  if (!tensor.defined() || tensor.dim() < 1 || tensor.size(0) != 1)
    detail::Invalid("batch.request",
                    "defined tensor with leading dimension one");
  return tensor;
}
bool Compatible(const PredictRequest& a, const PredictRequest& b) {
  const auto& x = a.context;
  const auto& y = b.context;
  if (x.model_version != y.model_version || x.project_id != y.project_id ||
      x.principal_id != y.principal_id ||
      x.priority_class != y.priority_class ||
      x.deadline_class != y.deadline_class ||
      x.deployment_id != y.deployment_id)
    return false;
  const auto t = RequestTensor(a), u = RequestTensor(b);
  if (t.dim() != u.dim() || t.scalar_type() != u.scalar_type() ||
      t.device() != u.device())
    return false;
  for (std::int64_t i = 1; i < t.dim(); ++i)
    if (t.size(i) != u.size(i)) return false;
  return true;
}
std::exception_ptr Failure(std::stop_token stop, const Deadline& deadline) {
  try {
    detail::Boundary(stop, deadline);
  } catch (...) {
    return std::current_exception();
  }
  return {};
}
std::exception_ptr Shutdown() {
  try {
    throw Cancelled("Batch scheduler shutting down",
                    detail::Details("batch.close", "accepting scheduler"));
  } catch (...) {
    return std::current_exception();
  }
}
}  // namespace
class DynamicBatcher::Impl {
 public:
  struct Work {
    PredictRequest request;
    std::stop_token stop;
    Clock::time_point queued_at;
    std::uint64_t bytes = 0;
    std::promise<PredictResponse> promise;
    bool resolved = false;  // Only the scheduler worker accesses completion.
  };
  std::shared_ptr<Predictor> predictor;
  BatchOptions options;
  mutable std::mutex mutex;
  std::mutex join_mutex;
  std::condition_variable cv;
  std::deque<std::shared_ptr<Work>> queue;
  BatchStatistics stats;
  std::size_t reservations = 0;
  std::thread worker;
  Impl(std::shared_ptr<Predictor> p, BatchOptions o)
      : predictor(std::move(p)), options(o) {
    if (!predictor || !options.max_queue_size ||
        !options.max_resident_tensor_bytes || !options.max_batch_size ||
        options.max_batch_size > 4096U || options.max_wait.count() < 0 ||
        options.max_wait > std::chrono::seconds(1) ||
        options.max_oom_retries > 8U)
      detail::Invalid(
          "batch.options",
          "positive bounded queue/bytes/batch and wait/retry limits");
    const auto info = predictor->info();
    if (info.health != HealthStatus::ready || !info.runtime.tensor_batching ||
        !info.runtime.dynamic_shapes ||
        options.max_batch_size > info.runtime.max_batch_size)
      detail::Invalid("batch.capability",
                      "ready explicitly batching-capable runtime");
    const auto schema = predictor->input_schema();
    if (schema.positional.size() != 1U || !schema.keyword.empty() ||
        schema.allow_extra_keywords)
      detail::Invalid("batch.schema",
                      "exactly one positional tensor and no kwargs");
    BatchSchema(schema.positional.front());
    BatchSchema(predictor->output_schema());
    worker = std::thread([this] { Run(); });
  }
  void Fail(const std::shared_ptr<Work>& work, std::exception_ptr failure) {
    if (work->resolved) return;
    work->resolved = true;
    work->promise.set_exception(std::move(failure));
    std::lock_guard lock(mutex);
    --stats.active;
    stats.resident_tensor_bytes -= work->bytes;
    ++stats.failed;
    cv.notify_all();
  }
  void Success(const std::shared_ptr<Work>& work, PredictResponse response) {
    if (work->resolved) return;
    work->resolved = true;
    work->promise.set_value(std::move(response));
    std::lock_guard lock(mutex);
    --stats.active;
    stats.resident_tensor_bytes -= work->bytes;
    ++stats.completed;
    cv.notify_all();
  }
  void Execute(const std::vector<std::shared_ptr<Work>>& original) {
    // Iterative splitting has one finite retry budget for the entire group.
    std::deque<std::vector<std::shared_ptr<Work>>> pending;
    pending.push_back(original);
    std::size_t retries = 0;
    while (!pending.empty()) {
      auto group = std::move(pending.front());
      pending.pop_front();
      std::vector<std::shared_ptr<Work>> live;
      for (const auto& work : group) {
        if (auto failure = Failure(work->stop, work->request.deadline))
          Fail(work, std::move(failure));
        else
          live.push_back(work);
      }
      if (live.empty()) continue;
      try {
        c10::InferenceMode guard;
        std::vector<Tensor> tensors;
        tensors.reserve(live.size());
        for (const auto& work : live)
          tensors.push_back(RequestTensor(work->request));
        auto merged = live.front()->request;
        merged.input.positional.front() = torch::cat(tensors, 0);
        // A single member's cancellation/deadline must never abort its peers.
        merged.deadline = {};
        auto response = predictor->predict(merged);
        if (!response.output.value.isTensor())
          detail::Invalid("batch.output", "one tensor");
        const auto result = response.output.value.toTensor();
        if (result.dim() < 1 ||
            result.size(0) != static_cast<std::int64_t>(live.size()))
          detail::Invalid("batch.output",
                          "leading extent equal to admitted batch size");
        // Prepare all independent slices before fulfilling any promises. A
        // failure cannot fulfill one promise twice or silently lose peers.
        std::vector<PredictResponse> responses;
        responses.reserve(live.size());
        for (std::size_t i = 0; i < live.size(); ++i) {
          auto item = response;
          item.output.value =
              result.narrow(0, static_cast<std::int64_t>(i), 1).clone();
          item.context = live[i]->request.context;
          item.timing.queue =
              detail::Seconds(Clock::now() - live[i]->queued_at) -
              response.timing.total;
          item.timing.queue = std::max(0.0, item.timing.queue);
          item.timing.total =
              detail::Seconds(Clock::now() - live[i]->queued_at);
          responses.push_back(std::move(item));
        }
        if (result.is_cuda()) torch::cuda::synchronize(result.get_device());
        for (std::size_t i = 0; i < live.size(); ++i) {
          if (auto failure = Failure(live[i]->stop, live[i]->request.deadline))
            Fail(live[i], std::move(failure));
          else
            Success(live[i], std::move(responses[i]));
        }
      } catch (const RetryableOutOfMemory&) {
        const auto info = predictor->info();
        if (info.health == HealthStatus::ready && info.runtime.oom_retry_safe &&
            live.size() > 1U && retries < options.max_oom_retries) {
          ++retries;
          {
            std::lock_guard lock(mutex);
            ++stats.oom_retries;
          }
          const auto middle =
              live.begin() + static_cast<std::ptrdiff_t>(live.size() / 2U);
          pending.emplace_front(middle, live.end());
          pending.emplace_front(live.begin(), middle);
        } else {
          const auto failure = std::current_exception();
          for (const auto& work : live) Fail(work, failure);
        }
      } catch (...) {
        const auto failure = std::current_exception();
        for (const auto& work : live) Fail(work, failure);
      }
    }
  }
  void Run() noexcept {
    std::vector<std::shared_ptr<Work>> batch;
    try {
      for (;;) {
        batch.clear();
        {
          std::unique_lock lock(mutex);
          cv.wait(lock, [&] { return !stats.accepting || !queue.empty(); });
          if (!stats.accepting) {
            while (!queue.empty()) {
              auto work = queue.front();
              queue.pop_front();
              --stats.queued;
              work->promise.set_exception(Shutdown());
              stats.resident_tensor_bytes -= work->bytes;
              ++stats.failed;
            }
            if (reservations == 0U) break;
            cv.wait(lock, [&] { return reservations == 0U; });
            continue;
          }
          auto end = queue.front()->queued_at + options.max_wait;
          if (queue.front()->request.deadline)
            end = std::min(end, *queue.front()->request.deadline);
          std::size_t count = 1;
          std::uint64_t charged = queue.front()->bytes;
          const auto limits = predictor->request_limits();
          const auto input_bound = limits.max_tensor_bytes;
          std::uint64_t capacity = std::min<std::uint64_t>(
              options.max_batch_size, limits.max_dimension);
          // Each queued row already passed resource validation. Bound every
          // merged shape prefix, including prefixes before a zero dimension.
          std::uint64_t prefix = 1;
          for (const auto extent :
               RequestTensor(queue.front()->request).sizes()) {
            if (extent == 0) break;
            prefix *= static_cast<std::uint64_t>(extent);
            capacity = std::min(capacity, limits.max_elements / prefix);
          }
          if (charged != 0)
            capacity = std::min(capacity, input_bound / charged);
          while (count < queue.size() && count < capacity &&
                 Compatible(queue.front()->request, queue[count]->request) &&
                 queue[count]->bytes <= input_bound - charged) {
            charged += queue[count]->bytes;
            if (queue[count]->request.deadline)
              end = std::min(end, *queue[count]->request.deadline);
            ++count;
          }
          if (!Failure(queue.front()->stop, queue.front()->request.deadline) &&
              count < capacity && count == queue.size() && Clock::now() < end) {
            // Short bounded waits notice externally owned stop_tokens without
            // callbacks that could outlive the scheduler.
            cv.wait_until(
                lock,
                std::min(end, Clock::now() + std::chrono::milliseconds(1)));
            continue;
          }
          for (std::size_t i = 0; i < count; ++i) {
            batch.push_back(queue.front());
            queue.pop_front();
            --stats.queued;
            ++stats.active;
          }
        }
        Execute(batch);
      }
    } catch (...) {
      const auto failure = std::current_exception();
      for (const auto& work : batch) {
        try {
          Fail(work, failure);
        } catch (...) {
        }
      }
      std::lock_guard lock(mutex);
      stats.accepting = false;
      while (!queue.empty()) {
        auto work = queue.front();
        queue.pop_front();
        --stats.queued;
        try {
          work->promise.set_exception(failure);
        } catch (...) {
        }
        stats.resident_tensor_bytes -= work->bytes;
        ++stats.failed;
      }
    }
    std::lock_guard lock(mutex);
    stats.stopped = true;
    cv.notify_all();
  }
};
DynamicBatcher::DynamicBatcher(std::shared_ptr<Predictor> predictor,
                               BatchOptions options)
    : impl_(std::make_unique<Impl>(std::move(predictor), options)) {}
DynamicBatcher::~DynamicBatcher() {
  try {
    close();
  } catch (...) {
  }
}
std::future<PredictResponse> DynamicBatcher::submit(
    const PredictRequest& request, std::stop_token stop) {
  return detail::Backend("batch.submit", [&] {
    detail::Boundary(stop, request.deadline);
    if (impl_->predictor->info().health != HealthStatus::ready)
      throw NotReady("Batch predictor unavailable",
                     detail::Details("batch.submit", "ready predictor"));
    const auto tensor = RequestTensor(request);
    if (tensor.numel() < 0 || tensor.element_size() == 0 ||
        static_cast<std::uint64_t>(tensor.numel()) >
            impl_->options.max_resident_tensor_bytes / tensor.element_size())
      detail::Resource("batch.resident_bytes");
    const auto bytes =
        static_cast<std::uint64_t>(tensor.numel()) * tensor.element_size();
    auto work = std::make_shared<Impl::Work>();
    work->stop = stop;
    work->bytes = bytes;
    auto future = work->promise.get_future();
    {
      std::lock_guard lock(impl_->mutex);
      if (!impl_->stats.accepting)
        throw NotReady("Batch scheduler closed",
                       detail::Details("batch.submit", "accepting scheduler"));
      if (impl_->stats.queued + impl_->reservations >=
              impl_->options.max_queue_size ||
          bytes > impl_->options.max_resident_tensor_bytes -
                      impl_->stats.resident_tensor_bytes)
        detail::Resource("batch.queue");
      ++impl_->reservations;
      impl_->stats.resident_tensor_bytes += bytes;
    }
    try {
      work->request = impl_->predictor->snapshot_request(request);
      detail::Boundary(stop, request.deadline);
      work->queued_at = Clock::now();
      std::lock_guard lock(impl_->mutex);
      if (!impl_->stats.accepting)
        throw NotReady("Batch scheduler draining",
                       detail::Details("batch.submit", "accepting scheduler"));
      impl_->queue.push_back(work);
      --impl_->reservations;
      ++impl_->stats.queued;
      ++impl_->stats.accepted;
      impl_->cv.notify_all();
    } catch (...) {
      std::lock_guard lock(impl_->mutex);
      --impl_->reservations;
      impl_->stats.resident_tensor_bytes -= bytes;
      impl_->cv.notify_all();
      throw;
    }
    return future;
  });
}
BatchStatistics DynamicBatcher::statistics() const {
  std::lock_guard lock(impl_->mutex);
  return impl_->stats;
}
std::shared_ptr<Predictor> DynamicBatcher::predictor() const {
  return impl_->predictor;
}
bool DynamicBatcher::close(Deadline deadline, std::stop_token stop) {
  std::unique_lock lock(impl_->mutex);
  impl_->stats.accepting = false;
  impl_->cv.notify_all();
  while (!impl_->stats.stopped || impl_->reservations) {
    if (stop.stop_requested() || (deadline && Clock::now() >= *deadline))
      return false;
    impl_->cv.wait_for(lock, std::chrono::milliseconds(1));
  }
  lock.unlock();
  std::lock_guard join(impl_->join_mutex);
  if (impl_->worker.joinable()) impl_->worker.join();
  return true;
}
}  // namespace jyppx::liuding::serving
