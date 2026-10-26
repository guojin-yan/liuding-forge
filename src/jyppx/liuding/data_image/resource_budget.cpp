// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data_image/resource_budget.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <algorithm>
#include <array>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

namespace jyppx::liuding::data::image {
namespace {

constexpr std::size_t kKindCount =
    static_cast<std::size_t>(ImageResourceKind::pipeline_inflight_bytes) + 1U;

std::size_t Index(ImageResourceKind kind) noexcept {
  return static_cast<std::size_t>(kind);
}

std::string_view KindName(ImageResourceKind kind) noexcept {
  switch (kind) {
    case ImageResourceKind::encoded_bytes:
      return "encoded_bytes";
    case ImageResourceKind::decoded_bytes:
      return "decoded_bytes";
    case ImageResourceKind::pixels:
      return "pixels";
    case ImageResourceKind::codec_scratch_bytes:
      return "codec_scratch_bytes";
    case ImageResourceKind::annotation_bytes:
      return "annotation_bytes";
    case ImageResourceKind::index_bytes:
      return "index_bytes";
    case ImageResourceKind::pinned_bytes:
      return "pinned_bytes";
    case ImageResourceKind::gpu_bytes:
      return "gpu_bytes";
    case ImageResourceKind::staged_batches:
      return "staged_batches";
    case ImageResourceKind::render_queue:
      return "render_queue";
    case ImageResourceKind::encoded_queue_bytes:
      return "encoded_queue_bytes";
    case ImageResourceKind::disk_spool_bytes:
      return "disk_spool_bytes";
    case ImageResourceKind::pipeline_inflight_batches:
      return "pipeline_inflight_batches";
    case ImageResourceKind::pipeline_inflight_bytes:
      return "pipeline_inflight_bytes";
  }
  return "unknown";
}

std::array<std::uint64_t, kKindCount> Limits(
    const ImageResourceBudgetLimits& limits) noexcept {
  return {limits.max_encoded_bytes,
          limits.max_decoded_bytes,
          limits.max_pixels,
          limits.max_codec_scratch_bytes,
          limits.max_annotation_bytes,
          limits.max_index_bytes,
          limits.max_pinned_bytes,
          limits.max_gpu_bytes,
          limits.max_staged_batches,
          limits.max_render_queue,
          limits.max_encoded_queue_bytes,
          limits.max_disk_spool_bytes,
          limits.max_pipeline_inflight_batches,
          limits.max_pipeline_inflight_bytes};
}

}  // namespace

namespace resource_detail {

struct ImageResourceBudgetState {
  explicit ImageResourceBudgetState(const ImageResourceBudgetLimits& values)
      : limits(Limits(values)) {}

  mutable std::mutex mutex;
  std::condition_variable_any condition;
  std::array<std::uint64_t, kKindCount> limits{};
  std::array<std::uint64_t, kKindCount> current{};
  std::array<std::uint64_t, kKindCount> peak{};
};

}  // namespace resource_detail

ImageResourceLease::ImageResourceLease(
    std::shared_ptr<resource_detail::ImageResourceBudgetState> state,
    ImageResourceKind kind, std::uint64_t amount) noexcept
    : state_(std::move(state)), kind_(kind), amount_(amount) {}

ImageResourceLease::~ImageResourceLease() { release(); }

ImageResourceLease::ImageResourceLease(ImageResourceLease&& other) noexcept
    : state_(std::move(other.state_)),
      kind_(other.kind_),
      amount_(std::exchange(other.amount_, 0)) {}

ImageResourceLease& ImageResourceLease::operator=(
    ImageResourceLease&& other) noexcept {
  if (this == &other) return *this;
  release();
  state_ = std::move(other.state_);
  kind_ = other.kind_;
  amount_ = std::exchange(other.amount_, 0);
  return *this;
}

void ImageResourceLease::release() noexcept {
  if (!state_ || amount_ == 0) {
    state_.reset();
    amount_ = 0;
    return;
  }
  auto state = std::move(state_);
  const auto index = Index(kind_);
  if (index < state->current.size()) {
    {
      std::scoped_lock lock(state->mutex);
      state->current[index] =
          amount_ > state->current[index] ? 0 : state->current[index] - amount_;
    }
    state->condition.notify_all();
  }
  amount_ = 0;
}

ImageResourceBudget::ImageResourceBudget(ImageResourceBudgetLimits limits)
    : state_(
          std::make_shared<resource_detail::ImageResourceBudgetState>(limits)) {}

ImageResourceBudget::~ImageResourceBudget() = default;

ImageResourceLease ImageResourceLease::reserve_related(
    ImageResourceKind kind, std::uint64_t amount) const {
  if (!active())
    throw InvalidArgument(
        "Cannot reserve from an inactive image resource lease",
        ErrorDetails{.component = "data.image.resource_budget",
                     .operation = "reserve.related"});
  return Reserve(state_, kind, amount);
}

ImageResourceLease ImageResourceLease::Reserve(
    const std::shared_ptr<resource_detail::ImageResourceBudgetState>& state,
    ImageResourceKind kind, std::uint64_t amount) {
  if (amount == 0) return {};
  const auto index = Index(kind);
  if (index >= state->limits.size()) {
    throw InvalidArgument(
        "Unknown image resource category",
        ErrorDetails{.component = "data.image.resource_budget",
                     .operation = "reserve.kind"});
  }

  std::uint64_t current = 0;
  std::uint64_t limit = 0;
  {
    std::scoped_lock lock(state->mutex);
    current = state->current[index];
    limit = state->limits[index];
    if (amount <= limit - std::min(current, limit) && current <= limit &&
        amount <= std::numeric_limits<std::uint64_t>::max() - current) {
      state->current[index] = current + amount;
      state->peak[index] = std::max(state->peak[index], current + amount);
      return ImageResourceLease(state, kind, amount);
    }
  }

  throw ResourceExhausted(
      "Image aggregate resource budget exceeded",
      ErrorDetails{
          .component = "data.image.resource_budget",
          .operation = "reserve." + std::string(KindName(kind)),
          .expected = std::to_string(limit),
          .actual = std::to_string(current) + "+" + std::to_string(amount),
          .hint =
              "Release an earlier stage lease or lower the batch/output size"});
}

ImageResourceLease ImageResourceBudget::reserve(ImageResourceKind kind,
                                                std::uint64_t amount) {
  return ImageResourceLease::Reserve(state_, kind, amount);
}

ImageResourceLease ImageResourceBudget::reserve_wait(ImageResourceKind kind,
                                                     std::uint64_t amount,
                                                     std::stop_token stop) {
  if (amount == 0) return {};
  const auto index = Index(kind);
  if (index >= state_->limits.size()) {
    throw InvalidArgument(
        "Unknown image resource category",
        ErrorDetails{.component = "data.image.resource_budget",
                     .operation = "reserve_wait.kind"});
  }
  std::unique_lock lock(state_->mutex);
  const auto limit = state_->limits[index];
  if (amount > limit) {
    throw ResourceExhausted(
        "Image reservation exceeds aggregate resource limit",
        ErrorDetails{.component = "data.image.resource_budget",
                     .operation = "reserve_wait." + std::string(KindName(kind)),
                     .expected = std::to_string(limit),
                     .actual = std::to_string(amount)});
  }
  if (stop.stop_requested())
    throw Cancelled("Image resource admission cancelled");
  if (!state_->condition.wait(lock, stop, [&] {
        return amount <= limit - state_->current[index];
      })) {
    throw Cancelled("Image resource admission cancelled");
  }
  if (stop.stop_requested())
    throw Cancelled("Image resource admission cancelled");
  state_->current[index] += amount;
  state_->peak[index] = std::max(state_->peak[index], state_->current[index]);
  return ImageResourceLease(state_, kind, amount);
}

ImageResourceLeaseSet ImageResourceBudget::reserve_wait_all(
    std::span<const ImageResourceRequest> requests, std::stop_token stop) {
  std::array<std::uint64_t, kKindCount> amounts{};
  for (const auto& request : requests) {
    const auto index = Index(request.kind);
    if (index >= amounts.size()) {
      throw InvalidArgument(
          "Unknown image resource category",
          ErrorDetails{.component = "data.image.resource_budget",
                       .operation = "reserve_wait_all.kind"});
    }
    if (request.amount >
        std::numeric_limits<std::uint64_t>::max() - amounts[index]) {
      throw ResourceExhausted("Image admission amount overflow");
    }
    amounts[index] += request.amount;
  }
  ImageResourceLeaseSet leases;
  leases.reserve(static_cast<std::size_t>(
      std::count_if(amounts.begin(), amounts.end(),
                    [](std::uint64_t amount) { return amount != 0; })));
  std::unique_lock lock(state_->mutex);
  for (std::size_t index = 0; index < amounts.size(); ++index) {
    if (amounts[index] > state_->limits[index]) {
      throw ResourceExhausted(
          "Image reservation exceeds aggregate resource limit",
          ErrorDetails{.component = "data.image.resource_budget",
                       .operation = "reserve_wait_all." +
                                    std::string(KindName(
                                        static_cast<ImageResourceKind>(index))),
                       .expected = std::to_string(state_->limits[index]),
                       .actual = std::to_string(amounts[index])});
    }
  }
  if (stop.stop_requested())
    throw Cancelled("Image resource admission cancelled");
  if (!state_->condition.wait(lock, stop, [&] {
        for (std::size_t index = 0; index < amounts.size(); ++index) {
          if (amounts[index] > state_->limits[index] - state_->current[index])
            return false;
        }
        return true;
      })) {
    throw Cancelled("Image resource admission cancelled");
  }
  if (stop.stop_requested())
    throw Cancelled("Image resource admission cancelled");
  for (std::size_t index = 0; index < amounts.size(); ++index) {
    if (!amounts[index]) continue;
    state_->current[index] += amounts[index];
    state_->peak[index] = std::max(state_->peak[index], state_->current[index]);
    leases.push_back(ImageResourceLease(
        state_, static_cast<ImageResourceKind>(index), amounts[index]));
  }
  return leases;
}

std::uint64_t ImageResourceBudget::limit(ImageResourceKind kind) const noexcept {
  const auto index = Index(kind);
  if (index >= state_->limits.size()) return 0;
  std::scoped_lock lock(state_->mutex);
  return state_->limits[index];
}

ImageResourceUsage ImageResourceBudget::usage(
    ImageResourceKind kind) const noexcept {
  const auto index = Index(kind);
  if (index >= state_->limits.size()) return {};
  std::scoped_lock lock(state_->mutex);
  return {.current = state_->current[index],
          .peak = state_->peak[index],
          .limit = state_->limits[index]};
}

}  // namespace jyppx::liuding::data::image
