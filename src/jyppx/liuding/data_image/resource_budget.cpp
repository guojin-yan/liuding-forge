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
#include <limits>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

namespace jyppx::liuding::data::image {
namespace {

constexpr std::size_t kKindCount =
    static_cast<std::size_t>(ImageResourceKind::disk_spool_bytes) + 1U;

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
          limits.max_disk_spool_bytes};
}

}  // namespace

namespace resource_detail {

struct ImageResourceBudgetState {
  explicit ImageResourceBudgetState(const ImageResourceBudgetLimits& values)
      : limits(Limits(values)) {}

  mutable std::mutex mutex;
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
  const auto index = Index(kind_);
  if (index < state_->current.size()) {
    std::scoped_lock lock(state_->mutex);
    state_->current[index] = amount_ > state_->current[index]
                                 ? 0
                                 : state_->current[index] - amount_;
  }
  state_.reset();
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
