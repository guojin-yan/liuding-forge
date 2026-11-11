// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "stb_resource_scope.hpp"

#include <jyppx/liuding/foundation/error.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <utility>

namespace jyppx::liuding::data::image::codec_detail {
namespace {
thread_local StbResourceScope* current_scope = nullptr;

}  // namespace

struct alignas(std::max_align_t) StbAllocation {
  std::size_t bytes;
  ImageResourceLease lease;
  StbResourceScope* owner;
  StbAllocation* previous;
  StbAllocation* next;
};

void release(void* pointer) noexcept;

StbResourceScope::StbResourceScope(
    std::shared_ptr<ImageResourceBudget> selected_budget,
    std::size_t limit) noexcept
    : budget(std::move(selected_budget)),
      limit_(limit),
      previous_(current_scope) {
  current_scope = this;
}

StbResourceScope::~StbResourceScope() {
  // Also reclaim any backend allocation left live on a failed decode.
  while (head_) release(head_ + 1);
  current_scope = previous_;
}

void StbResourceScope::check_capacity(std::size_t bytes) const {
  if (bytes > limit_ - current_)
    throw ResourceExhausted(
        "Image decode working limit exceeded",
        ErrorDetails{
            .component = "data.image.decoder",
            .operation = "decode.working",
            .expected = std::to_string(limit_),
            .actual = std::to_string(current_) + "+" + std::to_string(bytes)});
}

void StbResourceScope::rethrow_if_failed() const {
  if (!failure) return;
  try {
    std::rethrow_exception(failure);
  } catch (const std::bad_alloc&) {
    throw ResourceExhausted("Image codec allocation failed");
  }
}

void* allocate(std::size_t bytes) noexcept {
  try {
    auto* owner = current_scope;
    if (owner && owner->failure) return nullptr;
    if (owner) owner->check_capacity(bytes);
    if (bytes > std::numeric_limits<std::size_t>::max() - sizeof(StbAllocation))
      throw std::bad_alloc();
    ImageResourceLease lease;
    if (current_scope && current_scope->budget)
      lease = current_scope->budget->reserve(
          ImageResourceKind::codec_scratch_bytes, bytes);
    auto* storage = std::malloc(sizeof(StbAllocation) + bytes);
    if (!storage) throw std::bad_alloc();
    auto* allocation =
        new (storage) StbAllocation{bytes, std::move(lease), owner, nullptr,
                                    owner ? owner->head_ : nullptr};
    if (owner) {
      if (owner->head_) owner->head_->previous = allocation;
      owner->head_ = allocation;
      owner->current_ += bytes;
    }
    return allocation + 1;
  } catch (...) {
    if (current_scope && !current_scope->failure)
      current_scope->failure = std::current_exception();
    return nullptr;
  }
}

void release(void* pointer) noexcept {
  if (!pointer) return;
  auto* allocation = static_cast<StbAllocation*>(pointer) - 1;
  if (auto* owner = allocation->owner) {
    if (allocation->previous)
      allocation->previous->next = allocation->next;
    else
      owner->head_ = allocation->next;
    if (allocation->next) allocation->next->previous = allocation->previous;
    owner->current_ -= allocation->bytes;
  }
  // Hold the charge until the storage itself has been returned to malloc.
  auto lease = std::move(allocation->lease);
  allocation->~StbAllocation();
  std::free(allocation);
}

void* reallocate(void* pointer, std::size_t bytes) noexcept {
  if (!pointer) return allocate(bytes);
  if (bytes == 0) {
    release(pointer);
    return nullptr;
  }
  auto* replacement = allocate(bytes);
  if (!replacement) return nullptr;  // Preserve the old allocation on failure.
  const auto* allocation = static_cast<StbAllocation*>(pointer) - 1;
  std::memcpy(replacement, pointer, std::min(bytes, allocation->bytes));
  release(pointer);
  return replacement;
}

}  // namespace jyppx::liuding::data::image::codec_detail

#define STBI_MALLOC(size) \
  jyppx::liuding::data::image::codec_detail::allocate(size)
#define STBI_REALLOC(pointer, size) \
  jyppx::liuding::data::image::codec_detail::reallocate(pointer, size)
#define STBI_FREE(pointer) \
  jyppx::liuding::data::image::codec_detail::release(pointer)
#define STBI_NO_STDIO
#define STBI_MAX_DIMENSIONS 65536
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
