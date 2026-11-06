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
#include <utility>

namespace jyppx::liuding::data::image::codec_detail {
namespace {
thread_local StbResourceScope* current_scope = nullptr;

struct alignas(std::max_align_t) Allocation {
  std::size_t bytes;
  ImageResourceLease lease;
};
}  // namespace

StbResourceScope::StbResourceScope(
    std::shared_ptr<ImageResourceBudget> selected_budget) noexcept
    : budget(std::move(selected_budget)), previous_(current_scope) {
  current_scope = this;
}

StbResourceScope::~StbResourceScope() { current_scope = previous_; }

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
    if (current_scope && current_scope->failure) return nullptr;
    if (bytes > std::numeric_limits<std::size_t>::max() - sizeof(Allocation))
      throw std::bad_alloc();
    ImageResourceLease lease;
    if (current_scope && current_scope->budget)
      lease = current_scope->budget->reserve(
          ImageResourceKind::codec_scratch_bytes, bytes);
    auto* storage = std::malloc(sizeof(Allocation) + bytes);
    if (!storage) throw std::bad_alloc();
    auto* allocation = new (storage) Allocation{bytes, std::move(lease)};
    return allocation + 1;
  } catch (...) {
    if (current_scope && !current_scope->failure)
      current_scope->failure = std::current_exception();
    return nullptr;
  }
}

void release(void* pointer) noexcept {
  if (!pointer) return;
  auto* allocation = static_cast<Allocation*>(pointer) - 1;
  // Hold the charge until the storage itself has been returned to malloc.
  auto lease = std::move(allocation->lease);
  allocation->~Allocation();
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
  const auto* allocation = static_cast<Allocation*>(pointer) - 1;
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
