// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

#ifndef JYPPX_LIUDING_DATA_IMAGE_STB_WRITE_RESOURCE_SCOPE_HPP
#define JYPPX_LIUDING_DATA_IMAGE_STB_WRITE_RESOURCE_SCOPE_HPP

#include <jyppx/liuding/data_image/resource_budget.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>

namespace jyppx::liuding::data::image::codec_detail {

// The writer is compiled as C++. Its stretchy-buffer code requires allocation
// to succeed, so throwing hooks stop it before it can dereference a null
// pointer. The intrusive owner list releases every live allocation when an
// exception unwinds through stb, including failures in the output callback.
class StbWriteResourceScope final {
  struct alignas(std::max_align_t) Allocation {
    std::size_t bytes;
    ImageResourceLease lease;
    StbWriteResourceScope* owner;
    Allocation* previous;
    Allocation* next;
  };

 public:
  StbWriteResourceScope(std::shared_ptr<ImageResourceBudget> budget,
                        std::size_t limit, std::size_t staging) noexcept
      : budget_(std::move(budget)),
        limit_(limit),
        current_(staging),
        previous_(current_scope_) {
    current_scope_ = this;
  }

  ~StbWriteResourceScope() {
    while (head_) Free(head_ + 1);
    current_scope_ = previous_;
  }

  StbWriteResourceScope(const StbWriteResourceScope&) = delete;
  StbWriteResourceScope& operator=(const StbWriteResourceScope&) = delete;

  static void* Allocate(std::size_t bytes) {
    auto* owner = current_scope_;
    if (owner && bytes > owner->limit_ - owner->current_)
      throw ResourceExhausted(
          "Image encode working limit exceeded",
          ErrorDetails{.component = "data.image.encoder",
                       .operation = "encode.working",
                       .expected = std::to_string(owner->limit_),
                       .actual = std::to_string(owner->current_) + "+" +
                                 std::to_string(bytes)});
    if (bytes > std::numeric_limits<std::size_t>::max() - sizeof(Allocation))
      throw std::bad_alloc();
    ImageResourceLease lease;
    if (owner && owner->budget_)
      lease = owner->budget_->reserve(ImageResourceKind::codec_scratch_bytes,
                                      bytes);
    auto* storage = std::malloc(sizeof(Allocation) + bytes);
    if (!storage) throw std::bad_alloc();
    auto* allocation =
        new (storage) Allocation{bytes, std::move(lease), owner, nullptr,
                                 owner ? owner->head_ : nullptr};
    if (owner) {
      if (owner->head_) owner->head_->previous = allocation;
      owner->head_ = allocation;
      owner->current_ += bytes;
    }
    return allocation + 1;
  }

  static void Free(void* pointer) noexcept {
    if (!pointer) return;
    auto* allocation = static_cast<Allocation*>(pointer) - 1;
    if (auto* owner = allocation->owner) {
      if (allocation->previous)
        allocation->previous->next = allocation->next;
      else
        owner->head_ = allocation->next;
      if (allocation->next) allocation->next->previous = allocation->previous;
      owner->current_ -= allocation->bytes;
    }
    auto lease = std::move(allocation->lease);
    allocation->~Allocation();
    std::free(allocation);
  }

  static void* Reallocate(void* pointer, std::size_t bytes) {
    if (!pointer) return Allocate(bytes);
    if (!bytes) {
      Free(pointer);
      return nullptr;
    }
    // Charge both buffers before allocation; failure leaves the old buffer
    // registered so the scope can release it while unwinding.
    auto* replacement = Allocate(bytes);
    const auto* allocation = static_cast<Allocation*>(pointer) - 1;
    std::memcpy(replacement, pointer, std::min(bytes, allocation->bytes));
    Free(pointer);
    return replacement;
  }

 private:
  inline static thread_local StbWriteResourceScope* current_scope_ = nullptr;
  std::shared_ptr<ImageResourceBudget> budget_;
  std::size_t limit_;
  std::size_t current_;
  Allocation* head_ = nullptr;
  StbWriteResourceScope* previous_;
};

}  // namespace jyppx::liuding::data::image::codec_detail

#endif  // JYPPX_LIUDING_DATA_IMAGE_STB_WRITE_RESOURCE_SCOPE_HPP
