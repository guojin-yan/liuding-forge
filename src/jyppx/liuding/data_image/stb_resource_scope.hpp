// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

#ifndef JYPPX_LIUDING_DATA_IMAGE_STB_RESOURCE_SCOPE_HPP
#define JYPPX_LIUDING_DATA_IMAGE_STB_RESOURCE_SCOPE_HPP

#include <jyppx/liuding/data_image/resource_budget.hpp>

#include <cstddef>
#include <exception>
#include <memory>

namespace jyppx::liuding::data::image::codec_detail {

struct StbAllocation;

// Each synchronous codec call uses its calling thread's budget. Allocation
// callbacks return null on failure; exceptions are rethrown after stb returns.
class StbResourceScope final {
 public:
  explicit StbResourceScope(std::shared_ptr<ImageResourceBudget> budget,
                            std::size_t limit) noexcept;
  ~StbResourceScope();
  StbResourceScope(const StbResourceScope&) = delete;
  StbResourceScope& operator=(const StbResourceScope&) = delete;
  void rethrow_if_failed() const;

  std::shared_ptr<ImageResourceBudget> budget;
  std::exception_ptr failure;

 private:
  friend void* allocate(std::size_t bytes) noexcept;
  friend void release(void* pointer) noexcept;
  void check_capacity(std::size_t bytes) const;
  std::size_t limit_;
  std::size_t current_ = 0;
  StbAllocation* head_ = nullptr;
  StbResourceScope* previous_;
};

}  // namespace jyppx::liuding::data::image::codec_detail

#endif  // JYPPX_LIUDING_DATA_IMAGE_STB_RESOURCE_SCOPE_HPP
