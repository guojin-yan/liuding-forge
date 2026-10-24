// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_MODELS_DETAIL_HPP
#define JYPPX_LIUDING_MODELS_DETAIL_HPP
#include <jyppx/liuding/core/types.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <c10/util/Exception.h>
#include <exception>
#include <new>
#include <string>
#include <utility>
namespace jyppx::liuding::models::detail {
inline ErrorDetails Details(std::string operation, std::string expected) {
  return {.component = "models",
          .operation = std::move(operation),
          .expected = std::move(expected),
          .actual = {},
          .hint = "Inspect the documented model and weight contract."};
}
[[noreturn]] inline void Invalid(std::string operation, std::string expected) {
  throw InvalidArgument("Invalid model contract",
                        Details(std::move(operation), std::move(expected)));
}
template <class Function>
decltype(auto) Backend(const char* operation, Function&& function) {
  try {
    return std::forward<Function>(function)();
  } catch (const Error&) {
    throw;
  } catch (const c10::OutOfMemoryError&) {
    std::throw_with_nested(
        ResourceExhausted("Model backend memory exhausted",
                          Details(operation, "available memory")));
  } catch (const c10::Error& error) {
    jyppx::liuding::detail::rethrow_backend_error(
        error,
        {.component = "models",
         .operation = operation,
         .failure_kind = jyppx::liuding::detail::BackendFailureKind::unknown,
         .expected = "a supported model invocation",
         .hint = "Inspect the nested backend cause."});
  } catch (const std::bad_alloc&) {
    std::throw_with_nested(ResourceExhausted(
        "Model allocation failed", Details(operation, "available memory")));
  }
}
inline void FloatingInput(const Tensor& input, const Tensor& parameter,
                          std::int64_t rank) {
  if (!input.defined() || input.layout() != torch::kStrided ||
      input.dim() != rank || input.is_quantized() ||
      !input.is_floating_point() ||
      input.scalar_type() != parameter.scalar_type() ||
      input.device() != parameter.device())
    Invalid("forward.input",
            "dense floating input matching module rank/device/dtype");
  for (const auto extent : input.sizes())
    if (extent <= 0) Invalid("forward.shape", "positive input dimensions");
}
}  // namespace jyppx::liuding::models::detail
#endif
