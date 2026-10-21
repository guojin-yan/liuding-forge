// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
#ifndef JYPPX_SERVING_DETAIL_HPP
#define JYPPX_SERVING_DETAIL_HPP
#include <jyppx/foundation/error.hpp>
#include <jyppx/serving/runtime.hpp>

#include <exception>
#include <mutex>
#include <string>
#include <utility>
namespace jyppx::serving::detail {
inline ErrorDetails Details(std::string operation, std::string expected) {
  return {
      .component = "serving",
      .operation = std::move(operation),
      .expected = std::move(expected),
      .actual = {},
      .hint =
          "Inspect the declared serving backend, schema and resource limits."};
}
[[noreturn]] inline void Invalid(std::string operation, std::string expected) {
  throw InvalidArgument("Invalid serving contract",
                        Details(std::move(operation), std::move(expected)));
}
[[noreturn]] inline void Resource(std::string operation) {
  throw ResourceExhausted(
      "Serving resource budget exceeded",
      Details(std::move(operation), "within the declared budget"));
}
inline void Boundary(std::stop_token stop, const Deadline& deadline) {
  if (stop.stop_requested())
    throw Cancelled("Inference cancelled", Details("cancel", "active request"));
  if (deadline && std::chrono::steady_clock::now() >= *deadline)
    throw DeadlineExceeded("Inference deadline elapsed",
                           Details("deadline", "future deadline"));
}
inline double Seconds(std::chrono::steady_clock::duration d) {
  return std::chrono::duration<double>(d).count();
}
inline void Text(std::string_view text, bool required = false) {
  if ((required && text.empty()) || text.size() > 256U)
    Invalid("context", "bounded nonempty IDs where required");
  for (const unsigned char ch : text)
    if (ch < 0x20U || ch > 0x7eU)
      Invalid("context", "printable ASCII correlation IDs");
}
template <class F>
decltype(auto) Backend(const char* operation, F&& function) {
  try {
    return std::forward<F>(function)();
  } catch (const Error&) {
    throw;
  } catch (const c10::OutOfMemoryError&) {
    std::throw_with_nested(
        ResourceExhausted("Inference backend allocation failed",
                          Details(operation, "available backend memory")));
  } catch (const c10::Error& e) {
    jyppx::detail::rethrow_backend_error(
        e, {.component = "serving",
            .operation = operation,
            .failure_kind = jyppx::detail::BackendFailureKind::unknown,
            .expected = "supported inference operation",
            .hint = "Inspect nested backend cause."});
  } catch (const std::bad_alloc&) {
    std::throw_with_nested(ResourceExhausted(
        "Inference allocation failed", Details(operation, "available memory")));
  }
}
inline bool IsOom(const std::exception& error, std::size_t depth = 0) {
  if (dynamic_cast<const c10::OutOfMemoryError*>(&error) ||
      dynamic_cast<const std::bad_alloc*>(&error) ||
      dynamic_cast<const RetryableOutOfMemory*>(&error))
    return true;
  if (depth < 8U) {
    try {
      std::rethrow_if_nested(error);
    } catch (const std::exception& nested) {
      return IsOom(nested, depth + 1);
    } catch (...) {
    }
  }
  return false;
}
c10::IValue Snapshot(const c10::IValue& value);
nn::Invocation Snapshot(const nn::Invocation& value);
void Context(const RequestContext& context, std::string_view version);
bool Equal(const c10::IValue& actual, const c10::IValue& expected, double rtol,
           double atol);
}  // namespace jyppx::serving::detail
#endif
