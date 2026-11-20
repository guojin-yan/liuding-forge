// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
#ifndef JYPPX_FOUNDATION_NESTED_ERROR_HPP
#define JYPPX_FOUNDATION_NESTED_ERROR_HPP

#include <exception>
#include <type_traits>
#include <utility>

namespace jyppx::liuding::detail {

// exception_ptr may copy an exception while capturing or rethrowing it. Keep
// that copy distinct from initial cause capture: the existing nested_exception
// must be copied along with the error. The explicit copy constructor avoids a
// nested-exception copy failure with clang-cl and the MSVC STL.
template <class ErrorType>
class CapturedNestedError : public ErrorType, public std::nested_exception {
 public:
  explicit CapturedNestedError(ErrorType error) : ErrorType(std::move(error)) {}

  CapturedNestedError(const CapturedNestedError& error)
      : ErrorType(error), std::nested_exception(error) {}
};

template <class ErrorType>
[[noreturn]] void throw_nested_error(ErrorType&& error) {
  using Value = std::decay_t<ErrorType>;
  if constexpr (!std::is_class_v<Value> || std::is_final_v<Value> ||
                std::is_base_of_v<std::nested_exception, Value>) {
    throw std::forward<ErrorType>(error);
  } else {
    throw CapturedNestedError<Value>(std::forward<ErrorType>(error));
  }
}

}  // namespace jyppx::liuding::detail

#endif  // JYPPX_FOUNDATION_NESTED_ERROR_HPP
