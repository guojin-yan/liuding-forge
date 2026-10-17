#include "jyppx/foundation/error.hpp"

#include <exception>

namespace jyppx {
namespace {

std::string render_error(ErrorCode code, const std::string& message,
                         const ErrorDetails& details) {
  std::ostringstream output;
  output << '[' << error_code_name(code) << "] " << message;
  if (!details.component.empty())
    output << "\n  Component: " << details.component;
  if (!details.operation.empty())
    output << "\n  Operation: " << details.operation;
  if (!details.expected.empty())
    output << "\n  Expected:  " << details.expected;
  if (!details.actual.empty()) output << "\n  Actual:    " << details.actual;
  if (!details.hint.empty()) output << "\n  Hint:      " << details.hint;
  return output.str();
}

}  // namespace

std::string_view error_code_name(ErrorCode code) noexcept {
  switch (code) {
    case ErrorCode::invalid_argument:
      return "invalid_argument";
    case ErrorCode::not_implemented:
      return "not_implemented";
    case ErrorCode::device:
      return "device";
    case ErrorCode::serialization:
      return "serialization";
    case ErrorCode::numerical:
      return "numerical";
    case ErrorCode::io:
      return "io";
    case ErrorCode::resource_exhausted:
      return "resource_exhausted";
    case ErrorCode::cancelled:
      return "cancelled";
    case ErrorCode::compatibility:
      return "compatibility";
    case ErrorCode::backend:
      return "backend";
    case ErrorCode::internal:
      return "internal";
    case ErrorCode::not_ready:
      return "not_ready";
    case ErrorCode::deadline_exceeded:
      return "deadline_exceeded";
  }
  return "unknown";
}

Error::Error(ErrorCode code, std::string message, ErrorDetails details)
    : std::runtime_error(render_error(code, message, details)),
      code_(code),
      message_(std::move(message)),
      details_(std::move(details)) {}

Error::Error(ErrorCode code, std::string message, std::string hint)
    : Error(code, std::move(message),
            ErrorDetails{.component = {},
                         .operation = {},
                         .expected = {},
                         .actual = {},
                         .hint = std::move(hint)}) {}

namespace detail {

[[noreturn]] void rethrow_backend_error(const c10::Error& cause,
                                        BackendErrorContext context) {
  ErrorDetails details{
      .component = std::move(context.component),
      .operation = std::move(context.operation),
      .expected = std::move(context.expected),
      .actual = cause.what(),
      .hint = std::move(context.hint),
  };

  switch (context.failure_kind) {
    case BackendFailureKind::invalid_argument:
      std::throw_with_nested(
          InvalidArgument("Backend rejected an argument", std::move(details)));
    case BackendFailureKind::unsupported:
      std::throw_with_nested(NotImplemented("Backend operation is unsupported",
                                            std::move(details)));
    case BackendFailureKind::device:
      std::throw_with_nested(
          DeviceError("Backend device operation failed", std::move(details)));
    case BackendFailureKind::resource_exhausted:
      std::throw_with_nested(ResourceExhausted("Backend resource was exhausted",
                                               std::move(details)));
    case BackendFailureKind::unknown:
      std::throw_with_nested(
          BackendError("Backend operation failed", std::move(details)));
  }
  std::terminate();
}

}  // namespace detail
}  // namespace jyppx
