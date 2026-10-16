#ifndef JYPPX_FOUNDATION_ERROR_HPP
#define JYPPX_FOUNDATION_ERROR_HPP

/**
 * @file error.hpp
 * @brief Defines stable error codes, structured diagnostics, and backend error
 * translation for the foundation component.
 */

#include <c10/util/Exception.h>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace jyppx {

/**
 * @brief Identifies the machine-readable category of a JYPPX failure.
 *
 * Numeric values are serialized in diagnostics and must remain stable. Adding a
 * category requires a new value; existing values must never be renumbered.
 */
enum class ErrorCode : std::uint16_t {
  invalid_argument = 1,  ///< A caller supplied an invalid argument or option.
  not_implemented = 2,   ///< The requested operation has no implementation.
  device = 3,            ///< A device, driver, or accelerator operation failed.
  serialization = 4,     ///< Structured data could not be encoded or decoded.
  numerical = 5,  ///< A numerical precondition or finite-value check failed.
  io = 6,         ///< A filesystem or stream operation failed.
  resource_exhausted =
      7,          ///< Memory, storage, or another finite resource ran out.
  cancelled = 8,  ///< An operation stopped after a cancellation request.
  compatibility =
      9,          ///< Persisted data or runtime capabilities are incompatible.
  backend = 10,   ///< A backend failed without a more specific classification.
  internal = 11,  ///< A JYPPX invariant was violated.
  not_ready =
      12,  ///< A valid stateful result is unavailable until more input arrives.
};

/**
 * @brief Returns the stable lowercase name for an error code.
 * @param code Error category to render.
 * @return A process-lifetime string view, or `"unknown"` for an unrecognized
 * numeric value.
 */
std::string_view error_code_name(ErrorCode code) noexcept;

/**
 * @brief Carries structured, human-readable context for a JYPPX error.
 *
 * Empty fields are omitted from `Error::what()`. The structure owns all text
 * and is safe to copy between threads; concurrent mutation still requires
 * caller synchronization.
 */
struct ErrorDetails {
  std::string component;  ///< Framework component that detected the failure.
  std::string
      operation;         ///< Operation, field, or state transition that failed.
  std::string expected;  ///< Required value, shape, state, or capability.
  std::string actual;  ///< Observed value, shape, state, or backend diagnostic.
  std::string hint;    ///< Actionable remediation guidance for the caller.
};

/**
 * @brief Base class for stable JYPPX exceptions with structured diagnostics.
 *
 * Instances own their message and details, are immutable after construction,
 * and may be inspected concurrently. Backend adapters should preserve the
 * original exception with `std::throw_with_nested` when translating failures.
 */
class Error : public std::runtime_error {
 public:
  /**
   * @brief Creates an error from a stable code and structured context.
   * @param code Machine-readable error category.
   * @param message Concise description of the failed operation.
   * @param details Owned diagnostic fields appended to `what()` when nonempty.
   */
  Error(ErrorCode code, std::string message, ErrorDetails details = {});

  /**
   * @brief Creates an error with only remediation guidance.
   * @param code Machine-readable error category.
   * @param message Concise description of the failed operation.
   * @param hint Actionable remediation guidance for the caller.
   */
  Error(ErrorCode code, std::string message, std::string hint);

  /**
   * @brief Returns the stable machine-readable error category.
   * @return The code supplied at construction.
   */
  ErrorCode code() const noexcept { return code_; }

  /**
   * @brief Returns the concise message without rendered diagnostic fields.
   * @return A reference valid for the lifetime of this exception.
   */
  const std::string& message() const noexcept { return message_; }

  /**
   * @brief Returns the structured diagnostic context.
   * @return A reference valid for the lifetime of this exception.
   */
  const ErrorDetails& details() const noexcept { return details_; }

 private:
  ErrorCode code_;  ///< Stable machine-readable category for this failure.
  std::string
      message_;  ///< Concise message stored separately from rendered `what()`.
  ErrorDetails
      details_;  ///< Owned structured context for logging and recovery.
};

/** @brief Reports an invalid caller argument or option. */
class InvalidArgument : public Error {
 public:
  /**
   * @brief Creates an invalid-argument error with structured context.
   * @param message Concise description of the rejected input.
   * @param details Owned fields describing the expected and actual input.
   */
  explicit InvalidArgument(std::string message, ErrorDetails details = {})
      : Error(ErrorCode::invalid_argument, std::move(message),
              std::move(details)) {}

  /**
   * @brief Creates an invalid-argument error with remediation guidance.
   * @param message Concise description of the rejected input.
   * @param hint Actionable guidance for producing a valid input.
   */
  InvalidArgument(std::string message, std::string hint)
      : Error(ErrorCode::invalid_argument, std::move(message),
              std::move(hint)) {}
};

/** @brief Reports an operation that the current implementation does not
 * support. */
class NotImplemented : public Error {
 public:
  /**
   * @brief Creates an unsupported-operation error with structured context.
   * @param message Concise description of the unavailable operation.
   * @param details Owned fields describing the unsupported capability.
   */
  explicit NotImplemented(std::string message, ErrorDetails details = {})
      : Error(ErrorCode::not_implemented, std::move(message),
              std::move(details)) {}

  /**
   * @brief Creates an unsupported-operation error with remediation guidance.
   * @param message Concise description of the unavailable operation.
   * @param hint Alternative operation or configuration the caller may use.
   */
  NotImplemented(std::string message, std::string hint)
      : Error(ErrorCode::not_implemented, std::move(message), std::move(hint)) {
  }
};

/** @brief Reports a device, driver, or accelerator runtime failure. */
class DeviceError : public Error {
 public:
  /**
   * @brief Creates a device error with structured context.
   * @param message Concise description of the device failure.
   * @param details Owned fields describing the device operation and runtime.
   */
  explicit DeviceError(std::string message, ErrorDetails details = {})
      : Error(ErrorCode::device, std::move(message), std::move(details)) {}

  /**
   * @brief Creates a device error with remediation guidance.
   * @param message Concise description of the device failure.
   * @param hint Runtime, topology, or configuration guidance.
   */
  DeviceError(std::string message, std::string hint)
      : Error(ErrorCode::device, std::move(message), std::move(hint)) {}
};

/** @brief Reports a serialization or deserialization failure. */
class SerializationError : public Error {
 public:
  /**
   * @brief Creates a serialization error with structured context.
   * @param message Concise description of the encoding or decoding failure.
   * @param details Owned fields describing the expected and observed format.
   */
  explicit SerializationError(std::string message, ErrorDetails details = {})
      : Error(ErrorCode::serialization, std::move(message),
              std::move(details)) {}

  /**
   * @brief Creates a serialization error with remediation guidance.
   * @param message Concise description of the encoding or decoding failure.
   * @param hint Migration, repair, or format-selection guidance.
   */
  SerializationError(std::string message, std::string hint)
      : Error(ErrorCode::serialization, std::move(message), std::move(hint)) {}
};

/** @brief Reports a numerical validity or stability failure. */
class NumericalError : public Error {
 public:
  /**
   * @brief Creates a numerical error with structured context.
   * @param message Concise description of the numerical failure.
   * @param details Owned fields describing the violated numerical contract.
   */
  explicit NumericalError(std::string message, ErrorDetails details = {})
      : Error(ErrorCode::numerical, std::move(message), std::move(details)) {}

  /**
   * @brief Creates a numerical error with remediation guidance.
   * @param message Concise description of the numerical failure.
   * @param hint Stabilization or input-validation guidance.
   */
  NumericalError(std::string message, std::string hint)
      : Error(ErrorCode::numerical, std::move(message), std::move(hint)) {}
};

/** @brief Reports a filesystem, path, or stream failure. */
class IoError : public Error {
 public:
  /**
   * @brief Creates an I/O error with structured context.
   * @param message Concise description of the failed I/O operation.
   * @param details Owned fields describing the path, stream, or expected state.
   */
  explicit IoError(std::string message, ErrorDetails details = {})
      : Error(ErrorCode::io, std::move(message), std::move(details)) {}

  /**
   * @brief Creates an I/O error with remediation guidance.
   * @param message Concise description of the failed I/O operation.
   * @param hint Path, permission, capacity, or retry guidance.
   */
  IoError(std::string message, std::string hint)
      : Error(ErrorCode::io, std::move(message), std::move(hint)) {}
};

/** @brief Reports exhaustion of memory, storage, or another finite resource. */
class ResourceExhausted : public Error {
 public:
  /**
   * @brief Creates a resource-exhaustion error with structured context.
   * @param message Concise description of the unavailable resource.
   * @param details Owned fields describing the resource limit and demand.
   */
  explicit ResourceExhausted(std::string message, ErrorDetails details = {})
      : Error(ErrorCode::resource_exhausted, std::move(message),
              std::move(details)) {}

  /**
   * @brief Creates a resource-exhaustion error with remediation guidance.
   * @param message Concise description of the unavailable resource.
   * @param hint Capacity, batching, cleanup, or retry guidance.
   */
  ResourceExhausted(std::string message, std::string hint)
      : Error(ErrorCode::resource_exhausted, std::move(message),
              std::move(hint)) {}
};

/** @brief Reports that an operation honored a cancellation request. */
class Cancelled : public Error {
 public:
  /**
   * @brief Creates a cancellation error with structured context.
   * @param message Concise description of the cancelled operation.
   * @param details Owned fields describing the cancellation boundary.
   */
  explicit Cancelled(std::string message, ErrorDetails details = {})
      : Error(ErrorCode::cancelled, std::move(message), std::move(details)) {}

  /**
   * @brief Creates a cancellation error with remediation guidance.
   * @param message Concise description of the cancelled operation.
   * @param hint Resume, retry, or cleanup guidance.
   */
  Cancelled(std::string message, std::string hint)
      : Error(ErrorCode::cancelled, std::move(message), std::move(hint)) {}
};

/** @brief Reports incompatible persisted data, ABI, or runtime capabilities. */
class CompatibilityError : public Error {
 public:
  /**
   * @brief Creates a compatibility error with structured context.
   * @param message Concise description of the incompatibility.
   * @param details Owned fields describing required and observed versions.
   */
  explicit CompatibilityError(std::string message, ErrorDetails details = {})
      : Error(ErrorCode::compatibility, std::move(message),
              std::move(details)) {}

  /**
   * @brief Creates a compatibility error with remediation guidance.
   * @param message Concise description of the incompatibility.
   * @param hint Migration or environment-alignment guidance.
   */
  CompatibilityError(std::string message, std::string hint)
      : Error(ErrorCode::compatibility, std::move(message), std::move(hint)) {}
};

/** @brief Reports an unclassified failure raised by a dependency backend. */
class BackendError : public Error {
 public:
  /**
   * @brief Creates a backend error with structured context.
   * @param message Concise description of the backend failure.
   * @param details Owned fields containing the backend operation and
   * diagnostic.
   */
  explicit BackendError(std::string message, ErrorDetails details = {})
      : Error(ErrorCode::backend, std::move(message), std::move(details)) {}

  /**
   * @brief Creates a backend error with remediation guidance.
   * @param message Concise description of the backend failure.
   * @param hint Backend selection, version, or configuration guidance.
   */
  BackendError(std::string message, std::string hint)
      : Error(ErrorCode::backend, std::move(message), std::move(hint)) {}
};

/** @brief Reports violation of a framework invariant. */
class InternalError : public Error {
 public:
  /**
   * @brief Creates an internal error with structured context.
   * @param message Concise description of the violated invariant.
   * @param details Owned fields describing the expected internal state.
   */
  explicit InternalError(std::string message, ErrorDetails details = {})
      : Error(ErrorCode::internal, std::move(message), std::move(details)) {}

  /**
   * @brief Creates an internal error with remediation guidance.
   * @param message Concise description of the violated invariant.
   * @param hint Diagnostic collection or bug-report guidance.
   */
  InternalError(std::string message, std::string hint)
      : Error(ErrorCode::internal, std::move(message), std::move(hint)) {}
};

/** @brief Reports that a stateful result is not ready to be observed. */
class NotReady : public Error {
 public:
  /**
   * @brief Creates a not-ready error with structured context.
   * @param message Concise description of the unavailable result.
   * @param details Owned fields describing the required state transition.
   */
  explicit NotReady(std::string message, ErrorDetails details = {})
      : Error(ErrorCode::not_ready, std::move(message), std::move(details)) {}

  /**
   * @brief Creates a not-ready error with remediation guidance.
   * @param message Concise description of the unavailable result.
   * @param hint Actionable guidance for making the result available.
   */
  NotReady(std::string message, std::string hint)
      : Error(ErrorCode::not_ready, std::move(message), std::move(hint)) {}
};

namespace detail {

/**
 * @brief Selects an explicit JYPPX category for a backend exception.
 *
 * Translation must be based on the operation contract and typed backend state,
 * never on locale-dependent text returned by `c10::Error::what()`.
 */
enum class BackendFailureKind {
  invalid_argument,  ///< The backend rejected a caller-controlled argument.
  unsupported,  ///< The backend does not implement the requested operation.
  device,       ///< The active device or accelerator runtime failed.
  resource_exhausted,  ///< The backend exhausted a finite runtime resource.
  unknown,  ///< No narrower category can be justified by typed state.
};

/**
 * @brief Describes the operation-specific contract used to translate a backend
 * exception.
 */
struct BackendErrorContext {
  std::string component;  ///< Framework component invoking the backend.
  std::string
      operation;  ///< Backend operation or framework transition that failed.
  BackendFailureKind failure_kind =
      BackendFailureKind::unknown;  ///< Explicit target category for
                                    ///< translation.
  std::string
      expected;      ///< Backend state or capability required by the operation.
  std::string hint;  ///< Actionable remediation guidance for the caller.
};

/**
 * @brief Translates and rethrows a LibTorch exception as a JYPPX exception.
 * @param cause Backend exception retained as the nested cause.
 * @param context Owned operation context and explicit failure category.
 * @throws InvalidArgument When `failure_kind` is `invalid_argument`.
 * @throws NotImplemented When `failure_kind` is `unsupported`.
 * @throws DeviceError When `failure_kind` is `device`.
 * @throws ResourceExhausted When `failure_kind` is `resource_exhausted`.
 * @throws BackendError When `failure_kind` is `unknown`.
 */
[[noreturn]] void rethrow_backend_error(const c10::Error& cause,
                                        BackendErrorContext context);

}  // namespace detail

/**
 * @brief Validates a public API argument and throws structured diagnostics.
 *
 * `actual_value` is evaluated exactly once and rendered through `operator<<`.
 * Other macro arguments are evaluated only when the condition is false.
 */
#define JYPPX_CHECK_ARG(condition, component_value, field_value,  \
                        expected_value, actual_value, hint_value) \
  do {                                                            \
    if (!(condition)) {                                           \
      std::ostringstream _jyppx_actual;                           \
      _jyppx_actual << (actual_value);                            \
      ::jyppx::ErrorDetails _jyppx_details;                       \
      _jyppx_details.component = (component_value);               \
      _jyppx_details.operation = (field_value);                   \
      _jyppx_details.expected = (expected_value);                 \
      _jyppx_details.actual = _jyppx_actual.str();                \
      _jyppx_details.hint = (hint_value);                         \
      throw ::jyppx::InvalidArgument(                             \
          std::string(field_value) + " check failed",             \
          std::move(_jyppx_details));                             \
    }                                                             \
  } while (false)

/**
 * @brief Validates an internal invariant and throws `InternalError` on failure.
 *
 * `message_value` and `hint_value` are evaluated only when `condition` is
 * false.
 */
#define JYPPX_CHECK(condition, message_value, hint_value)          \
  do {                                                             \
    if (!(condition)) {                                            \
      throw ::jyppx::InternalError((message_value), (hint_value)); \
    }                                                              \
  } while (false)

/**
 * @brief Unconditionally reports an unsupported operation.
 *
 * The macro throws `NotImplemented`; callers must not place cleanup that must
 * execute after the expansion.
 */
#define JYPPX_NOT_IMPLEMENTED(what_value, hint_value) \
  throw ::jyppx::NotImplemented(                      \
      std::string("Not implemented: ") + (what_value), (hint_value))

}  // namespace jyppx

#endif  // JYPPX_FOUNDATION_ERROR_HPP
