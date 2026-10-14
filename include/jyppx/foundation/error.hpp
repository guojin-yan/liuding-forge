#ifndef JYPPX_FOUNDATION_ERROR_HPP
#define JYPPX_FOUNDATION_ERROR_HPP

#include <c10/util/Exception.h>

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace jyppx {

enum class ErrorCode : std::uint16_t {
    invalid_argument = 1,
    not_implemented = 2,
    device = 3,
    serialization = 4,
    numerical = 5,
    io = 6,
    resource_exhausted = 7,
    cancelled = 8,
    compatibility = 9,
    backend = 10,
    internal = 11,
};

std::string_view error_code_name(ErrorCode code) noexcept;

struct ErrorDetails {
    std::string component;
    std::string operation;
    std::string expected;
    std::string actual;
    std::string hint;
};

class Error : public std::runtime_error {
public:
    Error(ErrorCode code, std::string message, ErrorDetails details = {});
    Error(ErrorCode code, std::string message, std::string hint);

    ErrorCode code() const noexcept { return code_; }
    const std::string& message() const noexcept { return message_; }
    const ErrorDetails& details() const noexcept { return details_; }

private:
    ErrorCode code_;
    std::string message_;
    ErrorDetails details_;
};

#define JYPPX_DEFINE_ERROR_TYPE(name, code_value)                         \
    class name : public Error {                                           \
    public:                                                               \
        explicit name(std::string message, ErrorDetails details = {})     \
            : Error(ErrorCode::code_value, std::move(message),            \
                    std::move(details)) {}                                \
        name(std::string message, std::string hint)                       \
            : Error(ErrorCode::code_value, std::move(message),            \
                    std::move(hint)) {}                                   \
    }

JYPPX_DEFINE_ERROR_TYPE(InvalidArgument, invalid_argument);
JYPPX_DEFINE_ERROR_TYPE(NotImplemented, not_implemented);
JYPPX_DEFINE_ERROR_TYPE(DeviceError, device);
JYPPX_DEFINE_ERROR_TYPE(SerializationError, serialization);
JYPPX_DEFINE_ERROR_TYPE(NumericalError, numerical);
JYPPX_DEFINE_ERROR_TYPE(IoError, io);
JYPPX_DEFINE_ERROR_TYPE(ResourceExhausted, resource_exhausted);
JYPPX_DEFINE_ERROR_TYPE(Cancelled, cancelled);
JYPPX_DEFINE_ERROR_TYPE(CompatibilityError, compatibility);
JYPPX_DEFINE_ERROR_TYPE(BackendError, backend);
JYPPX_DEFINE_ERROR_TYPE(InternalError, internal);

#undef JYPPX_DEFINE_ERROR_TYPE

namespace detail {

enum class BackendFailureKind {
    invalid_argument,
    unsupported,
    device,
    resource_exhausted,
    unknown,
};

struct BackendErrorContext {
    std::string component;
    std::string operation;
    BackendFailureKind failure_kind = BackendFailureKind::unknown;
    std::string expected;
    std::string hint;
};

[[noreturn]] void rethrow_backend_error(
    const c10::Error& cause,
    BackendErrorContext context);

} // namespace detail

#define JYPPX_CHECK_ARG(condition, component_value, field_value,            \
                        expected_value, actual_value, hint_value)           \
    do {                                                                    \
        if (!(condition)) {                                                 \
            std::ostringstream _jyppx_actual;                               \
            _jyppx_actual << (actual_value);                                \
            ::jyppx::ErrorDetails _jyppx_details;                           \
            _jyppx_details.component = (component_value);                   \
            _jyppx_details.operation = (field_value);                       \
            _jyppx_details.expected = (expected_value);                     \
            _jyppx_details.actual = _jyppx_actual.str();                    \
            _jyppx_details.hint = (hint_value);                             \
            throw ::jyppx::InvalidArgument(                                \
                std::string(field_value) + " check failed",               \
                std::move(_jyppx_details));                                 \
        }                                                                   \
    } while (false)

#define JYPPX_CHECK(condition, message_value, hint_value)                   \
    do {                                                                    \
        if (!(condition)) {                                                 \
            throw ::jyppx::InternalError((message_value), (hint_value));    \
        }                                                                   \
    } while (false)

#define JYPPX_NOT_IMPLEMENTED(what_value, hint_value)                       \
    throw ::jyppx::NotImplemented(                                         \
        std::string("Not implemented: ") + (what_value), (hint_value))

} // namespace jyppx

#endif // JYPPX_FOUNDATION_ERROR_HPP
