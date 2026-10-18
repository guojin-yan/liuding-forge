#ifndef JYPPX_DATA_DETAIL_ERROR_HPP
#define JYPPX_DATA_DETAIL_ERROR_HPP

/**
 * @file error.hpp
 * @brief Provides structured error constructors shared by data components.
 */

#include <jyppx/foundation/error.hpp>

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>

namespace jyppx::data::detail {

/**
 * @brief Throws a structured invalid-argument error for a data API.
 * @param component Stable data component name.
 * @param operation Public operation or field that rejected the input.
 * @param expected Required value, shape, range, or state.
 * @param actual Observed invalid value or condition.
 * @param hint Actionable remediation guidance for the caller.
 * @throws jyppx::InvalidArgument Always.
 */
[[noreturn]] inline void throw_invalid(std::string_view component,
                                       std::string_view operation,
                                       std::string expected, std::string actual,
                                       std::string hint) {
  throw ::jyppx::InvalidArgument(
      "Invalid data API argument",
      ::jyppx::ErrorDetails{.component = std::string(component),
                            .operation = std::string(operation),
                            .expected = std::move(expected),
                            .actual = std::move(actual),
                            .hint = std::move(hint)});
}

/**
 * @brief Throws a structured data index error.
 * @param component Stable data component name.
 * @param operation Lookup operation that received the invalid index.
 * @param index Invalid zero-based index rendered for diagnostics.
 * @param size Number of addressable entries.
 * @throws jyppx::InvalidArgument Always.
 */
[[noreturn]] inline void throw_index_error(std::string_view component,
                                           std::string_view operation,
                                           std::size_t index,
                                           std::size_t size) {
  throw_invalid(component, operation, "index < size", std::to_string(index),
                "The valid index range is [0, " + std::to_string(size) + ")");
}

/**
 * @brief Throws a structured malformed-state error for a sampler.
 * @param component Stable sampler component name.
 * @param operation State operation being decoded.
 * @param actual Description of the malformed state.
 * @throws jyppx::SerializationError Always.
 */
[[noreturn]] inline void throw_state_error(std::string_view component,
                                           std::string_view operation,
                                           std::string actual) {
  throw ::jyppx::SerializationError(
      "Invalid data sampler state",
      ::jyppx::ErrorDetails{.component = std::string(component),
                            .operation = std::string(operation),
                            .expected = "state produced by the same sampler",
                            .actual = std::move(actual),
                            .hint = "Restore state from the matching sampler "
                                    "configuration"});
}

/**
 * @brief Throws a structured compatibility error for restored data state.
 * @param component Stable data component name.
 * @param operation Compatibility check that rejected the state.
 * @param expected Required schema or runtime topology.
 * @param actual Observed incompatible schema or topology.
 * @param hint Actionable migration or configuration guidance.
 * @throws jyppx::CompatibilityError Always.
 */
[[noreturn]] inline void throw_compatibility(std::string_view component,
                                             std::string_view operation,
                                             std::string expected,
                                             std::string actual,
                                             std::string hint) {
  throw ::jyppx::CompatibilityError(
      "Incompatible data state",
      ::jyppx::ErrorDetails{.component = std::string(component),
                            .operation = std::string(operation),
                            .expected = std::move(expected),
                            .actual = std::move(actual),
                            .hint = std::move(hint)});
}

/**
 * @brief Throws a structured serialization error for malformed data state.
 * @param component Stable data component name.
 * @param operation State field or decoding operation that failed.
 * @param expected Required serialized representation.
 * @param actual Observed malformed representation.
 * @param hint Actionable repair or checkpoint guidance.
 * @throws jyppx::SerializationError Always.
 */
[[noreturn]] inline void throw_serialization(std::string_view component,
                                             std::string_view operation,
                                             std::string expected,
                                             std::string actual,
                                             std::string hint) {
  throw ::jyppx::SerializationError(
      "Invalid serialized data state",
      ::jyppx::ErrorDetails{.component = std::string(component),
                            .operation = std::string(operation),
                            .expected = std::move(expected),
                            .actual = std::move(actual),
                            .hint = std::move(hint)});
}

}  // namespace jyppx::data::detail

#endif  // JYPPX_DATA_DETAIL_ERROR_HPP
