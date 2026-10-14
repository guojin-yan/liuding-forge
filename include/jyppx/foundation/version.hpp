#ifndef JYPPX_FOUNDATION_VERSION_HPP
#define JYPPX_FOUNDATION_VERSION_HPP

/**
 * @file version.hpp
 * @brief Exposes the framework version and reproducible build diagnostics.
 */

#include <jyppx/foundation/build_config.hpp>

#include <string>

#if defined(JYPPX_CMAKE_VERSION_MAJOR)
static_assert(JYPPX_VERSION_MAJOR == JYPPX_CMAKE_VERSION_MAJOR &&
                  JYPPX_VERSION_MINOR == JYPPX_CMAKE_VERSION_MINOR &&
                  JYPPX_VERSION_PATCH == JYPPX_CMAKE_VERSION_PATCH,
              "JYPPX generated version must match the CMake project version");
#endif

namespace jyppx {

/**
 * @brief Returns the complete semantic framework version.
 * @return A null-terminated process-lifetime string in `major.minor.patch`
 * format.
 */
inline const char* version() noexcept { return JYPPX_VERSION; }

/**
 * @brief Returns the semantic-version major component.
 * @return A non-negative major version integer.
 */
inline int version_major() noexcept { return JYPPX_VERSION_MAJOR; }

/**
 * @brief Returns the semantic-version minor component.
 * @return A non-negative minor version integer.
 */
inline int version_minor() noexcept { return JYPPX_VERSION_MINOR; }

/**
 * @brief Returns the semantic-version patch component.
 * @return A non-negative patch version integer.
 */
inline int version_patch() noexcept { return JYPPX_VERSION_PATCH; }

/**
 * @brief Provides a compact, allocation-free view of core build metadata.
 *
 * All character pointers refer to process-lifetime generated string literals.
 * The structure is immutable by convention and safe to copy between threads.
 */
struct BuildInfo {
  const char* framework_version;  ///< JYPPX semantic version string.
  const char*
      libtorch_version;    ///< LibTorch version selected at configure time.
  const char* compiler;    ///< Compiler family and version string.
  const char* build_type;  ///< CMake configuration such as Release or Debug.
  long cxx_standard;       ///< Numeric `__cplusplus` or `_MSVC_LANG` value.
};

/**
 * @brief Owns the complete build and runtime identity used in diagnostics.
 *
 * The structure contains no secrets or host paths. It is copyable and safe to
 * inspect concurrently after construction.
 */
struct BuildFingerprint {
  std::string jyppx_version;  ///< JYPPX semantic version string.
  std::string git_commit;  ///< Source commit, with `+dirty` for local changes.
  std::string compiler;  ///< Compiler family and version used for this binary.
  std::string cxx_standard;  ///< Numeric C++ language standard selected by the
                             ///< compiler.
  std::string os_arch;  ///< Configure-time operating system and architecture.
  std::string build_type;  ///< CMake configuration used to compile the library.
  std::string
      libtorch_version;  ///< LibTorch package version linked by the library.
  std::string libtorch_build;       ///< LibTorch ABI/build variant recorded at
                                    ///< configure time.
  std::string accelerator_runtime;  ///< Built accelerator and current runtime
                                    ///< availability.
  std::string dependency_lock_digest;  ///< Dependency-lock digest or `unlocked`
                                       ///< sentinel.
};

/**
 * @brief Returns allocation-free core build metadata.
 * @return A value whose character pointers remain valid for the process
 * lifetime.
 */
BuildInfo build_info() noexcept;

/**
 * @brief Captures configure-time identity and current accelerator availability.
 * @return An owned fingerprint suitable for logs and checkpoint manifests.
 * @throws DeviceError If LibTorch cannot inspect accelerator availability. The
 * original c10::Error is retained as the nested cause.
 */
BuildFingerprint current_build_fingerprint();

/**
 * @brief Serializes a build fingerprint as one compact JSON object.
 * @param fingerprint Metadata to serialize; all strings are JSON-escaped.
 * @return UTF-8 JSON without trailing whitespace or a newline.
 */
std::string build_fingerprint_json(const BuildFingerprint& fingerprint);

/**
 * @brief Captures and serializes the current build fingerprint.
 * @return UTF-8 JSON without trailing whitespace or a newline.
 * @throws DeviceError If LibTorch cannot inspect accelerator availability. The
 * original c10::Error is retained as the nested cause.
 */
std::string build_fingerprint_json();

/**
 * @brief Returns a compact diagnostic representation of the current build.
 * @return Semicolon-delimited `key=value` fields intended for human-readable
 * logs.
 * @throws DeviceError If LibTorch cannot inspect accelerator availability. The
 * original c10::Error is retained as the nested cause.
 */
std::string build_fingerprint();

}  // namespace jyppx

#endif  // JYPPX_FOUNDATION_VERSION_HPP
