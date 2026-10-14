#ifndef JYPPX_FOUNDATION_VERSION_HPP
#define JYPPX_FOUNDATION_VERSION_HPP

#include <jyppx/foundation/build_config.hpp>

#include <string>

#if defined(JYPPX_CMAKE_VERSION_MAJOR)
static_assert(JYPPX_VERSION_MAJOR == JYPPX_CMAKE_VERSION_MAJOR &&
              JYPPX_VERSION_MINOR == JYPPX_CMAKE_VERSION_MINOR &&
              JYPPX_VERSION_PATCH == JYPPX_CMAKE_VERSION_PATCH,
              "JYPPX generated version must match the CMake project version");
#endif

namespace jyppx {

inline const char* version() noexcept { return JYPPX_VERSION; }
inline int version_major() noexcept { return JYPPX_VERSION_MAJOR; }
inline int version_minor() noexcept { return JYPPX_VERSION_MINOR; }
inline int version_patch() noexcept { return JYPPX_VERSION_PATCH; }

struct BuildInfo {
    const char* framework_version;
    const char* libtorch_version;
    const char* compiler;
    const char* build_type;
    long cxx_standard;
};

struct BuildFingerprint {
    std::string jyppx_version;
    std::string git_commit;
    std::string compiler;
    std::string cxx_standard;
    std::string os_arch;
    std::string build_type;
    std::string libtorch_version;
    std::string libtorch_build;
    std::string accelerator_runtime;
    std::string dependency_lock_digest;
};

BuildInfo build_info() noexcept;
BuildFingerprint current_build_fingerprint();
std::string build_fingerprint_json(const BuildFingerprint& fingerprint);
std::string build_fingerprint_json();

/** Compact diagnostic representation retained as the primary string API. */
std::string build_fingerprint();

} // namespace jyppx

#endif // JYPPX_FOUNDATION_VERSION_HPP
