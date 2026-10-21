#include "jyppx/foundation/version.hpp"

#include "jyppx/foundation/build_metadata.hpp"
#include "jyppx/foundation/error.hpp"

#include <ATen/Context.h>
#include <sstream>
#include <string_view>

namespace jyppx {
namespace {

std::string json_escape(std::string_view value) {
  std::string output;
  output.reserve(value.size());
  for (const char ch : value) {
    switch (ch) {
      case '\\':
        output += "\\\\";
        break;
      case '"':
        output += "\\\"";
        break;
      case '\n':
        output += "\\n";
        break;
      case '\r':
        output += "\\r";
        break;
      case '\t':
        output += "\\t";
        break;
      default:
        output += ch;
        break;
    }
  }
  return output;
}

long language_standard() noexcept {
#if defined(_MSVC_LANG)
  return _MSVC_LANG;
#else
  return __cplusplus;
#endif
}

}  // namespace

BuildInfo build_info() noexcept {
  return {JYPPX_VERSION, JYPPX_BUILD_LIBTORCH_VERSION, JYPPX_BUILD_COMPILER,
          JYPPX_BUILD_TYPE, language_standard()};
}

BuildFingerprint current_build_fingerprint() {
  try {
    std::string accelerator = JYPPX_BUILD_ACCELERATOR;
    accelerator += at::globalContext().hasCUDA()
                       ? ":available:" + std::to_string(at::getNumGPUs())
                       : ":unavailable";
    return {
        .jyppx_version = JYPPX_VERSION,
        .git_commit = JYPPX_BUILD_GIT_COMMIT,
        .compiler = JYPPX_BUILD_COMPILER,
        .cxx_standard = std::to_string(language_standard()),
        .os_arch = JYPPX_BUILD_OS_ARCH,
        .build_type = JYPPX_BUILD_TYPE,
        .libtorch_version = JYPPX_BUILD_LIBTORCH_VERSION,
        .libtorch_build = JYPPX_BUILD_LIBTORCH_BUILD,
        .accelerator_runtime = std::move(accelerator),
        .dependency_lock_digest = JYPPX_BUILD_DEPENDENCY_LOCK_DIGEST,
    };
  } catch (const Error&) {
    throw;
  } catch (const c10::Error& cause) {
    detail::rethrow_backend_error(
        cause,
        {.component = "BuildFingerprint",
         .operation = "inspect_accelerator_runtime",
         .failure_kind = detail::BackendFailureKind::device,
         .expected = "queryable LibTorch accelerator runtime",
         .hint =
             "Check the LibTorch, accelerator driver, and runtime versions."});
  }
}

std::string build_fingerprint_json(const BuildFingerprint& fingerprint) {
  std::ostringstream output;
  output << '{' << "\"jyppx_version\":\""
         << json_escape(fingerprint.jyppx_version) << "\","
         << "\"git_commit\":\"" << json_escape(fingerprint.git_commit) << "\","
         << "\"compiler\":\"" << json_escape(fingerprint.compiler) << "\","
         << "\"cxx_standard\":\"" << json_escape(fingerprint.cxx_standard)
         << "\","
         << "\"os_arch\":\"" << json_escape(fingerprint.os_arch) << "\","
         << "\"build_type\":\"" << json_escape(fingerprint.build_type) << "\","
         << "\"libtorch_version\":\""
         << json_escape(fingerprint.libtorch_version) << "\","
         << "\"libtorch_build\":\"" << json_escape(fingerprint.libtorch_build)
         << "\","
         << "\"accelerator_runtime\":\""
         << json_escape(fingerprint.accelerator_runtime) << "\","
         << "\"dependency_lock_digest\":\""
         << json_escape(fingerprint.dependency_lock_digest) << "\"}";
  return output.str();
}

std::string build_fingerprint_json() {
  return build_fingerprint_json(current_build_fingerprint());
}

std::string build_fingerprint() {
  const auto info = current_build_fingerprint();
  return "jyppx=" + info.jyppx_version + ";libtorch=" + info.libtorch_version +
         ";compiler=" + info.compiler + ";build=" + info.build_type +
         ";cxx=" + info.cxx_standard + ";git=" + info.git_commit +
         ";platform=" + info.os_arch +
         ";accelerator=" + info.accelerator_runtime;
}

}  // namespace jyppx
