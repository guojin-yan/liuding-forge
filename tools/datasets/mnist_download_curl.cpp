// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "mnist_download_detail.hpp"

#include <jyppx/liuding/foundation/error.hpp>

#include <curl/curl.h>
#include <exception>
#include <limits>
#include <memory>
#include <string>

namespace jyppx::liuding::datasets::detail {
namespace {
class CurlRuntime final {
 public:
  CurlRuntime() {
    if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK)
      throw IoError("Could not initialize system libcurl");
  }
  ~CurlRuntime() { curl_global_cleanup(); }
  CurlRuntime(const CurlRuntime&) = delete;
  CurlRuntime& operator=(const CurlRuntime&) = delete;
};
void check(CURLcode result) {
  if (result != CURLE_OK)
    throw IoError(
        std::string("MNIST HTTPS operation failed in system libcurl: ") +
        curl_easy_strerror(result));
}
struct Receiver {
  const MnistHttpsCallbacks& callbacks;
  std::size_t header_bytes = 0U;
  std::exception_ptr error;
};
std::size_t receive_header(char*, std::size_t size, std::size_t count,
                           void* context) noexcept {
  auto& state = *static_cast<Receiver*>(context);
  try {
    if (size != 0U && count > std::numeric_limits<std::size_t>::max() / size)
      throw ResourceExhausted("MNIST HTTPS header length overflow");
    const auto bytes = size * count;
    if (bytes > 65536U - state.header_bytes)
      throw ResourceExhausted("MNIST HTTPS headers exceed 64 KiB");
    state.header_bytes += bytes;
    return bytes;
  } catch (...) {
    state.error = std::current_exception();
    return 0U;
  }
}
std::size_t receive(char* data, std::size_t size, std::size_t count,
                    void* context) noexcept {
  auto& state = *static_cast<Receiver*>(context);
  try {
    if (size != 0U && count > std::numeric_limits<std::size_t>::max() / size)
      throw ResourceExhausted("MNIST HTTPS callback length overflow");
    const auto bytes = size * count;
    // Body bytes are bounded and hashed in a private stage. Final response
    // metadata is delivered only after curl_easy_perform has completed.
    state.callbacks.bytes({reinterpret_cast<const std::byte*>(data), bytes});
    return bytes;
  } catch (...) {
    state.error = std::current_exception();
    return 0U;
  }
}
}  // namespace

void transfer_mnist_https(std::string_view url, std::uint64_t expected_bytes,
                          const MnistHttpsCallbacks& callbacks) {
  require_mnist_https_url(url);
  if (expected_bytes == 0U || expected_bytes > 9912422U)
    throw InvalidArgument("Invalid MNIST HTTPS size bound");
  static const CurlRuntime runtime;
  static_cast<void>(runtime);
  const auto* version = curl_version_info(CURLVERSION_NOW);
  if (!version || version->version_num < 0x075100U ||
      (version->features & CURL_VERSION_SSL) == 0 ||
      (version->features & CURL_VERSION_ASYNCHDNS) == 0)
    throw NotImplemented(
        "System libcurl >= 7.81 with TLS and asynchronous DNS is required");
  const std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> handle(
      curl_easy_init(), &curl_easy_cleanup);
  if (!handle) throw IoError("Could not create MNIST libcurl request");
  const std::string owned_url(url);
  Receiver receiver{callbacks, 0U, {}};
  check(curl_easy_setopt(handle.get(), CURLOPT_URL, owned_url.c_str()));
#if LIBCURL_VERSION_NUM >= 0x075500
  if (version->version_num < 0x075500U)
    throw NotImplemented("System libcurl runtime must match selected headers");
  check(curl_easy_setopt(handle.get(), CURLOPT_PROTOCOLS_STR, "https"));
  check(curl_easy_setopt(handle.get(), CURLOPT_REDIR_PROTOCOLS_STR, "https"));
#else
  check(curl_easy_setopt(handle.get(), CURLOPT_PROTOCOLS,
                         static_cast<long>(CURLPROTO_HTTPS)));
  check(curl_easy_setopt(handle.get(), CURLOPT_REDIR_PROTOCOLS,
                         static_cast<long>(CURLPROTO_HTTPS)));
#endif
  check(curl_easy_setopt(handle.get(), CURLOPT_FOLLOWLOCATION, 0L));
  check(curl_easy_setopt(handle.get(), CURLOPT_MAXREDIRS, 0L));
  check(curl_easy_setopt(handle.get(), CURLOPT_SSL_VERIFYPEER, 1L));
  check(curl_easy_setopt(handle.get(), CURLOPT_SSL_VERIFYHOST, 2L));
  check(curl_easy_setopt(handle.get(), CURLOPT_SSLVERSION,
                         static_cast<long>(CURL_SSLVERSION_TLSv1_2)));
  check(curl_easy_setopt(handle.get(), CURLOPT_NETRC,
                         static_cast<long>(CURL_NETRC_IGNORED)));
  check(curl_easy_setopt(handle.get(), CURLOPT_NOSIGNAL, 1L));
  check(curl_easy_setopt(handle.get(), CURLOPT_CONNECTTIMEOUT, 30L));
  check(curl_easy_setopt(handle.get(), CURLOPT_TIMEOUT, 120L));
  check(curl_easy_setopt(handle.get(), CURLOPT_LOW_SPEED_LIMIT, 1024L));
  check(curl_easy_setopt(handle.get(), CURLOPT_LOW_SPEED_TIME, 30L));
  check(curl_easy_setopt(handle.get(), CURLOPT_HTTP_VERSION,
                         static_cast<long>(CURL_HTTP_VERSION_1_1)));
  check(curl_easy_setopt(handle.get(), CURLOPT_ACCEPT_ENCODING, "identity"));
  check(curl_easy_setopt(handle.get(), CURLOPT_HTTP_CONTENT_DECODING, 0L));
  check(curl_easy_setopt(handle.get(), CURLOPT_MAXFILESIZE_LARGE,
                         static_cast<curl_off_t>(expected_bytes)));
  check(
      curl_easy_setopt(handle.get(), CURLOPT_USERAGENT, "Liuding-Forge/MNIST"));
  check(curl_easy_setopt(handle.get(), CURLOPT_WRITEFUNCTION, &receive));
  check(curl_easy_setopt(handle.get(), CURLOPT_WRITEDATA, &receiver));
  check(
      curl_easy_setopt(handle.get(), CURLOPT_HEADERFUNCTION, &receive_header));
  check(curl_easy_setopt(handle.get(), CURLOPT_HEADERDATA, &receiver));
  const auto result = curl_easy_perform(handle.get());
  if (receiver.error) std::rethrow_exception(receiver.error);
  check(result);
  // The selected 7.81 SDK documents getinfo results after transfer completion.
  // No response parsing or getinfo calls occur inside native callbacks.
  long status = 0L;
  curl_off_t length = -1;
  check(curl_easy_getinfo(handle.get(), CURLINFO_RESPONSE_CODE, &status));
  check(curl_easy_getinfo(handle.get(), CURLINFO_CONTENT_LENGTH_DOWNLOAD_T,
                          &length));
  if (status < 0L ||
      static_cast<unsigned long>(status) >
          std::numeric_limits<std::uint32_t>::max() ||
      length < -1)
    throw IoError("Invalid MNIST final HTTPS response metadata");
  callbacks.headers(static_cast<std::uint32_t>(status),
                    length < 0 ? std::nullopt
                               : std::optional<std::uint64_t>(
                                     static_cast<std::uint64_t>(length)));
}
}  // namespace jyppx::liuding::datasets::detail
