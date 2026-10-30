// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "mnist_download_detail.hpp"

#include <jyppx/liuding/foundation/error.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <optional>
#include <string>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winhttp.h>

namespace jyppx::liuding::datasets::detail {
namespace {
class InternetHandle final {
 public:
  explicit InternetHandle(HINTERNET handle) : handle_(handle) {
    if (!handle_)
      throw IoError("Could not open MNIST WinHTTP handle; OS code " +
                    std::to_string(GetLastError()));
  }
  ~InternetHandle() { WinHttpCloseHandle(handle_); }
  InternetHandle(const InternetHandle&) = delete;
  InternetHandle& operator=(const InternetHandle&) = delete;
  HINTERNET get() const noexcept { return handle_; }

 private:
  HINTERNET handle_;
};

void check(BOOL result) {
  if (!result)
    throw IoError("MNIST HTTPS operation failed in WinHTTP; OS code " +
                  std::to_string(GetLastError()));
}
}  // namespace

void transfer_mnist_https(std::string_view url, std::uint64_t expected_bytes,
                          const MnistHttpsCallbacks& callbacks) {
  require_mnist_https_url(url);
  if (expected_bytes == 0U || expected_bytes > 9912422U)
    throw InvalidArgument("Invalid MNIST HTTPS size bound");
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(120);
  const InternetHandle session(
      WinHttpOpen(L"Liuding-Forge/MNIST", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
                  WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0U));
  // Leave certificate chain/hostname validation enabled. Never set
  // SECURITY_FLAG_IGNORE_*, inject trust, or follow any redirect.
  DWORD protocols = WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_2;
  check(WinHttpSetOption(session.get(), WINHTTP_OPTION_SECURE_PROTOCOLS,
                         &protocols, sizeof(protocols)));
  const InternetHandle connection(
      WinHttpConnect(session.get(), L"storage.googleapis.com",
                     INTERNET_DEFAULT_HTTPS_PORT, 0U));
  constexpr std::string_view origin = "https://storage.googleapis.com";
  const std::wstring resource(url.begin() + origin.size(), url.end());
  const InternetHandle request(WinHttpOpenRequest(
      connection.get(), L"GET", resource.c_str(), nullptr, WINHTTP_NO_REFERER,
      WINHTTP_DEFAULT_ACCEPT_TYPES, WINHTTP_FLAG_SECURE));
  DWORD redirects = WINHTTP_OPTION_REDIRECT_POLICY_NEVER;
  check(WinHttpSetOption(request.get(), WINHTTP_OPTION_REDIRECT_POLICY,
                         &redirects, sizeof(redirects)));
  DWORD logon = WINHTTP_AUTOLOGON_SECURITY_LEVEL_HIGH;
  check(WinHttpSetOption(request.get(), WINHTTP_OPTION_AUTOLOGON_POLICY, &logon,
                         sizeof(logon)));
  DWORD maximum_headers = 64U * 1024U;
  check(WinHttpSetOption(request.get(), WINHTTP_OPTION_MAX_RESPONSE_HEADER_SIZE,
                         &maximum_headers, sizeof(maximum_headers)));
  const auto check_deadline = [&] {
    if (std::chrono::steady_clock::now() >= deadline)
      throw IoError("MNIST HTTPS exceeded the 120-second transfer deadline");
  };
  const auto set_remaining_timeout = [&] {
    const auto remaining =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now())
            .count();
    if (remaining <= 0)
      throw IoError("MNIST HTTPS exceeded the 120-second transfer deadline");
    const auto timeout =
        static_cast<int>(std::min<std::int64_t>(remaining, 30000));
    check(
        WinHttpSetTimeouts(request.get(), timeout, timeout, timeout, timeout));
  };
  set_remaining_timeout();
  check(WinHttpSendRequest(request.get(), L"Accept-Encoding: identity\r\n",
                           static_cast<DWORD>(-1L), WINHTTP_NO_REQUEST_DATA, 0U,
                           0U, 0U));
  check_deadline();
  set_remaining_timeout();
  check(WinHttpReceiveResponse(request.get(), nullptr));
  check_deadline();
  DWORD status = 0U;
  DWORD size = sizeof(status);
  check(WinHttpQueryHeaders(
      request.get(), WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
      WINHTTP_HEADER_NAME_BY_INDEX, &status, &size, WINHTTP_NO_HEADER_INDEX));
  std::optional<std::uint64_t> length;
  DWORD content_length = 0U;
  size = sizeof(content_length);
  if (WinHttpQueryHeaders(
          request.get(),
          WINHTTP_QUERY_CONTENT_LENGTH | WINHTTP_QUERY_FLAG_NUMBER,
          WINHTTP_HEADER_NAME_BY_INDEX, &content_length, &size,
          WINHTTP_NO_HEADER_INDEX)) {
    length = content_length;
  } else if (GetLastError() != ERROR_WINHTTP_HEADER_NOT_FOUND) {
    throw IoError("Could not read MNIST HTTPS Content-Length");
  }
  check_deadline();
  callbacks.headers(status, length);
  check_deadline();
  std::array<std::byte, kMnistDownloadChunkBytes> block{};
  for (;;) {
    set_remaining_timeout();
    DWORD count = 0U;
    check(WinHttpReadData(request.get(), block.data(),
                          static_cast<DWORD>(block.size()), &count));
    // Also check the final zero-byte read before accepting EOF. Per-phase
    // WinHTTP timeouts alone do not enforce a hard wall-clock cancellation.
    check_deadline();
    if (count == 0U) break;
    callbacks.bytes(std::span(block).first(count));
    check_deadline();
  }
  check_deadline();
}
}  // namespace jyppx::liuding::datasets::detail
