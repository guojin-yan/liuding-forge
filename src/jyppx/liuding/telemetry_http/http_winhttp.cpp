// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "http_transport.hpp"

#include <jyppx/liuding/foundation/error.hpp>

#include <array>
#include <utility>
#include <windows.h>
#include <winhttp.h>
#include <winsock2.h>
#include <ws2tcpip.h>

namespace jyppx::liuding::experiment::detail {
namespace {
class Handle final {
 public:
  explicit Handle(HINTERNET value) : value_(value) {
    if (!value_)
      throw IoError("Could not open OTLP WinHTTP handle; OS code " +
                    std::to_string(GetLastError()));
  }
  ~Handle() { WinHttpCloseHandle(value_); }
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  HINTERNET get() const noexcept { return value_; }

 private:
  HINTERNET value_;
};

void check(BOOL result, std::stop_token stop,
           std::chrono::steady_clock::time_point deadline) {
  const auto code = result ? ERROR_SUCCESS : GetLastError();
  check_telemetry_operation(stop, deadline);
  if (!result)
    throw IoError("OTLP WinHTTP operation failed; OS code " +
                  std::to_string(code));
}

template <typename Validate>
std::size_t validate_response_headers(
    HINTERNET request, DWORD query, Validate&& validate, std::stop_token stop,
    std::chrono::steady_clock::time_point deadline) {
  std::size_t count = 0U;
  DWORD index = 0U;
  for (;;) {
    DWORD next_index = index;
    DWORD bytes = 0U;
    const auto sized =
        WinHttpQueryHeaders(request, query, WINHTTP_HEADER_NAME_BY_INDEX,
                            WINHTTP_NO_OUTPUT_BUFFER, &bytes, &next_index);
    const auto code = sized ? ERROR_SUCCESS : GetLastError();
    check_telemetry_operation(stop, deadline);
    if (!sized && code == ERROR_WINHTTP_HEADER_NOT_FOUND) return count;
    if (sized || code != ERROR_INSUFFICIENT_BUFFER || bytes == 0U ||
        bytes > 64U * 1024U || bytes % sizeof(wchar_t) != 0U)
      throw IoError("Could not read bounded OTLP HTTP response metadata");
    std::wstring value(bytes / sizeof(wchar_t), L'\0');
    next_index = index;  // Size queries must not advance enumeration.
    check(WinHttpQueryHeaders(request, query, WINHTTP_HEADER_NAME_BY_INDEX,
                              value.data(), &bytes, &next_index),
          stop, deadline);
    if (next_index <= index)
      throw IoError("OTLP HTTP response metadata enumeration did not advance");
    index = next_index;
    const auto end = value.find(L'\0');
    if (end != std::wstring::npos) value.resize(end);
    std::string field;
    field.reserve(value.size());
    for (const auto character : value) {
      if (static_cast<unsigned int>(character) > 255U)
        throw IoError("OTLP HTTP response metadata is not a byte field");
      field.push_back(static_cast<char>(character));
    }
    validate(field);
    ++count;
  }
}

class Sender final : public OtlpRequestSender {
 public:
  Sender(OtlpHttpEndpoint endpoint, std::chrono::milliseconds timeout)
      : endpoint_(std::move(endpoint)), timeout_(timeout) {}

  std::string send(const PreparedOtlpRequest& prepared, std::size_t reply_limit,
                   std::stop_token stop,
                   TelemetryDeadline caller_deadline) override {
    validate_http_request(prepared, reply_limit);
    const auto deadline =
        http_request_deadline(stop, caller_deadline, timeout_);
    std::string reply;
    {
      const Handle session(
          WinHttpOpen(L"Liuding-Forge/OTLP", WINHTTP_ACCESS_TYPE_NO_PROXY,
                      WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0U));
      DWORD protocols = WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_2;
      check(WinHttpSetOption(session.get(), WINHTTP_OPTION_SECURE_PROTOCOLS,
                             &protocols, sizeof(protocols)),
            stop, deadline);
      const auto host = endpoint_.host.find(':') == std::string::npos
                            ? endpoint_.host
                            : "[" + endpoint_.host + "]";
      const std::wstring wide_host(host.begin(), host.end());
      const Handle connection(
          WinHttpConnect(session.get(), wide_host.c_str(), endpoint_.port, 0U));
      const auto path = OtlpProjectionEncoder::signal_path(prepared.signal);
      const std::wstring wide_path(path.begin(), path.end());
      const wchar_t* accept[] = {L"application/x-protobuf", nullptr};
      const Handle request(
          WinHttpOpenRequest(connection.get(), L"POST", wide_path.c_str(),
                             L"HTTP/1.1", WINHTTP_NO_REFERER, accept,
                             endpoint_.tls ? WINHTTP_FLAG_SECURE : 0U));
      DWORD redirects = WINHTTP_OPTION_REDIRECT_POLICY_NEVER;
      check(WinHttpSetOption(request.get(), WINHTTP_OPTION_REDIRECT_POLICY,
                             &redirects, sizeof(redirects)),
            stop, deadline);
      DWORD disabled = WINHTTP_DISABLE_AUTHENTICATION |
                       WINHTTP_DISABLE_COOKIES | WINHTTP_DISABLE_REDIRECTS;
      check(WinHttpSetOption(request.get(), WINHTTP_OPTION_DISABLE_FEATURE,
                             &disabled, sizeof(disabled)),
            stop, deadline);
      DWORD logon = WINHTTP_AUTOLOGON_SECURITY_LEVEL_HIGH;
      check(WinHttpSetOption(request.get(), WINHTTP_OPTION_AUTOLOGON_POLICY,
                             &logon, sizeof(logon)),
            stop, deadline);
      DWORD maximum_headers = 64U * 1024U;
      check(WinHttpSetOption(request.get(),
                             WINHTTP_OPTION_MAX_RESPONSE_HEADER_SIZE,
                             &maximum_headers, sizeof(maximum_headers)),
            stop, deadline);
      const auto set_timeout = [&] {
        const auto ms = http_remaining_milliseconds(stop, deadline);
        check(WinHttpSetTimeouts(request.get(), ms, ms, ms, ms), stop,
              deadline);
      };
      // Trust/hostname validation remains at WinHTTP's secure defaults.
      set_timeout();
      check(
          WinHttpSendRequest(
              request.get(),
              L"Content-Type: application/x-protobuf\r\nAccept-Encoding: "
              L"identity\r\n",
              static_cast<DWORD>(-1L), const_cast<char*>(prepared.body.data()),
              static_cast<DWORD>(prepared.body.size()),
              static_cast<DWORD>(prepared.body.size()), 0U),
          stop, deadline);
      set_timeout();
      check(WinHttpReceiveResponse(request.get(), nullptr), stop, deadline);
      DWORD status = 0U;
      DWORD size = sizeof(status);
      check(WinHttpQueryHeaders(
                request.get(),
                WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                WINHTTP_HEADER_NAME_BY_INDEX, &status, &size,
                WINHTTP_NO_HEADER_INDEX),
            stop, deadline);
      if (status != 200U)
        throw IoError("OTLP collector returned HTTP status " +
                      std::to_string(status));
      // Query WinHTTP's parsed response fields before reading or accepting a
      // protobuf body. Enumerate duplicates so an unsupported coding cannot
      // be hidden behind an earlier identity value.
      const auto content_types = validate_response_headers(
          request.get(), WINHTTP_QUERY_CONTENT_TYPE,
          [](std::string_view value) {
            validate_http_response_content_type(value);
          },
          stop, deadline);
      if (content_types != 1U)
        throw IoError("OTLP collector response requires one content type");
      static_cast<void>(validate_response_headers(
          request.get(), WINHTTP_QUERY_CONTENT_ENCODING,
          [](std::string_view value) {
            validate_http_response_content_encoding(value);
          },
          stop, deadline));
      DWORD length = 0U;
      size = sizeof(length);
      if (WinHttpQueryHeaders(
              request.get(),
              WINHTTP_QUERY_CONTENT_LENGTH | WINHTTP_QUERY_FLAG_NUMBER,
              WINHTTP_HEADER_NAME_BY_INDEX, &length, &size,
              WINHTTP_NO_HEADER_INDEX)) {
        if (length > reply_limit)
          throw ResourceExhausted(
              "OTLP response exceeds the encoded reply bound");
      } else if (GetLastError() != ERROR_WINHTTP_HEADER_NOT_FOUND) {
        throw IoError("Could not read OTLP response length");
      }
      std::array<char, 16U * 1024U> chunk{};
      for (;;) {
        set_timeout();
        DWORD count = 0U;
        check(WinHttpReadData(request.get(), chunk.data(),
                              static_cast<DWORD>(chunk.size()), &count),
              stop, deadline);
        if (count == 0U) break;
        if (count > reply_limit - reply.size())
          throw ResourceExhausted(
              "OTLP response exceeds the encoded reply bound");
        reply.append(chunk.data(), count);
      }
    }  // Destroy request/connection/session before the final deadline check.
    check_telemetry_operation(stop, deadline);
    return reply;
  }

 private:
  OtlpHttpEndpoint endpoint_;
  std::chrono::milliseconds timeout_;
};
}  // namespace

std::unique_ptr<OtlpRequestSender> make_http_sender(
    OtlpHttpEndpoint endpoint, std::chrono::milliseconds timeout) {
  validate_http_endpoint(endpoint);
  if (endpoint.host.find(':') != std::string::npos) {
    IN6_ADDR address{};
    if (InetPtonA(AF_INET6, endpoint.host.c_str(), &address) != 1)
      throw InvalidArgument("Invalid OTLP IPv6 literal");
  }
  if (timeout < std::chrono::milliseconds(1) ||
      timeout > std::chrono::milliseconds(30000))
    throw InvalidArgument("OTLP request timeout must be in [1, 30000] ms");
  return std::make_unique<Sender>(std::move(endpoint), timeout);
}
}  // namespace jyppx::liuding::experiment::detail
