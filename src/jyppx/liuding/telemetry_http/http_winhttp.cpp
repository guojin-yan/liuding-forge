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

#include <algorithm>
#include <array>
#include <atomic>
#include <memory>
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

struct Completion final {
  Completion() : ready(CreateEventW(nullptr, FALSE, FALSE, nullptr)) {
    if (!ready) throw IoError("Could not create OTLP completion event");
  }
  ~Completion() { CloseHandle(ready); }
  HANDLE ready;
  std::atomic<DWORD> status{0U};
  std::atomic<DWORD> error{ERROR_SUCCESS};
  std::atomic<DWORD> bytes{0U};
  std::atomic<DWORD> secure_failures{0U};
  // WinHTTP can still reference an async read/write buffer until
  // HANDLE_CLOSING. Keep a bounded chunk alive with the callback, never the
  // caller's payload.
  std::array<char, 16U * 1024U> chunk{};
};

using CallbackOwner = std::shared_ptr<Completion>;

void CALLBACK completed(HINTERNET, DWORD_PTR context, DWORD status,
                        void* information, DWORD length) noexcept {
  if (context == 0U) return;
  auto* owner = reinterpret_cast<CallbackOwner*>(context);
  const auto state = *owner;
  if (status == WINHTTP_CALLBACK_STATUS_HANDLE_CLOSING) {
    // The last notification releases the native callback's ownership even if
    // cancellation has already returned to the caller. No stack pointer
    // escapes.
    delete owner;
    return;
  }
  if (status == WINHTTP_CALLBACK_STATUS_SECURE_FAILURE) {
    // This notification precedes REQUEST_ERROR and supplies the validation
    // reason hidden by the generic ERROR_WINHTTP_SECURE_FAILURE result.
    if (information && length == sizeof(DWORD))
      state->secure_failures.fetch_or(*static_cast<DWORD*>(information));
    return;
  } else if (status == WINHTTP_CALLBACK_STATUS_REQUEST_ERROR) {
    state->error =
        information && length == sizeof(WINHTTP_ASYNC_RESULT)
            ? static_cast<WINHTTP_ASYNC_RESULT*>(information)->dwError
            : ERROR_INVALID_DATA;
  } else if (status == WINHTTP_CALLBACK_STATUS_SENDREQUEST_COMPLETE ||
             status == WINHTTP_CALLBACK_STATUS_HEADERS_AVAILABLE ||
             status == WINHTTP_CALLBACK_STATUS_READ_COMPLETE ||
             status == WINHTTP_CALLBACK_STATUS_WRITE_COMPLETE) {
    if (status == WINHTTP_CALLBACK_STATUS_READ_COMPLETE) {
      state->bytes = length;
    } else if (status == WINHTTP_CALLBACK_STATUS_WRITE_COMPLETE) {
      if (information && length == sizeof(DWORD))
        state->bytes = *static_cast<DWORD*>(information);
      else
        state->error = ERROR_INVALID_DATA;
    }
    state->status.store(status, std::memory_order_release);
  } else {
    return;
  }
  SetEvent(state->ready);
}

void check_certificate_error(DWORD code, DWORD secure_failures) {
  constexpr DWORD certificate_failures =
      WINHTTP_CALLBACK_STATUS_FLAG_CERT_REV_FAILED |
      WINHTTP_CALLBACK_STATUS_FLAG_INVALID_CERT |
      WINHTTP_CALLBACK_STATUS_FLAG_CERT_REVOKED |
      WINHTTP_CALLBACK_STATUS_FLAG_INVALID_CA |
      WINHTTP_CALLBACK_STATUS_FLAG_CERT_CN_INVALID |
      WINHTTP_CALLBACK_STATUS_FLAG_CERT_DATE_INVALID |
      WINHTTP_CALLBACK_STATUS_FLAG_CERT_WRONG_USAGE;
  const bool direct_certificate_error =
      code == ERROR_WINHTTP_SECURE_CERT_DATE_INVALID ||
      code == ERROR_WINHTTP_SECURE_CERT_CN_INVALID ||
      code == ERROR_WINHTTP_SECURE_INVALID_CA ||
      code == ERROR_WINHTTP_SECURE_CERT_REV_FAILED ||
      code == ERROR_WINHTTP_SECURE_INVALID_CERT ||
      code == ERROR_WINHTTP_SECURE_CERT_REVOKED ||
      code == ERROR_WINHTTP_SECURE_CERT_WRONG_USAGE;
  if (direct_certificate_error ||
      (code == ERROR_WINHTTP_SECURE_FAILURE &&
       (secure_failures & certificate_failures) != 0U))
    throw_http_certificate_error("WinHTTP OS code " + std::to_string(code) +
                                 "; secure failure flags " +
                                 std::to_string(secure_failures));
}

class Request final {
 public:
  explicit Request(HINTERNET handle)
      : handle_(handle), state_(std::make_shared<Completion>()) {
    auto owner = std::make_unique<CallbackOwner>(state_);
    context_ = reinterpret_cast<DWORD_PTR>(owner.get());
    if (!WinHttpSetOption(handle_.get(), WINHTTP_OPTION_CONTEXT_VALUE,
                          &context_, sizeof(context_)))
      throw IoError("Could not set OTLP asynchronous request context");
    if (WinHttpSetStatusCallback(handle_.get(), &completed,
                                 WINHTTP_CALLBACK_FLAG_ALL_COMPLETIONS |
                                     WINHTTP_CALLBACK_FLAG_HANDLES |
                                     WINHTTP_CALLBACK_FLAG_SECURE_FAILURE,
                                 0U) == WINHTTP_INVALID_STATUS_CALLBACK)
      throw IoError("Could not install OTLP asynchronous completion callback");
    owner.release();  // HANDLE_CLOSING now owns this one reference.
  }
  HINTERNET get() const noexcept { return handle_.get(); }
  DWORD_PTR context() const noexcept { return context_; }
  Completion& state() noexcept { return *state_; }
  void begin() {
    if (!ResetEvent(state_->ready))
      throw IoError("Could not reset OTLP completion event");
    state_->status = 0U;
    state_->error = ERROR_SUCCESS;
    state_->bytes = 0U;
    state_->secure_failures = 0U;
  }
  DWORD wait(BOOL started, DWORD expected, std::stop_token stop,
             std::chrono::steady_clock::time_point deadline) {
    const auto error = started ? ERROR_SUCCESS : GetLastError();
    check_telemetry_operation(stop, deadline);
    if (error == ERROR_WINHTTP_TIMEOUT)
      throw DeadlineExceeded("OTLP HTTP request deadline exceeded");
    check_certificate_error(error, 0U);
    if (error != ERROR_SUCCESS && error != ERROR_IO_PENDING)
      throw IoError("OTLP asynchronous WinHTTP operation failed; OS code " +
                    std::to_string(error));
    const std::stop_callback cancelled(stop, [&] { SetEvent(state_->ready); });
    for (;;) {
      const auto milliseconds = http_remaining_milliseconds(stop, deadline);
      const auto result = WaitForSingleObject(state_->ready, milliseconds);
      check_telemetry_operation(stop, deadline);
      if (result == WAIT_TIMEOUT) continue;
      if (result != WAIT_OBJECT_0)
        throw IoError("Could not wait for OTLP asynchronous completion");
      break;
    }
    const auto completed_error = state_->error.load(std::memory_order_acquire);
    if (completed_error == ERROR_WINHTTP_TIMEOUT)
      throw DeadlineExceeded("OTLP HTTP request deadline exceeded");
    check_certificate_error(completed_error, state_->secure_failures.load());
    if (completed_error != ERROR_SUCCESS ||
        state_->status.load(std::memory_order_acquire) != expected)
      throw IoError("OTLP asynchronous WinHTTP completion failed; OS code " +
                    std::to_string(completed_error));
    return state_->bytes.load();
  }

 private:
  // The callback independently owns state until the last HANDLE_CLOSING notice.
  Handle handle_;
  CallbackOwner state_;
  DWORD_PTR context_ = 0U;
};

template <typename Validate>
std::size_t validate_response_headers(
    HINTERNET request, DWORD query, Validate&& validate, std::stop_token stop,
    std::chrono::steady_clock::time_point deadline,
    const wchar_t* name = WINHTTP_HEADER_NAME_BY_INDEX) {
  std::size_t count = 0U;
  DWORD index = 0U;
  for (;;) {
    DWORD next_index = index;
    DWORD bytes = 0U;
    const auto sized = WinHttpQueryHeaders(
        request, query, name, WINHTTP_NO_OUTPUT_BUFFER, &bytes, &next_index);
    const auto code = sized ? ERROR_SUCCESS : GetLastError();
    check_telemetry_operation(stop, deadline);
    if (!sized && code == ERROR_WINHTTP_HEADER_NOT_FOUND) return count;
    if (sized || code != ERROR_INSUFFICIENT_BUFFER || bytes == 0U ||
        bytes > 64U * 1024U || bytes % sizeof(wchar_t) != 0U)
      throw IoError("Could not read bounded OTLP HTTP response metadata");
    std::wstring value(bytes / sizeof(wchar_t), L'\0');
    next_index = index;  // Size queries must not advance enumeration.
    check(WinHttpQueryHeaders(request, query, name, value.data(), &bytes,
                              &next_index),
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
    // Keep WinHTTP's process runtime initialized across isolated sessions.
    // This anchor never connects or sends; every request still owns a separate
    // session so a stalled TLS handshake can be closed on cancellation.
    static const Handle runtime(WinHttpOpen(
        L"Liuding-Forge/OTLP-runtime", WINHTTP_ACCESS_TYPE_NO_PROXY,
        WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, WINHTTP_FLAG_ASYNC));
    static_cast<void>(runtime);
    std::string reply;
    {
      const Handle session(WinHttpOpen(
          L"Liuding-Forge/OTLP", WINHTTP_ACCESS_TYPE_NO_PROXY,
          WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, WINHTTP_FLAG_ASYNC));
      // Completion callbacks only publish atomics and signal an event. They
      // never wait for callers or for WinHTTP when cancellation races a
      // completion.
      BOOL nonblocking = TRUE;
      check(WinHttpSetOption(session.get(),
                             WINHTTP_OPTION_ASSURED_NON_BLOCKING_CALLBACKS,
                             &nonblocking, sizeof(nonblocking)),
            stop, deadline);
      // Keep pending handshakes owned by this request's session, rather than
      // letting a global connection pool outlive cooperative cancellation.
      BOOL isolated = TRUE;
      check(
          WinHttpSetOption(session.get(), WINHTTP_OPTION_DISABLE_GLOBAL_POOLING,
                           &isolated, sizeof(isolated)),
          stop, deadline);
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
      Request request(
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
      request.begin();
      static_cast<void>(request.wait(
          WinHttpSendRequest(
              request.get(),
              L"Content-Type: application/x-protobuf\r\nAccept-Encoding: "
              L"identity\r\n",
              static_cast<DWORD>(-1L), WINHTTP_NO_REQUEST_DATA, 0U,
              static_cast<DWORD>(prepared.body.size()), request.context()),
          WINHTTP_CALLBACK_STATUS_SENDREQUEST_COMPLETE, stop, deadline));
      std::size_t offset = 0U;
      while (offset < prepared.body.size()) {
        set_timeout();
        auto& chunk = request.state().chunk;
        const auto count =
            std::min(chunk.size(), prepared.body.size() - offset);
        std::copy_n(prepared.body.data() + offset, count, chunk.data());
        request.begin();
        const auto written = request.wait(
            WinHttpWriteData(request.get(), chunk.data(),
                             static_cast<DWORD>(count), nullptr),
            WINHTTP_CALLBACK_STATUS_WRITE_COMPLETE, stop, deadline);
        if (written == 0U || written > count)
          throw IoError("Invalid asynchronous OTLP write completion");
        offset += written;
      }
      set_timeout();
      request.begin();
      static_cast<void>(request.wait(
          WinHttpReceiveResponse(request.get(), nullptr),
          WINHTTP_CALLBACK_STATUS_HEADERS_AVAILABLE, stop, deadline));
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
      static_cast<void>(validate_response_headers(
          request.get(), WINHTTP_QUERY_CUSTOM,
          [](std::string_view value) {
            validate_http_response_trailer_declaration(value);
          },
          stop, deadline, L"Trailer"));
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
      for (;;) {
        set_timeout();
        auto& chunk = request.state().chunk;
        request.begin();
        const auto count = request.wait(
            WinHttpReadData(request.get(), chunk.data(),
                            static_cast<DWORD>(chunk.size()), nullptr),
            WINHTTP_CALLBACK_STATUS_READ_COMPLETE, stop, deadline);
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
