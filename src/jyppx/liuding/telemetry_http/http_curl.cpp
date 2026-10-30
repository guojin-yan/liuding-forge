// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "http_transport.hpp"

#include <jyppx/liuding/foundation/error.hpp>

#include <arpa/inet.h>
#include <curl/curl.h>
#include <exception>
#include <limits>
#include <memory>
#include <utility>

namespace jyppx::liuding::experiment::detail {
namespace {
class Runtime final {
 public:
  Runtime() {
    if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK)
      throw IoError("Could not initialize OTLP system libcurl");
  }
  ~Runtime() { curl_global_cleanup(); }
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;
};
void check(CURLcode result) {
  if (result != CURLE_OK)
    throw IoError(std::string("OTLP system libcurl operation failed: ") +
                  curl_easy_strerror(result));
}
struct Receiver {
  std::stop_token stop;
  std::chrono::steady_clock::time_point deadline;
  std::size_t reply_limit;
  std::string reply;
  std::size_t header_bytes = 0U;
  bool content_type_seen = false;
  bool header_block_complete = false;
  std::exception_ptr error;
};
std::size_t byte_count(std::size_t size, std::size_t count) {
  if (size != 0U && count > std::numeric_limits<std::size_t>::max() / size)
    throw ResourceExhausted("OTLP HTTP callback byte count overflow");
  return size * count;
}
std::size_t receive_header(char* data, std::size_t size, std::size_t count,
                           void* context) noexcept {
  auto& receiver = *static_cast<Receiver*>(context);
  try {
    check_telemetry_operation(receiver.stop, receiver.deadline);
    const auto bytes = byte_count(size, count);
    if (bytes > 65536U - receiver.header_bytes)
      throw ResourceExhausted("OTLP HTTP headers exceed 64 KiB");
    receiver.header_bytes += bytes;
    // libcurl owns HTTP framing. Its complete header-line callback supplies
    // representation metadata, which must be checked before protobuf ACK.
    auto line = std::string_view(data, bytes);
    if (!line.empty() && line.back() == '\n') line.remove_suffix(1U);
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1U);
    if (line.starts_with("HTTP/")) {
      receiver.content_type_seen = false;  // A later final header block wins.
      receiver.header_block_complete = false;
    } else if (line.empty()) {
      receiver.header_block_complete = true;
    } else {
      if (line.front() == ' ' || line.front() == '\t')
        throw IoError("OTLP collector returned obsolete folded HTTP headers");
      const auto colon = line.find(':');
      if (colon != std::string_view::npos) {
        const auto name = line.substr(0U, colon);
        const auto value = line.substr(colon + 1U);
        if (http_ascii_case_equal(name, "Content-Type")) {
          if (receiver.content_type_seen || receiver.header_block_complete)
            throw IoError("OTLP collector returned duplicate content types");
          validate_http_response_content_type(value);
          receiver.content_type_seen = true;
        } else if (http_ascii_case_equal(name, "Content-Encoding")) {
          if (receiver.header_block_complete)
            throw IoError("OTLP collector returned a content-coding trailer");
          validate_http_response_content_encoding(value);
        }
      }
    }
    return bytes;
  } catch (...) {
    receiver.error = std::current_exception();
    return 0U;
  }
}
std::size_t receive_body(char* data, std::size_t size, std::size_t count,
                         void* context) noexcept {
  auto& receiver = *static_cast<Receiver*>(context);
  try {
    check_telemetry_operation(receiver.stop, receiver.deadline);
    const auto bytes = byte_count(size, count);
    if (bytes > receiver.reply_limit - receiver.reply.size())
      throw ResourceExhausted("OTLP response exceeds the encoded reply bound");
    receiver.reply.append(data, bytes);
    return bytes;
  } catch (...) {
    receiver.error = std::current_exception();
    return 0U;
  }
}
int progress(void* context, curl_off_t, curl_off_t, curl_off_t,
             curl_off_t) noexcept {
  auto& receiver = *static_cast<Receiver*>(context);
  try {
    check_telemetry_operation(receiver.stop, receiver.deadline);
    return 0;
  } catch (...) {
    receiver.error = std::current_exception();
    return 1;
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
    static const Runtime runtime;
    static_cast<void>(runtime);
    const auto* version = curl_version_info(CURLVERSION_NOW);
    if (!version || version->version_num < 0x075100U ||
        (version->features & CURL_VERSION_SSL) == 0 ||
        (version->features & CURL_VERSION_ASYNCHDNS) == 0)
      throw NotImplemented(
          "OTLP requires shared curl >= 7.81 with TLS and asynchronous DNS");
    Receiver receiver{.stop = stop,
                      .deadline = deadline,
                      .reply_limit = reply_limit,
                      .reply = {},
                      .header_bytes = 0U,
                      .content_type_seen = false,
                      .header_block_complete = false,
                      .error = {}};
    {
      std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)> headers(
          nullptr, &curl_slist_free_all);
      const std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> handle(
          curl_easy_init(), &curl_easy_cleanup);
      if (!handle)
        throw IoError("Could not create OTLP system libcurl request");
      const auto host = endpoint_.host.find(':') == std::string::npos
                            ? endpoint_.host
                            : "[" + endpoint_.host + "]";
      const auto url =
          std::string(endpoint_.tls ? "https://" : "http://") + host + ":" +
          std::to_string(endpoint_.port) +
          std::string(OtlpProjectionEncoder::signal_path(prepared.signal));
      check(curl_easy_setopt(handle.get(), CURLOPT_URL, url.c_str()));
#if LIBCURL_VERSION_NUM >= 0x075500
      if (version->version_num < 0x075500U)
        throw NotImplemented("Selected curl headers require a runtime >= 7.85");
      check(curl_easy_setopt(handle.get(), CURLOPT_PROTOCOLS_STR,
                             endpoint_.tls ? "https" : "http"));
      check(curl_easy_setopt(handle.get(), CURLOPT_REDIR_PROTOCOLS_STR,
                             endpoint_.tls ? "https" : "http"));
#else
      const auto protocol =
          static_cast<long>(endpoint_.tls ? CURLPROTO_HTTPS : CURLPROTO_HTTP);
      check(curl_easy_setopt(handle.get(), CURLOPT_PROTOCOLS, protocol));
      check(curl_easy_setopt(handle.get(), CURLOPT_REDIR_PROTOCOLS, protocol));
#endif
      check(curl_easy_setopt(handle.get(), CURLOPT_PROXY, ""));
      check(curl_easy_setopt(handle.get(), CURLOPT_NETRC,
                             static_cast<long>(CURL_NETRC_IGNORED)));
      check(curl_easy_setopt(handle.get(), CURLOPT_HTTPAUTH,
                             static_cast<long>(CURLAUTH_NONE)));
      check(curl_easy_setopt(handle.get(), CURLOPT_FOLLOWLOCATION, 0L));
      check(curl_easy_setopt(handle.get(), CURLOPT_MAXREDIRS, 0L));
      check(curl_easy_setopt(handle.get(), CURLOPT_SSL_VERIFYPEER, 1L));
      check(curl_easy_setopt(handle.get(), CURLOPT_SSL_VERIFYHOST, 2L));
      check(curl_easy_setopt(handle.get(), CURLOPT_SSLVERSION,
                             static_cast<long>(CURL_SSLVERSION_TLSv1_2)));
      check(curl_easy_setopt(handle.get(), CURLOPT_NOSIGNAL, 1L));
      const auto remaining =
          static_cast<long>(http_remaining_milliseconds(stop, deadline));
      check(
          curl_easy_setopt(handle.get(), CURLOPT_CONNECTTIMEOUT_MS, remaining));
      check(curl_easy_setopt(handle.get(), CURLOPT_TIMEOUT_MS, remaining));
      check(curl_easy_setopt(handle.get(), CURLOPT_HTTP_VERSION,
                             static_cast<long>(CURL_HTTP_VERSION_1_1)));
      check(
          curl_easy_setopt(handle.get(), CURLOPT_ACCEPT_ENCODING, "identity"));
      check(curl_easy_setopt(handle.get(), CURLOPT_HTTP_CONTENT_DECODING, 0L));
      check(curl_easy_setopt(handle.get(), CURLOPT_MAXFILESIZE_LARGE,
                             static_cast<curl_off_t>(reply_limit)));
      check(curl_easy_setopt(handle.get(), CURLOPT_USERAGENT,
                             "Liuding-Forge/OTLP"));
      check(curl_easy_setopt(handle.get(), CURLOPT_POST, 1L));
      check(curl_easy_setopt(handle.get(), CURLOPT_POSTFIELDS,
                             prepared.body.data()));
      check(curl_easy_setopt(handle.get(), CURLOPT_POSTFIELDSIZE_LARGE,
                             static_cast<curl_off_t>(prepared.body.size())));
      curl_slist* raw_headers = nullptr;
      for (const auto* header : {"Content-Type: application/x-protobuf",
                                 "Accept: application/x-protobuf", "Expect:"}) {
        auto* next = curl_slist_append(raw_headers, header);
        if (!next) {
          curl_slist_free_all(raw_headers);
          throw ResourceExhausted("Could not allocate OTLP fixed HTTP headers");
        }
        raw_headers = next;
      }
      headers.reset(raw_headers);
      check(curl_easy_setopt(handle.get(), CURLOPT_HTTPHEADER, headers.get()));
      check(curl_easy_setopt(handle.get(), CURLOPT_HEADERFUNCTION,
                             &receive_header));
      check(curl_easy_setopt(handle.get(), CURLOPT_HEADERDATA, &receiver));
      check(
          curl_easy_setopt(handle.get(), CURLOPT_WRITEFUNCTION, &receive_body));
      check(curl_easy_setopt(handle.get(), CURLOPT_WRITEDATA, &receiver));
      check(curl_easy_setopt(handle.get(), CURLOPT_NOPROGRESS, 0L));
      check(
          curl_easy_setopt(handle.get(), CURLOPT_XFERINFOFUNCTION, &progress));
      check(curl_easy_setopt(handle.get(), CURLOPT_XFERINFODATA, &receiver));
      const auto result = curl_easy_perform(handle.get());
      if (receiver.error) std::rethrow_exception(receiver.error);
      check_telemetry_operation(stop, deadline);
      check(result);
      long status = 0L;
      check(curl_easy_getinfo(handle.get(), CURLINFO_RESPONSE_CODE, &status));
      if (status != 200L)
        throw IoError("OTLP collector returned HTTP status " +
                      std::to_string(status));
      if (!receiver.content_type_seen)
        throw IoError("OTLP collector response is missing its content type");
    }  // Handle/header cleanup is included in the final cooperative check.
    check_telemetry_operation(stop, deadline);
    return std::move(receiver.reply);
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
    in6_addr address{};
    if (inet_pton(AF_INET6, endpoint.host.c_str(), &address) != 1)
      throw InvalidArgument("Invalid OTLP IPv6 literal");
  }
  if (timeout < std::chrono::milliseconds(1) ||
      timeout > std::chrono::milliseconds(30000))
    throw InvalidArgument("OTLP request timeout must be in [1, 30000] ms");
  return std::make_unique<Sender>(std::move(endpoint), timeout);
}
}  // namespace jyppx::liuding::experiment::detail
