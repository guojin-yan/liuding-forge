// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "http_transport.hpp"

#include <jyppx/liuding/foundation/error.hpp>

#include <algorithm>
#include <utility>

namespace jyppx::liuding::experiment {
namespace detail {
namespace {
std::string_view trim_http_ows(std::string_view value) noexcept {
  while (!value.empty() && (value.front() == ' ' || value.front() == '\t'))
    value.remove_prefix(1U);
  while (!value.empty() && (value.back() == ' ' || value.back() == '\t'))
    value.remove_suffix(1U);
  return value;
}

bool http_token_character(unsigned char value) noexcept {
  return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'z') ||
         (value >= 'A' && value <= 'Z') || value == 96U ||
         std::string_view("!#$%&'*+-.^_|~").find(static_cast<char>(value)) !=
             std::string_view::npos;
}
}  // namespace

bool http_ascii_case_equal(std::string_view left,
                           std::string_view right) noexcept {
  if (left.size() != right.size()) return false;
  const auto lower = [](unsigned char value) {
    return value >= 'A' && value <= 'Z' ? value + ('a' - 'A') : value;
  };
  for (std::size_t i = 0U; i < left.size(); ++i)
    if (lower(static_cast<unsigned char>(left[i])) !=
        lower(static_cast<unsigned char>(right[i])))
      return false;
  return true;
}

void validate_http_response_content_type(std::string_view value) {
  value = trim_http_ows(value);
  auto position = value.find(';');
  if (!http_ascii_case_equal(trim_http_ows(value.substr(0U, position)),
                             "application/x-protobuf"))
    throw IoError("OTLP collector response requires application/x-protobuf");
  if (position == std::string_view::npos) return;
  const auto invalid = [] {
    throw IoError("OTLP collector response has invalid media-type parameters");
  };
  const auto skip_ows = [&] {
    while (position < value.size() &&
           (value[position] == ' ' || value[position] == '\t'))
      ++position;
  };
  while (position < value.size()) {
    if (value[position++] != ';') invalid();
    skip_ows();
    const auto name = position;
    while (position < value.size() &&
           http_token_character(static_cast<unsigned char>(value[position])))
      ++position;
    if (position == name) invalid();
    skip_ows();
    if (position == value.size() || value[position++] != '=') invalid();
    skip_ows();
    if (position == value.size()) invalid();
    if (value[position] == '"') {
      ++position;
      bool closed = false;
      while (position < value.size()) {
        auto character = static_cast<unsigned char>(value[position++]);
        if (character == '"') {
          closed = true;
          break;
        }
        if (character == '\\') {
          if (position == value.size()) invalid();
          character = static_cast<unsigned char>(value[position++]);
          if (!(character == '\t' || character == ' ' ||
                (character >= 33U && character <= 126U) || character >= 128U))
            invalid();
        } else if (!(character == '\t' || character == ' ' ||
                     character == '!' ||
                     (character >= 35U && character <= 91U) ||
                     (character >= 93U && character <= 126U) ||
                     character >= 128U)) {
          invalid();
        }
      }
      if (!closed) invalid();
    } else {
      const auto start = position;
      while (position < value.size() &&
             http_token_character(static_cast<unsigned char>(value[position])))
        ++position;
      if (position == start) invalid();
    }
    skip_ows();
    if (position < value.size() && value[position] != ';') invalid();
  }
}

void validate_http_response_content_encoding(std::string_view value) {
  // HTTP libraries retain transfer-framing ownership. This policy only
  // accepts an untransformed protobuf representation, including identity.
  for (;;) {
    const auto comma = value.find(',');
    if (!http_ascii_case_equal(trim_http_ows(value.substr(0U, comma)),
                               "identity"))
      throw IoError("OTLP collector response has unsupported content encoding");
    if (comma == std::string_view::npos) return;
    value.remove_prefix(comma + 1U);
  }
}

void validate_http_endpoint(const OtlpHttpEndpoint& endpoint) {
  if (endpoint.host.empty() || endpoint.host.size() > 253U ||
      endpoint.port == 0U)
    throw InvalidArgument("OTLP requires an explicit bounded host and port");
  // An authority is structured data, never a user-supplied URL. Restrict its
  // alphabet before handing DNS/IP interpretation to the mature HTTP stack.
  // Bracket-free IPv6 is rebracketed only by the curl URL formatter below.
  const bool ipv6 = endpoint.host.find(':') != std::string::npos;
  for (const auto c : endpoint.host) {
    const bool digit = c >= '0' && c <= '9';
    const bool lower = c >= 'a' && c <= 'z';
    const bool upper = c >= 'A' && c <= 'Z';
    const bool hex = digit || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
    if (ipv6 ? !(hex || c == ':' || c == '.')
             : !(digit || lower || upper || c == '-' || c == '.'))
      throw InvalidArgument("OTLP host is not a plain ASCII DNS/IP authority");
  }
  if (!ipv6) {
    std::size_t start = 0U;
    while (start < endpoint.host.size()) {
      const auto dot = endpoint.host.find('.', start);
      const auto end = dot == std::string::npos ? endpoint.host.size() : dot;
      if (end == start || end - start > 63U || endpoint.host[start] == '-' ||
          endpoint.host[end - 1U] == '-')
        throw InvalidArgument("OTLP host contains an invalid DNS label");
      if (dot == std::string::npos) break;
      start = dot + 1U;
    }
  }
  if (!endpoint.tls && endpoint.host != "127.0.0.1" && endpoint.host != "::1")
    throw InvalidArgument("Cleartext OTLP is limited to literal loopback");
}

void validate_http_request(const PreparedOtlpRequest& request,
                           std::size_t maximum_reply_bytes) {
  static_cast<void>(OtlpProjectionEncoder::signal_path(request.signal));
  if (request.body.size() > 16U * 1024U * 1024U || maximum_reply_bytes == 0U ||
      maximum_reply_bytes > 1024U * 1024U)
    throw ResourceExhausted("OTLP HTTP encoded request/reply bound exceeded");
}

std::chrono::steady_clock::time_point http_request_deadline(
    std::stop_token stop, TelemetryDeadline caller_deadline,
    std::chrono::milliseconds timeout) {
  check_telemetry_operation(stop, caller_deadline);
  const auto local = std::chrono::steady_clock::now() + timeout;
  return caller_deadline ? std::min(local, *caller_deadline) : local;
}

int http_remaining_milliseconds(
    std::stop_token stop, std::chrono::steady_clock::time_point deadline) {
  check_telemetry_operation(stop, deadline);
  const auto remaining = std::chrono::ceil<std::chrono::milliseconds>(
      deadline - std::chrono::steady_clock::now());
  // Zero means infinite to some native APIs. Never pass it after rounding.
  if (remaining.count() <= 0) throw DeadlineExceeded("OTLP deadline exceeded");
  return static_cast<int>(std::min<std::int64_t>(30000, remaining.count()));
}

OtlpPreflightOptions preflight_options(OtlpHttpOptions options) {
  return {.mapping = std::move(options.mapping),
          .descriptor = std::move(options.descriptor),
          .max_seen = options.max_seen,
          .max_seen_bytes = options.max_seen_bytes,
          .max_batch_prepared_bytes = options.max_batch_prepared_bytes};
}
}  // namespace detail

class OtlpHttpTelemetrySink::Impl final {
 public:
  explicit Impl(OtlpHttpOptions options)
      : descriptor(options.descriptor),
        sender(detail::make_http_sender(options.endpoint,
                                        options.request_timeout)),
        delivery(detail::preflight_options(std::move(options)), *sender) {}
  TelemetrySinkDescriptor descriptor;
  std::unique_ptr<detail::OtlpRequestSender> sender;
  detail::OtlpDeliveryState delivery;
};

OtlpHttpTelemetrySink::OtlpHttpTelemetrySink(OtlpHttpOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}
OtlpHttpTelemetrySink::~OtlpHttpTelemetrySink() = default;
const TelemetrySinkDescriptor& OtlpHttpTelemetrySink::descriptor()
    const noexcept {
  return impl_->descriptor;
}
TelemetryEmitResult OtlpHttpTelemetrySink::emit(
    std::span<const TelemetryEnvelope> envelopes, std::stop_token stop,
    TelemetryDeadline deadline) {
  return impl_->delivery.emit(envelopes, stop, deadline);
}
TelemetryFlushResult OtlpHttpTelemetrySink::flush(std::stop_token stop,
                                                  TelemetryDeadline deadline) {
  return impl_->delivery.flush(stop, deadline);
}
TelemetryFlushResult OtlpHttpTelemetrySink::close(std::stop_token stop,
                                                  TelemetryDeadline deadline) {
  return impl_->delivery.close(stop, deadline);
}
}  // namespace jyppx::liuding::experiment
