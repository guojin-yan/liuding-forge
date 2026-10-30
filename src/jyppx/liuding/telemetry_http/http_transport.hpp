// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_PRIVATE_HTTP_TRANSPORT_HPP
#define JYPPX_LIUDING_PRIVATE_HTTP_TRANSPORT_HPP

#include "../telemetry_native/detail/otlp_preflight.hpp"

#include <jyppx/liuding/telemetry_http/otlp_http.hpp>

#include <string_view>

namespace jyppx::liuding::experiment::detail {

void validate_http_endpoint(const OtlpHttpEndpoint& endpoint);
void validate_http_request(const PreparedOtlpRequest& request,
                           std::size_t maximum_reply_bytes);
bool http_ascii_case_equal(std::string_view left,
                           std::string_view right) noexcept;
void validate_http_response_content_type(std::string_view value);
void validate_http_response_content_encoding(std::string_view value);
std::chrono::steady_clock::time_point http_request_deadline(
    std::stop_token stop, TelemetryDeadline caller_deadline,
    std::chrono::milliseconds timeout);
int http_remaining_milliseconds(std::stop_token stop,
                                std::chrono::steady_clock::time_point deadline);
std::unique_ptr<OtlpRequestSender> make_http_sender(
    OtlpHttpEndpoint endpoint, std::chrono::milliseconds timeout);

}  // namespace jyppx::liuding::experiment::detail
#endif
