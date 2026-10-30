// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/telemetry_http/otlp_http.hpp>

#include <array>
#include <chrono>
#include <gtest/gtest.h>
#include <stop_token>

namespace {
namespace forge = jyppx::liuding;
namespace ex = forge::experiment;
ex::OtlpHttpOptions Options() {
  ex::OtlpHttpOptions result;
  result.endpoint = {.host = "127.0.0.1", .port = 1U, .tls = false};
  result.mapping.resource_instance_id = "http.unit-test";
  return result;
}

TEST(NativeOtlpHttp, RejectsUrlCredentialAndCleartextNonLoopbackAuthorities) {
  for (const auto* host : {"", "http://127.0.0.1", "user@127.0.0.1",
                           "127.0.0.1/path", "127.0.0.1?key=1", "[::1]",
                           "fe80::1%eth0", "127.0.0.1\r\nInjected: value",
                           "localhost", "127.0.0.2", "collector.example"}) {
    auto options = Options();
    options.endpoint.host = host;
    EXPECT_THROW(ex::OtlpHttpTelemetrySink{options}, forge::InvalidArgument);
  }
}

TEST(NativeOtlpHttp, RequiresExplicitBoundedEndpointAndTimeout) {
  auto options = Options();
  options.endpoint.port = 0U;
  EXPECT_THROW(ex::OtlpHttpTelemetrySink{options}, forge::InvalidArgument);
  options = Options();
  options.request_timeout = std::chrono::milliseconds(0);
  EXPECT_THROW(ex::OtlpHttpTelemetrySink{options}, forge::InvalidArgument);
  options.request_timeout = std::chrono::milliseconds(30001);
  EXPECT_THROW(ex::OtlpHttpTelemetrySink{options}, forge::InvalidArgument);
}

TEST(NativeOtlpHttp, ValidatesIpLiteralsBeforeAnyConnection) {
  auto options = Options();
  options.endpoint.host = "::1";
  EXPECT_NO_THROW(ex::OtlpHttpTelemetrySink{options});
  options.endpoint.tls = true;
  options.endpoint.host = ":::";
  EXPECT_THROW(ex::OtlpHttpTelemetrySink{options}, forge::InvalidArgument);
  options.endpoint.host = "2001:db8::1";
  EXPECT_NO_THROW(ex::OtlpHttpTelemetrySink{options});
}

TEST(NativeOtlpHttp, InheritsMappingRoleAndIndexBoundsWithoutConnecting) {
  auto options = Options();
  options.mapping.resource_instance_id.clear();
  EXPECT_THROW(ex::OtlpHttpTelemetrySink{options}, forge::InvalidArgument);
  options = Options();
  options.descriptor.role = forge::TelemetrySinkRole::evidence;
  EXPECT_THROW(ex::OtlpHttpTelemetrySink{options}, forge::InvalidArgument);
  options = Options();
  ++options.max_seen;
  EXPECT_THROW(ex::OtlpHttpTelemetrySink{options}, forge::InvalidArgument);
  options = Options();
  ++options.max_seen_bytes;
  EXPECT_THROW(ex::OtlpHttpTelemetrySink{options}, forge::InvalidArgument);
}

TEST(NativeOtlpHttp, EmptyBatchFlushCloseAndDestructionDoNotConnect) {
  ex::OtlpHttpTelemetrySink sink(Options());
  EXPECT_EQ(sink.descriptor().role, forge::TelemetrySinkRole::observer);
  EXPECT_TRUE(sink.emit({}).dispositions.empty());
  EXPECT_EQ(sink.flush().exported_events, 0U);
  EXPECT_EQ(sink.close().exported_events, 0U);
  EXPECT_EQ(sink.close().exported_events, 0U);
  EXPECT_THROW(sink.emit({}), forge::NotReady);
}

TEST(NativeOtlpHttp, RejectsAlreadyCancelledAndExpiredOperations) {
  ex::OtlpHttpTelemetrySink sink(Options());
  std::stop_source stop;
  stop.request_stop();
  EXPECT_THROW(sink.emit({}, stop.get_token()), forge::Cancelled);
  EXPECT_THROW(sink.emit({}, {},
                         std::chrono::steady_clock::now() -
                             std::chrono::milliseconds(1)),
               forge::DeadlineExceeded);
  EXPECT_EQ(sink.flush().exported_events, 0U);
}
}  // namespace
