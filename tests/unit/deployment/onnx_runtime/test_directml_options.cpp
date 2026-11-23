// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "jyppx/liuding/deployment/onnx_runtime/run_admission.hpp"

#include <jyppx/liuding/deployment/onnx_runtime/session.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <future>
#include <gtest/gtest.h>
#include <mutex>
#include <stop_token>
#include <vector>

namespace {
using namespace jyppx::liuding;
namespace runtime = jyppx::liuding::onnx_runtime;
using namespace std::chrono_literals;

TEST(OnnxRuntimeDirectMlOptions, TypedSelectionAndInvalidRawOptions) {
  const auto defaults = runtime::directml_execution_provider({});
  EXPECT_EQ(defaults.name, "DmlExecutionProvider");
  EXPECT_EQ(defaults.options.size(), 1U);
  EXPECT_EQ(defaults.options.at("device_id"), "0");
  EXPECT_EQ(runtime::directml_execution_provider({.device_id = 7})
                .options.at("device_id"),
            "7");
  EXPECT_THROW(runtime::directml_execution_provider({.device_id = -1}),
               InvalidArgument);
  // Native MIGraphX and ORT ROCm providers are separate integrations. The
  // default ORT adapter must not advertise either one without a wired API.
  EXPECT_FALSE(runtime::provider_append_supported("ROCMExecutionProvider"));
  EXPECT_FALSE(
      runtime::provider_append_supported("MIGraphXExecutionProvider"));
  // Option validation must precede provider discovery and model parsing.
  const std::array bytes{std::byte{0}};
  const ArtifactDigest digest{.hex = sha256_hex(bytes)};
  runtime::SessionOptions options;
  options.provider = defaults;
  for (const auto* value : {"", "-1", "+0", " 0", "0x0", "1.0", "2147483648"}) {
    options.provider.options = {{"device_id", value}};
    EXPECT_THROW(runtime::Session(bytes, digest, options), InvalidArgument)
        << value;
  }
  for (const auto* key :
       {"gpu_mem_limit", "device_filter", "user_compute_stream"}) {
    options.provider.options = {{key, "0"}};
    EXPECT_THROW(runtime::Session(bytes, digest, options), InvalidArgument)
        << key;
  }
#ifdef JYPPX_TEST_ORT_DIRECTML
  EXPECT_TRUE(runtime::provider_append_supported(defaults.name));
#else
  EXPECT_FALSE(runtime::provider_append_supported(defaults.name));
  options.provider = defaults;
  EXPECT_THROW(runtime::Session(bytes, digest, options), CompatibilityError);
#endif
}

TEST(OnnxRuntimeRunAdmission, WaitingCancellationPreservesActiveOwner) {
  std::timed_mutex mutex;
  std::unique_lock active(mutex);
  std::stop_source stop;
  std::promise<void> entered;
  bool announced = false;
  auto waiting = std::async(std::launch::async, [&] {
    try {
      auto slot = runtime::detail::AcquireRunAdmission(mutex, true, [&] {
        if (!announced) {
          entered.set_value();
          announced = true;
        }
        if (stop.stop_requested()) throw Cancelled("waiting caller stopped");
      });
      return false;
    } catch (const Cancelled&) {
      return true;
    }
  });
  entered.get_future().wait();
  stop.request_stop();
  const auto status = waiting.wait_for(2s);
  EXPECT_EQ(status, std::future_status::ready);
  // Release even on a regression so a failed test cannot hang in
  // future::~future.
  EXPECT_TRUE(active.owns_lock());
  active.unlock();
  EXPECT_TRUE(waiting.get());
  ASSERT_TRUE(mutex.try_lock());
  mutex.unlock();
}

TEST(OnnxRuntimeRunAdmission, WaitingDeadlineAndIndependentSession) {
  std::timed_mutex mutex, independent;
  std::unique_lock active(mutex);
  auto other = runtime::detail::AcquireRunAdmission(independent, true, [] {});
  EXPECT_TRUE(other.owns_lock());
  const auto deadline = std::chrono::steady_clock::now() + 30ms;
  auto waiting = std::async(std::launch::async, [&] {
    try {
      auto slot = runtime::detail::AcquireRunAdmission(mutex, true, [&] {
        if (std::chrono::steady_clock::now() >= deadline)
          throw DeadlineExceeded("waiting caller deadline");
      });
      return false;
    } catch (const DeadlineExceeded&) {
      return true;
    }
  });
  EXPECT_EQ(waiting.wait_for(2s), std::future_status::ready);
  EXPECT_TRUE(active.owns_lock());
  active.unlock();
  EXPECT_TRUE(waiting.get());
}

TEST(OnnxRuntimeRunAdmission, SerializesAndReleasesOnExceptions) {
  std::timed_mutex mutex;
  std::atomic<unsigned> in_flight = 0, overlaps = 0, completed = 0;
  std::vector<std::future<void>> callers;
  for (unsigned i = 0; i < 8; ++i) {
    callers.push_back(std::async(std::launch::async, [&] {
      for (unsigned run = 0; run < 32; ++run) {
        auto slot = runtime::detail::AcquireRunAdmission(mutex, true, [] {});
        if (in_flight.fetch_add(1) != 0) ++overlaps;
        ++completed;
        --in_flight;
      }
    }));
  }
  for (auto& caller : callers) caller.get();
  EXPECT_EQ(completed.load(), 256U);
  EXPECT_EQ(overlaps.load(), 0U);
  unsigned checks = 0;
  EXPECT_THROW(runtime::detail::AcquireRunAdmission(
                   mutex, true,
                   [&] {
                     if (++checks == 2)
                       throw Cancelled("after lock acquisition");
                   }),
               Cancelled);
  auto next = runtime::detail::AcquireRunAdmission(mutex, true, [] {});
  EXPECT_TRUE(next.owns_lock());
  auto concurrent = runtime::detail::AcquireRunAdmission(mutex, false, [] {});
  EXPECT_FALSE(concurrent.owns_lock());
}
}  // namespace
