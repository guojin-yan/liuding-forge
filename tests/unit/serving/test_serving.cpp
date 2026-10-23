// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
#include "../../../examples/serving/linear_bundle.hpp"

#include <torch/cuda.h>

#include <ATen/autocast_mode.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <future>
#include <gtest/gtest.h>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <stop_token>
#include <thread>
namespace {
using namespace jyppx::liuding;
using namespace jyppx::liuding::serving;
namespace example = jyppx::liuding::examples::serving;
using namespace std::chrono_literals;
constexpr auto Cpu = c10::kCPU;
class ServingTest : public ::testing::Test {
 protected:
  example::BundleFixture fixture;
  void SetUp() override { fixture.Publish(); }
  std::shared_ptr<Predictor> Load(example::Hook hook = {},
                                  const LoadOptions& options = {}) {
    return fixture.Load(std::move(hook), options);
  }
  PredictRequest Request(std::string id = "request-1", float value = 1.0F) {
    return example::Request(Device(Cpu), std::move(id), value);
  }
};
TEST_F(ServingTest, RealVerifiedStateWarmupAndIndependentNumericResponses) {
  auto predictor = Load();
  auto request = Request();
  auto input = request.input.positional.front().toTensor();
  auto response = predictor->predict(request);
  EXPECT_TRUE(torch::equal(response.output.value.toTensor(),
                           torch::tensor({{2.0F, -3.0F}})));
  EXPECT_FALSE(response.output.value.toTensor().requires_grad());
  EXPECT_NE(response.output.value.toTensor().data_ptr(), input.data_ptr());
  EXPECT_EQ(response.context.request_id, request.context.request_id);
  EXPECT_EQ(predictor->info().accepted, 1U);
  EXPECT_EQ(predictor->info().completed, 1U);
  EXPECT_EQ(predictor->info().active, 0U);
  EXPECT_TRUE(predictor->close());
  EXPECT_TRUE(predictor->close());
  EXPECT_EQ(predictor->info().health, HealthStatus::closed);
  EXPECT_TRUE(torch::equal(response.output.value.toTensor(),
                           torch::tensor({{2.0F, -3.0F}})));
  EXPECT_THROW(predictor->predict(request), NotReady);
}
TEST_F(ServingTest,
       InputShapeDtypeSizeFiniteRangeAndCorrelationFailBeforeForward) {
  auto calls = std::make_shared<std::atomic<std::size_t>>(0);
  LoadOptions options;
  options.request_limits.max_tensor_bytes = 16;
  options.request_limits.minimum = -5;
  options.request_limits.maximum = 5;
  auto predictor = Load([calls](const Tensor&) { ++*calls; }, options);
  calls->store(0);
  auto bad = Request();
  bad.context.model_version = "wrong";
  EXPECT_THROW(predictor->predict(bad), InvalidArgument);
  bad = Request();
  bad.context.request_id = "bad\nid";
  EXPECT_THROW(predictor->predict(bad), InvalidArgument);
  bad = Request();
  bad.input.positional.front() = torch::ones({1, 3});
  EXPECT_THROW(predictor->predict(bad), InvalidArgument);
  bad = Request();
  bad.input.positional.front() = torch::ones({1, 2}, torch::kInt64);
  EXPECT_THROW(predictor->predict(bad), InvalidArgument);
  bad = Request();
  bad.input.positional.front() = torch::ones({3, 2});
  EXPECT_THROW(predictor->predict(bad), ResourceExhausted);
  bad = Request();
  bad.input.positional.front() =
      torch::full({1, 2}, std::numeric_limits<float>::quiet_NaN());
  EXPECT_THROW(predictor->predict(bad), NumericalError);
  bad = Request("range", 6);
  EXPECT_THROW(predictor->predict(bad), InvalidArgument);
  EXPECT_EQ(calls->load(), 0U);
  EXPECT_EQ(predictor->info().health, HealthStatus::ready);
  EXPECT_EQ(predictor->info().active, 0U);
  EXPECT_EQ(predictor->info().failed, 7U);
  EXPECT_NO_THROW(predictor->predict(Request()));
}
TEST_F(ServingTest, ExpiredAndCancelledCallsDoNotEnterOrDegradeModel) {
  auto predictor = Load();
  auto request = Request();
  request.deadline = std::chrono::steady_clock::now() - 1ms;
  EXPECT_THROW(predictor->predict(request), DeadlineExceeded);
  std::stop_source stop;
  stop.request_stop();
  request.deadline = {};
  EXPECT_THROW(predictor->predict(request, stop.get_token()), Cancelled);
  EXPECT_EQ(predictor->info().accepted, 0U);
  EXPECT_EQ(predictor->info().health, HealthStatus::ready);
}
TEST_F(ServingTest, ResourceTraversalAndExactDeviceLimits) {
  auto runtime = fixture.descriptor;
  RequestLimits limits;
  limits.max_nodes = 2;
  const nn::Invocation strings{.positional = {c10::ivalue::Tuple::create(
                                   {std::string("a"), std::string("b")})},
                               .keyword = {}};
  EXPECT_THROW(validate_request_resources(strings, limits, runtime),
               ResourceExhausted);
  limits = {};
  limits.max_string_bytes = 2;
  EXPECT_THROW(
      validate_request_resources(
          {.positional = {std::string("abc")}, .keyword = {}}, limits, runtime),
      ResourceExhausted);
  limits = {};
  limits.minimum = -1;
  limits.maximum = 1;
  EXPECT_THROW(
      validate_request_resources(
          {.positional = {std::int64_t(2)}, .keyword = {}}, limits, runtime),
      InvalidArgument);
  limits = {};
  limits.max_rank = 1;
  EXPECT_THROW(validate_request_resources(Request().input, limits, runtime),
               ResourceExhausted);
  limits = {};
  limits.max_dimension = 1;
  EXPECT_THROW(validate_request_resources(Request().input, limits, runtime),
               ResourceExhausted);
  limits = {};
  limits.max_elements = 1;
  EXPECT_THROW(validate_request_resources(Request().input, limits, runtime),
               ResourceExhausted);
  runtime.dtypes = {torch::kInt64};
  EXPECT_THROW(validate_request_resources(Request().input, {}, runtime),
               InvalidArgument);
}
TEST_F(ServingTest,
       IntegerRangesKeepScalarAndTensorPrecisionAtDoubleBoundaries) {
  auto runtime = fixture.descriptor;
  runtime.dtypes = {torch::kInt64};
  const auto check = [&](const char* name, std::optional<double> minimum,
                         std::optional<double> maximum,
                         std::initializer_list<std::int64_t> accepted,
                         std::initializer_list<std::int64_t> rejected) {
    SCOPED_TRACE(name);
    RequestLimits limits;
    limits.minimum = minimum;
    limits.maximum = maximum;
    const auto value = [&](std::int64_t number, bool allowed) {
      SCOPED_TRACE(number);
      for (const bool as_tensor : {false, true}) {
        SCOPED_TRACE(as_tensor ? "int64 tensor" : "int64 scalar");
        nn::Invocation input;
        if (as_tensor)
          input.positional.emplace_back(torch::tensor({number}, torch::kInt64));
        else
          input.positional.emplace_back(number);
        if (allowed) {
          EXPECT_NO_THROW(validate_request_resources(input, limits, runtime));
        } else {
          EXPECT_THROW(validate_request_resources(input, limits, runtime),
                       InvalidArgument);
        }
      }
    };
    for (const auto number : accepted) value(number, true);
    for (const auto number : rejected) value(number, false);
  };
  constexpr std::int64_t two53_integer = std::int64_t{1} << 53;
  const double two53 = std::ldexp(1.0, 53);
  const double two63 = std::ldexp(1.0, 63);
  const auto lowest = std::numeric_limits<std::int64_t>::min();
  const auto highest = std::numeric_limits<std::int64_t>::max();
  check(
      "exact 2^53 limits", -two53, two53,
      {-two53_integer, -two53_integer + 1, 0, two53_integer - 1, two53_integer},
      {-two53_integer - 1, two53_integer + 1});
  check("upper 2^63 accepts every int64", {}, two63,
        {lowest, lowest + 1, 0, highest - 1, highest}, {});
  check("lower 2^63 accepts no int64", two63, {}, {}, {lowest, 0, highest});
  check("lower -2^63 accepts every int64", -two63, {},
        {lowest, lowest + 1, 0, highest - 1, highest}, {});
  check("upper -2^63 accepts only INT64_MIN", {}, -two63, {lowest},
        {lowest + 1, 0, highest});
  const double below_lowest =
      std::nextafter(-two63, -std::numeric_limits<double>::infinity());
  check("lower below INT64_MIN", below_lowest, {}, {lowest, 0, highest}, {});
  check("upper below INT64_MIN", {}, below_lowest, {}, {lowest, 0, highest});
  const double below_two63 = std::nextafter(two63, 0.0);
  const auto highest_boundary = static_cast<std::int64_t>(below_two63);
  check("upper last double below 2^63", {}, below_two63,
        {lowest, 0, highest_boundary}, {highest_boundary + 1, highest});
  const double above_lowest = std::nextafter(-two63, 0.0);
  const auto lowest_boundary = static_cast<std::int64_t>(above_lowest);
  check("lower next double above -2^63", above_lowest, {},
        {lowest_boundary, 0, highest}, {lowest, lowest_boundary - 1});
  check("fractional positive lower bound", 0.5, {}, {1, highest},
        {lowest, -1, 0});
  check("fractional positive upper bound", {}, 0.5, {lowest, -1, 0},
        {1, highest});
  check("fractional negative lower bound", -0.5, {}, {0, 1, highest},
        {lowest, -1});
  check("fractional negative upper bound", {}, -0.5, {lowest, -1},
        {0, 1, highest});
}
TEST_F(ServingTest, ActualBundleHashTrustAndExactRuntimeAreEnforced) {
  NativeStateBackend backend(fixture.descriptor, example::Factory());
  const std::array<WarmupCase, 1> warmup{example::KnownVector(Device(Cpu))};
  interop::ModelBundleVerifyOptions trusted;
  trusted.require_trusted_signature = true;
  EXPECT_THROW(backend.load(fixture.directory, trusted, warmup),
               CompatibilityError);
  auto mismatch = fixture.descriptor;
  mismatch.backend_version = "2";
  EXPECT_THROW(NativeStateBackend(mismatch, example::Factory())
                   .load(fixture.directory, {}, warmup),
               CompatibilityError);
  mismatch = fixture.descriptor;
  mismatch.environment_lock = example::Digest(example::Bytes("other"));
  EXPECT_THROW(NativeStateBackend(mismatch, example::Factory())
                   .load(fixture.directory, {}, warmup),
               CompatibilityError);
  mismatch = fixture.descriptor;
  mismatch.dependencies.front().version = "not-linked";
  EXPECT_THROW(NativeStateBackend(mismatch, example::Factory()),
               CompatibilityError);
  mismatch = fixture.descriptor;
  mismatch.dtypes.push_back(torch::kFloat64);
  EXPECT_THROW(NativeStateBackend(mismatch, example::Factory())
                   .load(fixture.directory, {}, warmup),
               CompatibilityError);
  LoadOptions tiny;
  tiny.max_snapshot_bytes = 1;
  EXPECT_THROW(backend.load(fixture.directory, {}, warmup, tiny),
               ResourceExhausted);
  const auto file = fixture.manifest.files.front();
  auto source = fixture.directory / file.relative_path;
  std::ofstream overwrite(source, std::ios::binary | std::ios::app);
  overwrite.put('x');
  overwrite.close();
  EXPECT_THROW(backend.load(fixture.directory, {}, warmup), Error);
}
TEST_F(ServingTest, OwnerPipelineDescriptorKnownVectorsAndStrictStateMapping) {
  const std::array<WarmupCase, 1> warmup{example::KnownVector(Device(Cpu))};
  auto factory = example::Factory();
  NativeStateBackend wrong_binding(
      fixture.descriptor, [factory](const auto& b, const auto& d) {
        auto native = factory(b, d);
        native.pipeline.input.digest = example::Digest(example::Bytes("wrong"));
        return native;
      });
  EXPECT_THROW(wrong_binding.load(fixture.directory, {}, warmup),
               CompatibilityError);
  NativeStateBackend missing(fixture.descriptor,
                             [factory](const auto& b, const auto& d) {
                               auto native = factory(b, d);
                               native.adapter->state().module().register_buffer(
                                   "unexpected", torch::zeros({1}));
                               return native;
                             });
  EXPECT_THROW(missing.load(fixture.directory, {}, warmup), CompatibilityError);
  auto wrong = warmup;
  wrong.front().expected = torch::zeros({1, 2});
  NativeStateBackend backend(fixture.descriptor, factory);
  EXPECT_THROW(backend.load(fixture.directory, {}, wrong), NumericalError);
  EXPECT_THROW(
      backend.load(fixture.directory, {}, std::span<const WarmupCase>{}),
      InvalidArgument);
  LoadOptions stop;
  std::stop_source source;
  source.request_stop();
  stop.stop = source.get_token();
  EXPECT_THROW(backend.load(fixture.directory, {}, warmup, stop), Cancelled);
  stop = {};
  stop.deadline = std::chrono::steady_clock::now() - 1ms;
  EXPECT_THROW(backend.load(fixture.directory, {}, warmup, stop),
               DeadlineExceeded);
}
struct Gate {
  std::mutex mutex;
  std::condition_variable cv;
  bool entered = false, released = false;
  void Wait(const Tensor&) {
    std::unique_lock lock(mutex);
    entered = true;
    cv.notify_all();
    if (!cv.wait_for(lock, 5s, [&] { return released; }))
      throw std::runtime_error("Serving test gate was not released");
  }
  bool Entered() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [&] { return entered; });
  }
  void Release() {
    std::lock_guard lock(mutex);
    released = true;
    cv.notify_all();
  }
  ~Gate() { Release(); }
};
TEST_F(ServingTest, ConcurrentLeaseBoundTimedDrainAndNoPrematureUnload) {
  auto gate = std::make_shared<Gate>();
  auto enabled = std::make_shared<std::atomic<bool>>(false);
  LoadOptions options;
  options.max_active_requests = 1;
  auto predictor = Load(
      [gate, enabled](const Tensor& x) {
        if (enabled->load()) gate->Wait(x);
      },
      options);
  enabled->store(true);
  auto pending = std::async(std::launch::async,
                            [&] { return predictor->predict(Request()); });
  EXPECT_TRUE(gate->Entered());
  EXPECT_THROW(predictor->predict(Request("overflow")), ResourceExhausted);
  EXPECT_FALSE(predictor->close(std::chrono::steady_clock::now() + 2ms));
  EXPECT_EQ(predictor->info().health, HealthStatus::draining);
  EXPECT_EQ(predictor->info().active, 1U);
  EXPECT_THROW(predictor->predict(Request("draining")), NotReady);
  gate->Release();
  EXPECT_TRUE(torch::equal(pending.get().output.value.toTensor(),
                           torch::tensor({{2.0F, -3.0F}})));
  EXPECT_TRUE(predictor->close());
}
TEST_F(ServingTest,
       PostForwardFailureDegradesAndObservationsAreRedactedAndIsolated) {
  auto enabled = std::make_shared<std::atomic<bool>>(false);
  std::vector<TelemetryAttributes> seen;
  LoadOptions options;
  options.observe = [&](TelemetryLog log, TelemetryCorrelation,
                        TelemetryAttributes attrs) {
    EXPECT_TRUE(log.body == "serving.predict.completed" ||
                log.body == "serving.predict.failed");
    seen.push_back(std::move(attrs));
    throw std::runtime_error("observation failure");
  };
  auto predictor = Load(
      [enabled](const Tensor&) {
        if (enabled->load()) throw BackendError("private-token");
      },
      options);
  EXPECT_NO_THROW(predictor->predict(Request()));
  EXPECT_GT(predictor->info().observation_failures, 0U);
  enabled->store(true);
  EXPECT_THROW(predictor->predict(Request("failure")), BackendError);
  EXPECT_EQ(predictor->info().health, HealthStatus::degraded);
  EXPECT_THROW(predictor->predict(Request()), NotReady);
  EXPECT_FALSE(seen.empty());
  for (const auto& attrs : seen)
    for (const auto& [key, value] : attrs) {
      EXPECT_NE(key, "input");
      EXPECT_NE(key, "token");
      EXPECT_NE(value, "private-token");
    }
}
TEST_F(ServingTest, BatchingPreservesOrderOwnsSnapshotsAndSeparatesScopes) {
  auto sizes = std::make_shared<std::vector<std::int64_t>>();
  auto predictor =
      Load([sizes](const Tensor& x) { sizes->push_back(x.size(0)); });
  sizes->clear();
  BatchOptions options;
  options.max_batch_size = 2;
  options.max_wait = 20ms;
  DynamicBatcher batcher(predictor, options);
  auto first = Request("first", 1);
  auto second = Request("second", 2);
  auto one = batcher.submit(first);
  auto two = batcher.submit(second);
  first.input.positional.front().toTensor().fill_(100);
  auto x = one.get(), y = two.get();
  EXPECT_EQ(x.context.request_id, "first");
  EXPECT_EQ(y.context.request_id, "second");
  EXPECT_TRUE(
      torch::equal(x.output.value.toTensor(), torch::tensor({{2.0F, -3.0F}})));
  EXPECT_TRUE(
      torch::equal(y.output.value.toTensor(), torch::tensor({{4.0F, -6.0F}})));
  EXPECT_EQ(*sizes, (std::vector<std::int64_t>{2}));
  auto a = Request("a"), b = Request("b");
  b.context.project_id = "other-project";
  auto left = batcher.submit(a);
  auto right = batcher.submit(b);
  left.get();
  right.get();
  EXPECT_EQ(sizes->size(), 3U);
  EXPECT_EQ(sizes->at(1), 1);
  EXPECT_EQ(sizes->at(2), 1);
  EXPECT_TRUE(batcher.close());
  const auto stats = batcher.statistics();
  EXPECT_EQ(stats.completed, 4U);
  EXPECT_EQ(stats.resident_tensor_bytes, 0U);
  EXPECT_EQ(stats.active, 0U);
  EXPECT_TRUE(stats.stopped);
}
TEST_F(ServingTest, InBatchCancellationDoesNotCancelPeers) {
  auto gate = std::make_shared<Gate>();
  auto enabled = std::make_shared<std::atomic<bool>>(false);
  auto predictor = Load([gate, enabled](const Tensor& x) {
    if (enabled->load()) gate->Wait(x);
  });
  enabled->store(true);
  BatchOptions options;
  options.max_batch_size = 2;
  options.max_wait = 50ms;
  DynamicBatcher batcher(predictor, options);
  std::stop_source cancelled;
  auto first = batcher.submit(Request("first"), cancelled.get_token());
  auto second = batcher.submit(Request("second", 2));
  EXPECT_TRUE(gate->Entered());
  cancelled.request_stop();
  gate->Release();
  EXPECT_THROW(first.get(), Cancelled);
  EXPECT_TRUE(torch::equal(second.get().output.value.toTensor(),
                           torch::tensor({{4.0F, -6.0F}})));
  EXPECT_EQ(predictor->info().health, HealthStatus::ready);
  EXPECT_TRUE(batcher.close());
}
TEST_F(ServingTest,
       BatchingHonorsMergedElementDimensionAndByteLimitsWithoutRejectingRows) {
  const auto check = [&](const char* name, const RequestLimits& limits,
                         const std::vector<std::int64_t>& expected_batches) {
    SCOPED_TRACE(name);
    auto gate = std::make_shared<Gate>();
    auto enabled = std::make_shared<std::atomic<bool>>(false);
    auto batches = std::make_shared<std::vector<std::int64_t>>();
    LoadOptions load_options;
    load_options.request_limits = limits;
    auto predictor = Load(
        [gate, enabled, batches](const Tensor& x) {
          batches->push_back(x.size(0));
          if (enabled->load()) gate->Wait(x);
        },
        load_options);
    batches->clear();
    enabled->store(true);
    BatchOptions options;
    options.max_batch_size = 4;
    options.max_wait = 0ms;
    DynamicBatcher batcher(predictor, options);
    auto active = batcher.submit(Request("active"));
    EXPECT_TRUE(gate->Entered());
    // Queue all three peers behind active compute so coalescing is
    // deterministic.
    std::array<std::future<PredictResponse>, 3> queued;
    for (std::size_t i = 0; i < queued.size(); ++i)
      queued[i] = batcher.submit(
          Request("queued-" + std::to_string(i), static_cast<float>(i + 2)));
    gate->Release();
    EXPECT_NO_THROW(active.get());
    for (std::size_t i = 0; i < queued.size(); ++i) {
      EXPECT_NO_THROW({
        const auto response = queued[i].get();
        const auto value = static_cast<float>(i + 2);
        EXPECT_TRUE(torch::equal(response.output.value.toTensor(),
                                 torch::tensor({{2 * value, -3 * value}})));
      });
    }
    EXPECT_TRUE(batcher.close());
    EXPECT_EQ(*batches, expected_batches);
    EXPECT_EQ(batcher.statistics().completed, 4U);
    EXPECT_EQ(batcher.statistics().failed, 0U);
    EXPECT_EQ(batcher.statistics().resident_tensor_bytes, 0U);
    EXPECT_EQ(predictor->info().health, HealthStatus::ready);
  };
  RequestLimits elements;
  elements.max_elements = 2;
  check("elements", elements, {1, 1, 1, 1});
  RequestLimits dimensions;
  dimensions.max_dimension = 2;
  check("dimensions", dimensions, {1, 2, 1});
  RequestLimits bytes;
  bytes.max_tensor_bytes = 16;
  check("bytes", bytes, {1, 2, 1});
}
TEST_F(ServingTest, QueueByteCountDeadlineAndShutdownBounds) {
  auto gate = std::make_shared<Gate>();
  auto enabled = std::make_shared<std::atomic<bool>>(false);
  auto predictor = Load([gate, enabled](const Tensor& x) {
    if (enabled->load()) gate->Wait(x);
  });
  enabled->store(true);
  BatchOptions options;
  options.max_batch_size = 1;
  options.max_wait = 0ms;
  options.max_queue_size = 1;
  options.max_resident_tensor_bytes = 16;
  DynamicBatcher batcher(predictor, options);
  auto active = batcher.submit(Request("active"));
  EXPECT_TRUE(gate->Entered());
  auto queued = batcher.submit(Request("queued"));
  EXPECT_THROW(batcher.submit(Request("overflow")), ResourceExhausted);
  EXPECT_FALSE(batcher.close(std::chrono::steady_clock::now() + 2ms));
  EXPECT_THROW(batcher.submit(Request("closed")), NotReady);
  gate->Release();
  EXPECT_NO_THROW(active.get());
  EXPECT_THROW(queued.get(), Cancelled);
  EXPECT_TRUE(batcher.close());
  EXPECT_EQ(batcher.statistics().resident_tensor_bytes, 0U);
  DynamicBatcher expired(predictor, options);
  auto request = Request("expired");
  request.deadline = std::chrono::steady_clock::now() - 1ms;
  EXPECT_THROW(expired.submit(request), DeadlineExceeded);
  EXPECT_TRUE(expired.close());
}
TEST_F(ServingTest, OnlyExplicitConfirmedOomGetsFiniteSplitRetry) {
  auto seen = std::make_shared<std::atomic<std::size_t>>(0);
  auto predictor = Load([seen](const Tensor& x) {
    ++*seen;
    if (x.size(0) > 1) throw std::bad_alloc();
  });
  seen->store(0);
  BatchOptions options;
  options.max_batch_size = 4;
  options.max_wait = 100ms;
  options.max_oom_retries = 3;
  DynamicBatcher batcher(predictor, options);
  std::array<std::future<PredictResponse>, 4> futures;
  for (std::size_t i = 0; i < futures.size(); ++i)
    futures[i] = batcher.submit(Request("oom-" + std::to_string(i)));
  for (auto& f : futures) EXPECT_NO_THROW(f.get());
  EXPECT_EQ(seen->load(), 7U);
  EXPECT_EQ(batcher.statistics().oom_retries, 3U);
  EXPECT_EQ(predictor->info().health, HealthStatus::ready);
  EXPECT_TRUE(batcher.close());
  options.max_oom_retries = 0;
  DynamicBatcher no_retry(predictor, options);
  auto a = no_retry.submit(Request("a"));
  auto b = no_retry.submit(Request("b"));
  EXPECT_THROW(a.get(), RetryableOutOfMemory);
  EXPECT_THROW(b.get(), RetryableOutOfMemory);
  EXPECT_EQ(no_retry.statistics().oom_retries, 0U);
  EXPECT_TRUE(no_retry.close());
}
ProtocolAdapter Protocol() {
  return {.protocol_id = "example.fixed-f32",
          .schema_version = 1,
          .decode =
              [](std::span<const std::byte> bytes) {
                if (bytes.size() != 8)
                  throw SerializationError("private decode details");
                auto input = torch::empty({1, 2}, torch::kFloat32);
                std::memcpy(input.data_ptr(), bytes.data(), 8);
                return nn::Invocation{.positional = {input}, .keyword = {}};
              },
          .encode =
              [](const nn::InvocationResult& result) {
                const auto tensor = result.value.toTensor().contiguous();
                const auto bytes =
                    std::as_bytes(std::span(tensor.data_ptr<float>(), 2));
                return std::vector<std::byte>(bytes.begin(), bytes.end());
              },
          .authorize =
              [](const RequestContext& c) {
                return c.project_id == "project-example";
              }};
}
TEST_F(ServingTest, TransportAuthenticationScopeSizeMappingAndReadiness) {
  auto predictor = Load();
  InferenceServiceAdapter service(
      predictor, Protocol(), {.max_request_bytes = 8, .max_response_bytes = 8});
  const std::array<float, 2> values{1, 2};
  const auto bytes = std::as_bytes(std::span(values));
  TransportContext context{
      .correlation = Request().context, .authenticated = false, .deadline = {}};
  EXPECT_EQ(service.handle(bytes, context).status,
            ServiceStatus::unauthenticated);
  context.authenticated = true;
  context.correlation.project_id = "denied";
  EXPECT_EQ(service.handle(bytes, context).status,
            ServiceStatus::permission_denied);
  context.correlation.project_id = "project-example";
  auto reply = service.handle(bytes, context);
  EXPECT_EQ(reply.status, ServiceStatus::ok);
  std::array<float, 2> actual{};
  std::memcpy(actual.data(), reply.body.data(), reply.body.size());
  EXPECT_EQ(actual, (std::array<float, 2>{2, -6}));
  EXPECT_EQ(reply.request_id, context.correlation.request_id);
  EXPECT_EQ(service.handle(bytes.first(1), context).message,
            "invalid_argument");
  std::array<std::byte, 9> large{};
  EXPECT_EQ(service.handle(large, context).status,
            ServiceStatus::resource_exhausted);
  context.deadline = std::chrono::steady_clock::now() - 1ms;
  EXPECT_EQ(service.handle(bytes, context).status,
            ServiceStatus::deadline_exceeded);
  context.deadline = {};
  EXPECT_TRUE(service.health().ready);
  EXPECT_FALSE(service.health().streaming);
  EXPECT_TRUE(service.close());
  EXPECT_FALSE(service.health().ready);
  EXPECT_EQ(service.handle(bytes, context).status, ServiceStatus::unavailable);
}
TEST_F(ServingTest,
       TransportSanitizesBackendFailureAndChecksSchedulerIdentity) {
  auto enabled = std::make_shared<std::atomic<bool>>(false);
  auto predictor = Load([enabled](const Tensor&) {
    if (enabled->load()) throw BackendError("secret/token/artifact/path");
  });
  InferenceServiceAdapter service(predictor, Protocol());
  const std::array<float, 2> values{1, 2};
  TransportContext context{
      .correlation = Request().context, .authenticated = true, .deadline = {}};
  enabled->store(true);
  const auto reply = service.handle(std::as_bytes(std::span(values)), context);
  EXPECT_EQ(reply.status, ServiceStatus::internal);
  EXPECT_EQ(reply.message, "internal");
  EXPECT_TRUE(reply.body.empty());
  EXPECT_FALSE(service.health().ready);
  auto other = Load();
  auto batcher = std::make_shared<DynamicBatcher>(other);
  EXPECT_THROW(InferenceServiceAdapter(predictor, Protocol(), {}, batcher),
               InvalidArgument);
  EXPECT_TRUE(batcher->close());
}
TEST_F(ServingTest,
       NativeBoundarySupportsMultiInputTupleAndDictionarySnapshots) {
  example::BundleFixture multi;
  auto schema = example::Input(Device(Cpu));
  schema.positional.push_back(schema.positional.front());
  const auto output =
      ValueSchema::tuple({example::Output(Device(Cpu)),
                          ValueSchema::dictionary(
                              {{"prediction", example::Output(Device(Cpu))}})});
  const auto input_descriptor =
      canonical_value_schema(ValueSchema::tuple(schema.positional));
  const auto output_descriptor = canonical_value_schema(output);
  const auto replace = [&](std::string_view id, std::string_view content) {
    const auto source =
        std::find_if(multi.sources.begin(), multi.sources.end(),
                     [&](const auto& item) { return item.file_id == id; });
    example::Write(source->source_path, example::Bytes(content));
    auto file =
        std::find_if(multi.manifest.files.begin(), multi.manifest.files.end(),
                     [&](const auto& item) { return item.id == id; });
    file->artifact.digest = example::Digest(example::Bytes(content));
    file->artifact.size_bytes = content.size();
  };
  replace("input", input_descriptor);
  replace("output", output_descriptor);
  multi.Publish();
  NativeStateBackend backend(multi.descriptor, [schema, output,
                                                input_descriptor,
                                                output_descriptor](
                                                   const auto& bundle,
                                                   const auto& runtime) {
    const auto bind = [&](const auto& ref, std::string_view expected) {
      const auto& bytes = bundle.files.at(ref.file_id);
      if (bytes.size() != expected.size() ||
          !std::equal(bytes.begin(), bytes.end(),
                      example::Bytes(expected).begin()))
        throw CompatibilityError(
            "Structured fixture owner descriptor mismatch");
      return DescriptorBinding{.schema_id = ref.schema_id,
                               .schema_version = ref.schema_version,
                               .digest = example::Digest(bytes)};
    };
    auto module = std::make_shared<example::Linear>(runtime.device);
    auto adapter = std::make_shared<nn::ModelAdapter>(nn::adapt_module(
        module,
        [](example::Linear& model, const nn::Invocation& input) {
          const auto sum =
              input.positional[0].toTensor() + input.positional[1].toTensor();
          const auto value = model.forward(sum);
          c10::Dict<std::string, Tensor> dictionary;
          dictionary.insert("prediction", value);
          return c10::ivalue::Tuple::create({value, dictionary});
        },
        schema, output));
    const auto& manifest = bundle.verification.manifest;
    return NativeModel{
        .adapter = adapter,
        .pipeline = {
            .version = "structured-fixture-v1",
            .input = bind(manifest.input_schema, input_descriptor),
            .output = bind(manifest.output_schema, output_descriptor),
            .preprocessing = bind(manifest.preprocessing, example::kIdentity),
            .postprocessing = bind(manifest.postprocessing, example::kIdentity),
            .request_schema = schema,
            .response_schema = output,
            .preprocess = {},
            .postprocess = {}}};
  });
  auto input = Request().input;
  input.positional.push_back(torch::ones({1, 2}));
  c10::Dict<std::string, Tensor> expected_dictionary;
  expected_dictionary.insert("prediction", torch::tensor({{4.0F, -6.0F}}));
  const std::array<WarmupCase, 1> warmup{
      WarmupCase{.input = input,
                 .expected = c10::ivalue::Tuple::create(
                     {torch::tensor({{4.0F, -6.0F}}), expected_dictionary})}};
  auto predictor = backend.load(multi.directory, {}, warmup);
  auto request = Request();
  request.input = input;
  auto response = predictor->predict(request);
  const auto& tuple = response.output.value.toTupleRef().elements();
  ASSERT_EQ(tuple.size(), 2U);
  EXPECT_TRUE(
      torch::equal(tuple[0].toTensor(), torch::tensor({{4.0F, -6.0F}})));
  const auto field = tuple[1].toGenericDict().at("prediction").toTensor();
  EXPECT_NE(field.data_ptr(), tuple[0].toTensor().data_ptr());
  EXPECT_THROW(static_cast<void>(DynamicBatcher{predictor}), InvalidArgument);
  EXPECT_TRUE(predictor->close());
  EXPECT_TRUE(torch::equal(field, torch::tensor({{4.0F, -6.0F}})));
}
TEST_F(ServingTest, QueuedDeadlineExpiresAndNeverEntersLaterBatch) {
  auto gate = std::make_shared<Gate>();
  auto enabled = std::make_shared<std::atomic<bool>>(false);
  auto calls = std::make_shared<std::atomic<std::size_t>>(0);
  auto predictor = Load([gate, enabled, calls](const Tensor& x) {
    if (enabled->load()) {
      ++*calls;
      gate->Wait(x);
    }
  });
  enabled->store(true);
  BatchOptions options;
  options.max_batch_size = 1;
  options.max_wait = 0ms;
  DynamicBatcher batcher(predictor, options);
  auto active = batcher.submit(Request("active"));
  EXPECT_TRUE(gate->Entered());
  auto deadline = Request("queued");
  deadline.deadline = std::chrono::steady_clock::now() + 5ms;
  auto expired = batcher.submit(deadline);
  std::this_thread::sleep_for(10ms);
  gate->Release();
  EXPECT_NO_THROW(active.get());
  EXPECT_THROW(expired.get(), DeadlineExceeded);
  EXPECT_EQ(calls->load(), 1U);
  EXPECT_TRUE(batcher.close());
}
TEST_F(ServingTest, OrdinaryResourceFailuresNeverTriggerOomRetries) {
  auto enabled = std::make_shared<std::atomic<bool>>(false);
  auto predictor = Load([enabled](const Tensor&) {
    if (enabled->load())
      throw ResourceExhausted("Model resource limit, not an allocation error");
  });
  enabled->store(true);
  BatchOptions options;
  options.max_batch_size = 2;
  options.max_wait = 50ms;
  DynamicBatcher batcher(predictor, options);
  auto first = batcher.submit(Request("first"));
  auto second = batcher.submit(Request("second"));
  EXPECT_THROW(first.get(), ResourceExhausted);
  EXPECT_THROW(second.get(), ResourceExhausted);
  EXPECT_EQ(batcher.statistics().oom_retries, 0U);
  EXPECT_EQ(predictor->info().health, HealthStatus::degraded);
  EXPECT_TRUE(batcher.close());
}
TEST_F(ServingTest, TransportResponseLimitAndHostAuthorizationMustBeExplicit) {
  auto predictor = Load();
  auto protocol = Protocol();
  InferenceServiceAdapter service(
      predictor, protocol, {.max_request_bytes = 8, .max_response_bytes = 1});
  const std::array<float, 2> values{1, 2};
  TransportContext context{
      .correlation = Request().context, .authenticated = true, .deadline = {}};
  EXPECT_EQ(service.handle(std::as_bytes(std::span(values)), context).status,
            ServiceStatus::resource_exhausted);
  protocol.authorize = {};
  EXPECT_THROW(InferenceServiceAdapter(predictor, protocol), InvalidArgument);
  EXPECT_TRUE(service.close());
}
TEST(ServingCuda, ExplicitCudaNativeBundleNumericalForwardAndDrain) {
  if (!torch::cuda::is_available())
    GTEST_SKIP() << "CUDA unavailable in this validation environment";
  const Device device(c10::kCUDA, 0);
  example::BundleFixture fixture(device);
  fixture.Publish();
  auto predictor = fixture.Load();
  auto response = predictor->predict(example::Request(device));
  EXPECT_TRUE(torch::equal(response.output.value.toTensor().cpu(),
                           torch::tensor({{2.0F, -3.0F}})));
  EXPECT_EQ(response.runtime.runtime.device, device);
  EXPECT_TRUE(predictor->close());
}
TEST(ServingPreview, ManifestDoesNotClaimNetworkOrtReloadOrGaQualification) {
  const auto manifest = serving_preview_manifest_json();
  EXPECT_NE(manifest.find("\"default_enabled\":false"), std::string::npos);
  EXPECT_NE(manifest.find("\"network_server\":false"), std::string::npos);
  EXPECT_NE(manifest.find("\"onnx_runtime\":\"unavailable\""),
            std::string::npos);
  EXPECT_NE(manifest.find("requires-recorded-"), std::string::npos);
}
}  // namespace
