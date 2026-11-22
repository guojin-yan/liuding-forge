// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
#include "../../../integration/deployment/migraphx_serving_fixture.hpp"
#include "../../../integration/deployment/native_modules.hpp"

#include <jyppx/liuding/serving/reload.hpp>

#include <atomic>
#include <future>
#include <gtest/gtest.h>
#include <limits>

namespace {
using namespace jyppx_test::migraphx_serving;
using namespace std::chrono_literals;

mgx::EnvironmentLock Sample() {
  const auto build = current_build_fingerprint();
  return {.adapter_version = build.jyppx_version,
          .libtorch_version = build.libtorch_version,
          .os_arch = "test-x86_64",
          .migraphx_header_version = "2.18.0",
          .hip_header_version = "7.16.26385",
          .migraphx_c = {.sha256 = std::string(64, '1'), .size_bytes = 123},
          .hip_runtime = {.sha256 = std::string(64, '2'), .size_bytes = 456},
          .hip_runtime_version = 71626385,
          .hip_driver_version = 71626385,
          .device_id = 0,
          .device_architecture = "gfx1100",
          .device_uuid = std::string(32, 'a'),
          .device_pci_bus_id = "0000:03:00.0",
          .input_name = "input",
          .input_shape = {2, 8},
          .output_shape = {2, 3}};
}
TEST(MIGraphXServing, ExplicitDeclarationContracts) {
  const auto descriptor = migraphx_runtime_descriptor(Sample());
  const mgx::SessionOptions options{.input_shape = {2, 8}};
  EXPECT_NO_THROW(
      MIGraphXRuntimeBackend(descriptor, options, Bundle::Factory()));
  EXPECT_THROW(MIGraphXRuntimeBackend(descriptor, options, {}),
               InvalidArgument);
  for (const auto flag :
       {&RuntimeDescriptor::dynamic_shapes, &RuntimeDescriptor::tensor_batching,
        &RuntimeDescriptor::oom_retry_safe,
        &RuntimeDescriptor::allow_cpu_fallback}) {
    auto changed = descriptor;
    changed.*flag = true;
    EXPECT_THROW(MIGraphXRuntimeBackend(changed, options, Bundle::Factory()),
                 InvalidArgument);
  }
  for (const auto member :
       {&RuntimeDescriptor::backend_id, &RuntimeDescriptor::format,
        &RuntimeDescriptor::execution_provider}) {
    auto changed = descriptor;
    changed.*member = "wrong";
    EXPECT_THROW(MIGraphXRuntimeBackend(changed, options, Bundle::Factory()),
                 InvalidArgument);
  }
  auto changed = descriptor;
  changed.provider_options["device_id"] = "1";
  EXPECT_THROW(MIGraphXRuntimeBackend(changed, options, Bundle::Factory()),
               InvalidArgument);
  changed = descriptor;
  changed.dtypes = {torch::kFloat16};
  EXPECT_THROW(MIGraphXRuntimeBackend(changed, options, Bundle::Factory()),
               InvalidArgument);
  auto invalid = options;
  invalid.device_id = -1;
  EXPECT_THROW(MIGraphXRuntimeBackend(descriptor, invalid, Bundle::Factory()),
               InvalidArgument);
  invalid = options;
  invalid.input_shape = {0, 8};
  EXPECT_THROW(MIGraphXRuntimeBackend(descriptor, invalid, Bundle::Factory()),
               InvalidArgument);
  LoadOptions load;
  load.max_snapshot_bytes = 0;
  EXPECT_THROW(MIGraphXRuntimeBackend(descriptor, options, Bundle::Factory())
                   .load_reservation_bytes(load),
               ResourceExhausted);
}
TEST(MIGraphXServing, ExactFixedSchemas) {
  const std::vector<std::size_t> shape{2, 8};
  const auto schema = migraphx_invocation_schema("input", shape);
  ASSERT_EQ(schema.keyword.size(), 1U);
  EXPECT_TRUE(schema.positional.empty());
  EXPECT_FALSE(schema.allow_extra_keywords);
  EXPECT_NO_THROW(nn::validate_invocation(
      {.positional = {}, .keyword = {{"input", torch::zeros({2, 8})}}},
      schema));
  EXPECT_THROW(
      nn::validate_invocation(
          {.positional = {}, .keyword = {{"input", torch::zeros({1, 8})}}},
          schema),
      InvalidArgument);
  EXPECT_THROW(migraphx_invocation_schema("", shape), InvalidArgument);
  EXPECT_THROW(migraphx_invocation_schema("bad\nname", shape), InvalidArgument);
  for (const auto& bad : std::vector<std::vector<std::size_t>>{
           {},
           {2, 0},
           std::vector<std::size_t>(9, 1),
           {std::numeric_limits<std::size_t>::max(), 2}})
    EXPECT_THROW(migraphx_result_schema(bad), InvalidArgument);
}

#ifdef JYPPX_MIGRAPHX_SERVING_FIXTURES
std::filesystem::path Evidence(const char* name) {
  const auto path =
      std::filesystem::path(JYPPX_MIGRAPHX_SERVING_EVIDENCE) /
      (std::string(name) + "-" +
       std::to_string(
           std::chrono::system_clock::now().time_since_epoch().count()));
  if (!std::filesystem::create_directories(path))
    throw std::runtime_error("fresh evidence required");
  std::cout << "native_migraphx_serving_evidence=" << path.string() << '\n';
  return path;
}
void Fixed(int index, bool resnet = false) {
  const auto root = Evidence(resnet ? (index ? "resnet18-5" : "resnet18-2")
                                    : (index ? "linear5" : "linear2"));
  Fixture fixture(JYPPX_MIGRAPHX_SERVING_FIXTURES, index, resnet);
  Bundle bundle(root / "owner", fixture);
  const auto directory = bundle.Publish();
  const auto input_before = fixture.input.clone();
  const std::array known{fixture.Known()};
  const auto backend = bundle.Backend(fixture);
  EXPECT_EQ(backend.load_reservation_bytes({.max_snapshot_bytes = 1234}),
            1234U);
  auto predictor = backend.load(directory, {}, known);
  EXPECT_EQ(predictor->info().health, HealthStatus::ready);
  EXPECT_EQ(predictor->info().runtime.device, Device(c10::kCPU));
  EXPECT_EQ(predictor->info().runtime.execution_provider, "migraphx.gpu");
  EXPECT_FALSE(predictor->execution_schema_contract().empty());
  auto response = predictor->predict(fixture.Request("migraphx-v1"));
  Compare(response.output.value, fixture);
  WriteOutput(root / "actual.f32", response.output.value);
  auto first = std::async(std::launch::async, [&] {
    return predictor->predict(fixture.Request());
  });
  auto second = std::async(std::launch::async, [&] {
    return predictor->predict(fixture.Request());
  });
  auto a = first.get();
  auto b = second.get();
  Compare(a.output.value, fixture);
  Compare(b.output.value, fixture);
  {
    c10::InferenceMode edit;
    a.output.value.toTensor().zero_();
  }
  Compare(b.output.value, fixture);
  Compare(response.output.value, fixture);
  EXPECT_TRUE(torch::equal(fixture.input, input_before));
  auto wrong = fixture.Request();
  wrong.input.keyword["input"] = torch::zeros({1, 8});
  EXPECT_THROW(predictor->predict(wrong), InvalidArgument);
  wrong = fixture.Request();
  wrong.input.keyword["input"] =
      torch::full_like(fixture.input, std::numeric_limits<float>::quiet_NaN());
  EXPECT_THROW(predictor->predict(wrong), NumericalError);
  EXPECT_EQ(predictor->info().health, HealthStatus::ready);
  EXPECT_TRUE(predictor->close());
  EXPECT_EQ(predictor->info().health, HealthStatus::closed);
  EXPECT_THROW(predictor->predict(fixture.Request()), NotReady);
  predictor.reset();
  Compare(response.output.value, fixture);
  jyppx_test::Modules(root / "modules.txt");
}
TEST(MIGraphXServingGpu, FixedLinear2OwnershipConcurrencyAndClose) { Fixed(0); }
TEST(MIGraphXServingGpu, FixedLinear5OwnershipConcurrencyAndClose) { Fixed(1); }
TEST(MIGraphXServingGpu, FixedResNet18Batch2OwnershipConcurrencyAndClose) {
  Fixed(0, true);
}
TEST(MIGraphXServingGpu, FixedResNet18Batch5OwnershipConcurrencyAndClose) {
  Fixed(1, true);
}

TEST(MIGraphXServingGpu, RejectsUnverifiedContractsAndBudgets) {
  const auto root = Evidence("rejections");
  Fixture fixture(JYPPX_MIGRAPHX_SERVING_FIXTURES);
  Bundle bundle(root / "owner", fixture);
  const auto directory = bundle.Publish();
  const auto backend = bundle.Backend(fixture);
  const std::array known{fixture.Known()};
  EXPECT_THROW(backend.load(directory, {}, {}), InvalidArgument);
  LoadOptions limited;
  limited.max_snapshot_bytes = 1;
  EXPECT_THROW(backend.load(directory, {}, known, limited), ResourceExhausted);
  limited = {};
  limited.request_limits.max_tensor_bytes = 1;
  EXPECT_THROW(backend.load(directory, {}, known, limited), ResourceExhausted);
  limited = {};
  limited.output_limits.max_tensor_bytes = 1;
  EXPECT_THROW(backend.load(directory, {}, known, limited), ResourceExhausted);
  auto wrong_known = known;
  wrong_known[0].expected = torch::zeros_like(fixture.expected);
  EXPECT_THROW(backend.load(directory, {}, wrong_known), NumericalError);
  auto unbound = Bundle::Factory();
  const auto bad_binding = [unbound](const auto& b, const auto& e) {
    auto p = unbound(b, e);
    p.input.digest.hex = std::string(64, '0');
    return p;
  };
  EXPECT_THROW(bundle.Backend(fixture, bad_binding).load(directory, {}, known),
               CompatibilityError);
  const auto wrong_schema = [unbound](const auto& b, const auto& e) {
    auto p = unbound(b, e);
    const std::array<std::size_t, 2> shape{2, 4};
    p.response_schema = migraphx_result_schema(shape);
    return p;
  };
  EXPECT_THROW(bundle.Backend(fixture, wrong_schema).load(directory, {}, known),
               CompatibilityError);
  auto drift = fixture.environment;
  drift.hip_runtime.sha256[0] = drift.hip_runtime.sha256[0] == '0' ? '1' : '0';
  Bundle changed(root / "environment-drift", fixture, &drift);
  EXPECT_THROW(changed.Backend(fixture).load(changed.Publish(), {}, known),
               CompatibilityError);
  bundle.manifest.runtime.devices = {"cpu"};
  EXPECT_THROW(backend.load(bundle.Publish("wrong-domain"), {}, known),
               CompatibilityError);
  bundle.manifest.runtime.devices = {"rocm"};
  bundle.manifest.evidence.front().schema_id = "jyppx.test.wrong-lock";
  EXPECT_THROW(backend.load(bundle.Publish("wrong-lock-schema"), {}, known),
               CompatibilityError);
  bundle.manifest.evidence.front().schema_id =
      std::string(mgx::kEnvironmentLockSchemaId);
  auto duplicate =
      bundle.Description("duplicate-lock", interop::BundleFileRole::evidence,
                         mgx::serialize_environment_lock(fixture.environment));
  duplicate.schema_id = std::string(mgx::kEnvironmentLockSchemaId);
  bundle.manifest.evidence.push_back(std::move(duplicate));
  EXPECT_THROW(backend.load(bundle.Publish("duplicate-lock-schema"), {}, known),
               CompatibilityError);
  bundle.manifest.evidence.clear();
  // An unclaimed evidence file is rejected by the common manifest contract.
  EXPECT_THROW(bundle.Publish("missing-evidence"), InvalidArgument);
}
TEST(MIGraphXServingGpu, CooperativeInterruptionsPreserveReadySession) {
  const auto root = Evidence("interruptions");
  Fixture fixture(JYPPX_MIGRAPHX_SERVING_FIXTURES);
  Bundle bundle(root / "owner", fixture);
  const auto directory = bundle.Publish();
  const std::array known{fixture.Known()};
  const auto backend = bundle.Backend(fixture);
  std::stop_source cancelled;
  cancelled.request_stop();
  LoadOptions options;
  options.stop = cancelled.get_token();
  EXPECT_THROW(backend.load(directory, {}, known, options), Cancelled);
  options = {};
  options.deadline = std::chrono::steady_clock::now() - 1s;
  EXPECT_THROW(backend.load(directory, {}, known, options), DeadlineExceeded);
  std::atomic<bool> interrupt{false}, fail_output{false};
  std::stop_source during;
  const auto original = Bundle::Factory();
  const auto owner = [&](const auto& b, const auto& e) {
    auto pipeline = original(b, e);
    pipeline.postprocess = [&](const nn::InvocationResult& result) {
      if (interrupt.load()) during.request_stop();
      if (fail_output.load()) throw BackendError("trusted postprocess failed");
      return result;
    };
    return pipeline;
  };
  auto predictor = bundle.Backend(fixture, owner).load(directory, {}, known);
  EXPECT_THROW(predictor->predict(fixture.Request(), cancelled.get_token()),
               Cancelled);
  auto request = fixture.Request();
  request.deadline = std::chrono::steady_clock::now() - 1s;
  EXPECT_THROW(predictor->predict(request), DeadlineExceeded);
  interrupt = true;
  EXPECT_THROW(predictor->predict(fixture.Request(), during.get_token()),
               Cancelled);
  EXPECT_EQ(predictor->info().health, HealthStatus::ready);
  interrupt = false;
  Compare(predictor->predict(fixture.Request()).output.value, fixture);
  fail_output = true;
  EXPECT_THROW(predictor->predict(fixture.Request()), BackendError);
  EXPECT_EQ(predictor->info().health, HealthStatus::degraded);
  EXPECT_THROW(predictor->predict(fixture.Request()), NotReady);
  EXPECT_TRUE(predictor->close());
}
TEST(MIGraphXServingGpu, VersionPublicationRetainsLeasesAndRejectsBadWarmup) {
  const auto root = Evidence("versions");
  Fixture fixture(JYPPX_MIGRAPHX_SERVING_FIXTURES);
  Bundle bundle(root / "owner", fixture);
  const auto first = bundle.Publish();
  const auto second = bundle.Publish("migraphx-v2");
  const auto backend = bundle.Backend(fixture);
  const std::array known{fixture.Known()};
  LoadOptions load;
  load.max_snapshot_bytes = 1U << 20U;
  ReloadablePredictor versions(
      backend.load(first, {}, known, load),
      {.initial_reservation_bytes = backend.load_reservation_bytes(load),
       .max_resident_bytes = 4U << 20U});
  {
    auto retained = versions.acquire("migraphx-v1");
    auto wrong_known = known;
    wrong_known[0].expected = torch::zeros_like(fixture.expected);
    EXPECT_THROW(versions.reload(backend, second, {}, wrong_known, load),
                 NumericalError);
    EXPECT_EQ(versions.info().model_version, "migraphx-v1");
    const auto result =
        versions.reload(backend, second, {}, known, load,
                        {.candidate_timeout = 30s, .drain_timeout = 0ms});
    EXPECT_EQ(result.current_version, "migraphx-v2");
    EXPECT_FALSE(result.previous_drained);
    Compare(retained.predict(fixture.Request("migraphx-v1")).output.value,
            fixture);
    Compare(versions.predict(fixture.Request("migraphx-v2")).output.value,
            fixture);
    EXPECT_EQ(versions.statistics().resident_instances, 2U);
    EXPECT_FALSE(versions.drain_previous(std::chrono::steady_clock::now()));
  }
  EXPECT_TRUE(versions.drain_previous());
  EXPECT_EQ(versions.statistics().resident_instances, 1U);
  EXPECT_TRUE(versions.close());
  EXPECT_EQ(versions.statistics().reserved_bytes, 0U);
}
#endif
}  // namespace
