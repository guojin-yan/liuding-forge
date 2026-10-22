#include <jyppx/foundation/foundation.hpp>

#include <torch/torch.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <gtest/gtest.h>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

TEST(FoundationErrorTest, PreservesStableCodeAndStructuredContext) {
  jyppx::ErrorDetails details{
      .component = "DataPipeline",
      .operation = "batch_size",
      .expected = "> 0",
      .actual = "0",
      .hint = "Set a positive batch size.",
  };
  const jyppx::InvalidArgument error("Invalid batch size", details);

  EXPECT_EQ(error.code(), jyppx::ErrorCode::invalid_argument);
  EXPECT_EQ(error.details().component, "DataPipeline");
  EXPECT_NE(std::string(error.what()).find("Expected:  > 0"),
            std::string::npos);
  EXPECT_NE(std::string(error.what()).find("Actual:    0"), std::string::npos);
  EXPECT_NE(std::string(error.what()).find("Hint:"), std::string::npos);
}

TEST(FoundationErrorTest, KeepsSerializedCodeValuesAndNamesStable) {
  constexpr std::array<std::pair<jyppx::ErrorCode, const char*>, 13> kCases{{
      {jyppx::ErrorCode::invalid_argument, "invalid_argument"},
      {jyppx::ErrorCode::not_implemented, "not_implemented"},
      {jyppx::ErrorCode::device, "device"},
      {jyppx::ErrorCode::serialization, "serialization"},
      {jyppx::ErrorCode::numerical, "numerical"},
      {jyppx::ErrorCode::io, "io"},
      {jyppx::ErrorCode::resource_exhausted, "resource_exhausted"},
      {jyppx::ErrorCode::cancelled, "cancelled"},
      {jyppx::ErrorCode::compatibility, "compatibility"},
      {jyppx::ErrorCode::backend, "backend"},
      {jyppx::ErrorCode::internal, "internal"},
      {jyppx::ErrorCode::not_ready, "not_ready"},
      {jyppx::ErrorCode::deadline_exceeded, "deadline_exceeded"},
  }};

  for (std::size_t index = 0; index < kCases.size(); ++index) {
    const auto [code, name] = kCases[index];
    EXPECT_EQ(static_cast<std::uint16_t>(code), index + 1);
    EXPECT_EQ(jyppx::error_code_name(code), name);
  }
  EXPECT_EQ(jyppx::error_code_name(static_cast<jyppx::ErrorCode>(0)),
            "unknown");
}

TEST(FoundationErrorTest, BackendTranslationUsesExplicitKindAndNestsCause) {
  try {
    TORCH_CHECK(false, "synthetic backend failure");
  } catch (const c10::Error& cause) {
    try {
      jyppx::detail::rethrow_backend_error(
          cause, {.component = "FoundationTest",
                  .operation = "cuda_probe",
                  .failure_kind = jyppx::detail::BackendFailureKind::device,
                  .expected = "available device",
                  .hint = "Use a supported runtime."});
    } catch (const jyppx::DeviceError& converted) {
      EXPECT_EQ(converted.code(), jyppx::ErrorCode::device);
      EXPECT_EQ(converted.details().operation, "cuda_probe");
      EXPECT_THROW(std::rethrow_if_nested(converted), c10::Error);
      return;
    }
  }
  FAIL() << "backend error was not translated";
}

TEST(FoundationRandomTest, CpuStateRoundTripsAndHasVersion) {
  jyppx::core::RngGuard restore(20261014, false);
  const auto state = jyppx::core::get_rng_state(false);
  EXPECT_EQ(state.format_version, jyppx::core::kRngStateFormatVersion);
  const auto expected = torch::rand({8});
  (void)torch::rand({8});
  jyppx::core::set_rng_state(state, false);
  EXPECT_TRUE(torch::equal(torch::rand({8}), expected));
}

TEST(FoundationRandomTest, RejectsUnknownStateVersionWithoutMutation) {
  jyppx::core::RngGuard restore(17, false);
  const auto before = jyppx::core::get_rng_state(false);
  auto invalid = before;
  invalid.format_version += 1;
  EXPECT_THROW(jyppx::core::set_rng_state(invalid, false),
               jyppx::CompatibilityError);
  EXPECT_TRUE(torch::equal(jyppx::core::get_rng_state(false).cpu, before.cpu));
}

TEST(FoundationRandomTest, RejectsMalformedCpuStateWithoutMutation) {
  jyppx::core::RngGuard restore(77, false);
  const auto before = jyppx::core::get_rng_state(false);
  auto malformed = before;
  malformed.cpu = malformed.cpu.slice(0, 0, malformed.cpu.numel() - 1).clone();

  EXPECT_THROW(jyppx::core::set_rng_state(malformed, false),
               jyppx::InvalidArgument);
  EXPECT_TRUE(torch::equal(jyppx::core::get_rng_state(false).cpu, before.cpu));
}

TEST(FoundationRandomTest, RejectsEveryInvalidCpuRepresentation) {
  jyppx::core::RngGuard restore(78, false);
  const auto valid = jyppx::core::get_rng_state(false);

  auto undefined = valid;
  undefined.cpu = torch::Tensor{};
  EXPECT_THROW(jyppx::core::validate_rng_state(undefined, false),
               jyppx::InvalidArgument);

  auto wrong_type = valid;
  wrong_type.cpu = valid.cpu.to(torch::kInt64);
  EXPECT_THROW(jyppx::core::validate_rng_state(wrong_type, false),
               jyppx::InvalidArgument);

  auto wrong_rank = valid;
  wrong_rank.cpu = valid.cpu.reshape({1, valid.cpu.numel()});
  EXPECT_THROW(jyppx::core::validate_rng_state(wrong_rank, false),
               jyppx::InvalidArgument);

  auto unexpected_cuda = valid;
  unexpected_cuda.cuda.push_back(valid.cpu.clone());
  EXPECT_THROW(jyppx::core::validate_rng_state(unexpected_cuda, false),
               jyppx::InvalidArgument);
}

TEST(FoundationRandomTest, ManualSeedReproducesCpuSequence) {
  jyppx::core::RngGuard restore(79, false);
  jyppx::core::manual_seed(20261018, false);
  const auto expected = torch::rand({16});

  jyppx::core::manual_seed(20261018, false);
  EXPECT_TRUE(torch::equal(torch::rand({16}), expected));
}

TEST(FoundationRandomTest, GuardRestoresExactCpuContinuation) {
  jyppx::core::RngGuard restore(80, false);
  const auto before_scope = jyppx::core::get_rng_state(false);
  const auto expected = torch::rand({16});
  jyppx::core::set_rng_state(before_scope, false);

  {
    jyppx::core::RngGuard temporary_seed(1234, false);
    (void)torch::rand({64});
  }

  EXPECT_TRUE(torch::equal(torch::rand({16}), expected));
}

TEST(FoundationRandomTest, SeedDerivationIsStableAndSeparatesStreams) {
  const auto seed = jyppx::core::derive_seed(42, 7, 3, 1);
  EXPECT_EQ(seed, jyppx::core::derive_seed(42, 7, 3, 1));
  EXPECT_NE(seed, jyppx::core::derive_seed(42, 8, 3, 1));
  EXPECT_NE(seed, jyppx::core::derive_seed(42, 7, 4, 1));
  EXPECT_NE(seed, jyppx::core::derive_seed(42, 7, 3, 2));
}

TEST(FoundationVersionTest, ExposesStructuredAndJsonFingerprint) {
  EXPECT_STREQ(jyppx::version(), "0.3.0");
  const auto info = jyppx::build_info();
  EXPECT_STREQ(info.framework_version, jyppx::version());
  EXPECT_NE(std::string(info.libtorch_version).find('.'), std::string::npos);
  EXPECT_GE(info.cxx_standard, 202002L);

  const auto fingerprint = jyppx::current_build_fingerprint();
  EXPECT_EQ(fingerprint.jyppx_version, jyppx::version());
  EXPECT_FALSE(fingerprint.git_commit.empty());
  EXPECT_FALSE(fingerprint.compiler.empty());
  EXPECT_FALSE(fingerprint.os_arch.empty());
  EXPECT_FALSE(fingerprint.libtorch_version.empty());
  EXPECT_FALSE(fingerprint.libtorch_build.empty());
  EXPECT_EQ(fingerprint.libtorch_build.find_first_of("/\\"), std::string::npos);
#if defined(_GLIBCXX_USE_CXX11_ABI)
  EXPECT_TRUE(fingerprint.libtorch_build.ends_with(
      "cxx11-abi" + std::to_string(_GLIBCXX_USE_CXX11_ABI)));
#endif
  if (fingerprint.accelerator_runtime.starts_with("cuda:")) {
    EXPECT_TRUE(fingerprint.libtorch_build.starts_with("cuda-"));
  } else {
    EXPECT_TRUE(fingerprint.libtorch_build.starts_with("cpu-"));
  }
  EXPECT_FALSE(fingerprint.accelerator_runtime.empty());
  EXPECT_EQ(fingerprint.dependency_lock_digest.size(), 64U);
  EXPECT_EQ(fingerprint.dependency_lock_digest.find_first_not_of(
                "0123456789abcdefABCDEF"),
            std::string::npos);

  const auto json = jyppx::build_fingerprint_json(fingerprint);
  EXPECT_NE(json.find("\"jyppx_version\":\"0.3.0\""), std::string::npos);
  EXPECT_NE(json.find("\"git_commit\":"), std::string::npos);
  EXPECT_NE(json.find("\"accelerator_runtime\":"), std::string::npos);

  const auto compact = jyppx::build_fingerprint();
  EXPECT_NE(compact.find("jyppx=0.3.0"), std::string::npos);
  EXPECT_NE(compact.find("libtorch="), std::string::npos);
}

/** @brief Returns a deterministic non-host-specific test build identity. */
jyppx::BuildFingerprint test_build_fingerprint() {
  return {.jyppx_version = "0.3.0",
          .git_commit = "0123456789abcdef",
          .compiler = "test-compiler",
          .cxx_standard = "202002",
          .os_arch = "test-x86_64",
          .build_type = "Release",
          .libtorch_version = "2.5.1",
          .libtorch_build = "cuda-12.4-test-abi",
          .accelerator_runtime = "cuda:available:1",
          .dependency_lock_digest = std::string(64U, 'a')};
}

TEST(FoundationCapabilityTest, KeepsSchemaEnumNamesStable) {
  EXPECT_EQ(jyppx::capability_status_name(jyppx::CapabilityStatus::available),
            "available");
  EXPECT_EQ(jyppx::capability_implementation_name(
                jyppx::CapabilityImplementation::delegated),
            "delegated");
  EXPECT_EQ(jyppx::capability_source_name(
                jyppx::CapabilitySource::configured_instance),
            "configured-instance");
  EXPECT_EQ(
      jyppx::capability_status_name(static_cast<jyppx::CapabilityStatus>(0)),
      "unknown");
  EXPECT_EQ(jyppx::capability_implementation_name(
                static_cast<jyppx::CapabilityImplementation>(0)),
            "unknown");
  EXPECT_EQ(
      jyppx::capability_source_name(static_cast<jyppx::CapabilitySource>(0)),
      "unknown");
}

TEST(FoundationCapabilityTest, QueriesCurrentComponentBuildInventory) {
  const auto manifest = jyppx::current_capability_manifest();
  ASSERT_EQ(manifest.features.size(), 12U);
  EXPECT_EQ(manifest.build.jyppx_version, jyppx::version());
  EXPECT_EQ(manifest.features.front().feature_id, "jyppx.component.core");

  const std::array<std::pair<std::string_view, bool>, 12U> expected_components{{
      {"core", JYPPX_COMPONENT_CORE_ENABLED != 0},
      {"data", JYPPX_COMPONENT_DATA_ENABLED != 0},
      {"data_image", JYPPX_COMPONENT_DATA_IMAGE_ENABLED != 0},
      {"experiment", JYPPX_COMPONENT_EXPERIMENT_ENABLED != 0},
      {"foundation", true},
      {"interop", JYPPX_COMPONENT_INTEROP_ENABLED != 0},
      {"models", JYPPX_COMPONENT_MODELS_ENABLED != 0},
      {"nn", JYPPX_COMPONENT_NN_ENABLED != 0},
      {"optim", JYPPX_COMPONENT_OPTIM_ENABLED != 0},
      {"quantization", JYPPX_COMPONENT_QUANTIZATION_ENABLED != 0},
      {"serving", JYPPX_COMPONENT_SERVING_ENABLED != 0},
      {"training", JYPPX_COMPONENT_TRAINING_ENABLED != 0},
  }};
  for (std::size_t index = 0; index < expected_components.size(); ++index) {
    const auto& [component, enabled] = expected_components[index];
    const auto& feature = manifest.features[index];
    EXPECT_EQ(feature.feature_id, "jyppx.component." + std::string(component));
    const bool preview = component == "models" || component == "quantization" ||
                         component == "serving" || component == "data_image";
    EXPECT_EQ(feature.status,
              enabled ? (preview ? jyppx::CapabilityStatus::experimental
                                 : jyppx::CapabilityStatus::available)
                      : jyppx::CapabilityStatus::unsupported);
    EXPECT_EQ(feature.implementation,
              enabled ? jyppx::CapabilityImplementation::native
                      : jyppx::CapabilityImplementation::unsupported);
    EXPECT_EQ(feature.source, jyppx::CapabilitySource::build);
    EXPECT_EQ(feature.backend, "jyppx");
    EXPECT_EQ(feature.schemas, (std::vector<std::string>{"component/v1"}));
    EXPECT_TRUE(feature.scope_id.empty());
    EXPECT_EQ(feature.limitations.empty(), enabled && !preview);
  }

  const auto json = jyppx::current_capability_manifest_json();
  EXPECT_EQ(json, jyppx::current_capability_manifest_json());
  EXPECT_NE(json.find("\"feature_id\":\"jyppx.component.foundation\""),
            std::string::npos);
  EXPECT_NE(json.find("\"status\":\"available\""), std::string::npos);
}

TEST(FoundationCapabilityTest, CanonicalizesAndSerializesTypedManifest) {
  jyppx::CapabilityDescriptor second{
      .feature_id = "jyppx.training.strategy.single-device",
      .feature_version = 1,
      .status = jyppx::CapabilityStatus::available,
      .implementation = jyppx::CapabilityImplementation::native,
      .source = jyppx::CapabilitySource::runtime,
      .backend = "libtorch",
      .devices = {"cuda:0", "cpu"},
      .dtypes = {"Float", "Half"},
      .schemas = {"strategy/v1"},
      .operations = {"optimizer.step", "model.forward"},
      .limits = {{.name = "world_size", .value = 1, .unit = "ranks"}},
      .limitations = {},
  };
  jyppx::CapabilityDescriptor first{
      .feature_id = "jyppx.data.pipeline.map",
      .scope_id = "data.pipeline.primary",
      .feature_version = 1,
      .status = jyppx::CapabilityStatus::experimental,
      .implementation = jyppx::CapabilityImplementation::adapted,
      .source = jyppx::CapabilitySource::configured_instance,
      .backend = "libtorch",
      .devices = {"cuda:0"},
      .schemas = {"batch/v1", "fnv1a64/0123456789abcdef"},
      .operations = {"worker.prefetch", "epoch.explicit"},
      .limits = {{.name = "worker_count", .value = 4, .unit = "workers"},
                 {.name = "max_batch_bytes", .value = 4096, .unit = "bytes"}},
      .limitations = {"process workers are unsupported"},
  };

  const auto manifest = jyppx::make_capability_manifest(
      {second, first}, test_build_fingerprint());
  ASSERT_EQ(manifest.features.size(), 2U);
  EXPECT_EQ(manifest.features[0].feature_id, "jyppx.data.pipeline.map");
  EXPECT_EQ(manifest.features[0].scope_id, "data.pipeline.primary");
  EXPECT_EQ(manifest.features[0].operations,
            (std::vector<std::string>{"epoch.explicit", "worker.prefetch"}));
  EXPECT_EQ(manifest.features[0].limits.front().name, "max_batch_bytes");

  const auto json = jyppx::capability_manifest_json(manifest);
  EXPECT_EQ(json, jyppx::capability_manifest_json(manifest));
  EXPECT_LT(json.find("jyppx.data.pipeline.map"),
            json.find("jyppx.training.strategy.single-device"));
  EXPECT_NE(json.find("\"status\":\"experimental\""), std::string::npos);
  EXPECT_NE(json.find("\"scope_id\":\"data.pipeline.primary\""),
            std::string::npos);
  EXPECT_NE(json.find("\"value\":4096"), std::string::npos);
  EXPECT_EQ(json.find("\"value\":\"4096\""), std::string::npos);
  EXPECT_NE(json.find("\"build\":{\"jyppx_version\":\"0.3.0\""),
            std::string::npos);
}

TEST(FoundationCapabilityTest, RejectsContradictionsDuplicatesAndBounds) {
  jyppx::CapabilityDescriptor unsupported{
      .feature_id = "jyppx.distributed.collective",
      .feature_version = 1,
      .status = jyppx::CapabilityStatus::unsupported,
      .implementation = jyppx::CapabilityImplementation::native,
      .source = jyppx::CapabilitySource::build,
      .backend = "libtorch",
      .devices = {},
      .dtypes = {},
      .schemas = {},
      .operations = {},
      .limits = {},
      .limitations = {"collective backend is not implemented"},
  };
  EXPECT_THROW(jyppx::validate_capability_descriptor(unsupported),
               jyppx::InvalidArgument);

  unsupported.implementation = jyppx::CapabilityImplementation::unsupported;
  unsupported.operations = {"collective.all-reduce"};
  EXPECT_THROW(jyppx::validate_capability_descriptor(unsupported),
               jyppx::InvalidArgument);
  unsupported.operations.clear();
  EXPECT_NO_THROW(jyppx::validate_capability_descriptor(unsupported));

  auto duplicate_value = unsupported;
  duplicate_value.devices = {"cpu", "cpu"};
  EXPECT_THROW(jyppx::validate_capability_descriptor(duplicate_value),
               jyppx::InvalidArgument);

  auto uppercase_identifier = unsupported;
  uppercase_identifier.backend = "LibTorch";
  EXPECT_THROW(jyppx::validate_capability_descriptor(uppercase_identifier),
               jyppx::InvalidArgument);

  EXPECT_THROW(jyppx::make_capability_manifest({unsupported, unsupported},
                                               test_build_fingerprint()),
               jyppx::InvalidArgument);

  std::vector<jyppx::CapabilityDescriptor> excessive(
      jyppx::kMaxCapabilityFeatures + 1U, unsupported);
  for (std::size_t index = 0; index < excessive.size(); ++index) {
    excessive[index].feature_id =
        "jyppx.unsupported.feature-" + std::to_string(index);
  }
  EXPECT_THROW(jyppx::make_capability_manifest(std::move(excessive),
                                               test_build_fingerprint()),
               jyppx::ResourceExhausted);

  auto invalid_utf8 = unsupported;
  invalid_utf8.limitations = {std::string("\xC3\x28", 2)};
  EXPECT_THROW(jyppx::validate_capability_descriptor(invalid_utf8),
               jyppx::InvalidArgument);
}

TEST(FoundationCapabilityTest, RejectsEverySchemaV1ControlBoundary) {
  const jyppx::CapabilityDescriptor valid{
      .feature_id = "jyppx.test.feature",
      .feature_version = 1,
      .status = jyppx::CapabilityStatus::available,
      .implementation = jyppx::CapabilityImplementation::native,
      .source = jyppx::CapabilitySource::build,
      .backend = "libtorch",
      .devices = {"cpu"},
      .operations = {"test.execute"},
      .limits = {{.name = "items", .value = 1, .unit = "count"}},
  };
  const auto expect_invalid = [&valid](const auto& mutate) {
    auto candidate = valid;
    mutate(candidate);
    EXPECT_THROW(jyppx::validate_capability_descriptor(candidate),
                 jyppx::InvalidArgument);
  };

  expect_invalid([](auto& value) { value.feature_id.clear(); });
  expect_invalid([](auto& value) { value.feature_version = 0; });
  expect_invalid([](auto& value) {
    value.status = static_cast<jyppx::CapabilityStatus>(0);
  });
  expect_invalid([](auto& value) {
    value.implementation = static_cast<jyppx::CapabilityImplementation>(0);
  });
  expect_invalid([](auto& value) {
    value.source = static_cast<jyppx::CapabilitySource>(0);
  });
  expect_invalid([](auto& value) { value.scope_id = "unexpected.scope"; });
  expect_invalid([](auto& value) {
    value.source = jyppx::CapabilitySource::configured_instance;
  });
  expect_invalid(
      [](auto& value) { value.limits.push_back(value.limits.front()); });
  expect_invalid([](auto& value) {
    value.status = jyppx::CapabilityStatus::unsupported;
    value.implementation = jyppx::CapabilityImplementation::unsupported;
    value.operations.clear();
    value.limitations.clear();
  });

  auto too_many_values = valid;
  too_many_values.devices.assign(jyppx::kMaxCapabilityValues + 1U, "cpu");
  EXPECT_THROW(jyppx::validate_capability_descriptor(too_many_values),
               jyppx::ResourceExhausted);

  auto too_many_limits = valid;
  too_many_limits.limits.assign(jyppx::kMaxCapabilityValues + 1U,
                                {.name = "items", .value = 1, .unit = "count"});
  EXPECT_THROW(jyppx::validate_capability_descriptor(too_many_limits),
               jyppx::ResourceExhausted);

  auto oversized_string = valid;
  oversized_string.backend.assign(jyppx::kMaxCapabilityStringBytes + 1U, 'a');
  EXPECT_THROW(jyppx::validate_capability_descriptor(oversized_string),
               jyppx::ResourceExhausted);

  auto manifest =
      jyppx::make_capability_manifest({valid}, test_build_fingerprint());
  manifest.schema_version = 2;
  EXPECT_THROW(jyppx::validate_capability_manifest(manifest),
               jyppx::InvalidArgument);
  manifest.schema_version = jyppx::CapabilityManifest::format_version;
  manifest.build.git_commit.clear();
  EXPECT_THROW(jyppx::validate_capability_manifest(manifest),
               jyppx::InvalidArgument);
}

TEST(FoundationCapabilityTest,
     SupportsScopedInstancesAndRejectsDuplicateCompositeIdentity) {
  const jyppx::CapabilityDescriptor first{
      .feature_id = "jyppx.data.pipeline.map",
      .scope_id = "data.pipeline.first",
      .feature_version = 1,
      .status = jyppx::CapabilityStatus::available,
      .implementation = jyppx::CapabilityImplementation::native,
      .source = jyppx::CapabilitySource::configured_instance,
      .backend = "libtorch",
      .operations = {"source.map"},
  };
  auto second = first;
  second.scope_id = "data.pipeline.second";

  const auto manifest = jyppx::make_capability_manifest(
      {second, first}, test_build_fingerprint());
  ASSERT_EQ(manifest.features.size(), 2U);
  EXPECT_EQ(manifest.features[0].scope_id, "data.pipeline.first");
  EXPECT_EQ(manifest.features[1].scope_id, "data.pipeline.second");

  EXPECT_THROW(
      jyppx::make_capability_manifest({first, first}, test_build_fingerprint()),
      jyppx::InvalidArgument);
}

TEST(FoundationCapabilityTest, ValidatesUtf8AndEscapesJsonText) {
  jyppx::CapabilityDescriptor descriptor{
      .feature_id = "jyppx.test.utf8",
      .feature_version = 1,
      .status = jyppx::CapabilityStatus::available,
      .implementation = jyppx::CapabilityImplementation::native,
      .source = jyppx::CapabilitySource::runtime,
      .backend = "libtorch",
      .limitations = {"quoted \"text\" with \\ and \xC2\xA2 \xE2\x82\xAC "
                      "\xF0\x90\x8D\x88"},
  };
  EXPECT_NO_THROW(jyppx::validate_capability_descriptor(descriptor));
  const auto json = jyppx::capability_manifest_json(
      jyppx::make_capability_manifest({descriptor}, test_build_fingerprint()));
  EXPECT_NE(json.find("quoted \\\"text\\\" with \\\\"), std::string::npos);

  const std::vector<std::string> invalid_utf8{
      std::string("\x80", 1),         std::string("\xC2", 1),
      std::string("\xC2\x20", 2),     std::string("\xC0\x80", 2),
      std::string("\xED\xA0\x80", 3), std::string("\xF4\x90\x80\x80", 4),
  };
  for (const auto& invalid : invalid_utf8) {
    descriptor.limitations = {invalid};
    EXPECT_THROW(jyppx::validate_capability_descriptor(descriptor),
                 jyppx::InvalidArgument);
  }
}

}  // namespace
