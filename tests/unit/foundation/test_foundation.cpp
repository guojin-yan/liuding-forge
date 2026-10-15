#include <jyppx/foundation/foundation.hpp>

#include <array>
#include <cstdint>
#include <exception>
#include <gtest/gtest.h>
#include <string>
#include <utility>

namespace {

TEST(FoundationErrorTest, PreservesStableCodeAndStructuredContext) {
  jyppx::ErrorDetails details{
      .component = "DataLoader",
      .operation = "batch_size",
      .expected = "> 0",
      .actual = "0",
      .hint = "Set a positive batch size.",
  };
  const jyppx::InvalidArgument error("Invalid batch size", details);

  EXPECT_EQ(error.code(), jyppx::ErrorCode::invalid_argument);
  EXPECT_EQ(error.details().component, "DataLoader");
  EXPECT_NE(std::string(error.what()).find("Expected:  > 0"),
            std::string::npos);
  EXPECT_NE(std::string(error.what()).find("Actual:    0"), std::string::npos);
  EXPECT_NE(std::string(error.what()).find("Hint:"), std::string::npos);
}

TEST(FoundationErrorTest, KeepsSerializedCodeValuesAndNamesStable) {
  constexpr std::array<std::pair<jyppx::ErrorCode, const char*>, 11> kCases{{
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
  EXPECT_EQ(fingerprint.libtorch_build.find_first_of("/\\"),
            std::string::npos);
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

}  // namespace
