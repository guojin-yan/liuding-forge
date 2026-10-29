// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
#include <jyppx/liuding/deployment/onnx_export/exporter.hpp>
#include <jyppx/liuding/deployment/onnx_runtime/environment.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/foundation/version.hpp>

#include <torch/torch.h>

#include <algorithm>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <vector>

namespace {
using namespace jyppx::liuding;
namespace runtime = jyppx::liuding::onnx_runtime;
using Json = nlohmann::json;

ArtifactDigest Digest(std::string_view text) {
  return {.hex =
              sha256_hex(std::as_bytes(std::span(text.data(), text.size())))};
}

const onnx_export::ExportResult& Artifact() {
  static const auto artifact = [] {
    torch::nn::Linear model(3, 2);
    model->eval();
    return onnx_export::export_model(model, torch::zeros({2, 3}));
  }();
  return artifact;
}

class OnnxEnvironmentTest : public ::testing::Test {
 protected:
  runtime::Session session{Artifact().onnx, Artifact().digest};
  runtime::EnvironmentLock observed = runtime::observe_environment(session);
  std::string encoded = runtime::serialize_environment_lock(observed);
  Json document = Json::parse(encoded);

  void ExpectMismatch(const Json& changed) {
    const auto bytes = changed.dump();
    EXPECT_THROW(runtime::verify_environment_lock(session, bytes, Digest(bytes),
                                                  observed.runtime),
                 CompatibilityError);
  }
};

TEST_F(OnnxEnvironmentTest,
       PreservesLegacyWireSchemaAndObservesLinkedVersions) {
  const auto build = current_build_fingerprint();
  auto platform = build.os_arch;
  for (auto& byte : platform)
    if (byte >= 'A' && byte <= 'Z') byte = static_cast<char>(byte - 'A' + 'a');
  const Json legacy{
      {"format", "jyppx.liuding.onnx_runtime_lock"},
      {"schema_version", 1},
      {"backend_id", "jyppx.liuding.onnx_runtime"},
      {"backend_version", build.jyppx_version},
      {"dependencies",
       Json::array(
           {{{"name", "libtorch"}, {"version", build.libtorch_version}},
            {{"name", "liuding-forge"}, {"version", build.jyppx_version}},
            {{"name", "onnxruntime"},
             {"version", session.runtime_version()}}})},
      {"devices", Json::array({"cpu"})},
      {"dtypes", Json::array({"float32"})},
      {"os_arch", Json::array({platform})},
      {"dynamic_shapes", true},
      {"provider",
       {{"name", "CPUExecutionProvider"},
        {"options", Json::object()},
        {"allow_cpu_fallback", false}}}};
  EXPECT_EQ(encoded, legacy.dump());
  const auto parsed = runtime::verify_environment_lock(
      session, encoded, Digest(encoded), observed.runtime);
  EXPECT_EQ(runtime::serialize_environment_lock(parsed), encoded);
  EXPECT_EQ(parsed.runtime.dependencies.size(), 3U);
  EXPECT_TRUE(parsed.runtime.dynamic_shapes);
  EXPECT_EQ(parsed.provider.name, "CPUExecutionProvider");
}

TEST_F(OnnxEnvironmentTest, RejectsChangedBytesAndIndependentDeclarationDrift) {
  EXPECT_THROW(runtime::verify_environment_lock(
                   session, encoded + " ", Digest(encoded), observed.runtime),
               CompatibilityError);
  auto declared = observed.runtime;
  declared.dependencies.back().version = "not-the-linked-version";
  EXPECT_THROW(runtime::verify_environment_lock(session, encoded,
                                                Digest(encoded), declared),
               CompatibilityError);
  declared = observed.runtime;
  declared.devices = {"cuda"};
  EXPECT_THROW(runtime::verify_environment_lock(session, encoded,
                                                Digest(encoded), declared),
               CompatibilityError);
  EXPECT_THROW(runtime::verify_environment_lock(
                   session, encoded, {.hex = "wrong"}, observed.runtime),
               InvalidArgument);
}

TEST_F(OnnxEnvironmentTest,
       RejectsSelfConsistentButUnobservedProviderAndEnvironment) {
  auto changed = document;
  changed["provider"]["name"] = "CUDAExecutionProvider";
  ExpectMismatch(changed);
  changed = document;
  changed["provider"]["options"]["device_id"] = "1";
  ExpectMismatch(changed);
  changed = document;
  changed["provider"]["allow_cpu_fallback"] = true;
  ExpectMismatch(changed);
  changed = document;
  changed["os_arch"] = Json::array({"unobserved-os-arch"});
  ExpectMismatch(changed);
  changed = document;
  changed["dependencies"][2]["version"] = "0.0.0";
  ExpectMismatch(changed);
  changed = document;
  changed["dynamic_shapes"] = false;
  ExpectMismatch(changed);
}

TEST_F(OnnxEnvironmentTest, RequiresStrictSchemaTypesAndUniqueJsonFields) {
  std::vector<Json> invalid;
  for (const auto& version :
       std::vector<Json>{Json(1.0), Json(true), Json("1"), Json(2)}) {
    auto changed = document;
    changed["schema_version"] = version;
    invalid.push_back(std::move(changed));
  }
  auto changed = document;
  changed["provider"]["allow_cpu_fallback"] = 0;
  invalid.push_back(changed);
  changed = document;
  changed["provider"]["options"]["device_id"] = 0;
  invalid.push_back(changed);
  changed = document;
  changed["extra"] = "ignored-by-no-one";
  invalid.push_back(changed);
  changed = document;
  changed["dependencies"][0]["extra"] = "ignored-by-no-one";
  invalid.push_back(changed);
  changed = document;
  changed.erase("dtypes");
  invalid.push_back(changed);
  for (const auto& value : invalid)
    EXPECT_THROW(runtime::parse_environment_lock(value.dump()),
                 SerializationError);
  const auto duplicate = "{\"format\":\"duplicate\"," + encoded.substr(1);
  EXPECT_THROW(runtime::parse_environment_lock(duplicate), SerializationError);
  EXPECT_THROW(runtime::parse_environment_lock("{not-json}"),
               SerializationError);
}

TEST_F(OnnxEnvironmentTest,
       CanonicalizesDependencyOrderButRejectsDuplicatesAndBadText) {
  auto reversed = document;
  std::reverse(reversed["dependencies"].begin(),
               reversed["dependencies"].end());
  const auto bytes = reversed.dump();
  EXPECT_NO_THROW(runtime::verify_environment_lock(
      session, bytes, Digest(bytes), observed.runtime));
  EXPECT_EQ(runtime::serialize_environment_lock(
                runtime::parse_environment_lock(bytes)),
            encoded);
  auto changed = document;
  changed["dependencies"].push_back(changed["dependencies"][0]);
  EXPECT_THROW(runtime::parse_environment_lock(changed.dump()),
               SerializationError);
  changed = document;
  changed["devices"].push_back("cpu");
  EXPECT_THROW(runtime::parse_environment_lock(changed.dump()),
               SerializationError);
  changed = document;
  changed["provider"]["name"] = "CPU\nExecutionProvider";
  EXPECT_THROW(runtime::parse_environment_lock(changed.dump()),
               SerializationError);
  auto invalid = observed;
  invalid.provider.name.clear();
  EXPECT_THROW(runtime::serialize_environment_lock(invalid), InvalidArgument);
}

TEST_F(OnnxEnvironmentTest, BoundsPayloadAndCollectionsBeforeMetadataUse) {
  EXPECT_THROW(runtime::parse_environment_lock(
                   std::string(runtime::kEnvironmentLockMaxBytes + 1U, ' ')),
               ResourceExhausted);
  auto changed = document;
  changed["devices"] = Json::array();
  for (unsigned index = 0; index < 65U; ++index)
    changed["devices"].push_back("device-" + std::to_string(index));
  EXPECT_THROW(runtime::parse_environment_lock(changed.dump()),
               ResourceExhausted);
  Json nested = Json::array();
  for (unsigned depth = 0; depth < 12U; ++depth) nested = Json::array({nested});
  changed = document;
  changed["provider"]["options"]["deep"] = nested;
  EXPECT_THROW(runtime::parse_environment_lock(changed.dump()),
               ResourceExhausted);
  auto invalid = observed;
  invalid.provider.options["large"] = std::string(4097U, 'x');
  EXPECT_THROW(runtime::serialize_environment_lock(invalid), ResourceExhausted);
}
}  // namespace
