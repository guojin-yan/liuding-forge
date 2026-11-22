// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
#include <jyppx/liuding/deployment/migraphx_runtime/environment.hpp>

#include <gtest/gtest.h>
#include <limits>
#include <nlohmann/json.hpp>

namespace {
using namespace jyppx::liuding;
namespace runtime = jyppx::liuding::migraphx_runtime;
using Json = nlohmann::json;
runtime::EnvironmentLock Sample() {
  // Codec-only declarations; no claim that this synthetic identity was
  // observed.
  return {.adapter_version = "0.3.0",
          .libtorch_version = "2.14.0+rocm10.1.0",
          .os_arch = "linux-x86_64",
          .migraphx_header_version = "2.18.0",
          .hip_header_version = "7.16.26385",
          .migraphx_c = {.sha256 = std::string(64, '1'), .size_bytes = 1234},
          .hip_runtime = {.sha256 = std::string(64, '2'), .size_bytes = 5678},
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
Json Document() {
  return Json::parse(runtime::serialize_environment_lock(Sample()));
}
TEST(MIGraphXEnvironment, Roundtrip) {
  const auto encoded = runtime::serialize_environment_lock(Sample());
  EXPECT_EQ(runtime::serialize_environment_lock(
                runtime::parse_environment_lock(encoded)),
            encoded);
  const auto doc = Document();
  EXPECT_EQ(doc.at("format").get<std::string>(),
            runtime::kEnvironmentLockSchemaId);
  EXPECT_EQ(doc.at("schema_version"), 1);
  EXPECT_EQ(doc.at("program").at("target"), "gpu");
  EXPECT_EQ(doc.at("program").at("dtype"), "float32");
  EXPECT_EQ(doc.at("program").at("offload_copy"), true);
  EXPECT_EQ(doc.at("program").at("fast_math"), false);
  EXPECT_EQ(doc.at("program").at("exhaustive_tune"), false);
  const auto pretty = doc.dump(2);
  EXPECT_EQ(runtime::serialize_environment_lock(
                runtime::parse_environment_lock(pretty)),
            encoded);
}
TEST(MIGraphXEnvironment, RequiresExactSchema) {
  const auto doc = Document();
  for (const auto& [key, value] : doc.items()) {
    auto changed = doc;
    changed.erase(key);
    EXPECT_THROW(runtime::parse_environment_lock(changed.dump()),
                 SerializationError)
        << key;
  }
  for (const auto* pointer :
       {"", "/build", "/sdk_headers", "/modules", "/modules/migraphx_c",
        "/modules/hip_runtime", "/hip", "/program"}) {
    auto changed = doc;
    changed[Json::json_pointer(pointer)]["unknown"] = "rejected";
    EXPECT_THROW(runtime::parse_environment_lock(changed.dump()),
                 SerializationError)
        << pointer;
  }
  const auto encoded = doc.dump();
  EXPECT_THROW(runtime::parse_environment_lock("{\"format\":\"duplicate\"," +
                                               encoded.substr(1)),
               SerializationError);
  auto nested_duplicate = encoded;
  const auto start = nested_duplicate.find("\"build\":{");
  ASSERT_NE(start, std::string::npos);
  nested_duplicate.insert(start + std::string("\"build\":{").size(),
                          "\"adapter_version\":\"duplicate\",");
  EXPECT_THROW(runtime::parse_environment_lock(nested_duplicate),
               SerializationError);
  for (const auto* text : {"", "null", "[]", "{broken}", "{}{}"})
    EXPECT_THROW(runtime::parse_environment_lock(text), SerializationError);
}
TEST(MIGraphXEnvironment, RequiresExactTypes) {
  const auto doc = Document();
  for (const auto* pointer :
       {"/schema_version", "/hip/device_id", "/hip/runtime_version",
        "/hip/driver_version", "/modules/migraphx_c/size_bytes",
        "/program/input_shape/0"}) {
    for (const auto& invalid : std::vector<Json>{true, 1.0, "1", -1, nullptr}) {
      auto changed = doc;
      changed[Json::json_pointer(pointer)] = invalid;
      EXPECT_THROW(runtime::parse_environment_lock(changed.dump()),
                   SerializationError)
          << pointer;
    }
  }
  for (const auto* pointer :
       {"/format", "/build/adapter_version", "/hip/architecture", "/hip/uuid",
        "/program/input_name", "/modules/hip_runtime/sha256"}) {
    auto changed = doc;
    changed[Json::json_pointer(pointer)] = 123;
    EXPECT_THROW(runtime::parse_environment_lock(changed.dump()),
                 SerializationError);
  }
  for (const auto* pointer : {"/program/offload_copy", "/program/fast_math",
                              "/program/exhaustive_tune"}) {
    for (const auto& invalid : std::vector<Json>{0, "false", nullptr}) {
      auto changed = doc;
      changed[Json::json_pointer(pointer)] = invalid;
      EXPECT_THROW(runtime::parse_environment_lock(changed.dump()),
                   SerializationError);
    }
    auto changed = doc;
    changed[Json::json_pointer(pointer)] =
        !changed[Json::json_pointer(pointer)].get<bool>();
    EXPECT_THROW(runtime::parse_environment_lock(changed.dump()),
                 SerializationError);
  }
  for (const auto& [pointer, value] : std::vector<std::pair<std::string, Json>>{
           {"/schema_version", 0},
           {"/schema_version", 2},
           {"/program/target", "ref"},
           {"/program/dtype", "float16"},
           {"/program/input_shape", 2},
           {"/hip/device_id", std::numeric_limits<std::uint64_t>::max()}}) {
    auto changed = doc;
    changed[Json::json_pointer(pointer)] = value;
    EXPECT_THROW(runtime::parse_environment_lock(changed.dump()),
                 SerializationError);
  }
}
TEST(MIGraphXEnvironment, RejectsInvalidMetadata) {
  const auto doc = Document();
  for (const auto& [pointer, value] : std::vector<std::pair<std::string, Json>>{
           {"/build/adapter_version", ""},
           {"/build/os_arch", "line\nbreak"},
           {"/program/input_name", std::string("in\0put", 6)},
           {"/hip/uuid", std::string(32, '0')},
           {"/hip/uuid", std::string(32, 'G')},
           {"/hip/runtime_version", 0},
           {"/hip/driver_version", 0},
           {"/modules/migraphx_c/sha256", std::string(64, 'A')},
           {"/modules/hip_runtime/sha256", "bad"},
           {"/modules/hip_runtime/size_bytes", 0},
           {"/program/input_shape", Json::array()},
           {"/program/output_shape", Json::array({2, 0})},
           {"/program/input_shape",
            Json::array({std::numeric_limits<std::uint64_t>::max(), 2})}}) {
    auto changed = doc;
    changed[Json::json_pointer(pointer)] = value;
    EXPECT_THROW(runtime::parse_environment_lock(changed.dump()),
                 SerializationError)
        << pointer;
  }
  auto invalid = Sample();
  invalid.device_id = -1;
  EXPECT_THROW(runtime::serialize_environment_lock(invalid), InvalidArgument);
  invalid = Sample();
  invalid.input_shape = {std::numeric_limits<std::size_t>::max(), 2};
  EXPECT_THROW(runtime::serialize_environment_lock(invalid), InvalidArgument);
}
TEST(MIGraphXEnvironment, BoundsEncodedMetadata) {
  EXPECT_THROW(runtime::parse_environment_lock(
                   std::string(runtime::kEnvironmentLockMaxBytes + 1, ' ')),
               ResourceExhausted);
  for (const auto* pointer :
       {"/program/input_shape", "/program/output_shape"}) {
    auto doc = Document();
    doc[Json::json_pointer(pointer)] = std::vector<int>(9, 1);
    EXPECT_THROW(runtime::parse_environment_lock(doc.dump()),
                 ResourceExhausted);
  }
  auto doc = Document();
  doc["modules"]["hip_runtime"]["size_bytes"] =
      runtime::kEnvironmentModuleMaxBytes + 1;
  EXPECT_THROW(runtime::parse_environment_lock(doc.dump()), ResourceExhausted);
  Json nested = Json::array();
  for (unsigned i = 0; i < 10; ++i) nested = Json::array({nested});
  EXPECT_THROW(runtime::parse_environment_lock(nested.dump()),
               ResourceExhausted);
  EXPECT_THROW(
      runtime::parse_environment_lock(Json(std::vector<int>(600, 1)).dump()),
      ResourceExhausted);
  auto invalid = Sample();
  invalid.input_name.assign(256, 'x');
  EXPECT_THROW(runtime::serialize_environment_lock(invalid), ResourceExhausted);
  invalid = Sample();
  invalid.migraphx_c.size_bytes = runtime::kEnvironmentModuleMaxBytes + 1;
  EXPECT_THROW(runtime::serialize_environment_lock(invalid), ResourceExhausted);
}
}  // namespace
