// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/experiment/config.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using jyppx::liuding::experiment::config::ConfigResolver;
using jyppx::liuding::experiment::config::ConfigSchema;
using jyppx::liuding::experiment::config::ConfigSourceKind;

class TempJsonFile {
 public:
  explicit TempJsonFile(std::string content) {
    static std::atomic<std::uint64_t> counter{0};
    path_ = std::filesystem::temp_directory_path() /
            ("jyppx-config-" + std::to_string(++counter) + ".json");
    std::ofstream output(path_, std::ios::binary | std::ios::trunc);
    output.write(content.data(), static_cast<std::streamsize>(content.size()));
    output.close();
    if (!output) throw std::runtime_error("could not create config fixture");
  }

  ~TempJsonFile() {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  TempJsonFile(const TempJsonFile&) = delete;
  TempJsonFile& operator=(const TempJsonFile&) = delete;

  const std::filesystem::path& path() const noexcept { return path_; }

 private:
  std::filesystem::path path_;
};

jyppx::liuding::ValueSchema Scalar(jyppx::liuding::ValueKind kind, bool required = true,
                          bool nullable = false) {
  return jyppx::liuding::ValueSchema{
      .kind = kind, .required = required, .nullable = nullable};
}

c10::impl::GenericDict Dictionary() {
  return c10::impl::GenericDict(c10::StringType::get(), c10::AnyType::get());
}

ConfigSchema MakeSchema() {
  auto trainer = jyppx::liuding::ValueSchema::dictionary(
      {{"enabled", Scalar(jyppx::liuding::ValueKind::boolean)},
       {"epochs", Scalar(jyppx::liuding::ValueKind::integer)},
       {"learning_rate", Scalar(jyppx::liuding::ValueKind::floating)},
       {"tags", jyppx::liuding::ValueSchema::list(Scalar(jyppx::liuding::ValueKind::string))}});
  auto root = jyppx::liuding::ValueSchema::dictionary(
      {{"credentials", jyppx::liuding::ValueSchema::dictionary(
                           {{"token", Scalar(jyppx::liuding::ValueKind::string)}})},
       {"model", jyppx::liuding::ValueSchema::dictionary(
                     {{"name", Scalar(jyppx::liuding::ValueKind::string)}})},
       {"note", Scalar(jyppx::liuding::ValueKind::string, false, true)},
       {"trainer", std::move(trainer)}});

  auto trainer_defaults = Dictionary();
  trainer_defaults.insert("enabled", true);
  trainer_defaults.insert("epochs", std::int64_t{1});
  trainer_defaults.insert("learning_rate", 0.1);
  c10::impl::GenericList tags(c10::AnyType::get());
  tags.push_back("default");
  trainer_defaults.insert("tags", tags);

  auto credentials = Dictionary();
  credentials.insert("token", "secret://env/training/token?version=1");
  auto model = Dictionary();
  model.insert("name", "default-model");
  auto defaults = Dictionary();
  defaults.insert("credentials", credentials);
  defaults.insert("model", model);
  defaults.insert("trainer", trainer_defaults);

  return {.schema_id = "experiment.training",
          .version = 1,
          .root = std::move(root),
          .defaults = c10::IValue(defaults),
          .environment_bindings = {{"/model/name", "JYPPX_MODEL"},
                                   {"/trainer/epochs", "JYPPX_EPOCHS"}},
          .secret_paths = {"/credentials/token"}};
}

TEST(ConfigTest, ResolvesFixedPrecedenceAndExplainsEveryContribution) {
  TempJsonFile file(
      R"({"model":{"name":"file-model"},"trainer":{"epochs":2,"learning_rate":0.2,"tags":["file"]}})");
  ConfigResolver resolver(MakeSchema());
  resolver.add_cli_override("/trainer/epochs", "4");
  resolver.add_environment(
      {{"JYPPX_EPOCHS", "3"}, {"JYPPX_MODEL", "环境模型"}});
  resolver.add_json_file(file.path());
  resolver.add_cli_override("/trainer/learning_rate", "0.25");

  const auto resolved = resolver.resolve();
  EXPECT_EQ(resolved.integer("/trainer/epochs"), 4);
  EXPECT_DOUBLE_EQ(resolved.floating("/trainer/learning_rate"), 0.25);
  EXPECT_EQ(resolved.string("/model/name"), "环境模型");
  EXPECT_TRUE(resolved.boolean("/trainer/enabled"));

  const auto origins = resolved.origins("/trainer/epochs");
  ASSERT_EQ(origins.size(), 4U);
  EXPECT_EQ(origins[0].kind, ConfigSourceKind::schema_default);
  EXPECT_EQ(origins[1].kind, ConfigSourceKind::json_file);
  EXPECT_EQ(origins[2].kind, ConfigSourceKind::environment);
  EXPECT_EQ(origins[3].kind, ConfigSourceKind::command_line);
  EXPECT_EQ(resolved.explain("/trainer/epochs").display_value, "4");
  EXPECT_NE(resolved.explain_json("/trainer/epochs").find("command_line"),
            std::string::npos);
}

TEST(ConfigTest, DeepMergesObjectsAndReplacesLists) {
  TempJsonFile first(R"({"trainer":{"epochs":2,"tags":["first","second"]}})");
  TempJsonFile second(R"({"trainer":{"tags":["replacement"]}})");
  ConfigResolver resolver(MakeSchema());
  resolver.add_json_file(first.path());
  resolver.add_json_file(second.path());
  const auto resolved = resolver.resolve();

  EXPECT_EQ(resolved.integer("/trainer/epochs"), 2);
  const auto tags_value = resolved.at("/trainer/tags");
  const auto tags = tags_value.toListRef();
  ASSERT_EQ(tags.size(), 1U);
  EXPECT_EQ(tags[0].toStringRef(), "replacement");
  const auto tag_origins = resolved.origins("/trainer/tags/0");
  ASSERT_EQ(tag_origins.size(), 1U);
  EXPECT_EQ(tag_origins.front().kind, ConfigSourceKind::json_file);
}

TEST(ConfigTest, ParsesStructuredCliFormsInArgumentOrder) {
  ConfigResolver resolver(MakeSchema());
  const std::vector<std::string_view> arguments{
      "--set", "/trainer/epochs=5", "--set=/model/name=\"cli-model\"",
      "--set=/trainer/epochs=6"};
  resolver.add_cli(arguments);
  const auto resolved = resolver.resolve();
  EXPECT_EQ(resolved.integer("/trainer/epochs"), 6);
  EXPECT_EQ(resolved.string("/model/name"), "cli-model");
  EXPECT_EQ(resolved.origins("/trainer/epochs").size(), 3U);
}

TEST(ConfigTest, RejectsUnknownAndDuplicateJsonFields) {
  TempJsonFile unknown(R"({"trainer":{"epochz":2}})");
  ConfigResolver unknown_resolver(MakeSchema());
  unknown_resolver.add_json_file(unknown.path());
  try {
    (void)unknown_resolver.resolve();
    FAIL() << "unknown field was accepted";
  } catch (const jyppx::liuding::InvalidArgument& error) {
    EXPECT_EQ(error.code(), jyppx::liuding::ErrorCode::invalid_argument);
    EXPECT_NE(error.details().operation.find("epochz"), std::string::npos);
  }

  TempJsonFile duplicate(R"({"trainer":{"epochs":2,"epochs":3}})");
  ConfigResolver duplicate_resolver(MakeSchema());
  duplicate_resolver.add_json_file(duplicate.path());
  EXPECT_THROW((void)duplicate_resolver.resolve(), jyppx::liuding::SerializationError);
}

TEST(ConfigTest, RejectsMissingRequiredFieldsAfterAllSources) {
  auto schema = MakeSchema();
  auto defaults = schema.defaults.toGenericDict();
  (void)defaults.erase("model");
  schema.defaults = defaults;
  ConfigResolver resolver(std::move(schema));
  EXPECT_THROW((void)resolver.resolve(), jyppx::liuding::InvalidArgument);
}

TEST(ConfigTest, RejectsSchemaDefaultsAndOpenObjectsOutsideContract) {
  auto open_schema = MakeSchema();
  open_schema.root.allow_extra_fields = true;
  EXPECT_THROW(ConfigResolver(std::move(open_schema)), jyppx::liuding::InvalidArgument);

  auto wrong_default = MakeSchema();
  auto defaults = wrong_default.defaults.toGenericDict();
  auto trainer = defaults.at("trainer").toGenericDict();
  trainer.insert_or_assign("epochs", "many");
  defaults.insert_or_assign("trainer", trainer);
  wrong_default.defaults = defaults;
  EXPECT_THROW(ConfigResolver(std::move(wrong_default)),
               jyppx::liuding::InvalidArgument);

  auto secret_default = MakeSchema();
  auto secret_defaults = secret_default.defaults.toGenericDict();
  auto credentials = secret_defaults.at("credentials").toGenericDict();
  credentials.insert_or_assign("token", "plain-default-secret");
  secret_defaults.insert_or_assign("credentials", credentials);
  secret_default.defaults = secret_defaults;
  EXPECT_THROW(ConfigResolver(std::move(secret_default)),
               jyppx::liuding::InvalidArgument);
}

TEST(ConfigTest, ValidatesSecretReferencesAndRedactsEveryDisplaySurface) {
  ConfigResolver resolver(MakeSchema());
  resolver.add_cli_override(
      "/credentials/token",
      R"("secret://vault/team/service-token?version=blue")");
  const auto resolved = resolver.resolve();
  EXPECT_EQ(resolved.string("/credentials/token"),
            "secret://vault/team/service-token?version=blue");
  const auto explanation = resolved.explain("/credentials/token");
  EXPECT_TRUE(explanation.redacted);
  EXPECT_EQ(explanation.display_value, "<redacted>");
  EXPECT_EQ(resolved.canonical_json().find("service-token"), std::string::npos);
  EXPECT_EQ(resolved.explain_json("/credentials/token").find("service-token"),
            std::string::npos);
  const auto parent_explanation = resolved.explain("/credentials");
  EXPECT_TRUE(parent_explanation.redacted);
  EXPECT_EQ(parent_explanation.display_value.find("service-token"),
            std::string::npos);
  const auto root_explanation = resolved.explain("");
  EXPECT_TRUE(root_explanation.redacted);
  EXPECT_EQ(root_explanation.display_value.find("service-token"),
            std::string::npos);
  EXPECT_EQ(resolved.explain_json("").find("service-token"), std::string::npos);

  ConfigResolver invalid(MakeSchema());
  invalid.add_cli_override("/credentials/token",
                           R"("plain-text-secret-must-not-leak")");
  try {
    (void)invalid.resolve();
    FAIL() << "plain-text secret was accepted";
  } catch (const jyppx::liuding::InvalidArgument& error) {
    EXPECT_EQ(std::string(error.what()).find("plain-text-secret-must-not-leak"),
              std::string::npos);
    EXPECT_EQ(error.details().actual, "non-reference secret text");
  }
  EXPECT_TRUE(jyppx::liuding::experiment::config::is_secret_reference(
      "secret://env/training/token"));
  EXPECT_FALSE(jyppx::liuding::experiment::config::is_secret_reference(
      "secret://missing-segments"));
}

TEST(ConfigTest, SerializesDeterministicallyWithSortedUtf8Objects) {
  TempJsonFile file(
      R"({"trainer":{"learning_rate":0.5,"epochs":9},"model":{"name":"模型"}})");
  ConfigResolver first(MakeSchema());
  first.add_json_file(file.path());
  ConfigResolver second(MakeSchema());
  second.add_cli_override("/model/name", R"("模型")");
  second.add_cli_override("/trainer/epochs", "9");
  second.add_cli_override("/trainer/learning_rate", "0.5");
  const auto first_json = first.resolve().canonical_json();
  const auto second_json = second.resolve().canonical_json();
  EXPECT_EQ(first_json, second_json);
  EXPECT_EQ(
      first_json,
      R"({"credentials":{"token":"<redacted>"},"model":{"name":"模型"},"trainer":{"enabled":true,"epochs":9,"learning_rate":0.5,"tags":["default"]}})");
}

TEST(ConfigTest, ReturnsDetachedValuesInsteadOfMutableAliases) {
  auto input_schema = MakeSchema();
  ConfigResolver resolver(input_schema);
  auto input_defaults = input_schema.defaults.toGenericDict();
  auto input_trainer = input_defaults.at("trainer").toGenericDict();
  input_trainer.insert_or_assign("epochs", std::int64_t{98});
  input_defaults.insert_or_assign("trainer", input_trainer);

  const auto resolved = resolver.resolve();
  auto detached = resolved.value().toGenericDict();
  auto trainer = detached.at("trainer").toGenericDict();
  trainer.insert_or_assign("epochs", std::int64_t{99});
  detached.insert_or_assign("trainer", trainer);

  auto detached_schema = resolved.schema();
  auto detached_defaults = detached_schema.defaults.toGenericDict();
  auto detached_trainer = detached_defaults.at("trainer").toGenericDict();
  detached_trainer.insert_or_assign("epochs", std::int64_t{100});
  detached_defaults.insert_or_assign("trainer", detached_trainer);
  EXPECT_EQ(resolved.integer("/trainer/epochs"), 1);
  const auto preserved_schema = resolved.schema();
  const auto preserved_defaults = preserved_schema.defaults.toGenericDict();
  const auto preserved_trainer =
      preserved_defaults.at("trainer").toGenericDict();
  EXPECT_EQ(preserved_trainer.at("epochs").toInt(), 1);
}

TEST(ConfigTest, RejectsInvalidUtf8AndBoundedSources) {
  TempJsonFile invalid_utf8(std::string("{\"model\":\"") +
                            static_cast<char>(0xC3) + "(\"}");
  ConfigResolver invalid_resolver(MakeSchema());
  invalid_resolver.add_json_file(invalid_utf8.path());
  EXPECT_THROW((void)invalid_resolver.resolve(), jyppx::liuding::InvalidArgument);

  TempJsonFile oversized(R"({"trainer":{"epochs":2}})");
  auto limits = jyppx::liuding::experiment::config::ConfigLimits{};
  limits.max_file_bytes = 8;
  ConfigResolver size_resolver(MakeSchema(), limits);
  size_resolver.add_json_file(oversized.path());
  EXPECT_THROW((void)size_resolver.resolve(), jyppx::liuding::ResourceExhausted);

  limits = {};
  limits.max_nodes = 2;
  EXPECT_THROW(ConfigResolver(MakeSchema(), limits), jyppx::liuding::ResourceExhausted);
}

TEST(ConfigTest, RejectsMalformedCliAndDuplicateEnvironmentSnapshots) {
  ConfigResolver resolver(MakeSchema());
  EXPECT_THROW(resolver.add_cli_override("/trainer/unknown", "1"),
               jyppx::liuding::InvalidArgument);
  const std::vector<std::string_view> malformed{"--other", "x"};
  EXPECT_THROW(resolver.add_cli(malformed), jyppx::liuding::InvalidArgument);
  resolver.add_environment({});
  EXPECT_THROW(resolver.add_environment({}), jyppx::liuding::InvalidArgument);
}

TEST(ConfigTest, ExposesStructuredConfigCapabilityWithoutNewManifestTypes) {
  const auto descriptor =
      jyppx::liuding::experiment::config::make_config_capability(MakeSchema());
  EXPECT_EQ(descriptor.feature_id, "experiment.config");
  EXPECT_EQ(descriptor.scope_id, "experiment.training");
  EXPECT_EQ(descriptor.source, jyppx::liuding::CapabilitySource::configured_instance);
  EXPECT_EQ(descriptor.status, jyppx::liuding::CapabilityStatus::experimental);
  EXPECT_EQ(descriptor.backend, "nlohmann-json");
  EXPECT_NE(std::find(descriptor.operations.begin(),
                      descriptor.operations.end(), "explain"),
            descriptor.operations.end());
  EXPECT_NO_THROW(jyppx::liuding::validate_capability_descriptor(descriptor));
}

TEST(ConfigTest, CoversStableSourceKindNames) {
  EXPECT_EQ(jyppx::liuding::experiment::config::config_source_kind_name(
                ConfigSourceKind::schema_default),
            "schema_default");
  EXPECT_EQ(jyppx::liuding::experiment::config::config_source_kind_name(
                ConfigSourceKind::json_file),
            "json_file");
  EXPECT_EQ(jyppx::liuding::experiment::config::config_source_kind_name(
                ConfigSourceKind::environment),
            "environment");
  EXPECT_EQ(jyppx::liuding::experiment::config::config_source_kind_name(
                ConfigSourceKind::command_line),
            "command_line");
  EXPECT_EQ(jyppx::liuding::experiment::config::config_source_kind_name(
                static_cast<ConfigSourceKind>(255)),
            "unknown");
}

}  // namespace
