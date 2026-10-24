// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data/batch_schema.hpp>
#include <jyppx/liuding/data/data_pipeline.hpp>
#include <jyppx/liuding/data/samplers/sequential_sampler.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <utility>
#include <vector>

namespace {

/** @brief Creates a scalar schema for one integer-valued batch. */
jyppx::liuding::data::BatchSchema integer_batch_schema(std::string id) {
  return {.id = std::move(id),
          .version = 1,
          .value = jyppx::liuding::ValueSchema{
              .name = "count",
              .semantic_tag = "sample_count",
              .kind = jyppx::liuding::ValueKind::integer}};
}

/** @brief Creates a one-dimensional float32 tensor batch schema. */
jyppx::liuding::data::BatchSchema tensor_batch_schema(std::string id) {
  jyppx::liuding::TensorSchema tensor;
  tensor.dtype = torch::kFloat32;
  tensor.device = jyppx::liuding::Device(torch::kCPU);
  tensor.rank = 1;
  tensor.shape = std::vector<std::optional<std::int64_t>>{std::nullopt};
  return jyppx::liuding::data::BatchSchema::tensor(
      std::move(id), std::move(tensor), 1, "features");
}

TEST(ValueSchemaTest,
     ValidatesFieldsTensorConstraintsOptionalValuesAndSemanticTags) {
  jyppx::liuding::TensorSchema features;
  features.dtype = torch::kFloat32;
  features.device = jyppx::liuding::Device(torch::kCPU);
  features.rank = 2;
  features.shape = std::vector<std::optional<std::int64_t>>{std::nullopt, 4};

  auto note =
      jyppx::liuding::ValueSchema{.name = "note",
                                  .semantic_tag = "annotation",
                                  .kind = jyppx::liuding::ValueKind::string,
                                  .required = false,
                                  .nullable = true};
  const auto schema = jyppx::liuding::data::BatchSchema{
      .id = "training.supervised_batch",
      .version = 3,
      .value = jyppx::liuding::ValueSchema::dictionary(
          {{"features", jyppx::liuding::ValueSchema::tensor(
                            features, "features", "model_input")},
           {"target",
            jyppx::liuding::ValueSchema{
                .name = "target",
                .semantic_tag = "class_index",
                .kind = jyppx::liuding::ValueKind::integer}},
           {"note", note}},
          false, "supervised_batch", "training_batch")};

  c10::impl::GenericDict value(c10::StringType::get(), c10::AnyType::get());
  value.insert("features", torch::ones({2, 4}, torch::kFloat32));
  value.insert("target", std::int64_t{1});

  EXPECT_NO_THROW(jyppx::liuding::data::validate_batch_schema(schema));
  EXPECT_NO_THROW(
      jyppx::liuding::data::validate_batch_value(c10::IValue(value), schema));
  EXPECT_EQ(jyppx::liuding::data::batch_schema_fingerprint(schema).size(), 16U);

  value.insert("extra", std::int64_t{9});
  EXPECT_THROW(
      jyppx::liuding::data::validate_batch_value(c10::IValue(value), schema),
      jyppx::liuding::InvalidArgument);
}

TEST(ValueSchemaTest, CanonicalDescriptorCoversIdentityVersionAndSemantics) {
  auto original = tensor_batch_schema("training.features");
  auto changed_tag = original;
  changed_tag.value.semantic_tag = "target";
  auto changed_version = original;
  changed_version.version = 2;

  EXPECT_EQ(jyppx::liuding::data::canonical_batch_schema(original),
            jyppx::liuding::data::canonical_batch_schema(original));
  EXPECT_NE(jyppx::liuding::data::canonical_batch_schema(original),
            jyppx::liuding::data::canonical_batch_schema(changed_tag));
  EXPECT_NE(jyppx::liuding::data::canonical_batch_schema(original),
            jyppx::liuding::data::canonical_batch_schema(changed_version));
  EXPECT_NE(jyppx::liuding::data::batch_schema_fingerprint(original),
            jyppx::liuding::data::batch_schema_fingerprint(changed_tag));
}

TEST(ValueSchemaTest, RejectsUnboundedOrMalformedSchemaAndRuntimeTrees) {
  EXPECT_THROW(
      jyppx::liuding::data::validate_batch_schema(
          {.id = "", .version = 1, .value = jyppx::liuding::ValueSchema{}}),
      jyppx::liuding::InvalidArgument);
  EXPECT_THROW(jyppx::liuding::data::validate_batch_schema(
                   {.id = "schema",
                    .version = 0,
                    .value = jyppx::liuding::ValueSchema{}}),
               jyppx::liuding::InvalidArgument);

  auto nested =
      jyppx::liuding::ValueSchema::list(jyppx::liuding::ValueSchema::list(
          jyppx::liuding::ValueSchema::list(jyppx::liuding::ValueSchema{
              .kind = jyppx::liuding::ValueKind::integer})));
  EXPECT_THROW(
      jyppx::liuding::validate_value_schema(nested, "batch",
                                            {.max_depth = 2,
                                             .max_schema_nodes = 16,
                                             .max_metadata_bytes = 1024,
                                             .max_runtime_nodes = 16}),
      jyppx::liuding::ResourceExhausted);

  c10::impl::GenericList values(c10::IntType::get());
  values.push_back(std::int64_t{1});
  values.push_back(std::int64_t{2});
  values.push_back(std::int64_t{3});
  const auto list_schema = jyppx::liuding::ValueSchema::list(
      jyppx::liuding::ValueSchema{.kind = jyppx::liuding::ValueKind::integer});
  EXPECT_THROW(
      jyppx::liuding::validate_value(c10::IValue(values), list_schema, "batch",
                                     {.max_depth = 8,
                                      .max_schema_nodes = 16,
                                      .max_metadata_bytes = 1024,
                                      .max_runtime_nodes = 3}),
      jyppx::liuding::ResourceExhausted);

  c10::impl::GenericDict keyed_value(c10::StringType::get(),
                                     c10::AnyType::get());
  keyed_value.insert("oversized-key", std::int64_t{1});
  const auto open_dictionary =
      jyppx::liuding::ValueSchema::dictionary({}, true);
  EXPECT_THROW(jyppx::liuding::validate_value(c10::IValue(keyed_value),
                                              open_dictionary, "batch",
                                              {.max_depth = 8,
                                               .max_schema_nodes = 16,
                                               .max_metadata_bytes = 8,
                                               .max_runtime_nodes = 16}),
               jyppx::liuding::ResourceExhausted);

  jyppx::liuding::TensorSchema excessive_shape;
  excessive_shape.shape =
      std::vector<std::optional<std::int64_t>>(17, std::nullopt);
  EXPECT_THROW(
      jyppx::liuding::validate_value_schema(
          jyppx::liuding::ValueSchema::tensor(std::move(excessive_shape)),
          "value",
          {.max_depth = 8,
           .max_schema_nodes = 16,
           .max_metadata_bytes = 1024,
           .max_runtime_nodes = 16}),
      jyppx::liuding::ResourceExhausted);
}

TEST(ValueSchemaTest, CoversEveryStableKindNameAndRuntimeCategory) {
  EXPECT_EQ(jyppx::liuding::value_kind_name(jyppx::liuding::ValueKind::any),
            "any");
  EXPECT_EQ(jyppx::liuding::value_kind_name(jyppx::liuding::ValueKind::tensor),
            "tensor");
  EXPECT_EQ(jyppx::liuding::value_kind_name(jyppx::liuding::ValueKind::tuple),
            "tuple");
  EXPECT_EQ(jyppx::liuding::value_kind_name(jyppx::liuding::ValueKind::list),
            "list");
  EXPECT_EQ(
      jyppx::liuding::value_kind_name(jyppx::liuding::ValueKind::dictionary),
      "dictionary");
  EXPECT_EQ(jyppx::liuding::value_kind_name(jyppx::liuding::ValueKind::integer),
            "integer");
  EXPECT_EQ(
      jyppx::liuding::value_kind_name(jyppx::liuding::ValueKind::floating),
      "floating");
  EXPECT_EQ(jyppx::liuding::value_kind_name(jyppx::liuding::ValueKind::boolean),
            "boolean");
  EXPECT_EQ(jyppx::liuding::value_kind_name(jyppx::liuding::ValueKind::string),
            "string");
  EXPECT_EQ(jyppx::liuding::value_kind_name(jyppx::liuding::ValueKind::none),
            "none");
  EXPECT_EQ(jyppx::liuding::value_kind_name(jyppx::liuding::ValueKind::device),
            "device");
  EXPECT_EQ(jyppx::liuding::value_kind_name(
                static_cast<jyppx::liuding::ValueKind>(255)),
            "unknown");

  EXPECT_NO_THROW(jyppx::liuding::validate_value(
      c10::IValue(std::int64_t{7}),
      jyppx::liuding::ValueSchema{.kind = jyppx::liuding::ValueKind::integer}));
  EXPECT_NO_THROW(jyppx::liuding::validate_value(
      c10::IValue(1.25), jyppx::liuding::ValueSchema{
                             .kind = jyppx::liuding::ValueKind::floating}));
  EXPECT_NO_THROW(jyppx::liuding::validate_value(
      c10::IValue(true),
      jyppx::liuding::ValueSchema{.kind = jyppx::liuding::ValueKind::boolean}));
  EXPECT_NO_THROW(jyppx::liuding::validate_value(
      c10::IValue(std::string("annotation")),
      jyppx::liuding::ValueSchema{.kind = jyppx::liuding::ValueKind::string}));
  EXPECT_NO_THROW(jyppx::liuding::validate_value(
      c10::IValue(),
      jyppx::liuding::ValueSchema{.kind = jyppx::liuding::ValueKind::none}));
  EXPECT_NO_THROW(jyppx::liuding::validate_value(
      c10::IValue(c10::Device(c10::kCPU)),
      jyppx::liuding::ValueSchema{.kind = jyppx::liuding::ValueKind::device}));
  EXPECT_NO_THROW(jyppx::liuding::validate_value(
      c10::IValue(),
      jyppx::liuding::ValueSchema{.kind = jyppx::liuding::ValueKind::integer,
                                  .nullable = true}));

  c10::impl::GenericList list(c10::IntType::get());
  list.push_back(std::int64_t{3});
  list.push_back(std::int64_t{5});
  const auto list_schema = jyppx::liuding::ValueSchema::list(
      jyppx::liuding::ValueSchema{.kind = jyppx::liuding::ValueKind::integer});
  EXPECT_NO_THROW(
      jyppx::liuding::validate_value(c10::IValue(list), list_schema));

  const auto tuple_schema = jyppx::liuding::ValueSchema::tuple(
      {jyppx::liuding::ValueSchema{.kind = jyppx::liuding::ValueKind::integer},
       jyppx::liuding::ValueSchema{.kind =
                                       jyppx::liuding::ValueKind::floating}});
  const auto tuple = c10::ivalue::Tuple::create(
      {c10::IValue(std::int64_t{2}), c10::IValue(0.5)});
  EXPECT_NO_THROW(
      jyppx::liuding::validate_value(c10::IValue(tuple), tuple_schema));
  EXPECT_THROW(
      jyppx::liuding::validate_value(c10::IValue(c10::ivalue::Tuple::create(
                                         {c10::IValue(std::int64_t{2})})),
                                     tuple_schema),
      jyppx::liuding::InvalidArgument);

  EXPECT_THROW(
      jyppx::liuding::validate_value(
          c10::IValue(false),
          jyppx::liuding::ValueSchema{
              .name = "count", .kind = jyppx::liuding::ValueKind::integer}),
      jyppx::liuding::InvalidArgument);
}

TEST(ValueSchemaTest, RejectsEveryMalformedSchemaCombinationAndLimit) {
  auto limits = jyppx::liuding::ValueSchemaLimits{};
  limits.max_depth = 0;
  EXPECT_THROW(jyppx::liuding::validate_value_schema({}, "value", limits),
               jyppx::liuding::InvalidArgument);
  limits = {};
  limits.max_schema_nodes = 0;
  EXPECT_THROW(jyppx::liuding::validate_value_schema({}, "value", limits),
               jyppx::liuding::InvalidArgument);
  limits = {};
  limits.max_metadata_bytes = 0;
  EXPECT_THROW(jyppx::liuding::validate_value_schema({}, "value", limits),
               jyppx::liuding::InvalidArgument);
  limits = {};
  limits.max_runtime_nodes = 0;
  EXPECT_THROW(jyppx::liuding::validate_value_schema({}, "value", limits),
               jyppx::liuding::InvalidArgument);

  auto primitive =
      jyppx::liuding::ValueSchema{.kind = jyppx::liuding::ValueKind::integer};
  primitive.elements.push_back(jyppx::liuding::ValueSchema{});
  EXPECT_THROW(jyppx::liuding::validate_value_schema(primitive),
               jyppx::liuding::InvalidArgument);

  auto tensor_with_children =
      jyppx::liuding::ValueSchema::tensor({}, "invalid_tensor");
  tensor_with_children.fields.emplace("child", jyppx::liuding::ValueSchema{});
  EXPECT_THROW(jyppx::liuding::validate_value_schema(tensor_with_children),
               jyppx::liuding::InvalidArgument);
  auto tensor_with_extra_fields = jyppx::liuding::ValueSchema::tensor();
  tensor_with_extra_fields.allow_extra_fields = true;
  EXPECT_THROW(jyppx::liuding::validate_value_schema(tensor_with_extra_fields),
               jyppx::liuding::InvalidArgument);
  EXPECT_NO_THROW(jyppx::liuding::validate_value_schema(
      jyppx::liuding::ValueSchema{.kind = jyppx::liuding::ValueKind::tensor}));

  jyppx::liuding::TensorSchema negative_rank;
  negative_rank.rank = -1;
  EXPECT_THROW(jyppx::liuding::validate_value_schema(
                   jyppx::liuding::ValueSchema::tensor(negative_rank)),
               jyppx::liuding::InvalidArgument);

  jyppx::liuding::TensorSchema mismatched_rank;
  mismatched_rank.rank = 1;
  mismatched_rank.shape = std::vector<std::optional<std::int64_t>>{2, 3};
  EXPECT_THROW(jyppx::liuding::validate_value_schema(
                   jyppx::liuding::ValueSchema::tensor(mismatched_rank)),
               jyppx::liuding::InvalidArgument);

  jyppx::liuding::TensorSchema negative_extent;
  negative_extent.shape =
      std::vector<std::optional<std::int64_t>>{std::int64_t{-2}};
  EXPECT_THROW(jyppx::liuding::validate_value_schema(
                   jyppx::liuding::ValueSchema::tensor(negative_extent)),
               jyppx::liuding::InvalidArgument);

  jyppx::liuding::TensorSchema bounded_device;
  bounded_device.device = jyppx::liuding::Device(torch::kCUDA);
  EXPECT_THROW(jyppx::liuding::validate_value_schema(
                   jyppx::liuding::ValueSchema::tensor(bounded_device), "value",
                   {.max_depth = 8,
                    .max_schema_nodes = 16,
                    .max_metadata_bytes = 3,
                    .max_runtime_nodes = 16}),
               jyppx::liuding::ResourceExhausted);

  auto malformed_tuple = jyppx::liuding::ValueSchema::tuple({});
  malformed_tuple.allow_extra_fields = true;
  EXPECT_THROW(jyppx::liuding::validate_value_schema(malformed_tuple),
               jyppx::liuding::InvalidArgument);

  auto malformed_list =
      jyppx::liuding::ValueSchema{.kind = jyppx::liuding::ValueKind::list};
  EXPECT_THROW(jyppx::liuding::validate_value_schema(malformed_list),
               jyppx::liuding::InvalidArgument);
  malformed_list.elements = {jyppx::liuding::ValueSchema{},
                             jyppx::liuding::ValueSchema{}};
  EXPECT_THROW(jyppx::liuding::validate_value_schema(malformed_list),
               jyppx::liuding::InvalidArgument);

  auto malformed_dictionary = jyppx::liuding::ValueSchema::dictionary({});
  malformed_dictionary.elements.push_back(jyppx::liuding::ValueSchema{});
  EXPECT_THROW(jyppx::liuding::validate_value_schema(malformed_dictionary),
               jyppx::liuding::InvalidArgument);

  auto empty_key_dictionary = jyppx::liuding::ValueSchema::dictionary({});
  empty_key_dictionary.fields.emplace("", jyppx::liuding::ValueSchema{});
  EXPECT_THROW(jyppx::liuding::validate_value_schema(empty_key_dictionary),
               jyppx::liuding::InvalidArgument);

  auto invalid_kind = jyppx::liuding::ValueSchema{};
  invalid_kind.kind = static_cast<jyppx::liuding::ValueKind>(255);
  EXPECT_THROW(jyppx::liuding::validate_value_schema(invalid_kind),
               jyppx::liuding::InvalidArgument);

  EXPECT_THROW(jyppx::liuding::validate_value_schema(
                   jyppx::liuding::ValueSchema::any("oversized-name"), "value",
                   {.max_depth = 8,
                    .max_schema_nodes = 16,
                    .max_metadata_bytes = 4,
                    .max_runtime_nodes = 16}),
               jyppx::liuding::ResourceExhausted);

  const auto cumulative_metadata = jyppx::liuding::ValueSchema::dictionary(
      {{"field", jyppx::liuding::ValueSchema::any("child")}}, false, "root");
  EXPECT_THROW(
      jyppx::liuding::validate_value_schema(cumulative_metadata, "value",
                                            {.max_depth = 8,
                                             .max_schema_nodes = 16,
                                             .max_metadata_bytes = 8,
                                             .max_runtime_nodes = 16}),
      jyppx::liuding::ResourceExhausted);
}

TEST(ValueSchemaTest, ValidatesEveryTensorConstraintFailureAtomically) {
  EXPECT_THROW(
      jyppx::liuding::validate_value(c10::IValue(jyppx::liuding::Tensor()),
                                     jyppx::liuding::ValueSchema::tensor()),
      jyppx::liuding::InvalidArgument);

  jyppx::liuding::TensorSchema dtype;
  dtype.dtype = torch::kFloat32;
  EXPECT_THROW(jyppx::liuding::validate_value(
                   c10::IValue(torch::ones({2}, torch::kInt64)),
                   jyppx::liuding::ValueSchema::tensor(dtype)),
               jyppx::liuding::InvalidArgument);

  jyppx::liuding::TensorSchema device;
  device.device = jyppx::liuding::Device(torch::kCUDA);
  EXPECT_THROW(jyppx::liuding::validate_value(
                   c10::IValue(torch::ones({2}, torch::kFloat32)),
                   jyppx::liuding::ValueSchema::tensor(device)),
               jyppx::liuding::InvalidArgument);

  jyppx::liuding::TensorSchema rank;
  rank.rank = 2;
  EXPECT_THROW(jyppx::liuding::validate_value(
                   c10::IValue(torch::ones({2}, torch::kFloat32)),
                   jyppx::liuding::ValueSchema::tensor(rank)),
               jyppx::liuding::InvalidArgument);

  jyppx::liuding::TensorSchema shape_rank;
  shape_rank.shape = std::vector<std::optional<std::int64_t>>{2, std::nullopt};
  EXPECT_THROW(jyppx::liuding::validate_value(
                   c10::IValue(torch::ones({2}, torch::kFloat32)),
                   jyppx::liuding::ValueSchema::tensor(shape_rank)),
               jyppx::liuding::InvalidArgument);

  jyppx::liuding::TensorSchema extent;
  extent.shape = std::vector<std::optional<std::int64_t>>{3};
  EXPECT_THROW(jyppx::liuding::validate_value(
                   c10::IValue(torch::ones({2}, torch::kFloat32)),
                   jyppx::liuding::ValueSchema::tensor(extent)),
               jyppx::liuding::InvalidArgument);

  EXPECT_NO_THROW(jyppx::liuding::validate_value(
      c10::IValue(torch::ones({2}, torch::kFloat32)),
      jyppx::liuding::ValueSchema::tensor()));
}

TEST(ValueSchemaTest, TraversesAnyValuesAndOpenDictionaryExtrasWithinLimits) {
  c10::impl::GenericList list(c10::AnyType::get());
  list.push_back(std::int64_t{1});
  list.push_back(std::string("two"));

  c10::impl::GenericDict nested_dictionary(c10::StringType::get(),
                                           c10::AnyType::get());
  nested_dictionary.insert("items", c10::IValue(list));
  const auto nested = c10::IValue(c10::ivalue::Tuple::create(
      {c10::IValue(nested_dictionary), c10::IValue(list)}));

  EXPECT_NO_THROW(jyppx::liuding::validate_value(
      nested, jyppx::liuding::ValueSchema::any("payload", "untyped_payload")));
  EXPECT_THROW(jyppx::liuding::validate_value(
                   nested, jyppx::liuding::ValueSchema::any(), "value",
                   {.max_depth = 1,
                    .max_schema_nodes = 16,
                    .max_metadata_bytes = 1024,
                    .max_runtime_nodes = 32}),
               jyppx::liuding::ResourceExhausted);

  const auto open_schema = jyppx::liuding::ValueSchema::dictionary(
      {{"required",
        jyppx::liuding::ValueSchema{.kind =
                                        jyppx::liuding::ValueKind::integer}},
       {"optional",
        jyppx::liuding::ValueSchema{.kind = jyppx::liuding::ValueKind::string,
                                    .required = false}}},
      true, "open_payload");
  c10::impl::GenericDict open_value(c10::StringType::get(),
                                    c10::AnyType::get());
  open_value.insert("required", std::int64_t{4});
  open_value.insert("extra", nested);
  EXPECT_NO_THROW(
      jyppx::liuding::validate_value(c10::IValue(open_value), open_schema));

  c10::impl::GenericDict missing(c10::StringType::get(), c10::AnyType::get());
  EXPECT_THROW(
      jyppx::liuding::validate_value(c10::IValue(missing), open_schema),
      jyppx::liuding::InvalidArgument);

  c10::impl::GenericDict non_string_key(c10::IntType::get(),
                                        c10::AnyType::get());
  non_string_key.insert(std::int64_t{1}, std::int64_t{2});
  EXPECT_THROW(jyppx::liuding::validate_value(
                   c10::IValue(non_string_key),
                   jyppx::liuding::ValueSchema::dictionary({}, true)),
               jyppx::liuding::InvalidArgument);
}

TEST(ValueSchemaTest, DirectFingerprintIsStableAndCoversCompleteDescriptor) {
  const auto schema = jyppx::liuding::ValueSchema::tuple(
      {jyppx::liuding::ValueSchema::tensor({}, "features"),
       jyppx::liuding::ValueSchema::list(
           jyppx::liuding::ValueSchema{.kind =
                                           jyppx::liuding::ValueKind::integer},
           "labels")},
      "batch", "training_batch");
  const auto fingerprint = jyppx::liuding::value_schema_fingerprint(schema);
  EXPECT_EQ(fingerprint.size(), 16U);
  EXPECT_EQ(fingerprint, jyppx::liuding::value_schema_fingerprint(schema));

  auto changed = schema;
  changed.elements.front().semantic_tag = "model_input";
  EXPECT_NE(fingerprint, jyppx::liuding::value_schema_fingerprint(changed));
}

TEST(ValueSchemaPipelineTest,
     PublishesCompleteCapabilityAndValidatedEnvelopeSchema) {
  const auto schema = tensor_batch_schema("pipeline.features");
  jyppx::liuding::data::DataPipeline pipeline(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(4),
      {.pipeline_id = "data.schema.capability",
       .dataset_version = "dataset.schema/v1",
       .dataset_size = 4,
       .schema = schema,
       .batch_factory_id = "factory.schema.tensor/v1",
       .batch_size = 2},
      [](std::span<const std::size_t> indices,
         const jyppx::liuding::data::DataWorkerContext&, std::stop_token) {
        return c10::IValue(torch::ones(
            {static_cast<std::int64_t>(indices.size())}, torch::kFloat32));
      });

  const auto capability = pipeline.capability();
  EXPECT_TRUE(capability.map_style);
  EXPECT_TRUE(capability.finite_epoch);
  EXPECT_TRUE(capability.schema_validation);
  EXPECT_FALSE(capability.process_workers);
  EXPECT_EQ(capability.worker_model,
            jyppx::liuding::data::DataWorkerModel::inline_execution);
  EXPECT_EQ(capability.batch_schema.id, schema.id);
  EXPECT_EQ(capability.batch_schema_fingerprint,
            jyppx::liuding::data::batch_schema_fingerprint(schema));
  EXPECT_EQ(capability.max_batch_tensors, 4096U);
  EXPECT_EQ(capability.max_batch_value_nodes, 65536U);
  EXPECT_EQ(capability.max_batch_bytes, 1ULL << 30U);

  pipeline.set_epoch(0);
  const auto batch = pipeline.next();
  ASSERT_TRUE(batch.has_value());
  EXPECT_EQ(jyppx::liuding::data::canonical_batch_schema(batch->schema),
            jyppx::liuding::data::canonical_batch_schema(schema));
}

TEST(ValueSchemaPipelineTest,
     InlineAndWorkerValidationFailuresDoNotAdvanceCommittedCursor) {
  const auto make_invalid = [](std::string id, std::size_t workers) {
    return jyppx::liuding::data::DataPipeline(
        std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(4),
        {.pipeline_id = std::move(id),
         .dataset_version = "dataset.schema/v1",
         .dataset_size = 4,
         .schema = integer_batch_schema("pipeline.integer"),
         .batch_factory_id = "factory.schema.invalid/v1",
         .batch_size = 2,
         .num_workers = workers,
         .prefetch_batches = 2},
        [](std::span<const std::size_t>,
           const jyppx::liuding::data::DataWorkerContext&, std::stop_token) {
          return c10::IValue(torch::ones({2}, torch::kFloat32));
        });
  };

  auto inline_pipeline = make_invalid("data.schema.inline-failure", 0);
  inline_pipeline.set_epoch(0);
  EXPECT_THROW(inline_pipeline.next(), jyppx::liuding::InvalidArgument);
  EXPECT_EQ(inline_pipeline.state().next_cursor, 0U);
  EXPECT_EQ(inline_pipeline.state().next_sequence, 0U);

  auto worker_pipeline = make_invalid("data.schema.worker-failure", 2);
  worker_pipeline.set_epoch(0);
  EXPECT_THROW(worker_pipeline.next(), jyppx::liuding::InvalidArgument);
  EXPECT_EQ(worker_pipeline.state().next_cursor, 0U);
  EXPECT_EQ(worker_pipeline.state().next_sequence, 0U);
}

}  // namespace
