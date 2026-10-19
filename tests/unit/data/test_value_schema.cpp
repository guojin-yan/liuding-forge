// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/data/batch_schema.hpp>
#include <jyppx/data/data_pipeline.hpp>
#include <jyppx/data/samplers/sequential_sampler.hpp>
#include <jyppx/foundation/error.hpp>

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
jyppx::data::BatchSchema integer_batch_schema(std::string id) {
  return {.id = std::move(id),
          .version = 1,
          .value = jyppx::ValueSchema{.name = "count",
                                      .semantic_tag = "sample_count",
                                      .kind = jyppx::ValueKind::integer}};
}

/** @brief Creates a one-dimensional float32 tensor batch schema. */
jyppx::data::BatchSchema tensor_batch_schema(std::string id) {
  jyppx::TensorSchema tensor;
  tensor.dtype = torch::kFloat32;
  tensor.device = jyppx::Device(torch::kCPU);
  tensor.rank = 1;
  tensor.shape = std::vector<std::optional<std::int64_t>>{std::nullopt};
  return jyppx::data::BatchSchema::tensor(std::move(id), std::move(tensor), 1,
                                          "features");
}

TEST(ValueSchemaTest,
     ValidatesFieldsTensorConstraintsOptionalValuesAndSemanticTags) {
  jyppx::TensorSchema features;
  features.dtype = torch::kFloat32;
  features.device = jyppx::Device(torch::kCPU);
  features.rank = 2;
  features.shape = std::vector<std::optional<std::int64_t>>{std::nullopt, 4};

  auto note = jyppx::ValueSchema{.name = "note",
                                 .semantic_tag = "annotation",
                                 .kind = jyppx::ValueKind::string,
                                 .required = false,
                                 .nullable = true};
  const auto schema = jyppx::data::BatchSchema{
      .id = "training.supervised_batch",
      .version = 3,
      .value = jyppx::ValueSchema::dictionary(
          {{"features",
            jyppx::ValueSchema::tensor(features, "features", "model_input")},
           {"target", jyppx::ValueSchema{.name = "target",
                                         .semantic_tag = "class_index",
                                         .kind = jyppx::ValueKind::integer}},
           {"note", note}},
          false, "supervised_batch", "training_batch")};

  c10::impl::GenericDict value(c10::StringType::get(), c10::AnyType::get());
  value.insert("features", torch::ones({2, 4}, torch::kFloat32));
  value.insert("target", std::int64_t{1});

  EXPECT_NO_THROW(jyppx::data::validate_batch_schema(schema));
  EXPECT_NO_THROW(
      jyppx::data::validate_batch_value(c10::IValue(value), schema));
  EXPECT_EQ(jyppx::data::batch_schema_fingerprint(schema).size(), 16U);

  value.insert("extra", std::int64_t{9});
  EXPECT_THROW(jyppx::data::validate_batch_value(c10::IValue(value), schema),
               jyppx::InvalidArgument);
}

TEST(ValueSchemaTest, CanonicalDescriptorCoversIdentityVersionAndSemantics) {
  auto original = tensor_batch_schema("training.features");
  auto changed_tag = original;
  changed_tag.value.semantic_tag = "target";
  auto changed_version = original;
  changed_version.version = 2;

  EXPECT_EQ(jyppx::data::canonical_batch_schema(original),
            jyppx::data::canonical_batch_schema(original));
  EXPECT_NE(jyppx::data::canonical_batch_schema(original),
            jyppx::data::canonical_batch_schema(changed_tag));
  EXPECT_NE(jyppx::data::canonical_batch_schema(original),
            jyppx::data::canonical_batch_schema(changed_version));
  EXPECT_NE(jyppx::data::batch_schema_fingerprint(original),
            jyppx::data::batch_schema_fingerprint(changed_tag));
}

TEST(ValueSchemaTest, RejectsUnboundedOrMalformedSchemaAndRuntimeTrees) {
  EXPECT_THROW(jyppx::data::validate_batch_schema(
                   {.id = "", .version = 1, .value = jyppx::ValueSchema{}}),
               jyppx::InvalidArgument);
  EXPECT_THROW(
      jyppx::data::validate_batch_schema(
          {.id = "schema", .version = 0, .value = jyppx::ValueSchema{}}),
      jyppx::InvalidArgument);

  auto nested = jyppx::ValueSchema::list(
      jyppx::ValueSchema::list(jyppx::ValueSchema::list(
          jyppx::ValueSchema{.kind = jyppx::ValueKind::integer})));
  EXPECT_THROW(jyppx::validate_value_schema(nested, "batch",
                                            {.max_depth = 2,
                                             .max_schema_nodes = 16,
                                             .max_metadata_bytes = 1024,
                                             .max_runtime_nodes = 16}),
               jyppx::ResourceExhausted);

  c10::impl::GenericList values(c10::IntType::get());
  values.push_back(std::int64_t{1});
  values.push_back(std::int64_t{2});
  values.push_back(std::int64_t{3});
  const auto list_schema = jyppx::ValueSchema::list(
      jyppx::ValueSchema{.kind = jyppx::ValueKind::integer});
  EXPECT_THROW(jyppx::validate_value(c10::IValue(values), list_schema, "batch",
                                     {.max_depth = 8,
                                      .max_schema_nodes = 16,
                                      .max_metadata_bytes = 1024,
                                      .max_runtime_nodes = 3}),
               jyppx::ResourceExhausted);

  c10::impl::GenericDict keyed_value(c10::StringType::get(),
                                     c10::AnyType::get());
  keyed_value.insert("oversized-key", std::int64_t{1});
  const auto open_dictionary = jyppx::ValueSchema::dictionary({}, true);
  EXPECT_THROW(
      jyppx::validate_value(c10::IValue(keyed_value), open_dictionary, "batch",
                            {.max_depth = 8,
                             .max_schema_nodes = 16,
                             .max_metadata_bytes = 8,
                             .max_runtime_nodes = 16}),
      jyppx::ResourceExhausted);

  jyppx::TensorSchema excessive_shape;
  excessive_shape.shape =
      std::vector<std::optional<std::int64_t>>(17, std::nullopt);
  EXPECT_THROW(
      jyppx::validate_value_schema(
          jyppx::ValueSchema::tensor(std::move(excessive_shape)), "value",
          {.max_depth = 8,
           .max_schema_nodes = 16,
           .max_metadata_bytes = 1024,
           .max_runtime_nodes = 16}),
      jyppx::ResourceExhausted);
}

TEST(ValueSchemaTest, CoversEveryStableKindNameAndRuntimeCategory) {
  EXPECT_EQ(jyppx::value_kind_name(jyppx::ValueKind::any), "any");
  EXPECT_EQ(jyppx::value_kind_name(jyppx::ValueKind::tensor), "tensor");
  EXPECT_EQ(jyppx::value_kind_name(jyppx::ValueKind::tuple), "tuple");
  EXPECT_EQ(jyppx::value_kind_name(jyppx::ValueKind::list), "list");
  EXPECT_EQ(jyppx::value_kind_name(jyppx::ValueKind::dictionary), "dictionary");
  EXPECT_EQ(jyppx::value_kind_name(jyppx::ValueKind::integer), "integer");
  EXPECT_EQ(jyppx::value_kind_name(jyppx::ValueKind::floating), "floating");
  EXPECT_EQ(jyppx::value_kind_name(jyppx::ValueKind::boolean), "boolean");
  EXPECT_EQ(jyppx::value_kind_name(jyppx::ValueKind::string), "string");
  EXPECT_EQ(jyppx::value_kind_name(jyppx::ValueKind::none), "none");
  EXPECT_EQ(jyppx::value_kind_name(jyppx::ValueKind::device), "device");
  EXPECT_EQ(jyppx::value_kind_name(static_cast<jyppx::ValueKind>(255)),
            "unknown");

  EXPECT_NO_THROW(jyppx::validate_value(
      c10::IValue(std::int64_t{7}),
      jyppx::ValueSchema{.kind = jyppx::ValueKind::integer}));
  EXPECT_NO_THROW(jyppx::validate_value(
      c10::IValue(1.25),
      jyppx::ValueSchema{.kind = jyppx::ValueKind::floating}));
  EXPECT_NO_THROW(jyppx::validate_value(
      c10::IValue(true),
      jyppx::ValueSchema{.kind = jyppx::ValueKind::boolean}));
  EXPECT_NO_THROW(jyppx::validate_value(
      c10::IValue(std::string("annotation")),
      jyppx::ValueSchema{.kind = jyppx::ValueKind::string}));
  EXPECT_NO_THROW(jyppx::validate_value(
      c10::IValue(), jyppx::ValueSchema{.kind = jyppx::ValueKind::none}));
  EXPECT_NO_THROW(jyppx::validate_value(
      c10::IValue(c10::Device(c10::kCPU)),
      jyppx::ValueSchema{.kind = jyppx::ValueKind::device}));
  EXPECT_NO_THROW(jyppx::validate_value(
      c10::IValue(),
      jyppx::ValueSchema{.kind = jyppx::ValueKind::integer, .nullable = true}));

  c10::impl::GenericList list(c10::IntType::get());
  list.push_back(std::int64_t{3});
  list.push_back(std::int64_t{5});
  const auto list_schema = jyppx::ValueSchema::list(
      jyppx::ValueSchema{.kind = jyppx::ValueKind::integer});
  EXPECT_NO_THROW(jyppx::validate_value(c10::IValue(list), list_schema));

  const auto tuple_schema = jyppx::ValueSchema::tuple(
      {jyppx::ValueSchema{.kind = jyppx::ValueKind::integer},
       jyppx::ValueSchema{.kind = jyppx::ValueKind::floating}});
  const auto tuple = c10::ivalue::Tuple::create(
      {c10::IValue(std::int64_t{2}), c10::IValue(0.5)});
  EXPECT_NO_THROW(jyppx::validate_value(c10::IValue(tuple), tuple_schema));
  EXPECT_THROW(jyppx::validate_value(c10::IValue(c10::ivalue::Tuple::create(
                                         {c10::IValue(std::int64_t{2})})),
                                     tuple_schema),
               jyppx::InvalidArgument);

  EXPECT_THROW(jyppx::validate_value(
                   c10::IValue(false),
                   jyppx::ValueSchema{.name = "count",
                                      .kind = jyppx::ValueKind::integer}),
               jyppx::InvalidArgument);
}

TEST(ValueSchemaTest, RejectsEveryMalformedSchemaCombinationAndLimit) {
  auto limits = jyppx::ValueSchemaLimits{};
  limits.max_depth = 0;
  EXPECT_THROW(jyppx::validate_value_schema({}, "value", limits),
               jyppx::InvalidArgument);
  limits = {};
  limits.max_schema_nodes = 0;
  EXPECT_THROW(jyppx::validate_value_schema({}, "value", limits),
               jyppx::InvalidArgument);
  limits = {};
  limits.max_metadata_bytes = 0;
  EXPECT_THROW(jyppx::validate_value_schema({}, "value", limits),
               jyppx::InvalidArgument);
  limits = {};
  limits.max_runtime_nodes = 0;
  EXPECT_THROW(jyppx::validate_value_schema({}, "value", limits),
               jyppx::InvalidArgument);

  auto primitive = jyppx::ValueSchema{.kind = jyppx::ValueKind::integer};
  primitive.elements.push_back(jyppx::ValueSchema{});
  EXPECT_THROW(jyppx::validate_value_schema(primitive), jyppx::InvalidArgument);

  auto tensor_with_children = jyppx::ValueSchema::tensor({}, "invalid_tensor");
  tensor_with_children.fields.emplace("child", jyppx::ValueSchema{});
  EXPECT_THROW(jyppx::validate_value_schema(tensor_with_children),
               jyppx::InvalidArgument);
  auto tensor_with_extra_fields = jyppx::ValueSchema::tensor();
  tensor_with_extra_fields.allow_extra_fields = true;
  EXPECT_THROW(jyppx::validate_value_schema(tensor_with_extra_fields),
               jyppx::InvalidArgument);
  EXPECT_NO_THROW(jyppx::validate_value_schema(
      jyppx::ValueSchema{.kind = jyppx::ValueKind::tensor}));

  jyppx::TensorSchema negative_rank;
  negative_rank.rank = -1;
  EXPECT_THROW(
      jyppx::validate_value_schema(jyppx::ValueSchema::tensor(negative_rank)),
      jyppx::InvalidArgument);

  jyppx::TensorSchema mismatched_rank;
  mismatched_rank.rank = 1;
  mismatched_rank.shape = std::vector<std::optional<std::int64_t>>{2, 3};
  EXPECT_THROW(
      jyppx::validate_value_schema(jyppx::ValueSchema::tensor(mismatched_rank)),
      jyppx::InvalidArgument);

  jyppx::TensorSchema negative_extent;
  negative_extent.shape =
      std::vector<std::optional<std::int64_t>>{std::int64_t{-2}};
  EXPECT_THROW(
      jyppx::validate_value_schema(jyppx::ValueSchema::tensor(negative_extent)),
      jyppx::InvalidArgument);

  jyppx::TensorSchema bounded_device;
  bounded_device.device = jyppx::Device(torch::kCUDA);
  EXPECT_THROW(jyppx::validate_value_schema(
                   jyppx::ValueSchema::tensor(bounded_device), "value",
                   {.max_depth = 8,
                    .max_schema_nodes = 16,
                    .max_metadata_bytes = 3,
                    .max_runtime_nodes = 16}),
               jyppx::ResourceExhausted);

  auto malformed_tuple = jyppx::ValueSchema::tuple({});
  malformed_tuple.allow_extra_fields = true;
  EXPECT_THROW(jyppx::validate_value_schema(malformed_tuple),
               jyppx::InvalidArgument);

  auto malformed_list = jyppx::ValueSchema{.kind = jyppx::ValueKind::list};
  EXPECT_THROW(jyppx::validate_value_schema(malformed_list),
               jyppx::InvalidArgument);
  malformed_list.elements = {jyppx::ValueSchema{}, jyppx::ValueSchema{}};
  EXPECT_THROW(jyppx::validate_value_schema(malformed_list),
               jyppx::InvalidArgument);

  auto malformed_dictionary = jyppx::ValueSchema::dictionary({});
  malformed_dictionary.elements.push_back(jyppx::ValueSchema{});
  EXPECT_THROW(jyppx::validate_value_schema(malformed_dictionary),
               jyppx::InvalidArgument);

  auto empty_key_dictionary = jyppx::ValueSchema::dictionary({});
  empty_key_dictionary.fields.emplace("", jyppx::ValueSchema{});
  EXPECT_THROW(jyppx::validate_value_schema(empty_key_dictionary),
               jyppx::InvalidArgument);

  auto invalid_kind = jyppx::ValueSchema{};
  invalid_kind.kind = static_cast<jyppx::ValueKind>(255);
  EXPECT_THROW(jyppx::validate_value_schema(invalid_kind),
               jyppx::InvalidArgument);

  EXPECT_THROW(jyppx::validate_value_schema(
                   jyppx::ValueSchema::any("oversized-name"), "value",
                   {.max_depth = 8,
                    .max_schema_nodes = 16,
                    .max_metadata_bytes = 4,
                    .max_runtime_nodes = 16}),
               jyppx::ResourceExhausted);

  const auto cumulative_metadata = jyppx::ValueSchema::dictionary(
      {{"field", jyppx::ValueSchema::any("child")}}, false, "root");
  EXPECT_THROW(jyppx::validate_value_schema(cumulative_metadata, "value",
                                            {.max_depth = 8,
                                             .max_schema_nodes = 16,
                                             .max_metadata_bytes = 8,
                                             .max_runtime_nodes = 16}),
               jyppx::ResourceExhausted);
}

TEST(ValueSchemaTest, ValidatesEveryTensorConstraintFailureAtomically) {
  EXPECT_THROW(jyppx::validate_value(c10::IValue(jyppx::Tensor()),
                                     jyppx::ValueSchema::tensor()),
               jyppx::InvalidArgument);

  jyppx::TensorSchema dtype;
  dtype.dtype = torch::kFloat32;
  EXPECT_THROW(
      jyppx::validate_value(c10::IValue(torch::ones({2}, torch::kInt64)),
                            jyppx::ValueSchema::tensor(dtype)),
      jyppx::InvalidArgument);

  jyppx::TensorSchema device;
  device.device = jyppx::Device(torch::kCUDA);
  EXPECT_THROW(
      jyppx::validate_value(c10::IValue(torch::ones({2}, torch::kFloat32)),
                            jyppx::ValueSchema::tensor(device)),
      jyppx::InvalidArgument);

  jyppx::TensorSchema rank;
  rank.rank = 2;
  EXPECT_THROW(
      jyppx::validate_value(c10::IValue(torch::ones({2}, torch::kFloat32)),
                            jyppx::ValueSchema::tensor(rank)),
      jyppx::InvalidArgument);

  jyppx::TensorSchema shape_rank;
  shape_rank.shape = std::vector<std::optional<std::int64_t>>{2, std::nullopt};
  EXPECT_THROW(
      jyppx::validate_value(c10::IValue(torch::ones({2}, torch::kFloat32)),
                            jyppx::ValueSchema::tensor(shape_rank)),
      jyppx::InvalidArgument);

  jyppx::TensorSchema extent;
  extent.shape = std::vector<std::optional<std::int64_t>>{3};
  EXPECT_THROW(
      jyppx::validate_value(c10::IValue(torch::ones({2}, torch::kFloat32)),
                            jyppx::ValueSchema::tensor(extent)),
      jyppx::InvalidArgument);

  EXPECT_NO_THROW(
      jyppx::validate_value(c10::IValue(torch::ones({2}, torch::kFloat32)),
                            jyppx::ValueSchema::tensor()));
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

  EXPECT_NO_THROW(jyppx::validate_value(
      nested, jyppx::ValueSchema::any("payload", "untyped_payload")));
  EXPECT_THROW(jyppx::validate_value(nested, jyppx::ValueSchema::any(), "value",
                                     {.max_depth = 1,
                                      .max_schema_nodes = 16,
                                      .max_metadata_bytes = 1024,
                                      .max_runtime_nodes = 32}),
               jyppx::ResourceExhausted);

  const auto open_schema = jyppx::ValueSchema::dictionary(
      {{"required", jyppx::ValueSchema{.kind = jyppx::ValueKind::integer}},
       {"optional", jyppx::ValueSchema{.kind = jyppx::ValueKind::string,
                                       .required = false}}},
      true, "open_payload");
  c10::impl::GenericDict open_value(c10::StringType::get(),
                                    c10::AnyType::get());
  open_value.insert("required", std::int64_t{4});
  open_value.insert("extra", nested);
  EXPECT_NO_THROW(jyppx::validate_value(c10::IValue(open_value), open_schema));

  c10::impl::GenericDict missing(c10::StringType::get(), c10::AnyType::get());
  EXPECT_THROW(jyppx::validate_value(c10::IValue(missing), open_schema),
               jyppx::InvalidArgument);

  c10::impl::GenericDict non_string_key(c10::IntType::get(),
                                        c10::AnyType::get());
  non_string_key.insert(std::int64_t{1}, std::int64_t{2});
  EXPECT_THROW(jyppx::validate_value(c10::IValue(non_string_key),
                                     jyppx::ValueSchema::dictionary({}, true)),
               jyppx::InvalidArgument);
}

TEST(ValueSchemaTest, DirectFingerprintIsStableAndCoversCompleteDescriptor) {
  const auto schema = jyppx::ValueSchema::tuple(
      {jyppx::ValueSchema::tensor({}, "features"),
       jyppx::ValueSchema::list(
           jyppx::ValueSchema{.kind = jyppx::ValueKind::integer}, "labels")},
      "batch", "training_batch");
  const auto fingerprint = jyppx::value_schema_fingerprint(schema);
  EXPECT_EQ(fingerprint.size(), 16U);
  EXPECT_EQ(fingerprint, jyppx::value_schema_fingerprint(schema));

  auto changed = schema;
  changed.elements.front().semantic_tag = "model_input";
  EXPECT_NE(fingerprint, jyppx::value_schema_fingerprint(changed));
}

TEST(ValueSchemaPipelineTest,
     PublishesCompleteCapabilityAndValidatedEnvelopeSchema) {
  const auto schema = tensor_batch_schema("pipeline.features");
  jyppx::data::DataPipeline pipeline(
      std::make_shared<jyppx::data::samplers::SequentialSampler>(4),
      {.pipeline_id = "data.schema.capability",
       .dataset_version = "dataset.schema/v1",
       .dataset_size = 4,
       .schema = schema,
       .batch_factory_id = "factory.schema.tensor/v1",
       .batch_size = 2},
      [](std::span<const std::size_t> indices,
         const jyppx::data::DataWorkerContext&, std::stop_token) {
        return c10::IValue(torch::ones(
            {static_cast<std::int64_t>(indices.size())}, torch::kFloat32));
      });

  const auto capability = pipeline.capability();
  EXPECT_TRUE(capability.map_style);
  EXPECT_TRUE(capability.finite_epoch);
  EXPECT_TRUE(capability.schema_validation);
  EXPECT_FALSE(capability.process_workers);
  EXPECT_EQ(capability.worker_model,
            jyppx::data::DataWorkerModel::inline_execution);
  EXPECT_EQ(capability.batch_schema.id, schema.id);
  EXPECT_EQ(capability.batch_schema_fingerprint,
            jyppx::data::batch_schema_fingerprint(schema));
  EXPECT_EQ(capability.max_batch_tensors, 4096U);
  EXPECT_EQ(capability.max_batch_value_nodes, 65536U);
  EXPECT_EQ(capability.max_batch_bytes, 1ULL << 30U);

  pipeline.set_epoch(0);
  const auto batch = pipeline.next();
  ASSERT_TRUE(batch.has_value());
  EXPECT_EQ(jyppx::data::canonical_batch_schema(batch->schema),
            jyppx::data::canonical_batch_schema(schema));
}

TEST(ValueSchemaPipelineTest,
     InlineAndWorkerValidationFailuresDoNotAdvanceCommittedCursor) {
  const auto make_invalid = [](std::string id, std::size_t workers) {
    return jyppx::data::DataPipeline(
        std::make_shared<jyppx::data::samplers::SequentialSampler>(4),
        {.pipeline_id = std::move(id),
         .dataset_version = "dataset.schema/v1",
         .dataset_size = 4,
         .schema = integer_batch_schema("pipeline.integer"),
         .batch_factory_id = "factory.schema.invalid/v1",
         .batch_size = 2,
         .num_workers = workers,
         .prefetch_batches = 2},
        [](std::span<const std::size_t>, const jyppx::data::DataWorkerContext&,
           std::stop_token) {
          return c10::IValue(torch::ones({2}, torch::kFloat32));
        });
  };

  auto inline_pipeline = make_invalid("data.schema.inline-failure", 0);
  inline_pipeline.set_epoch(0);
  EXPECT_THROW(inline_pipeline.next(), jyppx::InvalidArgument);
  EXPECT_EQ(inline_pipeline.state().next_cursor, 0U);
  EXPECT_EQ(inline_pipeline.state().next_sequence, 0U);

  auto worker_pipeline = make_invalid("data.schema.worker-failure", 2);
  worker_pipeline.set_epoch(0);
  EXPECT_THROW(worker_pipeline.next(), jyppx::InvalidArgument);
  EXPECT_EQ(worker_pipeline.state().next_cursor, 0U);
  EXPECT_EQ(worker_pipeline.state().next_sequence, 0U);
}

}  // namespace
