// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data/batch_memory.hpp>
#include <jyppx/liuding/data/data_pipeline.hpp>
#include <jyppx/liuding/data/samplers/sequential_sampler.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

/** @brief In-memory participant store used by checkpoint continuation tests. */
class MemoryStateStore final : public jyppx::liuding::StateWriter,
                               public jyppx::liuding::StateReader {
 public:
  /** @brief Creates a store constrained by one participant manifest. */
  explicit MemoryStateStore(jyppx::liuding::StateManifest manifest)
      : manifest_(std::move(manifest)) {
    jyppx::liuding::validate_state_manifest(manifest_);
  }

  /** @brief Stores one declared bounded checkpoint field. */
  void write_blob(std::string_view name,
                  std::span<const std::byte> data) override {
    const auto field = std::find_if(
        manifest_.fields.begin(), manifest_.fields.end(),
        [name](const jyppx::liuding::StateFieldDescriptor& candidate) {
          return candidate.name == name;
        });
    if (field == manifest_.fields.end() || data.size() > field->max_bytes ||
        !blobs_
             .emplace(std::string(name),
                      std::vector<std::byte>(data.begin(), data.end()))
             .second) {
      throw std::runtime_error("invalid batch-memory checkpoint field");
    }
  }

  /** @brief Returns the immutable stored participant manifest. */
  const jyppx::liuding::StateManifest& manifest() const noexcept override {
    return manifest_;
  }

  /** @brief Reports whether a named field was written. */
  bool contains(std::string_view name) const override {
    return blobs_.contains(std::string(name));
  }

  /** @brief Returns a stored field after enforcing the reader limit. */
  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t max_bytes) const override {
    const auto found = blobs_.find(std::string(name));
    if (found == blobs_.end() || found->second.size() > max_bytes) {
      throw std::runtime_error("missing batch-memory checkpoint field");
    }
    return found->second;
  }

 private:
  jyppx::liuding::StateManifest
      manifest_;  ///< Descriptor enforced for every field.
  std::map<std::string, std::vector<std::byte>>
      blobs_;  ///< Independently owned checkpoint payloads.
};

/** @brief Creates a deterministic tensor batch from sequential sample IDs. */
jyppx::liuding::data::DataBatchFactory tensor_factory() {
  return [](std::span<const std::size_t> indices,
            const jyppx::liuding::data::DataWorkerContext&, std::stop_token) {
    std::vector<std::int64_t> values;
    values.reserve(indices.size());
    for (const auto index : indices) {
      values.push_back(static_cast<std::int64_t>(index));
    }
    return c10::IValue(torch::tensor(values, torch::kFloat32));
  };
}

/** @brief Creates a map pipeline with one explicit memory-stage topology. */
jyppx::liuding::data::DataPipeline make_pipeline(
    std::string id, jyppx::liuding::data::BatchMemoryOptions memory,
    std::size_t workers = 0, std::size_t prefetch = 2) {
  jyppx::liuding::TensorSchema tensor_schema;
  tensor_schema.dtype = torch::kFloat32;
  tensor_schema.device =
      memory.target_device.value_or(jyppx::liuding::Device(torch::kCPU));
  tensor_schema.rank = 1;
  tensor_schema.shape = std::vector<std::optional<std::int64_t>>{std::nullopt};
  return jyppx::liuding::data::DataPipeline(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(4),
      {.pipeline_id = std::move(id),
       .dataset_version = "batch-memory.dataset/v1",
       .dataset_size = 4,
       .schema = jyppx::liuding::data::BatchSchema::tensor(
           "batch-memory.tensor", std::move(tensor_schema), 1, "features"),
       .batch_factory_id = "batch-memory.factory/v1",
       .batch_size = 2,
       .num_workers = workers,
       .prefetch_batches = prefetch,
       .persistent_workers = workers != 0,
       .memory = std::move(memory)},
      tensor_factory());
}

TEST(BatchMemoryTest, DisabledStagePreservesTheValueWithoutTraversal) {
  const auto tensor = torch::arange(0, 4, torch::kFloat32);
  const auto result = jyppx::liuding::data::stage_batch_memory(
      c10::IValue(tensor), jyppx::liuding::data::BatchMemoryOptions{});

  EXPECT_FALSE(result.metadata.staged);
  EXPECT_EQ(result.metadata.tensor_count, 0U);
  EXPECT_TRUE(result.value.toTensor().is_same(tensor));
}

TEST(BatchMemoryTest, PreservesNestedContainerTypesAndReportsCpuMetadata) {
  c10::impl::GenericList list(c10::TensorType::get());
  list.push_back(torch::ones({2}, torch::kFloat32));
  c10::impl::GenericDict dictionary(c10::StringType::get(),
                                    c10::TensorType::get());
  dictionary.insert("target", torch::zeros({3}, torch::kFloat32));
  const auto value = c10::IValue(c10::ivalue::Tuple::create(
      {c10::IValue(list), c10::IValue(dictionary), c10::IValue(7)}));

  const auto result = jyppx::liuding::data::stage_batch_memory(
      value, {.target_device = jyppx::liuding::Device(torch::kCPU)});

  ASSERT_TRUE(result.value.isTuple());
  const auto& elements = result.value.toTupleRef().elements();
  ASSERT_EQ(elements.size(), 3U);
  EXPECT_TRUE(elements[0].isList());
  EXPECT_TRUE(elements[1].isGenericDict());
  EXPECT_EQ(elements[2].toInt(), 7);
  EXPECT_TRUE(result.metadata.staged);
  EXPECT_EQ(result.metadata.tensor_count, 2U);
  EXPECT_EQ(result.metadata.value_node_count, 7U);
  EXPECT_EQ(result.metadata.tensor_bytes, 20U);
  EXPECT_EQ(result.metadata.transferred_tensor_count, 0U);
  EXPECT_EQ(result.metadata.transfer_stream_policy,
            jyppx::liuding::data::BatchTransferStreamPolicy::caller_current);
  ASSERT_TRUE(result.metadata.target_device.has_value());
  EXPECT_TRUE(result.metadata.target_device->is_cpu());
}

TEST(BatchMemoryTest, RejectsModifiedPreparationMetadataBeforeTransfer) {
  const jyppx::liuding::data::BatchMemoryOptions options{
      .target_device = jyppx::liuding::Device(torch::kCPU)};
  auto prepared = jyppx::liuding::data::prepare_batch_memory(
      c10::IValue(torch::ones({2}, torch::kFloat32)), options);
  ++prepared.metadata.tensor_count;

  EXPECT_THROW(
      jyppx::liuding::data::transfer_batch_memory(std::move(prepared), options),
      jyppx::liuding::CompatibilityError);
}

TEST(BatchMemoryTest, RejectsInvalidPolicyAndPreflightsResourceLimits) {
  auto non_blocking = jyppx::liuding::data::BatchMemoryOptions{};
  non_blocking.non_blocking = true;
  EXPECT_THROW(
      jyppx::liuding::data::validate_batch_memory_options(non_blocking),
      jyppx::liuding::InvalidArgument);
  non_blocking.target_device = jyppx::liuding::Device(torch::kCPU);
  EXPECT_NO_THROW(
      jyppx::liuding::data::validate_batch_memory_options(non_blocking));

  auto bytes = jyppx::liuding::data::BatchMemoryOptions{};
  bytes.target_device = jyppx::liuding::Device(torch::kCPU);
  bytes.max_bytes = 4;
  EXPECT_THROW(jyppx::liuding::data::stage_batch_memory(
                   c10::IValue(torch::ones({2}, torch::kFloat32)), bytes),
               jyppx::liuding::ResourceExhausted);

  auto nodes = jyppx::liuding::data::BatchMemoryOptions{};
  nodes.target_device = jyppx::liuding::Device(torch::kCPU);
  nodes.max_value_nodes = 1;
  c10::impl::GenericList list(c10::IntType::get());
  list.push_back(1);
  EXPECT_THROW(
      jyppx::liuding::data::stage_batch_memory(c10::IValue(list), nodes),
      jyppx::liuding::ResourceExhausted);

  const auto sparse = torch::sparse_coo_tensor(
      torch::tensor({{0}, {1}}, torch::kInt64), torch::tensor({1.0F}), {2, 2});
  EXPECT_THROW(
      jyppx::liuding::data::stage_batch_memory(c10::IValue(sparse), bytes),
      jyppx::liuding::InvalidArgument);
}

TEST(BatchMemoryTest, MapsPinningBackendFailureToFoundationDeviceError) {
  const auto tensor = torch::ones({2}, torch::kFloat32);
  const jyppx::liuding::data::BatchMemoryOptions options{.pin_memory = true};
  if (torch::cuda::is_available()) {
    const auto result =
        jyppx::liuding::data::stage_batch_memory(c10::IValue(tensor), options);
    EXPECT_TRUE(result.value.toTensor().is_pinned());
    EXPECT_EQ(result.metadata.pinned_tensor_count, 1U);
  } else {
    EXPECT_THROW(
        jyppx::liuding::data::stage_batch_memory(c10::IValue(tensor), options),
        jyppx::liuding::DeviceError);
  }
}

TEST(BatchMemoryPipelineTest, ExposesIndependentStageCapacityAndMetadata) {
  auto pipeline =
      make_pipeline("batch-memory.capacity",
                    {.target_device = jyppx::liuding::Device(torch::kCPU),
                     .stage_capacity = 1},
                    2, 4);
  pipeline.set_epoch(3);

  const auto capability = pipeline.capability();
  EXPECT_TRUE(capability.memory_stage);
  EXPECT_EQ(capability.prefetch_batches, 4U);
  EXPECT_EQ(capability.memory_stage_capacity, 1U);
  EXPECT_EQ(capability.max_replayed_batches, 1U);
  EXPECT_EQ(capability.transfer_stream_policy,
            jyppx::liuding::data::BatchTransferStreamPolicy::caller_current);
  EXPECT_LE(pipeline.outstanding_batches(), 1U);

  const auto batch = pipeline.next();
  ASSERT_TRUE(batch.has_value());
  EXPECT_TRUE(batch->memory.staged);
  EXPECT_EQ(batch->memory.tensor_count, 1U);
  EXPECT_EQ(batch->sample_ids, (std::vector<std::uint64_t>{0U, 1U}));
  const auto state = pipeline.state();
  EXPECT_EQ(state.memory_stage_capacity, 1U);
  EXPECT_EQ(state.target_device, "cpu");
}

TEST(BatchMemoryPipelineTest, WorkerStageFailureDoesNotAdvanceCommittedCursor) {
  auto memory = jyppx::liuding::data::BatchMemoryOptions{};
  memory.target_device = jyppx::liuding::Device(torch::kCPU);
  memory.max_bytes = 1;
  auto pipeline = make_pipeline("batch-memory.worker-failure", memory, 2, 2);
  pipeline.set_epoch(5);
  const auto before = pipeline.state();

  EXPECT_THROW(pipeline.next(), jyppx::liuding::ResourceExhausted);
  EXPECT_EQ(pipeline.state().next_cursor, before.next_cursor);
  EXPECT_EQ(pipeline.state().next_sequence, before.next_sequence);
}

TEST(BatchMemoryPipelineTest,
     RejectsMemoryTopologyMismatchWithoutMutatingPreparedState) {
  auto source =
      make_pipeline("batch-memory.topology",
                    {.target_device = jyppx::liuding::Device(torch::kCPU),
                     .stage_capacity = 1});
  source.set_epoch(2);
  ASSERT_TRUE(source.next().has_value());
  MemoryStateStore checkpoint(source.describe_state());
  source.save_state(checkpoint);

  auto target = make_pipeline("batch-memory.topology", {});
  target.set_epoch(9);
  const auto before = target.state();
  EXPECT_THROW(
      target.load_state(checkpoint, jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::CompatibilityError);
  EXPECT_EQ(target.state().epoch, before.epoch);
  EXPECT_EQ(target.state().next_cursor, before.next_cursor);
  EXPECT_EQ(target.state().target_device, before.target_device);
}

TEST(BatchMemoryCudaTest, PinsCpuLeavesAndTransfersNestedBatchToCuda) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA runtime is unavailable";
  }
  EXPECT_THROW(jyppx::liuding::data::prepare_batch_memory(
                   c10::IValue(torch::ones({1}, torch::kCUDA)),
                   {.target_device = jyppx::liuding::Device(torch::kCUDA, 0)}),
               jyppx::liuding::InvalidArgument);
  const auto source = torch::arange(0, 6, torch::kFloat32).view({2, 3});
  c10::impl::GenericList list(c10::TensorType::get());
  list.push_back(source);
  const auto value = c10::IValue(
      c10::ivalue::Tuple::create({c10::IValue(list), c10::IValue(source + 1)}));

  auto prepared = jyppx::liuding::data::prepare_batch_memory(
      value, {.pin_memory = true,
              .target_device = jyppx::liuding::Device(torch::kCUDA, 0),
              .non_blocking = true});
  const auto& prepared_elements = prepared.value.toTupleRef().elements();
  EXPECT_TRUE(prepared_elements[0].toListRef()[0].toTensor().is_pinned());
  EXPECT_TRUE(prepared_elements[1].toTensor().is_pinned());
  EXPECT_FALSE(prepared_elements[0].toListRef()[0].toTensor().is_cuda());
  EXPECT_EQ(prepared.metadata.transferred_tensor_count, 0U);

  const auto result = jyppx::liuding::data::transfer_batch_memory(
      std::move(prepared),
      {.pin_memory = true,
       .target_device = jyppx::liuding::Device(torch::kCUDA, 0),
       .non_blocking = true});

  const auto& elements = result.value.toTupleRef().elements();
  ASSERT_TRUE(elements[0].toListRef()[0].toTensor().is_cuda());
  ASSERT_TRUE(elements[1].toTensor().is_cuda());
  EXPECT_TRUE(
      torch::equal(elements[0].toListRef()[0].toTensor().cpu(), source));
  EXPECT_EQ(result.metadata.tensor_count, 2U);
  EXPECT_EQ(result.metadata.pinned_tensor_count, 2U);
  EXPECT_EQ(result.metadata.transferred_tensor_count, 2U);
  EXPECT_TRUE(result.metadata.non_blocking_requested);
  EXPECT_EQ(result.metadata.transfer_stream_policy,
            jyppx::liuding::data::BatchTransferStreamPolicy::caller_current);
  EXPECT_FALSE(source.is_pinned());
  EXPECT_TRUE(source.device().is_cpu());
}

TEST(BatchMemoryCudaTest, EmptyTensorLeavesDoNotRequirePinnedStorage) {
  if (!torch::cuda::is_available())
    GTEST_SKIP() << "CUDA runtime is unavailable";
  const auto value = c10::IValue(c10::ivalue::Tuple::create(
      {torch::empty({0}, torch::kInt64), torch::ones({2}, torch::kFloat32)}));
  const jyppx::liuding::data::BatchMemoryOptions options{
      .pin_memory = true,
      .target_device = jyppx::liuding::Device(torch::kCUDA, 0),
      .non_blocking = true};
  auto prepared = jyppx::liuding::data::prepare_batch_memory(value, options);
  EXPECT_EQ(prepared.metadata.tensor_count, 2U);
  EXPECT_EQ(prepared.metadata.pinned_tensor_count, 1U);
  EXPECT_EQ(prepared.metadata.tensor_bytes, 8U);
  const auto result =
      jyppx::liuding::data::transfer_batch_memory(std::move(prepared), options);
  EXPECT_EQ(result.metadata.transferred_tensor_count, 2U);
  const auto empty = result.value.toTupleRef().elements()[0].toTensor();
  EXPECT_TRUE(empty.is_cuda());
  EXPECT_EQ(empty.numel(), 0);
  EXPECT_EQ(empty.scalar_type(), torch::kInt64);
}

TEST(BatchMemoryCudaTest, PipelineCheckpointContinuesExactlyOnCuda) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA runtime is unavailable";
  }
  const jyppx::liuding::data::BatchMemoryOptions memory{
      .pin_memory = true,
      .target_device = jyppx::liuding::Device(torch::kCUDA, 0),
      .non_blocking = true,
      .stage_capacity = 1};
  auto source = make_pipeline("batch-memory.cuda-checkpoint", memory, 2, 3);
  const auto capability =
      jyppx::liuding::data::describe_data_capability(source.capability());
  EXPECT_EQ(capability.devices, (std::vector<std::string>{"cuda:0"}));
  EXPECT_NE(
      std::find(capability.operations.begin(), capability.operations.end(),
                "memory.transfer.non-blocking"),
      capability.operations.end());
  EXPECT_NE(
      std::find(capability.operations.begin(), capability.operations.end(),
                "memory.stream.caller-current"),
      capability.operations.end());
  source.set_epoch(11);
  const auto first = source.next();
  ASSERT_TRUE(first.has_value());
  EXPECT_TRUE(first->value.toTensor().is_cuda());

  MemoryStateStore checkpoint(source.describe_state());
  source.save_state(checkpoint);
  const auto expected = source.next();
  ASSERT_TRUE(expected.has_value());

  auto restored = make_pipeline("batch-memory.cuda-checkpoint", memory, 2, 3);
  restored.load_state(checkpoint, jyppx::liuding::RestorePolicy::exact_only);
  const auto actual = restored.next();
  ASSERT_TRUE(actual.has_value());
  EXPECT_EQ(actual->sample_ids, expected->sample_ids);
  EXPECT_TRUE(actual->value.toTensor().is_cuda());
  EXPECT_TRUE(
      torch::equal(actual->value.toTensor(), expected->value.toTensor()));
  EXPECT_EQ(actual->memory.pinned_tensor_count, 1U);
  EXPECT_EQ(actual->memory.transferred_tensor_count, 1U);
  EXPECT_FALSE(restored.next().has_value());
}

}  // namespace
