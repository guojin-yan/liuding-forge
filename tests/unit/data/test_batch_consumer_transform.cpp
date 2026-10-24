// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data/iterable_data_pipeline.hpp>
#include <jyppx/liuding/data/samplers/sequential_sampler.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <algorithm>
#include <atomic>
#include <gtest/gtest.h>
#include <map>
#include <thread>

namespace {
using namespace jyppx::liuding;
using namespace jyppx::liuding::data;

class SavedBatchState final : public StateReader, public StateWriter {
 public:
  explicit SavedBatchState(StateManifest m) : manifest_(std::move(m)) {}
  const StateManifest& manifest() const noexcept override { return manifest_; }
  bool contains(std::string_view n) const override {
    return blobs_.contains(std::string(n));
  }
  std::span<const std::byte> read_blob(std::string_view n,
                                       std::uint64_t max) const override {
    const auto& b = blobs_.at(std::string(n));
    if (b.size() > max) throw ResourceExhausted("test state too large");
    return b;
  }
  void write_blob(std::string_view n, std::span<const std::byte> b) override {
    blobs_[std::string(n)] = {b.begin(), b.end()};
  }
  void corrupt_topology(int mode) {
    auto& b = blobs_.at("topology");
    if (mode == 0) b.pop_back();
    if (mode == 1) b.back() = std::byte{2};
    if (mode == 2) b.push_back(std::byte{0});
    if (mode == 3) {
      const std::string bytes(reinterpret_cast<const char*>(b.data()),
                              b.size());
      const auto offset = bytes.find("scale/half/v1") - 8U;
      for (std::size_t i = 0; i < 8; ++i) b.at(offset + i) = std::byte{255};
    }
  }

 private:
  StateManifest manifest_;
  std::map<std::string, std::vector<std::byte>> blobs_;
};

class SixCursor final : public IterableSourceCursor {
 public:
  std::optional<IterableSample> next(std::stop_token stop) override {
    if (closed_ || stop.stop_requested() || position_ == 6) return {};
    const auto i = position_++;
    return IterableSample{c10::IValue(static_cast<std::int64_t>(i)), i,
                          std::to_string(position_)};
  }
  std::string save_state() const override { return std::to_string(position_); }
  void load_state(std::string_view s) override {
    if (s.size() != 1 || s[0] < '0' || s[0] > '6')
      throw SerializationError("invalid test cursor");
    position_ = static_cast<std::uint64_t>(s[0] - '0');
  }
  void close() noexcept override { closed_ = true; }

 private:
  std::uint64_t position_ = 0;
  std::atomic<bool> closed_{false};
};

class SixSource final : public IterableSource {
 public:
  IterableSourceDescriptor descriptor() const override {
    return {.source_id = "six/v1",
            .finite_epoch = true,
            .deterministic_epoch = true,
            .checkpointable = true,
            .restore_level = RestoreLevel::exact,
            .max_replayed_samples = 0,
            .max_omitted_samples = 0};
  }
  std::unique_ptr<IterableSourceCursor> open_epoch(
      std::uint64_t) const override {
    return std::make_unique<SixCursor>();
  }
};

BatchSchema BytesSchema() {
  return BatchSchema::tensor(
      "bytes",
      {.dtype = torch::kUInt8, .device = Device(torch::kCPU), .rank = 1});
}
BatchSchema FloatsSchema() {
  return BatchSchema::tensor(
      "scaled",
      {.dtype = torch::kFloat32, .device = Device(torch::kCPU), .rank = 1});
}
BatchConsumerTransform ScaleTransform() {
  return {.id = "scale/half/v1",
          .input_schema = BytesSchema(),
          .deterministic = true,
          .apply = [](const c10::IValue& v, const BatchConsumerContext&,
                      std::stop_token) {
            return c10::IValue(v.toTensor().to(torch::kFloat32) * 0.5F);
          }};
}

std::unique_ptr<BatchPipeline> Build(
    int mode,
    std::optional<BatchConsumerTransform> transform = ScaleTransform(),
    BatchSchema output = FloatsSchema(), BatchMemoryOptions memory = {},
    bool deterministic_factory = true) {
  if (mode < 2) {
    DataPipelineOptions o;
    o.pipeline_id = "precommit.fixture";
    o.dataset_version = "six-fixture/v1";
    o.dataset_size = 6;
    o.batch_factory_id = "six/bytes/v1";
    o.batch_size = 2;
    o.num_workers = mode == 0 ? 0 : 2;
    o.prefetch_batches = 3;
    o.persistent_workers = mode != 0;
    o.schema = std::move(output);
    o.memory = memory;
    o.deterministic_batch_factory = deterministic_factory;
    o.consumer_transform = std::move(transform);
    return std::make_unique<DataPipeline>(
        std::make_shared<samplers::SequentialSampler>(6), o,
        [](std::span<const std::size_t> ids, const DataWorkerContext&,
           std::stop_token) {
          std::vector<std::uint8_t> values;
          for (auto i : ids) values.push_back(static_cast<std::uint8_t>(i));
          return c10::IValue(torch::tensor(values, torch::kUInt8));
        });
  }
  IterableDataPipelineOptions o;
  o.pipeline_id = "precommit.fixture";
  o.dataset_version = "six-fixture/v1";
  o.batch_factory_id = "six/bytes/v1";
  o.batch_size = 2;
  o.prefetch_batches = 3;
  o.schema = std::move(output);
  o.memory = memory;
  o.deterministic_batch_factory = deterministic_factory;
  o.consumer_transform = std::move(transform);
  return std::make_unique<IterableDataPipeline>(
      std::make_shared<SixSource>(), o,
      [](std::span<const IterableSample> samples, std::uint64_t, std::uint64_t,
         std::stop_token) {
        std::vector<std::uint8_t> values;
        for (const auto& s : samples)
          values.push_back(static_cast<std::uint8_t>(s.value.toInt()));
        return c10::IValue(torch::tensor(values, torch::kUInt8));
      });
}
std::uint64_t Committed(const BatchPipeline& p) {
  if (const auto* m = dynamic_cast<const DataPipeline*>(&p))
    return m->state().next_cursor;
  return dynamic_cast<const IterableDataPipeline&>(p).state().committed_samples;
}
std::size_t Outstanding(const BatchPipeline& p) {
  if (const auto* m = dynamic_cast<const DataPipeline*>(&p))
    return m->outstanding_batches();
  return dynamic_cast<const IterableDataPipeline&>(p).outstanding_batches();
}
SavedBatchState Save(const BatchPipeline& p) {
  SavedBatchState s(p.describe_state());
  p.save_state(s);
  return s;
}

class BatchConsumerTransformTest : public testing::TestWithParam<int> {};

TEST_P(BatchConsumerTransformTest, RunsOnCallerWithUncommittedLogicalContext) {
  const auto caller = std::this_thread::get_id();
  auto stage = ScaleTransform();
  std::size_t calls = 0;
  BatchPipeline* raw = nullptr;
  stage.apply = [&](const c10::IValue& v, const BatchConsumerContext& c,
                    std::stop_token stop) {
    EXPECT_EQ(std::this_thread::get_id(), caller);
    EXPECT_FALSE(stop.stop_requested());
    EXPECT_EQ(c.dataset_version, "six-fixture/v1");
    EXPECT_EQ(c.epoch, 9U);
    EXPECT_EQ(c.sequence, calls);
    EXPECT_EQ(Committed(*raw), calls * 2U);
    EXPECT_EQ(c.sample_ids.size(), 2U);
    EXPECT_EQ(c.sample_ids[0], calls * 2U);
    EXPECT_EQ(c.source_watermark,
              GetParam() == 2 ? std::to_string(calls * 2U + 2U) : "");
    ++calls;
    return c10::IValue(v.toTensor().to(torch::kFloat32) * 0.5F);
  };
  auto p = Build(GetParam(), stage);
  raw = p.get();
  p->set_epoch(9);
  for (int batch = 0; batch < 3; ++batch) {
    const auto e = p->next();
    ASSERT_TRUE(e);
    EXPECT_EQ(e->consumer_transform_id, "scale/half/v1");
    EXPECT_EQ(e->schema.id, "scaled");
    EXPECT_TRUE(
        torch::equal(e->value.toTensor(),
                     torch::tensor({float(batch), float(batch) + 0.5F})));
    ASSERT_TRUE(e->transformed_memory);
    EXPECT_EQ(e->transformed_memory->tensor_bytes, 8U);
    EXPECT_EQ(e->transformed_memory->tensor_count, 1U);
    EXPECT_FALSE(e->memory.staged);
    EXPECT_EQ(e->memory.tensor_bytes, 0U);
  }
  EXPECT_FALSE(p->next());
  EXPECT_EQ(calls, 3U);
}

TEST_P(BatchConsumerTransformTest,
       InputMismatchStopsBeforeCallbackAndCanRestore) {
  auto stage = ScaleTransform();
  stage.input_schema = FloatsSchema();
  bool called = false;
  stage.apply = [&](const c10::IValue& v, const BatchConsumerContext&,
                    std::stop_token) {
    called = true;
    return v;
  };
  auto p = Build(GetParam(), stage);
  p->set_epoch(0);
  const auto state = Save(*p);
  EXPECT_THROW(p->next(), InvalidArgument);
  EXPECT_FALSE(called);
  EXPECT_EQ(Committed(*p), 0U);
  EXPECT_EQ(Outstanding(*p), 0U);
  EXPECT_THROW(p->next(), InvalidArgument);
  EXPECT_NO_THROW(p->load_state(state, RestorePolicy::exact_only));
  EXPECT_THROW(p->next(), InvalidArgument);
}

TEST_P(BatchConsumerTransformTest, OutputSchemaMismatchStopsWithoutCommit) {
  auto stage = ScaleTransform();
  stage.apply = [](const c10::IValue& v, const BatchConsumerContext&,
                   std::stop_token) { return v; };
  auto p = Build(GetParam(), stage);
  p->set_epoch(3);
  EXPECT_THROW(p->next(), InvalidArgument);
  EXPECT_EQ(Committed(*p), 0U);
  EXPECT_EQ(Outstanding(*p), 0U);
  p->set_epoch(4);
  EXPECT_THROW(p->next(), InvalidArgument);
}

TEST_P(BatchConsumerTransformTest,
       ExceptionAfterFirstBatchRestoresSameContextExactly) {
  auto stage = ScaleTransform();
  bool fail = true;
  stage.apply = [&](const c10::IValue& v, const BatchConsumerContext& c,
                    std::stop_token) {
    if (c.sequence == 1 && fail)
      throw std::runtime_error("injected precommit failure");
    return c10::IValue(v.toTensor().to(torch::kFloat32) * 0.5F);
  };
  auto p = Build(GetParam(), stage);
  p->set_epoch(7);
  ASSERT_TRUE(p->next());
  EXPECT_THROW(p->next(), BackendError);
  EXPECT_EQ(Committed(*p), 2U);
  EXPECT_EQ(Outstanding(*p), 0U);
  const auto saved = Save(*p);
  EXPECT_THROW(p->next(), BackendError);
  fail = false;
  p->load_state(saved, RestorePolicy::exact_only);
  auto fresh = Build(GetParam(), stage);
  fresh->load_state(saved, RestorePolicy::exact_only);
  while (const auto e = p->next()) {
    const auto replay = fresh->next();
    ASSERT_TRUE(replay);
    EXPECT_EQ(replay->epoch, 7U);
    EXPECT_EQ(e->sequence, replay->sequence);
    EXPECT_EQ(e->sample_ids, replay->sample_ids);
    EXPECT_EQ(e->source_watermark, replay->source_watermark);
    EXPECT_TRUE(torch::equal(e->value.toTensor(), replay->value.toTensor()));
  }
  EXPECT_FALSE(fresh->next());
  EXPECT_EQ(Committed(*p), 6U);
}

TEST_P(BatchConsumerTransformTest,
       PreservesStructuredAndWrapsNonstandardFailures) {
  for (const bool structured : {false, true}) {
    auto stage = ScaleTransform();
    stage.apply = [structured](const c10::IValue&, const BatchConsumerContext&,
                               std::stop_token) -> c10::IValue {
      if (structured)
        throw ResourceExhausted("injected transform resource failure");
      throw 17;
    };
    auto p = Build(GetParam(), stage);
    p->set_epoch(0);
    if (structured) {
      EXPECT_THROW(p->next(), ResourceExhausted);
    } else {
      EXPECT_THROW(p->next(), BackendError);
    }
    EXPECT_EQ(Committed(*p), 0U);
    EXPECT_EQ(Outstanding(*p), 0U);
  }
}

TEST_P(BatchConsumerTransformTest,
       CancellationDuringCallbackRequiresExplicitRestore) {
  std::stop_source cancel;
  bool request = true;
  auto stage = ScaleTransform();
  stage.apply = [&](const c10::IValue& v, const BatchConsumerContext&,
                    std::stop_token stop) {
    if (request) cancel.request_stop();
    EXPECT_EQ(stop.stop_requested(), request);
    return c10::IValue(v.toTensor().to(torch::kFloat32));
  };
  auto p = Build(GetParam(), stage);
  p->set_epoch(1);
  EXPECT_FALSE(p->next(cancel.get_token()));
  EXPECT_EQ(Committed(*p), 0U);
  EXPECT_EQ(Outstanding(*p), 0U);
  EXPECT_THROW(p->next(), Cancelled);
  const auto saved = Save(*p);
  request = false;
  p->load_state(saved, RestorePolicy::exact_only);
  ASSERT_TRUE(p->next());
  EXPECT_EQ(Committed(*p), 2U);
}

TEST_P(BatchConsumerTransformTest,
       ChecksOutputByteTensorAndNodeBoundsWithStagingOff) {
  for (int bound = 0; bound < 3; ++bound) {
    auto stage = ScaleTransform();
    BatchMemoryOptions limits;
    if (bound == 0)
      limits.max_bytes = 4;  // input 2 uint8 bytes fits; output 8 does not
    if (bound == 1) limits.max_tensors = 1;
    if (bound == 2) limits.max_value_nodes = 2;
    if (bound != 0) {
      stage.apply = [](const c10::IValue& v, const BatchConsumerContext&,
                       std::stop_token) {
        return c10::IValue(c10::ivalue::Tuple::create({v, v}));
      };
    }
    auto p = Build(GetParam(), stage, BatchSchema::any("bounded"), limits);
    p->set_epoch(0);
    EXPECT_THROW(p->next(), ResourceExhausted);
    EXPECT_EQ(Committed(*p), 0U);
    EXPECT_EQ(Outstanding(*p), 0U);
  }
}

TEST_P(BatchConsumerTransformTest, InspectsInputBeforeInvokingCallback) {
  auto stage = ScaleTransform();
  bool called = false;
  stage.apply = [&](const c10::IValue& v, const BatchConsumerContext&,
                    std::stop_token) {
    called = true;
    return v;
  };
  auto p = Build(GetParam(), stage, BytesSchema(), {.max_bytes = 1});
  p->set_epoch(0);
  EXPECT_THROW(p->next(), ResourceExhausted);
  EXPECT_FALSE(called);
  EXPECT_EQ(Committed(*p), 0U);
}

TEST_P(BatchConsumerTransformTest,
       CanChangeValueStructureBeforeFinalValidation) {
  auto stage = ScaleTransform();
  stage.apply = [](const c10::IValue& v, const BatchConsumerContext& c,
                   std::stop_token) {
    return c10::IValue(
        c10::ivalue::Tuple::create({v.toTensor().to(torch::kFloat32),
                                    static_cast<std::int64_t>(c.sequence)}));
  };
  const BatchSchema output{
      .id = "tensor-and-sequence",
      .value = ValueSchema::tuple(
          {ValueSchema::tensor(), ValueSchema{.kind = ValueKind::integer}})};
  auto p = Build(GetParam(), stage, output);
  p->set_epoch(0);
  const auto e = p->next();
  ASSERT_TRUE(e);
  EXPECT_TRUE(e->value.isTuple());
  EXPECT_EQ(e->value.toTupleRef().elements()[1].toInt(), 0);
  ASSERT_TRUE(e->transformed_memory);
  EXPECT_EQ(e->transformed_memory->tensor_bytes, 8U);
  EXPECT_EQ(e->transformed_memory->value_node_count, 3U);
}

TEST_P(BatchConsumerTransformTest,
       RejectsChangedTransformContractWithoutMutation) {
  auto original = Build(GetParam());
  original->set_epoch(2);
  ASSERT_TRUE(original->next());
  const auto saved = Save(*original);
  for (int change = 0; change < 4; ++change) {
    auto stage = ScaleTransform();
    if (change == 0) stage.id = "scale/other/v2";
    if (change == 1) stage.input_schema.value.semantic_tag = "changed-range";
    if (change == 2) stage.deterministic = false;
    auto target = change == 3 ? Build(GetParam(), std::nullopt)
                              : Build(GetParam(), stage);
    target->set_epoch(8);
    EXPECT_THROW(target->load_state(saved, RestorePolicy::allow_bounded),
                 CompatibilityError);
    EXPECT_EQ(Committed(*target), 0U);
  }
}

TEST_P(BatchConsumerTransformTest,
       NondeterministicStageOrFactoryCannotClaimExact) {
  for (bool factory : {false, true}) {
    auto stage = ScaleTransform();
    stage.deterministic = factory;
    auto p = Build(GetParam(), stage, FloatsSchema(), {}, !factory);
    EXPECT_EQ(p->capability().restore_level, RestoreLevel::bounded);
    EXPECT_FALSE(p->capability().degradation_reasons.empty());
    p->set_epoch(0);
    const auto saved = Save(*p);
    EXPECT_EQ(saved.manifest().restore_level, RestoreLevel::bounded);
    EXPECT_THROW(p->load_state(saved, RestorePolicy::exact_only),
                 CompatibilityError);
    EXPECT_NO_THROW(p->load_state(saved, RestorePolicy::allow_bounded));
    EXPECT_NO_THROW(describe_data_capability(p->capability()));
  }
}

TEST_P(BatchConsumerTransformTest,
       RejectsMalformedTransformStateWithoutReplacingLiveEpoch) {
  auto source = Build(GetParam());
  source->set_epoch(0);
  const auto saved = Save(*source);
  for (int corruption = 0; corruption < 4; ++corruption) {
    auto bad = saved;
    bad.corrupt_topology(corruption);
    auto target = Build(GetParam());
    target->set_epoch(5);
    EXPECT_THROW(target->load_state(bad, RestorePolicy::exact_only), Error);
    EXPECT_EQ(Committed(*target), 0U);
    const auto next = target->next();
    ASSERT_TRUE(next);
    EXPECT_EQ(next->epoch, 5U);
  }
}

TEST_P(BatchConsumerTransformTest,
       CapabilityBindsInputOutputAndTransformPolicy) {
  auto p = Build(GetParam());
  const auto c = p->capability();
  EXPECT_EQ(c.consumer_transform_id, "scale/half/v1");
  ASSERT_TRUE(c.consumer_input_schema);
  EXPECT_EQ(c.consumer_input_schema->id, "bytes");
  EXPECT_EQ(c.batch_schema.id, "scaled");
  const auto d = describe_data_capability(c);
  EXPECT_NE(std::find(d.operations.begin(), d.operations.end(),
                      "batch.transform.consumer-precommit"),
            d.operations.end());
  auto invalid = c;
  invalid.consumer_input_schema.reset();
  EXPECT_THROW(describe_data_capability(invalid), InvalidArgument);
  invalid = c;
  invalid.consumer_transform_id.clear();
  EXPECT_THROW(describe_data_capability(invalid), InvalidArgument);
  invalid = c;
  invalid.deterministic_consumer_transform = false;
  EXPECT_THROW(describe_data_capability(invalid), InvalidArgument);
}

TEST_P(BatchConsumerTransformTest,
       ValidatesCallbackIdentityAndInputSchemaAtConstruction) {
  for (int failure = 0; failure < 4; ++failure) {
    auto stage = ScaleTransform();
    if (failure == 0) stage.id.clear();
    if (failure == 1) stage.apply = {};
    if (failure == 2) stage.input_schema.id.clear();
    if (failure == 3) stage.id.assign(4097, 'x');
    if (failure == 3) {
      EXPECT_THROW(Build(GetParam(), stage), ResourceExhausted);
    } else {
      EXPECT_THROW(Build(GetParam(), stage), InvalidArgument);
    }
  }
}

TEST_P(BatchConsumerTransformTest,
       DisabledStageLeavesOriginalEnvelopeUntransformed) {
  auto p = Build(GetParam(), std::nullopt, BytesSchema());
  p->set_epoch(0);
  const auto e = p->next();
  ASSERT_TRUE(e);
  EXPECT_TRUE(e->consumer_transform_id.empty());
  EXPECT_FALSE(e->transformed_memory);
  EXPECT_EQ(e->value.toTensor().scalar_type(), torch::kUInt8);
}

INSTANTIATE_TEST_SUITE_P(AllPipelines, BatchConsumerTransformTest,
                         testing::Values(0, 1, 2),
                         [](const testing::TestParamInfo<int>& p) {
                           return p.param == 0 ? "InlineMap"
                                               : (p.param == 1 ? "WorkerMap"
                                                               : "Iterable");
                         });

TEST(BatchMemoryInspectionTest, CountsOccurrencesWithoutCopyingOrRequiringCpu) {
  const auto tensor = torch::ones({2, 3}, torch::kFloat64);
  const auto value = c10::IValue(
      c10::ivalue::Tuple::create({tensor, tensor, std::int64_t{4}}));
  const auto usage = inspect_batch_memory(value);
  EXPECT_EQ(usage.tensor_count, 2U);
  EXPECT_EQ(usage.value_node_count, 4U);
  EXPECT_EQ(usage.tensor_bytes, 96U);
  EXPECT_EQ(value.toTupleRef().elements()[0].toTensor().data_ptr(),
            tensor.data_ptr());
  EXPECT_THROW(inspect_batch_memory(value, {.max_bytes = 95}),
               ResourceExhausted);
}
}  // namespace
