// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/nn/model_checkpoint_participant.hpp>

#include "model_checkpoint/record_function_observer.hpp"
#include <c10/core/StorageImpl.h>
#ifdef JYPPX_LIUDING_NN_CUDA_METRICS
#include <c10/cuda/CUDACachingAllocator.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

class MemoryStateStore final : public jyppx::liuding::StateWriter,
                               public jyppx::liuding::StateReader {
 public:
  explicit MemoryStateStore(jyppx::liuding::StateManifest manifest)
      : manifest_(std::move(manifest)) {
    jyppx::liuding::validate_state_manifest(manifest_);
  }

  void write_blob(std::string_view name,
                  std::span<const std::byte> data) override {
    const auto field = std::find_if(
        manifest_.fields.begin(), manifest_.fields.end(),
        [name](const jyppx::liuding::StateFieldDescriptor& candidate) {
          return candidate.name == name;
        });
    if (field == manifest_.fields.end()) {
      throw jyppx::liuding::SerializationError("unknown model test field");
    }
    if (data.size() > field->max_bytes) {
      throw jyppx::liuding::ResourceExhausted("oversized model test field");
    }
    if (!blobs_
             .emplace(std::string(name),
                      std::vector<std::byte>(data.begin(), data.end()))
             .second) {
      throw jyppx::liuding::SerializationError("duplicate model test field");
    }
  }

  const jyppx::liuding::StateManifest& manifest() const noexcept override {
    return manifest_;
  }

  bool contains(std::string_view name) const override {
    return blobs_.contains(std::string(name));
  }

  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t max_bytes) const override {
    const auto blob = blobs_.find(std::string(name));
    if (blob == blobs_.end()) {
      throw jyppx::liuding::SerializationError("missing model test field");
    }
    if (blob->second.size() > max_bytes) {
      throw jyppx::liuding::ResourceExhausted("model test read limit exceeded");
    }
    return blob->second;
  }

  std::vector<std::byte>& mutable_blob(std::string_view name) {
    return blobs_.at(std::string(name));
  }

  jyppx::liuding::StateManifest& mutable_manifest() { return manifest_; }

 private:
  jyppx::liuding::StateManifest manifest_;
  std::map<std::string, std::vector<std::byte>> blobs_;
};

class ProbeModelImpl final : public torch::nn::Module {
 public:
  ProbeModelImpl()
      : linear(register_module("linear", torch::nn::Linear(2, 2))),
        offset(register_buffer("offset", torch::tensor({1.0F, 2.0F}))),
        mask(register_buffer("mask",
                             torch::tensor({true, false}, torch::kBool))) {}

  torch::nn::Linear linear;
  torch::Tensor offset;
  torch::Tensor mask;
};

class AlternateModelImpl final : public torch::nn::Module {
 public:
  AlternateModelImpl()
      : branch(register_module("branch", torch::nn::Linear(2, 2))),
        shiftx(register_buffer("shiftx", torch::tensor({3.0F, 4.0F}))),
        flag(register_buffer("flag",
                             torch::tensor({false, true}, torch::kBool))) {}

  torch::nn::Linear branch;
  torch::Tensor shiftx;
  torch::Tensor flag;
};

class AliasedBufferModelImpl final : public torch::nn::Module {
 public:
  AliasedBufferModelImpl() {
    base = register_buffer("base", torch::arange(4, torch::kFloat32));
    matrix = register_buffer("matrix", base.view({2, 2}));
  }

  torch::Tensor base;
  torch::Tensor matrix;
};

class OverlappingBufferModelImpl final : public torch::nn::Module {
 public:
  OverlappingBufferModelImpl() {
    expanded = register_buffer("expanded", torch::zeros({1}).expand({4}));
  }

  torch::Tensor expanded;
};

class EmptyModelImpl final : public torch::nn::Module {};

std::shared_ptr<jyppx::liuding::nn::ModelStateView> state_for(
    const std::shared_ptr<torch::nn::Module>& module) {
  return std::make_shared<jyppx::liuding::nn::ModelStateView>(module);
}

MemoryStateStore save_participant(
    const jyppx::liuding::nn::ModelCheckpointParticipant& participant) {
  MemoryStateStore store(participant.describe_state());
  participant.save_state(store);
  return store;
}

TEST(ModelCheckpointParticipantTest,
     DescribesCanonicalRawFieldsWithoutExecutableArchive) {
  auto model = std::make_shared<ProbeModelImpl>();
  jyppx::liuding::nn::ModelCheckpointParticipant participant(state_for(model));

  const auto manifest = participant.describe_state();
  ASSERT_EQ(manifest.participant_id, "model.main");
  ASSERT_EQ(manifest.schema_version, 1U);
  ASSERT_EQ(manifest.restore_level, jyppx::liuding::RestoreLevel::exact);
  ASSERT_EQ(manifest.fields.size(), 5U);
  EXPECT_EQ(manifest.fields.front().name, "topology");
  EXPECT_EQ(manifest.fields[1].name, "tensor.0000000000");
  EXPECT_EQ(manifest.fields.back().name, "tensor.0000000003");

  const auto store = save_participant(participant);
  const auto topology = store.read_blob("topology", 1024 * 1024);
  const std::string magic(reinterpret_cast<const char*>(topology.data()), 8);
  EXPECT_EQ(magic, "JYPPXMDL");
  const auto weight = store.read_blob("tensor.0000000000", 1024);
  EXPECT_EQ(weight.size(), 4U * sizeof(float));
  ASSERT_GE(topology.size(), 2U);
  EXPECT_FALSE(topology[0] == std::byte{'P'} && topology[1] == std::byte{'K'});
}

TEST(ModelCheckpointParticipantTest,
     RoundTripsParametersBuffersAndPreservesMode) {
  auto model = std::make_shared<ProbeModelImpl>();
  model->eval();
  model->linear->weight.detach().copy_(
      torch::tensor({{1.0F, 2.0F}, {3.0F, 4.0F}}));
  model->linear->bias.detach().copy_(torch::tensor({5.0F, 6.0F}));
  model->offset.copy_(torch::tensor({7.0F, 8.0F}));
  model->mask.copy_(torch::tensor({true, false}, torch::kBool));
  jyppx::liuding::nn::ModelCheckpointParticipant participant(state_for(model));
  const auto store = save_participant(participant);

  model->linear->weight.detach().zero_();
  model->linear->bias.detach().zero_();
  model->offset.zero_();
  model->mask.fill_(true);
  participant.load_state(store, jyppx::liuding::RestorePolicy::exact_only);

  EXPECT_TRUE(torch::equal(model->linear->weight,
                           torch::tensor({{1.0F, 2.0F}, {3.0F, 4.0F}})));
  EXPECT_TRUE(torch::equal(model->linear->bias, torch::tensor({5.0F, 6.0F})));
  EXPECT_TRUE(torch::equal(model->offset, torch::tensor({7.0F, 8.0F})));
  EXPECT_TRUE(
      torch::equal(model->mask, torch::tensor({true, false}, torch::kBool)));
  EXPECT_FALSE(model->is_training());
}

TEST(ModelCheckpointParticipantTest,
     RejectsDifferentStateKeysBeforeMutatingTheModel) {
  auto original = std::make_shared<ProbeModelImpl>();
  jyppx::liuding::nn::ModelCheckpointParticipant source(state_for(original));
  const auto store = save_participant(source);

  auto alternate = std::make_shared<AlternateModelImpl>();
  const auto before_weight = alternate->branch->weight.detach().clone();
  const auto before_shift = alternate->shiftx.clone();
  jyppx::liuding::nn::ModelCheckpointParticipant target(state_for(alternate));
  const auto target_manifest = target.describe_state();
  const auto source_manifest = source.describe_state();
  ASSERT_EQ(target_manifest.fields.size(), source_manifest.fields.size());
  for (std::size_t index = 0; index < target_manifest.fields.size(); ++index) {
    EXPECT_EQ(target_manifest.fields[index].name,
              source_manifest.fields[index].name);
    EXPECT_EQ(target_manifest.fields[index].max_bytes,
              source_manifest.fields[index].max_bytes);
  }

  EXPECT_THROW(
      target.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::CompatibilityError);
  EXPECT_TRUE(torch::equal(alternate->branch->weight, before_weight));
  EXPECT_TRUE(torch::equal(alternate->shiftx, before_shift));
}

TEST(ModelCheckpointParticipantTest,
     RejectsMalformedTopologyAndTensorBytesWithoutMutation) {
  auto model = std::make_shared<ProbeModelImpl>();
  jyppx::liuding::nn::ModelCheckpointParticipant participant(state_for(model));
  auto store = save_participant(participant);
  model->linear->weight.detach().fill_(9.0F);
  const auto before = model->linear->weight.detach().clone();

  store.mutable_blob("topology").front() = std::byte{'X'};
  EXPECT_THROW(
      participant.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::SerializationError);
  EXPECT_TRUE(torch::equal(model->linear->weight, before));

  store = save_participant(participant);
  store.mutable_blob("tensor.0000000000").pop_back();
  EXPECT_THROW(
      participant.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::SerializationError);
  EXPECT_TRUE(torch::equal(model->linear->weight, before));
}

TEST(ModelCheckpointParticipantTest,
     RejectsMalformedBooleanBytesAndManifestBeforeMutation) {
  auto model = std::make_shared<ProbeModelImpl>();
  jyppx::liuding::nn::ModelCheckpointParticipant participant(state_for(model));
  auto store = save_participant(participant);
  model->mask.fill_(false);
  const auto before = model->mask.clone();

  store.mutable_blob("tensor.0000000003").front() = std::byte{2};
  EXPECT_THROW(
      participant.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::SerializationError);
  EXPECT_TRUE(torch::equal(model->mask, before));

  store = save_participant(participant);
  store.mutable_manifest().schema_version = 2;
  EXPECT_THROW(
      participant.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::CompatibilityError);
  EXPECT_TRUE(torch::equal(model->mask, before));
}

TEST(ModelCheckpointParticipantTest,
     RestoresAliasedViewsWithoutChangingStorageTopology) {
  auto model = std::make_shared<AliasedBufferModelImpl>();
  ASSERT_TRUE(model->base.is_alias_of(model->matrix));
  jyppx::liuding::nn::ModelCheckpointParticipant participant(state_for(model));
  const auto store = save_participant(participant);

  model->base.fill_(10.0F);
  participant.load_state(store, jyppx::liuding::RestorePolicy::exact_only);

  EXPECT_TRUE(
      torch::equal(model->base, torch::tensor({0.0F, 1.0F, 2.0F, 3.0F})));
  EXPECT_TRUE(
      torch::equal(model->matrix, torch::tensor({{0.0F, 1.0F}, {2.0F, 3.0F}})));
  EXPECT_TRUE(model->base.is_alias_of(model->matrix));
}

TEST(ModelCheckpointParticipantTest,
     EnforcesOptionsAndRejectsUnsupportedModelState) {
  auto model = std::make_shared<ProbeModelImpl>();
  EXPECT_THROW(jyppx::liuding::nn::ModelCheckpointParticipant(
                   state_for(model), {.max_tensors = 2}),
               jyppx::liuding::ResourceExhausted);
  EXPECT_THROW(jyppx::liuding::nn::ModelCheckpointParticipant(
                   state_for(model), {.max_tensors = 0}),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(jyppx::liuding::nn::ModelCheckpointParticipant(
                   state_for(std::make_shared<EmptyModelImpl>())),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(jyppx::liuding::nn::ModelCheckpointParticipant(
                   state_for(std::make_shared<OverlappingBufferModelImpl>())),
               jyppx::liuding::InvalidArgument);
}

TEST(ModelCheckpointParticipantTest,
     RetainsAdapterStateAndRejectsUnknownRestorePolicy) {
  auto model = std::make_shared<ProbeModelImpl>();
  std::weak_ptr<ProbeModelImpl> lifetime = model;
  auto adapter = jyppx::liuding::nn::adapt_module(
      model,
      [](ProbeModelImpl&, const jyppx::liuding::nn::Invocation&) {
        return c10::IValue(torch::zeros({1}));
      },
      {}, jyppx::liuding::ValueSchema::tensor());
  jyppx::liuding::nn::ModelCheckpointParticipant participant(adapter);
  model.reset();
  adapter = jyppx::liuding::nn::adapt_module(
      std::make_shared<ProbeModelImpl>(),
      [](ProbeModelImpl&, const jyppx::liuding::nn::Invocation&) {
        return c10::IValue(torch::zeros({1}));
      },
      {}, jyppx::liuding::ValueSchema::tensor());
  ASSERT_FALSE(lifetime.expired());
  const auto store = save_participant(participant);

  EXPECT_THROW(participant.load_state(
                   store, static_cast<jyppx::liuding::RestorePolicy>(255)),
               jyppx::liuding::CompatibilityError);
}

std::map<std::string, std::vector<std::byte>> raw_model_fields(
    const MemoryStateStore& store) {
  std::map<std::string, std::vector<std::byte>> result;
  for (const auto& field : store.manifest().fields) {
    const auto bytes = store.read_blob(field.name, field.max_bytes);
    result.emplace(field.name,
                   std::vector<std::byte>(bytes.begin(), bytes.end()));
  }
  return result;
}

class DenseScalarBufferModelImpl final : public torch::nn::Module {
 public:
  DenseScalarBufferModelImpl() {
    for (const auto dtype :
         {c10::ScalarType::Byte, c10::ScalarType::Char, c10::ScalarType::Short,
          c10::ScalarType::Int, c10::ScalarType::Long, c10::ScalarType::Half,
          c10::ScalarType::Float, c10::ScalarType::Double,
          c10::ScalarType::ComplexHalf, c10::ScalarType::ComplexFloat,
          c10::ScalarType::ComplexDouble, c10::ScalarType::Bool,
          c10::ScalarType::BFloat16}) {
      const auto name = "scalar_" + std::to_string(static_cast<int>(dtype));
      register_buffer(name, torch::arange(6, torch::kFloat64)
                                .to(dtype)
                                .reshape({2, 3})
                                .transpose(0, 1));
    }
    register_buffer("empty", torch::empty({0, 3}, torch::kFloat64));
  }
};

TEST(ModelCheckpointParticipantTest,
     RoundTripsDenseScalarTypesEmptyTensorsAndNoncontiguousViewsExactly) {
  auto model = std::make_shared<DenseScalarBufferModelImpl>();
  model->eval();
  jyppx::liuding::nn::ModelCheckpointParticipant participant(state_for(model));
  const auto saved = save_participant(participant);
  const auto expected = raw_model_fields(saved);
  for (const auto& item : model->named_buffers()) item.value().zero_();

  participant.load_state(saved, jyppx::liuding::RestorePolicy::exact_only);

  EXPECT_EQ(raw_model_fields(save_participant(participant)), expected);
  EXPECT_FALSE(model->is_training());
  for (const auto& item : model->named_buffers()) {
    if (item.key() != "empty") EXPECT_FALSE(item.value().is_contiguous());
  }
}

TEST(ModelCheckpointParticipantTest,
     InvalidLastTensorPreservesEveryParameterBufferAndTopology) {
  auto model = std::make_shared<ProbeModelImpl>();
  model->eval();
  jyppx::liuding::nn::ModelCheckpointParticipant participant(state_for(model));
  auto saved = save_participant(participant);
  model->linear->weight.detach().fill_(11.0F);
  model->linear->bias.detach().fill_(13.0F);
  model->offset.fill_(17.0F);
  model->mask.fill_(false);
  const auto before = raw_model_fields(save_participant(participant));
  const auto last = saved.manifest().fields.back().name;
  saved.mutable_blob(last).pop_back();

  EXPECT_THROW(
      participant.load_state(saved, jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::SerializationError);

  EXPECT_EQ(raw_model_fields(save_participant(participant)), before);
  EXPECT_FALSE(model->is_training());
}

TEST(ModelCheckpointParticipantTest,
     MalformedTopologyCannotPartiallyRestoreAnyModelState) {
  auto model = std::make_shared<ProbeModelImpl>();
  model->eval();
  jyppx::liuding::nn::ModelCheckpointParticipant participant(state_for(model));
  const auto source = save_participant(participant);
  model->linear->weight.detach().fill_(19.0F);
  model->linear->bias.detach().fill_(23.0F);
  model->offset.fill_(29.0F);
  model->mask.fill_(false);
  const auto before = raw_model_fields(save_participant(participant));
  enum class Rejection { malformed, incompatible, resource };
  struct Mutation {
    const char* name;
    std::size_t offset;
    std::size_t width;
    std::uint64_t value;
    Rejection rejection;
  };
  // The version-1 wire header is magic[8], u32 version, u32 count.
  // Each entry begins role/u8, dtype/u8, grad/u8, reserved/u8,
  // key-length/u32, rank/u32, alias-group/u32, offset/i64, bytes/u64.
  const std::vector<Mutation> mutations{
      {"future version", 8, 4, 2, Rejection::incompatible},
      {"zero tensors", 12, 4, 0, Rejection::malformed},
      {"excessive tensors", 12, 4, 0xffffffffU, Rejection::resource},
      {"unknown role", 16, 1, 255, Rejection::malformed},
      {"unknown dtype", 17, 1, 255, Rejection::malformed},
      {"nonboolean grad flag", 18, 1, 2, Rejection::malformed},
      {"unknown reserved flag", 19, 1, 1, Rejection::malformed},
      {"unbounded key", 20, 4, 0xffffffffU, Rejection::resource},
      {"unbounded rank", 24, 4, 0xffffffffU, Rejection::resource},
      {"forward alias reference", 28, 4, 1, Rejection::malformed},
      {"negative storage offset", 32, 8, 0xffffffffffffffffULL,
       Rejection::malformed},
      {"inconsistent tensor bytes", 40, 8, 17, Rejection::malformed},
      {"unbounded tensor bytes", 40, 8, 0xffffffffffffffffULL,
       Rejection::resource},
      {"embedded NUL in key", 48, 1, 0, Rejection::malformed},
  };
  for (const auto& mutation : mutations) {
    SCOPED_TRACE(mutation.name);
    auto corrupted = source;
    auto& bytes = corrupted.mutable_blob("topology");
    ASSERT_LE(mutation.offset + mutation.width, bytes.size());
    for (std::size_t i = 0; i < mutation.width; ++i) {
      bytes[mutation.offset + i] =
          static_cast<std::byte>(mutation.value >> (8 * i));
    }
    const auto restore = [&] {
      participant.load_state(corrupted,
                             jyppx::liuding::RestorePolicy::exact_only);
    };
    switch (mutation.rejection) {
      case Rejection::malformed:
        EXPECT_THROW(restore(), jyppx::liuding::SerializationError);
        break;
      case Rejection::incompatible:
        EXPECT_THROW(restore(), jyppx::liuding::CompatibilityError);
        break;
      case Rejection::resource:
        EXPECT_THROW(restore(), jyppx::liuding::ResourceExhausted);
        break;
    }
    EXPECT_EQ(raw_model_fields(save_participant(participant)), before);
    EXPECT_FALSE(model->is_training());
  }
}

TEST(ModelCheckpointParticipantTest,
     ExactResourceBoundsAreAcceptedAndSmallerLimitsPreserveTheModel) {
  auto model = std::make_shared<ProbeModelImpl>();
  auto view = state_for(model);
  jyppx::liuding::nn::ModelCheckpointParticipant participant(view);
  const auto& constant = participant;
  ASSERT_EQ(&participant.state(), view.get());
  ASSERT_EQ(&constant.state(), view.get());
  const auto source = save_participant(participant);
  const auto before = raw_model_fields(source);
  auto limits = participant.options();
  limits.max_tensors = participant.state().named_parameters().size() +
                       constant.state().named_buffers().size();
  limits.max_state_key_bytes = 1;
  for (const auto& item : participant.state().named_parameters()) {
    limits.max_state_key_bytes =
        std::max(limits.max_state_key_bytes, item.key().size());
  }
  for (const auto& item : constant.state().named_buffers()) {
    limits.max_state_key_bytes =
        std::max(limits.max_state_key_bytes, item.key().size());
  }
  limits.max_topology_bytes = source.manifest().fields.front().max_bytes;
  limits.max_tensor_bytes = 1;
  limits.max_total_bytes = 0;
  for (const auto& field : source.manifest().fields) {
    limits.max_total_bytes += field.max_bytes;
    if (field.name != "topology") {
      limits.max_tensor_bytes =
          std::max(limits.max_tensor_bytes, field.max_bytes);
    }
  }
  jyppx::liuding::nn::ModelCheckpointParticipant bounded(view, limits);
  EXPECT_EQ(raw_model_fields(save_participant(bounded)), before);
  bounded.load_state(source, jyppx::liuding::RestorePolicy::exact_only);
  EXPECT_EQ(raw_model_fields(save_participant(bounded)), before);

  auto smaller = limits;
  --smaller.max_tensors;
  EXPECT_THROW(jyppx::liuding::nn::ModelCheckpointParticipant(view, smaller),
               jyppx::liuding::ResourceExhausted);
  smaller = limits;
  --smaller.max_state_key_bytes;
  EXPECT_THROW(jyppx::liuding::nn::ModelCheckpointParticipant(view, smaller),
               jyppx::liuding::InvalidArgument);
  smaller = limits;
  --smaller.max_topology_bytes;
  EXPECT_THROW(jyppx::liuding::nn::ModelCheckpointParticipant(view, smaller),
               jyppx::liuding::ResourceExhausted);
  smaller = limits;
  --smaller.max_tensor_bytes;
  EXPECT_THROW(jyppx::liuding::nn::ModelCheckpointParticipant(view, smaller),
               jyppx::liuding::ResourceExhausted);
  smaller = limits;
  --smaller.max_total_bytes;
  EXPECT_THROW(jyppx::liuding::nn::ModelCheckpointParticipant(view, smaller),
               jyppx::liuding::ResourceExhausted);
  EXPECT_EQ(raw_model_fields(save_participant(participant)), before);
}

enum class StoreFault { allocation, backend, missing };

[[noreturn]] void raise_store_fault(StoreFault fault) {
  if (fault == StoreFault::allocation) throw std::bad_alloc();
  TORCH_CHECK(false, "injected checkpoint store backend failure");
}

class FaultingStateWriter final : public jyppx::liuding::StateWriter {
 public:
  FaultingStateWriter(jyppx::liuding::StateManifest manifest, StoreFault fault)
      : store_(std::move(manifest)), fault_(fault) {}

  void write_blob(std::string_view name,
                  std::span<const std::byte> bytes) override {
    if (name == "tensor.0000000001") raise_store_fault(fault_);
    store_.write_blob(name, bytes);
    ++completed_writes;
  }

  std::size_t completed_writes = 0;

 private:
  MemoryStateStore store_;
  StoreFault fault_;
};

class FaultingStateReader final : public jyppx::liuding::StateReader {
 public:
  FaultingStateReader(const MemoryStateStore& store, StoreFault fault)
      : store_(store),
        fault_(fault),
        last_(store.manifest().fields.back().name) {}

  const jyppx::liuding::StateManifest& manifest() const noexcept override {
    return store_.manifest();
  }

  bool contains(std::string_view name) const override {
    return !(fault_ == StoreFault::missing && name == last_) &&
           store_.contains(name);
  }

  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t max_bytes) const override {
    if (name == last_) raise_store_fault(fault_);
    const auto bytes = store_.read_blob(name, max_bytes);
    ++completed_reads;
    return bytes;
  }

  mutable std::size_t completed_reads = 0;

 private:
  const MemoryStateStore& store_;
  StoreFault fault_;
  std::string last_;
};

TEST(ModelCheckpointParticipantTest,
     StoreFailuresRetainTypedDiagnosticsAndEveryModelField) {
  for (const auto fault : {StoreFault::allocation, StoreFault::backend}) {
    for (const auto saving : {true, false}) {
      SCOPED_TRACE(static_cast<int>(fault));
      SCOPED_TRACE(saving);
      auto model = std::make_shared<ProbeModelImpl>();
      model->eval();
      jyppx::liuding::nn::ModelCheckpointParticipant participant(
          state_for(model));
      const auto saved = save_participant(participant);
      model->linear->weight.detach().fill_(31.0F);
      model->linear->bias.detach().fill_(37.0F);
      model->offset.fill_(41.0F);
      model->mask.fill_(false);
      const auto before = raw_model_fields(save_participant(participant));
      FaultingStateWriter writer(participant.describe_state(), fault);
      FaultingStateReader reader(saved, fault);

      try {
        if (saving)
          participant.save_state(writer);
        else
          participant.load_state(reader,
                                 jyppx::liuding::RestorePolicy::exact_only);
        FAIL() << "store failure must be reported";
      } catch (const jyppx::liuding::ResourceExhausted& error) {
        EXPECT_EQ(fault, StoreFault::allocation);
        EXPECT_EQ(error.details().component, "nn.model_checkpoint");
        EXPECT_EQ(error.details().operation, saving ? "save" : "restore");
        EXPECT_EQ(error.details().actual, "allocation failed");
        EXPECT_FALSE(error.details().hint.empty());
      } catch (const jyppx::liuding::BackendError& error) {
        EXPECT_EQ(fault, StoreFault::backend);
        EXPECT_EQ(error.details().component, "nn.model_checkpoint");
        EXPECT_EQ(error.details().operation,
                  saving ? "save model state" : "restore model state");
        bool nested_backend = false;
        try {
          std::rethrow_if_nested(error);
        } catch (const c10::Error& cause) {
          nested_backend = true;
          EXPECT_NE(std::string(cause.what())
                        .find("injected checkpoint store backend failure"),
                    std::string::npos);
        }
        EXPECT_TRUE(nested_backend);
      }

      EXPECT_EQ(raw_model_fields(save_participant(participant)), before);
      EXPECT_FALSE(model->is_training());
      if (saving)
        EXPECT_EQ(writer.completed_writes, 2U);
      else
        EXPECT_EQ(reader.completed_reads, 4U);
    }
  }
}

TEST(ModelCheckpointParticipantTest,
     EveryTopologyTruncationAndMissingFieldPreservesTheWholeModel) {
  auto model = std::make_shared<ProbeModelImpl>();
  model->eval();
  jyppx::liuding::nn::ModelCheckpointParticipant participant(state_for(model));
  const auto saved = save_participant(participant);
  model->linear->weight.detach().fill_(43.0F);
  model->linear->bias.detach().fill_(47.0F);
  model->offset.fill_(53.0F);
  model->mask.fill_(false);
  const auto before = raw_model_fields(save_participant(participant));
  const auto size = saved.read_blob("topology", 1024 * 1024).size();
  for (std::size_t prefix = 0; prefix < size; ++prefix) {
    SCOPED_TRACE(prefix);
    auto truncated = saved;
    truncated.mutable_blob("topology").resize(prefix);
    EXPECT_THROW(participant.load_state(
                     truncated, jyppx::liuding::RestorePolicy::exact_only),
                 jyppx::liuding::SerializationError);
    EXPECT_EQ(raw_model_fields(save_participant(participant)), before);
    EXPECT_FALSE(model->is_training());
  }
  FaultingStateReader missing(saved, StoreFault::missing);
  EXPECT_THROW(participant.load_state(
                   missing, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::SerializationError);
  EXPECT_EQ(missing.completed_reads, 0U);
  EXPECT_EQ(raw_model_fields(save_participant(participant)), before);
  EXPECT_FALSE(model->is_training());
}

TEST(ModelCheckpointParticipantTest,
     RejectsZeroAndUnrepresentableWireLimitsBeforeModelAllocation) {
  auto model = std::make_shared<ProbeModelImpl>();
  auto view = state_for(model);
  jyppx::liuding::nn::ModelCheckpointParticipant participant(view);
  const auto before = raw_model_fields(save_participant(participant));
  using Options = jyppx::liuding::nn::ModelCheckpointParticipantOptions;
  const auto above_wire =
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) + 1;
  for (const auto& options :
       {Options{.max_tensors = 0}, Options{.max_state_key_bytes = 0},
        Options{.max_topology_bytes = 0}, Options{.max_tensor_bytes = 0},
        Options{.max_total_bytes = 0}, Options{.max_tensors = above_wire},
        Options{.max_state_key_bytes = above_wire}}) {
    EXPECT_THROW(jyppx::liuding::nn::ModelCheckpointParticipant(view, options),
                 jyppx::liuding::InvalidArgument);
    EXPECT_EQ(raw_model_fields(save_participant(participant)), before);
  }
}

class StridedAliasModelImpl final : public torch::nn::Module {
 public:
  StridedAliasModelImpl(torch::Device device, torch::ScalarType dtype) {
    const auto options = torch::TensorOptions().device(device).dtype(dtype);
    base = register_buffer("base", torch::arange(12, options));
    matrix = register_buffer("matrix", base.view({3, 4}).transpose(0, 1));
    offset = register_buffer("offset", base.narrow(0, 2, 4));
    empty = register_buffer("empty", torch::empty({0}, options));
  }
  torch::Tensor base, matrix, offset, empty;
};

void expect_strided_alias_round_trip(torch::Device device) {
  for (const auto dtype :
       {torch::kFloat32, torch::kFloat16, torch::kBFloat16, torch::kFloat64}) {
    SCOPED_TRACE(static_cast<int>(dtype));
    auto model = std::make_shared<StridedAliasModelImpl>(device, dtype);
    jyppx::liuding::nn::ModelCheckpointParticipant participant(
        state_for(model));
    const auto source = save_participant(participant);
    const auto before = raw_model_fields(source);
    const auto identity = model->base.unsafeGetTensorImpl();
    const auto storage = model->base.storage().unsafeGetStorageImpl();
    const auto strides = model->matrix.strides().vec();
    ASSERT_FALSE(model->matrix.is_contiguous());
    ASSERT_EQ(model->offset.storage_offset(), 2);
    model->base.add_(100);
    participant.load_state(source, jyppx::liuding::RestorePolicy::exact_only);
    EXPECT_EQ(raw_model_fields(save_participant(participant)), before);
    EXPECT_EQ(model->base.unsafeGetTensorImpl(), identity);
    EXPECT_EQ(model->base.storage().unsafeGetStorageImpl(), storage);
    EXPECT_EQ(model->matrix.strides().vec(), strides);
    EXPECT_EQ(model->offset.storage_offset(), 2);
    EXPECT_TRUE(model->base.is_alias_of(model->matrix));
    EXPECT_TRUE(model->base.is_alias_of(model->offset));
    EXPECT_EQ(model->base.device(), device);
    EXPECT_EQ(model->base.scalar_type(), dtype);
    EXPECT_EQ(model->empty.numel(), 0);
  }
}

TEST(ModelCheckpointParticipantTest,
     CpuRestorePreservesStridesOffsetsAndAliases) {
  expect_strided_alias_round_trip(torch::Device(torch::kCPU));
}

TEST(ModelCheckpointParticipantTest,
     CudaRestorePreservesStridesOffsetsAndAliases) {
  if (!torch::cuda::is_available())
    GTEST_SKIP() << "CUDA device is unavailable";
  expect_strided_alias_round_trip(torch::Device(torch::kCUDA, 0));
}

// Observe the real aten::copy_ calls and inject a one-shot storage write
// failure. Snapshot reads are untouched; no fault hooks enter the library.
enum class CommitFaultKind { backend, backend_oom, host_oom };

struct CommitFault {
  std::vector<torch::Tensor> targets;
  std::vector<std::size_t> copies;
  std::size_t failures = 0;
  bool fail_rollback = false;
  CommitFaultKind kind = CommitFaultKind::backend;
};

thread_local CommitFault* active_commit_fault = nullptr;

void fail_model_storage_write(c10::StorageImpl* storage) {
  storage->clear_materializer();
  ++active_commit_fault->failures;
  // Simulate a backend that changed part of the destination before failing.
  *static_cast<float*>(storage->mutable_data()) = -901.0F;
  if (active_commit_fault->kind == CommitFaultKind::host_oom)
    throw std::bad_alloc();
  const c10::SourceLocation location{__func__, __FILE__,
                                     static_cast<std::uint32_t>(__LINE__)};
  if (active_commit_fault->kind == CommitFaultKind::backend_oom)
    throw c10::OutOfMemoryError(location, "injected model commit allocation");
  throw c10::Error(location, "injected model commit copy");
}

void observe_model_copy(const at::Tensor& target) {
  auto& fault = *active_commit_fault;
  for (std::size_t index = 0; index < fault.targets.size(); ++index) {
    if (target.unsafeGetTensorImpl() !=
        fault.targets[index].unsafeGetTensorImpl())
      continue;
    fault.copies.push_back(index);
    if (index == 2 &&
        (fault.failures == 0 || (fault.fail_rollback && fault.failures == 1))) {
      target.storage().unsafeGetStorageImpl()->set_materializer(
          fail_model_storage_write);
    }
  }
}

class ObserveCommitFault final {
 public:
  explicit ObserveCommitFault(CommitFault& fault) : fault_(fault) {
    active_commit_fault = &fault;
    handle_ = jyppx::liuding::tests::model_checkpoint::register_copy_observer(
        observe_model_copy);
  }
  ~ObserveCommitFault() {
    jyppx::liuding::tests::model_checkpoint::unregister_copy_observer(handle_);
    for (const auto& tensor : fault_.targets)
      tensor.storage().unsafeGetStorageImpl()->clear_materializer();
    active_commit_fault = nullptr;
  }

 private:
  CommitFault& fault_;
  std::uint64_t handle_;
};

TEST(ModelCheckpointParticipantTest,
     FailedCommitRestoresEveryAttemptedTensorInReverseOrder) {
  for (const auto kind :
       {CommitFaultKind::backend, CommitFaultKind::backend_oom,
        CommitFaultKind::host_oom}) {
    SCOPED_TRACE(static_cast<int>(kind));
    auto model = std::make_shared<ProbeModelImpl>();
    model->eval();
    jyppx::liuding::nn::ModelCheckpointParticipant participant(
        state_for(model));
    const auto saved = save_participant(participant);
    model->linear->weight.detach().fill_(31);
    model->linear->bias.detach().fill_(37);
    model->offset.fill_(41);
    const auto before = raw_model_fields(save_participant(participant));
    CommitFault fault{.targets = {model->linear->weight, model->linear->bias,
                                  model->offset, model->mask},
                      .kind = kind};
    {
      const ObserveCommitFault observer(fault);
      if (kind != CommitFaultKind::host_oom) {
        EXPECT_THROW(participant.load_state(
                         saved, jyppx::liuding::RestorePolicy::exact_only),
                     jyppx::liuding::BackendError);
      } else {
        EXPECT_THROW(participant.load_state(
                         saved, jyppx::liuding::RestorePolicy::exact_only),
                     jyppx::liuding::ResourceExhausted);
      }
    }
    EXPECT_EQ(fault.failures, 1U);
    EXPECT_EQ(fault.copies, (std::vector<std::size_t>{0, 1, 2, 2, 1, 0}));
    EXPECT_EQ(raw_model_fields(save_participant(participant)), before);
    EXPECT_FALSE(model->is_training());
  }
}

TEST(ModelCheckpointParticipantTest,
     IncompleteCommitRollbackStillRestoresTheOtherAttemptedTensors) {
  auto model = std::make_shared<ProbeModelImpl>();
  jyppx::liuding::nn::ModelCheckpointParticipant participant(state_for(model));
  const auto saved = save_participant(participant);
  model->linear->weight.detach().fill_(31);
  model->linear->bias.detach().fill_(37);
  model->offset.fill_(41);
  CommitFault fault{.targets = {model->linear->weight, model->linear->bias,
                                model->offset, model->mask},
                    .fail_rollback = true};
  {
    const ObserveCommitFault observer(fault);
    try {
      participant.load_state(saved, jyppx::liuding::RestorePolicy::exact_only);
      FAIL() << "Incomplete rollback must be reported";
    } catch (const jyppx::liuding::InternalError& error) {
      EXPECT_EQ(error.details().operation, "restore.rollback");
      EXPECT_EQ(error.details().actual, "offset");
      EXPECT_THROW(std::rethrow_if_nested(error), c10::Error);
    }
  }
  EXPECT_EQ(fault.failures, 2U);
  EXPECT_EQ(fault.copies, (std::vector<std::size_t>{0, 1, 2, 2, 1, 0}));
  EXPECT_TRUE(model->linear->weight.eq(31).all().item<bool>());
  EXPECT_TRUE(model->linear->bias.eq(37).all().item<bool>());
  EXPECT_EQ(model->offset[0].item<float>(), -901.0F);
}

#ifdef JYPPX_LIUDING_NN_CUDA_METRICS
TEST(ModelCheckpointParticipantTest, CudaRestoreStagesSnapshotsOnHost) {
  if (!torch::cuda::is_available())
    GTEST_SKIP() << "CUDA device is unavailable";
  auto model = std::make_shared<torch::nn::Module>();
  constexpr std::int64_t elements = 1LL << 24;
  constexpr std::int64_t fields = 4;
  const auto options = torch::TensorOptions().device(torch::kCUDA, 0);
  for (std::int64_t index = 0; index < fields; ++index)
    model->register_buffer("field" + std::to_string(index),
                           torch::full({elements}, index + 1.0, options));
  jyppx::liuding::nn::ModelCheckpointParticipant participant(state_for(model));
  const auto saved = save_participant(participant);
  for (const auto& item : model->named_buffers()) item.value().add_(16);
  torch::cuda::synchronize(0);
  c10::cuda::CUDACachingAllocator::resetPeakStats(0);
  const auto before = c10::cuda::CUDACachingAllocator::getDeviceStats(0)
                          .allocated_bytes[0]
                          .current;
  const auto start = std::chrono::steady_clock::now();
  participant.load_state(saved, jyppx::liuding::RestorePolicy::exact_only);
  torch::cuda::synchronize(0);
  const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::steady_clock::now() - start)
                           .count();
  const auto after = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
  const auto extra = after.allocated_bytes[0].peak - before;
  const auto payload =
      fields * elements * static_cast<std::int64_t>(sizeof(float));
  EXPECT_LE(extra, 16LL << 20);
  EXPECT_EQ(after.allocated_bytes[0].current, before);
  for (std::int64_t index = 0; index < fields; ++index)
    EXPECT_TRUE(model->named_buffers()["field" + std::to_string(index)]
                    .eq(index + 1.0)
                    .all()
                    .item<bool>());
  std::cout << "MODEL_RESTORE_METRICS payload_bytes=" << payload
            << " extra_cuda_allocated_peak_bytes=" << extra
            << " elapsed_us=" << elapsed << '\n';
}
#endif

}  // namespace
