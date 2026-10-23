// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/nn/model_checkpoint_participant.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <map>
#include <memory>
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
    const auto field =
        std::find_if(manifest_.fields.begin(), manifest_.fields.end(),
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

  EXPECT_THROW(target.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
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
  EXPECT_THROW(participant.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::SerializationError);
  EXPECT_TRUE(torch::equal(model->linear->weight, before));

  store = save_participant(participant);
  store.mutable_blob("tensor.0000000000").pop_back();
  EXPECT_THROW(participant.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
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
  EXPECT_THROW(participant.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::SerializationError);
  EXPECT_TRUE(torch::equal(model->mask, before));

  store = save_participant(participant);
  store.mutable_manifest().schema_version = 2;
  EXPECT_THROW(participant.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
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
  EXPECT_THROW(jyppx::liuding::nn::ModelCheckpointParticipant(state_for(model),
                                                     {.max_tensors = 2}),
               jyppx::liuding::ResourceExhausted);
  EXPECT_THROW(jyppx::liuding::nn::ModelCheckpointParticipant(state_for(model),
                                                     {.max_tensors = 0}),
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

  EXPECT_THROW(
      participant.load_state(store, static_cast<jyppx::liuding::RestorePolicy>(255)),
      jyppx::liuding::CompatibilityError);
}

}  // namespace
