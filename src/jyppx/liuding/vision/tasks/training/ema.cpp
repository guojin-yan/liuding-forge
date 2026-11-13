// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "checkpoint_codec.hpp"

#include <jyppx/liuding/vision/tasks/training/ema.hpp>

#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>
#include <utility>

namespace jyppx::liuding::vision::tasks {
namespace {
constexpr std::uint64_t kRecipeLimit = 64U << 10U;
constexpr std::uint64_t kControlMagic = 0x31414d45464cULL;

bool Integral(c10::ScalarType dtype) {
  return dtype == torch::kUInt8 || dtype == torch::kInt8 ||
         dtype == torch::kInt16 || dtype == torch::kInt32 ||
         dtype == torch::kInt64 || dtype == torch::kBool;
}

struct Entry {
  std::string key;
  Tensor source, shadow;
  std::vector<std::int64_t> sizes, strides;
  c10::ScalarType dtype;
  Device device;
  const c10::StorageImpl* source_storage;
  const c10::StorageImpl* shadow_storage;
};

class ModelReader final : public StateReader {
 public:
  ModelReader(const StateReader& source, StateManifest manifest)
      : source_(source), manifest_(std::move(manifest)) {}
  const StateManifest& manifest() const noexcept override { return manifest_; }
  bool contains(std::string_view name) const override {
    return source_.contains(name);
  }
  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t max_bytes) const override {
    return source_.read_blob(name, max_bytes);
  }

 private:
  const StateReader& source_;
  StateManifest manifest_;
};
}  // namespace

struct ModelEma::Impl {
  std::shared_ptr<nn::ModelAdapter> source, shadow;
  ModelEmaOptions options;
  std::unique_ptr<nn::ModelCheckpointParticipant> checkpoint;
  std::vector<Entry> entries;
  std::string recipe;
  std::uint64_t updates = 0;

  void ValidateBindings() const {
    std::size_t index = 0;
    const auto validate = [&](const auto& source_values,
                              const auto& shadow_values,
                              const std::string& prefix) {
      if (source_values.size() != shadow_values.size())
        throw CompatibilityError("EMA named state topology changed");
      for (const auto& item : source_values) {
        if (index >= entries.size() || !shadow_values.contains(item.key()))
          throw CompatibilityError("EMA named state topology changed");
        const auto& entry = entries[index++];
        const auto& live = item.value();
        const auto& target = shadow_values[item.key()];
        if (entry.key != prefix + item.key() ||
            live.unsafeGetTensorImpl() != entry.source.unsafeGetTensorImpl() ||
            target.unsafeGetTensorImpl() != entry.shadow.unsafeGetTensorImpl())
          throw CompatibilityError("EMA model tensor identity changed");
        for (const auto& value : {live, target})
          if (!value.defined() || value.layout() != torch::kStrided ||
              value.scalar_type() != entry.dtype ||
              value.device() != entry.device ||
              value.sizes().vec() != entry.sizes ||
              value.strides().vec() != entry.strides ||
              value.storage_offset() != 0)
            throw CompatibilityError(
                "EMA tensor placement or topology changed");
        if (live.storage().unsafeGetStorageImpl() != entry.source_storage ||
            target.storage().unsafeGetStorageImpl() != entry.shadow_storage ||
            target.requires_grad())
          throw CompatibilityError("EMA tensor storage/frozen state changed");
      }
    };
    validate(source->state().named_parameters(),
             shadow->state().named_parameters(), "parameter.");
    validate(source->state().named_buffers(), shadow->state().named_buffers(),
             "buffer.");
    if (index != entries.size())
      throw CompatibilityError("EMA named state count changed");
    if (shadow->state().is_training())
      throw NotReady("EMA shadow must remain in evaluation mode");
    for (const auto& module : shadow->state().module().modules(false))
      if (module->is_training())
        throw NotReady("EMA submodules must remain in evaluation mode");
  }
};

ModelEma::ModelEma(std::shared_ptr<nn::ModelAdapter> source,
                   std::shared_ptr<nn::ModelAdapter> shadow,
                   ModelEmaOptions options)
    : impl_(std::make_unique<Impl>()) {
  if (!source || !shadow ||
      source->state().shared_module() == shadow->state().shared_module() ||
      !state::Identifier(options.source_model_id) ||
      !std::isfinite(options.decay) || options.decay < 0 ||
      options.decay >= 1 || !std::isfinite(options.ramp_updates) ||
      options.ramp_updates <= 0)
    throw InvalidArgument("Invalid EMA models, identity or averaging recipe");
  impl_->source = std::move(source);
  impl_->shadow = std::move(shadow);
  impl_->options = std::move(options);
  const auto& limits = impl_->options.checkpoint;
  std::set<const c10::StorageImpl*> storages;
  std::uint64_t total = 0;
  const auto bind = [&](const auto& live_values, const auto& shadow_values,
                        const std::string& prefix) {
    if (live_values.size() != shadow_values.size())
      throw InvalidArgument("EMA source and shadow named state must match");
    for (const auto& item : live_values) {
      if (!shadow_values.contains(item.key()))
        throw InvalidArgument("EMA shadow is missing a named source tensor");
      const auto live = item.value();
      const auto target = shadow_values[item.key()];
      if (impl_->entries.size() >= limits.max_tensors)
        throw ResourceExhausted("EMA tensor count exceeds configured limit");
      for (const auto& value : {live, target}) {
        if (!value.defined() || value.layout() != torch::kStrided ||
            !value.is_contiguous() || value.storage_offset() != 0 ||
            !(value.device().is_cpu() || value.device().is_cuda()) ||
            !(value.scalar_type() == torch::kFloat32 ||
              value.scalar_type() == torch::kFloat64 ||
              Integral(value.scalar_type())) ||
            value.numel() == 0)
          throw InvalidArgument(
              "EMA requires nonempty contiguous FP32/FP64 or integral state");
        if (!storages.insert(value.storage().unsafeGetStorageImpl()).second)
          throw InvalidArgument(
              "EMA source/shadow storage aliases are unsupported");
      }
      if (live.sizes() != target.sizes() ||
          live.strides() != target.strides() ||
          live.scalar_type() != target.scalar_type() ||
          live.device() != target.device())
        throw InvalidArgument(
            "EMA source and shadow tensor metadata must match");
      const auto bytes =
          static_cast<std::uint64_t>(live.numel()) * live.element_size();
      if (bytes > limits.max_tensor_bytes || total > limits.max_total_bytes ||
          bytes > limits.max_total_bytes - total)
        throw ResourceExhausted("EMA state bytes exceed configured limit");
      total += bytes;
      impl_->entries.push_back({prefix + item.key(), live, target,
                                live.sizes().vec(), live.strides().vec(),
                                live.scalar_type(), live.device(),
                                live.storage().unsafeGetStorageImpl(),
                                target.storage().unsafeGetStorageImpl()});
    }
  };
  bind(impl_->source->state().named_parameters(),
       impl_->shadow->state().named_parameters(), "parameter.");
  bind(impl_->source->state().named_buffers(),
       impl_->shadow->state().named_buffers(), "buffer.");
  if (impl_->entries.empty()) throw InvalidArgument("EMA model has no state");
  // Reject configuration/topology before copying any tensor values.
  impl_->checkpoint =
      std::make_unique<nn::ModelCheckpointParticipant>(*impl_->shadow, limits);
  impl_->recipe =
      nlohmann::json{{"schema", "vision.model-ema/v1"},
                     {"source_model_id", impl_->options.source_model_id},
                     {"decay", impl_->options.decay},
                     {"ramp_updates", impl_->options.ramp_updates},
                     {"floating", "ramped-fp32-fp64"},
                     {"integral", "copy"}}
          .dump();
  const torch::NoGradGuard no_grad;
  for (auto& entry : impl_->entries) {
    entry.shadow.set_requires_grad(false);
    entry.shadow.copy_(entry.source);
  }
  impl_->shadow->state().eval();
  impl_->ValidateBindings();
}

ModelEma::~ModelEma() = default;
nn::ModelAdapter& ModelEma::model() const noexcept { return *impl_->shadow; }
const ModelEmaOptions& ModelEma::options() const noexcept {
  return impl_->options;
}
std::uint64_t ModelEma::completed_updates() const noexcept {
  return impl_->updates;
}
double ModelEma::current_decay() const noexcept {
  return impl_->options.decay *
         -std::expm1(-static_cast<double>(impl_->updates) /
                     impl_->options.ramp_updates);
}
void ModelEma::on_optimizer_update(std::uint64_t global_step) {
  if (impl_->updates == (std::numeric_limits<std::uint64_t>::max)() ||
      global_step != impl_->updates + 1)
    throw InvalidArgument("EMA requires one consecutive successful update");
  impl_->ValidateBindings();
  const auto decay =
      impl_->options.decay * -std::expm1(-static_cast<double>(global_step) /
                                         impl_->options.ramp_updates);
  const torch::NoGradGuard no_grad;
  for (auto& entry : impl_->entries) {
    if (entry.source.is_floating_point())
      entry.shadow.lerp_(entry.source.detach(), 1.0 - decay);
    else
      entry.shadow.copy_(entry.source);
  }
  impl_->updates = global_step;
}
std::string ModelEma::participant_id() const {
  return impl_->options.checkpoint.participant_id;
}
StateManifest ModelEma::describe_state() const {
  impl_->ValidateBindings();
  auto manifest = impl_->checkpoint->describe_state();
  manifest.fields.push_back({"ema_recipe",
                             "application/vnd.jyppx.model-ema-recipe-v1",
                             kRecipeLimit});
  manifest.fields.push_back(
      {"ema_control", "application/vnd.jyppx.model-ema-control-v1", 16});
  validate_state_manifest(manifest);
  return manifest;
}
void ModelEma::save_state(StateWriter& writer) const {
  impl_->ValidateBindings();
  std::vector<std::byte> control;
  state::Append(control, kControlMagic);
  state::Append(control, impl_->updates);
  writer.write_blob("ema_recipe", std::as_bytes(std::span(impl_->recipe)));
  writer.write_blob("ema_control", control);
  impl_->checkpoint->save_state(writer);
}
void ModelEma::load_state(const StateReader& reader, RestorePolicy policy) {
  state::VerifyManifest(reader.manifest(), describe_state(), policy);
  state::VerifyText(reader, "ema_recipe", impl_->recipe, kRecipeLimit);
  const auto control = reader.read_blob("ema_control", 16);
  std::size_t offset = 0;
  if (control.size() != 16 || state::Read(control, offset) != kControlMagic)
    throw SerializationError("Invalid EMA control state");
  const auto updates = state::Read(control, offset);
  // The delegated load stages all tensors before its transactional commit.
  const ModelReader tensors(reader, impl_->checkpoint->describe_state());
  impl_->checkpoint->load_state(tensors, policy);
  impl_->updates = updates;
}

}  // namespace jyppx::liuding::vision::tasks
