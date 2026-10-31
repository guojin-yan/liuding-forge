// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/random.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/nn/model_checkpoint_participant.hpp>
#include <jyppx/liuding/training/batch_norm_calibration.hpp>

#include <torch/nn/modules/batchnorm.h>
#include <torch/utils.h>

#include <algorithm>
#include <array>
#include <exception>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace jyppx::liuding::training {
namespace {

class ModelSnapshot final : public StateWriter, public StateReader {
 public:
  ModelSnapshot(StateManifest manifest, std::uint64_t maximum)
      : manifest_(std::move(manifest)), maximum_(maximum) {
    validate_state_manifest(manifest_);
    std::uint64_t declared = 0;
    for (const auto& field : manifest_.fields) {
      if (field.max_bytes > maximum_ - declared)
        throw ResourceExhausted("BN calibration model snapshot bound exceeded");
      declared += field.max_bytes;
    }
  }
  void write_blob(std::string_view name,
                  std::span<const std::byte> bytes) override {
    const auto field =
        std::find_if(manifest_.fields.begin(), manifest_.fields.end(),
                     [&](const auto& item) { return item.name == name; });
    if (field == manifest_.fields.end() || blobs_.contains(std::string(name)))
      throw SerializationError("Invalid BN calibration snapshot field");
    if (bytes.size() > field->max_bytes || bytes.size() > maximum_ - used_)
      throw ResourceExhausted("BN calibration model snapshot bound exceeded");
    blobs_.emplace(std::string(name),
                   std::vector<std::byte>(bytes.begin(), bytes.end()));
    used_ += bytes.size();
  }
  const StateManifest& manifest() const noexcept override { return manifest_; }
  bool contains(std::string_view name) const override {
    return blobs_.contains(std::string(name));
  }
  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t maximum) const override {
    const auto& blob = blobs_.at(std::string(name));
    if (blob.size() > maximum)
      throw ResourceExhausted("BN calibration snapshot read exceeds bound");
    return blob;
  }
  std::uint64_t bytes() const noexcept { return used_; }

 private:
  StateManifest manifest_;
  std::uint64_t maximum_, used_ = 0;
  std::map<std::string, std::vector<std::byte>> blobs_;
};

struct Mode {
  std::shared_ptr<torch::nn::Module> module;
  bool training;
};
struct Norm {
  std::shared_ptr<torch::nn::BatchNorm2dImpl> module;
  std::optional<double> momentum;
};

void add_integer(Sha256& hash, std::uint64_t value) {
  std::array<std::byte, 8> bytes{};
  for (unsigned index = 0; index < bytes.size(); ++index)
    bytes[index] = static_cast<std::byte>(value >> (8 * index));
  hash.update(bytes);
}
void add_text(Sha256& hash, std::string_view text) {
  add_integer(hash, text.size());
  hash.update({reinterpret_cast<const std::byte*>(text.data()), text.size()});
}

Sha256Digest protected_state(const nn::ModelAdapter& model,
                             const std::set<const void*>& allowed) {
  Sha256 hash;
  const auto add = [&](const auto& entries, bool parameters) {
    add_text(hash, parameters ? "parameter" : "buffer");
    for (const auto& entry : entries) {
      const auto& value = entry.value();
      if (!parameters && allowed.contains(value.unsafeGetTensorImpl()))
        continue;
      // Input/model validation admits CPU dense tensors. Only a non-contiguous
      // tensor can require one transient tensor-sized contiguous
      // materialization.
      const auto tensor = value.detach().contiguous();
      add_text(hash, entry.key());
      add_text(hash, c10::toString(tensor.scalar_type()));
      add_integer(hash, tensor.dim());
      for (const auto size : tensor.sizes()) add_integer(hash, size);
      hash.update({static_cast<const std::byte*>(tensor.const_data_ptr()),
                   tensor.nbytes()});
    }
  };
  add(model.state().named_parameters(), true);
  add(model.state().named_buffers(), false);
  return hash.finish();
}

void check_cpu_state(const nn::ModelAdapter& model) {
  const auto check = [](const auto& entries, bool parameters) {
    for (const auto& entry : entries) {
      const auto& tensor = entry.value();
      if (!tensor.defined() || !tensor.device().is_cpu() ||
          tensor.layout() != c10::Layout::Strided ||
          !tensor.is_non_overlapping_and_dense() ||
          (tensor.scalar_type() != torch::kFloat32 &&
           (parameters || tensor.scalar_type() != torch::kInt64)))
        throw InvalidArgument(
            "BN calibration requires dense CPU FP32 model "
            "parameters and CPU FP32/int64 buffers");
    }
  };
  check(model.state().named_parameters(), true);
  check(model.state().named_buffers(), false);
}

void restore_controls(const std::vector<Mode>& modes,
                      const std::vector<Norm>& norms,
                      const core::RngState& rng) {
  std::exception_ptr failure;
  const auto attempt = [&](auto&& action) {
    try {
      action();
    } catch (...) {
      if (!failure) failure = std::current_exception();
    }
  };
  // Pre-order traversal restores parents before children, preserving mixed
  // modes despite torch::nn::Module::train() recursively changing descendants.
  for (const auto& mode : modes)
    attempt([&] { mode.module->train(mode.training); });
  for (const auto& norm : norms)
    attempt([&] { norm.module->options.momentum(norm.momentum); });
  attempt([&] { core::set_rng_state(rng, false); });
  if (failure) std::rethrow_exception(failure);
}

void cancelled(std::stop_token stop) {
  if (stop.stop_requested()) throw Cancelled("BN calibration cancelled");
}
}  // namespace

BatchNormCalibrationResult calibrate_batch_norm(
    nn::ModelAdapter& model, const BatchNormCalibrationSource& source,
    const BatchNormCalibrationOptions& options, std::stop_token stop,
    const BatchNormCalibrationCommit& commit) {
  validate_artifact_digest(options.training_source);
  if (!source || options.expected_batches == 0 ||
      options.expected_samples == 0 || options.max_input_bytes == 0 ||
      options.max_model_snapshot_bytes == 0 ||
      options.expected_batches > options.expected_samples ||
      options.expected_batches >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    throw InvalidArgument(
        "BN calibration requires a finite non-empty source "
        "and explicit positive resource bounds");
  cancelled(stop);
  check_cpu_state(model);
  std::vector<Mode> modes;
  std::vector<Norm> norms;
  std::set<const void*> allowed;
  std::set<const void*> allowed_storages;
  std::map<std::string, const void*> allowed_names;
  std::set<const void*> registered_buffers;
  for (const auto& entry : model.state().named_buffers())
    registered_buffers.insert(entry.value().unsafeGetTensorImpl());
  for (const auto& named : model.state().module().named_modules()) {
    const auto& module = named.value();
    modes.push_back({module, module->is_training()});
    if (std::dynamic_pointer_cast<torch::nn::BatchNorm1dImpl>(module) ||
        std::dynamic_pointer_cast<torch::nn::BatchNorm3dImpl>(module))
      throw InvalidArgument("This BN calibration slice supports BN2d only");
    if (auto norm =
            std::dynamic_pointer_cast<torch::nn::BatchNorm2dImpl>(module)) {
      if (!norm->options.track_running_stats() ||
          !norm->running_mean.defined() || !norm->running_var.defined() ||
          !norm->num_batches_tracked.defined() ||
          norm->running_mean.scalar_type() != torch::kFloat32 ||
          norm->running_var.scalar_type() != torch::kFloat32 ||
          norm->num_batches_tracked.scalar_type() != torch::kInt64 ||
          norm->num_batches_tracked.numel() != 1)
        throw InvalidArgument(
            "BN calibration requires tracked FP32 BN2d state");
      for (const auto& value :
           {norm->running_mean, norm->running_var, norm->num_batches_tracked}) {
        if (!value.device().is_cpu() ||
            !registered_buffers.contains(value.unsafeGetTensorImpl()))
          throw InvalidArgument(
              "BN calibration statistics must be registered CPU buffers");
        if (!allowed.insert(value.unsafeGetTensorImpl()).second ||
            !allowed_storages.insert(value.storage().unsafeGetStorageImpl())
                 .second)
          throw InvalidArgument(
              "Aliased BN calibration buffers are unsupported");
      }
      const auto prefix =
          named.key().empty() ? std::string{} : named.key() + ".";
      for (const auto& [suffix, tensor] :
           std::array<std::pair<std::string_view, Tensor>, 3>{
               std::pair{"running_mean", norm->running_mean},
               std::pair{"running_var", norm->running_var},
               std::pair{"num_batches_tracked", norm->num_batches_tracked}}) {
        if (!allowed_names
                 .emplace(prefix + std::string(suffix),
                          tensor.unsafeGetTensorImpl())
                 .second)
          throw InvalidArgument(
              "BN statistics have ambiguous registered owners");
      }
      norms.push_back({norm, norm->options.momentum()});
    }
  }
  if (norms.empty()) throw InvalidArgument("Model has no tracked BN2d modules");
  const auto reject_aliases = [&](const auto& entries, bool parameters) {
    for (const auto& entry : entries) {
      const auto& tensor = entry.value();
      const auto owner = allowed_names.find(entry.key());
      const bool is_owned_statistic =
          !parameters && owner != allowed_names.end() &&
          owner->second == tensor.unsafeGetTensorImpl();
      if (allowed_storages.contains(tensor.storage().unsafeGetStorageImpl()) &&
          !is_owned_statistic)
        throw InvalidArgument("BN buffers alias protected model state");
    }
  };
  reject_aliases(model.state().named_parameters(), true);
  reject_aliases(model.state().named_buffers(), false);
  nn::ModelCheckpointParticipantOptions snapshot_options;
  snapshot_options.max_tensor_bytes = options.max_model_snapshot_bytes;
  snapshot_options.max_total_bytes = options.max_model_snapshot_bytes;
  nn::ModelCheckpointParticipant participant(model, snapshot_options);
  ModelSnapshot snapshot(participant.describe_state(),
                         options.max_model_snapshot_bytes);
  participant.save_state(snapshot);
  const auto protected_before = protected_state(model, allowed);
  const auto rng = core::get_rng_state(false);
  BatchNormCalibrationResult result{.batch_norm_modules = norms.size(),
                                    .model_snapshot_bytes = snapshot.bytes()};
  torch::NoGradGuard no_grad;
  try {
    model.state().eval();
    for (const auto& norm : norms) {
      norm.module->reset_running_stats();
      norm.module->options.momentum(std::nullopt);
      norm.module->train();
    }
    for (; result.batches < options.expected_batches; ++result.batches) {
      cancelled(stop);
      auto batch = source(stop);
      cancelled(stop);
      if (!batch) throw InvalidArgument("BN calibration source ended early");
      const auto& input = batch->images;
      if (!(batch->training_source == options.training_source))
        throw CompatibilityError("BN calibration received another data source");
      if (!input.defined() || !input.device().is_cpu() ||
          input.scalar_type() != torch::kFloat32 ||
          input.layout() != c10::Layout::Strided || input.dim() != 4 ||
          input.size(0) <= 0 || input.size(1) <= 0 || input.size(2) <= 0 ||
          input.size(3) <= 0)
        throw InvalidArgument("BN calibration input must be CPU FP32 NCHW");
      if (input.nbytes() > options.max_input_bytes)
        throw ResourceExhausted("BN calibration input exceeds its byte bound");
      const auto count = static_cast<std::uint64_t>(input.size(0));
      if (count > options.expected_samples - result.samples)
        throw InvalidArgument("BN calibration source exceeds sample count");
      if (!torch::isfinite(input).all().item<bool>())
        throw NumericalError("BN calibration input is not finite");
      static_cast<void>(model.invoke({.positional = {input}, .keyword = {}}));
      result.samples += count;
      cancelled(stop);
    }
    auto extra = source(stop);
    cancelled(stop);
    if (extra || result.samples != options.expected_samples)
      throw InvalidArgument(
          "BN calibration source has a different finite pass");
    for (const auto& norm : norms) {
      if (!torch::isfinite(norm.module->running_mean).all().item<bool>() ||
          !torch::isfinite(norm.module->running_var).all().item<bool>() ||
          norm.module->running_var.lt(0).any().item<bool>() ||
          norm.module->num_batches_tracked.item<std::int64_t>() !=
              static_cast<std::int64_t>(options.expected_batches))
        throw NumericalError("BN calibration produced invalid tracked state");
    }
    if (protected_state(model, allowed) != protected_before)
      throw CompatibilityError("BN calibration changed protected model state");
    restore_controls(modes, norms, rng);
    cancelled(stop);
    if (commit) commit(result);
    // All potentially throwing control restoration precedes durable commit.
    return result;
  } catch (...) {
    const auto original = std::current_exception();
    std::exception_ptr rollback_failure;
    try {
      participant.load_state(snapshot, RestorePolicy::exact_only);
    } catch (...) {
      rollback_failure = std::current_exception();
    }
    try {
      restore_controls(modes, norms, rng);
    } catch (...) {
      if (!rollback_failure) rollback_failure = std::current_exception();
    }
    if (rollback_failure) {
      try {
        std::rethrow_exception(rollback_failure);
      } catch (...) {
        std::throw_with_nested(InternalError(
            "BN calibration failed and rollback was incomplete",
            ErrorDetails{.component = "training.batch_norm_calibration",
                         .operation = "rollback",
                         .hint = "Discard this process and resume the durable "
                                 "READY checkpoint before using the model."}));
      }
    }
    std::rethrow_exception(original);
  }
}
}  // namespace jyppx::liuding::training
