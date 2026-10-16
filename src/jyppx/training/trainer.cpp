#include <jyppx/core/device.hpp>
#include <jyppx/foundation/random.hpp>
#include <jyppx/training/amp/autocast.hpp>
#include <jyppx/training/trainer.hpp>

#include <torch/serialize.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace jyppx {
namespace training {

namespace {

constexpr int64_t kCheckpointFormatVersion = 3;

void atomic_write_file(const std::string& path, std::string_view bytes) {
  static std::atomic<uint64_t> sequence{0};
  const std::filesystem::path destination(path);
  auto temporary = destination;
  temporary +=
      ".tmp-" +
      std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count()) +
      "-" + std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));

  try {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("Cannot create checkpoint: " + path);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    output.flush();
    if (!output)
      throw std::runtime_error("Failed to write checkpoint: " + path);
    output.close();

#ifdef _WIN32
    if (!MoveFileExW(temporary.c_str(), destination.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
      throw std::system_error(static_cast<int>(GetLastError()),
                              std::system_category(),
                              "Failed to atomically replace checkpoint");
    }
#else
    std::filesystem::rename(temporary, destination);
#endif
  } catch (...) {
    std::error_code error;
    std::filesystem::remove(temporary, error);
    throw;
  }
}

void write_scalar_state(torch::serialize::OutputArchive& archive,
                        const std::map<std::string, double>& state) {
  for (const auto& [name, value] : state) {
    archive.write(
        name,
        torch::tensor(value, torch::TensorOptions().dtype(torch::kFloat64)));
  }
}

std::map<std::string, double> read_scalar_state(
    torch::serialize::InputArchive& archive) {
  std::map<std::string, double> state;
  for (const auto& key : archive.keys()) {
    torch::Tensor value;
    if (archive.try_read(key, value) && value.defined() && value.numel() == 1) {
      state.emplace(key, value.item<double>());
    }
  }
  return state;
}

void write_grad_scaler_state(torch::serialize::OutputArchive& archive,
                             const amp::GradScalerState& state) {
  archive.write("format_version",
                torch::IValue(static_cast<int64_t>(state.format_version)));
  archive.write("scaler_id", torch::IValue(state.scaler_id));
  archive.write("init_scale", torch::IValue(state.init_scale));
  archive.write("growth_factor", torch::IValue(state.growth_factor));
  archive.write("backoff_factor", torch::IValue(state.backoff_factor));
  archive.write("growth_interval",
                torch::IValue(static_cast<int64_t>(state.growth_interval)));
  archive.write("enabled", torch::IValue(state.enabled));
  archive.write("scale", torch::IValue(state.scale));
  archive.write("growth_tracker",
                torch::IValue(static_cast<int64_t>(state.growth_tracker)));
}

amp::GradScalerState read_grad_scaler_state(
    torch::serialize::InputArchive& archive) {
  torch::IValue value;
  const auto require_int = [&](const char* key) {
    if (!archive.try_read(key, value) || !value.isInt()) {
      throw std::runtime_error(std::string("Invalid GradScaler field: ") + key);
    }
    return value.toInt();
  };
  const auto require_double = [&](const char* key) {
    if (!archive.try_read(key, value) || !value.isDouble()) {
      throw std::runtime_error(std::string("Invalid GradScaler field: ") + key);
    }
    return value.toDouble();
  };
  const auto require_bool = [&](const char* key) {
    if (!archive.try_read(key, value) || !value.isBool()) {
      throw std::runtime_error(std::string("Invalid GradScaler field: ") + key);
    }
    return value.toBool();
  };
  const auto require_string = [&](const char* key) {
    if (!archive.try_read(key, value) || !value.isString()) {
      throw std::runtime_error(std::string("Invalid GradScaler field: ") + key);
    }
    return value.toStringRef();
  };

  const auto format_version = require_int("format_version");
  const auto growth_interval = require_int("growth_interval");
  const auto growth_tracker = require_int("growth_tracker");
  if (format_version < 0 ||
      format_version > std::numeric_limits<std::uint32_t>::max() ||
      growth_interval < 0 ||
      growth_interval > std::numeric_limits<int>::max() || growth_tracker < 0) {
    throw std::runtime_error("GradScaler checkpoint integer is out of range");
  }
  return {.format_version = static_cast<std::uint32_t>(format_version),
          .scaler_id = require_string("scaler_id"),
          .init_scale = require_double("init_scale"),
          .growth_factor = require_double("growth_factor"),
          .backoff_factor = require_double("backoff_factor"),
          .growth_interval = static_cast<int>(growth_interval),
          .enabled = require_bool("enabled"),
          .scale = require_double("scale"),
          .growth_tracker = static_cast<std::uint64_t>(growth_tracker)};
}

void write_callback_state(torch::serialize::OutputArchive& archive,
                          const CallbackState& state) {
  torch::serialize::OutputArchive scalar_archive(archive.compilation_unit());
  write_scalar_state(scalar_archive, state.scalars);
  archive.write("scalars", scalar_archive);

  torch::serialize::OutputArchive string_archive(archive.compilation_unit());
  for (const auto& [name, value] : state.strings) {
    string_archive.write(name, torch::IValue(value));
  }
  archive.write("strings", string_archive);
}

CallbackState read_callback_state(torch::serialize::InputArchive& archive) {
  CallbackState state;
  torch::serialize::InputArchive scalar_archive;
  if (archive.try_read("scalars", scalar_archive)) {
    state.scalars = read_scalar_state(scalar_archive);
  }
  torch::serialize::InputArchive string_archive;
  if (archive.try_read("strings", string_archive)) {
    for (const auto& key : string_archive.keys()) {
      torch::IValue value;
      if (string_archive.try_read(key, value) && value.isString()) {
        state.strings.emplace(key, value.toStringRef());
      }
    }
  }
  return state;
}

void write_data_loader_state(torch::serialize::OutputArchive& archive,
                             const data::DataLoaderState& state) {
  if (state.dataset_size >
          static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
      state.batch_size >
          static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    throw std::overflow_error(
        "DataLoader dimensions cannot be represented in a checkpoint");
  }
  archive.write("version", torch::IValue(static_cast<int64_t>(state.version)));
  archive.write("dataset_size",
                torch::IValue(static_cast<int64_t>(state.dataset_size)));
  archive.write("batch_size",
                torch::IValue(static_cast<int64_t>(state.batch_size)));
  archive.write("drop_last", torch::IValue(state.drop_last));
  archive.write("epoch_prepared", torch::IValue(state.epoch_prepared));
  archive.write("sampler_state_supported",
                torch::IValue(state.sampler_state_supported));
  archive.write("sampler_state", torch::IValue(state.sampler_state));

  std::vector<int64_t> indices;
  indices.reserve(state.prepared_indices.size());
  for (const auto index : state.prepared_indices) {
    if (index > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
      throw std::overflow_error("DataLoader index cannot be checkpointed");
    }
    indices.push_back(static_cast<int64_t>(index));
  }
  archive.write(
      "prepared_indices",
      torch::tensor(indices, torch::TensorOptions().dtype(torch::kInt64)));
}

data::DataLoaderState read_data_loader_state(
    torch::serialize::InputArchive& archive) {
  data::DataLoaderState state;
  torch::IValue value;
  auto read_nonnegative = [&](const char* key) -> uint64_t {
    if (!archive.try_read(key, value) || !value.isInt() || value.toInt() < 0) {
      throw std::runtime_error(
          std::string("Invalid DataLoader checkpoint field: ") + key);
    }
    return static_cast<uint64_t>(value.toInt());
  };

  const auto version = read_nonnegative("version");
  const auto dataset_size = read_nonnegative("dataset_size");
  const auto batch_size = read_nonnegative("batch_size");
  if (version > std::numeric_limits<std::uint32_t>::max() ||
      dataset_size > std::numeric_limits<size_t>::max() ||
      batch_size > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error("DataLoader checkpoint field is out of range");
  }
  state.version = static_cast<std::uint32_t>(version);
  state.dataset_size = static_cast<size_t>(dataset_size);
  state.batch_size = static_cast<size_t>(batch_size);

  if (!archive.try_read("drop_last", value) || !value.isBool()) {
    throw std::runtime_error("Invalid DataLoader checkpoint field: drop_last");
  }
  state.drop_last = value.toBool();
  if (!archive.try_read("epoch_prepared", value) || !value.isBool()) {
    throw std::runtime_error(
        "Invalid DataLoader checkpoint field: epoch_prepared");
  }
  state.epoch_prepared = value.toBool();
  if (state.version >= 2) {
    if (!archive.try_read("sampler_state_supported", value) ||
        !value.isBool()) {
      throw std::runtime_error(
          "Invalid DataLoader checkpoint field: sampler_state_supported");
    }
    state.sampler_state_supported = value.toBool();
  } else {
    state.sampler_state_supported = true;
  }
  if (!archive.try_read("sampler_state", value) || !value.isString()) {
    throw std::runtime_error(
        "Invalid DataLoader checkpoint field: sampler_state");
  }
  state.sampler_state = value.toStringRef();

  Tensor indices;
  if (!archive.try_read("prepared_indices", indices) ||
      indices.scalar_type() != torch::kInt64 || indices.dim() != 1) {
    throw std::runtime_error("Invalid DataLoader checkpoint indices");
  }
  indices = indices.cpu().contiguous();
  const auto* values = indices.const_data_ptr<int64_t>();
  state.prepared_indices.reserve(static_cast<size_t>(indices.numel()));
  for (int64_t i = 0; i < indices.numel(); ++i) {
    if (values[i] < 0) {
      throw std::runtime_error(
          "DataLoader checkpoint contains a negative index");
    }
    state.prepared_indices.push_back(static_cast<size_t>(values[i]));
  }
  return state;
}

std::vector<std::shared_ptr<Callback>> checkpointable_callbacks(
    const std::vector<std::shared_ptr<Callback>>& callbacks) {
  std::vector<std::shared_ptr<Callback>> result;
  for (const auto& callback : callbacks) {
    if (callback && !callback->state_key().empty()) {
      result.push_back(callback);
    }
  }
  return result;
}

}  // namespace

Trainer::Trainer(std::shared_ptr<torch::nn::Module> model,
                 std::shared_ptr<optim::Optimizer> optimizer,
                 const TrainerOptions& options)
    : model_(std::move(model)),
      optimizer_(std::move(optimizer)),
      options_(options),
      forward_fn_(options_.forward_fn),
      loss_fn_(options_.loss_fn),
      global_step_(0),
      current_epoch_(0),
      next_epoch_(0),
      should_stop_(false),
      last_train_loss_(0.0),
      last_val_loss_(0.0),
      has_validation_loss_(false) {
  if (!model_) {
    throw std::invalid_argument("Trainer model cannot be null");
  }
  if (!optimizer_) {
    throw std::invalid_argument("Trainer optimizer cannot be null");
  }
  if (options_.use_amp && options_.amp_dtype == at::kHalf) {
    grad_scaler_ = std::make_unique<amp::GradScaler>(options_.amp_options);
  }
  validate_configuration();
}

void Trainer::validate_configuration() const {
  if (!model_) {
    throw std::invalid_argument("Trainer model cannot be null");
  }
  if (!optimizer_) {
    throw std::invalid_argument("Trainer optimizer cannot be null");
  }
  if (options_.max_epochs < 0) {
    throw std::invalid_argument("Trainer max_epochs must be non-negative");
  }
  if (options_.max_steps == 0 || options_.max_steps < -1) {
    throw std::invalid_argument("Trainer max_steps must be -1 or positive");
  }
  if (options_.accumulation_steps <= 0) {
    throw std::invalid_argument("Trainer accumulation_steps must be positive");
  }
  if (options_.gradient_clip_val < 0.0) {
    throw std::invalid_argument(
        "Trainer gradient_clip_val must be non-negative");
  }
  if (options_.gradient_clip_mode != "norm" &&
      options_.gradient_clip_mode != "value") {
    throw std::invalid_argument(
        "Trainer gradient_clip_mode must be 'norm' or 'value'");
  }
  if (options_.max_checkpoint_bytes == 0) {
    throw std::invalid_argument(
        "Trainer max_checkpoint_bytes must be positive");
  }
  if (!loss_fn_) {
    throw std::invalid_argument("Trainer loss function cannot be empty");
  }
  if (options_.use_amp) {
    if (options_.amp_dtype != at::kHalf &&
        options_.amp_dtype != at::kBFloat16) {
      throw std::invalid_argument(
          "Trainer AMP dtype must be torch::kFloat16 or torch::kBFloat16");
    }
    if (options_.amp_dtype == at::kHalf &&
        options_.device.type() != torch::kCUDA) {
      throw std::runtime_error("float16 AMP requires a CUDA device");
    }
    if (options_.device.type() == torch::kCUDA && !cuda_is_available()) {
      throw std::runtime_error(
          "Trainer AMP requires CUDA, but no CUDA device is available");
    }
  }
}

void Trainer::require_forward_fn() const {
  if (!forward_fn_) {
    throw std::invalid_argument(
        "Trainer requires a forward function; construct it with a concrete "
        "module, "
        "set TrainerOptions::forward_fn, or call Trainer::set_forward_fn");
  }
}

Tensor Trainer::compute_loss(const Tensor& input, const Tensor& target) const {
  if (options_.use_amp) {
    // The loss itself comes back in float32 where supported, so logging
    // and backward retain eager-mode numerical behavior.
    amp::AutocastGuard autocast{options_.device.type(), options_.amp_dtype};
    return loss_fn_(forward_fn_(input), target);
  }
  return loss_fn_(forward_fn_(input), target);
}

void Trainer::set_forward_fn(TrainerOptions::ForwardFn forward_fn) {
  if (!forward_fn) {
    throw std::invalid_argument("Trainer forward function cannot be empty");
  }
  forward_fn_ = std::move(forward_fn);
  options_.forward_fn = forward_fn_;
}

void Trainer::set_loss_fn(TrainerOptions::LossFn loss_fn) {
  if (!loss_fn) {
    throw std::invalid_argument("Trainer loss function cannot be empty");
  }
  loss_fn_ = std::move(loss_fn);
  options_.loss_fn = loss_fn_;
}

void Trainer::set_scheduler(
    std::shared_ptr<optim::schedulers::LRScheduler> scheduler) {
  if (scheduler &&
      scheduler->get_last_lrs().size() != optimizer_->param_groups().size()) {
    throw std::invalid_argument(
        "Trainer scheduler parameter groups must match the optimizer");
  }
  scheduler_ = std::move(scheduler);
}

void Trainer::add_callback(std::shared_ptr<Callback> callback) {
  if (!callback) {
    throw std::invalid_argument("Trainer callback cannot be null");
  }
  options_.callbacks.push_back(std::move(callback));
}

void Trainer::clip_gradients() {
  const auto parameters = model_->parameters();
  if (options_.gradient_clip_mode == "value") {
    torch::nn::utils::clip_grad_value_(parameters, options_.gradient_clip_val);
  } else {
    torch::nn::utils::clip_grad_norm_(parameters, options_.gradient_clip_val);
  }
}

void Trainer::scale_gradients(double scale) {
  if (scale == 1.0) {
    return;
  }
  torch::NoGradGuard no_grad;
  for (const auto& parameter : model_->parameters()) {
    if (!parameter.defined() || !parameter.requires_grad()) {
      continue;
    }
    auto gradient = parameter.grad();
    if (gradient.defined()) {
      gradient.mul_(scale);
    }
  }
}

void Trainer::defer_epoch_end_action(std::function<void()> action) {
  if (!action) {
    throw std::invalid_argument("Deferred epoch-end action cannot be empty");
  }
  if (dispatching_epoch_end_callbacks_) {
    deferred_epoch_end_actions_.push_back(std::move(action));
    return;
  }
  action();
}

void Trainer::save_checkpoint(const std::string& path) const {
  validate_configuration();
  if (path.empty()) {
    throw std::invalid_argument("Checkpoint path cannot be empty");
  }
  if (!checkpoint_at_epoch_boundary_) {
    throw std::logic_error(
        "Checkpoint save requires an epoch boundary; in-epoch exact state is "
        "not available");
  }

  torch::serialize::OutputArchive archive;
  archive.write("checkpoint/format_version",
                torch::IValue(kCheckpointFormatVersion));
  archive.write("checkpoint/consistency_boundary", torch::IValue("epoch"));
  archive.write("checkpoint/has_scheduler",
                torch::IValue(scheduler_ != nullptr));
  archive.write("checkpoint/has_grad_scaler",
                torch::IValue(grad_scaler_ != nullptr));
  archive.write(
      "config/accumulation_steps",
      torch::IValue(static_cast<int64_t>(options_.accumulation_steps)));
  archive.write("config/use_amp", torch::IValue(options_.use_amp));
  archive.write("config/amp_dtype",
                torch::IValue(static_cast<int64_t>(options_.amp_dtype)));
  archive.write("config/device_type",
                torch::IValue(static_cast<int64_t>(options_.device.type())));
  archive.write("config/device_index",
                torch::IValue(static_cast<int64_t>(options_.device.index())));

  torch::serialize::OutputArchive model_archive(archive.compilation_unit());
  model_->save(model_archive);
  archive.write("model", model_archive);

  torch::serialize::OutputArchive optimizer_archive(archive.compilation_unit());
  for (const auto& [name, value] : optimizer_->state_dict()) {
    if (value.defined()) {
      optimizer_archive.write(name, value.detach().cpu());
    }
  }
  archive.write("optimizer", optimizer_archive);

  if (scheduler_) {
    torch::serialize::OutputArchive scheduler_archive(
        archive.compilation_unit());
    write_scalar_state(scheduler_archive, scheduler_->state_dict());
    archive.write("scheduler", scheduler_archive);
  }

  if (grad_scaler_) {
    torch::serialize::OutputArchive scaler_archive(archive.compilation_unit());
    write_grad_scaler_state(scaler_archive, grad_scaler_->capture_state());
    archive.write("grad_scaler", scaler_archive);
  }

  archive.write("trainer/global_step", torch::IValue(global_step_));
  archive.write("trainer/current_epoch",
                torch::IValue(static_cast<int64_t>(current_epoch_)));
  archive.write("trainer/next_epoch",
                torch::IValue(static_cast<int64_t>(next_epoch_)));
  archive.write("trainer/last_train_loss", torch::IValue(last_train_loss_));
  archive.write("trainer/last_val_loss", torch::IValue(last_val_loss_));
  archive.write("trainer/has_validation_loss",
                torch::IValue(has_validation_loss_));
  archive.write("trainer/should_stop", torch::IValue(should_stop_));

  const auto rng_state = core::get_rng_state(options_.device.is_cuda());
  archive.write("rng/cpu", rng_state.cpu);
  const int64_t cuda_rng_count = static_cast<int64_t>(rng_state.cuda.size());
  archive.write("rng/cuda_count", torch::IValue(cuda_rng_count));
  for (int64_t i = 0; i < cuda_rng_count; ++i) {
    archive.write("rng/cuda/" + std::to_string(i),
                  rng_state.cuda[static_cast<size_t>(i)]);
  }

  const auto callbacks = checkpointable_callbacks(options_.callbacks);
  torch::serialize::OutputArchive callbacks_archive(archive.compilation_unit());
  callbacks_archive.write(
      "count", torch::IValue(static_cast<int64_t>(callbacks.size())));
  for (size_t i = 0; i < callbacks.size(); ++i) {
    torch::serialize::OutputArchive callback_archive(
        archive.compilation_unit());
    callback_archive.write("state_key",
                           torch::IValue(callbacks[i]->state_key()));
    write_callback_state(callback_archive, callbacks[i]->state_dict());
    callbacks_archive.write(std::to_string(i), callback_archive);
  }
  archive.write("callbacks", callbacks_archive);

  if (train_loader_state_) {
    if (!train_loader_state_->sampler_state_supported) {
      throw std::logic_error(
          "Training DataLoader sampler does not support checkpoint state");
    }
    torch::serialize::OutputArchive loader_archive(archive.compilation_unit());
    write_data_loader_state(loader_archive, *train_loader_state_);
    archive.write("data/train_loader", loader_archive);
  }
  if (validation_loader_state_) {
    if (!validation_loader_state_->sampler_state_supported) {
      throw std::logic_error(
          "Validation DataLoader sampler does not support checkpoint state");
    }
    torch::serialize::OutputArchive loader_archive(archive.compilation_unit());
    write_data_loader_state(loader_archive, *validation_loader_state_);
    archive.write("data/validation_loader", loader_archive);
  }

  std::ostringstream serialized(std::ios::binary | std::ios::out);
  archive.save_to(serialized);
  const std::string raw = serialized.str();
  if (raw.size() > options_.max_checkpoint_bytes) {
    throw std::runtime_error("Checkpoint exceeds max_checkpoint_bytes");
  }
  atomic_write_file(path, raw);
}

void Trainer::load_checkpoint(const std::string& path) {
  validate_configuration();
  if (path.empty()) {
    throw std::invalid_argument("Checkpoint path cannot be empty");
  }

  std::error_code file_error;
  const auto file_size = std::filesystem::file_size(path, file_error);
  if (file_error) {
    throw std::runtime_error("Cannot inspect checkpoint: " + path);
  }
  if (file_size > options_.max_checkpoint_bytes) {
    throw std::runtime_error("Checkpoint exceeds max_checkpoint_bytes");
  }

  torch::serialize::InputArchive archive;
  archive.load_from(path, options_.device);

  torch::IValue value;
  if (!archive.try_read("checkpoint/format_version", value) || !value.isInt() ||
      value.toInt() != kCheckpointFormatVersion) {
    throw std::runtime_error(
        "Unsupported or missing checkpoint format version");
  }

  std::optional<data::DataLoaderState> restored_train_loader;
  std::optional<data::DataLoaderState> restored_validation_loader;
  std::vector<std::pair<std::shared_ptr<Callback>, CallbackState>>
      restored_callbacks;
  Tensor restored_cpu_rng;
  std::vector<Tensor> restored_cuda_rng;

  {
    auto require_int = [&](const char* key) -> int64_t {
      if (!archive.try_read(key, value) || !value.isInt()) {
        throw std::runtime_error(std::string("Missing checkpoint field: ") +
                                 key);
      }
      return value.toInt();
    };
    auto require_bool = [&](const char* key) -> bool {
      if (!archive.try_read(key, value) || !value.isBool()) {
        throw std::runtime_error(std::string("Missing checkpoint field: ") +
                                 key);
      }
      return value.toBool();
    };

    if (require_int("config/accumulation_steps") !=
            options_.accumulation_steps ||
        require_bool("config/use_amp") != options_.use_amp ||
        require_int("config/amp_dtype") !=
            static_cast<int64_t>(options_.amp_dtype) ||
        require_int("config/device_type") !=
            static_cast<int64_t>(options_.device.type()) ||
        require_int("config/device_index") !=
            static_cast<int64_t>(options_.device.index())) {
      throw std::invalid_argument(
          "Checkpoint training configuration is incompatible with this "
          "Trainer");
    }
    if (!archive.try_read("checkpoint/consistency_boundary", value) ||
        !value.isString() || value.toStringRef() != "epoch") {
      throw std::runtime_error(
          "Checkpoint does not declare a supported consistency boundary");
    }
    if (require_bool("checkpoint/has_scheduler") != (scheduler_ != nullptr)) {
      throw std::invalid_argument(
          "Checkpoint scheduler configuration does not match this Trainer");
    }
    if (require_bool("checkpoint/has_grad_scaler") !=
        (grad_scaler_ != nullptr)) {
      throw std::invalid_argument(
          "Checkpoint GradScaler configuration does not match this Trainer");
    }

    torch::serialize::InputArchive callbacks_archive;
    if (!archive.try_read("callbacks", callbacks_archive)) {
      throw std::runtime_error("Checkpoint callback state is missing");
    }
    if (!callbacks_archive.try_read("count", value) || !value.isInt() ||
        value.toInt() < 0) {
      throw std::runtime_error("Checkpoint callback count is invalid");
    }
    const auto callbacks = checkpointable_callbacks(options_.callbacks);
    if (static_cast<uint64_t>(value.toInt()) != callbacks.size()) {
      throw std::invalid_argument(
          "Checkpoint callback configuration does not match this Trainer");
    }
    restored_callbacks.reserve(callbacks.size());
    for (size_t i = 0; i < callbacks.size(); ++i) {
      torch::serialize::InputArchive callback_archive;
      if (!callbacks_archive.try_read(std::to_string(i), callback_archive)) {
        throw std::runtime_error("Checkpoint callback state is incomplete");
      }
      if (!callback_archive.try_read("state_key", value) || !value.isString() ||
          value.toStringRef() != callbacks[i]->state_key()) {
        throw std::invalid_argument(
            "Checkpoint callback order or type does not match this Trainer");
      }
      restored_callbacks.emplace_back(callbacks[i],
                                      read_callback_state(callback_archive));
    }

    torch::serialize::InputArchive loader_archive;
    if (archive.try_read("data/train_loader", loader_archive)) {
      restored_train_loader = read_data_loader_state(loader_archive);
    }
    torch::serialize::InputArchive validation_archive;
    if (archive.try_read("data/validation_loader", validation_archive)) {
      restored_validation_loader = read_data_loader_state(validation_archive);
    }
    if (!archive.try_read("rng/cpu", restored_cpu_rng) ||
        !restored_cpu_rng.defined()) {
      throw std::runtime_error("Checkpoint CPU RNG state is missing");
    }
    const int64_t cuda_rng_count = require_int("rng/cuda_count");
    const int64_t available_cuda_rng_count = static_cast<int64_t>(
        core::get_rng_state(options_.device.is_cuda()).cuda.size());
    if (cuda_rng_count < 0 || cuda_rng_count != available_cuda_rng_count) {
      throw std::invalid_argument(
          "Checkpoint CUDA RNG topology does not match this Trainer");
    }
    restored_cuda_rng.reserve(static_cast<size_t>(cuda_rng_count));
    for (int64_t i = 0; i < cuda_rng_count; ++i) {
      Tensor rng_state;
      if (!archive.try_read("rng/cuda/" + std::to_string(i), rng_state) ||
          !rng_state.defined()) {
        throw std::runtime_error("Checkpoint CUDA RNG state is incomplete");
      }
      restored_cuda_rng.push_back(rng_state.cpu());
    }
    core::validate_rng_state(
        core::RngState{restored_cpu_rng.cpu(), restored_cuda_rng},
        options_.device.is_cuda());
  }

  torch::serialize::InputArchive model_archive;
  if (!archive.try_read("model", model_archive)) {
    throw std::runtime_error("Checkpoint model state is missing");
  }

  std::optional<std::map<std::string, Tensor>> restored_optimizer;
  torch::serialize::InputArchive optimizer_archive;
  if (archive.try_read("optimizer", optimizer_archive)) {
    std::map<std::string, Tensor> state;
    for (const auto& key : optimizer_archive.keys()) {
      Tensor tensor_value;
      if (optimizer_archive.try_read(key, tensor_value)) {
        state.emplace(key, std::move(tensor_value));
      }
    }
    restored_optimizer = std::move(state);
  } else {
    throw std::runtime_error("Checkpoint optimizer state is missing");
  }

  std::optional<std::map<std::string, double>> restored_scheduler;
  torch::serialize::InputArchive scheduler_archive;
  if (scheduler_ && archive.try_read("scheduler", scheduler_archive)) {
    restored_scheduler = read_scalar_state(scheduler_archive);
  } else if (scheduler_) {
    throw std::runtime_error("Checkpoint scheduler state is missing");
  }

  std::optional<amp::GradScalerState> restored_scaler;
  torch::serialize::InputArchive scaler_archive;
  if (grad_scaler_ && archive.try_read("grad_scaler", scaler_archive)) {
    restored_scaler = read_grad_scaler_state(scaler_archive);
    grad_scaler_->validate_state(*restored_scaler);
  } else if (grad_scaler_) {
    throw std::runtime_error("Checkpoint GradScaler state is missing");
  }

  auto read_int_field = [&](const char* key) -> std::optional<int64_t> {
    torch::IValue field;
    if (!archive.try_read(key, field)) {
      throw std::runtime_error(std::string("Missing checkpoint field: ") + key);
    }
    if (!field.isInt()) {
      throw std::runtime_error(std::string("Invalid checkpoint field: ") + key);
    }
    return field.toInt();
  };
  auto read_double_field = [&](const char* key) -> std::optional<double> {
    torch::IValue field;
    if (!archive.try_read(key, field)) {
      throw std::runtime_error(std::string("Missing checkpoint field: ") + key);
    }
    if (!field.isDouble()) {
      throw std::runtime_error(std::string("Invalid checkpoint field: ") + key);
    }
    return field.toDouble();
  };
  auto read_bool_field = [&](const char* key) -> std::optional<bool> {
    torch::IValue field;
    if (!archive.try_read(key, field)) {
      throw std::runtime_error(std::string("Missing checkpoint field: ") + key);
    }
    if (!field.isBool()) {
      throw std::runtime_error(std::string("Invalid checkpoint field: ") + key);
    }
    return field.toBool();
  };

  const auto restored_global_step = read_int_field("trainer/global_step");
  const auto restored_current_epoch = read_int_field("trainer/current_epoch");
  const auto restored_next_epoch = read_int_field("trainer/next_epoch");
  const auto restored_last_train_loss =
      read_double_field("trainer/last_train_loss");
  const auto restored_last_val_loss =
      read_double_field("trainer/last_val_loss");
  const auto restored_has_validation_loss =
      read_bool_field("trainer/has_validation_loss");
  const auto restored_should_stop = read_bool_field("trainer/should_stop");

  if (restored_global_step && *restored_global_step < 0) {
    throw std::runtime_error("Checkpoint global_step must be non-negative");
  }
  const auto validate_epoch = [](const std::optional<int64_t>& epoch,
                                 const char* name) {
    if (epoch && (*epoch < 0 || *epoch > std::numeric_limits<int>::max())) {
      throw std::runtime_error(std::string("Checkpoint ") + name +
                               " is out of range");
    }
  };
  validate_epoch(restored_current_epoch, "current_epoch");
  validate_epoch(restored_next_epoch, "next_epoch");

  const auto model_parameters = model_->parameters();
  const auto model_buffers = model_->buffers();
  std::vector<Tensor> parameter_snapshot;
  std::vector<Tensor> buffer_snapshot;
  parameter_snapshot.reserve(model_parameters.size());
  buffer_snapshot.reserve(model_buffers.size());
  for (const auto& parameter : model_parameters) {
    parameter_snapshot.push_back(parameter.detach().clone());
  }
  for (const auto& buffer : model_buffers) {
    buffer_snapshot.push_back(buffer.detach().clone());
  }
  const auto optimizer_snapshot = optimizer_->state_dict();
  const auto scheduler_snapshot =
      scheduler_ ? std::optional<std::map<std::string, double>>(
                       scheduler_->state_dict())
                 : std::nullopt;
  const auto scaler_snapshot =
      grad_scaler_
          ? std::optional<amp::GradScalerState>(grad_scaler_->capture_state())
          : std::nullopt;
  std::vector<std::pair<std::shared_ptr<Callback>, CallbackState>>
      callback_snapshot;
  callback_snapshot.reserve(restored_callbacks.size());
  for (const auto& [callback, state] : restored_callbacks) {
    (void)state;
    callback_snapshot.emplace_back(callback, callback->state_dict());
  }
  const auto rng_snapshot = core::get_rng_state(options_.device.is_cuda());
  const auto train_loader_snapshot = train_loader_state_;
  const auto validation_loader_snapshot = validation_loader_state_;
  const bool restore_train_loader_snapshot = restore_train_loader_state_;
  const bool restore_validation_loader_snapshot =
      restore_validation_loader_state_;
  const bool checkpoint_boundary_snapshot = checkpoint_at_epoch_boundary_;
  const int64_t global_step_snapshot = global_step_;
  const int current_epoch_snapshot = current_epoch_;
  const int next_epoch_snapshot = next_epoch_;
  const double last_train_loss_snapshot = last_train_loss_;
  const double last_val_loss_snapshot = last_val_loss_;
  const bool has_validation_loss_snapshot = has_validation_loss_;
  const bool should_stop_snapshot = should_stop_;

  try {
    model_->load(model_archive);
    if (restored_optimizer) {
      optimizer_->load_state_dict(*restored_optimizer);
    }
    if (scheduler_ && restored_scheduler) {
      scheduler_->load_state_dict(*restored_scheduler);
    }
    if (grad_scaler_ && restored_scaler) {
      grad_scaler_->restore_state(*restored_scaler);
    }
    for (auto& [callback, state] : restored_callbacks) {
      callback->load_state_dict(state);
    }

    if (restored_global_step) global_step_ = *restored_global_step;
    if (restored_current_epoch) {
      current_epoch_ = static_cast<int>(*restored_current_epoch);
    }
    if (restored_next_epoch) {
      next_epoch_ = static_cast<int>(*restored_next_epoch);
    }
    if (restored_last_train_loss) last_train_loss_ = *restored_last_train_loss;
    if (restored_last_val_loss) last_val_loss_ = *restored_last_val_loss;
    if (restored_has_validation_loss) {
      has_validation_loss_ = *restored_has_validation_loss;
    }
    if (restored_should_stop) should_stop_ = *restored_should_stop;

    train_loader_state_ = std::move(restored_train_loader);
    validation_loader_state_ = std::move(restored_validation_loader);
    restore_train_loader_state_ = train_loader_state_.has_value();
    restore_validation_loader_state_ = validation_loader_state_.has_value();
    if (restored_cpu_rng.defined()) {
      core::set_rng_state(
          core::RngState{restored_cpu_rng.cpu(), restored_cuda_rng},
          options_.device.is_cuda());
    }
    checkpoint_at_epoch_boundary_ = true;
  } catch (...) {
    const std::exception_ptr restore_error = std::current_exception();
    std::string rollback_errors;
    auto rollback = [&](const char* component, auto&& action) {
      try {
        action();
      } catch (const std::exception& error) {
        if (!rollback_errors.empty()) rollback_errors += "; ";
        rollback_errors += std::string(component) + ": " + error.what();
      } catch (...) {
        if (!rollback_errors.empty()) rollback_errors += "; ";
        rollback_errors += std::string(component) + ": unknown error";
      }
    };

    rollback("model", [&]() {
      torch::NoGradGuard no_grad;
      for (size_t i = 0; i < model_parameters.size(); ++i) {
        model_parameters[i].copy_(parameter_snapshot[i]);
      }
      for (size_t i = 0; i < model_buffers.size(); ++i) {
        model_buffers[i].copy_(buffer_snapshot[i]);
      }
    });
    rollback("optimizer",
             [&]() { optimizer_->load_state_dict(optimizer_snapshot); });
    if (scheduler_ && scheduler_snapshot) {
      rollback("scheduler",
               [&]() { scheduler_->load_state_dict(*scheduler_snapshot); });
    }
    if (grad_scaler_ && scaler_snapshot) {
      rollback("grad_scaler",
               [&]() { grad_scaler_->restore_state(*scaler_snapshot); });
    }
    for (auto& callback_state : callback_snapshot) {
      auto& callback = callback_state.first;
      auto& state = callback_state.second;
      rollback("callback", [&]() { callback->load_state_dict(state); });
    }
    rollback("rng", [&]() {
      core::set_rng_state(rng_snapshot, options_.device.is_cuda());
    });

    train_loader_state_ = train_loader_snapshot;
    validation_loader_state_ = validation_loader_snapshot;
    restore_train_loader_state_ = restore_train_loader_snapshot;
    restore_validation_loader_state_ = restore_validation_loader_snapshot;
    checkpoint_at_epoch_boundary_ = checkpoint_boundary_snapshot;
    global_step_ = global_step_snapshot;
    current_epoch_ = current_epoch_snapshot;
    next_epoch_ = next_epoch_snapshot;
    last_train_loss_ = last_train_loss_snapshot;
    last_val_loss_ = last_val_loss_snapshot;
    has_validation_loss_ = has_validation_loss_snapshot;
    should_stop_ = should_stop_snapshot;

    if (!rollback_errors.empty()) {
      try {
        std::rethrow_exception(restore_error);
      } catch (...) {
        std::throw_with_nested(std::runtime_error(
            "Checkpoint restore failed and rollback was incomplete: " +
            rollback_errors));
      }
    }
    std::rethrow_exception(restore_error);
  }
}

}  // namespace training
}  // namespace jyppx
