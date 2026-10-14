#ifndef JYPPX_TRAINING_TRAINER_HPP
#define JYPPX_TRAINING_TRAINER_HPP

#include <jyppx/core/types.hpp>
#include <jyppx/data/dataloader.hpp>
#include <jyppx/optim/optimizer.hpp>
#include <jyppx/optim/schedulers/lr_scheduler.hpp>
#include <jyppx/training/amp/grad_scaler.hpp>
#include <jyppx/training/callback.hpp>

#include <torch/nn/module.h>
#include <torch/nn/utils.h>

#include <ATen/core/ScalarType.h>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace jyppx {
namespace training {

/**
 * @brief Configures one Trainer's loop, device, precision, and callbacks.
 */
struct TrainerOptions {
  /** @brief Callable mapping one input tensor to one prediction tensor. */
  using ForwardFn = std::function<Tensor(const Tensor&)>;

  /** @brief Callable reducing prediction and target tensors to one scalar loss.
   */
  using LossFn = std::function<Tensor(const Tensor&, const Tensor&)>;

  int max_epochs =
      10;  ///< Non-negative exclusive upper bound for epoch indices.
  int max_steps = -1;  ///< Positive optimizer-step limit; -1 means unlimited.
  Device device = torch::kCPU;  ///< Device receiving model and batch tensors.
  double gradient_clip_val =
      0.0;  ///< Non-negative clip threshold; zero disables clipping.
  std::string gradient_clip_mode =
      "norm";            ///< Clipping algorithm: `norm` or `value`.
  bool use_amp = false;  ///< Whether forward and loss execute under autocast.
  at::ScalarType amp_dtype =
      at::kHalf;                       ///< CUDA Half/BFloat16 or CPU BFloat16.
  amp::GradScalerOptions amp_options;  ///< Float16 loss-scaling configuration.
  int accumulation_steps = 1;  ///< Positive batch count per optimizer update.
  ForwardFn
      forward_fn;  ///< Optional invocation override for non-standard modules.

  /** @brief Scalar objective; defaults to mean-squared error. */
  LossFn loss_fn = [](const Tensor& prediction, const Tensor& target) {
    return torch::mse_loss(prediction, target);
  };
  std::vector<std::shared_ptr<Callback>>
      callbacks;  ///< Ordered shared lifecycle observers.
  uint64_t max_checkpoint_bytes =
      4ULL * 1024ULL * 1024ULL *
      1024ULL;  ///< Positive archive read/write limit in bytes.
};

/**
 * @brief Coordinates model training, validation, callbacks, and recovery.
 *
 * The Trainer shares ownership of the model, optimizer, scheduler, and
 * callbacks. It owns the GradScaler and loop state. Instances are not
 * thread-safe and checkpoint operations require an epoch consistency boundary.
 */
class Trainer {
 public:
  /**
   * @brief Creates a trainer for a type-erased module.
   * @param model Shared non-null LibTorch module.
   * @param optimizer Shared non-null optimizer referencing model parameters.
   * @param options Valid training configuration.
   * @throws std::invalid_argument A required object or option is invalid.
   * @throws std::runtime_error Requested AMP is unavailable on the device.
   */
  Trainer(std::shared_ptr<torch::nn::Module> model,
          std::shared_ptr<optim::Optimizer> optimizer,
          const TrainerOptions& options = TrainerOptions{});

  /**
   * @brief Construct a trainer from a concrete LibTorch module.
   *
   * The concrete module type supplies the type-safe `forward(Tensor)` call;
   * the trainer stores the resulting callable in a type-erased form.
   *
   * @tparam ModuleT Concrete LibTorch module with `forward(Tensor)`.
   * @param model Shared non-null concrete module.
   * @param optimizer Shared non-null optimizer referencing model parameters.
   * @param options Valid training configuration.
   * @throws std::invalid_argument A required object or option is invalid.
   * @throws std::runtime_error Requested AMP is unavailable on the device.
   */
  template <typename ModuleT,
            typename =
                std::enable_if_t<std::is_base_of_v<torch::nn::Module, ModuleT>>>
  Trainer(std::shared_ptr<ModuleT> model,
          std::shared_ptr<optim::Optimizer> optimizer,
          const TrainerOptions& options = TrainerOptions{})
      : Trainer(std::static_pointer_cast<torch::nn::Module>(model), optimizer,
                options) {
    if (!options.forward_fn) {
      forward_fn_ = [model](const Tensor& input) {
        return model->forward(input);
      };
    }
  }

  /**
   * @brief Fits the model until the epoch, step, or callback stop limit.
   * @tparam T Supported batch sample type (`Tensor` or `data::Example`).
   * @param train_loader Non-empty training loader.
   * @param val_loader Optional validation loader; borrowed for the call.
   */
  template <typename T = Tensor>
  void fit(data::DataLoader<T>& train_loader,
           data::DataLoader<T>* val_loader = nullptr);

  /**
   * @brief Evaluates mean loss without updating parameters.
   * @tparam T Supported batch sample type.
   * @param val_loader Non-empty validation loader.
   * @return Sample-weighted mean validation loss.
   */
  template <typename T = Tensor>
  double validate(data::DataLoader<T>& val_loader);

  /**
   * @brief Evaluates mean test loss without validation callbacks.
   * @tparam T Supported batch sample type.
   * @param test_loader Non-empty test loader.
   * @return Sample-weighted mean test loss.
   */
  template <typename T = Tensor>
  double test(data::DataLoader<T>& test_loader);

  /**
   * @brief Sets or replaces the type-erased model forward function.
   * @param forward_fn Non-empty callable retained by value.
   * @throws std::invalid_argument @p forward_fn is empty.
   */
  void set_forward_fn(TrainerOptions::ForwardFn forward_fn);

  /**
   * @brief Sets or replaces the scalar loss function.
   * @param loss_fn Non-empty callable retained by value.
   * @throws std::invalid_argument @p loss_fn is empty.
   */
  void set_loss_fn(TrainerOptions::LossFn loss_fn);

  /**
   * @brief Attaches a scheduler stepped after each epoch.
   * @param scheduler Shared scheduler, or null to detach the current scheduler.
   * @throws std::invalid_argument Scheduler and optimizer group counts differ.
   */
  void set_scheduler(std::shared_ptr<optim::schedulers::LRScheduler> scheduler);

  /**
   * @brief Appends a callback to lifecycle order.
   * @param callback Shared non-null callback retained by the Trainer.
   * @throws std::invalid_argument @p callback is null.
   */
  void add_callback(std::shared_ptr<Callback> callback);

  /** @brief Requests that fit stop after the current safe boundary. */
  void request_stop() { should_stop_ = true; }

  /**
   * @brief Runs an action after every callback observes the epoch end.
   * @param action Non-empty action, usually atomic checkpoint publication.
   * @throws std::invalid_argument @p action is empty.
   */
  void defer_epoch_end_action(std::function<void()> action);

  /**
   * @brief Save the versioned training state at a consistent epoch boundary.
   *
   * Saving from a batch callback is rejected because the current loader does
   * not yet expose an exact in-epoch cursor or accumulated-gradient state.
   *
   * @param path Non-empty destination path atomically replaced on success.
   * @throws std::logic_error The Trainer is not at an epoch boundary.
   * @throws std::runtime_error Serialization, size checks, or publication fail.
   */
  void save_checkpoint(const std::string& path) const;

  /**
   * @brief Restores a complete checkpoint as one transaction.
   *
   * All archive fields are parsed before mutation. If any participant fails,
   * every already-applied serializable state is rolled back.
   *
   * @param path Existing checkpoint path within max_checkpoint_bytes.
   * @throws std::invalid_argument Runtime topology or configuration differs.
   * @throws std::runtime_error The archive is missing, malformed, or
   * incomplete.
   */
  void load_checkpoint(const std::string& path);

  /** @brief Returns the shared model used by the training loop. */
  const std::shared_ptr<torch::nn::Module>& model() const { return model_; }

  /** @brief Returns the shared optimizer used by the training loop. */
  const std::shared_ptr<optim::Optimizer>& optimizer() const {
    return optimizer_;
  }

  /** @brief Returns the attached scheduler, or null when none is configured. */
  const std::shared_ptr<optim::schedulers::LRScheduler>& scheduler() const {
    return scheduler_;
  }

  /** @brief Active gradient scaler, or null for disabled AMP and bfloat16. */
  const std::unique_ptr<amp::GradScaler>& grad_scaler() const {
    return grad_scaler_;
  }

  /** @brief Returns the last completed epoch's sample-weighted train loss. */
  double last_train_loss() const { return last_train_loss_; }

  /** @brief Returns the latest sample-weighted validation loss. */
  double last_val_loss() const { return last_val_loss_; }

  /** @brief Returns whether last_val_loss() contains a completed result. */
  bool has_validation_loss() const { return has_validation_loss_; }

  /** @brief Returns whether a callback requested training termination. */
  bool should_stop() const { return should_stop_; }

  /**
   * @brief Returns completed optimizer updates across all epochs.
   */
  int64_t global_step() const { return global_step_; }

  /**
   * @brief Returns the zero-based active or most recently completed epoch.
   */
  int current_epoch() const { return current_epoch_; }

 private:
  std::shared_ptr<torch::nn::Module>
      model_;  ///< Shared model whose tensors are mutated.
  std::shared_ptr<optim::Optimizer>
      optimizer_;  ///< Shared optimizer for model parameters.
  std::shared_ptr<optim::schedulers::LRScheduler>
      scheduler_;  ///< Optional shared epoch scheduler.
  std::unique_ptr<amp::GradScaler>
      grad_scaler_;  ///< Owned float16 loss scaler, if required.
  TrainerOptions
      options_;  ///< Validated immutable loop configuration and callbacks.
  TrainerOptions::ForwardFn forward_fn_;  ///< Type-erased model invocation.
  TrainerOptions::LossFn
      loss_fn_;          ///< Scalar training and evaluation objective.
  int64_t global_step_;  ///< Completed optimizer updates; always non-negative.
  int current_epoch_;    ///< Zero-based active or last completed loop epoch.
  int next_epoch_;       ///< First epoch to execute after checkpoint recovery.
  bool should_stop_;     ///< Callback-controlled fit termination request.
  double last_train_loss_;  ///< Sample-weighted mean from the last completed
                            ///< train epoch.
  double last_val_loss_;    ///< Sample-weighted mean from the latest validation
                            ///< pass.
  bool has_validation_loss_;  ///< Whether last_val_loss_ is valid.
  std::optional<data::DataLoaderState>
      train_loader_state_;  ///< Saved train order at a boundary.
  std::optional<data::DataLoaderState>
      validation_loader_state_;  ///< Saved validation order.
  bool restore_train_loader_state_ =
      false;  ///< Whether fit must apply train_loader_state_.
  bool restore_validation_loader_state_ =
      false;  ///< Whether fit must apply validation state.
  bool checkpoint_at_epoch_boundary_ =
      true;  ///< Whether exact checkpoint save is permitted.
  bool dispatching_epoch_end_callbacks_ =
      false;  ///< Whether epoch-end hooks are active.
  std::vector<std::function<void()>>
      deferred_epoch_end_actions_;  ///< Post-hook actions.

  /**
   * @brief Converts supported batch representations to input and target
   * tensors.
   * @tparam T Batch sample type.
   * @param batch Loaded batch.
   * @return Input/target pair borrowed from the batch tensors.
   */
  template <typename T>
  static std::pair<Tensor, Tensor> unpack_batch(const data::Batch<T>& batch) {
    if constexpr (std::is_same_v<T, Tensor>) {
      // A tensor-only loader has no separate labels. Treat it as an
      // autoencoder-style batch so the default MSE loss remains useful.
      return {batch.data, batch.data};
    } else if constexpr (std::is_same_v<T, data::Example>) {
      return {batch.data, batch.targets};
    } else {
      throw std::invalid_argument(
          "Trainer supports Tensor and TensorDataset::Example batches; "
          "provide a custom training loop for other sample types");
    }
  }

  /**
   * @brief Validates all Trainer options and owned participant pointers.
   * @throws std::invalid_argument A configuration field is invalid.
   * @throws std::runtime_error Requested AMP is unavailable.
   */
  void validate_configuration() const;

  /**
   * @brief Rejects training when no forward function is available.
   * @throws std::invalid_argument forward_fn_ is empty.
   */
  void require_forward_fn() const;

  /**
   * @brief Invokes the model and scalar loss under configured autocast.
   * @param input Batch input tensor on the configured device.
   * @param target Batch target tensor on the configured device.
   * @return Loss tensor produced by loss_fn_.
   */
  Tensor compute_loss(const Tensor& input, const Tensor& target) const;

  /** @brief Applies the configured norm or value gradient clipping. */
  void clip_gradients();

  /**
   * @brief Multiplies every defined model gradient by @p scale.
   * @param scale Finite correction applied in place to accumulated gradients.
   */
  void scale_gradients(double scale);

  /**
   * @brief Runs one validation pass and optionally dispatches callbacks.
   * @tparam T Supported batch sample type.
   * @param loader Non-empty loader borrowed for the call.
   * @param fire_callbacks Whether validation lifecycle hooks are dispatched.
   * @return Sample-weighted mean loss for the pass.
   */
  template <typename T>
  double run_validation(data::DataLoader<T>& loader, bool fire_callbacks);
};

template <typename T>
void Trainer::fit(data::DataLoader<T>& train_loader,
                  data::DataLoader<T>* val_loader) {
  validate_configuration();
  require_forward_fn();
  if (train_loader.size() == 0) {
    throw std::invalid_argument("Trainer requires a non-empty training loader");
  }

  if (restore_train_loader_state_) {
    train_loader.load_state_dict(*train_loader_state_);
    restore_train_loader_state_ = false;
  }
  if (val_loader != nullptr && restore_validation_loader_state_) {
    val_loader->load_state_dict(*validation_loader_state_);
    restore_validation_loader_state_ = false;
  } else if (val_loader == nullptr) {
    validation_loader_state_.reset();
    restore_validation_loader_state_ = false;
  }

  train_loader_state_ = train_loader.state_dict();
  if (val_loader != nullptr) {
    validation_loader_state_ = val_loader->state_dict();
  }
  checkpoint_at_epoch_boundary_ = true;

  model_->to(options_.device);
  model_->train();
  optimizer_->zero_grad();
  should_stop_ = false;
  has_validation_loss_ = false;

  for (const auto& callback : options_.callbacks) {
    if (callback) {
      callback->on_fit_start(this);
    }
  }

  bool reached_max_steps = false;
  for (current_epoch_ = next_epoch_;
       current_epoch_ < options_.max_epochs &&
       (options_.max_steps < 0 || global_step_ < options_.max_steps);
       ++current_epoch_) {
    checkpoint_at_epoch_boundary_ = false;
    for (const auto& callback : options_.callbacks) {
      if (callback) {
        callback->on_epoch_start(this);
        callback->on_train_epoch_start(this);
      }
    }

    model_->train();
    optimizer_->zero_grad();
    double weighted_loss = 0.0;
    size_t sample_count = 0;
    int pending_batches = 0;
    int batch_index = 0;

    for (const auto batch : train_loader) {
      for (const auto& callback : options_.callbacks) {
        if (callback) {
          callback->on_train_batch_start(this, batch_index);
        }
      }

      auto [input, target] = unpack_batch<T>(batch);
      if (!input.defined() || !target.defined() || input.size(0) == 0) {
        throw std::invalid_argument(
            "Trainer received an empty or undefined batch");
      }
      input = input.to(options_.device);
      target = target.to(options_.device);

      Tensor loss = compute_loss(input, target);
      if (!loss.defined() || loss.numel() != 1) {
        throw std::invalid_argument(
            "Trainer loss function must return one scalar tensor");
      }
      const auto batch_size = batch.batch_size();
      const double batch_loss = loss.detach().item<double>();
      weighted_loss += batch_loss * static_cast<double>(batch_size);
      sample_count += batch_size;

      const Tensor backward_loss =
          grad_scaler_
              ? grad_scaler_->scale(loss) /
                    static_cast<double>(options_.accumulation_steps)
              : loss / static_cast<double>(options_.accumulation_steps);
      backward_loss.backward();
      ++pending_batches;
      const bool accumulation_boundary =
          pending_batches == options_.accumulation_steps;
      const bool last_batch =
          (batch_index + 1) >= static_cast<int>(train_loader.size());
      if (accumulation_boundary || last_batch) {
        if (pending_batches < options_.accumulation_steps) {
          scale_gradients(static_cast<double>(options_.accumulation_steps) /
                          static_cast<double>(pending_batches));
        }
        if (grad_scaler_) {
          // Unscale first so clipping sees true gradient magnitudes;
          // step() reuses the unscale result and skips on overflow.
          if (options_.gradient_clip_val > 0.0) {
            grad_scaler_->unscale(optimizer_);
          }
          grad_scaler_->step(optimizer_);
          grad_scaler_->update();
        } else {
          if (options_.gradient_clip_val > 0.0) {
            clip_gradients();
          }
          optimizer_->step();
        }
        optimizer_->zero_grad();
        pending_batches = 0;
        ++global_step_;
        if (options_.max_steps > 0 && global_step_ >= options_.max_steps) {
          reached_max_steps = true;
        }
      }

      for (const auto& callback : options_.callbacks) {
        if (callback) {
          callback->on_train_batch_end(this, batch_index, batch_loss);
        }
      }
      ++batch_index;
      if (reached_max_steps) {
        break;
      }
    }

    last_train_loss_ = sample_count == 0
                           ? 0.0
                           : weighted_loss / static_cast<double>(sample_count);
    for (const auto& callback : options_.callbacks) {
      if (callback) {
        callback->on_train_epoch_end(this);
      }
    }

    if (val_loader != nullptr) {
      last_val_loss_ = run_validation(*val_loader, true);
      has_validation_loss_ = true;
    } else {
      has_validation_loss_ = false;
    }

    if (scheduler_) {
      if (scheduler_->requires_metric()) {
        if (!has_validation_loss_) {
          throw std::invalid_argument(
              "Metric-driven scheduler requires a validation loader");
        }
        scheduler_->step(last_val_loss_);
      } else {
        scheduler_->step();
      }
    }
    next_epoch_ = current_epoch_ + 1;
    train_loader_state_ = train_loader.state_dict();
    if (val_loader != nullptr) {
      validation_loader_state_ = val_loader->state_dict();
    }
    checkpoint_at_epoch_boundary_ = true;
    deferred_epoch_end_actions_.clear();
    dispatching_epoch_end_callbacks_ = true;
    try {
      for (const auto& callback : options_.callbacks) {
        if (callback) {
          callback->on_epoch_end(this);
        }
      }
    } catch (...) {
      dispatching_epoch_end_callbacks_ = false;
      deferred_epoch_end_actions_.clear();
      throw;
    }
    dispatching_epoch_end_callbacks_ = false;
    auto deferred_actions = std::move(deferred_epoch_end_actions_);
    deferred_epoch_end_actions_.clear();
    for (auto& action : deferred_actions) {
      action();
    }
    if (reached_max_steps || should_stop_) {
      break;
    }
  }

  for (const auto& callback : options_.callbacks) {
    if (callback) {
      callback->on_fit_end(this);
    }
  }
}

template <typename T>
double Trainer::validate(data::DataLoader<T>& val_loader) {
  validate_configuration();
  require_forward_fn();
  return run_validation(val_loader, true);
}

template <typename T>
double Trainer::test(data::DataLoader<T>& test_loader) {
  validate_configuration();
  require_forward_fn();
  return run_validation(test_loader, false);
}

template <typename T>
double Trainer::run_validation(data::DataLoader<T>& loader,
                               bool fire_callbacks) {
  if (loader.size() == 0) {
    throw std::invalid_argument(
        "Trainer requires a non-empty validation loader");
  }
  if (fire_callbacks) {
    for (const auto& callback : options_.callbacks) {
      if (callback) {
        callback->on_validation_epoch_start(this);
      }
    }
  }

  const bool was_training = model_->is_training();
  /** @brief Restores a borrowed module's entry training mode on scope exit. */
  struct TrainingModeGuard {
    torch::nn::Module& module;  ///< Borrowed module whose mode is restored.
    bool was_training;          ///< Training flag observed before validation.

    /** @brief Restores the captured training flag. */
    ~TrainingModeGuard() { module.train(was_training); }
  } mode_guard{*model_, was_training};
  model_->eval();
  torch::NoGradGuard no_grad;
  double weighted_loss = 0.0;
  size_t sample_count = 0;
  int batch_index = 0;
  for (const auto batch : loader) {
    for (const auto& callback : options_.callbacks) {
      if (fire_callbacks && callback) {
        callback->on_validation_batch_start(this, batch_index);
      }
    }
    auto [input, target] = unpack_batch<T>(batch);
    if (!input.defined() || !target.defined() || input.size(0) == 0) {
      throw std::invalid_argument(
          "Trainer received an empty or undefined validation batch");
    }
    input = input.to(options_.device);
    target = target.to(options_.device);
    Tensor loss = compute_loss(input, target);
    if (!loss.defined() || loss.numel() != 1) {
      throw std::invalid_argument(
          "Trainer loss function must return one scalar tensor");
    }
    const auto batch_size = batch.batch_size();
    weighted_loss += loss.item<double>() * static_cast<double>(batch_size);
    sample_count += batch_size;
    for (const auto& callback : options_.callbacks) {
      if (fire_callbacks && callback) {
        callback->on_validation_batch_end(this, batch_index);
      }
    }
    ++batch_index;
  }

  const double mean_loss =
      sample_count == 0 ? 0.0
                        : weighted_loss / static_cast<double>(sample_count);
  if (fire_callbacks) {
    last_val_loss_ = mean_loss;
    has_validation_loss_ = true;
    for (const auto& callback : options_.callbacks) {
      if (callback) {
        callback->on_validation_epoch_end(this);
      }
    }
  }
  return mean_loss;
}

}  // namespace training
}  // namespace jyppx

#endif  // JYPPX_TRAINING_TRAINER_HPP
