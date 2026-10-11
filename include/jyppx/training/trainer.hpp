#ifndef JYPPX_TRAINING_TRAINER_HPP
#define JYPPX_TRAINING_TRAINER_HPP

#include <jyppx/core/types.hpp>
#include <jyppx/optim/optimizer.hpp>
#include <jyppx/optim/schedulers/lr_scheduler.hpp>
#include <jyppx/data/dataloader.hpp>
#include <jyppx/training/amp/grad_scaler.hpp>
#include <jyppx/training/callback.hpp>
#include <torch/nn/module.h>
#include <torch/nn/utils.h>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace jyppx {
namespace training {

/**
 * @brief Trainer options
 */
struct TrainerOptions {
    using ForwardFn = std::function<Tensor(const Tensor&)>;
    using LossFn = std::function<Tensor(const Tensor&, const Tensor&)>;

    int max_epochs = 10;                        ///< Maximum number of epochs
    int max_steps = -1;                         ///< Maximum optimizer steps (-1 = unlimited)
    Device device = torch::kCPU;                ///< Device to train on
    double gradient_clip_val = 0.0;             ///< Gradient clipping value (0 = disabled)
    std::string gradient_clip_mode = "norm";   ///< "norm" or "value"
    bool use_amp = false;                       ///< Automatic mixed precision (CUDA only)
    amp::GradScalerOptions amp_options;         ///< GradScaler settings for use_amp
    int accumulation_steps = 1;                 ///< Gradient accumulation steps
    ForwardFn forward_fn;                       ///< Optional type-erased model forward function
    LossFn loss_fn = [](const Tensor& prediction, const Tensor& target) {
        return torch::mse_loss(prediction, target);
    };
    std::vector<std::shared_ptr<Callback>> callbacks;  ///< Callbacks
};

/**
 * @brief High-level training abstraction
 */
class Trainer {
public:
    Trainer(
        std::shared_ptr<torch::nn::Module> model,
        std::shared_ptr<optim::Optimizer> optimizer,
        const TrainerOptions& options = TrainerOptions{}
    );

    /**
     * @brief Construct a trainer from a concrete LibTorch module.
     *
     * The concrete module type supplies the type-safe `forward(Tensor)` call;
     * the trainer stores the resulting callable in a type-erased form.
     */
    template <typename ModuleT,
              typename = std::enable_if_t<std::is_base_of_v<torch::nn::Module, ModuleT>>>
    Trainer(
        std::shared_ptr<ModuleT> model,
        std::shared_ptr<optim::Optimizer> optimizer,
        const TrainerOptions& options = TrainerOptions{})
        : Trainer(
              std::static_pointer_cast<torch::nn::Module>(model), optimizer, options) {
        if (!options.forward_fn) {
            forward_fn_ = [model](const Tensor& input) { return model->forward(input); };
        }
    }

    /**
     * @brief Fit the model
     */
    template<typename T = Tensor>
    void fit(
        data::DataLoader<T>& train_loader,
        data::DataLoader<T>* val_loader = nullptr
    );

    /**
     * @brief Validate the model
     */
    template<typename T = Tensor>
    double validate(data::DataLoader<T>& val_loader);

    /**
     * @brief Test the model
     */
    template<typename T = Tensor>
    double test(data::DataLoader<T>& test_loader);

    /** @brief Set or replace the type-erased model forward function. */
    void set_forward_fn(TrainerOptions::ForwardFn forward_fn);

    /** @brief Set or replace the loss function. */
    void set_loss_fn(TrainerOptions::LossFn loss_fn);

    /** @brief Attach a learning-rate scheduler stepped after each epoch. */
    void set_scheduler(std::shared_ptr<optim::schedulers::LRScheduler> scheduler);

    /** @brief Add a callback to the lifecycle callback list. */
    void add_callback(std::shared_ptr<Callback> callback);

    /** @brief Ask the fit loop to stop after the current epoch. */
    void request_stop() { should_stop_ = true; }

    /** @brief Save model, optimizer and loop counters to a LibTorch archive. */
    void save_checkpoint(const std::string& path) const;

    /** @brief Restore model, optimizer and loop counters from an archive. */
    void load_checkpoint(const std::string& path);

    /** @brief Current model and optimizer, exposed for callbacks. */
    const std::shared_ptr<torch::nn::Module>& model() const { return model_; }
    const std::shared_ptr<optim::Optimizer>& optimizer() const { return optimizer_; }
    const std::shared_ptr<optim::schedulers::LRScheduler>& scheduler() const {
        return scheduler_;
    }
    /** @brief Active gradient scaler, or null when use_amp is disabled. */
    const std::unique_ptr<amp::GradScaler>& grad_scaler() const { return grad_scaler_; }

    /** @brief Last epoch's mean train and validation losses. */
    double last_train_loss() const { return last_train_loss_; }
    double last_val_loss() const { return last_val_loss_; }
    bool has_validation_loss() const { return has_validation_loss_; }
    bool should_stop() const { return should_stop_; }

    /**
     * @brief Get global step count
     */
    int64_t global_step() const { return global_step_; }

    /**
     * @brief Get current epoch
     */
    int current_epoch() const { return current_epoch_; }

private:
    std::shared_ptr<torch::nn::Module> model_;
    std::shared_ptr<optim::Optimizer> optimizer_;
    std::shared_ptr<optim::schedulers::LRScheduler> scheduler_;
    std::unique_ptr<amp::GradScaler> grad_scaler_;
    TrainerOptions options_;
    TrainerOptions::ForwardFn forward_fn_;
    TrainerOptions::LossFn loss_fn_;
    int64_t global_step_;
    int current_epoch_;
    int next_epoch_;
    bool should_stop_;
    double last_train_loss_;
    double last_val_loss_;
    bool has_validation_loss_;

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

    void validate_configuration() const;
    void require_forward_fn() const;
    Tensor compute_loss(const Tensor& input, const Tensor& target) const;
    void clip_gradients();
    void scale_gradients(double scale);

    template <typename T>
    double run_validation(data::DataLoader<T>& loader, bool fire_callbacks);
};

template <typename T>
void Trainer::fit(data::DataLoader<T>& train_loader, data::DataLoader<T>* val_loader) {
    validate_configuration();
    require_forward_fn();
    if (train_loader.size() == 0) {
        throw std::invalid_argument("Trainer requires a non-empty training loader");
    }

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
                throw std::invalid_argument("Trainer received an empty or undefined batch");
            }
            input = input.to(options_.device);
            target = target.to(options_.device);

            Tensor loss = compute_loss(input, target);
            if (!loss.defined() || loss.numel() != 1) {
                throw std::invalid_argument("Trainer loss function must return one scalar tensor");
            }
            const auto batch_size = batch.batch_size();
            const double batch_loss = loss.detach().item<double>();
            weighted_loss += batch_loss * static_cast<double>(batch_size);
            sample_count += batch_size;

            const Tensor backward_loss = options_.use_amp
                ? grad_scaler_->scale(loss) /
                      static_cast<double>(options_.accumulation_steps)
                : loss / static_cast<double>(options_.accumulation_steps);
            backward_loss.backward();
            ++pending_batches;
            const bool accumulation_boundary =
                pending_batches == options_.accumulation_steps;
            const bool last_batch = (batch_index + 1) >= static_cast<int>(train_loader.size());
            if (accumulation_boundary || last_batch) {
                if (pending_batches < options_.accumulation_steps) {
                    scale_gradients(static_cast<double>(options_.accumulation_steps) /
                                    static_cast<double>(pending_batches));
                }
                if (options_.use_amp) {
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
            scheduler_->step();
        }
        next_epoch_ = current_epoch_ + 1;
        for (const auto& callback : options_.callbacks) {
            if (callback) {
                callback->on_epoch_end(this);
            }
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
double Trainer::run_validation(data::DataLoader<T>& loader, bool fire_callbacks) {
    if (loader.size() == 0) {
        throw std::invalid_argument("Trainer requires a non-empty validation loader");
    }
    if (fire_callbacks) {
        for (const auto& callback : options_.callbacks) {
            if (callback) {
                callback->on_validation_epoch_start(this);
            }
        }
    }

    const bool was_training = model_->is_training();
    struct TrainingModeGuard {
        torch::nn::Module& module;
        bool was_training;
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
            throw std::invalid_argument("Trainer received an empty or undefined validation batch");
        }
        input = input.to(options_.device);
        target = target.to(options_.device);
        Tensor loss = compute_loss(input, target);
        if (!loss.defined() || loss.numel() != 1) {
            throw std::invalid_argument("Trainer loss function must return one scalar tensor");
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

    const double mean_loss = sample_count == 0
        ? 0.0
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

} // namespace training
} // namespace jyppx

#endif // JYPPX_TRAINING_TRAINER_HPP
