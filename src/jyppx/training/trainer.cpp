#include <jyppx/training/trainer.hpp>

#include <jyppx/core/device.hpp>
#include <jyppx/training/amp/autocast.hpp>

#include <torch/serialize.h>

#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace jyppx {
namespace training {

namespace {

void write_scalar_state(
    torch::serialize::OutputArchive& archive,
    const std::map<std::string, double>& state) {
    for (const auto& [name, value] : state) {
        archive.write(
            name,
            torch::tensor(value, torch::TensorOptions().dtype(torch::kFloat64)));
    }
}

std::map<std::string, double> read_scalar_state(torch::serialize::InputArchive& archive) {
    std::map<std::string, double> state;
    for (const auto& key : archive.keys()) {
        torch::Tensor value;
        if (archive.try_read(key, value) && value.defined() && value.numel() == 1) {
            state.emplace(key, value.item<double>());
        }
    }
    return state;
}

} // namespace

Trainer::Trainer(
    std::shared_ptr<torch::nn::Module> model,
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
    if (options_.use_amp) {
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
        throw std::invalid_argument("Trainer gradient_clip_val must be non-negative");
    }
    if (options_.gradient_clip_mode != "norm" && options_.gradient_clip_mode != "value") {
        throw std::invalid_argument("Trainer gradient_clip_mode must be 'norm' or 'value'");
    }
    if (!loss_fn_) {
        throw std::invalid_argument("Trainer loss function cannot be empty");
    }
    if (options_.use_amp) {
        // Mixed precision training currently requires a CUDA device: float16
        // autocast is only wired up for CUDA in this framework.
        if (options_.device.type() != torch::kCUDA) {
            throw std::runtime_error(
                "Trainer AMP requires a CUDA device; set TrainerOptions::device to cuda");
        }
        if (!cuda_is_available()) {
            throw std::runtime_error(
                "Trainer AMP requires CUDA, but no CUDA device is available");
        }
    }
}

void Trainer::require_forward_fn() const {
    if (!forward_fn_) {
        throw std::invalid_argument(
            "Trainer requires a forward function; construct it with a concrete module, "
            "set TrainerOptions::forward_fn, or call Trainer::set_forward_fn");
    }
}

Tensor Trainer::compute_loss(const Tensor& input, const Tensor& target) const {
    if (options_.use_amp) {
        // Forward pass runs in float16 via autocast; the loss itself comes
        // back in float32 so logging and backward behave as in eager mode.
        amp::AutocastGuard autocast{options_.device.type(), at::kHalf};
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

void Trainer::set_scheduler(std::shared_ptr<optim::schedulers::LRScheduler> scheduler) {
    if (scheduler && scheduler->get_last_lrs().size() != optimizer_->param_groups().size()) {
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

void Trainer::save_checkpoint(const std::string& path) const {
    validate_configuration();
    if (path.empty()) {
        throw std::invalid_argument("Checkpoint path cannot be empty");
    }

    torch::serialize::OutputArchive archive;
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
        torch::serialize::OutputArchive scheduler_archive(archive.compilation_unit());
        write_scalar_state(scheduler_archive, scheduler_->state_dict());
        archive.write("scheduler", scheduler_archive);
    }

    if (grad_scaler_) {
        torch::serialize::OutputArchive scaler_archive(archive.compilation_unit());
        write_scalar_state(scaler_archive, grad_scaler_->state_dict());
        archive.write("grad_scaler", scaler_archive);
    }

    archive.write("trainer/global_step", torch::IValue(global_step_));
    archive.write("trainer/current_epoch", torch::IValue(static_cast<int64_t>(current_epoch_)));
    archive.write("trainer/next_epoch", torch::IValue(static_cast<int64_t>(next_epoch_)));
    archive.write("trainer/last_train_loss", torch::IValue(last_train_loss_));
    archive.write("trainer/last_val_loss", torch::IValue(last_val_loss_));
    archive.write("trainer/has_validation_loss", torch::IValue(has_validation_loss_));
    archive.save_to(path);
}

void Trainer::load_checkpoint(const std::string& path) {
    validate_configuration();
    if (path.empty()) {
        throw std::invalid_argument("Checkpoint path cannot be empty");
    }

    torch::serialize::InputArchive archive;
    archive.load_from(path, options_.device);

    torch::serialize::InputArchive model_archive;
    archive.read("model", model_archive);
    model_->load(model_archive);

    torch::serialize::InputArchive optimizer_archive;
    if (archive.try_read("optimizer", optimizer_archive)) {
        std::map<std::string, Tensor> state;
        for (const auto& key : optimizer_archive.keys()) {
            Tensor value;
            if (optimizer_archive.try_read(key, value)) {
                state.emplace(key, std::move(value));
            }
        }
        optimizer_->load_state_dict(state);
    }

    torch::serialize::InputArchive scheduler_archive;
    if (scheduler_ && archive.try_read("scheduler", scheduler_archive)) {
        scheduler_->load_state_dict(read_scalar_state(scheduler_archive));
    }

    torch::serialize::InputArchive scaler_archive;
    if (grad_scaler_ && archive.try_read("grad_scaler", scaler_archive)) {
        grad_scaler_->load_state_dict(read_scalar_state(scaler_archive));
    }

    torch::IValue value;
    if (archive.try_read("trainer/global_step", value) && value.isInt()) {
        global_step_ = value.toInt();
    }
    if (archive.try_read("trainer/current_epoch", value) && value.isInt()) {
        current_epoch_ = static_cast<int>(value.toInt());
    }
    if (archive.try_read("trainer/next_epoch", value) && value.isInt()) {
        next_epoch_ = static_cast<int>(value.toInt());
    } else if (current_epoch_ < std::numeric_limits<int>::max()) {
        // Older checkpoints stored the active epoch, before next_epoch was
        // recorded explicitly.
        next_epoch_ = current_epoch_ + 1;
    }
    if (archive.try_read("trainer/last_train_loss", value) && value.isDouble()) {
        last_train_loss_ = value.toDouble();
    }
    if (archive.try_read("trainer/last_val_loss", value) && value.isDouble()) {
        last_val_loss_ = value.toDouble();
    }
    if (archive.try_read("trainer/has_validation_loss", value) && value.isBool()) {
        has_validation_loss_ = value.toBool();
    }
}

} // namespace training
} // namespace jyppx
