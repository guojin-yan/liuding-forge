#include <jyppx/optim/adagrad.hpp>

#include <stdexcept>

namespace jyppx {
namespace optim {

namespace {

void validate_options(const AdagradOptions& options) {
    if (options.lr < 0.0) {
        throw std::invalid_argument("Adagrad learning rate must be non-negative");
    }
    if (options.lr_decay < 0.0) {
        throw std::invalid_argument("Adagrad lr_decay must be non-negative");
    }
    if (options.weight_decay < 0.0) {
        throw std::invalid_argument("Adagrad weight decay must be non-negative");
    }
    if (options.eps <= 0.0) {
        throw std::invalid_argument("Adagrad epsilon must be positive");
    }
}

} // namespace

Adagrad::Adagrad(const TensorList& params, const AdagradOptions& options)
    : Optimizer(params), options_(options), last_synced_options_(options) {
    validate_options(options_);
    auto& group = param_groups_.front();
    group.options["lr"] = options_.lr;
    group.options["lr_decay"] = options_.lr_decay;
    group.options["weight_decay"] = options_.weight_decay;
    group.options["eps"] = options_.eps;
}

void Adagrad::sync_options_to_default_group() {
    auto& group = param_groups_.front();
    if (options_.lr != last_synced_options_.lr) {
        group.options["lr"] = options_.lr;
    }
    if (options_.lr_decay != last_synced_options_.lr_decay) {
        group.options["lr_decay"] = options_.lr_decay;
    }
    if (options_.weight_decay != last_synced_options_.weight_decay) {
        group.options["weight_decay"] = options_.weight_decay;
    }
    if (options_.eps != last_synced_options_.eps) {
        group.options["eps"] = options_.eps;
    }
    last_synced_options_ = options_;
}

void Adagrad::step() {
    validate_options(options_);
    sync_options_to_default_group();
    torch::NoGradGuard no_grad;
    ++step_count_;

    for (const auto& group : param_groups_) {
        const auto option_of = [&group](const char* name, double fallback) {
            const auto it = group.options.find(name);
            return it == group.options.end() ? fallback : it->second;
        };
        const double lr = option_of("lr", options_.lr);
        const double lr_decay = option_of("lr_decay", options_.lr_decay);
        const double weight_decay = option_of("weight_decay", options_.weight_decay);
        const double eps = option_of("eps", options_.eps);

        for (const auto& parameter : group.params) {
            if (!parameter.defined() || !parameter.requires_grad()) {
                continue;
            }
            auto gradient = parameter.grad();
            if (!gradient.defined()) {
                continue;
            }
            if (weight_decay != 0.0) {
                gradient = gradient.add(parameter, weight_decay);
            }

            auto& parameter_state = state_[parameter.unsafeGetTensorImpl()];
            auto& sum = parameter_state["sum"];
            if (!sum.defined()) {
                sum = torch::zeros_like(parameter);
            }
            // Per-parameter step count drives the schedule decay.
            auto& parameter_step = parameter_state["step"];
            if (!parameter_step.defined()) {
                parameter_step = torch::zeros({}, torch::TensorOptions().dtype(torch::kInt64));
            }
            parameter_step.add_(1);
            const auto step = parameter_step.item<int64_t>();

            sum.addcmul_(gradient, gradient, 1.0);
            const double clr = lr / (1.0 + static_cast<double>(step - 1) * lr_decay);
            const auto std_dev = sum.sqrt().add_(eps);
            parameter.addcdiv_(gradient, std_dev, -clr);
        }
    }
}

} // namespace optim
} // namespace jyppx
