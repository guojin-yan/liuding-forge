#include <jyppx/optim/nadam.hpp>

#include <cmath>
#include <stdexcept>

namespace jyppx {
namespace optim {

namespace {

void validate_options(const NAdamOptions& options) {
    if (options.lr < 0.0) {
        throw std::invalid_argument("NAdam learning rate must be non-negative");
    }
    if (options.beta1 < 0.0 || options.beta1 >= 1.0 || options.beta2 < 0.0 ||
        options.beta2 >= 1.0) {
        throw std::invalid_argument("NAdam betas must be in [0, 1)");
    }
    if (options.eps <= 0.0) {
        throw std::invalid_argument("NAdam epsilon must be positive");
    }
    if (options.weight_decay < 0.0) {
        throw std::invalid_argument("NAdam weight decay must be non-negative");
    }
    if (options.momentum_decay != 0.0) {
        throw std::invalid_argument(
            "NAdam momentum_decay > 0 is not supported in this build; keep 0.0");
    }
}

} // namespace

NAdam::NAdam(const TensorList& params, const NAdamOptions& options)
    : Optimizer(params), options_(options), last_synced_options_(options) {
    validate_options(options_);
    auto& group = param_groups_.front();
    group.options["lr"] = options_.lr;
    group.options["beta1"] = options_.beta1;
    group.options["beta2"] = options_.beta2;
    group.options["eps"] = options_.eps;
    group.options["weight_decay"] = options_.weight_decay;
    group.options["momentum_decay"] = options_.momentum_decay;
}

void NAdam::sync_options_to_default_group() {
    auto& group = param_groups_.front();
    if (options_.lr != last_synced_options_.lr) {
        group.options["lr"] = options_.lr;
    }
    if (options_.beta1 != last_synced_options_.beta1) {
        group.options["beta1"] = options_.beta1;
    }
    if (options_.beta2 != last_synced_options_.beta2) {
        group.options["beta2"] = options_.beta2;
    }
    if (options_.eps != last_synced_options_.eps) {
        group.options["eps"] = options_.eps;
    }
    if (options_.weight_decay != last_synced_options_.weight_decay) {
        group.options["weight_decay"] = options_.weight_decay;
    }
    last_synced_options_ = options_;
}

void NAdam::step() {
    validate_options(options_);
    sync_options_to_default_group();
    torch::NoGradGuard no_grad;
    ++step_count_;
    const auto step = static_cast<double>(step_count_);

    // Momentum schedule (PyTorch NAdam, schedule_delay = 1).
    const double beta1 = options_.beta1;
    const double mu_t = beta1 * (1.0 - 0.5 * std::pow(0.96, step));
    mu_product_ *= mu_t;
    const double bias_correction2 = 1.0 - std::pow(options_.beta2, step);
    const double bias_correction_sqrt = std::sqrt(bias_correction2);
    // Per-step coefficients (torch.optim.nadam, _single_tensor_nadam).
    const double grad_coef =
        -options_.lr * (1.0 - mu_t) / (bias_correction_sqrt * (1.0 - mu_product_));
    const double exp_avg_coef =
        -options_.lr * mu_t / (bias_correction_sqrt * (1.0 - mu_product_));

    for (const auto& group : param_groups_) {
        const auto option_of = [&group](const char* name, double fallback) {
            const auto it = group.options.find(name);
            return it == group.options.end() ? fallback : it->second;
        };
        const double lr = option_of("lr", options_.lr);
        const double lr_scale = lr / options_.lr; // scheduler support
        const double beta2 = option_of("beta2", options_.beta2);
        const double eps = option_of("eps", options_.eps);
        const double weight_decay = option_of("weight_decay", options_.weight_decay);
        (void)beta2; // options carried per group; schedule uses member values

        const double group_grad_coef = grad_coef * lr_scale;
        const double group_exp_avg_coef = exp_avg_coef * lr_scale;
        const double group_bc2 = 1.0 - std::pow(beta2, step);

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
            auto& exp_avg = parameter_state["exp_avg"];
            auto& exp_avg_sq = parameter_state["exp_avg_sq"];
            if (!exp_avg.defined()) {
                exp_avg = torch::zeros_like(parameter);
                exp_avg_sq = torch::zeros_like(parameter);
            }
            exp_avg.mul_(beta1).add_(gradient, 1.0 - beta1);
            exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, 1.0 - beta2);

            auto denom = (exp_avg_sq / group_bc2).sqrt_().add_(eps);
            parameter.addcdiv_(gradient, denom, group_grad_coef);
            parameter.addcdiv_(exp_avg, denom, group_exp_avg_coef);
        }
    }
}

} // namespace optim
} // namespace jyppx
