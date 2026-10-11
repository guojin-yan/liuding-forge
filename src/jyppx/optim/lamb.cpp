#include <jyppx/optim/lamb.hpp>

#include <cmath>
#include <stdexcept>

namespace jyppx {
namespace optim {

namespace {

void validate_options(const LambOptions& options) {
    if (options.lr < 0.0) {
        throw std::invalid_argument("Lamb learning rate must be non-negative");
    }
    if (options.beta1 < 0.0 || options.beta1 >= 1.0 || options.beta2 < 0.0 ||
        options.beta2 >= 1.0) {
        throw std::invalid_argument("Lamb betas must be in [0, 1)");
    }
    if (options.eps < 0.0) {
        throw std::invalid_argument("Lamb epsilon must be non-negative");
    }
    if (options.weight_decay < 0.0) {
        throw std::invalid_argument("Lamb weight decay must be non-negative");
    }
}

} // namespace

Lamb::Lamb(const TensorList& params, const LambOptions& options)
    : Optimizer(params), options_(options), last_synced_options_(options) {
    validate_options(options_);
    auto& group = param_groups_.front();
    group.options["lr"] = options_.lr;
    group.options["beta1"] = options_.beta1;
    group.options["beta2"] = options_.beta2;
    group.options["eps"] = options_.eps;
    group.options["weight_decay"] = options_.weight_decay;
}

void Lamb::sync_options_to_default_group() {
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

void Lamb::step() {
    validate_options(options_);
    sync_options_to_default_group();
    torch::NoGradGuard no_grad;
    ++step_count_;
    const auto step = static_cast<double>(step_count_);
    const double bias_correction1 = 1.0 - std::pow(options_.beta1, step);
    const double bias_correction2 = 1.0 - std::pow(options_.beta2, step);

    for (const auto& group : param_groups_) {
        const auto option_of = [&group](const char* name, double fallback) {
            const auto it = group.options.find(name);
            return it == group.options.end() ? fallback : it->second;
        };
        const double lr = option_of("lr", options_.lr);
        const double beta1 = option_of("beta1", options_.beta1);
        const double beta2 = option_of("beta2", options_.beta2);
        const double eps = option_of("eps", options_.eps);
        const double weight_decay = option_of("weight_decay", options_.weight_decay);
        const double group_bc1 = 1.0 - std::pow(beta1, step);
        const double group_bc2 = 1.0 - std::pow(beta2, step);
        (void)bias_correction1;
        (void)bias_correction2;

        for (const auto& parameter : group.params) {
            if (!parameter.defined() || !parameter.requires_grad()) {
                continue;
            }
            auto gradient = parameter.grad();
            if (!gradient.defined()) {
                continue;
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

            auto adam_step = (exp_avg / group_bc1) /
                             ((exp_avg_sq / group_bc2).sqrt().add_(eps));
            if (weight_decay != 0.0) {
                adam_step = adam_step.add(parameter, weight_decay);
            }

            // Trust ratio with the reference clamp on the weight norm.
            const double weight_norm =
                parameter.norm().item<double>();
            const double clamped_weight_norm = std::min(weight_norm, 10.0);
            const double adam_norm = adam_step.norm().item<double>();
            const double trust_ratio =
                (clamped_weight_norm == 0.0 || adam_norm == 0.0)
                    ? 1.0
                    : clamped_weight_norm / adam_norm;

            parameter.add_(adam_step, -lr * trust_ratio);
        }
    }
}

} // namespace optim
} // namespace jyppx
