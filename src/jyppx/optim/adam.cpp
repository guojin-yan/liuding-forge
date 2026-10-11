#include <jyppx/optim/adam.hpp>

#include <cmath>
#include <stdexcept>

namespace jyppx {
namespace optim {

namespace {

void validate_options(const AdamOptions& options) {
    if (options.lr < 0.0) {
        throw std::invalid_argument("Adam learning rate must be non-negative");
    }
    if (options.beta1 < 0.0 || options.beta1 >= 1.0 || options.beta2 < 0.0 ||
        options.beta2 >= 1.0) {
        throw std::invalid_argument("Adam betas must be in [0, 1)");
    }
    if (options.eps <= 0.0) {
        throw std::invalid_argument("Adam epsilon must be positive");
    }
    if (options.weight_decay < 0.0) {
        throw std::invalid_argument("Adam weight decay must be non-negative");
    }
}

} // namespace

Adam::Adam(const TensorList& params, const AdamOptions& options)
    : Optimizer(params), options_(options), last_synced_options_(options) {
    validate_options(options_);
    auto& group = param_groups_.front();
    group.options["lr"] = options_.lr;
    group.options["weight_decay"] = options_.weight_decay;
    group.options["beta1"] = options_.beta1;
    group.options["beta2"] = options_.beta2;
    group.options["eps"] = options_.eps;
    group.options["amsgrad"] = options_.amsgrad ? 1.0 : 0.0;
}

void Adam::sync_options_to_default_group() {
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
    if (options_.amsgrad != last_synced_options_.amsgrad) {
        group.options["amsgrad"] = options_.amsgrad ? 1.0 : 0.0;
    }
    last_synced_options_ = options_;
}

void Adam::step() {
    validate_options(options_);
    sync_options_to_default_group();
    torch::NoGradGuard no_grad;
    ++step_count_;

    for (const auto& group : param_groups_) {
        const auto lr_it = group.options.find("lr");
        const auto decay_it = group.options.find("weight_decay");
        const double lr = lr_it == group.options.end() ? options_.lr : lr_it->second;
        const double weight_decay =
            decay_it == group.options.end() ? options_.weight_decay : decay_it->second;
        const double beta1 = group.options.count("beta1") ? group.options.at("beta1") : options_.beta1;
        const double beta2 = group.options.count("beta2") ? group.options.at("beta2") : options_.beta2;
        const double eps = group.options.count("eps") ? group.options.at("eps") : options_.eps;
        const bool amsgrad = group.options.count("amsgrad") ? group.options.at("amsgrad") != 0.0 : options_.amsgrad;

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
            auto& parameter_step = parameter_state["step"];
            if (!parameter_step.defined()) {
                parameter_step = torch::zeros({}, torch::TensorOptions().dtype(torch::kInt64));
            }
            parameter_step.add_(1);
            const auto step = parameter_step.item<int64_t>();

            exp_avg.mul_(beta1).add_(gradient, 1.0 - beta1);
            exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, 1.0 - beta2);

            const double bias_correction1 = 1.0 - std::pow(beta1, step);
            const double bias_correction2 = 1.0 - std::pow(beta2, step);
            const double bias_correction2_sqrt = std::sqrt(bias_correction2);
            torch::Tensor denominator;
            if (amsgrad) {
                auto& max_exp_avg_sq = parameter_state["max_exp_avg_sq"];
                if (!max_exp_avg_sq.defined()) {
                    max_exp_avg_sq = torch::zeros_like(parameter);
                }
                max_exp_avg_sq.copy_(torch::maximum(max_exp_avg_sq, exp_avg_sq));
                // PyTorch scales sqrt(v) by 1/sqrt(bias_correction2) before
                // adding eps, so eps is effectively scaled as well.
                denominator = (max_exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps);
            } else {
                denominator = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps);
            }

            const double step_size = lr / bias_correction1;
            parameter.addcdiv_(exp_avg, denominator, -step_size);
        }
    }
}

} // namespace optim
} // namespace jyppx
