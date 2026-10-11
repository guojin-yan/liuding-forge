#include <jyppx/optim/rmsprop.hpp>

#include <stdexcept>

namespace jyppx {
namespace optim {

namespace {

void validate_options(const RMSpropOptions& options) {
    if (options.lr < 0.0) {
        throw std::invalid_argument("RMSprop learning rate must be non-negative");
    }
    if (options.alpha < 0.0 || options.alpha > 1.0) {
        throw std::invalid_argument("RMSprop alpha must be in [0, 1]");
    }
    if (options.eps <= 0.0) {
        throw std::invalid_argument("RMSprop epsilon must be positive");
    }
    if (options.weight_decay < 0.0) {
        throw std::invalid_argument("RMSprop weight decay must be non-negative");
    }
    if (options.momentum < 0.0) {
        throw std::invalid_argument("RMSprop momentum must be non-negative");
    }
}

} // namespace

RMSprop::RMSprop(const TensorList& params, const RMSpropOptions& options)
    : Optimizer(params), options_(options), last_synced_options_(options) {
    validate_options(options_);
    auto& group = param_groups_.front();
    group.options["lr"] = options_.lr;
    group.options["alpha"] = options_.alpha;
    group.options["eps"] = options_.eps;
    group.options["weight_decay"] = options_.weight_decay;
    group.options["momentum"] = options_.momentum;
    group.options["centered"] = options_.centered ? 1.0 : 0.0;
}

void RMSprop::sync_options_to_default_group() {
    auto& group = param_groups_.front();
    if (options_.lr != last_synced_options_.lr) {
        group.options["lr"] = options_.lr;
    }
    if (options_.alpha != last_synced_options_.alpha) {
        group.options["alpha"] = options_.alpha;
    }
    if (options_.eps != last_synced_options_.eps) {
        group.options["eps"] = options_.eps;
    }
    if (options_.weight_decay != last_synced_options_.weight_decay) {
        group.options["weight_decay"] = options_.weight_decay;
    }
    if (options_.momentum != last_synced_options_.momentum) {
        group.options["momentum"] = options_.momentum;
    }
    if (options_.centered != last_synced_options_.centered) {
        group.options["centered"] = options_.centered ? 1.0 : 0.0;
    }
    last_synced_options_ = options_;
}

void RMSprop::step() {
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
        const double alpha = option_of("alpha", options_.alpha);
        const double eps = option_of("eps", options_.eps);
        const double weight_decay = option_of("weight_decay", options_.weight_decay);
        const double momentum = option_of("momentum", options_.momentum);
        const bool centered = option_of("centered", options_.centered ? 1.0 : 0.0) != 0.0;

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
            auto& square_avg = parameter_state["square_avg"];
            if (!square_avg.defined()) {
                square_avg = torch::zeros_like(parameter);
            }
            square_avg.mul_(alpha).addcmul_(gradient, gradient, 1.0 - alpha);

            torch::Tensor average;
            if (centered) {
                auto& grad_avg = parameter_state["grad_avg"];
                if (!grad_avg.defined()) {
                    grad_avg = torch::zeros_like(parameter);
                }
                grad_avg.mul_(alpha).add_(gradient, 1.0 - alpha);
                average = (square_avg - grad_avg.pow(2)).sqrt_().add_(eps);
            } else {
                average = square_avg.sqrt().add_(eps);
            }

            if (momentum != 0.0) {
                // torch::optim::RMSprop applies momentum to the normalized
                // gradient (grad / avg) and steps by -lr * buffer. Note this
                // differs from torch.optim.rmsprop in Python, which keeps the
                // raw gradient in the buffer and divides by avg on update;
                // the two coincide only on the first step.
                auto& momentum_buffer = parameter_state["momentum_buffer"];
                const torch::Tensor normalized = gradient / average;
                if (!momentum_buffer.defined()) {
                    momentum_buffer = normalized.clone();
                } else {
                    momentum_buffer.mul_(momentum).add_(normalized);
                }
                parameter.add_(momentum_buffer, -lr);
            } else {
                parameter.addcdiv_(gradient, average, -lr);
            }
        }
    }
}

} // namespace optim
} // namespace jyppx
