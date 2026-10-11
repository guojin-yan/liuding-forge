#include <jyppx/optim/sgd.hpp>

#include <stdexcept>

namespace jyppx {
namespace optim {

namespace {

void validate_options(const SGDOptions& options) {
    if (options.lr < 0.0) {
        throw std::invalid_argument("SGD learning rate must be non-negative");
    }
    if (options.momentum < 0.0) {
        throw std::invalid_argument("SGD momentum must be non-negative");
    }
    if (options.dampening < 0.0 || options.dampening > 1.0) {
        throw std::invalid_argument("SGD dampening must be in [0, 1]");
    }
    if (options.weight_decay < 0.0) {
        throw std::invalid_argument("SGD weight decay must be non-negative");
    }
    if (options.nesterov && (options.momentum <= 0.0 || options.dampening != 0.0)) {
        throw std::invalid_argument(
            "SGD Nesterov momentum requires momentum > 0 and dampening == 0");
    }
}

} // namespace

SGD::SGD(const TensorList& params, const SGDOptions& options)
    : Optimizer(params), options_(options), last_synced_options_(options) {
    validate_options(options_);
    param_groups_.front().options["lr"] = options_.lr;
    param_groups_.front().options["momentum"] = options_.momentum;
    param_groups_.front().options["dampening"] = options_.dampening;
    param_groups_.front().options["weight_decay"] = options_.weight_decay;
    param_groups_.front().options["nesterov"] = options_.nesterov ? 1.0 : 0.0;
}

void SGD::sync_options_to_default_group() {
    auto& group = param_groups_.front();
    if (options_.lr != last_synced_options_.lr) {
        group.options["lr"] = options_.lr;
    }
    if (options_.momentum != last_synced_options_.momentum) {
        group.options["momentum"] = options_.momentum;
    }
    if (options_.dampening != last_synced_options_.dampening) {
        group.options["dampening"] = options_.dampening;
    }
    if (options_.weight_decay != last_synced_options_.weight_decay) {
        group.options["weight_decay"] = options_.weight_decay;
    }
    if (options_.nesterov != last_synced_options_.nesterov) {
        group.options["nesterov"] = options_.nesterov ? 1.0 : 0.0;
    }
    last_synced_options_ = options_;
}

void SGD::step() {
    validate_options(options_);
    sync_options_to_default_group();
    torch::NoGradGuard no_grad;
    ++step_count_;

    for (const auto& group : param_groups_) {
        const auto lr_it = group.options.find("lr");
        const double lr = lr_it == group.options.end() ? options_.lr : lr_it->second;
        const auto momentum_it = group.options.find("momentum");
        const auto dampening_it = group.options.find("dampening");
        const auto decay_it = group.options.find("weight_decay");
        const auto nesterov_it = group.options.find("nesterov");
        const double momentum = momentum_it == group.options.end() ? options_.momentum : momentum_it->second;
        const double dampening = dampening_it == group.options.end() ? options_.dampening : dampening_it->second;
        const double weight_decay = decay_it == group.options.end() ? options_.weight_decay : decay_it->second;
        const bool nesterov = nesterov_it == group.options.end() ? options_.nesterov : nesterov_it->second != 0.0;
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

            if (momentum != 0.0) {
                auto& parameter_state = state_[parameter.unsafeGetTensorImpl()];
                auto state_it = parameter_state.find("momentum_buffer");
                if (state_it == parameter_state.end()) {
                    state_it = parameter_state.emplace("momentum_buffer", gradient.clone()).first;
                } else {
                    state_it->second.mul_(momentum).add_(gradient, 1.0 - dampening);
                }

                if (nesterov) {
                    gradient = gradient.add(state_it->second, momentum);
                } else {
                    gradient = state_it->second;
                }
            }

            parameter.add_(gradient, -lr);
        }
    }
}

} // namespace optim
} // namespace jyppx
