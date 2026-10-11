#include <jyppx/training/amp/grad_scaler.hpp>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace jyppx {
namespace training {
namespace amp {

namespace {

void validate_options(const GradScalerOptions& options) {
    if (options.init_scale <= 0.0) {
        throw std::invalid_argument("GradScaler init_scale must be positive");
    }
    if (options.growth_factor <= 1.0) {
        throw std::invalid_argument("GradScaler growth_factor must exceed 1.0");
    }
    if (options.backoff_factor <= 0.0 || options.backoff_factor >= 1.0) {
        throw std::invalid_argument("GradScaler backoff_factor must be in (0, 1)");
    }
    if (options.growth_interval <= 0) {
        throw std::invalid_argument("GradScaler growth_interval must be positive");
    }
}

} // namespace

GradScaler::GradScaler(GradScalerOptions options)
    : options_(options),
      scale_(options_.enabled ? options_.init_scale : 1.0),
      growth_tracker_(0),
      found_inf_(false),
      unscaled_(false) {
    validate_options(options_);
}

Tensor GradScaler::scale(const Tensor& loss) const {
    if (!options_.enabled) {
        return loss;
    }
    return loss * scale_;
}

void GradScaler::unscale(const std::shared_ptr<optim::Optimizer>& optimizer) {
    if (!optimizer) {
        throw std::invalid_argument("GradScaler optimizer cannot be null");
    }
    if (!options_.enabled || unscaled_) {
        return;
    }

    found_inf_ = false;
    const torch::NoGradGuard no_grad;
    for (const auto& group : optimizer->param_groups()) {
        for (const auto& parameter : group.params) {
            if (!parameter.defined() || !parameter.grad().defined()) {
                continue;
            }
            auto& gradient = parameter.mutable_grad();
            gradient.div_(scale_);
            if (torch::isinf(gradient).any().item<bool>() ||
                torch::isnan(gradient).any().item<bool>()) {
                found_inf_ = true;
            }
        }
    }
    unscaled_ = true;
}

void GradScaler::step(const std::shared_ptr<optim::Optimizer>& optimizer) {
    if (!optimizer) {
        throw std::invalid_argument("GradScaler optimizer cannot be null");
    }
    if (!options_.enabled) {
        optimizer->step();
        return;
    }

    unscale(optimizer);
    if (found_inf_) {
        // Skip the update and shrink the scale so the next attempt is smaller.
        scale_ *= options_.backoff_factor;
        growth_tracker_ = 0;
        return;
    }
    optimizer->step();
}

void GradScaler::update() {
    if (!options_.enabled) {
        return;
    }
    if (!found_inf_) {
        ++growth_tracker_;
        if (growth_tracker_ >= options_.growth_interval) {
            scale_ *= options_.growth_factor;
            growth_tracker_ = 0;
        }
    }
    // Reset per-iteration bookkeeping: gradients are consumed (or discarded)
    // once update() runs, so the next scale()/step() cycle starts fresh.
    unscaled_ = false;
}

double GradScaler::get_scale() const {
    return scale_;
}

bool GradScaler::found_inf() const {
    return found_inf_;
}

std::map<std::string, double> GradScaler::state_dict() const {
    return {
        {"scale", scale_},
        {"growth_tracker", static_cast<double>(growth_tracker_)},
    };
}

void GradScaler::load_state_dict(const std::map<std::string, double>& state) {
    const auto scale_it = state.find("scale");
    if (scale_it != state.end()) {
        scale_ = scale_it->second;
    }
    const auto tracker_it = state.find("growth_tracker");
    if (tracker_it != state.end()) {
        growth_tracker_ = static_cast<int>(tracker_it->second);
    }
}

} // namespace amp
} // namespace training
} // namespace jyppx
