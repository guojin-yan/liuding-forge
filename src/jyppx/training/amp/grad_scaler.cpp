#include <jyppx/training/amp/grad_scaler.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace jyppx {
namespace training {
namespace amp {

namespace {

void validate_options(const GradScalerOptions& options) {
    if (!std::isfinite(options.init_scale) || options.init_scale <= 0.0) {
        throw std::invalid_argument("GradScaler init_scale must be positive");
    }
    if (!std::isfinite(options.growth_factor) || options.growth_factor <= 1.0) {
        throw std::invalid_argument("GradScaler growth_factor must exceed 1.0");
    }
    if (!std::isfinite(options.backoff_factor) || options.backoff_factor <= 0.0 ||
        options.backoff_factor >= 1.0) {
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
      found_inf_(false) {
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
    if (!options_.enabled) {
        return;
    }

    if (optimizer_states_.empty()) {
        found_inf_ = false;
    }
    const auto existing = optimizer_states_.find(optimizer.get());
    if (existing != optimizer_states_.end()) {
        if (existing->second.stage == OptimizerStage::stepped) {
            throw std::logic_error(
                "GradScaler cannot unscale an optimizer after step() before update()");
        }
        return;
    }

    OptimizerState state;
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
                state.found_inf = true;
            }
        }
    }
    found_inf_ = found_inf_ || state.found_inf;
    optimizer_states_.emplace(optimizer.get(), state);
}

void GradScaler::step(const std::shared_ptr<optim::Optimizer>& optimizer) {
    if (!optimizer) {
        throw std::invalid_argument("GradScaler optimizer cannot be null");
    }
    if (!options_.enabled) {
        optimizer->step();
        return;
    }

    const auto existing = optimizer_states_.find(optimizer.get());
    if (existing != optimizer_states_.end() &&
        existing->second.stage == OptimizerStage::stepped) {
        throw std::logic_error(
            "GradScaler.step() for this optimizer must be followed by update()");
    }
    unscale(optimizer);
    auto& state = optimizer_states_.at(optimizer.get());
    if (!state.found_inf) {
        optimizer->step();
    }
    state.stage = OptimizerStage::stepped;
}

void GradScaler::update() {
    if (!options_.enabled) {
        return;
    }
    if (optimizer_states_.empty()) {
        throw std::logic_error("GradScaler.update() requires a preceding step()");
    }
    for (const auto& [optimizer, state] : optimizer_states_) {
        (void)optimizer;
        if (state.stage != OptimizerStage::stepped) {
            throw std::logic_error(
                "GradScaler.update() requires step() for every unscaled optimizer");
        }
    }
    if (found_inf_) {
        scale_ = std::max(scale_ * options_.backoff_factor,
                          std::numeric_limits<double>::min());
        growth_tracker_ = 0;
    } else {
        ++growth_tracker_;
        if (growth_tracker_ >= options_.growth_interval) {
            const double grown = scale_ * options_.growth_factor;
            if (!std::isfinite(grown)) {
                throw std::overflow_error("GradScaler scale overflow");
            }
            scale_ = grown;
            growth_tracker_ = 0;
        }
    }
    optimizer_states_.clear();
}

double GradScaler::get_scale() const {
    return scale_;
}

bool GradScaler::found_inf() const {
    return found_inf_;
}

std::map<std::string, double> GradScaler::state_dict() const {
    if (!optimizer_states_.empty()) {
        throw std::logic_error(
            "GradScaler state can only be saved after update() completes the step cycle");
    }
    return {
        {"format_version", 1.0},
        {"scale", scale_},
        {"growth_tracker", static_cast<double>(growth_tracker_)},
        {"enabled", options_.enabled ? 1.0 : 0.0},
        {"init_scale", options_.init_scale},
        {"growth_factor", options_.growth_factor},
        {"backoff_factor", options_.backoff_factor},
        {"growth_interval", static_cast<double>(options_.growth_interval)},
    };
}

void GradScaler::load_state_dict(const std::map<std::string, double>& state) {
    const auto scale_it = state.find("scale");
    const auto tracker_it = state.find("growth_tracker");
    if (scale_it == state.end() || tracker_it == state.end() ||
        !std::isfinite(scale_it->second) || scale_it->second <= 0.0 ||
        !std::isfinite(tracker_it->second) || tracker_it->second < 0.0 ||
        std::floor(tracker_it->second) != tracker_it->second ||
        tracker_it->second > static_cast<double>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("Invalid GradScaler state");
    }
    const auto version = state.find("format_version");
    const bool has_format = version != state.end();
    if (has_format && version->second != 1.0) {
        throw std::invalid_argument("Unsupported GradScaler state format version");
    }
    const auto require_config = [&](const char* key, double expected) {
        const auto it = state.find(key);
        if (has_format && it == state.end()) {
            throw std::invalid_argument(
                std::string("GradScaler checkpoint is missing option: ") + key);
        }
        if (it != state.end() && (!std::isfinite(it->second) || it->second != expected)) {
            throw std::invalid_argument(std::string("GradScaler checkpoint option mismatch: ") + key);
        }
    };
    require_config("enabled", options_.enabled ? 1.0 : 0.0);
    require_config("init_scale", options_.init_scale);
    require_config("growth_factor", options_.growth_factor);
    require_config("backoff_factor", options_.backoff_factor);
    require_config("growth_interval", static_cast<double>(options_.growth_interval));

    scale_ = scale_it->second;
    growth_tracker_ = static_cast<int>(tracker_it->second);
    found_inf_ = false;
    optimizer_states_.clear();
}

} // namespace amp
} // namespace training
} // namespace jyppx
