#include <jyppx/optim/schedulers/reduce_lr_on_plateau.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace jyppx {
namespace optim {
namespace schedulers {

ReduceLROnPlateau::ReduceLROnPlateau(
    std::shared_ptr<Optimizer> optimizer,
    ReduceLROnPlateauOptions options)
    : LRScheduler(std::move(optimizer)),
      options_(options),
      best_(options_.mode == "max"
                ? -std::numeric_limits<double>::infinity()
                : std::numeric_limits<double>::infinity()),
      num_bad_epochs_(0) {
    if (options_.mode != "min" && options_.mode != "max") {
        throw std::invalid_argument("ReduceLROnPlateau mode must be 'min' or 'max'");
    }
    if (!std::isfinite(options_.factor) || options_.factor <= 0.0 || options_.factor >= 1.0) {
        throw std::invalid_argument("ReduceLROnPlateau factor must be in (0, 1)");
    }
    if (options_.patience < 0) {
        throw std::invalid_argument("ReduceLROnPlateau patience must be non-negative");
    }
    if (!std::isfinite(options_.threshold) || options_.threshold < 0.0) {
        throw std::invalid_argument("ReduceLROnPlateau threshold must be non-negative");
    }
    if (options_.threshold_mode != "rel" && options_.threshold_mode != "abs") {
        throw std::invalid_argument("ReduceLROnPlateau threshold_mode must be 'rel' or 'abs'");
    }
    if (!std::isfinite(options_.min_lr) || options_.min_lr < 0.0) {
        throw std::invalid_argument("ReduceLROnPlateau min_lr must be finite and non-negative");
    }
    const auto lrs = get_last_lrs();
    for (const double lr : lrs) {
        if (options_.min_lr > lr) {
            throw std::invalid_argument(
                "ReduceLROnPlateau min_lr cannot exceed the optimizer learning rate");
        }
    }
    configure_descriptor(
        "jyppx.scheduler.reduce_lr_on_plateau/v1", SchedulerStepMode::metric,
        {{"factor", descriptor_number(options_.factor)},
         {"min_lr", descriptor_number(options_.min_lr)},
         {"mode", options_.mode},
         {"patience", descriptor_integer(options_.patience)},
         {"threshold", descriptor_number(options_.threshold)},
         {"threshold_mode", options_.threshold_mode}});
}

void ReduceLROnPlateau::step() {
    throw std::logic_error(
        "ReduceLROnPlateau requires a metric; call step(metric)");
}

bool ReduceLROnPlateau::is_better(double current) const {
    if (options_.mode == "min") {
        return options_.threshold_mode == "rel"
            ? current < best_ * (1.0 - options_.threshold)
            : current < best_ - options_.threshold;
    }
    return options_.threshold_mode == "rel"
        ? current > best_ * (1.0 + options_.threshold)
        : current > best_ + options_.threshold;
}

void ReduceLROnPlateau::step(double metric) {
    if (!std::isfinite(metric)) {
        throw std::invalid_argument("ReduceLROnPlateau metric must be finite");
    }
    ++step_count_;
    if (is_better(metric)) {
        best_ = metric;
        num_bad_epochs_ = 0;
        return;
    }

    ++num_bad_epochs_;
    if (num_bad_epochs_ > options_.patience) {
        const auto current_lrs = optimizer_->get_lrs();
        std::vector<double> decayed;
        decayed.reserve(current_lrs.size());
        for (const double lr : current_lrs) {
            decayed.push_back(std::max(lr * options_.factor, options_.min_lr));
        }
        set_last_lrs(decayed);
        num_bad_epochs_ = 0;
    }
}

std::map<std::string, double> ReduceLROnPlateau::state_dict() const {
    auto state = LRScheduler::state_dict();
    state["best"] = best_;
    state["num_bad_epochs"] = static_cast<double>(num_bad_epochs_);
    return state;
}

void ReduceLROnPlateau::load_state_dict(const std::map<std::string, double>& state) {
    const bool has_format = state.count("format_version") != 0;
    double restored_best = best_;
    int restored_bad_epochs = num_bad_epochs_;
    const auto best_it = state.find("best");
    if (best_it != state.end()) {
        if (std::isnan(best_it->second)) {
            throw std::invalid_argument("ReduceLROnPlateau best metric cannot be NaN");
        }
        restored_best = best_it->second;
    } else if (has_format) {
        throw std::invalid_argument("ReduceLROnPlateau checkpoint is missing best");
    }
    const auto bad_it = state.find("num_bad_epochs");
    if (bad_it != state.end()) {
        if (!std::isfinite(bad_it->second) || bad_it->second < 0.0 ||
            std::floor(bad_it->second) != bad_it->second) {
            throw std::invalid_argument(
                "ReduceLROnPlateau num_bad_epochs must be a non-negative integer");
        }
        if (bad_it->second > static_cast<double>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("ReduceLROnPlateau num_bad_epochs is too large");
        }
        restored_bad_epochs = static_cast<int>(bad_it->second);
    } else if (has_format) {
        throw std::invalid_argument(
            "ReduceLROnPlateau checkpoint is missing num_bad_epochs");
    }
    LRScheduler::load_state_dict(state);
    best_ = restored_best;
    num_bad_epochs_ = restored_bad_epochs;
}

} // namespace schedulers
} // namespace optim
} // namespace jyppx
