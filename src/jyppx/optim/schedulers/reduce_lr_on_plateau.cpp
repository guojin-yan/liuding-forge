#include <jyppx/optim/schedulers/reduce_lr_on_plateau.hpp>

#include <algorithm>
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
    if (options_.factor <= 0.0 || options_.factor >= 1.0) {
        throw std::invalid_argument("ReduceLROnPlateau factor must be in (0, 1)");
    }
    if (options_.patience < 0) {
        throw std::invalid_argument("ReduceLROnPlateau patience must be non-negative");
    }
    if (options_.threshold < 0.0) {
        throw std::invalid_argument("ReduceLROnPlateau threshold must be non-negative");
    }
    if (options_.threshold_mode != "rel" && options_.threshold_mode != "abs") {
        throw std::invalid_argument("ReduceLROnPlateau threshold_mode must be 'rel' or 'abs'");
    }
    const auto lrs = get_last_lrs();
    for (const double lr : lrs) {
        if (options_.min_lr > lr) {
            throw std::invalid_argument(
                "ReduceLROnPlateau min_lr cannot exceed the optimizer learning rate");
        }
    }
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

} // namespace schedulers
} // namespace optim
} // namespace jyppx
