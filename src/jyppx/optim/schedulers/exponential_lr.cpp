#include <jyppx/optim/schedulers/exponential_lr.hpp>

#include <cmath>
#include <stdexcept>

namespace jyppx {
namespace optim {
namespace schedulers {

ExponentialLR::ExponentialLR(std::shared_ptr<Optimizer> optimizer, double gamma)
    : LRScheduler(std::move(optimizer)), gamma_(gamma) {
    if (!std::isfinite(gamma_) || gamma_ <= 0.0) {
        throw std::invalid_argument("ExponentialLR gamma must be positive");
    }
    configure_descriptor(
        "jyppx.scheduler.exponential_lr/v1", SchedulerStepMode::epoch,
        {{"gamma", descriptor_number(gamma_)}});
}

void ExponentialLR::step() {
    ++step_count_;
    const double factor = std::pow(gamma_, step_count_);
    std::vector<double> lrs;
    lrs.reserve(base_lrs_.size());
    for (const auto base_lr : base_lrs_) {
        lrs.push_back(base_lr * factor);
    }
    set_last_lrs(lrs);
}

} // namespace schedulers
} // namespace optim
} // namespace jyppx
