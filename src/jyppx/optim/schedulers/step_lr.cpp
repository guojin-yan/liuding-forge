#include <jyppx/optim/schedulers/step_lr.hpp>

#include <cmath>
#include <stdexcept>

namespace jyppx {
namespace optim {
namespace schedulers {

StepLR::StepLR(std::shared_ptr<Optimizer> optimizer, int64_t step_size, double gamma)
    : LRScheduler(std::move(optimizer)), step_size_(step_size), gamma_(gamma) {
    if (step_size_ <= 0) {
        throw std::invalid_argument("StepLR step_size must be positive");
    }
    if (!std::isfinite(gamma_) || gamma_ <= 0.0) {
        throw std::invalid_argument("StepLR gamma must be positive");
    }
    configure_descriptor(
        "jyppx.scheduler.step_lr/v1", SchedulerStepMode::epoch,
        {{"gamma", descriptor_number(gamma_)},
         {"step_size", descriptor_integer(step_size_)}});
}

void StepLR::step() {
    ++step_count_;
    const double factor = std::pow(gamma_, step_count_ / step_size_);
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
