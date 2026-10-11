#include <jyppx/optim/schedulers/multi_step_lr.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace jyppx {
namespace optim {
namespace schedulers {

MultiStepLR::MultiStepLR(
    std::shared_ptr<Optimizer> optimizer,
    std::vector<int64_t> milestones,
    double gamma)
    : LRScheduler(std::move(optimizer)), milestones_(std::move(milestones)), gamma_(gamma) {
    if (milestones_.empty()) {
        throw std::invalid_argument("MultiStepLR requires at least one milestone");
    }
    if (!std::is_sorted(milestones_.begin(), milestones_.end())) {
        throw std::invalid_argument("MultiStepLR milestones must be sorted ascending");
    }
    if (milestones_.front() <= 0) {
        throw std::invalid_argument("MultiStepLR milestones must be positive");
    }
    if (gamma_ <= 0.0) {
        throw std::invalid_argument("MultiStepLR gamma must be positive");
    }
}

void MultiStepLR::step() {
    ++step_count_;
    const auto milestones_passed = static_cast<int64_t>(
        std::count_if(
            milestones_.begin(), milestones_.end(),
            [this](int64_t milestone) { return milestone <= step_count_; }));
    const double factor = std::pow(gamma_, static_cast<double>(milestones_passed));
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
