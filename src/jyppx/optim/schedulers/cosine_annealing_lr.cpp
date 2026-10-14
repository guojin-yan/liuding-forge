#include <jyppx/optim/schedulers/cosine_annealing_lr.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace jyppx {
namespace optim {
namespace schedulers {

CosineAnnealingLR::CosineAnnealingLR(
    std::shared_ptr<Optimizer> optimizer,
    int64_t T_max,
    double eta_min)
    : LRScheduler(std::move(optimizer)), T_max_(T_max), eta_min_(eta_min) {
    if (T_max_ <= 0) {
        throw std::invalid_argument("CosineAnnealingLR T_max must be positive");
    }
    if (!std::isfinite(eta_min_) || eta_min_ < 0.0) {
        throw std::invalid_argument("CosineAnnealingLR eta_min must be non-negative");
    }
    for (const auto base_lr : base_lrs_) {
        if (eta_min_ > base_lr) {
            throw std::invalid_argument("CosineAnnealingLR eta_min cannot exceed every base lr");
        }
    }
    configure_descriptor(
        "jyppx.scheduler.cosine_annealing_lr/v1", SchedulerStepMode::epoch,
        {{"T_max", descriptor_integer(T_max_)},
         {"eta_min", descriptor_number(eta_min_)}});
}

void CosineAnnealingLR::step() {
    ++step_count_;
    constexpr double pi = 3.141592653589793238462643383279502884;
    const double progress = static_cast<double>(step_count_) / static_cast<double>(T_max_);
    const double cosine = (1.0 + std::cos(pi * progress)) / 2.0;
    std::vector<double> lrs;
    lrs.reserve(base_lrs_.size());
    for (const auto base_lr : base_lrs_) {
        lrs.push_back(std::max(eta_min_, eta_min_ + (base_lr - eta_min_) * cosine));
    }
    set_last_lrs(lrs);
}

} // namespace schedulers
} // namespace optim
} // namespace jyppx
