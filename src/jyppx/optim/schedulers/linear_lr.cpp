#include <jyppx/optim/schedulers/linear_lr.hpp>

#include <cmath>
#include <stdexcept>

namespace jyppx::optim::schedulers {
LinearLR::LinearLR(std::shared_ptr<Optimizer> optimizer, double start_factor, double end_factor, int64_t total_iters)
    : LRScheduler(std::move(optimizer)), start_factor_(start_factor), end_factor_(end_factor), total_iters_(total_iters) {
    if (!std::isfinite(start_factor_) || start_factor_ <= 0.0 || start_factor_ > 1.0) throw std::invalid_argument("LinearLR start_factor must be in (0, 1]");
    if (!std::isfinite(end_factor_) || end_factor_ <= 0.0 || end_factor_ > 1.0) throw std::invalid_argument("LinearLR end_factor must be in (0, 1]");
    if (total_iters_ < 0) throw std::invalid_argument("LinearLR total_iters must be non-negative");
    configure_descriptor(
        "jyppx.scheduler.linear_lr/v1", SchedulerStepMode::epoch,
        {{"end_factor", descriptor_number(end_factor_)},
         {"start_factor", descriptor_number(start_factor_)},
         {"total_iters", descriptor_integer(total_iters_)}});
    std::vector<double> initial; for (double base : base_lrs_) initial.push_back(base * start_factor_); set_last_lrs(initial);
}
void LinearLR::step() {
    ++step_count_;
    const double pct = total_iters_ == 0 ? 1.0 : std::min(1.0, static_cast<double>(step_count_) / static_cast<double>(total_iters_));
    const double factor = start_factor_ + (end_factor_ - start_factor_) * pct;
    std::vector<double> lrs; for (double base : base_lrs_) lrs.push_back(base * factor); set_last_lrs(lrs);
}
} // namespace jyppx::optim::schedulers
