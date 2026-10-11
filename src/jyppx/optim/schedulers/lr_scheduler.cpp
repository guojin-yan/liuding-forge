#include <jyppx/optim/schedulers/lr_scheduler.hpp>

#include <stdexcept>

namespace jyppx {
namespace optim {
namespace schedulers {

LRScheduler::LRScheduler(std::shared_ptr<Optimizer> optimizer)
    : optimizer_(std::move(optimizer)), step_count_(0), base_lr_(0.0), last_lr_(0.0) {
    if (!optimizer_) {
        throw std::invalid_argument("Scheduler optimizer cannot be null");
    }
    const auto lrs = optimizer_->get_lrs();
    if (lrs.empty()) {
        throw std::invalid_argument("Scheduler optimizer must contain a parameter group");
    }
    base_lrs_ = lrs;
    last_lrs_ = lrs;
    base_lr_ = lrs.front();
    last_lr_ = lrs.front();
}

double LRScheduler::get_last_lr() const {
    return last_lr_;
}

void LRScheduler::set_last_lrs(const std::vector<double>& lrs) {
    if (lrs.size() != base_lrs_.size()) {
        throw std::invalid_argument("Scheduler learning-rate count must match parameter groups");
    }
    optimizer_->set_lrs(lrs);
    last_lrs_ = lrs;
    last_lr_ = lrs.front();
}

} // namespace schedulers
} // namespace optim
} // namespace jyppx
