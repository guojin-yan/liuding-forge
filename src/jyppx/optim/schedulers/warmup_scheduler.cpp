#include <jyppx/optim/schedulers/warmup_scheduler.hpp>

#include <cmath>
#include <stdexcept>

namespace jyppx::optim::schedulers {

WarmupScheduler::WarmupScheduler(
    std::shared_ptr<Optimizer> optimizer, int64_t warmup_iters,
    double start_factor, std::shared_ptr<LRScheduler> after_scheduler)
    : LRScheduler(std::move(optimizer)),
      warmup_iters_(warmup_iters),
      start_factor_(start_factor),
      after_scheduler_(std::move(after_scheduler)) {
    if (warmup_iters_ <= 0 || !std::isfinite(start_factor_) ||
        start_factor_ <= 0.0 || start_factor_ > 1.0) {
        throw std::invalid_argument(
            "WarmupScheduler requires warmup_iters > 0 and start_factor in (0, 1]");
    }
    std::vector<SchedulerDescriptor> children;
    bool checkpoint_safe = true;
    if (after_scheduler_) {
        children.push_back(after_scheduler_->descriptor());
        checkpoint_safe = after_scheduler_->descriptor().checkpoint_safe;
    }
    configure_descriptor(
        "jyppx.scheduler.warmup/v1", SchedulerStepMode::epoch,
        {{"start_factor", descriptor_number(start_factor_)},
         {"warmup_iters", descriptor_integer(warmup_iters_)}},
        std::move(children), checkpoint_safe);
    std::vector<double> initial;
    initial.reserve(base_lrs_.size());
    for (const double base : base_lrs_) initial.push_back(base * start_factor_);
    set_last_lrs(initial);
}

void WarmupScheduler::step() {
    ++step_count_;
    if (step_count_ <= warmup_iters_) {
        const double progress = static_cast<double>(step_count_) /
                                static_cast<double>(warmup_iters_);
        const double factor = start_factor_ + (1.0 - start_factor_) * progress;
        std::vector<double> lrs;
        lrs.reserve(base_lrs_.size());
        for (const double base : base_lrs_) lrs.push_back(base * factor);
        set_last_lrs(lrs);
        return;
    }
    if (after_scheduler_) {
        after_scheduler_->step();
        set_last_lrs(after_scheduler_->get_last_lrs());
    } else {
        set_last_lrs(base_lrs_);
    }
}

std::map<std::string, double> WarmupScheduler::state_dict() const {
    auto state = LRScheduler::state_dict();
    if (after_scheduler_) {
        for (const auto& [name, value] : after_scheduler_->state_dict()) {
            state["after_" + name] = value;
        }
    }
    return state;
}

void WarmupScheduler::load_state_dict(
    const std::map<std::string, double>& state) {
    validate_descriptor(state);
    if (!after_scheduler_) {
        LRScheduler::load_state_dict(state);
        return;
    }

    std::map<std::string, double> after;
    for (const auto& [name, value] : state) {
        if (name.rfind("after_", 0) == 0) {
            after[name.substr(6)] = value;
        }
    }
    if (after.empty()) {
        throw std::invalid_argument("WarmupScheduler child state is missing");
    }
    after_scheduler_->validate_descriptor(after);
    const auto parent_snapshot = LRScheduler::state_dict();
    const auto child_snapshot = after_scheduler_->state_dict();
    try {
        after_scheduler_->load_state_dict(after);
        LRScheduler::load_state_dict(state);
    } catch (...) {
        after_scheduler_->load_state_dict(child_snapshot);
        LRScheduler::load_state_dict(parent_snapshot);
        throw;
    }
}

} // namespace jyppx::optim::schedulers
