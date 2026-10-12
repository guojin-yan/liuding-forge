#include <jyppx/optim/schedulers/lr_scheduler.hpp>

#include <cmath>
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

std::map<std::string, double> LRScheduler::state_dict() const {
    std::map<std::string, double> state{{"step_count", static_cast<double>(step_count_)}};
    for (size_t i = 0; i < last_lrs_.size(); ++i) {
        state["last_lr_" + std::to_string(i)] = last_lrs_[i];
    }
    return state;
}

void LRScheduler::load_state_dict(const std::map<std::string, double>& state) {
    const auto step_it = state.find("step_count");
    if (step_it != state.end()) {
        if (!std::isfinite(step_it->second) || step_it->second < 0.0 ||
            std::floor(step_it->second) != step_it->second) {
            throw std::invalid_argument("Scheduler step_count must be a non-negative integer");
        }
        step_count_ = static_cast<int64_t>(step_it->second);
    }

    std::vector<double> lrs = optimizer_->get_lrs();
    bool has_saved_lrs = false;
    for (size_t i = 0; i < lrs.size(); ++i) {
        const auto lr_it = state.find("last_lr_" + std::to_string(i));
        if (lr_it == state.end()) {
            continue;
        }
        if (!std::isfinite(lr_it->second) || lr_it->second < 0.0) {
            throw std::invalid_argument(
                "Scheduler learning rates must be finite and non-negative");
        }
        lrs[i] = lr_it->second;
        has_saved_lrs = true;
    }
    if (has_saved_lrs) {
        set_last_lrs(lrs);
    }
}

} // namespace schedulers
} // namespace optim
} // namespace jyppx
