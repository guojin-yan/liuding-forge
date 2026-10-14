#include <jyppx/optim/schedulers/composite_schedulers.hpp>

#include <stdexcept>

namespace jyppx::optim::schedulers {
namespace {

std::map<std::string, double> child_state(
    const std::map<std::string, double>& state,
    size_t index) {
    std::map<std::string, double> child;
    const std::string prefix = "scheduler_" + std::to_string(index) + "_";
    for (const auto& [key, value] : state) {
        if (key.rfind(prefix, 0) == 0) {
            child[key.substr(prefix.size())] = value;
        }
    }
    if (child.empty()) {
        throw std::invalid_argument("Composite scheduler child state is missing");
    }
    return child;
}

std::vector<SchedulerDescriptor> child_descriptors(
    const std::vector<std::shared_ptr<LRScheduler>>& schedulers) {
    std::vector<SchedulerDescriptor> result;
    result.reserve(schedulers.size());
    for (const auto& scheduler : schedulers) {
        result.push_back(scheduler->descriptor());
    }
    return result;
}

bool children_checkpoint_safe(
    const std::vector<std::shared_ptr<LRScheduler>>& schedulers) {
    for (const auto& scheduler : schedulers) {
        if (!scheduler->descriptor().checkpoint_safe) return false;
    }
    return true;
}

void validate_children(
    const std::shared_ptr<Optimizer>& optimizer,
    const std::vector<std::shared_ptr<LRScheduler>>& schedulers) {
    if (schedulers.empty()) {
        throw std::invalid_argument(
            "Composite scheduler requires at least one child scheduler");
    }
    for (const auto& scheduler : schedulers) {
        if (!scheduler) {
            throw std::invalid_argument("Composite scheduler child cannot be null");
        }
        if (scheduler->optimizer().get() != optimizer.get()) {
            throw std::invalid_argument(
                "Composite scheduler children must share the parent optimizer");
        }
    }
}

void restore_transaction(
    LRScheduler& parent,
    const std::map<std::string, double>& parent_snapshot,
    const std::vector<std::shared_ptr<LRScheduler>>& children,
    const std::vector<std::map<std::string, double>>& child_snapshots) {
    for (size_t i = 0; i < children.size(); ++i) {
        children[i]->load_state_dict(child_snapshots[i]);
    }
    parent.LRScheduler::load_state_dict(parent_snapshot);
}

} // namespace

SequentialLR::SequentialLR(
    std::shared_ptr<Optimizer> optimizer,
    std::vector<std::shared_ptr<LRScheduler>> schedulers,
    std::vector<int64_t> milestones)
    : LRScheduler(std::move(optimizer)),
      schedulers_(std::move(schedulers)),
      milestones_(std::move(milestones)) {
    validate_children(optimizer_, schedulers_);
    if (milestones_.size() + 1 != schedulers_.size()) {
        throw std::invalid_argument(
            "SequentialLR requires one fewer milestone than schedulers");
    }
    std::string encoded_milestones;
    for (size_t i = 0; i < milestones_.size(); ++i) {
        if (milestones_[i] <= 0 ||
            (i > 0 && milestones_[i] <= milestones_[i - 1])) {
            throw std::invalid_argument(
                "SequentialLR milestones must be positive and strictly increasing");
        }
        if (i != 0) encoded_milestones.push_back(',');
        encoded_milestones += descriptor_integer(milestones_[i]);
    }
    configure_descriptor(
        "jyppx.scheduler.sequential_lr/v1", SchedulerStepMode::epoch,
        {{"milestones", std::move(encoded_milestones)}},
        child_descriptors(schedulers_), children_checkpoint_safe(schedulers_));
}

void SequentialLR::step() {
    ++step_count_;
    size_t index = 0;
    while (index < milestones_.size() && step_count_ >= milestones_[index]) ++index;
    schedulers_[index]->step();
    set_last_lrs(schedulers_[index]->get_last_lrs());
}

std::map<std::string, double> SequentialLR::state_dict() const {
    auto state = LRScheduler::state_dict();
    for (size_t i = 0; i < schedulers_.size(); ++i) {
        for (const auto& [key, value] : schedulers_[i]->state_dict()) {
            state["scheduler_" + std::to_string(i) + "_" + key] = value;
        }
    }
    return state;
}

void SequentialLR::load_state_dict(const std::map<std::string, double>& state) {
    validate_descriptor(state);
    std::vector<std::map<std::string, double>> decoded;
    std::vector<std::map<std::string, double>> snapshots;
    decoded.reserve(schedulers_.size());
    snapshots.reserve(schedulers_.size());
    for (size_t i = 0; i < schedulers_.size(); ++i) {
        decoded.push_back(child_state(state, i));
        schedulers_[i]->validate_descriptor(decoded.back());
        snapshots.push_back(schedulers_[i]->state_dict());
    }
    const auto parent_snapshot = LRScheduler::state_dict();
    try {
        for (size_t i = 0; i < schedulers_.size(); ++i) {
            schedulers_[i]->load_state_dict(decoded[i]);
        }
        LRScheduler::load_state_dict(state);
    } catch (...) {
        restore_transaction(*this, parent_snapshot, schedulers_, snapshots);
        throw;
    }
}

ChainedScheduler::ChainedScheduler(
    std::shared_ptr<Optimizer> optimizer,
    std::vector<std::shared_ptr<LRScheduler>> schedulers)
    : LRScheduler(std::move(optimizer)), schedulers_(std::move(schedulers)) {
    validate_children(optimizer_, schedulers_);
    configure_descriptor(
        "jyppx.scheduler.chained/v1", SchedulerStepMode::epoch, {},
        child_descriptors(schedulers_), children_checkpoint_safe(schedulers_));
}

void ChainedScheduler::step() {
    ++step_count_;
    auto lrs = optimizer_->get_lrs();
    for (const auto& scheduler : schedulers_) {
        const auto before = scheduler->get_last_lrs();
        scheduler->step();
        const auto after = scheduler->get_last_lrs();
        for (size_t i = 0; i < lrs.size(); ++i) {
            const double factor = before[i] == 0.0 ? 0.0 : after[i] / before[i];
            lrs[i] *= factor;
        }
    }
    set_last_lrs(lrs);
}

std::map<std::string, double> ChainedScheduler::state_dict() const {
    auto state = LRScheduler::state_dict();
    for (size_t i = 0; i < schedulers_.size(); ++i) {
        for (const auto& [key, value] : schedulers_[i]->state_dict()) {
            state["scheduler_" + std::to_string(i) + "_" + key] = value;
        }
    }
    return state;
}

void ChainedScheduler::load_state_dict(const std::map<std::string, double>& state) {
    validate_descriptor(state);
    std::vector<std::map<std::string, double>> decoded;
    std::vector<std::map<std::string, double>> snapshots;
    decoded.reserve(schedulers_.size());
    snapshots.reserve(schedulers_.size());
    for (size_t i = 0; i < schedulers_.size(); ++i) {
        decoded.push_back(child_state(state, i));
        schedulers_[i]->validate_descriptor(decoded.back());
        snapshots.push_back(schedulers_[i]->state_dict());
    }
    const auto parent_snapshot = LRScheduler::state_dict();
    try {
        for (size_t i = 0; i < schedulers_.size(); ++i) {
            schedulers_[i]->load_state_dict(decoded[i]);
        }
        LRScheduler::load_state_dict(state);
    } catch (...) {
        restore_transaction(*this, parent_snapshot, schedulers_, snapshots);
        throw;
    }
}

} // namespace jyppx::optim::schedulers
