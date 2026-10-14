#include <jyppx/optim/schedulers/cosine_annealing_warm_restarts.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace jyppx::optim::schedulers {
CosineAnnealingWarmRestarts::CosineAnnealingWarmRestarts(
    std::shared_ptr<Optimizer> optimizer, int64_t T_0, int64_t T_mult, double eta_min)
    : LRScheduler(std::move(optimizer)), T_0_(T_0), T_mult_(T_mult),
      eta_min_(eta_min), cycle_length_(T_0) {
    if (T_0_ <= 0) throw std::invalid_argument("CosineAnnealingWarmRestarts T_0 must be positive");
    if (T_mult_ < 1) throw std::invalid_argument("CosineAnnealingWarmRestarts T_mult must be >= 1");
    if (!std::isfinite(eta_min_) || eta_min_ < 0.0) throw std::invalid_argument("CosineAnnealingWarmRestarts eta_min must be non-negative and finite");
    for (double base : base_lrs_) if (eta_min_ > base) throw std::invalid_argument("CosineAnnealingWarmRestarts eta_min cannot exceed base lr");
    configure_descriptor(
        "jyppx.scheduler.cosine_warm_restarts/v1", SchedulerStepMode::epoch,
        {{"T_0", descriptor_integer(T_0_)},
         {"T_mult", descriptor_integer(T_mult_)},
         {"eta_min", descriptor_number(eta_min_)}});
}
void CosineAnnealingWarmRestarts::step() {
    ++step_count_;
    ++cycle_step_;
    if (cycle_step_ >= cycle_length_) {
        cycle_step_ = 0;
        ++cycle_;
        if (T_mult_ > 1) {
            if (cycle_length_ > std::numeric_limits<int64_t>::max() / T_mult_) throw std::overflow_error("CosineAnnealingWarmRestarts cycle length overflow");
            cycle_length_ *= T_mult_;
        }
    }
    constexpr double pi = 3.141592653589793238462643383279502884;
    const double pct = static_cast<double>(cycle_step_) / static_cast<double>(cycle_length_);
    std::vector<double> lrs;
    for (double base : base_lrs_) lrs.push_back(eta_min_ + (base - eta_min_) * (1.0 + std::cos(pi * pct)) / 2.0);
    set_last_lrs(lrs);
}
std::map<std::string, double> CosineAnnealingWarmRestarts::state_dict() const {
    auto state = LRScheduler::state_dict();
    state["cycle"] = static_cast<double>(cycle_);
    state["cycle_step"] = static_cast<double>(cycle_step_);
    state["cycle_length"] = static_cast<double>(cycle_length_);
    return state;
}
void CosineAnnealingWarmRestarts::load_state_dict(const std::map<std::string, double>& state) {
    const bool has_format = state.count("format_version") != 0;
    auto get_int = [&](const char* name, int64_t fallback) {
        auto it = state.find(name);
        if (it == state.end()) {
            if (has_format) throw std::invalid_argument(std::string("CosineAnnealingWarmRestarts state is missing ") + name);
            return fallback;
        }
        if (!std::isfinite(it->second) || it->second < 0 ||
            std::floor(it->second) != it->second ||
            it->second > 9007199254740991.0) {
            throw std::invalid_argument(std::string("Invalid CosineAnnealingWarmRestarts state field: ") + name);
        }
        return static_cast<int64_t>(it->second);
    };
    const int64_t restored_cycle = get_int("cycle", cycle_);
    const int64_t restored_step = get_int("cycle_step", cycle_step_);
    const int64_t restored_length = get_int("cycle_length", cycle_length_);
    if (restored_length <= 0 || restored_step >= restored_length) {
        throw std::invalid_argument("Invalid CosineAnnealingWarmRestarts state");
    }
    LRScheduler::load_state_dict(state);
    cycle_ = restored_cycle;
    cycle_step_ = restored_step;
    cycle_length_ = restored_length;
}
} // namespace jyppx::optim::schedulers
