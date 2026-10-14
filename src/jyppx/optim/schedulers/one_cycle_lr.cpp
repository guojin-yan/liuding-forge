#include <jyppx/optim/schedulers/one_cycle_lr.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace jyppx::optim::schedulers {

namespace {
constexpr double kPi = 3.141592653589793238462643383279502884;
bool finite(double value) { return std::isfinite(value); }
}

OneCycleLR::OneCycleLR(std::shared_ptr<Optimizer> optimizer,
                       double max_lr,
                       int64_t total_steps,
                       double pct_start,
                       std::string anneal_strategy,
                       double div_factor,
                       double final_div_factor,
                       bool three_phase,
                       bool cycle_momentum,
                       double base_momentum,
                       double max_momentum)
    : OneCycleLR(std::move(optimizer), max_lr, total_steps, pct_start,
                 std::move(anneal_strategy), div_factor, final_div_factor,
                 three_phase, cycle_momentum, base_momentum, max_momentum, 0) {}

OneCycleLR::OneCycleLR(std::shared_ptr<Optimizer> optimizer,
                       double max_lr,
                       int64_t epochs,
                       int64_t steps_per_epoch,
                       double pct_start,
                       std::string anneal_strategy,
                       double div_factor,
                       double final_div_factor,
                       bool three_phase,
                       bool cycle_momentum,
                       double base_momentum,
                       double max_momentum)
    : OneCycleLR(std::move(optimizer), max_lr,
                 epochs > 0 && steps_per_epoch > 0 ? epochs * steps_per_epoch : 0,
                 pct_start, std::move(anneal_strategy), div_factor,
                 final_div_factor, three_phase, cycle_momentum, base_momentum,
                 max_momentum, 0) {
    if (epochs <= 0 || steps_per_epoch <= 0) {
        throw std::invalid_argument("OneCycleLR epochs and steps_per_epoch must be positive");
    }
}

OneCycleLR::OneCycleLR(std::shared_ptr<Optimizer> optimizer,
                       double max_lr,
                       int64_t total_steps,
                       double pct_start,
                       std::string anneal_strategy,
                       double div_factor,
                       double final_div_factor,
                       bool three_phase,
                       bool cycle_momentum,
                       double base_momentum,
                       double max_momentum,
                       int)
    : LRScheduler(std::move(optimizer)),
      total_steps_(total_steps),
      pct_start_(pct_start),
      anneal_strategy_(std::move(anneal_strategy)),
      div_factor_(div_factor),
      final_div_factor_(final_div_factor),
      three_phase_(three_phase),
      cycle_momentum_(cycle_momentum),
      base_momentum_(base_momentum),
      max_momentum_(max_momentum) {
    if (total_steps_ <= 0) {
        throw std::invalid_argument("OneCycleLR total_steps must be positive");
    }
    if (!finite(max_lr) || max_lr <= 0.0) {
        throw std::invalid_argument("OneCycleLR max_lr must be positive and finite");
    }
    if (!finite(pct_start_) || pct_start_ < 0.0 || pct_start_ > 1.0) {
        throw std::invalid_argument("OneCycleLR pct_start must be in [0, 1]");
    }
    if (anneal_strategy_ != "cos" && anneal_strategy_ != "linear") {
        throw std::invalid_argument("OneCycleLR anneal_strategy must be 'cos' or 'linear'");
    }
    if (!finite(div_factor_) || div_factor_ <= 0.0 ||
        !finite(final_div_factor_) || final_div_factor_ <= 0.0) {
        throw std::invalid_argument("OneCycleLR division factors must be positive and finite");
    }
    if (!finite(base_momentum_) || !finite(max_momentum_) ||
        base_momentum_ < 0.0 || max_momentum_ < base_momentum_) {
        throw std::invalid_argument("OneCycleLR momentum range is invalid");
    }

    max_lrs_.assign(base_lrs_.size(), max_lr);
    initial_lrs_.reserve(base_lrs_.size());
    final_lrs_.reserve(base_lrs_.size());
    for (size_t i = 0; i < base_lrs_.size(); ++i) {
        initial_lrs_.push_back(max_lrs_[i] / div_factor_);
        final_lrs_.push_back(max_lrs_[i] / (div_factor_ * final_div_factor_));
    }
    if (cycle_momentum_) {
        for (const auto& group : optimizer_->param_groups()) {
            const bool has_momentum = group.options.count("momentum") != 0;
            const bool has_beta1 = group.options.count("beta1") != 0;
            if (!has_momentum && !has_beta1) {
                throw std::invalid_argument(
                    "OneCycleLR cycle_momentum requires SGD momentum or Adam beta1");
            }
            base_momentums_.push_back(base_momentum_);
            max_momentums_.push_back(max_momentum_);
        }
    }
    configure_descriptor(
        "jyppx.scheduler.one_cycle_lr/v1", SchedulerStepMode::batch,
        {{"anneal_strategy", anneal_strategy_},
         {"base_momentum", descriptor_number(base_momentum_)},
         {"cycle_momentum", descriptor_boolean(cycle_momentum_)},
         {"div_factor", descriptor_number(div_factor_)},
         {"final_div_factor", descriptor_number(final_div_factor_)},
         {"max_lr", descriptor_number(max_lr)},
         {"max_momentum", descriptor_number(max_momentum_)},
         {"pct_start", descriptor_number(pct_start_)},
         {"three_phase", descriptor_boolean(three_phase_)},
         {"total_steps", descriptor_integer(total_steps_)}});
    // PyTorch applies the initial step during construction.
    set_last_lrs(initial_lrs_);
    if (cycle_momentum_) {
        set_momentum(max_momentum_);
    }
}

double OneCycleLR::anneal(double start, double end, double pct,
                          const std::string& strategy) {
    if (strategy == "linear") {
        return start + (end - start) * pct;
    }
    return end + (start - end) * (1.0 + std::cos(kPi * pct)) / 2.0;
}

void OneCycleLR::set_momentum(double value) {
    for (auto& group : optimizer_->param_groups()) {
        if (group.options.count("momentum")) {
            group.options["momentum"] = value;
        }
        if (group.options.count("beta1")) {
            group.options["beta1"] = value;
        }
    }
}

void OneCycleLR::step() {
    if (step_count_ >= total_steps_) {
        throw std::out_of_range("OneCycleLR has exceeded total_steps");
    }
    ++step_count_;
    const double first_end = pct_start_ * static_cast<double>(total_steps_) - 1.0;
    const double second_end = three_phase_ ? 2.0 * pct_start_ * total_steps_ - 2.0
                                          : static_cast<double>(total_steps_) - 1.0;
    std::vector<double> lrs;
    lrs.reserve(max_lrs_.size());
    double pct = 0.0;
    int phase = 0;
    if (static_cast<double>(step_count_) <= first_end) {
        pct = static_cast<double>(step_count_) / first_end;
        phase = 0;
    } else if (!three_phase_ || static_cast<double>(step_count_) <= second_end) {
        pct = (static_cast<double>(step_count_) - first_end) /
              (second_end - first_end);
        phase = 1;
    } else {
        pct = (static_cast<double>(step_count_) - second_end) /
              (static_cast<double>(total_steps_) - 1.0 - second_end);
        phase = 2;
    }
    for (size_t i = 0; i < max_lrs_.size(); ++i) {
        if (phase == 0) {
            lrs.push_back(anneal(initial_lrs_[i], max_lrs_[i], pct, anneal_strategy_));
        } else if (phase == 1) {
            lrs.push_back(anneal(max_lrs_[i], three_phase_ ? initial_lrs_[i] : final_lrs_[i],
                                 pct, anneal_strategy_));
        } else {
            lrs.push_back(anneal(initial_lrs_[i], final_lrs_[i], pct, anneal_strategy_));
        }
    }
    // OneCycleLR deliberately mirrors PyTorch's final scheduled step. With
    // linear annealing that step can extrapolate below zero because PyTorch
    // does not clamp phase progress. Keep Optimizer::set_lrs validation for
    // ordinary callers while applying this scheduler-specific value here.
    for (size_t i = 0; i < lrs.size(); ++i) {
        optimizer_->param_groups()[i].options["lr"] = lrs[i];
    }
    last_lrs_ = lrs;
    last_lr_ = lrs.front();
    if (cycle_momentum_) {
        double momentum = max_momentum_;
        if (phase == 0) momentum = anneal(max_momentum_, base_momentum_, pct, anneal_strategy_);
        else if (phase == 1) momentum = anneal(base_momentum_, max_momentum_, pct, anneal_strategy_);
        else momentum = max_momentum_;
        set_momentum(momentum);
    }
}

std::map<std::string, double> OneCycleLR::state_dict() const {
    auto state = LRScheduler::state_dict();
    if (cycle_momentum_) {
        for (size_t i = 0; i < optimizer_->param_groups().size(); ++i) {
            const auto& group = optimizer_->param_groups()[i];
            if (auto momentum_it = group.options.find("momentum"); momentum_it != group.options.end()) {
                state["momentum_" + std::to_string(i)] = momentum_it->second;
            } else if (auto beta_it = group.options.find("beta1"); beta_it != group.options.end()) {
                state["momentum_" + std::to_string(i)] = beta_it->second;
            }
        }
    }
    return state;
}

void OneCycleLR::load_state_dict(const std::map<std::string, double>& state) {
    const bool has_format = state.count("format_version") != 0;
    int64_t restored_step = step_count_;
    std::vector<double> lrs;
    decode_state_dict(state, true, restored_step, lrs);

    std::vector<double> momentums(optimizer_->param_groups().size(), 0.0);
    std::vector<bool> has_momentum(momentums.size(), false);

    if (cycle_momentum_) {
        for (size_t i = 0; i < optimizer_->param_groups().size(); ++i) {
            auto it = state.find("momentum_" + std::to_string(i));
            if (it != state.end()) {
                if (!finite(it->second) || it->second < 0.0) {
                    throw std::invalid_argument("OneCycleLR momentum must be finite and non-negative");
                }
                momentums[i] = it->second;
                has_momentum[i] = true;
            } else if (has_format) {
                throw std::invalid_argument(
                    "OneCycleLR checkpoint is missing parameter-group momentum");
            }
        }
    }

    for (size_t i = 0; i < lrs.size(); ++i) {
        optimizer_->param_groups()[i].options["lr"] = lrs[i];
    }
    last_lrs_ = lrs;
    last_lr_ = lrs.front();
    step_count_ = restored_step;
    for (size_t i = 0; i < momentums.size(); ++i) {
        if (!has_momentum[i]) continue;
        auto& group = optimizer_->param_groups()[i];
        if (group.options.count("momentum")) group.options["momentum"] = momentums[i];
        if (group.options.count("beta1")) group.options["beta1"] = momentums[i];
    }
}

} // namespace jyppx::optim::schedulers
