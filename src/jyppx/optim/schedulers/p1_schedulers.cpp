#include <jyppx/optim/schedulers/p1_schedulers.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace jyppx::optim::schedulers {

CyclicLR::CyclicLR(std::shared_ptr<Optimizer> optimizer, double base_lr, double max_lr,
                   int64_t step_size_up, int64_t step_size_down,
                   std::string mode, double gamma, bool cycle_momentum,
                   double base_momentum, double max_momentum,
                   std::function<double(double)> scale_fn,
                   std::string scale_mode)
    : LRScheduler(std::move(optimizer)), base_lr_(base_lr), max_lr_(max_lr), gamma_(gamma),
      base_momentum_(base_momentum), max_momentum_(max_momentum),
      step_up_(step_size_up), step_down_(step_size_down > 0 ? step_size_down : step_size_up),
      mode_(std::move(mode)), scale_mode_(std::move(scale_mode)),
      cycle_momentum_(cycle_momentum), scale_fn_(std::move(scale_fn)) {
    if (base_lr_ < 0.0 || max_lr_ <= base_lr_ || !std::isfinite(base_lr_) || !std::isfinite(max_lr_)) throw std::invalid_argument("CyclicLR requires 0 <= base_lr < max_lr");
    if (step_up_ <= 0 || step_down_ <= 0) throw std::invalid_argument("CyclicLR step sizes must be positive");
    if (!scale_fn_ && mode_ != "triangular" && mode_ != "triangular2" && mode_ != "exp_range") throw std::invalid_argument("CyclicLR mode is invalid");
    if (scale_mode_ != "cycle" && scale_mode_ != "iterations") throw std::invalid_argument("CyclicLR scale_mode must be 'cycle' or 'iterations'");
    if (mode_ == "exp_range" && (gamma_ <= 0.0 || !std::isfinite(gamma_))) throw std::invalid_argument("CyclicLR gamma must be positive and finite");
    if (cycle_momentum_) {
        if (!std::isfinite(base_momentum_) || !std::isfinite(max_momentum_) ||
            base_momentum_ > max_momentum_) {
            throw std::invalid_argument(
                "CyclicLR requires finite base_momentum <= max_momentum");
        }
        for (auto& group : optimizer_->param_groups()) {
            if (!group.options.count("momentum") && !group.options.count("beta1")) {
                throw std::invalid_argument(
                    "CyclicLR cycle_momentum requires momentum or beta1");
            }
            const char* key = group.options.count("beta1") ? "beta1" : "momentum";
            group.options[key] = max_momentum_;
            group.options["base_momentum"] = base_momentum_;
            group.options["max_momentum"] = max_momentum_;
        }
    }
    configure_descriptor(
        "jyppx.scheduler.cyclic_lr/v1", SchedulerStepMode::batch,
        {{"base_lr", descriptor_number(base_lr_)},
         {"base_momentum", descriptor_number(base_momentum_)},
         {"cycle_momentum", descriptor_boolean(cycle_momentum_)},
         {"gamma", descriptor_number(gamma_)},
         {"has_custom_scale_fn", descriptor_boolean(static_cast<bool>(scale_fn_))},
         {"max_lr", descriptor_number(max_lr_)},
         {"max_momentum", descriptor_number(max_momentum_)},
         {"mode", mode_},
         {"scale_mode", scale_mode_},
         {"step_size_down", descriptor_integer(step_down_)},
         {"step_size_up", descriptor_integer(step_up_)}},
        {}, !scale_fn_);
    std::vector<double> initial(base_lrs_.size(), base_lr_); set_last_lrs(initial);
}
void CyclicLR::step() {
    ++step_count_;
    const int64_t cycle_len = step_up_ + step_down_;
    const double cycle =
        std::floor(1.0 + static_cast<double>(step_count_) / cycle_len);
    const double x =
        1.0 + static_cast<double>(step_count_) / cycle_len - cycle;
    const double step_ratio = static_cast<double>(step_up_) / cycle_len;
    const double pct = x <= step_ratio ? x / step_ratio
                                       : (x - 1.0) / (step_ratio - 1.0);
    const double scale_argument =
        scale_mode_ == "cycle" ? cycle : static_cast<double>(step_count_);
    double scale = 1.0;
    if (scale_fn_) {
        scale = scale_fn_(scale_argument);
    } else if (mode_ == "triangular2") {
        scale = std::pow(0.5, cycle - 1.0);
    } else if (mode_ == "exp_range") {
        scale = std::pow(gamma_, static_cast<double>(step_count_));
    }
    if (!std::isfinite(scale)) {
        throw std::invalid_argument("CyclicLR scale function must return a finite value");
    }
    const double amplitude = std::max(0.0, pct) * scale;
    const double lr = base_lr_ + (max_lr_ - base_lr_) * amplitude;
    set_last_lrs(std::vector<double>(base_lrs_.size(), lr));
    if (cycle_momentum_) {
        const double momentum =
            max_momentum_ - (max_momentum_ - base_momentum_) * amplitude;
        for (auto& group : optimizer_->param_groups()) {
            const char* key = group.options.count("beta1") ? "beta1" : "momentum";
            group.options[key] = momentum;
        }
    }
}

PolynomialLR::PolynomialLR(std::shared_ptr<Optimizer> optimizer, int64_t total_iters, double power)
    : LRScheduler(std::move(optimizer)), total_iters_(total_iters), power_(power) {
    if (total_iters_ <= 0 || power_ < 0.0 || !std::isfinite(power_)) throw std::invalid_argument("PolynomialLR total_iters must be positive and power non-negative");
    configure_descriptor(
        "jyppx.scheduler.polynomial_lr/v1", SchedulerStepMode::epoch,
        {{"power", descriptor_number(power_)},
         {"total_iters", descriptor_integer(total_iters_)}});
}
void PolynomialLR::step() {
    ++step_count_;
    const double pct = std::min(1.0, static_cast<double>(step_count_) / total_iters_);
    const double factor = std::pow(1.0 - pct, power_);
    std::vector<double> lrs; for (double base : base_lrs_) lrs.push_back(base * factor); set_last_lrs(lrs);
}

ConstantLR::ConstantLR(std::shared_ptr<Optimizer> optimizer, double factor, int64_t total_iters)
    : LRScheduler(std::move(optimizer)), factor_(factor), total_iters_(total_iters) {
    if (factor_ <= 0.0 || factor_ > 1.0 || total_iters_ < 0 || !std::isfinite(factor_)) throw std::invalid_argument("ConstantLR factor must be in (0,1] and total_iters non-negative");
    configure_descriptor(
        "jyppx.scheduler.constant_lr/v1", SchedulerStepMode::epoch,
        {{"factor", descriptor_number(factor_)},
         {"total_iters", descriptor_integer(total_iters_)}});
    std::vector<double> initial; for (double base : base_lrs_) initial.push_back(base * factor_); set_last_lrs(initial);
}
void ConstantLR::step() {
    ++step_count_;
    const double factor = step_count_ < total_iters_ ? factor_ : 1.0;
    std::vector<double> lrs; for (double base : base_lrs_) lrs.push_back(base * factor); set_last_lrs(lrs);
}

LambdaLR::LambdaLR(std::shared_ptr<Optimizer> optimizer, std::function<double(int64_t)> lr_lambda)
    : LambdaLR(std::move(optimizer), std::vector<std::function<double(int64_t)>>{std::move(lr_lambda)}) {}
LambdaLR::LambdaLR(std::shared_ptr<Optimizer> optimizer, std::vector<std::function<double(int64_t)>> lr_lambdas)
    : LRScheduler(std::move(optimizer)), lambdas_(std::move(lr_lambdas)) {
    if (lambdas_.empty() || (lambdas_.size() != 1 && lambdas_.size() != base_lrs_.size())) throw std::invalid_argument("LambdaLR lambda count must be one or match parameter groups");
    for (const auto& fn : lambdas_) if (!fn) throw std::invalid_argument("LambdaLR lambda cannot be empty");
    configure_descriptor(
        "jyppx.scheduler.lambda_lr/runtime-v1", SchedulerStepMode::epoch,
        {{"callable_count", descriptor_integer(static_cast<int64_t>(lambdas_.size()))}},
        {}, false);
    std::vector<double> initial_lrs;
    initial_lrs.reserve(base_lrs_.size());
    for (size_t index = 0; index < base_lrs_.size(); ++index) {
        const double factor = lambdas_.size() == 1
                                  ? lambdas_.front()(0)
                                  : lambdas_[index](0);
        if (!std::isfinite(factor) || factor < 0.0) {
            throw std::invalid_argument(
                "LambdaLR lambda must return finite non-negative values");
        }
        initial_lrs.push_back(base_lrs_[index] * factor);
    }
    set_last_lrs(initial_lrs);
}
void LambdaLR::step() {
    ++step_count_;
    std::vector<double> lrs; lrs.reserve(base_lrs_.size());
    for (size_t i = 0; i < base_lrs_.size(); ++i) {
        const double factor = lambdas_.size() == 1 ? lambdas_.front()(step_count_) : lambdas_[i](step_count_);
        if (!std::isfinite(factor) || factor < 0.0) throw std::invalid_argument("LambdaLR lambda must return finite non-negative values");
        lrs.push_back(base_lrs_[i] * factor);
    }
    set_last_lrs(lrs);
}
} // namespace jyppx::optim::schedulers
