#ifndef JYPPX_OPTIM_SCHEDULERS_ONE_CYCLE_LR_HPP
#define JYPPX_OPTIM_SCHEDULERS_ONE_CYCLE_LR_HPP

#include <jyppx/optim/schedulers/lr_scheduler.hpp>

#include <string>

namespace jyppx::optim::schedulers {

/**
 * @brief The one-cycle learning-rate policy from Smith (2018).
 *
 * The scheduler is stepped once per optimizer update.  It supports the same
 * total-step and epoch/step-per-epoch construction modes as PyTorch and also
 * updates SGD momentum or Adam beta1 in the opposite direction when
 * cycle_momentum is enabled.
 */
class OneCycleLR : public LRScheduler {
public:
    OneCycleLR(std::shared_ptr<Optimizer> optimizer,
               double max_lr,
               int64_t total_steps,
               double pct_start = 0.3,
               std::string anneal_strategy = "cos",
               double div_factor = 25.0,
               double final_div_factor = 1e4,
               bool three_phase = false,
               bool cycle_momentum = true,
               double base_momentum = 0.85,
               double max_momentum = 0.95);

    OneCycleLR(std::shared_ptr<Optimizer> optimizer,
               double max_lr,
               int64_t epochs,
               int64_t steps_per_epoch,
               double pct_start = 0.3,
               std::string anneal_strategy = "cos",
               double div_factor = 25.0,
               double final_div_factor = 1e4,
               bool three_phase = false,
               bool cycle_momentum = true,
               double base_momentum = 0.85,
               double max_momentum = 0.95);

    using LRScheduler::step;
    void step() override;
    std::map<std::string, double> state_dict() const override;
    void load_state_dict(const std::map<std::string, double>& state) override;

private:
    OneCycleLR(std::shared_ptr<Optimizer> optimizer,
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
               int);
    static double anneal(double start, double end, double pct, const std::string& strategy);
    void set_momentum(double value);

    int64_t total_steps_;
    double pct_start_;
    std::string anneal_strategy_;
    double div_factor_;
    double final_div_factor_;
    bool three_phase_;
    bool cycle_momentum_;
    double base_momentum_;
    double max_momentum_;
    std::vector<double> max_lrs_;
    std::vector<double> initial_lrs_;
    std::vector<double> final_lrs_;
    std::vector<double> base_momentums_;
    std::vector<double> max_momentums_;
};

} // namespace jyppx::optim::schedulers

// The design document exposes schedulers as jyppx::optim::X.  Keep the
// implementation in the existing schedulers namespace and provide the
// documented spelling as a direct type alias.
namespace jyppx::optim {
using OneCycleLR = schedulers::OneCycleLR;
}

#endif
