#ifndef JYPPX_OPTIM_SCHEDULERS_COSINE_ANNEALING_WARM_RESTARTS_HPP
#define JYPPX_OPTIM_SCHEDULERS_COSINE_ANNEALING_WARM_RESTARTS_HPP

#include <jyppx/optim/schedulers/lr_scheduler.hpp>

namespace jyppx::optim::schedulers {

/** @brief Cosine annealing with periodic warm restarts (SGDR). */
class CosineAnnealingWarmRestarts : public LRScheduler {
public:
    CosineAnnealingWarmRestarts(std::shared_ptr<Optimizer> optimizer,
                                int64_t T_0,
                                int64_t T_mult = 1,
                                double eta_min = 0.0);
    using LRScheduler::step;
    void step() override;
    std::map<std::string, double> state_dict() const override;
    void load_state_dict(const std::map<std::string, double>& state) override;
    int64_t current_cycle() const { return cycle_; }
    int64_t cycle_step() const { return cycle_step_; }

private:
    int64_t T_0_;
    int64_t T_mult_;
    double eta_min_;
    int64_t cycle_ = 0;
    int64_t cycle_step_ = 0;
    int64_t cycle_length_;
};

} // namespace jyppx::optim::schedulers
namespace jyppx::optim { using CosineAnnealingWarmRestarts = schedulers::CosineAnnealingWarmRestarts; }
#endif
