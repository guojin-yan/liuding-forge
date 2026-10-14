#ifndef JYPPX_OPTIM_SCHEDULERS_WARMUP_SCHEDULER_HPP
#define JYPPX_OPTIM_SCHEDULERS_WARMUP_SCHEDULER_HPP

#include <jyppx/optim/schedulers/lr_scheduler.hpp>

namespace jyppx::optim::schedulers {

/**
 * Linear learning-rate warmup followed by an optional scheduler.
 * Construction immediately applies `start_factor`; after `warmup_iters`
 * calls to step(), the optimizer reaches its original base learning rate.
 */
class WarmupScheduler : public LRScheduler {
public:
    WarmupScheduler(std::shared_ptr<Optimizer> optimizer,
                    int64_t warmup_iters,
                    double start_factor = 0.001,
                    std::shared_ptr<LRScheduler> after_scheduler = nullptr);

    using LRScheduler::step;
    void step() override;
    std::map<std::string, double> state_dict() const override;
    void load_state_dict(const std::map<std::string, double>& state) override;

private:
    int64_t warmup_iters_;
    double start_factor_;
    std::shared_ptr<LRScheduler> after_scheduler_;
};

} // namespace jyppx::optim::schedulers

namespace jyppx::optim {
using WarmupScheduler = schedulers::WarmupScheduler;
}

#endif
