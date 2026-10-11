#ifndef JYPPX_OPTIM_SCHEDULERS_STEP_LR_HPP
#define JYPPX_OPTIM_SCHEDULERS_STEP_LR_HPP

#include <jyppx/optim/schedulers/lr_scheduler.hpp>

namespace jyppx {
namespace optim {
namespace schedulers {

/**
 * @brief Step learning rate scheduler
 *
 * Decays the learning rate by gamma every step_size epochs
 */
class StepLR : public LRScheduler {
public:
    StepLR(std::shared_ptr<Optimizer> optimizer, int64_t step_size, double gamma = 0.1);

    using LRScheduler::step;  // keep the metric-less overload set visible

    void step() override;

private:
    int64_t step_size_;
    double gamma_;
};

} // namespace schedulers
} // namespace optim
} // namespace jyppx

#endif // JYPPX_OPTIM_SCHEDULERS_STEP_LR_HPP
