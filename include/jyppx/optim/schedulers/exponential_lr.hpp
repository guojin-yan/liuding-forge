#ifndef JYPPX_OPTIM_SCHEDULERS_EXPONENTIAL_LR_HPP
#define JYPPX_OPTIM_SCHEDULERS_EXPONENTIAL_LR_HPP

#include <jyppx/optim/schedulers/lr_scheduler.hpp>

namespace jyppx {
namespace optim {
namespace schedulers {

/**
 * @brief Exponential learning rate scheduler
 *
 * Decays the learning rate by gamma every epoch
 */
class ExponentialLR : public LRScheduler {
public:
    ExponentialLR(std::shared_ptr<Optimizer> optimizer, double gamma);

    using LRScheduler::step;  // keep the metric-less overload set visible

    void step() override;

private:
    double gamma_;
};

} // namespace schedulers
} // namespace optim
} // namespace jyppx

#endif // JYPPX_OPTIM_SCHEDULERS_EXPONENTIAL_LR_HPP
