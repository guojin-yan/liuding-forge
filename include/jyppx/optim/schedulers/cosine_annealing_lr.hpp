#ifndef JYPPX_OPTIM_SCHEDULERS_COSINE_ANNEALING_LR_HPP
#define JYPPX_OPTIM_SCHEDULERS_COSINE_ANNEALING_LR_HPP

#include <jyppx/optim/schedulers/lr_scheduler.hpp>

namespace jyppx {
namespace optim {
namespace schedulers {

/**
 * @brief Cosine annealing learning rate scheduler
 */
class CosineAnnealingLR : public LRScheduler {
public:
    CosineAnnealingLR(
        std::shared_ptr<Optimizer> optimizer,
        int64_t T_max,
        double eta_min = 0.0
    );

    using LRScheduler::step;  // keep the metric-less overload set visible

    void step() override;

private:
    int64_t T_max_;
    double eta_min_;
};

} // namespace schedulers
} // namespace optim
} // namespace jyppx

#endif // JYPPX_OPTIM_SCHEDULERS_COSINE_ANNEALING_LR_HPP
