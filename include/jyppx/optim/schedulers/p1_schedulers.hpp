#ifndef JYPPX_OPTIM_SCHEDULERS_P1_SCHEDULERS_HPP
#define JYPPX_OPTIM_SCHEDULERS_P1_SCHEDULERS_HPP

#include <jyppx/optim/schedulers/lr_scheduler.hpp>
#include <functional>

namespace jyppx::optim::schedulers {

class CyclicLR : public LRScheduler {
public:
    CyclicLR(std::shared_ptr<Optimizer> optimizer, double base_lr, double max_lr,
             int64_t step_size_up, int64_t step_size_down = 0,
             std::string mode = "triangular", double gamma = 1.0,
             bool cycle_momentum = true, double base_momentum = 0.8,
             double max_momentum = 0.9,
             std::function<double(double)> scale_fn = {},
             std::string scale_mode = "cycle");
    using LRScheduler::step;
    void step() override;
private:
    double base_lr_, max_lr_, gamma_, base_momentum_, max_momentum_;
    int64_t step_up_, step_down_;
    std::string mode_, scale_mode_;
    bool cycle_momentum_;
    std::function<double(double)> scale_fn_;
};

class PolynomialLR : public LRScheduler {
public:
    PolynomialLR(std::shared_ptr<Optimizer> optimizer, int64_t total_iters,
                 double power = 1.0);
    using LRScheduler::step;
    void step() override;
private:
    int64_t total_iters_; double power_;
};

class ConstantLR : public LRScheduler {
public:
    ConstantLR(std::shared_ptr<Optimizer> optimizer, double factor = 1.0 / 3.0,
               int64_t total_iters = 5);
    using LRScheduler::step;
    void step() override;
private:
    double factor_; int64_t total_iters_;
};

class LambdaLR : public LRScheduler {
public:
    explicit LambdaLR(std::shared_ptr<Optimizer> optimizer,
                      std::function<double(int64_t)> lr_lambda);
    LambdaLR(std::shared_ptr<Optimizer> optimizer,
             std::vector<std::function<double(int64_t)>> lr_lambdas);
    using LRScheduler::step;
    void step() override;
private:
    std::vector<std::function<double(int64_t)>> lambdas_;
};

} // namespace jyppx::optim::schedulers
namespace jyppx::optim {
using CyclicLR = schedulers::CyclicLR;
using PolynomialLR = schedulers::PolynomialLR;
using ConstantLR = schedulers::ConstantLR;
using LambdaLR = schedulers::LambdaLR;
}
#endif
