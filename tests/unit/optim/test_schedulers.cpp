// Unit tests for jyppx::optim::schedulers: StepLR, ExponentialLR and
// CosineAnnealingLR.

#include <jyppx/optim/schedulers/cosine_annealing_lr.hpp>
#include <jyppx/optim/schedulers/exponential_lr.hpp>
#include <jyppx/optim/schedulers/step_lr.hpp>
#include <jyppx/optim/sgd.hpp>

#include "test_utils/jyppx_test_helpers.hpp"

#include <gtest/gtest.h>

#include <limits>

namespace {

using namespace jyppx;
using namespace jyppx_test;

class SchedulerTest : public ::testing::Test {
protected:
    void SetUp() override {
        torch::manual_seed(99);
        params_ = make_params({{2, 2}});
        optimizer_ = std::make_shared<optim::SGD>(
            params_, optim::SGDOptions{.lr = base_lr_});
    }

    static constexpr double base_lr_ = 0.1;
    TensorList params_;
    std::shared_ptr<optim::SGD> optimizer_;
};

TEST_F(SchedulerTest, StepLRHoldsThenDecaysPerStepSize) {
    optim::schedulers::StepLR scheduler(optimizer_, /*step_size=*/2, /*gamma=*/0.1);

    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), base_lr_);
    scheduler.step();
    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), base_lr_);      // gamma^0
    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), base_lr_ * 0.1, 1e-12); // gamma^1
    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), base_lr_ * 0.1, 1e-12);
    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), base_lr_ * 0.01, 1e-12); // gamma^2
    EXPECT_EQ(scheduler.get_step(), 4);
}

TEST_F(SchedulerTest, ExponentialLRDecaysEveryStep) {
    optim::schedulers::ExponentialLR scheduler(optimizer_, /*gamma=*/0.5);

    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), base_lr_ * 0.5, 1e-12);
    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), base_lr_ * 0.25, 1e-12);
    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), base_lr_ * 0.125, 1e-12);
}

TEST_F(SchedulerTest, CosineAnnealingFollowsTheCosineProfile) {
    optim::schedulers::CosineAnnealingLR scheduler(optimizer_, /*T_max=*/4, /*eta_min=*/0.0);

    constexpr double pi = 3.14159265358979323846;
    for (int64_t step = 1; step <= 4; ++step) {
        scheduler.step();
        const double expected =
            base_lr_ * (1.0 + std::cos(pi * static_cast<double>(step) / 4.0)) / 2.0;
        EXPECT_NEAR(scheduler.get_last_lr(), expected, 1e-12) << "step " << step;
    }
    EXPECT_NEAR(scheduler.get_last_lr(), 0.0, 1e-12);
}

TEST_F(SchedulerTest, SchedulerUpdatesTheOptimizerLearningRate) {
    optim::schedulers::ExponentialLR scheduler(optimizer_, /*gamma=*/0.5);
    scheduler.step();
    EXPECT_DOUBLE_EQ(optimizer_->get_lrs()[0], scheduler.get_last_lrs()[0]);
    scheduler.step();
    EXPECT_DOUBLE_EQ(optimizer_->get_lrs()[0], base_lr_ * 0.25);
}

TEST_F(SchedulerTest, DescriptorIdentifiesTypeConfigurationAndStepMode) {
    optim::schedulers::StepLR scheduler(optimizer_, 2, 0.1);
    const auto& descriptor = scheduler.descriptor();
    EXPECT_EQ(descriptor.type_id, "jyppx.scheduler.step_lr/v1");
    EXPECT_EQ(descriptor.step_mode, optim::schedulers::SchedulerStepMode::epoch);
    EXPECT_EQ(descriptor.config.at("step_size"), "2");
    EXPECT_EQ(descriptor.config.at("gamma"), "0.10000000000000001");
    EXPECT_TRUE(descriptor.checkpoint_safe);
    EXPECT_EQ(scheduler.descriptor_fingerprint().size(), 16u);
}

TEST_F(SchedulerTest, DescriptorMismatchFailsBeforeStateMutation) {
    optim::schedulers::StepLR source(optimizer_, 2, 0.1);
    source.step();
    const auto state = source.state_dict();

    auto other_optimizer = std::make_shared<optim::SGD>(
        make_params({{2, 2}}), optim::SGDOptions{.lr = base_lr_});
    optim::schedulers::StepLR different(other_optimizer, 3, 0.1);
    const auto before = different.state_dict();
    EXPECT_THROW(different.load_state_dict(state), std::invalid_argument);
    EXPECT_EQ(different.state_dict(), before);
    EXPECT_DOUBLE_EQ(other_optimizer->get_lrs().front(), base_lr_);
}

TEST_F(SchedulerTest, InvalidArgumentsThrow) {
    EXPECT_THROW(
        optim::schedulers::StepLR(optimizer_, /*step_size=*/0), std::invalid_argument);
    EXPECT_THROW(
        optim::schedulers::StepLR(optimizer_, 2, /*gamma=*/0.0), std::invalid_argument);
    EXPECT_THROW(
        optim::schedulers::ExponentialLR(optimizer_, 0.0), std::invalid_argument);
    EXPECT_THROW(
        optim::schedulers::CosineAnnealingLR(optimizer_, 0), std::invalid_argument);
    EXPECT_THROW(
        optim::schedulers::CosineAnnealingLR(optimizer_, 4, /*eta_min=*/1.0),
        std::invalid_argument);
    EXPECT_THROW(
        optim::schedulers::ExponentialLR(nullptr, 0.5), std::invalid_argument);
    EXPECT_THROW(
        optim::schedulers::StepLR(
            optimizer_, 2, std::numeric_limits<double>::quiet_NaN()),
        std::invalid_argument);
    EXPECT_THROW(
        optim::schedulers::CosineAnnealingLR(
            optimizer_, 4, std::numeric_limits<double>::infinity()),
        std::invalid_argument);
}

TEST_F(SchedulerTest, SchedulerWithoutOptimizerLrThrows) {
    auto params = make_params({{2}});
    // An optimizer constructed through the base class without group options
    // has no lr; schedulers must reject it.
    class BareOptimizer : public optim::Optimizer {
    public:
        explicit BareOptimizer(const TensorList& p) : Optimizer(p) {}
        void step() override {}
    };
    auto bare = std::make_shared<BareOptimizer>(params);
    EXPECT_THROW(optim::schedulers::ExponentialLR(bare, 0.5), std::runtime_error);
}

} // namespace
