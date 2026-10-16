// Unit tests for the remaining legacy StepLR surface.

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
        optim::schedulers::StepLR(
            optimizer_, 2, std::numeric_limits<double>::quiet_NaN()),
        std::invalid_argument);
}

} // namespace
