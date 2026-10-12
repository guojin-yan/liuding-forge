// Unit tests for MultiStepLR and ReduceLROnPlateau.

#include <jyppx/optim/schedulers/multi_step_lr.hpp>
#include <jyppx/optim/schedulers/reduce_lr_on_plateau.hpp>
#include <jyppx/optim/sgd.hpp>

#include "test_utils/jyppx_test_helpers.hpp"

#include <gtest/gtest.h>

namespace {

using namespace jyppx;
using namespace jyppx_test;

class NewSchedulersTest : public ::testing::Test {
protected:
    void SetUp() override {
        torch::manual_seed(77);
        params_ = make_params({{2, 2}});
        optimizer_ = std::make_shared<optim::SGD>(
            params_, optim::SGDOptions{.lr = base_lr_});
    }

    static constexpr double base_lr_ = 0.1;
    TensorList params_;
    std::shared_ptr<optim::SGD> optimizer_;
};

TEST_F(NewSchedulersTest, MultiStepLRDecaysAtMilestones) {
    optim::schedulers::MultiStepLR scheduler(optimizer_, {3, 5}, /*gamma=*/0.1);

    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), base_lr_);
    scheduler.step();
    scheduler.step();
    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), base_lr_);          // 2 steps, no milestone
    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), base_lr_ * 0.1, 1e-12);  // milestone 3 passed
    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), base_lr_ * 0.1, 1e-12);
    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), base_lr_ * 0.01, 1e-12); // milestones 3 and 5
    EXPECT_EQ(scheduler.get_step(), 5);
}

TEST_F(NewSchedulersTest, MultiStepLRValidatesArguments) {
    EXPECT_THROW(
        optim::schedulers::MultiStepLR(optimizer_, {}, 0.1), std::invalid_argument);
    EXPECT_THROW(
        optim::schedulers::MultiStepLR(optimizer_, {5, 3}, 0.1), std::invalid_argument);
    EXPECT_THROW(
        optim::schedulers::MultiStepLR(optimizer_, {0, 3}, 0.1), std::invalid_argument);
    EXPECT_THROW(
        optim::schedulers::MultiStepLR(optimizer_, {3}, 0.0), std::invalid_argument);
}

TEST_F(NewSchedulersTest, ReduceLROnPlateauDecaysAfterPatience) {
    optim::schedulers::ReduceLROnPlateau scheduler(
        optimizer_,
        {.mode = "min", .factor = 0.5, .patience = 1, .threshold = 0.0});

    // Improvement resets the bad-epoch counter.
    scheduler.step(1.0);
    scheduler.step(0.5);
    EXPECT_DOUBLE_EQ(scheduler.best(), 0.5);
    EXPECT_EQ(scheduler.num_bad_epochs(), 0);
    EXPECT_DOUBLE_EQ(scheduler.get_last_lr(), base_lr_);

    // Two non-improving steps (patience=1): the second triggers a decay.
    scheduler.step(0.5);
    EXPECT_EQ(scheduler.num_bad_epochs(), 1);
    scheduler.step(0.6);
    EXPECT_NEAR(scheduler.get_last_lr(), base_lr_ * 0.5, 1e-12);
    EXPECT_EQ(scheduler.num_bad_epochs(), 0); // counter resets after decay

    // Decays apply to the current learning rate, not the base one.
    scheduler.step(0.6);
    scheduler.step(0.6);
    EXPECT_NEAR(scheduler.get_last_lr(), base_lr_ * 0.25, 1e-12);
    EXPECT_DOUBLE_EQ(optimizer_->get_lrs()[0], scheduler.get_last_lr());
}

TEST_F(NewSchedulersTest, ReduceLROnPlateauRespectsMinLr) {
    optim::schedulers::ReduceLROnPlateau scheduler(
        optimizer_,
        {.mode = "min", .factor = 0.1, .patience = 0, .threshold = 0.0, .min_lr = 0.01});

    for (int i = 0; i < 5; ++i) {
        scheduler.step(1.0); // never improves
    }
    EXPECT_NEAR(scheduler.get_last_lr(), 0.01, 1e-12);
    EXPECT_NEAR(optimizer_->get_lrs()[0], 0.01, 1e-12);
}

TEST_F(NewSchedulersTest, ReduceLROnPlateauMaxModeUsesRelativeThreshold) {
    optim::schedulers::ReduceLROnPlateau scheduler(
        optimizer_,
        {.mode = "max", .factor = 0.5, .patience = 0, .threshold = 0.1,
         .threshold_mode = "rel"});

    scheduler.step(1.0);           // first value establishes the best
    EXPECT_EQ(scheduler.num_bad_epochs(), 0);
    scheduler.step(1.05);          // within 10% of best: not an improvement
    EXPECT_NEAR(scheduler.get_last_lr(), base_lr_ * 0.5, 1e-12);
    scheduler.step(1.2);           // > 1.1: improvement
    EXPECT_DOUBLE_EQ(scheduler.best(), 1.2);
    EXPECT_NEAR(scheduler.get_last_lr(), base_lr_ * 0.5, 1e-12); // unchanged
}

TEST_F(NewSchedulersTest, ReduceLROnPlateauRejectsMetriclessStep) {
    optim::schedulers::ReduceLROnPlateau scheduler(optimizer_, {.patience = 0});
    EXPECT_THROW(scheduler.step(), std::logic_error);
}

TEST_F(NewSchedulersTest, EpochSchedulersRejectMetricStep) {
    optim::schedulers::StepLR step_scheduler(optimizer_, 2);
    EXPECT_THROW(step_scheduler.step(0.5), std::logic_error);
}

TEST_F(NewSchedulersTest, SchedulerStateRoundTripsCurrentLearningRate) {
    optim::schedulers::StepLR source(optimizer_, 2, 0.1);
    source.step();
    source.step();
    ASSERT_EQ(source.get_step(), 2);

    auto restored_optimizer = std::make_shared<optim::SGD>(
        params_, optim::SGDOptions{.lr = base_lr_});
    optim::schedulers::StepLR restored(restored_optimizer, 2, 0.1);
    restored.load_state_dict(source.state_dict());

    EXPECT_EQ(restored.get_step(), source.get_step());
    EXPECT_DOUBLE_EQ(restored.get_last_lr(), source.get_last_lr());
    EXPECT_DOUBLE_EQ(restored_optimizer->get_lrs().front(), source.get_last_lr());
    restored.step();
    EXPECT_DOUBLE_EQ(restored.get_last_lr(), base_lr_ * 0.1);
}

TEST_F(NewSchedulersTest, ReduceLROnPlateauStateRestoresMetricHistory) {
    optim::schedulers::ReduceLROnPlateau source(
        optimizer_, {.mode = "min", .factor = 0.5, .patience = 1, .threshold = 0.0});
    source.step(1.0);
    source.step(1.1);
    const auto state = source.state_dict();

    auto restored_optimizer = std::make_shared<optim::SGD>(
        params_, optim::SGDOptions{.lr = base_lr_});
    optim::schedulers::ReduceLROnPlateau restored(
        restored_optimizer,
        {.mode = "min", .factor = 0.5, .patience = 1, .threshold = 0.0});
    restored.load_state_dict(state);

    EXPECT_DOUBLE_EQ(restored.best(), source.best());
    EXPECT_EQ(restored.num_bad_epochs(), source.num_bad_epochs());
    restored.step(1.2);
    EXPECT_NEAR(restored.get_last_lr(), base_lr_ * 0.5, 1e-12);
}

TEST_F(NewSchedulersTest, ReduceLROnPlateauValidatesArguments) {
    EXPECT_THROW(
        optim::schedulers::ReduceLROnPlateau(
            optimizer_, {.mode = "up"}), std::invalid_argument);
    EXPECT_THROW(
        optim::schedulers::ReduceLROnPlateau(
            optimizer_, {.factor = 1.0}), std::invalid_argument);
    EXPECT_THROW(
        optim::schedulers::ReduceLROnPlateau(
            optimizer_, {.patience = -1}), std::invalid_argument);
    EXPECT_THROW(
        optim::schedulers::ReduceLROnPlateau(
            optimizer_, {.threshold_mode = "abs-"}), std::invalid_argument);
    EXPECT_THROW(
        optim::schedulers::ReduceLROnPlateau(
            optimizer_, {.min_lr = 1.0}), std::invalid_argument);
}

} // namespace
