// Unit tests for legacy advanced schedulers not yet migrated.

#include <jyppx/optim/schedulers/reduce_lr_on_plateau.hpp>
#include <jyppx/optim/schedulers/one_cycle_lr.hpp>
#include <jyppx/optim/schedulers/cosine_annealing_warm_restarts.hpp>
#include <jyppx/optim/sgd.hpp>

#include "test_utils/jyppx_test_helpers.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

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

TEST_F(NewSchedulersTest, SchedulerStateRejectsTopologyAndPartialState) {
    optim::schedulers::StepLR source(optimizer_, 2, 0.1);
    auto state = source.state_dict();

    auto second = torch::ones({1}, torch::TensorOptions().requires_grad(true));
    optimizer_->add_param_group(optim::ParamGroup({second}));
    EXPECT_THROW(source.load_state_dict(state), std::invalid_argument);

    auto fresh_optimizer = std::make_shared<optim::SGD>(
        params_, optim::SGDOptions{.lr = base_lr_});
    optim::schedulers::StepLR fresh(fresh_optimizer, 2, 0.1);
    state.erase("last_lr_0");
    EXPECT_THROW(fresh.load_state_dict(state), std::invalid_argument);
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

    auto incomplete = state;
    incomplete.erase("best");
    EXPECT_THROW(restored.load_state_dict(incomplete), std::invalid_argument);
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
    EXPECT_THROW(
        optim::schedulers::ReduceLROnPlateau(
            optimizer_, {.factor = std::numeric_limits<double>::quiet_NaN()}),
        std::invalid_argument);
    optim::schedulers::ReduceLROnPlateau scheduler(optimizer_);
    EXPECT_THROW(
        scheduler.step(std::numeric_limits<double>::quiet_NaN()),
        std::invalid_argument);
}

TEST_F(NewSchedulersTest, CosineWarmRestartsRestartsAtCycleBoundary) {
    optim::schedulers::CosineAnnealingWarmRestarts scheduler(optimizer_, 3, 2, 0.0);
    constexpr double pi = 3.14159265358979323846;
    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), base_lr_ * (1.0 + std::cos(pi / 3.0)) / 2.0, 1e-12);
    scheduler.step();
    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), base_lr_, 1e-12);
    EXPECT_EQ(scheduler.current_cycle(), 1);
    EXPECT_EQ(scheduler.cycle_step(), 0);
    scheduler.step();
    EXPECT_EQ(scheduler.cycle_step(), 1);
}

TEST_F(NewSchedulersTest, OneCycleLRSupportsCosineAndMomentum) {
    optim::schedulers::OneCycleLR scheduler(
        optimizer_, 0.2, 10, 0.3, "cos", 2.0, 10.0, false, true, 0.8, 0.95);
    EXPECT_NEAR(scheduler.get_last_lr(), 0.1, 1e-12);
    EXPECT_NEAR(optimizer_->param_groups().front().options.at("momentum"), 0.95, 1e-12);
    scheduler.step();
    EXPECT_GT(scheduler.get_last_lr(), 0.1);
    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.2, 1e-12);
    for (int i = 0; i < 7; ++i) scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.2 / 20.0, 1e-12);
    EXPECT_NEAR(optimizer_->param_groups().front().options.at("momentum"), 0.95, 1e-12);
    scheduler.step();
    const double pct = 8.0 / 7.0;
    const double expected = 0.01 + (0.2 - 0.01) *
                                       (1.0 + std::cos(3.14159265358979323846 * pct)) /
                                       2.0;
    EXPECT_NEAR(scheduler.get_last_lr(), expected, 1e-12);
    const double expected_momentum =
        0.95 + (0.8 - 0.95) *
                   (1.0 + std::cos(3.14159265358979323846 * pct)) / 2.0;
    EXPECT_NEAR(optimizer_->param_groups().front().options.at("momentum"),
                expected_momentum, 1e-12);
    EXPECT_THROW(scheduler.step(), std::out_of_range);
}

TEST_F(NewSchedulersTest, OneCycleLRLinearFinalStepMatchesPyTorchExtrapolation) {
    optim::schedulers::OneCycleLR scheduler(
        optimizer_, 0.2, 10, 0.3, "linear", 2.0, 10.0, false, false);
    for (int i = 0; i < 10; ++i) scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), -0.017142857142857154, 1e-12);
    EXPECT_DOUBLE_EQ(optimizer_->param_groups().front().options.at("lr"),
                     scheduler.get_last_lr());

    auto restored_optimizer =
        std::make_shared<optim::SGD>(params_, optim::SGDOptions{.lr = base_lr_});
    optim::schedulers::OneCycleLR restored(
        restored_optimizer, 0.2, 10, 0.3, "linear", 2.0, 10.0, false, false);
    restored.load_state_dict(scheduler.state_dict());
    EXPECT_EQ(restored.get_step(), scheduler.get_step());
    EXPECT_DOUBLE_EQ(restored.get_last_lr(), scheduler.get_last_lr());
    EXPECT_DOUBLE_EQ(restored_optimizer->param_groups().front().options.at("lr"),
                     scheduler.get_last_lr());
}

TEST_F(NewSchedulersTest, OneCycleLRThreePhaseMatchesPyTorchLinearSchedule) {
    optim::schedulers::OneCycleLR scheduler(
        optimizer_, 0.2, 10, 0.3, "linear", 2.0, 10.0, true, true, 0.8, 0.95);

    const std::vector<double> expected_lrs{
        0.15, 0.20, 0.15, 0.10, 0.082, 0.064, 0.046, 0.028, 0.010};
    const std::vector<double> expected_momentums{
        0.875, 0.8, 0.875, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95};
    for (size_t i = 0; i < expected_lrs.size(); ++i) {
        scheduler.step();
        EXPECT_NEAR(scheduler.get_last_lr(), expected_lrs[i], 1e-12) << i;
        EXPECT_NEAR(optimizer_->param_groups().front().options.at("momentum"),
                    expected_momentums[i], 1e-12)
            << i;
    }
}

TEST_F(NewSchedulersTest, OneCycleLRValidatesAndRestoresState) {
    EXPECT_THROW(optim::schedulers::OneCycleLR(optimizer_, 0.1, 0), std::invalid_argument);
    EXPECT_THROW(optim::schedulers::OneCycleLR(optimizer_, 0.1, 5, -0.1), std::invalid_argument);
    EXPECT_THROW(optim::schedulers::OneCycleLR(optimizer_, 0.1, 5, 0.3, "bad"), std::invalid_argument);
    optim::schedulers::OneCycleLR source(optimizer_, 0.2, 5, 0.3, "linear", 2.0, 10.0, false, false);
    source.step();
    auto restored_optimizer = std::make_shared<optim::SGD>(params_, optim::SGDOptions{.lr = base_lr_});
    optim::schedulers::OneCycleLR restored(restored_optimizer, 0.2, 5, 0.3, "linear", 2.0, 10.0, false, false);
    restored.load_state_dict(source.state_dict());
    EXPECT_EQ(restored.get_step(), source.get_step());
    EXPECT_DOUBLE_EQ(restored.get_last_lr(), source.get_last_lr());
}

} // namespace
