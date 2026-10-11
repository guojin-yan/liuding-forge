// Unit tests for jyppx::optim::SGD and the shared Optimizer base behavior.

#include <jyppx/optim/sgd.hpp>

#include <torch/optim/sgd.h>

#include "test_utils/jyppx_test_helpers.hpp"

#include <gtest/gtest.h>

namespace {

using namespace jyppx;
using namespace jyppx_test;

TEST(SGDTest, VanillaSgdMatchesTorch) {
    torch::manual_seed(42);
    auto jyppx_params = make_params({{4, 3}, {3}});
    torch::manual_seed(42);
    auto torch_params = make_params({{4, 3}, {3}});

    optim::SGD jyppx_optimizer(
        jyppx_params, optim::SGDOptions{.lr = 0.05, .momentum = 0.0, .weight_decay = 0.0});
    torch::optim::SGD torch_optimizer(
        torch_params, torch::optim::SGDOptions(0.05));

    expect_optimizer_parity(jyppx_optimizer, jyppx_params, torch_optimizer, torch_params, 10);
}

TEST(SGDTest, MomentumMatchesTorch) {
    torch::manual_seed(7);
    auto jyppx_params = make_params({{6}, {2, 2}});
    torch::manual_seed(7);
    auto torch_params = make_params({{6}, {2, 2}});

    optim::SGD jyppx_optimizer(
        jyppx_params,
        optim::SGDOptions{.lr = 0.03, .momentum = 0.9, .dampening = 0.0, .weight_decay = 0.0});
    torch::optim::SGD torch_optimizer(
        torch_params, torch::optim::SGDOptions(0.03).momentum(0.9));

    expect_optimizer_parity(jyppx_optimizer, jyppx_params, torch_optimizer, torch_params, 10);
}

TEST(SGDTest, WeightDecayAndNesterovMatchTorch) {
    torch::manual_seed(11);
    auto jyppx_params = make_params({{3, 3}, {3}});
    torch::manual_seed(11);
    auto torch_params = make_params({{3, 3}, {3}});

    optim::SGD jyppx_optimizer(
        jyppx_params,
        optim::SGDOptions{
            .lr = 0.02, .momentum = 0.9, .dampening = 0.0, .weight_decay = 0.01,
            .nesterov = true});
    torch::optim::SGD torch_optimizer(
        torch_params,
        torch::optim::SGDOptions(0.02).momentum(0.9).weight_decay(0.01).nesterov(true));

    expect_optimizer_parity(jyppx_optimizer, jyppx_params, torch_optimizer, torch_params, 10);
}

TEST(SGDTest, EmptyParameterListThrows) {
    EXPECT_THROW(optim::SGD({}, optim::SGDOptions{.lr = 0.01}), std::invalid_argument);
}

TEST(SGDTest, InvalidOptionsThrow) {
    torch::manual_seed(1);
    auto params = make_params({{2}});
    EXPECT_THROW(optim::SGD(params, optim::SGDOptions{.lr = -0.1}), std::invalid_argument);
    // Nesterov requires momentum and no dampening.
    EXPECT_THROW(
        optim::SGD(params, optim::SGDOptions{.lr = 0.1, .momentum = 0.0, .nesterov = true}),
        std::invalid_argument);
}

TEST(OptimizerTest, ZeroGradZeroesAndClearsGradients) {
    torch::manual_seed(3);
    auto params = make_params({{2, 2}});
    optim::SGD optimizer(params, optim::SGDOptions{.lr = 0.01});

    quadratic_loss(params).backward();
    ASSERT_TRUE(params[0].grad().defined());

    optimizer.zero_grad(/*set_to_none=*/false);
    EXPECT_TRUE(params[0].grad().defined());
    EXPECT_TRUE(torch::equal(params[0].grad(), torch::zeros_like(params[0])));

    optimizer.zero_grad(/*set_to_none=*/true);
    EXPECT_FALSE(params[0].grad().defined());
}

TEST(OptimizerTest, LearningRateAccessors) {
    torch::manual_seed(4);
    auto params = make_params({{2}});
    optim::SGD optimizer(params, optim::SGDOptions{.lr = 0.5});

    const auto lrs = optimizer.get_lrs();
    ASSERT_EQ(lrs.size(), 1);
    EXPECT_DOUBLE_EQ(lrs[0], 0.5);

    optimizer.set_lr(0.25);
    EXPECT_DOUBLE_EQ(optimizer.get_lrs()[0], 0.25);

    optimizer.set_lrs({0.125});
    EXPECT_DOUBLE_EQ(optimizer.get_lrs()[0], 0.125);
    EXPECT_THROW(optimizer.set_lrs({0.1, 0.2}), std::invalid_argument);
    EXPECT_THROW(optimizer.set_lr(-1.0), std::invalid_argument);
}

TEST(OptimizerTest, StepCountIncrements) {
    torch::manual_seed(5);
    auto params = make_params({{2}});
    optim::SGD optimizer(params, optim::SGDOptions{.lr = 0.01});

    for (int i = 0; i < 4; ++i) {
        optimizer.zero_grad();
        quadratic_loss(params).backward();
        optimizer.step();
    }
    EXPECT_EQ(optimizer.step_count(), 4);
}

TEST(SGDTest, StateDictRestoresMomentumAndStepCount) {
    torch::manual_seed(21);
    auto params_a = make_params({{3}});
    optim::SGD continuous(
        params_a, optim::SGDOptions{.lr = 0.05, .momentum = 0.9});

    for (int i = 0; i < 3; ++i) {
        continuous.zero_grad();
        quadratic_loss(params_a).backward();
        continuous.step();
    }
    ASSERT_EQ(continuous.step_count(), 3);

    // Fork the run: resume from the same weights and optimizer state.
    auto params_b = clone_leaf_params(params_a);
    optim::SGD resumed(params_b, optim::SGDOptions{.lr = 0.05, .momentum = 0.9});
    resumed.load_state_dict(continuous.state_dict());
    ASSERT_EQ(resumed.step_count(), 3);

    for (int i = 0; i < 5; ++i) {
        continuous.zero_grad();
        quadratic_loss(params_a).backward();
        continuous.step();
        resumed.zero_grad();
        quadratic_loss(params_b).backward();
        resumed.step();
    }
    expect_tensor_lists_close(params_a, params_b);
}

} // namespace
