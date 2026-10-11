// Unit tests for jyppx::optim::AdamW, including parity against torch::optim.

#include <jyppx/optim/adamw.hpp>

#include <torch/optim/adamw.h>

#include "test_utils/jyppx_test_helpers.hpp"

#include <gtest/gtest.h>

namespace {

using namespace jyppx;
using namespace jyppx_test;

TEST(AdamWTest, DefaultsMatchTorch) {
    torch::manual_seed(42);
    auto jyppx_params = make_params({{4, 3}, {3}});
    torch::manual_seed(42);
    auto torch_params = make_params({{4, 3}, {3}});

    optim::AdamW jyppx_optimizer(jyppx_params, optim::AdamWOptions{.lr = 0.01});
    torch::optim::AdamW torch_optimizer(
        torch_params, torch::optim::AdamWOptions(0.01));

    expect_optimizer_parity(jyppx_optimizer, jyppx_params, torch_optimizer, torch_params, 10);
}

TEST(AdamWTest, DecoupledWeightDecayMatchesTorch) {
    torch::manual_seed(17);
    auto jyppx_params = make_params({{3, 3}, {3}});
    torch::manual_seed(17);
    auto torch_params = make_params({{3, 3}, {3}});

    optim::AdamW jyppx_optimizer(
        jyppx_params,
        optim::AdamWOptions{
            .lr = 0.05, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-8, .weight_decay = 0.1});
    torch::optim::AdamW torch_optimizer(
        torch_params, torch::optim::AdamWOptions(0.05).weight_decay(0.1));

    expect_optimizer_parity(jyppx_optimizer, jyppx_params, torch_optimizer, torch_params, 10);
}

// AdamW must be distinguishable from Adam-with-L2: the decay applies to the
// parameter, not the gradient. A parameter whose gradient is zero still decays.
TEST(AdamWTest, DecaysParameterEvenWithoutGradientFlow) {
    torch::manual_seed(31);
    auto params = make_params({{4}});
    const double initial_norm = params[0].norm().item<double>();

    optim::AdamW optimizer(params, optim::AdamWOptions{.lr = 0.1, .weight_decay = 0.5});
    // Zero gradient: only the decoupled decay should act.
    optimizer.zero_grad();
    const auto zero_grad = torch::zeros_like(params[0]);
    params[0].mutable_grad() = zero_grad.clone();
    optimizer.step();

    const double decayed_norm = params[0].norm().item<double>();
    EXPECT_LT(decayed_norm, initial_norm);
    // Expected multiplicative decay: (1 - lr * weight_decay) = 0.95.
    EXPECT_NEAR(decayed_norm, initial_norm * 0.95, initial_norm * 1e-9);
}

TEST(AdamWTest, EmptyParameterListThrows) {
    EXPECT_THROW(optim::AdamW({}, optim::AdamWOptions{}), std::invalid_argument);
}

} // namespace
