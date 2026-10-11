// Unit tests for jyppx::optim::Adam, including parity against torch::optim.

#include <jyppx/optim/adam.hpp>

#include <torch/optim/adam.h>

#include "test_utils/jyppx_test_helpers.hpp"

#include <gtest/gtest.h>

namespace {

using namespace jyppx;
using namespace jyppx_test;

TEST(AdamTest, DefaultsMatchTorch) {
    torch::manual_seed(42);
    auto jyppx_params = make_params({{4, 3}, {3}});
    torch::manual_seed(42);
    auto torch_params = make_params({{4, 3}, {3}});

    optim::Adam jyppx_optimizer(jyppx_params, optim::AdamOptions{.lr = 0.01});
    torch::optim::Adam torch_optimizer(
        torch_params, torch::optim::AdamOptions(0.01));

    expect_optimizer_parity(jyppx_optimizer, jyppx_params, torch_optimizer, torch_params, 10);
}

TEST(AdamTest, CustomBetasWeightDecayMatchTorch) {
    torch::manual_seed(9);
    auto jyppx_params = make_params({{5}, {2, 2}});
    torch::manual_seed(9);
    auto torch_params = make_params({{5}, {2, 2}});

    optim::Adam jyppx_optimizer(
        jyppx_params,
        optim::AdamOptions{
            .lr = 0.02, .beta1 = 0.85, .beta2 = 0.995, .eps = 1e-8, .weight_decay = 0.005});
    torch::optim::Adam torch_optimizer(
        torch_params,
        torch::optim::AdamOptions(0.02)
            .betas({0.85, 0.995})
            .weight_decay(0.005));

    expect_optimizer_parity(jyppx_optimizer, jyppx_params, torch_optimizer, torch_params, 10);
}

TEST(AdamTest, AmsgradMatchesTorch) {
    torch::manual_seed(13);
    auto jyppx_params = make_params({{3, 2}});
    torch::manual_seed(13);
    auto torch_params = make_params({{3, 2}});

    optim::Adam jyppx_optimizer(
        jyppx_params, optim::AdamOptions{.lr = 0.01, .amsgrad = true});
    torch::optim::Adam torch_optimizer(
        torch_params, torch::optim::AdamOptions(0.01).amsgrad(true));

    expect_optimizer_parity(jyppx_optimizer, jyppx_params, torch_optimizer, torch_params, 10);
}

TEST(AdamTest, InvalidOptionsThrow) {
    torch::manual_seed(2);
    auto params = make_params({{2}});
    EXPECT_THROW(optim::Adam(params, optim::AdamOptions{.lr = -0.1}), std::invalid_argument);
    EXPECT_THROW(optim::Adam(params, optim::AdamOptions{.beta1 = 1.0}), std::invalid_argument);
    EXPECT_THROW(optim::Adam(params, optim::AdamOptions{.eps = 0.0}), std::invalid_argument);
}

TEST(AdamTest, EmptyParameterListThrows) {
    EXPECT_THROW(optim::Adam({}, optim::AdamOptions{}), std::invalid_argument);
}

} // namespace
