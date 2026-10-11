// Unit tests for jyppx::optim::RMSprop and jyppx::optim::Adagrad, including
// parity against torch::optim.

#include <jyppx/optim/adagrad.hpp>
#include <jyppx/optim/rmsprop.hpp>

#include <torch/optim/adagrad.h>
#include <torch/optim/rmsprop.h>

#include "test_utils/jyppx_test_helpers.hpp"

#include <gtest/gtest.h>

#include <sstream>

namespace {

using namespace jyppx;
using namespace jyppx_test;

TEST(RMSpropTest, DefaultsMatchTorch) {
    torch::manual_seed(42);
    auto jyppx_params = make_params({{4, 3}, {3}});
    torch::manual_seed(42);
    auto torch_params = make_params({{4, 3}, {3}});

    optim::RMSprop jyppx_optimizer(jyppx_params, optim::RMSpropOptions{.lr = 0.01});
    torch::optim::RMSprop torch_optimizer(
        torch_params, torch::optim::RMSpropOptions(0.01));

    expect_optimizer_parity(jyppx_optimizer, jyppx_params, torch_optimizer, torch_params, 10);
}

TEST(RMSpropTest, AlphaMomentumCenteredMatchTorch) {
    torch::manual_seed(9);
    auto jyppx_params = make_params({{5}, {2, 2}});
    torch::manual_seed(9);
    auto torch_params = make_params({{5}, {2, 2}});

    optim::RMSprop jyppx_optimizer(
        jyppx_params,
        optim::RMSpropOptions{
            .lr = 0.02, .alpha = 0.9, .eps = 1e-8, .weight_decay = 0.005,
            .momentum = 0.5, .centered = true});
    torch::optim::RMSprop torch_optimizer(
        torch_params,
        torch::optim::RMSpropOptions(0.02)
            .alpha(0.9)
            .eps(1e-8)
            .weight_decay(0.005)
            .momentum(0.5)
            .centered(true));

    expect_optimizer_parity(jyppx_optimizer, jyppx_params, torch_optimizer, torch_params, 10);
}

TEST(RMSpropTest, MomentumOnlyMatchesTorch) {
    torch::manual_seed(19);
    auto jyppx_params = make_params({{5}, {2, 2}});
    torch::manual_seed(19);
    auto torch_params = make_params({{5}, {2, 2}});

    optim::RMSprop jyppx_optimizer(
        jyppx_params, optim::RMSpropOptions{.lr = 0.02, .momentum = 0.5});
    torch::optim::RMSprop torch_optimizer(
        torch_params, torch::optim::RMSpropOptions(0.02).momentum(0.5));

    expect_optimizer_parity(jyppx_optimizer, jyppx_params, torch_optimizer, torch_params, 10);
}

TEST(RMSpropTest, CenteredOnlyMatchesTorch) {
    torch::manual_seed(23);
    auto jyppx_params = make_params({{5}, {2, 2}});
    torch::manual_seed(23);
    auto torch_params = make_params({{5}, {2, 2}});

    optim::RMSprop jyppx_optimizer(
        jyppx_params, optim::RMSpropOptions{.lr = 0.02, .alpha = 0.9, .centered = true});
    torch::optim::RMSprop torch_optimizer(
        torch_params, torch::optim::RMSpropOptions(0.02).alpha(0.9).centered(true));

    expect_optimizer_parity(jyppx_optimizer, jyppx_params, torch_optimizer, torch_params, 10);
}

TEST(RMSpropTest, InvalidOptionsThrow) {
    torch::manual_seed(3);
    auto params = make_params({{2}});
    EXPECT_THROW(optim::RMSprop(params, optim::RMSpropOptions{.lr = -0.1}), std::invalid_argument);
    EXPECT_THROW(optim::RMSprop(params, optim::RMSpropOptions{.alpha = 1.5}), std::invalid_argument);
    EXPECT_THROW(optim::RMSprop(params, optim::RMSpropOptions{.eps = 0.0}), std::invalid_argument);
    EXPECT_THROW(optim::RMSprop(params, optim::RMSpropOptions{.momentum = -1.0}), std::invalid_argument);
}

TEST(AdagradTest, DefaultsMatchTorch) {
    torch::manual_seed(42);
    auto jyppx_params = make_params({{4, 3}, {3}});
    torch::manual_seed(42);
    auto torch_params = make_params({{4, 3}, {3}});

    optim::Adagrad jyppx_optimizer(jyppx_params, optim::AdagradOptions{.lr = 0.05});
    torch::optim::Adagrad torch_optimizer(torch_params, torch::optim::AdagradOptions(0.05));

    expect_optimizer_parity(jyppx_optimizer, jyppx_params, torch_optimizer, torch_params, 10);
}

TEST(AdagradTest, DecayAndWeightDecayMatchTorch) {
    torch::manual_seed(13);
    auto jyppx_params = make_params({{3, 3}, {3}});
    torch::manual_seed(13);
    auto torch_params = make_params({{3, 3}, {3}});

    optim::Adagrad jyppx_optimizer(
        jyppx_params,
        optim::AdagradOptions{.lr = 0.1, .lr_decay = 0.01, .weight_decay = 0.01, .eps = 1e-10});
    torch::optim::Adagrad torch_optimizer(
        torch_params,
        torch::optim::AdagradOptions(0.1).lr_decay(0.01).weight_decay(0.01).eps(1e-10));

    expect_optimizer_parity(jyppx_optimizer, jyppx_params, torch_optimizer, torch_params, 10);
}

TEST(AdagradTest, InvalidOptionsThrow) {
    torch::manual_seed(4);
    auto params = make_params({{2}});
    EXPECT_THROW(optim::Adagrad(params, optim::AdagradOptions{.lr = -0.1}), std::invalid_argument);
    EXPECT_THROW(optim::Adagrad(params, optim::AdagradOptions{.lr_decay = -0.5}), std::invalid_argument);
    EXPECT_THROW(optim::Adagrad(params, optim::AdagradOptions{.eps = 0.0}), std::invalid_argument);
}

} // namespace


