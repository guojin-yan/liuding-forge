// Unit tests for jyppx::nn::init.

#include <jyppx/nn/init.hpp>

#include <torch/nn/init.h>

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

namespace {

using namespace jyppx;

// Delegation must reproduce torch::nn::init bit-for-bit when both draw from
// the same seeded generator stream.
TEST(InitParityTest, MatchesTorchInitWithSameSeed) {
    const auto run_pair = [](auto&& jyppx_fn, auto&& torch_fn) {
        torch::Tensor ours = torch::empty({64, 32});
        torch::Tensor theirs = torch::empty({64, 32});
        torch::manual_seed(11);
        jyppx_fn(ours);
        torch::manual_seed(11);
        torch_fn(theirs);
        return torch::equal(ours, theirs);
    };

    EXPECT_TRUE(run_pair(
        [](Tensor& t) { nn::init::uniform_(t, -0.5, 0.5); },
        [](Tensor t) { torch::nn::init::uniform_(t, -0.5, 0.5); }));
    EXPECT_TRUE(run_pair(
        [](Tensor& t) { nn::init::normal_(t, 1.0, 2.0); },
        [](Tensor t) { torch::nn::init::normal_(t, 1.0, 2.0); }));
    EXPECT_TRUE(run_pair(
        [](Tensor& t) { nn::init::xavier_uniform_(t, 1.5); },
        [](Tensor t) { torch::nn::init::xavier_uniform_(t, 1.5); }));
    EXPECT_TRUE(run_pair(
        [](Tensor& t) { nn::init::xavier_normal_(t, 1.5); },
        [](Tensor t) { torch::nn::init::xavier_normal_(t, 1.5); }));
    EXPECT_TRUE(run_pair(
        [](Tensor& t) { nn::init::kaiming_uniform_(t, 0.0, "fan_in"); },
        [](Tensor t) { torch::nn::init::kaiming_uniform_(t, 0.0, torch::kFanIn, torch::kLeakyReLU); }));
    EXPECT_TRUE(run_pair(
        [](Tensor& t) { nn::init::kaiming_normal_(t, 0.0, "fan_out"); },
        [](Tensor t) { torch::nn::init::kaiming_normal_(t, 0.0, torch::kFanOut, torch::kLeakyReLU); }));
}

TEST(InitBasicTest, ConstantZerosOnesFillExactly) {
    auto tensor = torch::zeros({3, 4});

    nn::init::constant_(tensor, 2.5);
    EXPECT_TRUE(torch::equal(tensor, torch::full({3, 4}, 2.5)));

    nn::init::ones_(tensor);
    EXPECT_TRUE(torch::equal(tensor, torch::ones({3, 4})));

    nn::init::zeros_(tensor);
    EXPECT_TRUE(torch::equal(tensor, torch::zeros({3, 4})));
}

TEST(InitBasicTest, UniformRespectsBounds) {
    auto tensor = torch::empty({1000}, torch::TensorOptions().dtype(torch::kFloat64));
    nn::init::uniform_(tensor, -1.0, 3.0);

    const double min = tensor.min().item<double>();
    const double max = tensor.max().item<double>();
    EXPECT_GE(min, -1.0);
    EXPECT_LT(max, 3.0);
    EXPECT_NEAR((tensor.mean().item<double>()), 1.0, 0.1); // midpoint of [-1, 3]
}

TEST(InitBasicTest, NormalMatchesRequestedStatistics) {
    auto tensor = torch::empty({200000}, torch::TensorOptions().dtype(torch::kFloat64));
    nn::init::normal_(tensor, 2.0, 0.5);

    const double mean = tensor.mean().item<double>();
    const double std = tensor.std().item<double>();
    EXPECT_NEAR(mean, 2.0, 0.02);
    EXPECT_NEAR(std, 0.5, 0.02);
}

TEST(InitXavierTest, UniformBoundFollowsFanFormula) {
    // {64, 32}: fan_in=32, fan_out=64 -> bound = gain*sqrt(6/96).
    auto tensor = torch::empty({64, 32}, torch::TensorOptions().dtype(torch::kFloat64));
    const double gain = 1.5;
    const double bound = gain * std::sqrt(6.0 / (32.0 + 64.0));

    nn::init::xavier_uniform_(tensor, gain);
    EXPECT_LE(tensor.abs().max().item<double>(), bound * (1.0 + 1e-6));
    EXPECT_LT(tensor.min().item<double>(), 0.0); // spans the full range
    EXPECT_NEAR(tensor.mean().item<double>(), 0.0, 0.01);
}

TEST(InitXavierTest, NormalStdFollowsFanFormula) {
    auto tensor = torch::empty({64, 64}, torch::TensorOptions().dtype(torch::kFloat64));
    const double gain = 1.0;
    const double expected_std = gain * std::sqrt(2.0 / (64.0 + 64.0));

    nn::init::xavier_normal_(tensor, gain);
    EXPECT_NEAR(tensor.std().item<double>(), expected_std, 0.01);
    EXPECT_NEAR(tensor.mean().item<double>(), 0.0, 0.02);
}

TEST(InitKaimingTest, UniformBoundFollowsFanFormula) {
    // a=0 -> leaky_relu gain = sqrt(2) (torch calculate_gain); {64, 32}
    // fan_in=32 -> bound = sqrt(3)*sqrt(2)/sqrt(32) = sqrt(6/32).
    auto tensor = torch::empty({64, 32}, torch::TensorOptions().dtype(torch::kFloat64));
    const double bound = std::sqrt(6.0 / 32.0);

    nn::init::kaiming_uniform_(tensor, 0.0, "fan_in");
    EXPECT_LE(tensor.abs().max().item<double>(), bound * (1.0 + 1e-6));

    // fan_out mode uses dimension 0 instead.
    auto fan_out_tensor = torch::empty({64, 32}, torch::TensorOptions().dtype(torch::kFloat64));
    const double fan_out_bound = std::sqrt(6.0 / 64.0);
    nn::init::kaiming_uniform_(fan_out_tensor, 0.0, "fan_out");
    EXPECT_LE(fan_out_tensor.abs().max().item<double>(), fan_out_bound * (1.0 + 1e-6));
}

TEST(InitKaimingTest, NonZeroAGainShrinksTheBound) {
    auto tensor = torch::empty({64, 32}, torch::TensorOptions().dtype(torch::kFloat64));
    // a = sqrt(2) -> leaky_relu gain = sqrt(2/(1+2)); bound = sqrt(3)*gain/sqrt(32).
    const double gain = std::sqrt(2.0 / 3.0);
    const double bound = std::sqrt(3.0) * gain / std::sqrt(32.0);

    nn::init::kaiming_uniform_(tensor, std::sqrt(2.0), "fan_in");
    EXPECT_LE(tensor.abs().max().item<double>(), bound * (1.0 + 1e-6));
}

TEST(InitKaimingTest, NormalStdFollowsFanFormula) {
    // {8, 16, 3, 3} conv weight: fan_in = 16 * 9 = 144 -> std = sqrt(2/144).
    auto tensor =
        torch::empty({8, 16, 3, 3}, torch::TensorOptions().dtype(torch::kFloat64));
    const double expected_std = std::sqrt(2.0 / 144.0);

    nn::init::kaiming_normal_(tensor, 0.0, "fan_in");
    EXPECT_NEAR(tensor.std().item<double>(), expected_std, 0.005);
}

TEST(InitKaimingTest, OneDimensionalTensorIsRejectedLikeTorchCxx) {
    // Unlike torch.nn.init in Python (fan = numel for 1D), the torch::nn C++
    // implementation requires at least two dimensions to compute fans.
    auto tensor = torch::empty({100}, torch::TensorOptions().dtype(torch::kFloat64));
    EXPECT_THROW(nn::init::kaiming_uniform_(tensor, 0.0, "fan_in"), std::exception);
}

TEST(InitKaimingTest, InvalidModeThrows) {
    auto tensor = torch::empty({4, 4});
    EXPECT_THROW(nn::init::kaiming_uniform_(tensor, 0.0, "sideways"), std::invalid_argument);
    EXPECT_THROW(nn::init::kaiming_normal_(tensor, 0.0, ""), std::invalid_argument);
}

TEST(InitTest, WorksOnRequiresGradParameters) {
    // Init functions are used on freshly created module weights; they must not
    // trip autograd's in-place checks nor record gradients.
    auto tensor = torch::empty({8, 8}, torch::TensorOptions().requires_grad(true));
    nn::init::constant_(tensor, 0.5);
    EXPECT_TRUE(torch::equal(tensor, torch::full({8, 8}, 0.5)));
    EXPECT_FALSE(tensor.grad().defined());
}

} // namespace
