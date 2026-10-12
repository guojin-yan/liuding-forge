// NaN/Inf handling tests for AMP gradient scaling.

#include <jyppx/core/tensor_utils.hpp>
#include <jyppx/optim/sgd.hpp>
#include <jyppx/training/amp/grad_scaler.hpp>

#include <gtest/gtest.h>

#include <limits>
#include <memory>

namespace {

using jyppx::Tensor;
using jyppx::optim::SGD;
using jyppx::optim::SGDOptions;
using jyppx::training::amp::GradScaler;
using jyppx::training::amp::GradScalerOptions;

TEST(NaNInfTest, NaNGradientIsDetectedAndOptimizerStepIsSkipped) {
    auto parameter = torch::ones({2}, torch::TensorOptions().dtype(torch::kFloat64)
                                             .requires_grad(true));
    const auto before = parameter.clone();
    parameter.mutable_grad() = torch::tensor(
        {std::numeric_limits<double>::quiet_NaN(), 1.0},
        torch::TensorOptions().dtype(torch::kFloat64));
    auto optimizer = std::make_shared<SGD>(
        jyppx::TensorList{parameter}, SGDOptions{.lr = 0.25});
    GradScaler scaler({.init_scale = 8.0, .backoff_factor = 0.5});

    scaler.step(optimizer);
    EXPECT_TRUE(scaler.found_inf());
    EXPECT_DOUBLE_EQ(scaler.get_scale(), 4.0);
    EXPECT_TRUE(torch::equal(parameter, before));
    EXPECT_FALSE(jyppx::is_finite(parameter.grad()));
}

TEST(NaNInfTest, InfGradientIsDetectedAndScaleBacksOff) {
    auto parameter = torch::ones({2}, torch::TensorOptions().dtype(torch::kFloat64)
                                             .requires_grad(true));
    parameter.mutable_grad() = torch::tensor(
        {std::numeric_limits<double>::infinity(), -1.0},
        torch::TensorOptions().dtype(torch::kFloat64));
    auto optimizer = std::make_shared<SGD>(jyppx::TensorList{parameter});
    GradScaler scaler({.init_scale = 16.0, .backoff_factor = 0.25});

    scaler.step(optimizer);
    EXPECT_TRUE(scaler.found_inf());
    EXPECT_DOUBLE_EQ(scaler.get_scale(), 4.0);
}

TEST(NaNInfTest, FiniteGradientUpdatesParameterAndGrowsAfterInterval) {
    auto parameter = torch::ones({2}, torch::TensorOptions().dtype(torch::kFloat64)
                                             .requires_grad(true));
    parameter.mutable_grad() = torch::full_like(parameter, 4.0);
    auto optimizer = std::make_shared<SGD>(jyppx::TensorList{parameter},
                                            SGDOptions{.lr = 0.25});
    GradScaler scaler({.init_scale = 2.0, .growth_factor = 2.0, .growth_interval = 1});

    scaler.step(optimizer);
    scaler.update();
    EXPECT_FALSE(scaler.found_inf());
    EXPECT_DOUBLE_EQ(scaler.get_scale(), 4.0);
    EXPECT_TRUE(jyppx::is_finite(parameter));
    EXPECT_TRUE(torch::allclose(parameter, torch::full({2}, 0.5, parameter.options())));
}

} // namespace
