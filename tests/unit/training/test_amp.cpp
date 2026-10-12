// Unit tests for jyppx::training::amp: GradScaler and the Trainer AMP path.

#include <jyppx/training/amp/grad_scaler.hpp>
#include <jyppx/training/trainer.hpp>

#include "test_utils/jyppx_test_helpers.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>

namespace {

using namespace jyppx;
using namespace jyppx_test;
using jyppx::training::amp::GradScaler;
using jyppx::training::amp::GradScalerOptions;

TEST(GradScalerTest, InitialScaleMatchesOptions) {
    GradScaler scaler;
    EXPECT_DOUBLE_EQ(scaler.get_scale(), 65536.0);

    GradScaler custom({.init_scale = 128.0});
    EXPECT_DOUBLE_EQ(custom.get_scale(), 128.0);
}

TEST(GradScalerTest, ScaleMultipliesLoss) {
    GradScaler scaler({.init_scale = 4.0});
    auto loss = torch::tensor(1.5, torch::TensorOptions().dtype(torch::kFloat32));
    EXPECT_NEAR(scaler.scale(loss).item<double>(), 6.0, 1e-6);
}

TEST(GradScalerTest, DisabledScalerPassesThrough) {
    GradScaler scaler({.init_scale = 4.0, .enabled = false});
    EXPECT_DOUBLE_EQ(scaler.get_scale(), 1.0);

    auto loss = torch::tensor(1.5, torch::TensorOptions().dtype(torch::kFloat32));
    EXPECT_NEAR(scaler.scale(loss).item<double>(), 1.5, 1e-6);

    // step() forwards to the optimizer with the raw gradient: no scaling, no
    // unscale, no overflow handling.
    torch::manual_seed(1);
    auto params = make_params({{2}});
    const auto initial = params[0].detach().clone();
    auto optimizer = std::make_shared<optim::SGD>(params, optim::SGDOptions{.lr = 0.1});
    params[0].mutable_grad() =
        torch::full({2}, 8.0, torch::TensorOptions().dtype(torch::kFloat64));
    scaler.step(optimizer);
    EXPECT_TRUE(torch::allclose(params[0], initial - 0.8, 1e-12, 1e-12));
}

TEST(GradScalerTest, StepUnscalesGradientsBeforeStepping) {
    torch::manual_seed(2);
    auto params = make_params({{4}});
    const auto initial = params[0].detach().clone();
    auto optimizer = std::make_shared<optim::SGD>(params, optim::SGDOptions{.lr = 0.1});

    GradScaler scaler({.init_scale = 8.0});
    // Simulate a scaled gradient of 8.0; unscale must divide it back to 1.0.
    params[0].mutable_grad() = torch::full({4}, 8.0, torch::TensorOptions().dtype(torch::kFloat64));
    scaler.step(optimizer);

    // param = initial - lr * (8 / 8)
    const auto expected = initial - 0.1;
    EXPECT_TRUE(torch::allclose(params[0], expected, 1e-12, 1e-12));
    EXPECT_FALSE(scaler.found_inf());
}

TEST(GradScalerTest, InfGradientSkipsStepAndBacksOff) {
    torch::manual_seed(3);
    auto params = make_params({{4}});
    const auto initial = params[0].detach().clone();
    auto optimizer = std::make_shared<optim::SGD>(params, optim::SGDOptions{.lr = 0.1});

    GradScaler scaler({.init_scale = 2.0, .backoff_factor = 0.5});
    params[0].mutable_grad() = torch::full(
        {4}, std::numeric_limits<double>::infinity(),
        torch::TensorOptions().dtype(torch::kFloat64));

    scaler.step(optimizer);
    EXPECT_TRUE(scaler.found_inf());
    EXPECT_TRUE(torch::equal(params[0], initial)); // update skipped
    EXPECT_DOUBLE_EQ(scaler.get_scale(), 1.0);     // 2.0 * 0.5
}

TEST(GradScalerTest, UnscaleBeforeClipAndStepIsIdempotent) {
    torch::manual_seed(4);
    auto params = make_params({{4}});
    const auto initial = params[0].detach().clone();
    auto optimizer = std::make_shared<optim::SGD>(params, optim::SGDOptions{.lr = 0.1});

    GradScaler scaler({.init_scale = 4.0});
    params[0].mutable_grad() =
        torch::full({4}, 4.0, torch::TensorOptions().dtype(torch::kFloat64));

    // Trainer-style flow: explicit unscale, then step() must not rescale.
    scaler.unscale(optimizer);
    EXPECT_NEAR(params[0].grad().max().item<double>(), 1.0, 1e-12);
    scaler.step(optimizer);

    const auto expected = initial - 0.1;
    EXPECT_TRUE(torch::allclose(params[0], expected, 1e-12, 1e-12));
}

TEST(GradScalerTest, UpdateGrowsScaleAfterInterval) {
    GradScaler scaler({.init_scale = 1.0, .growth_factor = 2.0, .growth_interval = 2});

    scaler.update();
    EXPECT_DOUBLE_EQ(scaler.get_scale(), 1.0); // interval not reached
    scaler.update();
    EXPECT_DOUBLE_EQ(scaler.get_scale(), 2.0); // growth applied

    // Successful steps keep counting; the tracker restarts after growth.
    scaler.update();
    scaler.update();
    EXPECT_DOUBLE_EQ(scaler.get_scale(), 4.0);
}

TEST(GradScalerTest, OverflowingStepsDoNotCountTowardGrowth) {
    torch::manual_seed(5);
    auto params = make_params({{2}});
    auto optimizer = std::make_shared<optim::SGD>(params, optim::SGDOptions{.lr = 0.1});

    GradScaler scaler({.init_scale = 1.0, .growth_interval = 2});

    // One overflowing step resets the tracker.
    params[0].mutable_grad() = torch::full(
        {2}, std::numeric_limits<double>::infinity(),
        torch::TensorOptions().dtype(torch::kFloat64));
    scaler.step(optimizer);
    scaler.update();
    EXPECT_DOUBLE_EQ(scaler.get_scale(), 0.5);
    EXPECT_EQ(scaler.state_dict().at("growth_tracker"), 0.0);

    // Two healthy steps reach the interval again.
    params[0].mutable_grad() = torch::ones_like(params[0]);
    scaler.step(optimizer);
    scaler.update();
    params[0].mutable_grad() = torch::ones_like(params[0]);
    scaler.step(optimizer);
    scaler.update();
    EXPECT_DOUBLE_EQ(scaler.get_scale(), 1.0); // 0.5 * 2.0 after two clean steps
}

TEST(GradScalerTest, StateDictRoundTrip) {
    GradScaler scaler({.init_scale = 100.0, .growth_interval = 3});
    scaler.update();
    scaler.update();

    GradScaler restored;
    restored.load_state_dict(scaler.state_dict());
    EXPECT_DOUBLE_EQ(restored.get_scale(), scaler.get_scale());
    EXPECT_EQ(
        restored.state_dict().at("growth_tracker"),
        scaler.state_dict().at("growth_tracker"));
}

TEST(GradScalerTest, InvalidOptionsThrow) {
    EXPECT_THROW(GradScaler({.init_scale = 0.0}), std::invalid_argument);
    EXPECT_THROW(GradScaler({.growth_factor = 1.0}), std::invalid_argument);
    EXPECT_THROW(GradScaler({.backoff_factor = 1.0}), std::invalid_argument);
    EXPECT_THROW(GradScaler({.growth_interval = 0}), std::invalid_argument);
}

TEST(GradScalerTest, NullOptimizerThrows) {
    GradScaler scaler;
    std::shared_ptr<optim::Optimizer> null_optimizer;
    EXPECT_THROW(scaler.unscale(null_optimizer), std::invalid_argument);
    EXPECT_THROW(scaler.step(null_optimizer), std::invalid_argument);
}

// ---- Trainer integration --------------------------------------------------

TEST(TrainerAMPTest, AmpOnCpuThrowsAtConstruction) {
    torch::manual_seed(42);
    auto model = std::make_shared<LinearModel>(2, 1);
    auto optimizer =
        std::make_shared<optim::Adam>(model->parameters(), optim::AdamOptions{});

    EXPECT_THROW(
        training::Trainer(
            model, optimizer, training::TrainerOptions{.use_amp = true}),
        std::runtime_error);
}

TEST(TrainerAMPTest, AmpFitsOnCudaAndReducesLoss) {
    if (!cuda_is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    const auto [inputs, targets] = make_linear_data(64);

    auto make_and_fit = [&](bool use_amp) {
        torch::manual_seed(2026);
        auto model = std::make_shared<LinearModel>(2, 1);
        auto optimizer = std::make_shared<optim::Adam>(
            model->parameters(), optim::AdamOptions{.lr = 0.1});
        training::Trainer trainer(
            model, optimizer,
            training::TrainerOptions{
                .max_epochs = 10,
                .device = torch::kCUDA,
                .use_amp = use_amp});
        auto loader = data::DataLoader<data::Example>(
            std::make_shared<jyppx::data::TensorDataset>(inputs, targets),
            data::DataLoaderOptions{.batch_size = 16});
        trainer.fit(loader);
        return std::make_pair(trainer.last_train_loss(), trainer.global_step());
    };

    const auto [loss_fp32, steps_fp32] = make_and_fit(false);
    const auto [loss_amp, steps_amp] = make_and_fit(true);

    EXPECT_EQ(steps_fp32, steps_amp);
    // Half-precision rounding makes the AMP trajectory diverge slightly from
    // the FP32 one; both must converge well below the initial loss (~30).
    EXPECT_LT(loss_amp, 0.2);
    EXPECT_LT(loss_fp32, 0.2);
    EXPECT_LT(std::abs(loss_amp - loss_fp32), 0.15);
}

} // namespace
