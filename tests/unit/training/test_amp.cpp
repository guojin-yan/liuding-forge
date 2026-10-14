// Unit tests for jyppx::training::amp: GradScaler and the Trainer AMP path.

#include "test_utils/jyppx_test_helpers.hpp"

#include <jyppx/training/amp/autocast.hpp>
#include <jyppx/training/amp/grad_scaler.hpp>
#include <jyppx/training/trainer.hpp>

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <memory>

namespace {

using namespace jyppx;
using namespace jyppx_test;
using jyppx::training::amp::AutocastGuard;
using jyppx::training::amp::GradScaler;
using jyppx::training::amp::GradScalerOptions;

TEST(AutocastGuardTest, RestoresCpuStateAndRejectsUnsupportedDtype) {
  const bool previous_enabled =
      at::autocast::is_autocast_enabled(c10::DeviceType::CPU);
  const auto previous_dtype =
      at::autocast::get_autocast_dtype(c10::DeviceType::CPU);
  at::autocast::set_autocast_enabled(c10::DeviceType::CPU, false);
  at::autocast::set_autocast_dtype(c10::DeviceType::CPU, at::kBFloat16);

  {
    AutocastGuard guard(c10::DeviceType::CPU, at::kBFloat16);
    EXPECT_TRUE(at::autocast::is_autocast_enabled(c10::DeviceType::CPU));
    EXPECT_EQ(at::autocast::get_autocast_dtype(c10::DeviceType::CPU),
              at::kBFloat16);
  }
  EXPECT_FALSE(at::autocast::is_autocast_enabled(c10::DeviceType::CPU));
  EXPECT_EQ(at::autocast::get_autocast_dtype(c10::DeviceType::CPU),
            at::kBFloat16);
  EXPECT_THROW(AutocastGuard(c10::DeviceType::CPU, at::kHalf),
               std::invalid_argument);

  at::autocast::set_autocast_dtype(c10::DeviceType::CPU, previous_dtype);
  at::autocast::set_autocast_enabled(c10::DeviceType::CPU, previous_enabled);
}

TEST(AutocastGuardCudaTest, NestedGuardRestoresOuterDtype) {
  if (!cuda_is_available()) {
    GTEST_SKIP() << "CUDA not available";
  }

  const bool previous_enabled =
      at::autocast::is_autocast_enabled(c10::DeviceType::CUDA);
  const auto previous_dtype =
      at::autocast::get_autocast_dtype(c10::DeviceType::CUDA);
  {
    AutocastGuard outer(c10::DeviceType::CUDA, at::kHalf);
    {
      AutocastGuard inner(c10::DeviceType::CUDA, at::kBFloat16);
      EXPECT_EQ(at::autocast::get_autocast_dtype(c10::DeviceType::CUDA),
                at::kBFloat16);
    }
    EXPECT_TRUE(at::autocast::is_autocast_enabled(c10::DeviceType::CUDA));
    EXPECT_EQ(at::autocast::get_autocast_dtype(c10::DeviceType::CUDA),
              at::kHalf);
  }
  EXPECT_EQ(at::autocast::is_autocast_enabled(c10::DeviceType::CUDA),
            previous_enabled);
  EXPECT_EQ(at::autocast::get_autocast_dtype(c10::DeviceType::CUDA),
            previous_dtype);
}

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
  auto optimizer =
      std::make_shared<optim::SGD>(params, optim::SGDOptions{.lr = 0.1});
  params[0].mutable_grad() =
      torch::full({2}, 8.0, torch::TensorOptions().dtype(torch::kFloat64));
  scaler.step(optimizer);
  EXPECT_TRUE(torch::allclose(params[0], initial - 0.8, 1e-12, 1e-12));
}

TEST(GradScalerTest, StepUnscalesGradientsBeforeStepping) {
  torch::manual_seed(2);
  auto params = make_params({{4}});
  const auto initial = params[0].detach().clone();
  auto optimizer =
      std::make_shared<optim::SGD>(params, optim::SGDOptions{.lr = 0.1});

  GradScaler scaler({.init_scale = 8.0});
  // Simulate a scaled gradient of 8.0; unscale must divide it back to 1.0.
  params[0].mutable_grad() =
      torch::full({4}, 8.0, torch::TensorOptions().dtype(torch::kFloat64));
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
  auto optimizer =
      std::make_shared<optim::SGD>(params, optim::SGDOptions{.lr = 0.1});

  GradScaler scaler({.init_scale = 2.0, .backoff_factor = 0.5});
  params[0].mutable_grad() =
      torch::full({4}, std::numeric_limits<double>::infinity(),
                  torch::TensorOptions().dtype(torch::kFloat64));

  scaler.step(optimizer);
  EXPECT_TRUE(scaler.found_inf());
  EXPECT_TRUE(torch::equal(params[0], initial));  // update skipped
  EXPECT_DOUBLE_EQ(scaler.get_scale(), 2.0);      // update owns scale changes
  scaler.update();
  EXPECT_DOUBLE_EQ(scaler.get_scale(), 1.0);  // 2.0 * 0.5
}

TEST(GradScalerTest, TracksMultipleOptimizersIndependently) {
  auto finite_params = make_params({{2}});
  auto overflow_params = make_params({{2}});
  const auto finite_initial = finite_params[0].detach().clone();
  const auto overflow_initial = overflow_params[0].detach().clone();
  auto finite_optimizer =
      std::make_shared<optim::SGD>(finite_params, optim::SGDOptions{.lr = 0.1});
  auto overflow_optimizer = std::make_shared<optim::SGD>(
      overflow_params, optim::SGDOptions{.lr = 0.1});

  GradScaler scaler({.init_scale = 8.0, .backoff_factor = 0.5});
  finite_params[0].mutable_grad() = torch::full_like(finite_params[0], 8.0);
  overflow_params[0].mutable_grad() = torch::full_like(
      overflow_params[0], std::numeric_limits<double>::infinity());

  scaler.step(finite_optimizer);
  scaler.step(overflow_optimizer);
  EXPECT_TRUE(
      torch::allclose(finite_params[0], finite_initial - 0.1, 1e-12, 1e-12));
  EXPECT_TRUE(torch::equal(overflow_params[0], overflow_initial));
  EXPECT_TRUE(scaler.found_inf());
  EXPECT_DOUBLE_EQ(scaler.get_scale(), 8.0);
  scaler.update();
  EXPECT_DOUBLE_EQ(scaler.get_scale(), 4.0);
}

TEST(GradScalerCudaTest, MultipleOptimizersUseRealCudaGradients) {
  if (!cuda_is_available()) {
    GTEST_SKIP() << "CUDA not available";
  }

  const auto options =
      torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
  std::vector<Tensor> finite_params{
      torch::ones({2}, options).requires_grad_(true)};
  std::vector<Tensor> overflow_params{
      torch::ones({2}, options).requires_grad_(true)};
  auto finite_optimizer =
      std::make_shared<optim::SGD>(finite_params, optim::SGDOptions{.lr = 0.1});
  auto overflow_optimizer = std::make_shared<optim::SGD>(
      overflow_params, optim::SGDOptions{.lr = 0.1});

  GradScaler scaler({.init_scale = 8.0, .backoff_factor = 0.5});
  finite_params[0].mutable_grad() = torch::full_like(finite_params[0], 8.0);
  overflow_params[0].mutable_grad() = torch::full_like(
      overflow_params[0], std::numeric_limits<float>::infinity());

  scaler.step(finite_optimizer);
  scaler.step(overflow_optimizer);
  scaler.update();

  EXPECT_TRUE(torch::allclose(finite_params[0].cpu(), torch::full({2}, 0.9f)));
  EXPECT_TRUE(torch::equal(overflow_params[0].cpu(), torch::ones({2})));
  EXPECT_DOUBLE_EQ(scaler.get_scale(), 4.0);
}

TEST(GradScalerTest, UnscaleBeforeClipAndStepIsIdempotent) {
  torch::manual_seed(4);
  auto params = make_params({{4}});
  const auto initial = params[0].detach().clone();
  auto optimizer =
      std::make_shared<optim::SGD>(params, optim::SGDOptions{.lr = 0.1});

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
  GradScaler scaler(
      {.init_scale = 1.0, .growth_factor = 2.0, .growth_interval = 2});
  auto params = make_params({{1}});
  auto optimizer =
      std::make_shared<optim::SGD>(params, optim::SGDOptions{.lr = 0.1});
  const auto successful_step = [&] {
    params[0].mutable_grad() = torch::ones_like(params[0]);
    scaler.step(optimizer);
    scaler.update();
  };

  successful_step();
  EXPECT_DOUBLE_EQ(scaler.get_scale(), 1.0);  // interval not reached
  successful_step();
  EXPECT_DOUBLE_EQ(scaler.get_scale(), 2.0);  // growth applied

  // Successful steps keep counting; the tracker restarts after growth.
  successful_step();
  successful_step();
  EXPECT_DOUBLE_EQ(scaler.get_scale(), 4.0);
}

TEST(GradScalerTest, OverflowingStepsDoNotCountTowardGrowth) {
  torch::manual_seed(5);
  auto params = make_params({{2}});
  auto optimizer =
      std::make_shared<optim::SGD>(params, optim::SGDOptions{.lr = 0.1});

  GradScaler scaler({.init_scale = 1.0, .growth_interval = 2});

  // One overflowing step resets the tracker.
  params[0].mutable_grad() =
      torch::full({2}, std::numeric_limits<double>::infinity(),
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
  EXPECT_DOUBLE_EQ(scaler.get_scale(), 1.0);  // 0.5 * 2.0 after two clean steps
}

TEST(GradScalerTest, StateDictRoundTrip) {
  GradScaler scaler({.init_scale = 100.0, .growth_interval = 3});
  auto params = make_params({{1}});
  auto optimizer = std::make_shared<optim::SGD>(params);
  for (int i = 0; i < 2; ++i) {
    params[0].mutable_grad() = torch::ones_like(params[0]);
    scaler.step(optimizer);
    scaler.update();
  }

  GradScaler restored({.init_scale = 100.0, .growth_interval = 3});
  restored.load_state_dict(scaler.state_dict());
  EXPECT_DOUBLE_EQ(restored.get_scale(), scaler.get_scale());
  EXPECT_EQ(restored.state_dict().at("growth_tracker"),
            scaler.state_dict().at("growth_tracker"));
}

TEST(GradScalerTest, InvalidOptionsThrow) {
  EXPECT_THROW(GradScaler({.init_scale = 0.0}), std::invalid_argument);
  EXPECT_THROW(GradScaler({.growth_factor = 1.0}), std::invalid_argument);
  EXPECT_THROW(GradScaler({.backoff_factor = 1.0}), std::invalid_argument);
  EXPECT_THROW(GradScaler({.growth_interval = 0}), std::invalid_argument);
  EXPECT_THROW(
      GradScaler({.init_scale = std::numeric_limits<double>::quiet_NaN()}),
      std::invalid_argument);
}

TEST(GradScalerTest, EnforcesStepUpdateStateMachineAndValidatesState) {
  auto params = make_params({{1}});
  auto optimizer = std::make_shared<optim::SGD>(params);
  GradScaler scaler({.init_scale = 2.0});
  EXPECT_THROW(scaler.update(), std::logic_error);

  params[0].mutable_grad() = torch::full_like(params[0], 2.0);
  scaler.step(optimizer);
  EXPECT_THROW(scaler.state_dict(), std::logic_error);
  EXPECT_THROW(scaler.step(optimizer), std::logic_error);
  EXPECT_THROW(scaler.unscale(optimizer), std::logic_error);
  scaler.update();

  auto invalid = scaler.state_dict();
  invalid["scale"] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(scaler.load_state_dict(invalid), std::invalid_argument);

  GradScaler incompatible({.init_scale = 4.0});
  EXPECT_THROW(incompatible.load_state_dict(scaler.state_dict()),
               std::invalid_argument);

  auto incomplete = scaler.state_dict();
  incomplete.erase("growth_factor");
  EXPECT_THROW(scaler.load_state_dict(incomplete), std::invalid_argument);
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

  EXPECT_THROW(training::Trainer(model, optimizer,
                                 training::TrainerOptions{.use_amp = true}),
               std::runtime_error);
}

TEST(TrainerAMPTest, BFloat16AmpOnCpuDoesNotCreateLossScaler) {
  auto model = std::make_shared<LinearModel>(2, 1);
  auto optimizer =
      std::make_shared<optim::Adam>(model->parameters(), optim::AdamOptions{});

  training::Trainer trainer(
      model, optimizer,
      training::TrainerOptions{.use_amp = true, .amp_dtype = at::kBFloat16});

  EXPECT_FALSE(trainer.grad_scaler());
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
            .max_epochs = 10, .device = torch::kCUDA, .use_amp = use_amp});
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

}  // namespace
