// Unit tests for jyppx::optim::SGD and the shared Optimizer base behavior.

#include "test_utils/jyppx_test_helpers.hpp"

#include <jyppx/optim/sgd.hpp>

#include <torch/optim/sgd.h>

#include <gtest/gtest.h>
#include <limits>

namespace {

using namespace jyppx;
using namespace jyppx_test;

TEST(SGDTest, VanillaSgdMatchesTorch) {
  torch::manual_seed(42);
  auto jyppx_params = make_params({{4, 3}, {3}});
  torch::manual_seed(42);
  auto torch_params = make_params({{4, 3}, {3}});

  optim::SGD jyppx_optimizer(
      jyppx_params,
      optim::SGDOptions{.lr = 0.05, .momentum = 0.0, .weight_decay = 0.0});
  torch::optim::SGD torch_optimizer(torch_params,
                                    torch::optim::SGDOptions(0.05));

  expect_optimizer_parity(jyppx_optimizer, jyppx_params, torch_optimizer,
                          torch_params, 10);
}

TEST(SGDTest, MomentumMatchesTorch) {
  torch::manual_seed(7);
  auto jyppx_params = make_params({{6}, {2, 2}});
  torch::manual_seed(7);
  auto torch_params = make_params({{6}, {2, 2}});

  optim::SGD jyppx_optimizer(
      jyppx_params,
      optim::SGDOptions{
          .lr = 0.03, .momentum = 0.9, .dampening = 0.0, .weight_decay = 0.0});
  torch::optim::SGD torch_optimizer(
      torch_params, torch::optim::SGDOptions(0.03).momentum(0.9));

  expect_optimizer_parity(jyppx_optimizer, jyppx_params, torch_optimizer,
                          torch_params, 10);
}

TEST(SGDTest, WeightDecayAndNesterovMatchTorch) {
  torch::manual_seed(11);
  auto jyppx_params = make_params({{3, 3}, {3}});
  torch::manual_seed(11);
  auto torch_params = make_params({{3, 3}, {3}});

  optim::SGD jyppx_optimizer(jyppx_params,
                             optim::SGDOptions{.lr = 0.02,
                                               .momentum = 0.9,
                                               .dampening = 0.0,
                                               .weight_decay = 0.01,
                                               .nesterov = true});
  torch::optim::SGD torch_optimizer(
      torch_params,
      torch::optim::SGDOptions(0.02).momentum(0.9).weight_decay(0.01).nesterov(
          true));

  expect_optimizer_parity(jyppx_optimizer, jyppx_params, torch_optimizer,
                          torch_params, 10);
}

TEST(SGDTest, EmptyParameterListThrows) {
  EXPECT_THROW(optim::SGD({}, optim::SGDOptions{.lr = 0.01}),
               std::invalid_argument);
}

TEST(SGDTest, InvalidOptionsThrow) {
  torch::manual_seed(1);
  auto params = make_params({{2}});
  EXPECT_THROW(optim::SGD(params, optim::SGDOptions{.lr = -0.1}),
               std::invalid_argument);
  // Nesterov requires momentum and no dampening.
  EXPECT_THROW(optim::SGD(params,
                          optim::SGDOptions{
                              .lr = 0.1, .momentum = 0.0, .nesterov = true}),
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

  auto second = torch::ones({1}, torch::TensorOptions().requires_grad(true));
  optimizer.add_param_group(optim::ParamGroup({second}));
  optimizer.set_lrs({0.125, 0.25});
  EXPECT_THROW(
      optimizer.set_lrs({0.5, std::numeric_limits<double>::quiet_NaN()}),
      std::invalid_argument);
  EXPECT_EQ(optimizer.get_lrs(), (std::vector<double>{0.125, 0.25}));
}

TEST(OptimizerTest, AddParamGroupInheritsDefaultsAndRoundTripsState) {
  auto first = torch::tensor(
      {1.0}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  auto second = torch::tensor(
      {1.0}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  optim::SGD optimizer({first}, optim::SGDOptions{.lr = 0.1, .momentum = 0.9});

  optim::ParamGroup second_group({second});
  second_group.options["lr"] = 0.01;
  optimizer.add_param_group(std::move(second_group));

  ASSERT_EQ(optimizer.param_groups().size(), 2);
  EXPECT_DOUBLE_EQ(optimizer.param_groups()[1].options.at("lr"), 0.01);
  EXPECT_DOUBLE_EQ(optimizer.param_groups()[1].options.at("momentum"), 0.9);

  first.mutable_grad() = torch::ones_like(first);
  second.mutable_grad() = torch::ones_like(second);
  optimizer.step();
  EXPECT_NEAR(first.item<double>(), 0.9, 1e-12);
  EXPECT_NEAR(second.item<double>(), 0.99, 1e-12);

  auto resumed_first = first.detach().clone().set_requires_grad(true);
  auto resumed_second = second.detach().clone().set_requires_grad(true);
  optim::SGD resumed({resumed_first},
                     optim::SGDOptions{.lr = 0.1, .momentum = 0.9});
  resumed.add_param_group(optim::ParamGroup({resumed_second}));
  resumed.load_state_dict(optimizer.state_dict());
  ASSERT_EQ(resumed.get_lrs().size(), 2);
  EXPECT_DOUBLE_EQ(resumed.get_lrs()[1], 0.01);

  first.mutable_grad() = torch::ones_like(first);
  second.mutable_grad() = torch::ones_like(second);
  resumed_first.mutable_grad() = torch::ones_like(resumed_first);
  resumed_second.mutable_grad() = torch::ones_like(resumed_second);
  optimizer.step();
  resumed.step();
  EXPECT_TRUE(torch::allclose(first, resumed_first, 1e-12, 1e-12));
  EXPECT_TRUE(torch::allclose(second, resumed_second, 1e-12, 1e-12));
}

TEST(OptimizerTest, AddParamGroupRejectsInvalidParameters) {
  auto parameter =
      torch::tensor({1.0}, torch::TensorOptions().requires_grad(true));
  auto duplicate = parameter.detach();
  optim::SGD optimizer({parameter});

  EXPECT_THROW(optimizer.add_param_group(optim::ParamGroup{}),
               std::invalid_argument);
  EXPECT_THROW(optimizer.add_param_group(optim::ParamGroup({Tensor{}})),
               std::invalid_argument);
  EXPECT_THROW(optimizer.add_param_group(optim::ParamGroup({parameter})),
               std::invalid_argument);
  EXPECT_THROW(
      optimizer.add_param_group(optim::ParamGroup({duplicate, duplicate})),
      std::invalid_argument);
}

TEST(OptimizerTest, ConstructorAndLearningRatesRejectInvalidValues) {
  auto parameter =
      torch::tensor({1.0}, torch::TensorOptions().requires_grad(true));
  EXPECT_THROW(optim::SGD({Tensor{}}), std::invalid_argument);
  EXPECT_THROW(optim::SGD({parameter, parameter}), std::invalid_argument);

  optim::SGD optimizer({parameter});
  EXPECT_THROW(optimizer.set_lr(std::numeric_limits<double>::quiet_NaN()),
               std::invalid_argument);
  EXPECT_THROW(optimizer.set_lrs({std::numeric_limits<double>::infinity()}),
               std::invalid_argument);
}

TEST(OptimizerTest, StateRestoreValidatesTopologyAndReplacesOldBuffers) {
  auto parameter = torch::tensor(
      {1.0}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  optim::SGD trained({parameter},
                     optim::SGDOptions{.lr = 0.1, .momentum = 0.9});
  parameter.mutable_grad() = torch::ones_like(parameter);
  trained.step();

  auto fresh_parameter = torch::tensor(
      {1.0}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  optim::SGD fresh({fresh_parameter},
                   optim::SGDOptions{.lr = 0.1, .momentum = 0.9});
  trained.load_state_dict(fresh.state_dict());
  parameter.mutable_grad() = torch::ones_like(parameter);
  trained.step();
  EXPECT_NEAR(parameter.item<double>(), 0.8, 1e-12);

  auto first = torch::ones({1}, torch::TensorOptions().requires_grad(true));
  auto second = torch::ones({1}, torch::TensorOptions().requires_grad(true));
  optim::SGD two_parameters({first, second});
  EXPECT_THROW(two_parameters.load_state_dict(fresh.state_dict()),
               std::invalid_argument);

  auto malformed = fresh.state_dict();
  malformed["step_count"] = torch::tensor(-1, torch::kInt64);
  EXPECT_THROW(fresh.load_state_dict(malformed), std::invalid_argument);

  auto incomplete = trained.state_dict();
  incomplete.erase("param_0_state_momentum_buffer");
  EXPECT_THROW(fresh.load_state_dict(incomplete), std::invalid_argument);

  incomplete = trained.state_dict();
  incomplete.erase("step_count");
  EXPECT_THROW(fresh.load_state_dict(incomplete), std::invalid_argument);

  incomplete = trained.state_dict();
  incomplete.erase("__optimizer_format_version");
  EXPECT_THROW(fresh.load_state_dict(incomplete), std::invalid_argument);
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
  optim::SGD continuous(params_a,
                        optim::SGDOptions{.lr = 0.05, .momentum = 0.9});

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

}  // namespace
