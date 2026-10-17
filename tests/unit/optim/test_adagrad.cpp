// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

// Unit tests for the remaining hand-written Adagrad implementation.

#include "test_utils/jyppx_test_helpers.hpp"

#include <jyppx/optim/adagrad.hpp>

#include <torch/optim/adagrad.h>

#include <gtest/gtest.h>

namespace {

using namespace jyppx;
using namespace jyppx_test;

TEST(AdagradTest, DefaultsMatchTorch) {
  torch::manual_seed(42);
  auto jyppx_params = make_params({{4, 3}, {3}});
  torch::manual_seed(42);
  auto torch_params = make_params({{4, 3}, {3}});

  optim::Adagrad jyppx_optimizer(jyppx_params,
                                 optim::AdagradOptions{.lr = 0.05});
  torch::optim::Adagrad torch_optimizer(torch_params,
                                        torch::optim::AdagradOptions(0.05));

  expect_optimizer_parity(jyppx_optimizer, jyppx_params, torch_optimizer,
                          torch_params, 10);
}

TEST(AdagradTest, DecayAndWeightDecayMatchTorch) {
  torch::manual_seed(13);
  auto jyppx_params = make_params({{3, 3}, {3}});
  torch::manual_seed(13);
  auto torch_params = make_params({{3, 3}, {3}});

  optim::Adagrad jyppx_optimizer(
      jyppx_params,
      optim::AdagradOptions{
          .lr = 0.1, .lr_decay = 0.01, .weight_decay = 0.01, .eps = 1e-10});
  torch::optim::Adagrad torch_optimizer(
      torch_params,
      torch::optim::AdagradOptions(0.1).lr_decay(0.01).weight_decay(0.01).eps(
          1e-10));

  expect_optimizer_parity(jyppx_optimizer, jyppx_params, torch_optimizer,
                          torch_params, 10);
}

TEST(AdagradTest, InvalidOptionsThrow) {
  torch::manual_seed(4);
  auto params = make_params({{2}});
  EXPECT_THROW(optim::Adagrad(params, optim::AdagradOptions{.lr = -0.1}),
               std::invalid_argument);
  EXPECT_THROW(optim::Adagrad(params, optim::AdagradOptions{.lr_decay = -0.5}),
               std::invalid_argument);
  EXPECT_THROW(optim::Adagrad(params, optim::AdagradOptions{.eps = 0.0}),
               std::invalid_argument);
}

}  // namespace
