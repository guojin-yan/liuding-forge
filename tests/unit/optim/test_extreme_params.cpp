// Extreme optimizer parameter tests.

#include <jyppx/core/tensor_utils.hpp>
#include <jyppx/data/dataloader.hpp>
#include <jyppx/data/tensor_dataset.hpp>
#include <jyppx/optim/adam.hpp>
#include <jyppx/optim/adamw.hpp>

#include <gtest/gtest.h>
#include <memory>

namespace {

using jyppx::Tensor;

TEST(ExtremeOptimizerParamsTest, AdamVariantsRemainFiniteAtLargeLearningRate) {
  for (const bool decoupled : {false, true}) {
    auto parameter = torch::ones(
        {128},
        torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
    parameter.mutable_grad() = torch::full_like(parameter, 0.5);
    if (decoupled) {
      jyppx::optim::AdamW optimizer({parameter},
                                    {.lr = 1e2, .weight_decay = 1e-1});
      optimizer.step();
    } else {
      jyppx::optim::Adam optimizer({parameter},
                                   {.lr = 1e2, .weight_decay = 1e-1});
      optimizer.step();
    }
    EXPECT_TRUE(jyppx::is_finite(parameter));
  }
}

TEST(ExtremeOptimizerParamsTest, VeryLargeBatchIsBatchedWithoutOverflow) {
  auto data =
      torch::arange(0, 8192, torch::TensorOptions().dtype(torch::kFloat32))
          .view({8192, 1});
  auto targets = data * 2.0F;
  auto dataset = std::make_shared<jyppx::data::TensorDataset>(data, targets);
  jyppx::data::DataLoader loader(dataset, {.batch_size = 8192});

  ASSERT_EQ(loader.size(), 1U);
  for (const auto& batch : loader) {
    ASSERT_EQ(batch.data.size(0), 8192);
    EXPECT_TRUE(jyppx::is_finite(batch.data));
    EXPECT_TRUE(jyppx::is_finite(batch.targets));
  }
}

}  // namespace
