// Unit tests for the remaining jyppx::nn legacy loss helpers.

#include <jyppx/nn/losses.hpp>

#include <gtest/gtest.h>

namespace {

namespace functional = jyppx::nn::functional;

TEST(LossReductionTest, ApplyReductionSemantics) {
  const auto per_element = torch::tensor({1.0, 2.0, 3.0}, torch::kFloat64);
  EXPECT_NEAR(functional::apply_reduction(per_element, "mean").item<double>(),
              2.0, 1e-12);
  EXPECT_NEAR(functional::apply_reduction(per_element, "sum").item<double>(),
              6.0, 1e-12);
  EXPECT_TRUE(torch::equal(functional::apply_reduction(per_element, "none"),
                           per_element));
  EXPECT_THROW(functional::apply_reduction(per_element, "avg"),
               std::invalid_argument);
}

}  // namespace
