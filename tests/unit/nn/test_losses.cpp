// Unit tests for jyppx::nn loss functions.

#include <jyppx/nn/losses.hpp>

#include <torch/nn/functional.h>

#include <gtest/gtest.h>

#include <cmath>

namespace {

using namespace jyppx;

TEST(DiceLossTest, PerfectPredictionScoresNearZero) {
    // One-hot probabilities on the right classes -> Dice = 1 -> loss ~ 0.
    auto probs = torch::tensor({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}},
                               torch::TensorOptions().dtype(torch::kFloat64));
    auto logits = torch::log(probs.clamp_min(1e-12));  // softmax(log p) = p
    auto target = torch::tensor({0, 1}, torch::kLong);

    const auto loss = nn::functional::dice_loss(logits, target);
    EXPECT_NEAR(loss.item<double>(), 0.0, 1e-6);
}

TEST(DiceLossTest, HandComputedMulticlassValue) {
    // Two samples, two classes. probs = softmax(logits) computed by hand.
    auto logits = torch::tensor({{0.0, 0.0}, {0.0, 0.0}},
                                torch::TensorOptions().dtype(torch::kFloat64)
                                    .requires_grad(true));
    auto target = torch::tensor({0, 1}, torch::kLong);

    // softmax of zeros = 0.5 everywhere; intersection per class = 0.5,
    // card = 1 + 1 = 2 -> dice = (2*0.5 + 1) / (2 + 1) = 2/3 -> loss = 1/3.
    const auto loss = nn::functional::dice_loss(logits, target);
    EXPECT_NEAR(loss.item<double>(), 1.0 / 3.0, 1e-12);
}

TEST(DiceLossTest, BinaryPathMatchesHandComputation) {
    // probs = sigmoid(0) = 0.5 for every element; truth all ones.
    // dice = (2 * 4 * 0.5 + 1) / (4 * 0.5 + 4 + 1) = 5 / 7.
    auto logits = torch::zeros({2, 2}, torch::TensorOptions().dtype(torch::kFloat64)
                                            .requires_grad(true));
    auto target = torch::ones({2, 2}, torch::kFloat64);

    const auto loss = nn::functional::dice_loss(logits, target);
    EXPECT_NEAR(loss.item<double>(), 1.0 - 5.0 / 7.0, 1e-12);
}

TEST(DiceLossTest, GradientsFlowToLogits) {
    auto logits = torch::randn({4, 3, 2}, torch::TensorOptions().dtype(torch::kFloat64)
                                               .requires_grad(true));
    auto target = torch::randint(0, 3, {4, 2}, torch::kLong);

    auto loss = nn::functional::dice_loss(logits, target);
    loss.backward();
    ASSERT_TRUE(logits.grad().defined());
    EXPECT_TRUE(torch::isfinite(logits.grad()).all().item<bool>());
}

TEST(DiceLossTest, ValidatesArguments) {
    auto logits = torch::randn({4, 3});
    auto bad_target = torch::full({4}, 7, torch::kLong); // class 7 out of range
    EXPECT_THROW(nn::functional::dice_loss(logits, bad_target), std::invalid_argument);
    EXPECT_THROW(nn::functional::dice_loss(logits, torch::randint(0, 3, {4}), 1.0, "sum"),
                 std::invalid_argument);
    EXPECT_THROW(nn::functional::dice_loss(logits, torch::randint(0, 3, {4}), -1.0),
                 std::invalid_argument);
}

TEST(LossModulesTest, CriterionWrappersCallThrough) {
    auto logits = torch::randn({6, 4}, torch::TensorOptions().dtype(torch::kFloat64));
    auto target = torch::randint(0, 4, {6}, torch::kLong);
    auto float_target = torch::randint(0, 2, {6, 1}, torch::kFloat64).to(torch::kFloat64);
    // Make the binary target well-formed: {0, 1} values.
    float_target = (torch::rand({6, 1}) > 0.5).to(torch::kFloat64);

    nn::DiceLoss dice(1.0);
    const auto dice_value = dice.forward(logits, target);
    EXPECT_TRUE(torch::allclose(
        dice_value, nn::functional::dice_loss(logits, target), 1e-12, 1e-12));

}

TEST(LossReductionTest, ApplyReductionSemantics) {
    auto per_element = torch::tensor({1.0, 2.0, 3.0}, torch::kFloat64);
    EXPECT_NEAR(nn::functional::apply_reduction(per_element, "mean").item<double>(),
                2.0, 1e-12);
    EXPECT_NEAR(nn::functional::apply_reduction(per_element, "sum").item<double>(),
                6.0, 1e-12);
    EXPECT_TRUE(torch::equal(nn::functional::apply_reduction(per_element, "none"),
                             per_element));
    EXPECT_THROW(nn::functional::apply_reduction(per_element, "avg"),
                 std::invalid_argument);
}

} // namespace
