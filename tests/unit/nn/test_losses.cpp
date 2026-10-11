// Unit tests for jyppx::nn loss functions.

#include <jyppx/nn/losses.hpp>

#include <torch/nn/functional.h>

#include <gtest/gtest.h>

#include <cmath>

namespace {

using namespace jyppx;

TEST(FocalLossTest, GammaZeroRecoversWeightedCrossEntropy) {
    // gamma = 0 collapses the focal term: -alpha * CE.
    auto logits = torch::randn({6, 4}, torch::TensorOptions().dtype(torch::kFloat64)
                                                     .requires_grad(true));
    auto target = torch::randint(0, 4, {6}, torch::kLong);

    const auto focal = nn::functional::focal_loss(logits, target, 1.0, 0.0);
    const auto ce = torch::nn::functional::cross_entropy(
        logits, target,
        torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kMean));
    EXPECT_TRUE(torch::allclose(focal, ce, 1e-10, 1e-10));
}

TEST(FocalLossTest, FocusingPenalizesEasyExamplesLess) {
    // Confident correct predictions lose weight as gamma grows.
    auto confident = torch::tensor({{10.0, -10.0}, {-10.0, 10.0}},
                                   torch::TensorOptions().dtype(torch::kFloat64));
    auto target = torch::tensor({0, 1}, torch::kLong);

    const double plain = nn::functional::focal_loss(confident, target, 1.0, 0.0)
                             .item<double>();
    const double focused = nn::functional::focal_loss(confident, target, 1.0, 2.0)
                               .item<double>();
    EXPECT_GT(plain, 0.0);
    EXPECT_LT(focused, plain * 1e-4); // (1 - pt)^2 with pt ~ 1 crushes the loss
}

TEST(FocalLossTest, HandComputedBinaryValue) {
    // logits 0 -> pt = sigmoid(0) = 0.5; alpha = 0.25, gamma = 2.
    // Positive pair: -0.25 * (1 - 0.5)^2 * log(0.5)
    auto logits = torch::tensor({{0.0}}, torch::TensorOptions().dtype(torch::kFloat64)
                                               .requires_grad(true));
    auto target = torch::tensor({{1.0}}, torch::kFloat64);

    const auto loss = nn::functional::focal_loss(logits, target, 0.25, 2.0);
    const double expected = -0.25 * std::pow(0.5, 2.0) * std::log(0.5);
    EXPECT_NEAR(loss.item<double>(), expected, 1e-12);
}

TEST(FocalLossTest, AlphaWeightsThePositiveClass) {
    auto logits = torch::tensor({{0.0}, {0.0}},
                                torch::TensorOptions().dtype(torch::kFloat64));
    auto target = torch::tensor({{1.0}, {0.0}}, torch::kFloat64);

    const auto loss = nn::functional::focal_loss(logits, target, 0.25, 0.0, "none");
    // alpha * log(2) for the positive sample, (1 - alpha) * log(2) for the negative.
    EXPECT_NEAR(loss[0].item<double>(), 0.25 * std::log(2.0), 1e-12);
    EXPECT_NEAR(loss[1].item<double>(), 0.75 * std::log(2.0), 1e-12);
}

TEST(FocalLossTest, GradientsFlowToLogits) {
    auto logits = torch::randn({8, 3}, torch::TensorOptions().dtype(torch::kFloat64)
                                            .requires_grad(true));
    auto target = torch::randint(0, 3, {8}, torch::kLong);

    auto loss = nn::functional::focal_loss(logits, target, 0.75, 2.0);
    loss.backward();
    ASSERT_TRUE(logits.grad().defined());
    EXPECT_TRUE(torch::isfinite(logits.grad()).all().item<bool>());
}

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

TEST(LabelSmoothingTest, ZeroSmoothingRecoversCrossEntropy) {
    auto logits = torch::randn({10, 5}, torch::TensorOptions().dtype(torch::kFloat64)
                                             .requires_grad(true));
    auto target = torch::randint(0, 5, {10}, torch::kLong);

    const auto ours = nn::functional::label_smoothing_cross_entropy(logits, target, 0.0);
    const auto reference = torch::nn::functional::cross_entropy(
        logits, target,
        torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kMean));
    EXPECT_TRUE(torch::allclose(ours, reference, 1e-10, 1e-10));
}

TEST(LabelSmoothingTest, MatchesTorchLabelSmoothing) {
    auto logits = torch::randn({10, 5}, torch::TensorOptions().dtype(torch::kFloat64));
    auto target = torch::randint(0, 5, {10}, torch::kLong);

    const auto ours =
        nn::functional::label_smoothing_cross_entropy(logits, target, 0.2);
    const auto reference = torch::nn::functional::cross_entropy(
        logits, target,
        torch::nn::functional::CrossEntropyFuncOptions()
            .reduction(torch::kMean)
            .label_smoothing(0.2));
    EXPECT_TRUE(torch::allclose(ours, reference, 1e-10, 1e-10));
}

TEST(LabelSmoothingTest, SmoothingRaisesTheFloorOnConfidentErrors) {
    // A confidently wrong prediction: smoothing keeps the loss finite and
    // larger than plain CE would... (CE is also finite; instead verify the
    // smoothed loss exceeds the hard CE for this case).
    auto logits = torch::tensor({{10.0, -10.0}},
                                torch::TensorOptions().dtype(torch::kFloat64));
    auto target = torch::tensor({1}, torch::kLong); // wrong class

    const double hard = nn::functional::label_smoothing_cross_entropy(
                            logits, target, 0.0)
                            .item<double>();
    const double smoothed = nn::functional::label_smoothing_cross_entropy(
                                logits, target, 0.5)
                                .item<double>();
    EXPECT_GT(hard, 0.0);
    // Smoothing redistributes mass away from the (wrong) confident target.
    EXPECT_LT(smoothed, hard);
}

TEST(ContrastiveLossTest, HandComputedValues) {
    // x1 = [0,0], x2a = [3,0] (d = 3, matching), x2b = [1,0] (d = 1, not matching).
    auto x1 = torch::tensor({{0.0, 0.0}, {0.0, 0.0}},
                            torch::TensorOptions().dtype(torch::kFloat64));
    auto x2 = torch::tensor({{3.0, 0.0}, {1.0, 0.0}},
                            torch::TensorOptions().dtype(torch::kFloat64)
                                .requires_grad(true));
    auto target = torch::tensor({1.0, 0.0}, torch::kFloat64);

    // margin = 2: matching pair -> d^2 = 9; non-matching -> (2 - 1)^2 = 1.
    const auto loss = nn::functional::contrastive_loss(x1, x2, target, 2.0, "none");
    EXPECT_NEAR(loss[0].item<double>(), 9.0, 1e-12);
    EXPECT_NEAR(loss[1].item<double>(), 1.0, 1e-12);
}

TEST(ContrastiveLossTest, MarginClampsDistantNonMatchingPairs) {
    auto x1 = torch::zeros({1, 2}, torch::TensorOptions().dtype(torch::kFloat64)
                                       .requires_grad(true));
    auto x2 = torch::tensor({{100.0, 0.0}}, torch::kFloat64);
    auto target = torch::tensor({0.0}, torch::kFloat64);

    // d = 100 > margin = 2: the hinge saturates at 0 and no gradient flows.
    const auto loss = nn::functional::contrastive_loss(x1, x2, target, 2.0);
    EXPECT_NEAR(loss.item<double>(), 0.0, 1e-12);
    loss.backward();
    EXPECT_TRUE(torch::all(x1.grad().abs() == 0.0).item<bool>());
}

TEST(ContrastiveLossTest, ValidatesArguments) {
    auto x1 = torch::randn({4, 3});
    auto x2 = torch::randn({4, 3});
    EXPECT_THROW(
        nn::functional::contrastive_loss(x1, x2, torch::full({4}, 2.0)),
        std::invalid_argument);                       // label outside {0, 1}
    EXPECT_THROW(
        nn::functional::contrastive_loss(x1, x2, torch::randint(0, 2, {4}), -1.0),
        std::invalid_argument);                       // margin <= 0
    EXPECT_THROW(
        nn::functional::contrastive_loss(x1, torch::randn({4, 5}),
                                         torch::randint(0, 2, {4})),
        std::invalid_argument);                       // shape mismatch
}

TEST(TripletLossTest, MatchesTorchImplementation) {
    torch::manual_seed(5);
    auto anchor = torch::randn({8, 4}, torch::TensorOptions().dtype(torch::kFloat64));
    auto positive = torch::randn({8, 4}, torch::TensorOptions().dtype(torch::kFloat64));
    auto negative = torch::randn({8, 4}, torch::TensorOptions().dtype(torch::kFloat64));

    const auto ours = nn::functional::triplet_margin_loss(anchor, positive, negative, 0.5);
    const auto reference = torch::nn::functional::triplet_margin_loss(
        anchor, positive, negative,
        torch::nn::functional::TripletMarginLossFuncOptions()
            .margin(0.5)
            .reduction(torch::kMean));
    EXPECT_TRUE(torch::allclose(ours, reference, 1e-12, 1e-12));
}

TEST(TripletLossTest, HandComputedSingleTriplet) {
    auto anchor = torch::zeros({1, 2}, torch::kFloat64);
    auto positive = torch::tensor({{1.0, 0.0}}, torch::kFloat64);
    auto negative = torch::tensor({{0.5, 0.0}}, torch::kFloat64);

    // d(a, p) = 1, d(a, n) = 0.5, margin = 1 -> max(0, 1 - 0.5 + 1) = 1.5.
    const auto loss = nn::functional::triplet_margin_loss(anchor, positive, negative, 1.0);
    EXPECT_NEAR(loss.item<double>(), 1.5, 1e-12);
}

TEST(LossModulesTest, CriterionWrappersCallThrough) {
    auto logits = torch::randn({6, 4}, torch::TensorOptions().dtype(torch::kFloat64));
    auto target = torch::randint(0, 4, {6}, torch::kLong);
    auto float_target = torch::randint(0, 2, {6, 1}, torch::kFloat64).to(torch::kFloat64);
    // Make the binary target well-formed: {0, 1} values.
    float_target = (torch::rand({6, 1}) > 0.5).to(torch::kFloat64);

    nn::FocalLoss focal(0.25, 2.0);
    EXPECT_DOUBLE_EQ(focal.alpha(), 0.25);
    const auto focal_value = focal.forward(logits, target);
    EXPECT_TRUE(torch::allclose(
        focal_value,
        nn::functional::focal_loss(logits, target, 0.25, 2.0), 1e-12, 1e-12));

    nn::DiceLoss dice(1.0);
    const auto dice_value = dice.forward(logits, target);
    EXPECT_TRUE(torch::allclose(
        dice_value, nn::functional::dice_loss(logits, target), 1e-12, 1e-12));

    nn::LabelSmoothingCrossEntropy smoothed(0.1);
    const auto smoothed_value = smoothed.forward(logits, target);
    EXPECT_TRUE(torch::allclose(
        smoothed_value,
        nn::functional::label_smoothing_cross_entropy(logits, target, 0.1),
        1e-12, 1e-12));

    auto x1 = torch::randn({6, 3}, torch::TensorOptions().dtype(torch::kFloat64));
    auto x2 = torch::randn({6, 3}, torch::TensorOptions().dtype(torch::kFloat64));
    auto pair_target = (torch::rand({6}) > 0.5).to(torch::kFloat64);
    nn::ContrastiveLoss contrastive(1.0);
    const auto contrastive_value = contrastive.forward(x1, x2, pair_target);
    EXPECT_TRUE(torch::allclose(
        contrastive_value,
        nn::functional::contrastive_loss(x1, x2, pair_target), 1e-12, 1e-12));

    auto anchor = torch::randn({6, 3}, torch::TensorOptions().dtype(torch::kFloat64));
    auto positive = torch::randn({6, 3}, torch::TensorOptions().dtype(torch::kFloat64));
    auto negative = torch::randn({6, 3}, torch::TensorOptions().dtype(torch::kFloat64));
    nn::TripletLoss triplet(0.8);
    EXPECT_DOUBLE_EQ(triplet.margin(), 0.8);
    const auto triplet_value = triplet.forward(anchor, positive, negative);
    EXPECT_TRUE(torch::allclose(
        triplet_value,
        nn::functional::triplet_margin_loss(anchor, positive, negative, 0.8),
        1e-12, 1e-12));

    // The float_target is exercised through the binary focal path.
    const auto binary_value = focal.forward(float_target.sigmoid(), float_target);
    EXPECT_TRUE(torch::isfinite(binary_value).all().item<bool>());
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
