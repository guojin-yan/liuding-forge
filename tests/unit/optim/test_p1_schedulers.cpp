#include <jyppx/optim/schedulers/p1_schedulers.hpp>
#include <jyppx/optim/sgd.hpp>
#include <gtest/gtest.h>

#include "test_utils/jyppx_test_helpers.hpp"

namespace {
using namespace jyppx;
using namespace jyppx_test;
class P1SchedulerTest : public ::testing::Test {
protected:
 void SetUp() override { params_ = make_params({{2,2}}); optimizer_ = std::make_shared<optim::SGD>(params_, optim::SGDOptions{.lr = 0.1}); }
 TensorList params_; std::shared_ptr<optim::SGD> optimizer_;
};
TEST_F(P1SchedulerTest, LambdaAndCyclicModesWork) {
 optim::schedulers::LambdaLR lambda(optimizer_, [](int64_t step) { return 1.0 / (step + 1); });
 EXPECT_NEAR(lambda.get_last_lr(), 0.1, 1e-12);
 lambda.step(); EXPECT_NEAR(lambda.get_last_lr(), 0.05, 1e-12); lambda.step(); EXPECT_NEAR(lambda.get_last_lr(), 0.1 / 3.0, 1e-12);
 auto opt = std::make_shared<optim::SGD>(params_, optim::SGDOptions{.lr = 0.1, .momentum = 0.9});
 optim::schedulers::CyclicLR cyclic(opt, 0.01, 0.1, 2, 2, "triangular", 1.0, true);
 EXPECT_NEAR(opt->param_groups()[0].options.at("momentum"), 0.9, 1e-12);
 cyclic.step(); EXPECT_NEAR(cyclic.get_last_lr(), 0.055, 1e-12); EXPECT_NEAR(opt->param_groups()[0].options.at("momentum"), 0.85, 1e-12);
 cyclic.step(); EXPECT_NEAR(cyclic.get_last_lr(), 0.1, 1e-12); EXPECT_NEAR(opt->param_groups()[0].options.at("momentum"), 0.8, 1e-12);
 cyclic.step(); EXPECT_NEAR(cyclic.get_last_lr(), 0.055, 1e-12);
}
TEST_F(P1SchedulerTest, CyclicScalingMatchesPyTorchProfiles) {
 auto exp_opt = std::make_shared<optim::SGD>(params_, optim::SGDOptions{.lr = 0.1});
 optim::schedulers::CyclicLR exp_range(exp_opt, 0.01, 0.1, 1, 1, "exp_range", 0.5, false);
 exp_range.step();
 EXPECT_NEAR(exp_range.get_last_lr(), 0.055, 1e-12);

 auto custom_opt = std::make_shared<optim::SGD>(params_, optim::SGDOptions{.lr = 0.1, .momentum = 0.9});
 optim::schedulers::CyclicLR custom(
     custom_opt, 0.01, 0.1, 1, 1, "ignored", 1.0, true, 0.7, 0.95,
     [](double iteration) { return 1.0 / (iteration + 1.0); },
     "iterations");
 custom.step();
 EXPECT_NEAR(custom.get_last_lr(), 0.055, 1e-12);
 EXPECT_NEAR(custom_opt->param_groups()[0].options.at("momentum"), 0.825,
             1e-12);

 auto triangular_opt = std::make_shared<optim::SGD>(
     params_, optim::SGDOptions{.lr = 0.1, .momentum = 0.9});
 optim::schedulers::CyclicLR triangular2(
     triangular_opt, 0.01, 0.1, 1, 1, "triangular2", 1.0, true);
 triangular2.step();
 triangular2.step();
 triangular2.step();
 EXPECT_NEAR(triangular2.get_last_lr(), 0.055, 1e-12);
 EXPECT_NEAR(triangular_opt->param_groups()[0].options.at("momentum"), 0.85,
             1e-12);
}
TEST_F(P1SchedulerTest, InvalidParametersThrow) {
 EXPECT_THROW(optim::schedulers::LambdaLR(optimizer_, std::function<double(int64_t)>{}), std::invalid_argument);
 EXPECT_THROW(optim::schedulers::CyclicLR(optimizer_, 0.1, 0.01, 1), std::invalid_argument);
}
}
