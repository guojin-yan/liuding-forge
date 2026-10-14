#include <jyppx/optim/schedulers/composite_schedulers.hpp>
#include <jyppx/optim/schedulers/exponential_lr.hpp>
#include <jyppx/optim/schedulers/lambda_lr.hpp>
#include <jyppx/optim/sgd.hpp>

#include <gtest/gtest.h>

namespace {

using namespace jyppx;

std::shared_ptr<optim::SGD> make_optimizer() {
    auto parameter = torch::tensor(
        {1.0}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
    return std::make_shared<optim::SGD>(TensorList{parameter},
                                        optim::SGDOptions{.lr = 0.1});
}

std::vector<std::shared_ptr<optim::schedulers::LRScheduler>> make_children(
    const std::shared_ptr<optim::SGD>& optimizer) {
    return {
        std::make_shared<optim::schedulers::LambdaLR>(
            optimizer, [](int64_t step) { return 1.0 / (step + 1.0); }),
        std::make_shared<optim::schedulers::ExponentialLR>(optimizer, 0.5),
    };
}

TEST(CompositeSchedulerTest, SequentialSwitchesAndRoundTripsState) {
    auto optimizer = make_optimizer();
    optim::schedulers::SequentialLR scheduler(optimizer, make_children(optimizer), {2});
    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.05, 1e-12);

    const auto state_before_switch = scheduler.state_dict();
    auto early_optimizer = make_optimizer();
    optim::schedulers::SequentialLR early_restored(
        early_optimizer, make_children(early_optimizer), {2});
    early_restored.load_state_dict(state_before_switch);
    EXPECT_NEAR(early_restored.get_last_lr(), 0.05, 1e-12);
    EXPECT_NEAR(early_optimizer->get_lrs().front(), 0.05, 1e-12);

    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.05, 1e-12);
    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.025, 1e-12);

    auto restored_optimizer = make_optimizer();
    optim::schedulers::SequentialLR restored(
        restored_optimizer, make_children(restored_optimizer), {2});
    restored.load_state_dict(scheduler.state_dict());
    EXPECT_EQ(restored.get_step(), scheduler.get_step());
    EXPECT_NEAR(restored_optimizer->get_lrs().front(), scheduler.get_last_lr(), 1e-12);
    scheduler.step();
    restored.step();
    EXPECT_NEAR(restored.get_last_lr(), scheduler.get_last_lr(), 1e-12);
}

TEST(CompositeSchedulerTest, ChainedMultipliesChildFactorsAndRestores) {
    auto optimizer = make_optimizer();
    optim::schedulers::ChainedScheduler scheduler(optimizer, make_children(optimizer));
    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.025, 1e-12);
    scheduler.step();
    EXPECT_NEAR(scheduler.get_last_lr(), 0.1 / 3.0 * 0.25, 1e-12);

    auto restored_optimizer = make_optimizer();
    optim::schedulers::ChainedScheduler restored(
        restored_optimizer, make_children(restored_optimizer));
    restored.load_state_dict(scheduler.state_dict());
    EXPECT_NEAR(restored_optimizer->get_lrs().front(), scheduler.get_last_lr(), 1e-12);
    scheduler.step();
    restored.step();
    EXPECT_NEAR(restored.get_last_lr(), scheduler.get_last_lr(), 1e-12);
}

TEST(CompositeSchedulerTest, RejectsInvalidStructures) {
    auto optimizer = make_optimizer();
    EXPECT_THROW(optim::schedulers::SequentialLR(optimizer, {}, {}),
                 std::invalid_argument);
    EXPECT_THROW(optim::schedulers::SequentialLR(
                     optimizer, make_children(optimizer), {}),
                 std::invalid_argument);
    EXPECT_THROW(optim::schedulers::SequentialLR(
                     optimizer,
                     {make_children(optimizer)[0], make_children(optimizer)[1],
                      make_children(optimizer)[0]},
                     {2, 2}),
                 std::invalid_argument);
    EXPECT_THROW(optim::schedulers::ChainedScheduler(optimizer, {}),
                 std::invalid_argument);
}

TEST(CompositeSchedulerTest, LoadFailureRollsBackChildrenParentAndOptimizer) {
    auto source_optimizer = make_optimizer();
    optim::schedulers::SequentialLR source(
        source_optimizer, make_children(source_optimizer), {2});
    source.step();
    auto corrupted = source.state_dict();
    corrupted["scheduler_1_last_lr_0"] = -1.0;

    auto target_optimizer = make_optimizer();
    optim::schedulers::SequentialLR target(
        target_optimizer, make_children(target_optimizer), {2});
    const auto before = target.state_dict();
    const auto lr_before = target_optimizer->get_lrs();

    EXPECT_THROW(target.load_state_dict(corrupted), std::invalid_argument);
    EXPECT_EQ(target.state_dict(), before);
    EXPECT_EQ(target_optimizer->get_lrs(), lr_before);
}

TEST(CompositeSchedulerTest, DescriptorIncludesOrderedChildrenAndSafety) {
    auto optimizer = make_optimizer();
    optim::schedulers::SequentialLR scheduler(
        optimizer, make_children(optimizer), {2});
    const auto& descriptor = scheduler.descriptor();
    ASSERT_EQ(descriptor.children.size(), 2u);
    EXPECT_EQ(descriptor.children[0].type_id,
              "jyppx.scheduler.lambda_lr/runtime-v1");
    EXPECT_EQ(descriptor.children[1].type_id,
              "jyppx.scheduler.exponential_lr/v1");
    EXPECT_FALSE(descriptor.checkpoint_safe);
}

} // namespace
