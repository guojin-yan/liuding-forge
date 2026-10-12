// End-to-end unit tests for jyppx::training::Trainer: fitting, validation,
// gradient accumulation, checkpointing and option validation.

#include <jyppx/training/trainer.hpp>
#include <jyppx/optim/schedulers/reduce_lr_on_plateau.hpp>
#include <jyppx/optim/schedulers/step_lr.hpp>

#include "test_utils/jyppx_test_helpers.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <memory>

namespace {

using namespace jyppx;
using namespace jyppx_test;

struct TrainerFixture {
    std::shared_ptr<LinearModel> model;
    std::shared_ptr<optim::Adam> optimizer;
    std::unique_ptr<training::Trainer> trainer;
    std::shared_ptr<data::DataLoader<data::Example>> train_loader;
    std::shared_ptr<data::DataLoader<data::Example>> val_loader;

    static TrainerFixture create(int max_epochs = 15, double lr = 0.1) {
        TrainerFixture fx;
        torch::manual_seed(2026);
        fx.model = std::make_shared<LinearModel>(2, 1);
        fx.optimizer = std::make_shared<optim::Adam>(fx.model->parameters(),
                                                     optim::AdamOptions{.lr = lr});
        fx.trainer = std::make_unique<training::Trainer>(
            fx.model, fx.optimizer, training::TrainerOptions{.max_epochs = max_epochs});

        const auto [inputs, targets] = make_linear_data(32);
        fx.train_loader = std::make_shared<data::DataLoader<data::Example>>(
            std::make_shared<data::TensorDataset>(inputs, targets),
            data::DataLoaderOptions{.batch_size = 8});
        fx.val_loader = std::make_shared<data::DataLoader<data::Example>>(
            std::make_shared<data::TensorDataset>(inputs, targets),
            data::DataLoaderOptions{.batch_size = 8});
        return fx;
    }
};

TEST(TrainerTest, FitReducesLossOnLinearRegression) {
    auto fx = TrainerFixture::create();

    // Sanity: the untrained loss is large.
    const double initial_loss = fx.trainer->validate(*fx.val_loader);
    EXPECT_GT(initial_loss, 0.1);

    fx.trainer->fit(*fx.train_loader);

    EXPECT_LT(fx.trainer->last_train_loss(), initial_loss);
    EXPECT_GT(fx.trainer->global_step(), 0);
    // y = 2*x0 - x1 + 0.5 with lr=0.05 Adam over 15 epochs should converge
    // comfortably below 1e-2 mean squared error.
    EXPECT_LT(fx.trainer->last_train_loss(), 1e-2);
}

TEST(TrainerTest, ValidationReturnsFiniteMeanLoss) {
    auto fx = TrainerFixture::create();
    const double loss = fx.trainer->validate(*fx.val_loader);
    EXPECT_TRUE(std::isfinite(loss));
    EXPECT_GT(loss, 0.0);

    const double test_loss = fx.trainer->test(*fx.val_loader);
    EXPECT_DOUBLE_EQ(test_loss, loss); // same data, eval mode is deterministic
}

TEST(TrainerTest, ValidationRestoresTrainingMode) {
    auto fx = TrainerFixture::create();
    fx.trainer->fit(*fx.train_loader);
    EXPECT_TRUE(fx.model->is_training());

    fx.trainer->validate(*fx.val_loader);
    EXPECT_TRUE(fx.model->is_training());

    fx.model->eval();
    fx.trainer->validate(*fx.val_loader);
    EXPECT_FALSE(fx.model->is_training());
}

TEST(TrainerTest, MaxStepsStopsTrainingEarly) {
    auto fx = TrainerFixture::create(/*max_epochs=*/10);
    fx.trainer = std::make_unique<training::Trainer>(
        fx.model, fx.optimizer,
        training::TrainerOptions{.max_epochs = 10, .max_steps = 3});

    fx.trainer->fit(*fx.train_loader);
    EXPECT_EQ(fx.trainer->global_step(), 3);
    EXPECT_EQ(fx.trainer->current_epoch(), 0); // stopped mid-epoch
}

TEST(TrainerTest, GradientAccumulationMatchesSingleLargeBatch) {
    // Two accumulated half-batches must produce the same update as one full
    // batch, because mean(batch) == mean(half1)/2 + mean(half2)/2.
    auto full = TrainerFixture::create(/*max_epochs=*/1, /*lr=*/0.1);
    auto accumulated = TrainerFixture::create(/*max_epochs=*/1, /*lr=*/0.1);
    copy_module_weights(*full.model, *accumulated.model);

    full.trainer->fit(*full.train_loader);

    auto half_loader = std::make_shared<data::DataLoader<data::Example>>(
        std::make_shared<data::TensorDataset>(
            make_linear_data(32).first, make_linear_data(32).second),
        data::DataLoaderOptions{.batch_size = 4});
    accumulated.trainer = std::make_unique<training::Trainer>(
        accumulated.model, accumulated.optimizer,
        training::TrainerOptions{.max_epochs = 1, .accumulation_steps = 2});
    accumulated.trainer->fit(*half_loader);

    EXPECT_EQ(full.trainer->global_step(), 4); // 32 samples / batch 8
    EXPECT_EQ(accumulated.trainer->global_step(), 4); // 8 micro-batches / 2
    expect_tensor_lists_close(
        full.model->parameters(), accumulated.model->parameters(), 1e-5);
}

TEST(TrainerTest, CheckpointRoundTripPreservesWeightsAndCounters) {
    // Two epochs of a 32-sample loader with batch size 8 == 8 optimizer steps.
    auto source = TrainerFixture::create(/*max_epochs=*/2);
    source.trainer->fit(*source.train_loader);
    ASSERT_EQ(source.trainer->global_step(), 8);

    const auto checkpoint_path =
        (std::filesystem::temp_directory_path() / "jyppx_trainer_roundtrip.pt").string();
    source.trainer->save_checkpoint(checkpoint_path);

    // A fresh trainer starts from different weights...
    auto restored = TrainerFixture::create();
    restored.trainer->load_checkpoint(checkpoint_path);

    // ...and after loading matches the source exactly.
    expect_tensor_lists_close(source.model->parameters(), restored.model->parameters());
    EXPECT_EQ(restored.trainer->global_step(), source.trainer->global_step());
    EXPECT_EQ(restored.trainer->current_epoch(), source.trainer->current_epoch());

    // Resuming continues from next_epoch instead of restarting: two more
    // epochs produce exactly eight additional optimizer steps.
    const int resume_epoch = source.trainer->current_epoch(); // 2
    const int64_t steps_before = restored.trainer->global_step();
    restored.trainer = std::make_unique<training::Trainer>(
        restored.model, restored.optimizer,
        training::TrainerOptions{.max_epochs = resume_epoch + 2});
    restored.trainer->load_checkpoint(checkpoint_path);
    restored.trainer->fit(*restored.train_loader);
    EXPECT_EQ(restored.trainer->global_step(), steps_before + 8);
    EXPECT_EQ(restored.trainer->current_epoch(), resume_epoch + 2);

    std::filesystem::remove(checkpoint_path);
}

TEST(TrainerTest, MetricDrivenSchedulerConsumesValidationLoss) {
    auto fx = TrainerFixture::create(/*max_epochs=*/2);
    auto scheduler = std::make_shared<optim::schedulers::ReduceLROnPlateau>(
        fx.optimizer,
        optim::schedulers::ReduceLROnPlateauOptions{
            .mode = "min", .factor = 0.5, .patience = 0, .threshold = 0.0});
    fx.trainer->set_scheduler(scheduler);

    EXPECT_NO_THROW(fx.trainer->fit(*fx.train_loader, fx.val_loader.get()));
    EXPECT_EQ(scheduler->get_step(), 2);
    EXPECT_TRUE(fx.trainer->has_validation_loss());
}

TEST(TrainerTest, MetricDrivenSchedulerRejectsMissingValidationLoader) {
    auto fx = TrainerFixture::create(/*max_epochs=*/2);
    auto scheduler = std::make_shared<optim::schedulers::ReduceLROnPlateau>(
        fx.optimizer,
        optim::schedulers::ReduceLROnPlateauOptions{.patience = 0});
    fx.trainer->set_scheduler(scheduler);

    EXPECT_THROW(fx.trainer->fit(*fx.train_loader), std::invalid_argument);
}

TEST(TrainerTest, CheckpointRestoresSchedulerState) {
    auto source = TrainerFixture::create(/*max_epochs=*/1);
    auto source_scheduler = std::make_shared<optim::schedulers::StepLR>(
        source.optimizer, /*step_size=*/1, /*gamma=*/0.5);
    source.trainer->set_scheduler(source_scheduler);
    source.trainer->fit(*source.train_loader);
    ASSERT_EQ(source_scheduler->get_step(), 1);

    const auto checkpoint_path =
        (std::filesystem::temp_directory_path() / "jyppx_trainer_scheduler.pt").string();
    source.trainer->save_checkpoint(checkpoint_path);

    auto restored = TrainerFixture::create(/*max_epochs=*/2);
    auto restored_scheduler = std::make_shared<optim::schedulers::StepLR>(
        restored.optimizer, /*step_size=*/1, /*gamma=*/0.5);
    restored.trainer->set_scheduler(restored_scheduler);
    restored.trainer->load_checkpoint(checkpoint_path);

    EXPECT_EQ(restored_scheduler->get_step(), source_scheduler->get_step());
    EXPECT_DOUBLE_EQ(restored_scheduler->get_last_lr(), source_scheduler->get_last_lr());
    EXPECT_DOUBLE_EQ(
        restored.optimizer->get_lrs().front(), source.optimizer->get_lrs().front());
    std::filesystem::remove(checkpoint_path);
}

TEST(TrainerTest, OptionValidationThrows) {
    auto fx = TrainerFixture::create();

    // Null model / optimizer.
    EXPECT_THROW(
        training::Trainer(nullptr, fx.optimizer, training::TrainerOptions{}),
        std::invalid_argument);
    EXPECT_THROW(
        training::Trainer(fx.model, nullptr, training::TrainerOptions{}),
        std::invalid_argument);

    // Bad option values.
    EXPECT_THROW(
        training::Trainer(fx.model, fx.optimizer, training::TrainerOptions{.max_epochs = -1}),
        std::invalid_argument);
    EXPECT_THROW(
        training::Trainer(fx.model, fx.optimizer,
                          training::TrainerOptions{.max_steps = 0}),
        std::invalid_argument);
    EXPECT_THROW(
        training::Trainer(fx.model, fx.optimizer,
                          training::TrainerOptions{.accumulation_steps = 0}),
        std::invalid_argument);
    EXPECT_THROW(
        training::Trainer(fx.model, fx.optimizer,
                          training::TrainerOptions{.gradient_clip_val = -1.0}),
        std::invalid_argument);
    EXPECT_THROW(
        training::Trainer(fx.model, fx.optimizer,
                          training::TrainerOptions{.gradient_clip_mode = "bogus"}),
        std::invalid_argument);
    // AMP is not implemented in this build yet and must fail loudly.
    EXPECT_THROW(
        training::Trainer(fx.model, fx.optimizer, training::TrainerOptions{.use_amp = true}),
        std::runtime_error);
}

TEST(TrainerTest, FitRejectsLoaderWithoutBatches) {
    auto fx = TrainerFixture::create();
    // batch_size larger than the dataset with drop_last yields zero batches.
    auto empty_loader = data::DataLoader<data::Example>(
        std::make_shared<data::TensorDataset>(make_linear_data(4).first,
                                              make_linear_data(4).second),
        data::DataLoaderOptions{.batch_size = 8, .drop_last = true});
    EXPECT_THROW(fx.trainer->fit(empty_loader), std::invalid_argument);
}

TEST(TrainerTest, GradientClippingLimitsParameterMovement) {
    // SGD with lr=1.0 would normally move parameters by O(1) per step on this
    // problem. With the gradient norm clipped to 1e-3, one epoch (4 steps)
    // can move each parameter by at most ~lr * max_norm * steps = 4e-3.
    torch::manual_seed(2026);
    auto model = std::make_shared<LinearModel>(2, 1);
    const auto initial = clone_tensors(model->parameters());
    auto optimizer =
        std::make_shared<optim::SGD>(model->parameters(), optim::SGDOptions{.lr = 1.0});
    training::Trainer trainer(
        model, optimizer,
        training::TrainerOptions{.max_epochs = 1, .gradient_clip_val = 1e-3});

    const auto [inputs, targets] = make_linear_data(32);
    auto loader = data::DataLoader<data::Example>(
        std::make_shared<data::TensorDataset>(inputs, targets),
        data::DataLoaderOptions{.batch_size = 8});
    trainer.fit(loader);

    const auto& trained = model->parameters();
    ASSERT_EQ(initial.size(), trained.size());
    for (size_t i = 0; i < trained.size(); ++i) {
        const double movement =
            (trained[i] - initial[i]).norm().item<double>();
        EXPECT_LT(movement, 0.1) << "parameter " << i;
    }
}

} // namespace
