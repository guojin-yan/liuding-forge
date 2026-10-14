// Unit tests for the callback system: lifecycle ordering, EarlyStopping and
// ModelCheckpoint.

#include "test_utils/jyppx_test_helpers.hpp"

#include <jyppx/training/callbacks/early_stopping.hpp>
#include <jyppx/training/callbacks/model_checkpoint.hpp>
#include <jyppx/training/trainer.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <string>

namespace {

using namespace jyppx;
using namespace jyppx_test;

// Small deterministic regression problem: y = 2*x - 1 over one input column.
class SingleFeatureModel : public torch::nn::Module {
 public:
  SingleFeatureModel()
      : linear(register_module("linear", torch::nn::Linear(1, 1))) {}

  Tensor forward(const Tensor& input) { return linear->forward(input); }

  torch::nn::Linear linear{nullptr};
};

std::shared_ptr<data::TensorDataset> make_dataset(int samples) {
  torch::manual_seed(2024);
  const auto inputs = torch::rand({samples, 1});
  const auto targets = 2.0 * inputs - 1.0;
  return std::make_shared<data::TensorDataset>(inputs, targets);
}

struct TrainerBundle {
  std::shared_ptr<SingleFeatureModel> model;
  std::shared_ptr<optim::SGD> optimizer;
  std::unique_ptr<training::Trainer> trainer;
};

TrainerBundle make_trainer(int max_epochs = 3) {
  TrainerBundle bundle;
  bundle.model = std::make_shared<SingleFeatureModel>();
  bundle.optimizer = std::make_shared<optim::SGD>(bundle.model->parameters(),
                                                  optim::SGDOptions{.lr = 0.1});
  bundle.trainer = std::make_unique<training::Trainer>(
      bundle.model, bundle.optimizer,
      training::TrainerOptions{.max_epochs = max_epochs});
  return bundle;
}

TEST(CallbackLifecycleTest, EventsFireInOrderWithoutValidation) {
  auto bundle = make_trainer(/*max_epochs=*/2);
  auto recording = std::make_shared<RecordingCallback>();
  bundle.trainer->add_callback(recording);

  data::DataLoader loader(make_dataset(8), {.batch_size = 4});
  bundle.trainer->fit(loader);

  const auto& events = recording->events();
  ASSERT_GE(events.size(), 2u);
  EXPECT_EQ(events.front(), "fit_start");
  EXPECT_EQ(events.back(), "fit_end");
  // Two epochs worth of epoch events.
  EXPECT_EQ(std::count(events.begin(), events.end(), "epoch_start"), 2);
  EXPECT_EQ(std::count(events.begin(), events.end(), "epoch_end"), 2);
  // Per epoch: train batches 0 and 1.
  EXPECT_EQ(std::count(events.begin(), events.end(), "train_batch_start:0"), 2);
  EXPECT_EQ(std::count(events.begin(), events.end(), "train_batch_start:1"), 2);
  // No validation events without a validation loader.
  EXPECT_EQ(std::count(events.begin(), events.end(), "validation_epoch_start"),
            0);
}

TEST(CallbackLifecycleTest, ValidationEventsFirePerEpoch) {
  auto bundle = make_trainer(/*max_epochs=*/1);
  auto recording = std::make_shared<RecordingCallback>();
  bundle.trainer->add_callback(recording);

  data::DataLoader train_loader(make_dataset(8), {.batch_size = 4});
  data::DataLoader val_loader(make_dataset(8), {.batch_size = 4});
  bundle.trainer->fit(train_loader, &val_loader);

  const auto& events = recording->events();
  const auto epoch_start =
      std::find(events.begin(), events.end(), "epoch_start");
  ASSERT_NE(epoch_start, events.end());
  // The remainder after the epoch starts must follow the documented order.
  const std::vector<std::string> expected_tail{
      "epoch_start",
      "train_epoch_start",
      "train_batch_start:0",
      "train_batch_end:0",
      "train_batch_start:1",
      "train_batch_end:1",
      "train_epoch_end",
      "validation_epoch_start",
      "validation_batch_start:0",
      "validation_batch_end:0",
      "validation_batch_start:1",
      "validation_batch_end:1",
      "validation_epoch_end",
      "epoch_end",
      "fit_end",
  };
  ASSERT_GE(events.size(), expected_tail.size());
  EXPECT_TRUE(std::equal(expected_tail.begin(), expected_tail.end(),
                         events.end() - expected_tail.size()));
}

TEST(EarlyStoppingTest, StopsAfterPatienceExhausted) {
  auto bundle = make_trainer();
  const auto early_stopping =
      std::make_shared<training::callbacks::EarlyStopping>("val_loss",
                                                           /*patience=*/2);
  bundle.trainer->add_callback(early_stopping);

  data::DataLoader loader(make_dataset(8), {.batch_size = 4});
  // validate() publishes last_val_loss; on_epoch_end consumes it. The loss
  // stays constant between calls, so every call after the first counts as
  // "no improvement".
  for (int call = 0; call < 10; ++call) {
    bundle.trainer->validate(loader);
    early_stopping->on_epoch_end(bundle.trainer.get());
    if (early_stopping->should_stop()) {
      break;
    }
  }
  EXPECT_TRUE(early_stopping->should_stop());
  EXPECT_TRUE(bundle.trainer->should_stop());
  // First call improves on +inf, then two non-improving calls exhaust
  // patience=2: three validate()/on_epoch_end rounds in total.
  EXPECT_LE(bundle.trainer->current_epoch(), 0);
}

TEST(EarlyStoppingTest, ResetClearsState) {
  const auto early_stopping =
      std::make_shared<training::callbacks::EarlyStopping>("val_loss",
                                                           /*patience=*/0);
  early_stopping->reset();
  EXPECT_FALSE(early_stopping->should_stop());
}

TEST(EarlyStoppingTest, InvalidArgumentsThrow) {
  EXPECT_THROW(training::callbacks::EarlyStopping("", 1),
               std::invalid_argument);
  EXPECT_THROW(training::callbacks::EarlyStopping("val_loss", /*patience=*/-1),
               std::invalid_argument);
  EXPECT_THROW(
      training::callbacks::EarlyStopping("val_loss", 1, /*min_delta=*/-0.5),
      std::invalid_argument);
  EXPECT_THROW(training::callbacks::EarlyStopping("val_loss", 1, 0.0,
                                                  /*mode=*/"max-min"),
               std::invalid_argument);
  EXPECT_THROW(training::callbacks::EarlyStopping(
                   "val_loss", 1, std::numeric_limits<double>::quiet_NaN()),
               std::invalid_argument);
}

TEST(EarlyStoppingTest, InvalidCheckpointStateDoesNotMutateCallback) {
  training::callbacks::EarlyStopping callback("val_loss", /*patience=*/2);
  const auto before = callback.state_dict();
  auto invalid = before;
  invalid.scalars["wait_count"] = 0.5;

  EXPECT_THROW(callback.load_state_dict(invalid), std::invalid_argument);
  EXPECT_EQ(callback.state_dict().scalars, before.scalars);
  EXPECT_EQ(callback.state_dict().strings, before.strings);
}

TEST(EarlyStoppingTest, CheckpointRestoresPatienceState) {
  const auto checkpoint_path =
      (std::filesystem::temp_directory_path() / "jyppx_early_stopping_state.pt")
          .string();
  auto source = make_trainer();
  auto source_callback = std::make_shared<training::callbacks::EarlyStopping>(
      "val_loss", /*patience=*/2);
  source.trainer->add_callback(source_callback);
  data::DataLoader loader(make_dataset(8), {.batch_size = 4});

  source.trainer->validate(loader);
  source_callback->on_epoch_end(source.trainer.get());
  source_callback->on_epoch_end(source.trainer.get());
  ASSERT_FALSE(source_callback->should_stop());
  source.trainer->save_checkpoint(checkpoint_path);

  auto restored = make_trainer();
  auto restored_callback = std::make_shared<training::callbacks::EarlyStopping>(
      "val_loss", /*patience=*/2);
  restored.trainer->add_callback(restored_callback);
  restored.trainer->load_checkpoint(checkpoint_path);
  restored_callback->on_epoch_end(restored.trainer.get());

  EXPECT_TRUE(restored_callback->should_stop());
  EXPECT_TRUE(restored.trainer->should_stop());
  std::filesystem::remove(checkpoint_path);
}

TEST(ModelCheckpointTest, SavesWhenValidationImproves) {
  const auto directory =
      (std::filesystem::temp_directory_path() / "jyppx_test_checkpoints")
          .string();
  std::filesystem::remove_all(directory);
  const auto checkpoint_path =
      (std::filesystem::path(directory) / "best.pt").string();

  auto bundle = make_trainer();
  const auto checkpoint =
      std::make_shared<training::callbacks::ModelCheckpoint>(
          checkpoint_path, "val_loss", "min", /*save_top_k=*/1);
  bundle.trainer->add_callback(checkpoint);

  data::DataLoader loader(make_dataset(8), {.batch_size = 4});
  bundle.trainer->validate(loader);
  checkpoint->on_epoch_end(bundle.trainer.get());
  EXPECT_TRUE(std::filesystem::exists(checkpoint_path));

  // Re-saving at the same score must not rewrite the checkpoint.
  const auto written_time = std::filesystem::last_write_time(checkpoint_path);
  checkpoint->on_epoch_end(bundle.trainer.get());
  EXPECT_EQ(std::filesystem::last_write_time(checkpoint_path), written_time);

  std::filesystem::remove_all(directory);
}

TEST(ModelCheckpointTest, InvalidArgumentsThrow) {
  EXPECT_THROW(training::callbacks::ModelCheckpoint(""), std::invalid_argument);
  EXPECT_THROW(training::callbacks::ModelCheckpoint("a.pt", ""),
               std::invalid_argument);
  EXPECT_THROW(training::callbacks::ModelCheckpoint("a.pt", "val_loss", "up"),
               std::invalid_argument);
  EXPECT_THROW(
      training::callbacks::ModelCheckpoint("a.pt", "val_loss", "min", -2),
      std::invalid_argument);
}

TEST(ModelCheckpointTest, InvalidCheckpointStateDoesNotMutateCallback) {
  training::callbacks::ModelCheckpoint callback("a.pt", "val_loss", "min",
                                                /*save_top_k=*/1);
  const auto before = callback.state_dict();
  auto invalid = before;
  invalid.scalars["saved_count"] = 0.5;

  EXPECT_THROW(callback.load_state_dict(invalid), std::invalid_argument);
  EXPECT_EQ(callback.state_dict().scalars, before.scalars);
  EXPECT_EQ(callback.state_dict().strings, before.strings);
}

TEST(ModelCheckpointTest, SaveAllCreatesParentDirectory) {
  const auto directory =
      std::filesystem::temp_directory_path() / "jyppx_test_checkpoints_all";
  std::filesystem::remove_all(directory);

  auto bundle = make_trainer();
  training::callbacks::ModelCheckpoint checkpoint(
      (directory / "epoch.pt").string(), "train_loss", "min", -1);
  checkpoint.on_epoch_end(bundle.trainer.get());

  EXPECT_TRUE(std::filesystem::exists(directory / "epoch-1.pt"));
  std::filesystem::remove_all(directory);
}

TEST(ModelCheckpointTest, CapturesCallbacksThatRunLaterInTheSameEpoch) {
  const auto checkpoint_path = (std::filesystem::temp_directory_path() /
                                "jyppx_checkpoint_callback_boundary.pt")
                                   .string();
  std::filesystem::remove(checkpoint_path);

  auto source = make_trainer();
  auto checkpoint = std::make_shared<training::callbacks::ModelCheckpoint>(
      checkpoint_path, "val_loss", "min", 1);
  auto early_stopping = std::make_shared<training::callbacks::EarlyStopping>(
      "val_loss", /*patience=*/2);
  source.trainer->add_callback(checkpoint);
  source.trainer->add_callback(early_stopping);

  data::DataLoader train_loader(make_dataset(8), {.batch_size = 4});
  data::DataLoader validation_loader(make_dataset(8), {.batch_size = 4});
  source.trainer->fit(train_loader, &validation_loader);
  ASSERT_TRUE(std::filesystem::exists(checkpoint_path));

  auto restored = make_trainer();
  auto restored_checkpoint =
      std::make_shared<training::callbacks::ModelCheckpoint>(
          checkpoint_path, "val_loss", "min", 1);
  auto restored_early_stopping =
      std::make_shared<training::callbacks::EarlyStopping>("val_loss",
                                                           /*patience=*/2);
  restored.trainer->add_callback(restored_checkpoint);
  restored.trainer->add_callback(restored_early_stopping);
  restored.trainer->load_checkpoint(checkpoint_path);

  const auto restored_state = restored_early_stopping->state_dict();
  ASSERT_TRUE(restored_state.scalars.contains("best_score"));
  EXPECT_TRUE(std::isfinite(restored_state.scalars.at("best_score")));
  EXPECT_EQ(restored_state.scalars.at("wait_count"), 0.0);

  std::filesystem::remove(checkpoint_path);
}

}  // namespace
