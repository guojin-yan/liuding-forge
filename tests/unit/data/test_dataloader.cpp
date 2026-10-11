// Unit tests for jyppx::data::DataLoader.

#include <jyppx/data/dataloader.hpp>
#include <jyppx/data/tensor_dataset.hpp>
#include <jyppx/training/trainer.hpp>

#include "test_utils/jyppx_test_helpers.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <vector>

namespace {

using namespace jyppx;

// Minimal map-style dataset yielding raw tensors, used to exercise the
// tensor-typed DataLoader specializations.
class TensorRowsDataset : public data::Dataset<Tensor> {
public:
    explicit TensorRowsDataset(Tensor data) : data_(std::move(data)) {}

    Tensor get(size_t index) const override {
        return data_[static_cast<int64_t>(index)];
    }
    size_t size() const override {
        return static_cast<size_t>(data_.size(0));
    }

private:
    Tensor data_;
};

class DataLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        inputs_ = torch::arange(0, 20, torch::TensorOptions().dtype(torch::kFloat32)).view({10, 2});
        targets_ = torch::arange(0, 10, torch::TensorOptions().dtype(torch::kFloat32));
        dataset_ = std::make_shared<data::TensorDataset>(inputs_, targets_);
    }

    Tensor inputs_;
    Tensor targets_;
    std::shared_ptr<data::TensorDataset> dataset_;
};

TEST_F(DataLoaderTest, BatchesWithoutShufflePreserveOrder) {
    data::DataLoader loader(dataset_, {.batch_size = 4});

    ASSERT_EQ(loader.size(), 3); // 4 + 4 + 2
    int batch_index = 0;
    for (const auto& batch : loader) {
        EXPECT_EQ(batch.data.size(0), batch_index < 2 ? 4 : 2);
        EXPECT_EQ(batch.targets.size(0), batch_index < 2 ? 4 : 2);
        // Sequential loading keeps dataset order.
        EXPECT_TRUE(torch::equal(batch.data, inputs_.narrow(0, batch_index * 4,
                                                     batch_index < 2 ? 4 : 2)));
        ++batch_index;
    }
    EXPECT_EQ(batch_index, 3);
}

TEST_F(DataLoaderTest, DropLastRemovesIncompleteBatch) {
    data::DataLoader loader(dataset_, {.batch_size = 4, .drop_last = true});
    ASSERT_EQ(loader.size(), 2);
    for (const auto& batch : loader) {
        EXPECT_EQ(batch.data.size(0), 4);
    }
}

TEST_F(DataLoaderTest, ShuffleWithSeedIsDeterministicAcrossLoaders) {
    data::DataLoader first(dataset_, {.batch_size = 5, .shuffle = true, .seed = 99});
    data::DataLoader second(dataset_, {.batch_size = 5, .shuffle = true, .seed = 99});

    std::vector<Tensor> first_batches;
    for (const auto& batch : first) {
        first_batches.push_back(batch.data.clone());
    }
    int index = 0;
    for (const auto& batch : second) {
        EXPECT_TRUE(torch::equal(batch.data, first_batches[index++]));
    }
}

TEST_F(DataLoaderTest, ShuffledBatchesAreAPermutationOfTheDataset) {
    data::DataLoader loader(dataset_, {.batch_size = 3, .shuffle = true, .seed = 5});

    std::set<float> seen;
    for (const auto& batch : loader) {
        for (int64_t row = 0; row < batch.data.size(0); ++row) {
            seen.insert(batch.data[row][0].item<float>());
        }
    }
    EXPECT_EQ(seen.size(), 10); // every sample appears exactly once
}

TEST_F(DataLoaderTest, ResetStartsNewEpochWithFreshOrder) {
    data::DataLoader loader(dataset_, {.batch_size = 5, .shuffle = true, .seed = 11});

    std::vector<Tensor> first_epoch;
    for (const auto& batch : loader) {
        first_epoch.push_back(batch.data.clone());
    }
    loader.reset();
    int index = 0;
    bool differs = false;
    for (const auto& batch : loader) {
        if (!torch::equal(batch.data, first_epoch[index++])) {
            differs = true;
        }
    }
    EXPECT_TRUE(differs); // reshuffled for the next epoch
    EXPECT_EQ(index, 2);
}

TEST_F(DataLoaderTest, CustomCollateIsApplied) {
    // Double the inputs, keep the targets. Verifies the collate function is
    // actually consulted for Example batches.
    data::DataLoader loader(
        dataset_,
        {.batch_size = 4},
        [](const std::vector<data::Example>& samples) {
            std::vector<Tensor> data;
            std::vector<Tensor> targets;
            for (const auto& sample : samples) {
                data.push_back(sample.data * 2.0);
                targets.push_back(sample.target);
            }
            return data::Batch<data::Example>{torch::stack(data), torch::stack(targets)};
        });

    int batch_index = 0;
    for (const auto& batch : loader) {
        const int64_t rows = batch_index < 2 ? 4 : 2;
        EXPECT_TRUE(
            torch::allclose(batch.data, inputs_.narrow(0, batch_index * 4, rows) * 2.0));
        ++batch_index;
    }
    EXPECT_EQ(batch_index, 3);
}

TEST_F(DataLoaderTest, TensorValuedLoaderStacksSamples) {
    auto rows = std::make_shared<TensorRowsDataset>(
        torch::arange(0, 8, torch::TensorOptions().dtype(torch::kFloat32)).view({8, 1}));

    data::DataLoader tensor_loader(rows, {.batch_size = 4});

    ASSERT_EQ(tensor_loader.size(), 2);
    int batch_index = 0;
    for (const auto& batch : tensor_loader) {
        ASSERT_TRUE(batch.data.defined());
        EXPECT_EQ(batch.data.size(0), 4);
        // Default collate stacks rows into a new leading dimension.
        EXPECT_EQ(batch.data.size(1), 1);
        EXPECT_TRUE(torch::equal(batch.data.squeeze(1),
                                 torch::arange(batch_index * 4, batch_index * 4 + 4,
                                               torch::TensorOptions().dtype(torch::kFloat32))));
        ++batch_index;
    }
    EXPECT_EQ(batch_index, 2);
}

TEST_F(DataLoaderTest, ZeroBatchSizeThrows) {
    EXPECT_THROW(data::DataLoader(dataset_, {.batch_size = 0}), std::invalid_argument);
}

TEST_F(DataLoaderTest, NullDatasetThrows) {
    std::shared_ptr<data::Dataset<data::Example>> null_dataset;
    EXPECT_THROW(
        data::DataLoader(null_dataset, {.batch_size = 2}), std::invalid_argument);
}

TEST_F(DataLoaderTest, DatasetSizeExposesSampleCount) {
    data::DataLoader loader(dataset_, {.batch_size = 4});
    EXPECT_EQ(loader.dataset_size(), 10);
}

// ---- Multi-worker loading --------------------------------------------------

// Dataset that throws for every access, used to verify worker exception
// propagation.
class ExplodingDataset : public data::Dataset<data::Example> {
public:
    data::Example get(size_t) const override {
        throw std::runtime_error("worker boom");
    }
    size_t size() const override { return 8; }
};

TEST_F(DataLoaderTest, MultiWorkerMatchesSingleThreadOutput) {
    // shuffle=false: multi-worker output must be identical to single-threaded.
    std::vector<Tensor> single;
    {
        data::DataLoader loader(dataset_, {.batch_size = 3});
        for (const auto& batch : loader) {
            single.push_back(batch.data.clone());
        }
    }

    std::vector<Tensor> multi;
    {
        data::DataLoader loader(dataset_, {.batch_size = 3, .num_workers = 4});
        for (const auto& batch : loader) {
            multi.push_back(batch.data.clone());
        }
    }

    ASSERT_EQ(single.size(), multi.size());
    for (size_t i = 0; i < single.size(); ++i) {
        EXPECT_TRUE(torch::equal(single[i], multi[i])) << "batch " << i;
    }
}

TEST_F(DataLoaderTest, MultiWorkerShuffledIsAPermutation) {
    data::DataLoader loader(
        dataset_,
        {.batch_size = 3, .shuffle = true, .num_workers = 3, .seed = 5});

    std::set<float> seen;
    for (const auto& batch : loader) {
        for (int64_t row = 0; row < batch.data.size(0); ++row) {
            seen.insert(batch.data[row][0].item<float>());
        }
    }
    EXPECT_EQ(seen.size(), 10);
}

TEST_F(DataLoaderTest, MultiWorkerWithCustomCollate) {
    data::DataLoader loader(
        dataset_,
        {.batch_size = 4, .num_workers = 2},
        [](const std::vector<data::Example>& samples) {
            std::vector<Tensor> data;
            std::vector<Tensor> targets;
            for (const auto& sample : samples) {
                data.push_back(sample.data * 2.0);
                targets.push_back(sample.target);
            }
            return data::Batch<data::Example>{torch::stack(data), torch::stack(targets)};
        });

    int batch_index = 0;
    for (const auto& batch : loader) {
        const int64_t rows = batch_index < 2 ? 4 : 2;
        EXPECT_TRUE(
            torch::allclose(batch.data, inputs_.narrow(0, batch_index * 4, rows) * 2.0));
        ++batch_index;
    }
    EXPECT_EQ(batch_index, 3);
}

TEST_F(DataLoaderTest, MultiWorkerPropagatesExceptions) {
    auto exploding = std::make_shared<ExplodingDataset>();
    data::DataLoader loader(exploding, {.batch_size = 2, .num_workers = 2});

    EXPECT_THROW(
        [] (auto& l) {
            for (const auto& batch : l) {
                (void)batch;
            }
        }(loader),
        std::runtime_error);
}

TEST_F(DataLoaderTest, MultiWorkerEarlyBreakThenReuseIsClean) {
    data::DataLoader loader(dataset_, {.batch_size = 2, .num_workers = 3});

    // Abort the first epoch after two batches; in-flight work must not leak
    // into the next epoch.
    int consumed = 0;
    for (const auto& batch : loader) {
        (void)batch;
        if (++consumed == 2) {
            break;
        }
    }
    EXPECT_EQ(consumed, 2);

    int batch_index = 0;
    for (const auto& batch : loader) {
        // 10 samples / batch size 2 = exactly 5 full batches.
        ASSERT_EQ(batch.data.size(0), 2);
        EXPECT_TRUE(torch::equal(
            batch.data, inputs_.narrow(0, batch_index * 2, 2)));
        ++batch_index;
    }
    EXPECT_EQ(batch_index, 5);
}

TEST_F(DataLoaderTest, MultiWorkerStressManyEpochs) {
    data::DataLoader loader(
        dataset_,
        {.batch_size = 3, .shuffle = true, .num_workers = 8, .seed = 1});

    for (int epoch = 0; epoch < 20; ++epoch) {
        std::size_t samples = 0;
        for (const auto& batch : loader) {
            samples += batch.data.size(0);
        }
        EXPECT_EQ(samples, 10) << "epoch " << epoch;
    }
}

TEST_F(DataLoaderTest, MultiWorkerPrefetchFactorMustBePositive) {
    EXPECT_THROW(
        data::DataLoader(dataset_, {.batch_size = 2, .num_workers = 2, .prefetch_factor = 0}),
        std::invalid_argument);
}

TEST_F(DataLoaderTest, MultiWorkerTrainerFit) {
    // The Trainer consumes the loader epoch by epoch; verify the multi-worker
    // pipeline slots into fit() cleanly.
    auto [inputs, targets] = jyppx_test::make_linear_data(32);
    data::DataLoader loader(
        std::make_shared<data::TensorDataset>(inputs, targets),
        {.batch_size = 8, .num_workers = 2});

    torch::manual_seed(2026);
    auto model = std::make_shared<jyppx_test::LinearModel>(2, 1);
    auto optimizer =
        std::make_shared<optim::Adam>(model->parameters(), optim::AdamOptions{.lr = 0.1});
    training::Trainer trainer(
        model, optimizer, training::TrainerOptions{.max_epochs = 3});
    trainer.fit(loader);

    EXPECT_EQ(trainer.global_step(), 12); // 3 epochs x 4 batches
    EXPECT_LT(trainer.last_train_loss(), 0.5); // clearly below the ~4.0 start
}

} // namespace
