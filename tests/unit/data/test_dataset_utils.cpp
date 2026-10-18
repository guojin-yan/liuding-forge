// Unit tests for dataset utilities: Subset, ConcatDataset, random_split.

#include <jyppx/data/concat_dataset.hpp>
#include <jyppx/data/dataloader.hpp>
#include <jyppx/data/subset.hpp>
#include <jyppx/data/tensor_dataset.hpp>
#include <jyppx/foundation/error.hpp>

#include <gtest/gtest.h>
#include <numeric>

namespace {

using namespace jyppx;

class DatasetUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    inputs_ =
        torch::arange(0, 20, torch::TensorOptions().dtype(torch::kFloat32))
            .view({10, 2});
    targets_ =
        torch::arange(0, 10, torch::TensorOptions().dtype(torch::kFloat32));
    dataset_ = std::make_shared<data::TensorDataset>(inputs_, targets_);
  }

  Tensor inputs_;
  Tensor targets_;
  std::shared_ptr<data::TensorDataset> dataset_;
};

TEST_F(DatasetUtilsTest, SubsetExposesSelectedSamples) {
  const data::Subset<data::Example> subset(dataset_, {3, 0, 7});

  ASSERT_EQ(subset.size(), 3);
  EXPECT_TRUE(torch::equal(subset.get(0).data, inputs_[3]));
  EXPECT_TRUE(torch::equal(subset.get(1).data, inputs_[0]));
  EXPECT_TRUE(torch::equal(subset.get(2).data, inputs_[7]));
  EXPECT_TRUE(torch::equal(subset.get(2).target, targets_[7]));
}

TEST_F(DatasetUtilsTest, SubsetValidatesArguments) {
  EXPECT_THROW(data::Subset<data::Example>(nullptr, {0}),
               jyppx::InvalidArgument);
  EXPECT_THROW(data::Subset<data::Example>(dataset_, {10}),
               jyppx::InvalidArgument);
  const data::Subset<data::Example> subset(dataset_, {0});
  EXPECT_THROW(subset.get(1), jyppx::InvalidArgument);
}

TEST_F(DatasetUtilsTest, SubsetFeedsDataLoader) {
  auto subset = std::make_shared<data::Subset<data::Example>>(
      dataset_, std::vector<size_t>{1, 3, 5, 7, 9});
  data::DataLoader loader(subset, {.batch_size = 2});

  ASSERT_EQ(loader.size(), 3);
  std::vector<float> first_column;
  for (const auto& batch : loader) {
    for (int64_t row = 0; row < batch.data.size(0); ++row) {
      first_column.push_back(batch.data[row][0].item<float>());
    }
  }
  // Odd samples only, in subset order.
  const std::vector<float> expected{2.f, 6.f, 10.f, 14.f, 18.f};
  EXPECT_EQ(first_column, expected);
}

TEST_F(DatasetUtilsTest, ConcatDatasetSpansDatasets) {
  auto first = std::make_shared<data::Subset<data::Example>>(
      dataset_, std::vector<size_t>{0, 1, 2});
  auto second = std::make_shared<data::Subset<data::Example>>(
      dataset_, std::vector<size_t>{8, 9});
  const data::ConcatDataset<data::Example> concatenated({first, second});

  EXPECT_EQ(concatenated.size(), 5);
  EXPECT_EQ(concatenated.cumulative_sizes(), (std::vector<size_t>{3, 5}));
  EXPECT_TRUE(torch::equal(concatenated.get(0).data, inputs_[0]));
  EXPECT_TRUE(torch::equal(concatenated.get(3).data, inputs_[8]));
  EXPECT_TRUE(torch::equal(concatenated.get(4).target, targets_[9]));
}

TEST_F(DatasetUtilsTest, ConcatDatasetValidatesArguments) {
  EXPECT_THROW(data::ConcatDataset<data::Example>({}), jyppx::InvalidArgument);

  std::shared_ptr<data::Dataset<data::Example>> null_dataset;
  auto valid = std::make_shared<data::Subset<data::Example>>(
      dataset_, std::vector<size_t>{0});
  EXPECT_THROW(data::ConcatDataset<data::Example>({valid, null_dataset}),
               jyppx::InvalidArgument);

  const data::ConcatDataset<data::Example> concatenated({valid});
  EXPECT_THROW(concatenated.get(1), jyppx::InvalidArgument);
}

TEST_F(DatasetUtilsTest, RandomSplitPartitionsEverySample) {
  const auto subsets =
      data::random_split(dataset_, std::vector<size_t>{6, 4}, /*seed=*/123);

  ASSERT_EQ(subsets.size(), 2);
  EXPECT_EQ(subsets[0]->size(), 6);
  EXPECT_EQ(subsets[1]->size(), 4);

  std::vector<size_t> seen;
  for (const auto& subset : subsets) {
    for (size_t i = 0; i < subset->size(); ++i) {
      seen.push_back(subset->indices()[i]);
    }
  }
  std::sort(seen.begin(), seen.end());
  for (size_t i = 0; i < seen.size(); ++i) {
    EXPECT_EQ(seen[i], i);  // partition covers each sample exactly once
  }
}

TEST_F(DatasetUtilsTest, RandomSplitIsDeterministicWithSeed) {
  const auto first =
      data::random_split(dataset_, std::vector<size_t>{5, 5}, /*seed=*/7);
  const auto second =
      data::random_split(dataset_, std::vector<size_t>{5, 5}, /*seed=*/7);

  for (size_t s = 0; s < first.size(); ++s) {
    EXPECT_EQ(first[s]->indices(), second[s]->indices());
    EXPECT_TRUE(torch::equal(first[s]->get(0).data, second[s]->get(0).data));
  }
}

TEST_F(DatasetUtilsTest, RandomSplitValidatesLengths) {
  EXPECT_THROW(data::random_split(dataset_, std::vector<size_t>{5, 6}, 1),
               jyppx::InvalidArgument);
  EXPECT_THROW(
      data::random_split(std::shared_ptr<data::Dataset<data::Example>>(),
                         std::vector<size_t>{10}, 1),
      jyppx::InvalidArgument);
}

TEST_F(DatasetUtilsTest, RandomSplitTrainValLoadersWork) {
  // The canonical train/validation split flows through DataLoader.
  const auto subsets =
      data::random_split(dataset_, std::vector<size_t>{8, 2}, /*seed=*/42);
  data::DataLoader train_loader(subsets[0], {.batch_size = 4});
  data::DataLoader val_loader(subsets[1], {.batch_size = 2});

  EXPECT_EQ(train_loader.size(), 2);
  EXPECT_EQ(val_loader.size(), 1);
  for (const auto& batch : train_loader) {
    EXPECT_EQ(batch.data.size(0), 4);
  }
  for (const auto& batch : val_loader) {
    EXPECT_EQ(batch.data.size(0), 2);
  }
}

}  // namespace
