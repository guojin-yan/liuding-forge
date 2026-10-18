// Unit tests for jyppx::data: TensorDataset.

#include <jyppx/data/tensor_dataset.hpp>
#include <jyppx/foundation/error.hpp>

#include <gtest/gtest.h>

namespace {

using namespace jyppx;

class TensorDatasetTest : public ::testing::Test {
 protected:
  void SetUp() override {
    inputs_ =
        torch::arange(0, 12, torch::TensorOptions().dtype(torch::kFloat32))
            .view({6, 2});
    targets_ =
        torch::arange(0, 6, torch::TensorOptions().dtype(torch::kFloat32));
    dataset_ = std::make_shared<data::TensorDataset>(inputs_, targets_);
  }

  Tensor inputs_;
  Tensor targets_;
  std::shared_ptr<data::TensorDataset> dataset_;
};

TEST_F(TensorDatasetTest, SizeMatchesFirstDimension) {
  EXPECT_EQ(dataset_->size(), 6);
  EXPECT_FALSE(dataset_->empty());
}

TEST_F(TensorDatasetTest, GetReturnsMatchingSlices) {
  const auto sample = dataset_->get(3);
  EXPECT_TRUE(torch::equal(sample.data, inputs_[3]));
  EXPECT_TRUE(torch::equal(sample.target, targets_[3]));
  EXPECT_EQ(sample.data.numel(), 2);
}

TEST_F(TensorDatasetTest, OperatorIndexesDataset) {
  const auto sample = (*dataset_)[5];
  EXPECT_TRUE(torch::equal(sample.data, inputs_[5]));
  EXPECT_TRUE(torch::equal(sample.target, targets_[5]));
}

TEST_F(TensorDatasetTest, UnderlyingTensorsAreExposed) {
  EXPECT_TRUE(torch::equal(dataset_->data(), inputs_));
  EXPECT_TRUE(torch::equal(dataset_->targets(), targets_));
}

TEST_F(TensorDatasetTest, MismatchedSizesThrow) {
  const auto too_few_targets = torch::zeros({3});
  EXPECT_THROW(data::TensorDataset(inputs_, too_few_targets),
               jyppx::InvalidArgument);
}

TEST_F(TensorDatasetTest, OutOfRangeGetThrows) {
  try {
    (void)dataset_->get(6);
    FAIL() << "out-of-range access must throw";
  } catch (const jyppx::InvalidArgument& error) {
    EXPECT_EQ(error.code(), jyppx::ErrorCode::invalid_argument);
    EXPECT_EQ(error.details().component, "data.tensor_dataset");
    EXPECT_EQ(error.details().operation, "get");
  }
}

}  // namespace
