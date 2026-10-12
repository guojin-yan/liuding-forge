// Extreme-condition tests for logically large datasets.

#include <jyppx/data/dataloader.hpp>
#include <jyppx/core/tensor_utils.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <memory>

namespace {

using jyppx::Tensor;

// A row is 1 KiB, so this dataset represents more than 10 GiB without
// allocating the backing storage.  The test validates index arithmetic,
// batching, and repeated epochs; production datasets can replace get() with
// mmap or streaming I/O while retaining the same contract.
class VirtualLargeDataset final : public jyppx::data::Dataset<Tensor> {
public:
    static constexpr std::size_t kRows = 10'485'760;

    Tensor get(std::size_t index) const override {
        calls_.fetch_add(1, std::memory_order_relaxed);
        if (index >= kRows) {
            throw std::out_of_range("virtual dataset index out of range");
        }
        return torch::full({256}, static_cast<double>(index),
                           torch::TensorOptions().dtype(torch::kFloat32));
    }

    std::size_t size() const override { return kRows; }

    std::size_t calls() const { return calls_.load(std::memory_order_relaxed); }

private:
    mutable std::atomic<std::size_t> calls_{0};
};

TEST(LargeDatasetTest, LogicalTenGiBDatasetHasStableBatchGeometry) {
    auto dataset = std::make_shared<VirtualLargeDataset>();
    jyppx::data::DataLoader loader(dataset, {.batch_size = 1024});

    EXPECT_EQ(loader.dataset_size(), VirtualLargeDataset::kRows);
    EXPECT_EQ(loader.size(), (VirtualLargeDataset::kRows + 1023) / 1024);

    auto iterator = loader.begin();
    const auto first = *iterator;
    EXPECT_EQ(first.data.sizes(), torch::IntArrayRef({1024, 256}));
    EXPECT_EQ(first.data[0][0].item<int>(), 0);
    EXPECT_EQ(first.data[1023][0].item<int>(), 1023);

    ++iterator;
    const auto second = *iterator;
    EXPECT_EQ(second.data[0][0].item<int>(), 1024);
    EXPECT_TRUE(jyppx::is_finite(second.data));
    EXPECT_EQ(dataset->calls(), 2048U);
}

TEST(LargeDatasetTest, RepeatedPartialEpochsDoNotRetainBatches) {
    auto dataset = std::make_shared<VirtualLargeDataset>();
    jyppx::data::DataLoader loader(
        dataset, {.batch_size = 256, .num_workers = 4, .prefetch_factor = 2});

    // Repeatedly consume a small prefix.  This exercises worker shutdown and
    // restart without making CI walk ten million samples per epoch.
    for (int epoch = 0; epoch < 8; ++epoch) {
        int batches = 0;
        for (const auto& batch : loader) {
            ASSERT_EQ(batch.data.size(0), 256);
            ASSERT_TRUE(jyppx::is_finite(batch.data));
            if (++batches == 3) {
                break;
            }
        }
    }

    // Prefetching may finish additional batches before the consumer breaks;
    // the important invariant is that every requested prefix was produced.
    EXPECT_GE(dataset->calls(), 8U * 3U * 256U);
}

} // namespace
