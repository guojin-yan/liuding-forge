// Worker-pool and interruption stress tests.

#include <jyppx/data/dataloader.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <future>
#include <memory>
#include <utility>

namespace {

using jyppx::Tensor;

class CountingDataset final : public jyppx::data::Dataset<Tensor> {
public:
    explicit CountingDataset(std::size_t count) : count_(count) {}

    Tensor get(std::size_t index) const override {
        calls_.fetch_add(1, std::memory_order_relaxed);
        return torch::tensor({static_cast<float>(index)});
    }

    std::size_t size() const override { return count_; }
    std::size_t calls() const { return calls_.load(std::memory_order_relaxed); }

private:
    std::size_t count_;
    mutable std::atomic<std::size_t> calls_{0};
};

TEST(ConcurrentStressTest, SixtyFourWorkersPreserveOrderAndCompleteness) {
    auto dataset = std::make_shared<CountingDataset>(512);
    jyppx::data::DataLoader loader(
        dataset, {.batch_size = 8, .num_workers = 64, .prefetch_factor = 1});

    std::size_t samples = 0;
    float expected = 0.0F;
    for (const auto& batch : loader) {
        ASSERT_EQ(batch.data.size(0), 8);
        for (int64_t row = 0; row < batch.data.size(0); ++row) {
            EXPECT_FLOAT_EQ(batch.data[row].item<float>(), expected++);
        }
        samples += static_cast<std::size_t>(batch.data.size(0));
    }
    EXPECT_EQ(samples, 512U);
    EXPECT_EQ(dataset->calls(), 512U);
}

TEST(ConcurrentStressTest, MixedLoaderInstancesCanRunConcurrently) {
    auto run_loader = [](float offset) {
        auto dataset = std::make_shared<CountingDataset>(128);
        jyppx::data::DataLoader loader(
            dataset, {.batch_size = 4, .num_workers = 8, .prefetch_factor = 2});
        float checksum = 0.0F;
        for (const auto& batch : loader) {
            checksum += batch.data.sum().item<float>() + offset;
        }
        return std::pair<float, std::size_t>{checksum, dataset->calls()};
    };

    auto first = std::async(std::launch::async, run_loader, 1.0F);
    auto second = std::async(std::launch::async, run_loader, 2.0F);
    const auto first_result = first.get();
    const auto second_result = second.get();
    EXPECT_EQ(first_result.second, 128U);
    EXPECT_EQ(second_result.second, 128U);
    EXPECT_GT(second_result.first, first_result.first);
}

TEST(ConcurrentStressTest, InterruptAndResumeAcrossManyEpochsIsClean) {
    auto dataset = std::make_shared<CountingDataset>(96);
    jyppx::data::DataLoader loader(
        dataset, {.batch_size = 6, .shuffle = true, .num_workers = 16, .seed = 17});

    for (int epoch = 0; epoch < 12; ++epoch) {
        int batches = 0;
        for (const auto& batch : loader) {
            ASSERT_FALSE(batch.empty());
            if (++batches == 2) {
                break;
            }
        }
        loader.reset();
    }
    EXPECT_GE(dataset->calls(), 12U * 2U * 6U);
}

} // namespace
