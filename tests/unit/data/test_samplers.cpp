// Unit tests for jyppx::data samplers: SequentialSampler, RandomSampler and
// BatchSampler.

#include <jyppx/data/samplers/batch_sampler.hpp>
#include <jyppx/data/samplers/random_sampler.hpp>
#include <jyppx/data/samplers/sequential_sampler.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <set>

namespace {

using namespace jyppx::data;

TEST(SequentialSamplerTest, YieldsAscendingIndices) {
    const samplers::SequentialSampler sampler(5);
    const auto indices = sampler.indices();
    const std::vector<size_t> expected{0, 1, 2, 3, 4};
    EXPECT_EQ(indices, expected);
    EXPECT_EQ(sampler.size(), 5);
}

TEST(RandomSamplerTest, WithSeedIsDeterministic) {
    const samplers::RandomSampler first(16, 123);
    const samplers::RandomSampler second(16, 123);
    EXPECT_EQ(first.indices(), second.indices());
}

TEST(RandomSamplerTest, ProducesAPermutation) {
    const samplers::RandomSampler sampler(32, 7);
    const auto indices = sampler.indices();

    ASSERT_EQ(indices.size(), 32);
    std::set<size_t> unique(indices.begin(), indices.end());
    EXPECT_EQ(unique.size(), indices.size());
    EXPECT_TRUE(*std::min_element(indices.begin(), indices.end()) == 0);
    EXPECT_TRUE(*std::max_element(indices.begin(), indices.end()) == 31);
}

TEST(RandomSamplerTest, UnseededCallsVary) {
    const samplers::RandomSampler sampler(64);
    const auto first = sampler.indices();
    const auto second = sampler.indices();
    // Two draws from an unseeded generator are virtually never identical.
    EXPECT_NE(first, second);
}

TEST(BatchSamplerTest, SplitsIntoFullBatchesAndRemainder) {
    auto base = std::make_shared<samplers::SequentialSampler>(10);
    const samplers::BatchSampler sampler(base, 4, /*drop_last=*/false);

    const auto batches = sampler.indices();
    // The batch sampler flattens all indices; batches are derived by size().
    EXPECT_EQ(batches.size(), 10);
    EXPECT_EQ(sampler.size(), 3); // 4 + 4 + 2
    EXPECT_EQ(sampler.batch_size(), 4);
}

TEST(BatchSamplerTest, DropLastRemovesIncompleteBatch) {
    auto base = std::make_shared<samplers::SequentialSampler>(10);
    const samplers::BatchSampler sampler(base, 4, /*drop_last=*/true);

    const auto indices = sampler.indices();
    EXPECT_EQ(indices.size(), 8);
    EXPECT_EQ(sampler.size(), 2);
}

TEST(BatchSamplerTest, ExactMultipleHasNoRemainder) {
    auto base = std::make_shared<samplers::SequentialSampler>(8);
    const samplers::BatchSampler sampler(base, 4, /*drop_last=*/false);
    EXPECT_EQ(sampler.size(), 2);
}

} // namespace
