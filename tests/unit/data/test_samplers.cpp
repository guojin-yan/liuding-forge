// Unit tests for jyppx::data samplers: SequentialSampler, RandomSampler and
// BatchSampler.

#include <jyppx/data/samplers/batch_sampler.hpp>
#include <jyppx/data/samplers/random_sampler.hpp>
#include <jyppx/data/samplers/sequential_sampler.hpp>
#include <jyppx/data/samplers/weighted_random_sampler.hpp>
#include <jyppx/foundation/error.hpp>

#include <algorithm>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <string>

namespace {

using namespace jyppx::data;

class SizeOnlySampler final : public samplers::Sampler {
 public:
  explicit SizeOnlySampler(size_t size) : size_(size) {}

  std::vector<size_t> indices() const override { return {}; }

  size_t size() const override { return size_; }

 private:
  size_t size_;
};

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
  EXPECT_EQ(sampler.size(), 3);  // 4 + 4 + 2
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

TEST(BatchSamplerTest, RejectsNullSamplerAndZeroBatchSize) {
  EXPECT_THROW(samplers::BatchSampler(nullptr, 1), jyppx::InvalidArgument);
  auto base = std::make_shared<samplers::SequentialSampler>(8);
  EXPECT_THROW(samplers::BatchSampler(base, 0), jyppx::InvalidArgument);
}

TEST(BatchSamplerTest, CountsMaximumPopulationWithoutOverflow) {
  constexpr size_t kMaximum = std::numeric_limits<size_t>::max();
  auto base = std::make_shared<SizeOnlySampler>(kMaximum);

  const samplers::BatchSampler keep_tail(base, 2, /*drop_last=*/false);
  const samplers::BatchSampler drop_tail(base, 2, /*drop_last=*/true);

  EXPECT_EQ(keep_tail.size(), kMaximum / 2 + 1);
  EXPECT_EQ(drop_tail.size(), kMaximum / 2);
}

TEST(DistributedSamplerTest, CountsMaximumPopulationWithoutOverflow) {
  constexpr size_t kMaximum = std::numeric_limits<size_t>::max();
  const samplers::DistributedSampler keep_tail(
      kMaximum, /*rank=*/0, /*world_size=*/2, /*shuffle=*/false,
      /*seed=*/std::nullopt, /*drop_last=*/false);
  const samplers::DistributedSampler drop_tail(
      kMaximum, /*rank=*/0, /*world_size=*/2, /*shuffle=*/false,
      /*seed=*/std::nullopt, /*drop_last=*/true);

  EXPECT_EQ(keep_tail.size(), kMaximum / 2 + 1);
  EXPECT_EQ(drop_tail.size(), kMaximum / 2);
}

TEST(WeightedRandomSamplerTest, RestoresExactContinuationInBothModes) {
  const auto expect_exact_continuation = [](bool replacement) {
    samplers::WeightedRandomSampler sampler({1.0, 0.0, 2.0, 4.0}, 3,
                                            replacement, 20261018);
    (void)sampler.indices();
    const auto state = sampler.save_state();
    const auto expected = sampler.indices();
    (void)sampler.indices();

    sampler.load_state(state);
    EXPECT_EQ(sampler.indices(), expected);
  };

  expect_exact_continuation(true);
  expect_exact_continuation(false);
}

TEST(WeightedRandomSamplerTest, NeverSelectsZeroMassWithoutReplacement) {
  samplers::WeightedRandomSampler sampler({1.0, 0.0, 2.0, 0.0, 4.0}, 3,
                                          /*replacement=*/false, 91);
  const auto indices = sampler.indices();

  EXPECT_EQ(indices.size(), 3U);
  EXPECT_EQ(std::set<size_t>(indices.begin(), indices.end()),
            std::set<size_t>({0, 2, 4}));
}

TEST(WeightedRandomSamplerTest, RejectsNonTerminatingInputs) {
  EXPECT_THROW(
      samplers::WeightedRandomSampler({1.0, 0.0}, /*num_samples=*/2,
                                      /*replacement=*/false, /*seed=*/7),
      jyppx::InvalidArgument);
  EXPECT_THROW(
      samplers::WeightedRandomSampler(
          {1.0, std::numeric_limits<double>::infinity()}, /*num_samples=*/1),
      jyppx::InvalidArgument);
}

TEST(WeightedRandomSamplerTest, InvalidStateDoesNotAdvanceGenerator) {
  samplers::WeightedRandomSampler sampler({1.0, 2.0, 3.0}, 16, true, 9);
  samplers::WeightedRandomSampler control({1.0, 2.0, 3.0}, 16, true, 9);
  const std::string invalid_state = sampler.save_state() + " trailing";

  try {
    sampler.load_state(invalid_state);
    FAIL() << "malformed sampler state must be rejected";
  } catch (const jyppx::SerializationError& error) {
    EXPECT_EQ(error.code(), jyppx::ErrorCode::serialization);
    EXPECT_EQ(error.details().component, "data.weighted_random_sampler");
    EXPECT_EQ(error.details().operation, "load_state");
  }
  EXPECT_EQ(sampler.indices(), control.indices());
}

TEST(DistributedSamplerTest, RejectsEmptyPopulation) {
  EXPECT_THROW(samplers::DistributedSampler(0, /*rank=*/0, /*world_size=*/1),
               jyppx::InvalidArgument);
}

TEST(SubsetRandomSamplerTest, RestoresExactContinuation) {
  samplers::SubsetRandomSampler sampler({3, 5, 8, 13, 21}, 34);
  (void)sampler.indices();
  const auto state = sampler.save_state();
  const auto expected = sampler.indices();
  (void)sampler.indices();

  sampler.load_state(state);
  EXPECT_EQ(sampler.indices(), expected);
}

}  // namespace
