// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

// Unit tests for jyppx::liuding::data samplers: SequentialSampler,
// RandomSampler and BatchSampler.

#include <jyppx/liuding/data/samplers/batch_sampler.hpp>
#include <jyppx/liuding/data/samplers/random_sampler.hpp>
#include <jyppx/liuding/data/samplers/sequential_sampler.hpp>
#include <jyppx/liuding/data/samplers/weighted_random_sampler.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <algorithm>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <string>

namespace {

using namespace jyppx::liuding::data;

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

TEST(RandomSamplerTest, MatchesIndependentPortableGoldenVectors) {
  // Frozen independent integer reference; these are identical on
  // MSVC/libstdc++.
  samplers::RandomSampler sampler(10, 7);
  EXPECT_EQ(samplers::RandomSampler::algorithm_id(),
            "splitmix64-fisher-yates-rejection/v1");
  EXPECT_EQ(sampler.configuration_identity(),
            "jyppx.sampler.random/v2:algorithm="
            "splitmix64-fisher-yates-rejection/v1:size=10");
  EXPECT_EQ(sampler.indices(),
            (std::vector<size_t>{8, 1, 5, 9, 0, 4, 3, 2, 6, 7}));
  EXPECT_EQ(sampler.save_state(),
            "JYPPX-RANDOM-SAMPLER/v2\n10\n7\n10372713005361028292");
  EXPECT_EQ(sampler.indices(),
            (std::vector<size_t>{3, 2, 8, 9, 0, 7, 6, 4, 1, 5}));
  sampler.set_epoch(1);
  EXPECT_EQ(sampler.indices(),
            (std::vector<size_t>{5, 7, 0, 3, 6, 4, 8, 1, 9, 2}));
  sampler.set_epoch(std::numeric_limits<std::uint64_t>::max());
  EXPECT_EQ(sampler.indices(),
            (std::vector<size_t>{9, 7, 8, 4, 0, 3, 1, 6, 5, 2}));
}

TEST(RandomSamplerTest, DefinesSignedSeedAndEpochWrap) {
  samplers::RandomSampler sampler(10, -1);
  EXPECT_EQ(sampler.indices(),
            (std::vector<size_t>{3, 4, 2, 7, 5, 0, 8, 1, 9, 6}));
  sampler.set_epoch(1);
  EXPECT_EQ(sampler.indices(),
            (std::vector<size_t>{6, 3, 2, 9, 8, 1, 4, 7, 0, 5}));
  sampler.set_epoch(1);
  EXPECT_EQ(sampler.indices(), samplers::RandomSampler(10, 0).indices());
}

TEST(RandomSamplerTest, MatchesFullEpochOrderReferenceDigests) {
  // SHA-256 of decimal indices joined with commas, independently frozen.
  const std::string expected[] = {
      "65382656f547a74b06eb3ba45d2fd03fd38c9409721ca9c233141006f19a1874",
      "076656a26406241031176feb473cdbd6892842b9dbafe52bdaba8d599036c2d3",
      "c0d083b5d8b5195a49e151178b8b2f02641e091e1522867d70468e16d9adf413"};
  samplers::RandomSampler sampler(1000, 20261029);
  for (std::uint64_t epoch = 0; epoch < 3; ++epoch) {
    sampler.set_epoch(epoch);
    std::string csv;
    for (const auto index : sampler.indices()) {
      if (!csv.empty()) csv += ',';
      csv += std::to_string(index);
    }
    EXPECT_EQ(jyppx::liuding::sha256_hex(
                  {reinterpret_cast<const std::byte*>(csv.data()), csv.size()}),
              expected[epoch]);
  }
}

TEST(RandomSamplerTest, RejectsBiasedWordBeforeBoundedSelection) {
  // This seed's first output is zero. Bound 3 must reject it (2^64 mod 3 = 1).
  const samplers::RandomSampler sampler(3, INT64_C(7046029254386353131));
  EXPECT_EQ(sampler.indices(), (std::vector<size_t>{2, 0, 1}));
  EXPECT_EQ(
      sampler.save_state(),
      "JYPPX-RANDOM-SAMPLER/v2\n3\n7046029254386353131\n4354685564936845354");
}

TEST(RandomSamplerTest, RestoresPortableStateAndFutureEpochSeed) {
  samplers::RandomSampler sampler(10, 999);
  // This canonical byte string is equally consumable on both platforms.
  sampler.load_state("JYPPX-RANDOM-SAMPLER/v2\n10\n7\n10372713005361028292");
  EXPECT_EQ(sampler.indices(),
            (std::vector<size_t>{3, 2, 8, 9, 0, 7, 6, 4, 1, 5}));
  sampler.set_epoch(1);
  EXPECT_EQ(sampler.indices(),
            (std::vector<size_t>{5, 7, 0, 3, 6, 4, 8, 1, 9, 2}));
}

TEST(RandomSamplerTest, RejectsInvalidOrRetiredStateWithoutMutation) {
  samplers::RandomSampler sampler(10, 7);
  (void)sampler.indices();
  const auto initial_state = sampler.save_state();
  for (const std::string state :
       {"", "JYPPX-RANDOM-SAMPLER/v1\n7\n0",
        "JYPPX-RANDOM-SAMPLER/v3\n10\n7\n0",
        "JYPPX-RANDOM-SAMPLER/v2\n11\n7\n0",
        "JYPPX-RANDOM-SAMPLER/v2\n10\n-1\n0",
        "JYPPX-RANDOM-SAMPLER/v2\n10\n+7\n0",
        "JYPPX-RANDOM-SAMPLER/v2\n010\n7\n0",
        "JYPPX-RANDOM-SAMPLER/v2\n10\n7\n18446744073709551616",
        "JYPPX-RANDOM-SAMPLER/v2\n10\n7\n",
        "JYPPX-RANDOM-SAMPLER/v2\n10\n7\n0\n",
        "JYPPX-RANDOM-SAMPLER/v2\n10\n7\n0 trailing",
        "JYPPX-RANDOM-SAMPLER/v2\n10\r\n7\n0"}) {
    SCOPED_TRACE(state);
    EXPECT_THROW(sampler.load_state(state), jyppx::liuding::SerializationError);
    EXPECT_EQ(sampler.save_state(), initial_state);
  }
}

TEST(RandomSamplerTest, SingletonConsumesNoRandomWordsAndEmptyIsRejected) {
  const samplers::RandomSampler sampler(1, 7);
  const auto state = sampler.save_state();
  EXPECT_EQ(sampler.indices(), (std::vector<size_t>{0}));
  EXPECT_EQ(sampler.save_state(), state);
  EXPECT_THROW(samplers::RandomSampler(0, 7), jyppx::liuding::InvalidArgument);
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
  EXPECT_THROW(samplers::BatchSampler(nullptr, 1),
               jyppx::liuding::InvalidArgument);
  auto base = std::make_shared<samplers::SequentialSampler>(8);
  EXPECT_THROW(samplers::BatchSampler(base, 0),
               jyppx::liuding::InvalidArgument);
}

TEST(BatchSamplerTest, CountsMaximumPopulationWithoutOverflow) {
  constexpr size_t kMaximum = std::numeric_limits<size_t>::max();
  auto base = std::make_shared<SizeOnlySampler>(kMaximum);

  const samplers::BatchSampler keep_tail(base, 2, /*drop_last=*/false);
  const samplers::BatchSampler drop_tail(base, 2, /*drop_last=*/true);

  EXPECT_EQ(keep_tail.size(), kMaximum / 2 + 1);
  EXPECT_EQ(drop_tail.size(), kMaximum / 2);
}

TEST(SamplerContractTest, PublishesAndRestoresCurrentStateCapabilities) {
  const SizeOnlySampler state_less(4);
  EXPECT_FALSE(state_less.supports_state());

  auto subset = std::make_shared<samplers::SubsetRandomSampler>(
      std::vector<size_t>{2, 4, 6, 8}, 2026);
  EXPECT_EQ(subset->size(), 4U);
  EXPECT_TRUE(subset->supports_state());

  samplers::BatchSampler batches(subset, 2, /*drop_last=*/false);
  EXPECT_TRUE(batches.supports_state());
  const auto batch_state = batches.save_state();
  const auto expected_subset = subset->indices();
  batches.load_state(batch_state);
  EXPECT_EQ(subset->indices(), expected_subset);

  samplers::DistributedSampler distributed(
      /*size=*/9, /*rank=*/1, /*world_size=*/2, /*shuffle=*/true,
      /*seed=*/17, /*drop_last=*/false);
  EXPECT_TRUE(distributed.supports_epoch_control());
  EXPECT_TRUE(distributed.supports_state());
  EXPECT_NE(distributed.configuration_identity().find("world_size=2"),
            std::string::npos);
  distributed.set_epoch(7);
  const auto distributed_state = distributed.save_state();
  distributed.set_epoch(8);
  distributed.load_state(distributed_state);
  EXPECT_EQ(distributed.epoch(), 7U);
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
      jyppx::liuding::InvalidArgument);
  EXPECT_THROW(
      samplers::WeightedRandomSampler(
          {1.0, std::numeric_limits<double>::infinity()}, /*num_samples=*/1),
      jyppx::liuding::InvalidArgument);
}

TEST(WeightedRandomSamplerTest, InvalidStateDoesNotAdvanceGenerator) {
  samplers::WeightedRandomSampler sampler({1.0, 2.0, 3.0}, 16, true, 9);
  samplers::WeightedRandomSampler control({1.0, 2.0, 3.0}, 16, true, 9);
  const std::string invalid_state = sampler.save_state() + " trailing";

  try {
    sampler.load_state(invalid_state);
    FAIL() << "malformed sampler state must be rejected";
  } catch (const jyppx::liuding::SerializationError& error) {
    EXPECT_EQ(error.code(), jyppx::liuding::ErrorCode::serialization);
    EXPECT_EQ(error.details().component, "data.weighted_random_sampler");
    EXPECT_EQ(error.details().operation, "load_state");
  }
  EXPECT_EQ(sampler.indices(), control.indices());
}

TEST(DistributedSamplerTest, RejectsEmptyPopulation) {
  EXPECT_THROW(samplers::DistributedSampler(0, /*rank=*/0, /*world_size=*/1),
               jyppx::liuding::InvalidArgument);
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
