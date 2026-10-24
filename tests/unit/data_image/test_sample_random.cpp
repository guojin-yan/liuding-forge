// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data_image/sample_random.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <array>
#include <future>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace {
using namespace jyppx::liuding::data::image;
using jyppx::liuding::InvalidArgument;

SampleRandomKey ReferenceKey() {
  return {.dataset_id = "dataset-sha256:v1",
          .sample_id = "image/猫.png",
          .transform_id = "geometry-cpu:v1",
          .seed = 1234,
          .epoch = 7,
          .occurrence = 2,
          .view = 3};
}

TEST(SampleRandomTest, MatchesIndependentPythonHashlibCounterVectors) {
  const SampleRandom random(ReferenceKey());
  EXPECT_EQ(random.identity(),
            "b8146d164efa308e7620b181b982ef86019a9a2654b82a6110e6b3bd576b43ef");
  const std::array<std::uint64_t, 6> expected{
      17600931540693795272ULL, 3548571629548599345ULL, 3693858113676629156ULL,
      2591400760710424990ULL,  3289164206408506529ULL, 17953479984429372879ULL};
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(random.bits(i), expected[i]);
    EXPECT_DOUBLE_EQ(random.unit(i),
                     static_cast<double>(expected[i] >> 11U) * 0x1.0p-53);
  }
  // Draw 1 rejects attempt zero for this bound, then accepts attempt one.
  EXPECT_EQ(random.bounded(9223372036854775809ULL, 1), 98459335617357023ULL);
  EXPECT_EQ(random.bits(1), expected[1]);
}

TEST(SampleRandomTest, EveryLogicalCoordinateChangesIdentity) {
  auto key = ReferenceKey();
  const auto identity = SampleRandom(key).identity();
  auto Check = [&](auto change) {
    auto other = key;
    change(other);
    EXPECT_NE(SampleRandom(other).identity(), identity);
  };
  Check([](auto& k) { k.dataset_id += "2"; });
  Check([](auto& k) { k.sample_id += "2"; });
  Check([](auto& k) { k.transform_id += "2"; });
  Check([](auto& k) { ++k.seed; });
  Check([](auto& k) { ++k.epoch; });
  Check([](auto& k) { ++k.occurrence; });
  Check([](auto& k) { ++k.view; });
  key.algorithm_version = 2;
  EXPECT_THROW(SampleRandom{key}, InvalidArgument);
}

TEST(SampleRandomTest, LengthPrefixesSeparateConcatenationAndEmbeddedNulls) {
  auto a = ReferenceKey();
  auto b = a;
  a.dataset_id = "a";
  a.sample_id = "bc";
  b.dataset_id = "ab";
  b.sample_id = "c";
  EXPECT_NE(SampleRandom(a).identity(), SampleRandom(b).identity());
  b = a;
  b.dataset_id = std::string("a\0", 2);
  EXPECT_NE(SampleRandom(a).identity(), SampleRandom(b).identity());
  a.sample_id.clear();
  EXPECT_THROW(SampleRandom{a}, InvalidArgument);
  a = ReferenceKey();
  a.transform_id.assign(4097, 'x');
  EXPECT_THROW(SampleRandom{a}, InvalidArgument);
}

TEST(SampleRandomTest, FullWidthBoundsAndParallelReadOrderAreStable) {
  const SampleRandom random(ReferenceKey());
  EXPECT_THROW(random.bounded(0, 0), InvalidArgument);
  EXPECT_EQ(random.bounded(1, 0), 0U);
  EXPECT_LT(random.bounded(std::numeric_limits<std::uint64_t>::max(), 0),
            std::numeric_limits<std::uint64_t>::max());
  std::vector<std::future<std::uint64_t>> tasks;
  for (std::uint64_t i = 0; i < 32; ++i)
    tasks.push_back(std::async(std::launch::async,
                               [&random, i] { return random.bits(31 - i); }));
  for (std::size_t i = 0; i < tasks.size(); ++i) {
    EXPECT_EQ(tasks[i].get(), random.bits(31 - i));
    EXPECT_GE(random.unit(i), 0);
    EXPECT_LT(random.unit(i), 1);
    EXPECT_LT(random.bounded(17, i), 17U);
  }
}
}  // namespace
