// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data/chain_dataset.hpp>
#include <jyppx/liuding/data/stack_dataset.hpp>
#include <jyppx/liuding/data/subset.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <vector>

namespace {

namespace data = jyppx::liuding::data;
constexpr auto kMaximum = std::numeric_limits<std::size_t>::max();

// A logical range without a sample array, even at the largest legal size.
class IndexDataset final : public data::Dataset<std::size_t> {
 public:
  explicit IndexDataset(std::size_t count) : count_(count) {}

  std::size_t get(std::size_t index) const override {
    if (index >= count_) {
      throw jyppx::liuding::InvalidArgument("IndexDataset index out of range");
    }
    return index;
  }

  std::size_t size() const override { return count_; }

 private:
  std::size_t count_;
};

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Action>
void require_invalid(Action action, std::string_view component,
                     std::string_view operation) {
  try {
    action();
  } catch (const jyppx::liuding::InvalidArgument& error) {
    require(error.details().component == component &&
                error.details().operation == operation,
            "Incorrect structured dataset error");
    return;
  }
  throw std::runtime_error("Invalid dataset size was accepted");
}

void concat_contract() {
  auto maximum = std::make_shared<IndexDataset>(kMaximum);
  auto one = std::make_shared<IndexDataset>(1);
  auto empty = std::make_shared<IndexDataset>(0);
  require_invalid([&] { data::ConcatDataset<std::size_t> view({maximum, one}); },
                  "data.concat_dataset", "constructor.sizes");
  require_invalid([&] { data::ChainDataset<std::size_t> view({maximum, one}); },
                  "data.concat_dataset", "constructor.sizes");

  auto prefix = std::make_shared<IndexDataset>(kMaximum - 1);
  const data::ConcatDataset<std::size_t> view({empty, prefix, empty, one, empty});
  require(view.size() == kMaximum && view.get(0) == 0 &&
              view.get(kMaximum - 2) == kMaximum - 2 &&
              view.get(kMaximum - 1) == 0,
          "Maximum concatenation or empty-child addressing failed");
  require_invalid([&] { (void)view.get(kMaximum); }, "data.concat_dataset",
                  "get");
}

void split_contract() {
  auto dataset = std::make_shared<IndexDataset>(10);
  require_invalid(
      [&] { (void)data::random_split(dataset, {kMaximum, 11}, 42); },
      "data.random_split", "lengths");
  require_invalid([&] { (void)data::random_split(dataset, {11}, 42); },
                  "data.random_split", "lengths");
  require_invalid([&] { (void)data::random_split(dataset, {4, 5}, 42); },
                  "data.random_split", "lengths");

  const std::vector<std::size_t> lengths{0, 4, 0, 6, 0};
  const auto first = data::random_split(dataset, lengths, 42);
  const auto second = data::random_split(dataset, lengths, 42);
  std::vector<std::size_t> seen;
  for (std::size_t i = 0; i < lengths.size(); ++i) {
    require(first[i]->size() == lengths[i] &&
                first[i]->indices() == second[i]->indices(),
            "Zero-length partition or seeded replay changed");
    for (std::size_t j = 0; j < first[i]->size(); ++j) {
      seen.push_back(first[i]->get(j));
    }
  }
  std::sort(seen.begin(), seen.end());
  for (std::size_t i = 0; i < dataset->size(); ++i) {
    require(seen.at(i) == i, "Split omitted or repeated a sample");
  }
  auto empty = std::make_shared<IndexDataset>(0);
  const auto empty_views = data::random_split(empty, {0, 0}, 42);
  require(empty_views.size() == 2 && empty_views[0]->empty() &&
              empty_views[1]->empty(),
          "Empty dataset split changed");
}

void stack_contract() {
  auto maximum = std::make_shared<IndexDataset>(kMaximum);
  const data::StackDataset<std::size_t, std::size_t> view(maximum, maximum);
  const auto last = view.get(kMaximum - 1);
  require(view.size() == kMaximum && std::get<0>(last) == kMaximum - 1 &&
              std::get<1>(last) == kMaximum - 1,
          "Maximum stack size was mistaken for an empty dataset");
  require_invalid([&] { (void)view.get(kMaximum); }, "data.stack_dataset",
                  "get");
  auto small = std::make_shared<IndexDataset>(3);
  const data::StackDataset<std::size_t, std::size_t> prefix(maximum, small);
  require(prefix.size() == 3 && std::get<0>(prefix.get(2)) == 2 &&
              std::get<1>(prefix.get(2)) == 2,
          "Stack minimum-prefix behavior changed");
  auto empty = std::make_shared<IndexDataset>(0);
  require_invalid(
      [&] { data::StackDataset<std::size_t, std::size_t> invalid(maximum, empty); },
      "data.stack_dataset", "constructor.datasets");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string_view mode = argc == 2 ? argv[1] : "";
    if (mode == "concat") {
      concat_contract();
    } else if (mode == "split") {
      split_contract();
    } else if (mode == "stack") {
      stack_contract();
    } else {
      throw std::invalid_argument("Expected concat|split|stack");
    }
    std::cout << "dataset_view=" << mode << " passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
