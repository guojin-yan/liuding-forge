// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data/samplers/weighted_random_sampler.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

int main(int argc, char** argv) {
  try {
    const std::string_view mode = argc == 2 ? argv[1] : "";
    const double tiny = std::numeric_limits<double>::denorm_min();
    std::vector<double> weights;
    if (mode == "tiny") {
      weights = {tiny, 0.0};
    } else if (mode == "leading-zero") {
      weights = {0.0, tiny, 0.0};
    } else if (mode == "ordinary") {
      weights = {1.0, 0.0, 2.0, 0.0};
    } else {
      throw std::invalid_argument("Expected tiny|leading-zero|ordinary");
    }
    jyppx::liuding::data::samplers::WeightedRandomSampler sampler(
        weights, 4096, true, 7);
    const auto draws = sampler.indices();
    const auto zero_count = std::count_if(
        draws.begin(), draws.end(), [&](std::size_t index) {
          return index >= weights.size() || weights[index] == 0.0;
        });
    std::cout << "weights=" << mode << " draws=" << draws.size()
              << " zero_weight_draws=" << zero_count
              << " sha256="
              << jyppx::liuding::sha256_hex(std::as_bytes(std::span(draws)))
              << '\n';
    if (zero_count != 0) {
      throw std::runtime_error("Replacement sampler selected zero weight");
    }
    const auto state = sampler.save_state();
    const auto continuation = sampler.indices();
    sampler.load_state(state);
    if (sampler.indices() != continuation) {
      throw std::runtime_error("Weighted sampler continuation changed");
    }
    if (mode == "ordinary" &&
        (std::find(draws.begin(), draws.end(), 0) == draws.end() ||
         std::find(draws.begin(), draws.end(), 2) == draws.end())) {
      throw std::runtime_error("Ordinary positive weights were not sampled");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
