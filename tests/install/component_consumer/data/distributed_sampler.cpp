// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data/samplers/weighted_random_sampler.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <exception>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

int main(int argc, char** argv) {
  try {
    const std::string_view mode = argc == 2 ? argv[1] : "";
    jyppx::liuding::data::samplers::DistributedSampler sampler(
        9, 1, 2, true, 17, false);
    if (mode == "ordinary") {
      for (const auto epoch :
           {std::uint64_t{0}, std::uint64_t{7},
            std::numeric_limits<std::uint64_t>::max()}) {
        sampler.set_epoch(epoch);
        const auto expected = sampler.indices();
        const auto state = sampler.save_state();
        sampler.set_epoch(3);
        sampler.load_state(state);
        const auto actual = sampler.indices();
        if (sampler.epoch() != epoch || sampler.save_state() != state ||
            actual != expected || actual.size() != sampler.size()) {
          throw std::runtime_error("Canonical distributed epoch did not restore");
        }
        std::cout << "epoch=" << epoch << " draws=" << actual.size()
                  << " sha256="
                  << jyppx::liuding::sha256_hex(std::as_bytes(std::span(actual)))
                  << '\n';
      }
      return 0;
    }

    std::vector<std::string> invalid;
    if (mode == "negative") {
      invalid = {"-1", " -1", "\t-1", "\n-1"};
    } else if (mode == "malformed") {
      invalid = {"", "+1", " 1", "1 ", "01", "\t1", "1\n",
                 "18446744073709551616", std::string("1\0junk", 6)};
    } else {
      throw std::invalid_argument("Expected negative|malformed|ordinary");
    }
    std::size_t rejected = 0;
    std::size_t accepted = 0;
    std::size_t mutated = 0;
    for (const auto& state : invalid) {
      sampler.set_epoch(7);
      const auto before = sampler.save_state();
      const auto before_indices = sampler.indices();
      try {
        sampler.load_state(state);
        ++accepted;
      } catch (const jyppx::liuding::SerializationError&) {
        ++rejected;
      }
      if (sampler.save_state() != before || sampler.indices() != before_indices) {
        ++mutated;
      }
    }
    std::cout << "case=" << mode << " rejected=" << rejected
              << " accepted=" << accepted << " mutated=" << mutated << '\n';
    if (accepted != 0 || mutated != 0 || rejected != invalid.size()) {
      throw std::runtime_error("Invalid epoch was accepted or changed live state");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
