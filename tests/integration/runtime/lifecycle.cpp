// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/core/runtime.hpp>

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <charconv>
#include <iostream>
#include <stdexcept>
#include <string_view>

int main(int argc, char** argv) {
  try {
    if (argc != 3) return 2;
    int threads = 0;
    const std::string_view count(argv[1]), mode(argv[2]);
    const auto parsed =
        std::from_chars(count.data(), count.data() + count.size(), threads);
    if (parsed.ec != std::errc{} || parsed.ptr != count.data() + count.size() ||
        threads < 2 || threads > 8 || (mode != "normal" && mode != "unwind"))
      return 2;
    const auto before = at::get_num_threads();
    const auto interop = at::get_num_interop_threads();
    jyppx::liuding::initialize_runtime();
    if (at::get_num_threads() != before ||
        at::get_num_interop_threads() != interop)
      throw std::runtime_error(
          "Runtime initialization changed configured parallelism");
    at::set_num_threads(threads);
    jyppx::liuding::initialize_runtime();
    if (at::get_num_threads() != threads ||
        at::get_num_interop_threads() != interop)
      throw std::runtime_error(
          "Repeated runtime initialization changed parallelism");
    bool unwound = false;
    try {
      auto input = at::arange(1, 65537, at::TensorOptions().dtype(at::kDouble));
      input.set_requires_grad(true);
      const auto loss = input.square().sum();
      loss.backward();
      if (!at::equal(input.grad(), input.detach() * 2))
        throw std::runtime_error(
            "CPU gradient disagrees with independent derivative");
      if (mode == "unwind")
        throw std::runtime_error("intentional task failure");
    } catch (const std::runtime_error& error) {
      if (mode != "unwind" ||
          std::string_view(error.what()) != "intentional task failure")
        throw;
      unwound = true;
    }
    std::cout << "{\"body_completed\":true,\"threads\":"
              << at::get_num_threads()
              << ",\"gradient_checked\":true,\"unwound\":"
              << (unwound ? "true" : "false") << "}\n"
              << std::flush;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
