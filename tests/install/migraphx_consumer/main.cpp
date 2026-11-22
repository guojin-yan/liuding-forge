// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/sha256.hpp>
#ifndef JYPPX_MIGRAPHX_CONSUMER_FOUNDATION
#include <jyppx/liuding/deployment/migraphx_runtime/environment.hpp>
#include <jyppx/liuding/deployment/migraphx_runtime/session.hpp>
#endif
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
  try {
#ifdef JYPPX_MIGRAPHX_CONSUMER_FOUNDATION
    const std::vector<std::byte> empty;
    return jyppx::liuding::sha256_hex(empty) ==
                   "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b785"
                   "2b855"
               ? 0
               : 1;
#else
    if (argc != 2) throw std::runtime_error("expected immutable fixture root");
    const auto root = std::filesystem::path(argv[1]) / "linear";
    const auto read = [&](const char* name, const char* digest) {
      const auto path = root / name;
      const auto size = std::filesystem::file_size(path);
      if (size == 0 || size > (1U << 20U))
        throw std::runtime_error("fixture size");
      std::vector<std::byte> bytes(static_cast<std::size_t>(size));
      std::ifstream stream(path, std::ios::binary);
      stream.read(reinterpret_cast<char*>(bytes.data()),
                  static_cast<std::streamsize>(size));
      if (!stream || jyppx::liuding::sha256_hex(bytes) != digest)
        throw std::runtime_error("fixture SHA");
      return bytes;
    };
    constexpr const char* graph_sha =
        "7b4f9bf96cd72143e22c906b0063958ef05b10534785eac3e9fc1b07afcb990a";
    const auto graph = read("model.onnx", graph_sha);
    const auto input_bytes = read(
        "case-0-input.f32",
        "d55f7abaa5bb2c75cc098aa3dc97579967a294f802b308803833fef7964a94e9");
    const auto expected_bytes = read(
        "case-0-output.f32",
        "3b106db38a6dca7aafe95d1af546237ca38ac4d74a084a8f1b0cea5a361c709a");
    if (input_bytes.size() != 16U * sizeof(float) ||
        expected_bytes.size() != 6U * sizeof(float))
      throw std::runtime_error("fixture byte count");
    std::vector<float> input(16), expected(6);
    std::memcpy(input.data(), input_bytes.data(), input_bytes.size());
    std::memcpy(expected.data(), expected_bytes.data(), expected_bytes.size());
    namespace mgx = jyppx::liuding::migraphx_runtime;
    mgx::RunResult result;
    {
      mgx::Session session(graph, {.hex = graph_sha}, {.input_shape = {2, 8}});
      const auto observed = mgx::observe_environment(session);
      const auto encoded = mgx::serialize_environment_lock(observed);
      const auto lock_digest =
          [](std::string_view text) -> jyppx::liuding::ArtifactDigest {
        return {.hex = jyppx::liuding::sha256_hex(
                    std::as_bytes(std::span(text.data(), text.size())))};
      };
      static_cast<void>(
          mgx::verify_environment_lock(session, encoded, lock_digest(encoded)));
      auto drift = observed;
      --drift.hip_runtime_version;
      const auto changed = mgx::serialize_environment_lock(drift);
      bool rejected = false;
      try {
        static_cast<void>(mgx::verify_environment_lock(session, changed,
                                                       lock_digest(changed)));
      } catch (const jyppx::liuding::CompatibilityError&) {
        rejected = true;
      }
      if (!rejected) throw std::runtime_error("environment drift accepted");
      std::ofstream lock("environment-lock.json", std::ios::binary);
      lock << encoded;
      lock.close();
      if (!lock) throw std::runtime_error("environment evidence write failure");
      result = session.run(input);
    }
    if (result.shape != std::vector<std::size_t>{2, 3} ||
        result.values.size() != 6 || result.instructions.empty())
      throw std::runtime_error("output metadata");
    for (std::size_t i = 0; i < 6; ++i)
      if (!std::isfinite(result.values[i]) ||
          std::abs(static_cast<double>(result.values[i]) - expected[i]) >
              1e-5 + 1e-4 * std::abs(static_cast<double>(expected[i])))
        throw std::runtime_error("numerical mismatch");
    const auto bytes = std::as_bytes(std::span(result.values));
    std::ofstream output("actual.f32", std::ios::binary);
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    output.close();
    std::ofstream trace("dispatch.txt");
    for (const auto& name : result.instructions) trace << name << '\n';
    trace.close();
    if (!output || !trace) throw std::runtime_error("evidence write failure");
    std::cout << "installed_migraphx_output_sha256="
              << jyppx::liuding::sha256_hex(bytes) << '\n';
    return 0;
#endif
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
