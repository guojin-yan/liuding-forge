// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
#include "../../../examples/serving/linear_bundle.hpp"

#include <array>
#include <iostream>
#include <stdexcept>
int main() {
  try {
    namespace example = jyppx::liuding::examples::serving;
    using namespace jyppx::liuding::serving;
    example::BundleFixture fixture;
    fixture.Publish();
    auto predictor = fixture.Load();
    const auto result =
        predictor->predict(example::Request(jyppx::liuding::Device(c10::kCPU)));
    if (!torch::equal(result.output.value.toTensor(),
                      torch::tensor({{2.0F, -3.0F}})))
      throw std::runtime_error("Installed native serving result mismatch");
    BatchOptions options;
    options.max_batch_size = 2;
    DynamicBatcher batcher(predictor, options);
    auto first =
        batcher.submit(example::Request(jyppx::liuding::Device(c10::kCPU), "first"));
    auto second =
        batcher.submit(example::Request(jyppx::liuding::Device(c10::kCPU), "second", 2));
    if (first.get().context.request_id != "first" ||
        !torch::equal(second.get().output.value.toTensor(),
                      torch::tensor({{4.0F, -6.0F}})) ||
        !batcher.close() || !predictor->close())
      throw std::runtime_error("Installed serving batch/lifecycle mismatch");
    if (serving_preview_manifest_json().find("\"default_enabled\":false") ==
        std::string::npos)
      throw std::runtime_error("Missing serving preview manifest");
    std::cout << "Verified installed jyppx::liuding::serving native bundle, forward, "
                 "batch and drain\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
