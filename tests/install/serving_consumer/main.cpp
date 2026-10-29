// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
#include "../../../examples/serving/linear_bundle.hpp"

#include <array>
#include <chrono>
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
    auto first = batcher.submit(
        example::Request(jyppx::liuding::Device(c10::kCPU), "first"));
    auto second = batcher.submit(
        example::Request(jyppx::liuding::Device(c10::kCPU), "second", 2));
    if (first.get().context.request_id != "first" ||
        !torch::equal(second.get().output.value.toTensor(),
                      torch::tensor({{4.0F, -6.0F}})) ||
        !batcher.close() || !predictor->close())
      throw std::runtime_error("Installed serving batch/lifecycle mismatch");
    if (serving_preview_manifest_json().find("\"default_enabled\":false") ==
        std::string::npos)
      throw std::runtime_error("Missing serving preview manifest");
    example::BundleFixture candidate;
    candidate.manifest.bundle_id = "example-linear-v2";
    candidate.Publish();
    LoadOptions load;
    load.max_state_bytes = 1024;
    load.max_snapshot_bytes = 65536;
    constexpr std::uint64_t reservation = 2U * 1024U + 65536U;
    ReloadablePredictor versioned(fixture.Load({}, load),
                                  {.initial_reservation_bytes = reservation,
                                   .max_resident_bytes = 2U * reservation});
    auto lease = std::make_unique<PredictionLease>(versioned.acquire());
    NativeStateBackend backend(candidate.descriptor, example::Factory());
    const std::array<WarmupCase, 1> warmup{
        example::KnownVector(jyppx::liuding::Device(c10::kCPU))};
    const auto publication =
        versioned.reload(backend, candidate.directory, {}, warmup, load,
                         {.drain_timeout = std::chrono::milliseconds(0)});
    auto request = example::Request(jyppx::liuding::Device(c10::kCPU));
    request.context.model_version.clear();
    if (publication.previous_drained ||
        lease->predict(request).runtime.model_version != "example-linear-v1" ||
        versioned.predict(request).runtime.model_version != "example-linear-v2")
      throw std::runtime_error("Installed serving version lease mismatch");
    lease.reset();
    if (!versioned.drain_previous() || !versioned.close())
      throw std::runtime_error("Installed serving version drain mismatch");
    std::cout
        << "Verified installed jyppx::liuding::serving native bundle, forward, "
           "batch, reload, pinned lease and drain\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
