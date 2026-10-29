// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// This example shares the generated data-only bundle builder with the JSONL
// example. It needs no downloaded weights, Python process or network listener.
#include "linear_bundle.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
namespace {
namespace app = jyppx::liuding::examples::serving;
using namespace jyppx::liuding;
using namespace jyppx::liuding::serving;
void SetCandidateState(app::BundleFixture& bundle) {
  bundle.manifest.bundle_id = "example-linear-v2";
  const std::array<float, 2> weights{5.0F, 7.0F};
  const interop::SafeTensorSource tensor{
      .name = "weight",
      .dtype = interop::SafeTensorDtype::f32,
      .shape = {2},
      .data = std::as_bytes(std::span(weights)),
      .byte_order = interop::SafeTensorByteOrder::little};
  const auto shards =
      interop::shard_safetensors_for_bundle(std::span(&tensor, 1), 4096);
  if (shards.size() != 1)
    throw std::runtime_error("Expected one small example state shard");
  const auto file =
      std::find_if(bundle.manifest.files.begin(), bundle.manifest.files.end(),
                   [](const auto& item) {
                     return item.role == interop::BundleFileRole::weights;
                   });
  if (file == bundle.manifest.files.end())
    throw std::runtime_error("Missing example state shard");
  const auto source =
      std::find_if(bundle.sources.begin(), bundle.sources.end(),
                   [&](const auto& item) { return item.file_id == file->id; });
  if (source == bundle.sources.end())
    throw std::runtime_error("Missing example state source");
  app::Write(source->source_path, shards.front().bytes);
  *file = shards.front().file;
  bundle.manifest.models.front().file_ids = {file->id};
  bundle.manifest.models.front().tensors = shards.front().tensors;
}
}  // namespace
int main() {
  try {
    using namespace std::chrono_literals;
    app::BundleFixture initial;
    initial.Publish();
    app::BundleFixture candidate;
    SetCandidateState(candidate);
    candidate.Publish();

    LoadOptions load;
    load.max_state_bytes = 1024;
    load.max_snapshot_bytes = 65536;
    const auto reservation = 2 * load.max_state_bytes + load.max_snapshot_bytes;
    ReloadablePredictor predictor(initial.Load({}, load),
                                  {.initial_reservation_bytes = reservation,
                                   .max_resident_bytes = 2 * reservation});
    // Acquiring before publication pins v1 for the entire request lifetime.
    auto previous = std::make_unique<PredictionLease>(predictor.acquire());
    auto known = app::KnownVector(Device(c10::kCPU));
    known.expected = torch::tensor({{5.0F, 7.0F}});
    const std::array<WarmupCase, 1> warmup{known};
    NativeStateBackend backend(candidate.descriptor, app::Factory());
    const auto publication = predictor.reload(
        backend, candidate.directory, {}, warmup, load, {.drain_timeout = 0ms});
    auto request = app::Request(Device(c10::kCPU));
    request.context.model_version.clear();
    const auto old_result = previous->predict(request);
    const auto new_result = predictor.predict(request);
    if (publication.previous_drained ||
        old_result.runtime.model_version != "example-linear-v1" ||
        new_result.runtime.model_version != "example-linear-v2" ||
        !torch::equal(old_result.output.value.toTensor(),
                      torch::tensor({{2.0F, -3.0F}})) ||
        !torch::equal(new_result.output.value.toTensor(),
                      torch::tensor({{5.0F, 7.0F}})))
      throw std::runtime_error(
          "Version publication or numerical result mismatch");
    std::cout << "pinned " << old_result.runtime.model_version << ": [2,-3]\n"
              << "current " << new_result.runtime.model_version << ": [5,7]\n";
    previous.reset();
    if (!predictor.drain_previous(std::chrono::steady_clock::now() + 1s))
      throw std::runtime_error("Previous version did not drain");
    const auto state = predictor.statistics();
    std::cout << "resident_instances=" << state.resident_instances
              << ", peak_reserved_bytes=" << state.peak_reserved_bytes << '\n';
    if (!predictor.close(std::chrono::steady_clock::now() + 1s))
      throw std::runtime_error("Current version did not close");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
