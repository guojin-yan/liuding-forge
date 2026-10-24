// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "../../examples/serving/linear_bundle.hpp"
#include "resource_torch_support.hpp"

#include <atomic>
#include <future>
#include <optional>
#include <vector>

int main(int argc, char** argv) {
  try {
    using namespace jyppx::liuding::benchmarks;
    namespace fixture = jyppx::liuding::examples::serving;
    namespace serving = jyppx::liuding::serving;
    const auto args = arguments(argc, argv);
    require(args.profile == "serving", "unknown serving profile");
    constexpr std::size_t direct_count = 128, rounds = 32, width = 8;
    auto sample = report(args,
                         {{"fixture", "examples/serving/linear_bundle.hpp"},
                          {"operation", "y[n,c]=x[n,c]*[2,-3][c]"},
                          {"direct_requests", direct_count},
                          {"batch_rounds", rounds},
                          {"batch_width_limit", width},
                          {"max_wait_ms", 5},
                          {"request_shape", {1, 2}},
                          {"request_value", 1.0},
                          {"warmup_predictions", 1},
                          {"transport", "in-process public API; no HTTP"}},
                         direct_count + rounds * width);
    const auto device = configure_device(args, sample);
    torch::NoGradGuard guard;
    fixture::BundleFixture bundle(device);
    auto& metrics = sample["metrics"];
    metrics["publish_seconds"] =
        metric(seconds([&] { bundle.Publish(); }), "seconds", "lower", false);
    metrics["output_bytes"] =
        metric(directory_bytes(bundle.directory), "bytes");
    std::atomic<std::uint64_t> calls{0}, rows{0}, largest{0};
    const auto hook = [&](const jyppx::liuding::Tensor& input) {
      const auto count = static_cast<std::uint64_t>(input.size(0));
      calls.fetch_add(1);
      rows.fetch_add(count);
      auto previous = largest.load();
      while (previous < count &&
             !largest.compare_exchange_weak(previous, count)) {
      }
    };
    std::shared_ptr<serving::Predictor> predictor;
    metrics["load_seconds"] =
        metric(device_seconds(device, [&] { predictor = bundle.Load(hook); }),
               "seconds", "lower", false,
               "bundle verification, state import, and public backend "
               "known-vector warmup");
    require(predictor->info().runtime.device == device &&
                predictor->info().health == serving::HealthStatus::ready,
            "predictor not ready on requested device");
    const auto request = fixture::Request(device);
    const auto expected =
        torch::tensor({{2.0F, -3.0F}}, torch::TensorOptions().device(device));
    const auto verify = [&](const serving::PredictResponse& response) {
      const auto output = response.output.value.toTensor();
      verify_tensor(output, device, 2);
      require(output.sizes().vec() == std::vector<std::int64_t>{1, 2} &&
                  torch::equal(output, expected) &&
                  response.runtime.runtime.device == device,
              "serving exact response/device mismatch");
    };
    std::optional<serving::PredictResponse> warmed;
    metrics["warmup_seconds"] = metric(
        device_seconds(device, [&] { warmed = predictor->predict(request); }),
        "seconds", "lower", false,
        "one additional public Predictor prediction");
    verify(*warmed);
    std::vector<serving::PredictResponse> direct;
    direct.reserve(direct_count);
    const auto direct_time = device_seconds(device, [&] {
      for (std::size_t index = 0; index < direct_count; ++index)
        direct.push_back(predictor->predict(request));
    });
    for (const auto& response : direct) verify(response);
    metrics["forward_seconds"] = metric(
        direct_time / direct_count, "seconds", "lower", true,
        "complete Predictor boundary including validation and owned snapshots");
    calls = 0;
    rows = 0;
    largest = 0;
    serving::DynamicBatcher batcher(predictor,
                                    {.max_queue_size = 64,
                                     .max_batch_size = width,
                                     .max_wait = std::chrono::milliseconds(5)});
    std::vector<serving::PredictResponse> batched;
    batched.reserve(rounds * width);
    const auto batch_time = device_seconds(device, [&] {
      for (std::size_t round = 0; round < rounds; ++round) {
        std::vector<std::future<serving::PredictResponse>> futures;
        for (std::size_t index = 0; index < width; ++index)
          futures.push_back(batcher.submit(fixture::Request(
              device, "batch-" + std::to_string(round * width + index))));
        for (auto& future : futures) batched.push_back(future.get());
      }
    });
    for (const auto& response : batched) verify(response);
    require(batcher.close(), "batcher shutdown failed");
    const auto stats = batcher.statistics();
    require(stats.accepted == rounds * width &&
                stats.completed == rounds * width && stats.failed == 0 &&
                stats.oom_retries == 0 && stats.queued == 0 &&
                stats.active == 0 && stats.resident_tensor_bytes == 0 &&
                stats.stopped,
            "batch scheduler did not complete frozen requests");
    require(rows.load() == rounds * width && calls.load() > 0 &&
                calls.load() < rounds * width && largest.load() > 1 &&
                largest.load() <= width,
            "workload did not execute real compatible batching");
    require(predictor->info().failed == 0 && predictor->info().active == 0 &&
                predictor->close(),
            "predictor failure or drain mismatch");
    metrics["batch_seconds"] =
        metric(batch_time / rounds, "seconds", "lower", true,
               "mean round of eight public submit/future requests; actual "
               "coalescing counts retained");
    metrics["throughput_requests_per_second"] =
        metric(rounds * width / batch_time, "requests/second", "higher");
    sample["execution"]["path"] =
        "NativeStateBackend plus Predictor plus DynamicBatcher; exact "
        "arithmetic fixture";
    sample["validation"]["batch_forward_calls"] = calls.load();
    sample["validation"]["batch_rows"] = rows.load();
    sample["validation"]["largest_actual_batch"] = largest.load();
    sample["validation"]["output_sum"] = -1.0;
    sample["validation"]["output_kind"] = "finite exact FP32 known vector";
    cuda_metrics(device, sample);
    finish(sample, direct.size() + batched.size());
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "serving resource benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
