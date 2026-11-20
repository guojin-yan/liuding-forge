// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "accelerator_support.hpp"
#include "resource_support.hpp"

#include <jyppx/liuding/liuding.hpp>
#include <jyppx/liuding/optim/step_lr_scheduler.hpp>
#include <jyppx/liuding/training/amp/autocast.hpp>
#include <jyppx/liuding/training/amp/grad_scaler_checkpoint_participant.hpp>
#include <jyppx/liuding/training/file_checkpoint_store.hpp>

#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
namespace lf = jyppx::liuding;
namespace bench = lf::benchmarks;
namespace opt = lf::optim;
namespace training = lf::training;
namespace amp = training::amp;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
constexpr std::uint64_t MiB = 1024ULL * 1024ULL;

struct Options {
  std::string precision;
  std::filesystem::path directory;
  std::uint64_t seconds = 180;
  std::uint64_t checkpoint_seconds = 15;
  std::uint64_t max_steps = 1000000;
  std::uint64_t pressure_mib = 2048;
};

std::uint64_t number(std::string_view value) {
  std::uint64_t parsed = 0;
  const auto [end, status] =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  bench::require(
      status == std::errc{} && end == value.data() + value.size() && parsed > 0,
      "expected positive integer option");
  return parsed;
}

Options options(int argc, char** argv) {
  Options result;
  for (int index = 1; index < argc; index += 2) {
    bench::require(index + 1 < argc, "each option requires a value");
    const std::string_view key(argv[index]), value(argv[index + 1]);
    if (key == "--precision")
      result.precision = value;
    else if (key == "--work-dir")
      result.directory = value;
    else if (key == "--seconds")
      result.seconds = number(value);
    else if (key == "--checkpoint-seconds")
      result.checkpoint_seconds = number(value);
    else if (key == "--max-steps")
      result.max_steps = number(value);
    else if (key == "--pressure-mib")
      result.pressure_mib = number(value);
    else
      throw std::invalid_argument("unknown stability option");
  }
  bench::require(result.precision == "fp32" || result.precision == "fp16" ||
                     result.precision == "bf16",
                 "--precision must be fp32, fp16 or bf16");
  bench::require(result.seconds >= 2 && result.seconds <= 7200 &&
                     result.checkpoint_seconds < result.seconds &&
                     result.max_steps <= 20000000 && result.pressure_mib <= 4096,
                 "stability budget outside supported bounds");
  bench::require(!result.directory.empty() &&
                     std::filesystem::create_directory(result.directory),
                 "--work-dir must be a new directory with an existing parent");
  return result;
}

std::pair<std::size_t, std::size_t> device_memory() {
  std::size_t free = 0, total = 0;
#ifdef JYPPX_BENCH_ROCM
  bench::require(hipMemGetInfo(&free, &total) == hipSuccess,
                 "HIP memory observation failed");
#else
  bench::require(cudaMemGetInfo(&free, &total) == cudaSuccess,
                 "CUDA memory observation failed");
#endif
  bench::require(free > 0 && total >= free,
                 "invalid device memory observation");
  return {free, total};
}

Json memory_snapshot() {
  torch::cuda::synchronize(0);
  const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
  const auto [free, total] = device_memory();
  std::uint64_t rss = 0, peak = 0;
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS memory{};
  bench::require(
      GetProcessMemoryInfo(GetCurrentProcess(), &memory, sizeof(memory)) != 0,
      "RSS observation failed");
  rss = memory.WorkingSetSize;
  peak = memory.PeakWorkingSetSize;
#else
  std::ifstream input("/proc/self/status");
  std::string key, value, unit;
  while (std::getline(input, value)) {
    std::istringstream row(value);
    std::uint64_t kib = 0;
    if (row >> key >> kib >> unit) {
      if (key == "VmRSS:") rss = kib * 1024;
      if (key == "VmHWM:") peak = kib * 1024;
    }
  }
#endif
  bench::require(rss > 0 && peak >= rss, "invalid RSS observation");
  return {{"allocated_bytes", stats.allocated_bytes[0].current},
          {"reserved_bytes", stats.reserved_bytes[0].current},
          {"peak_allocated_bytes", stats.allocated_bytes[0].peak},
          {"peak_reserved_bytes", stats.reserved_bytes[0].peak},
          {"device_free_bytes", free},
          {"device_total_bytes", total},
          {"rss_bytes", rss},
          {"peak_rss_bytes", peak}};
}

void allocator_recovery() {
  // A process-local allocator quota provokes OOM without exhausting physical
  // VRAM.
  c10::cuda::CUDACachingAllocator::emptyCache();
  const auto [free, total] = device_memory();
  bench::require(free >= 512 * MiB,
                 "insufficient free memory for bounded recovery check");
  struct RestoreQuota {
    ~RestoreQuota() {
      try {
        c10::cuda::CUDACachingAllocator::setMemoryFraction(1.0, 0);
      } catch (...) {
      }
    }
  } restore;
  c10::cuda::CUDACachingAllocator::setMemoryFraction(
      static_cast<double>(64 * MiB) / total, 0);
  bool observed = false;
  try {
    auto rejected = torch::empty(
        {static_cast<std::int64_t>(128 * MiB)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, 0));
  } catch (const c10::OutOfMemoryError&) {
    observed = true;
  }
  c10::cuda::CUDACachingAllocator::setMemoryFraction(1.0, 0);
  bench::require(observed,
                 "allocator quota did not reject oversized allocation");
  const auto recovered =
      torch::ones({8}, torch::TensorOptions().device(torch::kCUDA, 0)) + 2;
  bench::require(recovered.sum().item<float>() == 24,
                 "execution did not recover after allocator quota OOM");
  bool invalid_shape = false;
  try {
    static_cast<void>(
        recovered.matmul(torch::ones({3, 2}, recovered.options())));
  } catch (const c10::Error&) {
    invalid_shape = true;
  }
  bench::require(invalid_shape, "invalid matrix shape was not rejected");
  bench::require(recovered.square().sum().item<float>() == 72,
                 "execution did not recover after shape rejection");
}

class Model final : public torch::nn::Module {
 public:
  Model() : dense(register_module("dense", torch::nn::Linear(256, 256))) {}
  lf::Tensor forward(const lf::Tensor& input) { return dense->forward(input); }
  torch::nn::Linear dense{nullptr};
};

struct State {
  std::shared_ptr<Model> holder;
  std::shared_ptr<lf::nn::ModelAdapter> model;
  std::shared_ptr<opt::AdamWOptimizerAdapter> optimizer;
  std::shared_ptr<opt::StepLrScheduler> scheduler;
  std::shared_ptr<amp::GradScaler> scaler;
  training::CheckpointParticipantRegistry registry;
  c10::ScalarType dtype;

  explicit State(c10::ScalarType type)
      : holder(std::make_shared<Model>()), dtype(type) {
    holder->to(torch::Device(torch::kCUDA, 0));
    model = std::make_shared<lf::nn::ModelAdapter>(
        lf::nn::adapt_tensor_module(holder));
    std::vector<opt::OptimizerParameterBinding> bindings;
    for (const auto& entry : model->state().named_parameters())
      bindings.push_back(
          {.parameter_id = entry.key(), .parameter = entry.value()});
    optimizer = std::make_shared<opt::AdamWOptimizerAdapter>(
        "stability.adamw", std::vector<opt::AdamWParameterGroup>{
                               {.group_id = "main",
                                .parameters = std::move(bindings),
                                .options = {.learning_rate = 1e-4}}});
    scheduler = std::make_shared<opt::StepLrScheduler>(
        optimizer,
        opt::StepLrSchedulerOptions{.step_size = 10000, .gamma = 0.99});
    scaler = std::make_shared<amp::GradScaler>(
        amp::GradScalerOptions{.init_scale = 128,
                               .growth_interval = 1000000000,
                               .enabled = dtype == torch::kFloat16});
    registry.register_participant(
        std::make_shared<lf::nn::ModelCheckpointParticipant>(
            *model, lf::nn::ModelCheckpointParticipantOptions{
                        .max_tensors = 4,
                        .max_topology_bytes = MiB,
                        .max_tensor_bytes = MiB,
                        .max_total_bytes = 4 * MiB}));
    registry.register_participant(
        std::make_shared<opt::AdamWOptimizerCheckpointParticipant>(
            optimizer,
            opt::AdamWOptimizerCheckpointOptions{.max_groups = 1,
                                                 .max_parameters = 4,
                                                 .max_topology_bytes = MiB,
                                                 .max_control_bytes = MiB,
                                                 .max_tensor_bytes = MiB,
                                                 .max_total_bytes = 8 * MiB}));
    registry.register_participant(
        std::make_shared<opt::LrSchedulerCheckpointParticipant>(
            scheduler,
            opt::LrSchedulerCheckpointOptions{.max_groups = 1,
                                              .max_topology_bytes = MiB,
                                              .max_control_bytes = MiB,
                                              .max_total_bytes = 2 * MiB}));
    registry.register_participant(
        std::make_shared<amp::GradScalerCheckpointParticipant>(scaler));
  }

  double step(const lf::Tensor& input, bool overflow = false) {
    optimizer->zero_grad();
    lf::Tensor prediction;
    {
      std::unique_ptr<amp::AutocastGuard> autocast;
      if (dtype != torch::kFloat32)
        autocast =
            std::make_unique<amp::AutocastGuard>(c10::DeviceType::CUDA, dtype);
      prediction = holder->forward(input);
    }
    bench::require(prediction.scalar_type() == dtype,
                   "autocast did not produce the selected precision");
    const auto target =
        input * 0.1 +
        static_cast<double>(optimizer->completed_steps() % 17) * 0.001;
    auto loss = (prediction.to(torch::kFloat32) - target).square().mean();
    const double value = loss.item<double>();
    bench::require(std::isfinite(value), "nonfinite stability loss");
    scaler->scale(loss).backward();
    if (overflow)
      holder->parameters().front().mutable_grad().fill_(
          std::numeric_limits<float>::infinity());
    scaler->unscale(optimizer);
    const bool allowed = scaler->step_allowed(optimizer);
    bench::require(allowed != overflow, "unexpected scaler commit decision");
    if (allowed) {
      for (const auto& parameter : holder->parameters())
        bench::require(parameter.grad().defined() &&
                           torch::isfinite(parameter.grad()).all().item<bool>(),
                       "nonfinite or missing stability gradient");
      optimizer->step();
      scheduler->step();
    }
    scaler->mark_step(optimizer, allowed);
    scaler->update();
    return value;
  }
};

void equal(const State& left, const State& right) {
  const auto a = left.holder->named_parameters(),
             b = right.holder->named_parameters();
  bench::require(a.size() == b.size(), "restored parameter topology differs");
  for (const auto& item : a) {
    bench::require(torch::equal(item.value(), b[item.key()]),
                   "restored parameter differs");
    bench::require(torch::equal(item.value().grad(), b[item.key()].grad()),
                   "continued gradient differs");
  }
  bench::require(
      left.optimizer->completed_steps() == right.optimizer->completed_steps() &&
          left.scheduler->completed_steps() ==
              right.scheduler->completed_steps() &&
          left.scheduler->learning_rates() ==
              right.scheduler->learning_rates() &&
          left.scaler->get_scale() == right.scaler->get_scale() &&
          left.scaler->capture_state().growth_tracker ==
              right.scaler->capture_state().growth_tracker,
      "restored control state differs");
}

training::FileCheckpointStoreOptions checkpoint_options() {
  training::FileCheckpointStoreOptions options;
  options.max_artifact_bytes = 64 * MiB;
  options.max_field_bytes = 16 * MiB;
  options.min_free_storage_bytes = 256 * MiB;
  return options;
}

Json run(const Options& args, const Json& identity) {
  allocator_recovery();
  const auto dtype = args.precision == "fp32"   ? torch::kFloat32
                     : args.precision == "fp16" ? torch::kFloat16
                                                : torch::kBFloat16;
  auto state = std::make_unique<State>(dtype);
  const auto input = torch::linspace(-0.5, 0.5, 32 * 256)
                         .reshape({32, 256})
                         .to(torch::Device(torch::kCUDA, 0));
  if (dtype == torch::kFloat16) {
    state->step(input, true);
    bench::require(state->optimizer->completed_steps() == 0 &&
                       state->scaler->get_scale() == 64,
                   "injected overflow did not skip and back off");
  }
  {
    const auto original = state->holder->parameters().front().detach().clone();
    state->step(input);
    bench::require(!torch::equal(original, state->holder->parameters().front()),
                   "initial finite step did not update model parameters");
  }
  auto pressure = lf::Tensor();
  const std::uint64_t pressure_bytes = args.pressure_mib * MiB;
  auto allocate_pressure = [&] {
    const auto [free, total] = device_memory();
    bench::require(
        pressure_bytes <= free / 4 && free - pressure_bytes >= 2 * 1024 * MiB,
        "pressure request exceeds conservative free-memory budget");
    pressure = torch::empty(
        {static_cast<std::int64_t>(pressure_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, 0));
    pressure.fill_(90);
    bench::require(
        pressure[0].item<int>() == 90 && pressure[-1].item<int>() == 90,
        "pressure payload write/read failed");
    torch::cuda::synchronize(0);
  };
  allocate_pressure();
  const auto initial = memory_snapshot();
  Json snapshots = Json::array();
  std::ofstream progress(args.directory / "progress.ndjson");
  bench::require(progress.good(), "cannot open stability progress evidence");
  const auto checkpoint_path = args.directory / "latest.checkpoint";
  std::uint64_t checkpoints = 0, compared_steps = 0;
  const auto started = Clock::now();
  auto last_checkpoint = started;
  double final_loss = 0;
  while (std::chrono::duration<double>(Clock::now() - started).count() <
             args.seconds &&
         state->optimizer->completed_steps() + 3 < args.max_steps) {
    final_loss = state->step(input);
    if (std::chrono::duration<double>(Clock::now() - last_checkpoint).count() <
        args.checkpoint_seconds)
      continue;
    {
      training::FileCheckpointStore store(checkpoint_path,
                                          checkpoint_options());
      training::CheckpointCoordinator coordinator(state->registry);
      coordinator.save(store);
    }
    auto resumed = std::make_unique<State>(dtype);
    {
      training::FileCheckpointStore reopened(checkpoint_path,
                                             checkpoint_options());
      training::CheckpointCoordinator coordinator(resumed->registry);
      coordinator.restore(reopened);
    }
    for (int step = 0; step < 3; ++step) {
      state->step(input);
      resumed->step(input);
      equal(*state, *resumed);
      ++compared_steps;
    }
    state = std::move(resumed);
    ++checkpoints;
    const auto before_release = memory_snapshot();
    pressure = lf::Tensor();
    torch::cuda::synchronize(0);
    const auto released = memory_snapshot();
    bench::require(before_release["allocated_bytes"].get<std::int64_t>() -
                           released["allocated_bytes"].get<std::int64_t>() >=
                       static_cast<std::int64_t>(pressure_bytes),
                   "pressure allocation did not release its live payload");
    allocate_pressure();
    bench::require(bench::benchmark_identity() == identity,
                   "device/build/thread identity changed during stability run");
    auto snapshot = memory_snapshot();
    snapshot["elapsed_seconds"] =
        std::chrono::duration<double>(Clock::now() - started).count();
    snapshot["committed_steps"] = state->optimizer->completed_steps();
    snapshot["released_allocated_bytes"] = released["allocated_bytes"];
    snapshot["checkpoint_bytes"] = std::filesystem::file_size(checkpoint_path);
    snapshots.push_back(snapshot);
    progress << snapshot.dump() << '\n' << std::flush;
    bench::require(progress.good(),
                   "cannot persist stability progress evidence");
    last_checkpoint = Clock::now();
  }
  torch::cuda::synchronize(0);
  const double elapsed =
      std::chrono::duration<double>(Clock::now() - started).count();
  bench::require(
      elapsed >= args.seconds && checkpoints > 0,
      "stability duration or checkpoint requirement was not reached");
  const auto final = memory_snapshot();
  pressure = lf::Tensor();
  torch::cuda::synchronize(0);
  const auto released = memory_snapshot();
  bench::require(final["allocated_bytes"].get<std::int64_t>() -
                         released["allocated_bytes"].get<std::int64_t>() >=
                     static_cast<std::int64_t>(pressure_bytes),
                 "final pressure payload did not release");
  bench::require(
      final["allocated_bytes"].get<std::int64_t>() <=
          initial["allocated_bytes"].get<std::int64_t>() +
              16 * static_cast<std::int64_t>(MiB),
      "live GPU allocation grew beyond the fixed-workload allowance");
  for (const auto& parameter : state->holder->parameters())
    bench::require(torch::isfinite(parameter).all().item<bool>(),
                   "nonfinite final parameter");
  bench::require(bench::benchmark_identity() == identity,
                 "final identity mismatch");
  return {{"schema_version", 1},
          {"kind", "jyppx-" JYPPX_BENCH_BACKEND "-stability-sample"},
          {"identity", identity},
          {"precision", args.precision},
          {"requested_seconds", args.seconds},
          {"elapsed_seconds", elapsed},
          {"committed_steps", state->optimizer->completed_steps()},
          {"checkpoint_restores", checkpoints},
          {"exact_continuation_steps", compared_steps},
          {"pressure_bytes", pressure_bytes},
          {"allocator_quota_oom_recovered", true},
          {"allocator_quota_bytes", 64 * MiB},
          {"allocator_rejected_bytes", 128 * MiB},
          {"shape_rejection_recovered", true},
          {"injected_overflow_skips", dtype == torch::kFloat16 ? 1 : 0},
          {"final_loss", final_loss},
          {"initial_memory", initial},
          {"final_memory", final},
          {"released_memory", released},
          {"snapshots", snapshots}};
}
}  // namespace

int main(int argc, char** argv) {
  try {
    bench::require(
        lf::cuda_is_available(),
        "stability target requires its matching accelerator runtime");
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);
    torch::manual_seed(20261120);
    const auto identity = bench::benchmark_identity();
    if (argc == 2 && std::string_view(argv[1]) == "--identity") {
      std::cout << identity.dump() << '\n';
      return 0;
    }
    std::cout << run(options(argc, argv), identity).dump() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "accelerator stability failure: " << error.what() << '\n';
    return 2;
  }
}
