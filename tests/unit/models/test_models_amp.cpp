// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/models/dino.hpp>
#include <jyppx/liuding/optim/sgd_adapter.hpp>
#include <jyppx/liuding/training/amp/autocast.hpp>
#include <jyppx/liuding/training/amp/grad_scaler.hpp>

#include <torch/torch.h>

#ifdef JYPPX_LIUDING_MODELS_CUDA_METRICS
#include <c10/cuda/CUDACachingAllocator.h>
#endif

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
// psapi.h requires Windows declarations.
// clang-format off
#include <windows.h>
#include <psapi.h>
// clang-format on
#elif defined(__linux__)
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace {

using namespace jyppx::liuding;

std::uint64_t ProcessPeakRssBytes() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS counters{};
  if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
    return static_cast<std::uint64_t>(counters.PeakWorkingSetSize);
  }
#elif defined(__linux__)
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) == 0) {
    return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024U;
  }
#endif
  return 0U;
}

std::uint64_t ProcessCurrentRssBytes() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS counters{};
  if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
    return static_cast<std::uint64_t>(counters.WorkingSetSize);
  }
#elif defined(__linux__)
  std::ifstream statm("/proc/self/statm");
  std::uint64_t total_pages = 0U;
  std::uint64_t resident_pages = 0U;
  const auto page_size = sysconf(_SC_PAGESIZE);
  if ((statm >> total_pages >> resident_pages) && page_size > 0) {
    return resident_pages * static_cast<std::uint64_t>(page_size);
  }
#endif
  return 0U;
}

TEST(ModelsAmpCuda, DinoV2GiantHalfStorageDynamicScalerResourceGate) {
  const auto* enabled =
      std::getenv("JYPPX_LIUDING_DINOV2_GIANT_CUDA_SCALER_GATE");
  if (enabled == nullptr || std::string_view(enabled) != "1") {
    GTEST_SKIP()
        << "Set JYPPX_LIUDING_DINOV2_GIANT_CUDA_SCALER_GATE=1 for the serial "
           "Giant FP16-storage dynamic scaling and overflow recovery gate";
  }
  if (!torch::cuda::is_available() || torch::cuda::device_count() == 0) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }
#ifndef JYPPX_LIUDING_MODELS_CUDA_METRICS
  GTEST_SKIP() << "CUDA allocator metrics are required for this resource gate";
#else
  torch::manual_seed(20261027);
  torch::cuda::synchronize(0);
  c10::cuda::CUDACachingAllocator::emptyCache();
  c10::cuda::CUDACachingAllocator::resetPeakStats(0);
  const auto start = std::chrono::steady_clock::now();
  {
    const torch::Device device(torch::kCUDA, 0);
    auto model =
        models::dinov2_vit_giant14(models::DinoV2Options{.num_classes = 3});
    ASSERT_TRUE(model);
    model->to(device, torch::kHalf);
    model->train();

    std::vector<optim::OptimizerParameterBinding> bindings;
    for (const auto& entry : model->named_parameters(true)) {
      ASSERT_TRUE(entry.value().device().is_cuda());
      ASSERT_EQ(entry.value().scalar_type(), torch::kHalf);
      bindings.push_back(
          {.parameter_id = entry.key(), .parameter = entry.value()});
    }
    auto optimizer = std::make_shared<optim::SgdOptimizerAdapter>(
        "optimizer.giant.amp", std::vector<optim::SgdParameterGroup>{
                                   {.group_id = "group.main",
                                    .parameters = std::move(bindings),
                                    .options = {.learning_rate = 0.01}}});
    training::amp::GradScaler scaler({.init_scale = 128.0,
                                      .growth_interval = 2,
                                      .scaler_id = "grad_scaler.giant"});
    const auto named = model->named_parameters(true);
    ASSERT_TRUE(named.contains("head.weight"));
    ASSERT_TRUE(named.contains("patch_embed.proj.weight"));
    const auto head = named["head.weight"];
    const auto patch = named["patch_embed.proj.weight"];
    const auto input =
        torch::randn({1, 3, 14, 14},
                     torch::TensorOptions().dtype(torch::kHalf).device(device));
    constexpr std::array<double, 4> expected_scales{128.0, 64.0, 64.0, 128.0};
    constexpr std::array<std::uint64_t, 4> expected_steps{1U, 1U, 2U, 3U};
    constexpr std::array<std::uint64_t, 4> expected_trackers{1U, 0U, 1U, 0U};

    for (std::size_t cycle = 0; cycle < expected_scales.size(); ++cycle) {
      SCOPED_TRACE(cycle);
      const bool inject_inf = cycle == 1;
      ASSERT_TRUE(scaler.checkpoint_ready());
      optimizer->zero_grad();
      EXPECT_FALSE(head.grad().defined());
      EXPECT_FALSE(patch.grad().defined());
      const auto head_before = head.detach().clone();
      const auto patch_before = patch.detach().clone();
      const double scale_before = scaler.get_scale();
      Tensor loss;
      {
        training::amp::AutocastGuard autocast(c10::DeviceType::CUDA, at::kHalf);
        const auto logits = model->forward(input);
        ASSERT_EQ(logits.sizes(), torch::IntArrayRef({1, 3}));
        ASSERT_EQ(logits.scalar_type(), torch::kHalf);
        loss = logits.to(torch::kFloat32).square().mean();
      }
      ASSERT_TRUE(torch::isfinite(loss).item<bool>());
      scaler.scale(loss).backward();
      loss = Tensor{};
      ASSERT_TRUE(head.grad().defined());
      ASSERT_TRUE(patch.grad().defined());
      ASSERT_EQ(head.grad().scalar_type(), torch::kHalf);
      ASSERT_EQ(patch.grad().scalar_type(), torch::kHalf);
      ASSERT_TRUE(torch::isfinite(head.grad()).all().item<bool>());
      ASSERT_TRUE(torch::isfinite(patch.grad()).all().item<bool>());
      const auto expected_head_gradient =
          head.grad().clone().div_(scale_before);
      if (inject_inf) {
        const torch::NoGradGuard no_grad;
        head.grad().view({-1})[0].fill_(std::numeric_limits<float>::infinity());
      }

      scaler.unscale(optimizer);
      EXPECT_FALSE(scaler.checkpoint_ready());
      const bool applied = scaler.cycle_allowed();
      ASSERT_EQ(applied, !inject_inf);
      EXPECT_EQ(scaler.step_allowed(optimizer), applied);
      EXPECT_EQ(scaler.found_inf(), inject_inf);
      if (applied) {
        EXPECT_TRUE(torch::equal(head.grad(), expected_head_gradient));
        EXPECT_TRUE(torch::isfinite(patch.grad()).all().item<bool>());
        EXPECT_GT(head.grad().abs().max().item<double>(), 0.0);
        EXPECT_GT(patch.grad().abs().max().item<double>(), 0.0);
        optimizer->step();
        EXPECT_FALSE(torch::equal(head, head_before));
        EXPECT_TRUE(torch::isfinite(head).all().item<bool>());
        EXPECT_TRUE(torch::isfinite(patch).all().item<bool>());
      } else {
        EXPECT_TRUE(torch::equal(head, head_before));
        EXPECT_TRUE(torch::equal(patch, patch_before));
      }
      scaler.mark_step(optimizer, applied);
      scaler.update();
      EXPECT_EQ(optimizer->completed_steps(), expected_steps[cycle]);
      EXPECT_DOUBLE_EQ(scaler.get_scale(), expected_scales[cycle]);
      EXPECT_EQ(scaler.found_inf(), inject_inf);
      ASSERT_TRUE(scaler.checkpoint_ready());
      EXPECT_EQ(scaler.capture_state().growth_tracker,
                expected_trackers[cycle]);
      std::cout << "SCALER_CYCLE model=vit_giant_patch14_dinov2 cycle=" << cycle
                << " injected_inf=" << inject_inf << " applied=" << applied
                << " completed_steps=" << optimizer->completed_steps()
                << " scale_before=" << scale_before
                << " scale_after=" << scaler.get_scale() << '\n';
    }
    optimizer->zero_grad();
    EXPECT_FALSE(head.grad().defined());
    EXPECT_FALSE(patch.grad().defined());
  }

  torch::cuda::synchronize(0);
  const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - start)
                              .count();
  const auto peak_rss = ProcessPeakRssBytes();
  const auto current_rss = ProcessCurrentRssBytes();
  ASSERT_GT(peak_rss, 0U);
  ASSERT_GT(current_rss, 0U);
  EXPECT_LE(peak_rss, 8ULL << 30U);
  EXPECT_LE(current_rss, 3ULL << 30U);
  const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
  const auto peak_cuda = stats.allocated_bytes[0].peak;
  const auto live_cuda = stats.allocated_bytes[0].current;
  ASSERT_GT(peak_cuda, 0);
  EXPECT_LE(peak_cuda, 5LL << 30U);
  EXPECT_LE(live_cuda, 32LL << 20U);
  c10::cuda::CUDACachingAllocator::emptyCache();
  const auto empty_stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
  const auto reserved_cuda = empty_stats.reserved_bytes[0].current;
  EXPECT_LE(reserved_cuda, 64LL << 20U);
  std::cout << "RESOURCE_METRICS model=vit_giant_patch14_dinov2"
            << " operation=cuda-half-storage-dynamic-scaler elapsed_ms="
            << elapsed_ms << " process_peak_rss_bytes=" << peak_rss
            << " process_current_rss_after_destroy_bytes=" << current_rss
            << " peak_cuda_allocated_bytes=" << peak_cuda
            << " cuda_allocated_after_destroy_bytes=" << live_cuda
            << " cuda_reserved_after_destroy_bytes="
            << stats.reserved_bytes[0].current
            << " cuda_reserved_after_empty_cache_bytes=" << reserved_cuda
            << '\n';
#endif
}

}  // namespace

int main(int argc, char** argv) {
  torch::set_num_threads(2);
  torch::set_num_interop_threads(1);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
