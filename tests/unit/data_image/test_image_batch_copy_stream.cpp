// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data_image/batch_transform.hpp>
#include <jyppx/liuding/data_image/image_batch_copy_stream.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <gtest/gtest.h>

#if defined(JYPPX_LIUDING_IMAGE_TEST_CUDA)
#include <c10/core/Event.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <chrono>
#include <condition_variable>
#include <cuda_runtime_api.h>
#include <mutex>
#endif

#include <array>

namespace {
using namespace jyppx::liuding;
using namespace jyppx::liuding::data;
using namespace jyppx::liuding::data::image;

BatchMemoryOptions Memory() {
  BatchMemoryOptions memory;
  memory.pin_memory = true;
  memory.non_blocking = true;
  memory.target_device = Device(torch::kCUDA, 0);
  return memory;
}

TEST(ImageBatchCopyStreamTest, RejectsInvalidPolicyBeforeBackendAccess) {
  auto memory = Memory();
  EXPECT_THROW(ImageBatchCopyStream(memory, 0), InvalidArgument);
  EXPECT_THROW(ImageBatchCopyStream(memory, 4097), InvalidArgument);
  memory.pin_memory = false;
  EXPECT_THROW(ImageBatchCopyStream stream(memory), InvalidArgument);
  memory = Memory();
  memory.non_blocking = false;
  EXPECT_THROW(ImageBatchCopyStream stream(memory), InvalidArgument);
  memory = Memory();
  memory.target_device = Device(torch::kCPU);
  EXPECT_THROW(ImageBatchCopyStream stream(memory), InvalidArgument);
  memory.target_device = Device(torch::kCUDA);
  EXPECT_THROW(ImageBatchCopyStream stream(memory), InvalidArgument);
  memory = Memory();
  memory.max_bytes = 0;
  EXPECT_THROW(ImageBatchCopyStream stream(memory), InvalidArgument);
  ImageBatchLimits limits;
  limits.max_samples = 0;
  EXPECT_THROW(ImageBatchCopyStream(Memory(), 2, false, limits),
               InvalidArgument);
}

#if !defined(JYPPX_LIUDING_IMAGE_TEST_CUDA)
TEST(ImageBatchCopyStreamTest, CpuBuildExplicitlyRejectsCuda) {
  EXPECT_THROW(ImageBatchCopyStream stream(Memory()), DeviceError);
}
#else

ImageBatch Batch(ImageBatchLayout layout = ImageBatchLayout::stack,
                 bool mask = false, std::size_t count = 2) {
  std::vector<VisionSample> samples;
  std::vector<ImageBatchKey> keys;
  auto categories = std::make_shared<const std::vector<VisionCategory>>(
      std::vector<VisionCategory>{{0, "zero"}, {1, "one"}, {2, "two"}});
  for (std::size_t i = 0; i < count; ++i) {
    VisionSample sample;
    sample.width = layout == ImageBatchLayout::stack ? 16 : 16 + i;
    sample.height = 16;
    sample.image_path = "image-" + std::to_string(i);
    sample.image =
        torch::arange(3 * sample.width * sample.height, torch::kInt64)
            .reshape({3, sample.height, sample.width})
            .remainder(251)
            .to(torch::kUInt8);
    sample.geometry.emplace(ImageExtent{sample.width, sample.height});
    sample.labels = {i % 3};
    sample.label_categories = categories;
    if (mask) {
      sample.mask_target.emplace();
      sample.mask_target->ids = torch::full(
          {sample.height, sample.width}, 9007199254740993LL + i, torch::kInt64);
      sample.mask_target->width = sample.width;
      sample.mask_target->height = sample.height;
      sample.mask_target->ignore_id = -7;
      sample.mask_target->void_id = -9;
    } else {
      sample.paired_target = sample.image.flip({2});
      sample.paired_target_pixels = sample.pixels;
    }
    samples.push_back(std::move(sample));
    keys.push_back({"sample-" + std::to_string(i), i, i + 1});
  }
  ImageCollateOptions options;
  options.layout = layout;
  options.mask_padding_id = -7;
  return collate_images(samples, keys, options);
}

c10::cuda::CUDAStream CopyStream(const ImageBatchTransfer& ticket) {
  const auto timeline = ticket.timeline();
  return c10::cuda::CUDAStream(c10::Stream::unpack3(
      timeline.copy_stream_id, timeline.device.index(), c10::DeviceType::CUDA));
}

// A deterministic device-side gate. The callback performs no CUDA calls and
// has its own bound, so a failed assertion cannot strand ticket destruction.
class Gate {
 public:
  explicit Gate(c10::cuda::CUDAStream stream) : stream_(stream) {
    if (cudaLaunchHostFunc(stream.stream(), Callback, this) != cudaSuccess)
      throw std::runtime_error("Cannot enqueue CUDA test gate");
    std::unique_lock lock(mutex_);
    if (!condition_.wait_for(lock, std::chrono::seconds(3),
                             [&] { return entered_; })) {
      released_ = true;
      condition_.notify_all();
      lock.unlock();
      stream_.synchronize();
      throw std::runtime_error("CUDA test gate did not enter");
    }
  }
  ~Gate() {
    Release();
    stream_.synchronize();
    EXPECT_FALSE(timed_out_);
  }
  void Release() {
    std::lock_guard lock(mutex_);
    released_ = true;
    condition_.notify_all();
  }

 private:
  static void CUDART_CB Callback(void* opaque) {
    auto& gate = *static_cast<Gate*>(opaque);
    std::unique_lock lock(gate.mutex_);
    gate.entered_ = true;
    gate.condition_.notify_all();
    if (!gate.condition_.wait_for(lock, std::chrono::seconds(10),
                                  [&] { return gate.released_; }))
      gate.timed_out_ = true;
  }
  c10::cuda::CUDAStream stream_;
  std::mutex mutex_;
  std::condition_variable condition_;
  bool entered_ = false;
  bool released_ = false;
  bool timed_out_ = false;
};

TEST(ImageBatchCopyStreamTest, CopiesAllLayoutsAndTargetsOnDistinctStream) {
  ASSERT_TRUE(torch::cuda::is_available());
  const auto consumer = c10::cuda::getStreamFromPool(true, 0);
  const c10::cuda::CUDAStreamGuard guard(consumer);
  for (auto layout : {ImageBatchLayout::stack, ImageBatchLayout::pad,
                      ImageBatchLayout::ragged}) {
    for (bool mask : {false, true}) {
      const auto input = Batch(layout, mask);
      const auto prepared = prepare_image_batch(input, Memory());
      ImageBatchCopyStream stream(Memory(), 2, true);
      auto ticket = stream.submit(prepared);
      EXPECT_EQ(c10::cuda::getCurrentCUDAStream(0), consumer);
      auto output = ticket.receive();
      EXPECT_EQ(c10::cuda::getCurrentCUDAStream(0), consumer);
      EXPECT_EQ(output.layout, input.layout);
      EXPECT_EQ(output.memory.transfer_stream_policy,
                BatchTransferStreamPolicy::explicit_copy);
      const auto same = [](const Tensor& a, const Tensor& b) {
        if (!a.defined()) return !b.defined();
        return b.device().is_cuda() && torch::equal(a, b.cpu());
      };
      EXPECT_TRUE(same(input.images, output.images));
      EXPECT_TRUE(same(input.valid, output.valid));
      EXPECT_TRUE(same(input.labels, output.labels));
      if (mask && input.masks) EXPECT_TRUE(same(*input.masks, *output.masks));
      if (!mask && input.paired_images)
        EXPECT_TRUE(same(*input.paired_images, *output.paired_images));
      for (std::size_t i = 0; i < input.image_list.size(); ++i) {
        EXPECT_TRUE(same(input.image_list[i], output.image_list[i]));
        EXPECT_TRUE(same(input.valid_list[i], output.valid_list[i]));
        if (mask)
          EXPECT_TRUE(same(input.mask_list[i], output.mask_list[i]));
        else
          EXPECT_TRUE(
              same(input.paired_image_list[i], output.paired_image_list[i]));
      }
      EXPECT_EQ(output.label_offsets, input.label_offsets);
      EXPECT_EQ(output.items.back().key.id, input.items.back().key.id);
      EXPECT_EQ(output.items.back().key.view, input.items.back().key.view);
      EXPECT_EQ(output.items.back().label_categories->size(), 3);
      EXPECT_EQ(output.items.back().geometry->state().output.width,
                input.items.back().geometry->state().output.width);
      EXPECT_THROW(ticket.receive(), InvalidArgument);
      ticket.wait();
      EXPECT_TRUE(ticket.ready());
      const auto timeline = ticket.timeline();
      EXPECT_NE(timeline.copy_stream_id, consumer.id());
      EXPECT_EQ(timeline.consumer_stream_id, consumer.id());
      ASSERT_TRUE(timeline.copy_elapsed_ms);
      EXPECT_GE(*timeline.copy_elapsed_ms, 0.0);
      EXPECT_EQ(stream.pending(), 0);
    }
  }
}

TEST(ImageBatchCopyStreamTest, RetainsSourceAndLeasesUntilCopyCompletion) {
  ASSERT_TRUE(torch::cuda::is_available());
  auto budget = std::make_shared<ImageResourceBudget>();
  ImageBatchCopyStream stream(Memory(), 1, true, {}, budget);
  auto input = prepare_image_batch(Batch(), Memory(), {}, budget);
  c10::cuda::CUDAStream copy = [&] {
    auto probe = stream.submit(input);
    const auto selected = CopyStream(probe);
    probe.wait();
    return selected;
  }();
  auto owner = std::make_shared<ImageResourceLeaseSet>();
  owner->push_back(
      budget->reserve(ImageResourceKind::pipeline_inflight_bytes, 17));
  std::weak_ptr<ImageResourceLeaseSet> weak = owner;
  input.resource_lease_sets.push_back(std::move(owner));
  Gate gate(copy);
  auto ticket = stream.submit(std::move(input));
  ASSERT_FALSE(ticket.ready());
  EXPECT_FALSE(weak.expired());
  EXPECT_GT(budget->usage(ImageResourceKind::pinned_bytes).current, 0);
  EXPECT_GT(budget->usage(ImageResourceKind::gpu_bytes).current, 0);
  {
    auto output = ticket.receive();
    // Returning before a deliberately blocked event proves no CPU wait.
    EXPECT_FALSE(ticket.ready());
    EXPECT_EQ(output.items.size(), 2);
  }
  EXPECT_GT(budget->usage(ImageResourceKind::gpu_bytes).current, 0);
  gate.Release();
  ticket.wait();
  ticket.wait();
  EXPECT_TRUE(weak.expired());
  EXPECT_EQ(stream.pending(), 0);
  for (std::uint8_t i = 0; i <= 13; ++i)
    EXPECT_EQ(budget->usage(static_cast<ImageResourceKind>(i)).current, 0);
}

TEST(ImageBatchCopyStreamTest, CapacityCancelAndPolicyFailuresPreserveState) {
  ASSERT_TRUE(torch::cuda::is_available());
  ImageBatchCopyStream stream(Memory(), 1);
  const auto prepared = prepare_image_batch(Batch(), Memory());
  auto first = stream.submit(prepared);
  EXPECT_EQ(stream.pending(), 1);
  EXPECT_THROW(stream.submit(prepared), ResourceExhausted);
  std::stop_source cancel;
  cancel.request_stop();
  EXPECT_THROW(stream.submit(prepared, cancel.get_token()), Cancelled);
  first.wait();
  EXPECT_EQ(stream.pending(), 0);
  auto damaged = prepared;
  damaged.memory.non_blocking_requested = false;
  const auto before = c10::cuda::getCurrentCUDAStream(0);
  EXPECT_THROW(stream.submit(damaged), CompatibilityError);
  EXPECT_EQ(c10::cuda::getCurrentCUDAStream(0), before);
  damaged = prepared;
  damaged.images = torch::empty(prepared.images.sizes(), torch::kUInt8);
  damaged.images.copy_(prepared.images);
  ASSERT_FALSE(damaged.images.is_pinned());
  EXPECT_THROW(stream.submit(damaged), CompatibilityError);
  EXPECT_EQ(stream.pending(), 0);
  auto next = stream.submit(prepared);
  {
    const c10::cuda::CUDAStreamGuard guard(CopyStream(next));
    EXPECT_THROW(next.receive(), CompatibilityError);
  }
  EXPECT_TRUE(torch::equal(next.receive().images.cpu(), prepared.images));
  next.wait();
  EXPECT_FALSE(next.timeline().copy_elapsed_ms);
}

TEST(ImageBatchCopyStreamTest, DestinationAdmissionPrecedesDeviceAllocation) {
  ASSERT_TRUE(torch::cuda::is_available());
  ImageResourceBudgetLimits limits;
  limits.max_gpu_bytes = 0;
  auto budget = std::make_shared<ImageResourceBudget>(limits);
  ImageBatchCopyStream stream(Memory(), 1, false, {}, budget);
  const auto input = prepare_image_batch(Batch(), Memory());
  const auto before = c10::cuda::getCurrentCUDAStream(0);
  EXPECT_THROW(stream.submit(input), ResourceExhausted);
  EXPECT_EQ(stream.pending(), 0);
  EXPECT_EQ(c10::cuda::getCurrentCUDAStream(0), before);
  EXPECT_EQ(budget->usage(ImageResourceKind::gpu_bytes).peak, 0);
  for (std::uint8_t i = 0; i <= 13; ++i)
    EXPECT_EQ(budget->usage(static_cast<ImageResourceKind>(i)).current, 0);
}

TEST(ImageBatchCopyStreamTest, CloseMoveAndDestructionKeepUnreadBatch) {
  ASSERT_TRUE(torch::cuda::is_available());
  const auto input = prepare_image_batch(Batch(), Memory());
  auto stream = std::make_unique<ImageBatchCopyStream>(Memory());
  auto original = stream->submit(input);
  auto ticket = std::move(original);
  EXPECT_THROW(static_cast<void>(original.ready()), InvalidArgument);
  ImageBatchCopyStream moved = std::move(*stream);
  EXPECT_EQ(stream->pending(), 0);
  EXPECT_THROW(stream->submit(input), InvalidArgument);
  moved.close();
  moved.close();
  EXPECT_EQ(moved.pending(), 0);
  EXPECT_TRUE(ticket.ready());
  EXPECT_THROW(moved.submit(input), InvalidArgument);
  auto output = ticket.receive();
  EXPECT_TRUE(torch::equal(output.images.cpu(), input.images));
  stream = std::make_unique<ImageBatchCopyStream>(Memory());
  // Replacing a live ticket drains its old copy before changing ownership.
  ticket = stream->submit(input);
  stream.reset();
  EXPECT_TRUE(ticket.ready());
  EXPECT_TRUE(torch::equal(ticket.receive().images.cpu(), input.images));
}

TEST(ImageBatchCopyStreamTest, RecordsDestinationStorageOnConsumerStream) {
  ASSERT_TRUE(torch::cuda::is_available());
  const auto consumer = c10::cuda::getStreamFromPool(true, 0);
  const c10::cuda::CUDAStreamGuard guard(consumer);
  ImageBatchCopyStream stream(Memory());
  const auto prepared = prepare_image_batch(Batch(), Memory());
  // Finish first-use allocation and fill-kernel initialization before gating:
  // driver initialization can synchronize streams and invalidate this race.
  auto snapshot = torch::empty(prepared.images.sizes(),
                               prepared.images.options().device(torch::kCUDA));
  auto ticket = stream.submit(prepared);
  const auto copy = CopyStream(ticket);
  ticket.wait();
  {
    const c10::cuda::CUDAStreamGuard copy_guard(copy);
    const auto warm =
        torch::full(prepared.images.sizes(), 255,
                    prepared.images.options().device(torch::kCUDA));
    copy.synchronize();
  }
  Gate gate(consumer);
  {
    auto output = ticket.receive();
    snapshot.copy_(output.images);
  }
  // Release the ticket's second destination owner before forcing allocator
  // reuse on the allocation stream while the consumer clone is still blocked.
  stream.close();
  std::vector<Tensor> replacements;
  {
    const c10::cuda::CUDAStreamGuard copy_guard(copy);
    for (int i = 0; i < 32; ++i)
      replacements.push_back(
          torch::full(prepared.images.sizes(), 255,
                      prepared.images.options().device(torch::kCUDA)));
  }
  gate.Release();
  EXPECT_TRUE(torch::equal(snapshot.cpu(), prepared.images));
}

TEST(ImageBatchCopyStreamTest, TwoOutstandingTicketsKeepIndependentResults) {
  ASSERT_TRUE(torch::cuda::is_available());
  ImageBatchCopyStream stream(Memory(), 2);
  const auto first = prepare_image_batch(Batch(), Memory());
  auto second = prepare_image_batch(Batch(), Memory());
  second.images.fill_(42);
  auto a = stream.submit(first);
  auto b = stream.submit(second);
  EXPECT_EQ(stream.pending(), 2);
  EXPECT_THROW(stream.submit(first), ResourceExhausted);
  EXPECT_TRUE(torch::equal(b.receive().images.cpu(), second.images));
  b.wait();
  EXPECT_EQ(stream.pending(), 1);
  EXPECT_TRUE(torch::equal(a.receive().images.cpu(), first.images));
  a.wait();
  EXPECT_EQ(stream.pending(), 0);
}

TEST(ImageBatchCopyStreamTest,
     RejectsDeviceSourceEvenWithCopiedPreparationMetadata) {
  ASSERT_TRUE(torch::cuda::is_available());
  const auto prepared = prepare_image_batch(Batch(), Memory());
  auto device_batch = transfer_image_batch(prepared, Memory());
  device_batch.memory = prepared.memory;
  ImageResourceBudgetLimits limits;
  limits.max_gpu_bytes = 0;
  auto budget = std::make_shared<ImageResourceBudget>(limits);
  ImageBatchCopyStream stream(Memory(), 1, false, {}, budget);
  EXPECT_THROW(stream.submit(std::move(device_batch)), CompatibilityError);
  EXPECT_EQ(stream.pending(), 0);
  EXPECT_EQ(budget->usage(ImageResourceKind::gpu_bytes).peak, 0);
}

TEST(ImageBatchCopyStreamTest, RejectsGraphCaptureBeforeQueuingWork) {
  ASSERT_TRUE(torch::cuda::is_available());
  const auto consumer = c10::cuda::getStreamFromPool(true, 0);
  const c10::cuda::CUDAStreamGuard guard(consumer);
  ImageBatchCopyStream stream(Memory());
  const auto prepared = prepare_image_batch(Batch(), Memory());
  auto ticket = stream.submit(prepared);
  ticket.wait();
  ASSERT_EQ(cudaStreamBeginCapture(consumer.stream(),
                                   cudaStreamCaptureModeThreadLocal),
            cudaSuccess);
  EXPECT_THROW(stream.submit(prepared), CompatibilityError);
  EXPECT_THROW(ticket.receive(), CompatibilityError);
  cudaGraph_t graph = nullptr;
  EXPECT_EQ(cudaStreamEndCapture(consumer.stream(), &graph), cudaSuccess);
  if (graph) EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
  EXPECT_EQ(stream.pending(), 0);
  EXPECT_TRUE(torch::equal(ticket.receive().images.cpu(), prepared.images));
}

TEST(ImageBatchCopyStreamTest, MatchesCallerCurrentForwardBackwardAndUpdates) {
  ASSERT_TRUE(torch::cuda::is_available());
  const auto consumer = c10::cuda::getStreamFromPool(true, 0);
  const c10::cuda::CUDAStreamGuard guard(consumer);
  const auto make = [] {
    auto network =
        torch::nn::Sequential(torch::nn::Flatten(), torch::nn::Linear(768, 8),
                              torch::nn::ReLU(), torch::nn::Linear(8, 3));
    network->to(Device(torch::kCUDA, 0));
    torch::NoGradGuard no_grad;
    for (auto& parameter : network->parameters())
      parameter.copy_(torch::arange(parameter.numel(), parameter.options())
                          .reshape(parameter.sizes())
                          .remainder(19)
                          .sub(9)
                          .mul(0.001));
    return network;
  };
  auto reference = make();
  auto copied = make();
  torch::optim::SGD reference_optimizer(reference->parameters(), 0.01);
  torch::optim::SGD copied_optimizer(copied->parameters(), 0.01);
  const auto prepared =
      prepare_image_batch(Batch(ImageBatchLayout::stack, true, 4), Memory());
  ImageBatchCopyStream stream(Memory(), 2, true);
  for (int step = 0; step < 4; ++step) {
    const auto standard =
        transform_image_batch(transfer_image_batch(prepared, Memory()));
    reference_optimizer.zero_grad();
    const auto expected = reference->forward(standard.images);
    auto expected_loss =
        torch::nn::functional::cross_entropy(expected, standard.labels);
    expected_loss.backward();
    auto ticket = stream.submit(prepared);
    auto actual_batch = transform_image_batch(ticket.receive());
    copied_optimizer.zero_grad();
    const auto actual = copied->forward(actual_batch.images);
    auto actual_loss =
        torch::nn::functional::cross_entropy(actual, actual_batch.labels);
    actual_loss.backward();
    EXPECT_TRUE(torch::equal(actual, expected));
    EXPECT_TRUE(torch::equal(actual_loss, expected_loss));
    const auto a = copied->parameters();
    const auto b = reference->parameters();
    for (std::size_t i = 0; i < a.size(); ++i)
      EXPECT_TRUE(torch::equal(a[i].grad(), b[i].grad()));
    reference_optimizer.step();
    copied_optimizer.step();
    for (std::size_t i = 0; i < a.size(); ++i)
      EXPECT_TRUE(torch::equal(a[i], b[i]));
    ticket.wait();
    EXPECT_EQ(c10::cuda::getCurrentCUDAStream(0), consumer);
  }
  EXPECT_EQ(stream.pending(), 0);
}
#endif
}  // namespace
