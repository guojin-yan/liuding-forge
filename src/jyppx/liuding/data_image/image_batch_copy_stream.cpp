// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "../data/batch_memory_detail.hpp"
#include "image_batch_detail.hpp"

#include <jyppx/liuding/data_image/image_batch_copy_stream.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <algorithm>
#include <c10/core/Event.h>
#include <c10/core/StreamGuard.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <exception>
#include <mutex>

namespace jyppx::liuding::data::image {
namespace copy_stream_detail {

struct Transfer {
  // Reservations must survive temporary/destination Tensor destruction.
  std::shared_ptr<ImageResourceLeaseSet> leases;
  std::optional<ImageBatch> source;
  std::optional<ImageBatch> output;
  std::optional<c10::Event> begin;
  c10::Event end;
  std::optional<std::int64_t> consumer_stream;
  bool received = false;
  bool drained = false;

  explicit Transfer(ImageBatch input, bool timing)
      : source(std::move(input)),
        end(c10::DeviceType::CUDA, timing ? c10::EventFlag::BACKEND_DEFAULT
                                          : c10::EventFlag::PYTORCH_DEFAULT) {
    if (timing)
      begin.emplace(c10::DeviceType::CUDA, c10::EventFlag::BACKEND_DEFAULT);
  }
};

struct Pool {
  std::mutex mutex;
  data::BatchMemoryOptions memory;
  ImageBatchLimits limits;
  std::shared_ptr<ImageResourceBudget> budget;
  c10::Stream stream;
  std::vector<std::weak_ptr<Transfer>> transfers;
  std::size_t capacity;
  bool timing;
  bool closed = false;

  Pool(data::BatchMemoryOptions policy, ImageBatchLimits bounds,
       std::shared_ptr<ImageResourceBudget> resources, c10::Stream selected,
       std::size_t count, bool record_timing)
      : memory(std::move(policy)),
        limits(bounds),
        budget(std::move(resources)),
        stream(selected),
        capacity(count),
        timing(record_timing) {
    transfers.reserve(capacity);
  }

  void Prune() {
    std::erase_if(transfers, [](const auto& weak) {
      const auto transfer = weak.lock();
      return !transfer || transfer->drained;
    });
  }
};
}  // namespace copy_stream_detail

namespace {
using copy_stream_detail::Pool;
using copy_stream_detail::Transfer;

template <class F>
decltype(auto) Backend(const char* operation, F&& call) {
  try {
    return std::forward<F>(call)();
  } catch (const c10::Error& cause) {
    ::jyppx::liuding::detail::rethrow_backend_error(
        cause,
        {.component = "data_image",
         .operation = operation,
         .failure_kind = ::jyppx::liuding::detail::BackendFailureKind::device,
         .expected = "available CUDA device and stream"});
  }
}

void Drain(Transfer& transfer) {
  if (transfer.drained) return;
  transfer.end.synchronize();
  transfer.source.reset();
  if (transfer.received) transfer.output.reset();
  transfer.leases.reset();
  transfer.drained = true;
}

void RecordStorage(const ImageBatch& batch, const c10::Stream& stream) {
  c10::impl::VirtualGuardImpl backend(c10::DeviceType::CUDA);
  const auto record = [&](const Tensor& tensor) {
    if (tensor.defined() && tensor.has_storage())
      backend.recordDataPtrOnStream(tensor.storage().data_ptr(), stream);
  };
  record(batch.images);
  record(batch.valid);
  record(batch.labels);
  if (batch.paired_images) record(*batch.paired_images);
  if (batch.masks) record(*batch.masks);
  for (const auto* list : {&batch.image_list, &batch.paired_image_list,
                           &batch.mask_list, &batch.valid_list})
    for (const auto& tensor : *list) record(tensor);
}

void RequireTicket(const std::shared_ptr<Pool>& pool,
                   const std::shared_ptr<Transfer>& transfer) {
  if (!pool || !transfer) throw InvalidArgument("Empty image transfer ticket");
}

void RequireOrdinaryStream(const c10::Stream& stream) {
  if (stream.is_capturing())
    throw CompatibilityError(
        "Image copy stream does not support CUDA Graph capture");
}
}  // namespace

ImageBatchTransfer::ImageBatchTransfer(std::shared_ptr<Pool> pool,
                                       std::shared_ptr<Transfer> transfer)
    : pool_(std::move(pool)), transfer_(std::move(transfer)) {}

ImageBatchTransfer::~ImageBatchTransfer() { DrainNoexcept(); }
ImageBatchTransfer::ImageBatchTransfer(ImageBatchTransfer&& other) noexcept =
    default;
ImageBatchTransfer& ImageBatchTransfer::operator=(
    ImageBatchTransfer&& other) noexcept {
  if (this != &other) {
    DrainNoexcept();
    transfer_ = std::move(other.transfer_);
    pool_ = std::move(other.pool_);
  }
  return *this;
}

void ImageBatchTransfer::DrainNoexcept() noexcept {
  if (!transfer_) return;
  try {
    wait();
  } catch (...) {
    // Releasing pinned memory before DMA completion would be unsafe. An
    // explicit wait reports the backend error while retaining these owners.
    std::terminate();
  }
}

ImageBatch ImageBatchTransfer::receive() {
  RequireTicket(pool_, transfer_);
  std::lock_guard lock(pool_->mutex);
  if (transfer_->received)
    throw InvalidArgument("Image transfer ticket was already received");
  return Backend("copy_stream.receive", [&] {
    c10::impl::VirtualGuardImpl backend(c10::DeviceType::CUDA);
    const auto consumer = backend.getStream(pool_->stream.device());
    RequireOrdinaryStream(consumer);
    RequireOrdinaryStream(pool_->stream);
    if (consumer == pool_->stream)
      throw CompatibilityError("Image consumer must use a different stream");
    // Copy metadata before any observable handoff. The ticket keeps a second
    // owner until DMA completes even if the receiver immediately drops it.
    auto result = *transfer_->output;
    transfer_->end.block(consumer);
    RecordStorage(result, consumer);
    transfer_->consumer_stream = consumer.id();
    transfer_->received = true;
    if (transfer_->drained) transfer_->output.reset();
    return result;
  });
}

bool ImageBatchTransfer::ready() const {
  RequireTicket(pool_, transfer_);
  std::lock_guard lock(pool_->mutex);
  return Backend("copy_stream.query",
                 [&] { return transfer_->drained || transfer_->end.query(); });
}

void ImageBatchTransfer::wait() {
  RequireTicket(pool_, transfer_);
  std::lock_guard lock(pool_->mutex);
  Backend("copy_stream.wait", [&] { Drain(*transfer_); });
  pool_->Prune();
}

ImageBatchCopyTimeline ImageBatchTransfer::timeline() const {
  RequireTicket(pool_, transfer_);
  std::lock_guard lock(pool_->mutex);
  return Backend("copy_stream.timeline", [&] {
    ImageBatchCopyTimeline result{
        .device = pool_->stream.device(),
        .copy_stream_id = pool_->stream.id(),
        .consumer_stream_id = transfer_->consumer_stream,
        .copy_complete = transfer_->drained || transfer_->end.query()};
    if (result.copy_complete && transfer_->begin)
      result.copy_elapsed_ms = transfer_->begin->elapsedTime(transfer_->end);
    return result;
  });
}

ImageBatchCopyStream::ImageBatchCopyStream(
    data::BatchMemoryOptions memory, std::size_t max_in_flight,
    bool record_timing, ImageBatchLimits limits,
    std::shared_ptr<ImageResourceBudget> resource_budget) {
  data::validate_batch_memory_options(memory);
  detail::CheckBatchLimits(limits);
  if (!memory.pin_memory || !memory.non_blocking || !memory.target_device ||
      !memory.target_device->is_cuda() || memory.target_device->index() < 0 ||
      max_in_flight == 0 || max_in_flight > 4096)
    throw InvalidArgument(
        "Image copy stream requires pinned nonblocking indexed CUDA policy "
        "and capacity in [1,4096]");
  Backend("copy_stream.create", [&] {
    c10::impl::VirtualGuardImpl backend(c10::DeviceType::CUDA);
    const auto device = *memory.target_device;
    if (device.index() >= backend.deviceCount())
      throw DeviceError("Image copy stream CUDA device is unavailable");
    const auto current = backend.getStream(device);
    auto selected = backend.getStreamFromGlobalPool(device);
    if (selected == current) selected = backend.getStreamFromGlobalPool(device);
    if (selected == current)
      throw DeviceError("No distinct CUDA copy stream is available");
    pool_ = std::make_shared<Pool>(std::move(memory), limits,
                                   std::move(resource_budget), selected,
                                   max_in_flight, record_timing);
  });
}

ImageBatchCopyStream::~ImageBatchCopyStream() { CloseNoexcept(); }
ImageBatchCopyStream::ImageBatchCopyStream(
    ImageBatchCopyStream&& other) noexcept = default;
ImageBatchCopyStream& ImageBatchCopyStream::operator=(
    ImageBatchCopyStream&& other) noexcept {
  if (this != &other) {
    CloseNoexcept();
    pool_ = std::move(other.pool_);
  }
  return *this;
}

void ImageBatchCopyStream::CloseNoexcept() noexcept {
  try {
    close();
  } catch (...) {
    std::terminate();
  }
}

ImageBatchTransfer ImageBatchCopyStream::submit(ImageBatch prepared,
                                                std::stop_token stop) {
  if (!pool_) throw InvalidArgument("Empty image copy stream");
  std::lock_guard lock(pool_->mutex);
  if (pool_->closed) throw InvalidArgument("Image copy stream is closed");
  if (stop.stop_requested()) throw Cancelled("Image copy submission cancelled");
  pool_->Prune();
  if (pool_->transfers.size() == pool_->capacity)
    throw ResourceExhausted("Image copy stream in-flight capacity exhausted");
  return Backend("copy_stream.submit", [&] {
    c10::impl::VirtualGuardImpl backend(c10::DeviceType::CUDA);
    RequireOrdinaryStream(backend.getStream(pool_->stream.device()));
    RequireOrdinaryStream(pool_->stream);
    auto state = std::make_shared<Transfer>(std::move(prepared), pool_->timing);
    bool work_started = false;
    try {
      c10::StreamGuard guard(pool_->stream);
      auto value = image_batch_value(*state->source, pool_->limits);
      const auto& first = state->source->layout == ImageBatchLayout::ragged
                              ? state->source->image_list.front()
                              : state->source->images;
      // Structural validation above requires a common device for every leaf.
      // Copied preparation metadata must not admit an already-device batch:
      // its producer stream would otherwise have no dependency on this copy.
      if (!first.device().is_cpu())
        throw CompatibilityError(
            "Image copy stream requires CPU source tensors");
      auto staged = data::detail::transfer_batch_memory_with_admission(
          {std::move(value), state->source->memory}, pool_->memory, [&] {
            const detail::BatchResourceProjection projection{
                .target_device = pool_->memory.target_device};
            state->leases = detail::ReserveBatchResources(
                *state->source, pool_->budget, pool_->limits, stop, true,
                &projection);
            if (stop.stop_requested())
              throw Cancelled("Image copy submission cancelled");
            work_started = true;
            if (state->begin) state->begin->record(pool_->stream);
          });
      auto output = image_batch_from_value(staged.value, pool_->limits);
      output.memory = staged.metadata;
      output.memory.transfer_stream_policy =
          data::BatchTransferStreamPolicy::explicit_copy;
      detail::RetainBatchSourceResources(*state->source, output);
      if (state->leases) output.resource_lease_sets.push_back(state->leases);
      state->output.emplace(std::move(output));
      state->end.record(pool_->stream);
      pool_->transfers.push_back(state);
      return ImageBatchTransfer(pool_, std::move(state));
    } catch (...) {
      if (work_started) {
        try {
          // A failed recursive transfer can already have queued some leaves.
          // Keep both the source and allocation reservations through drainage.
          pool_->stream.synchronize();
        } catch (...) {
          std::terminate();
        }
      }
      throw;
    }
  });
}

std::size_t ImageBatchCopyStream::pending() const {
  if (!pool_) return 0;
  std::lock_guard lock(pool_->mutex);
  pool_->Prune();
  return pool_->transfers.size();
}

void ImageBatchCopyStream::close() {
  if (!pool_) return;
  std::lock_guard lock(pool_->mutex);
  pool_->closed = true;
  Backend("copy_stream.close", [&] {
    for (const auto& weak : pool_->transfers)
      if (const auto state = weak.lock()) Drain(*state);
  });
  pool_->Prune();
}
}  // namespace jyppx::liuding::data::image
