// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_WORKFLOW_BATCH_TRANSFER_HPP
#define JYPPX_LIUDING_VISION_TASKS_WORKFLOW_BATCH_TRANSFER_HPP

#include <jyppx/liuding/data_image/image_batch.hpp>

namespace jyppx::liuding::vision::tasks::detail {
inline data::image::ImageBatch transfer_task_batch(
    const data::image::ImageBatch& batch, const Device& device,
    const data::image::ImageBatchLimits& limits = {}) {
  data::BatchMemoryOptions options;
  options.target_device = device;
  auto prepared = data::image::prepare_image_batch(batch, options, limits);
  return data::image::transfer_image_batch(std::move(prepared), options,
                                           limits);
}
}  // namespace jyppx::liuding::vision::tasks::detail
#endif
