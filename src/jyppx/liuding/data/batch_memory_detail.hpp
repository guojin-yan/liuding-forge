// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_DATA_BATCH_MEMORY_DETAIL_HPP
#define JYPPX_LIUDING_DATA_BATCH_MEMORY_DETAIL_HPP

#include <jyppx/liuding/data/batch_memory.hpp>

#include <functional>

namespace jyppx::liuding::data::detail {

// Internal cooperating-stage admission. The callback runs once after the
// existing value, policy and (for transfer) metadata checks, before any pin or
// destination Tensor allocation. A disabled stage keeps its existing bypass
// behavior, but still invokes admission for an unchanged output value.
using BatchMemoryAdmission = std::function<void()>;

BatchMemoryStageResult prepare_batch_memory_with_admission(
    const c10::IValue& value, const BatchMemoryOptions& options,
    const BatchMemoryAdmission& admit);
BatchMemoryStageResult transfer_batch_memory_with_admission(
    BatchMemoryStageResult prepared, const BatchMemoryOptions& options,
    const BatchMemoryAdmission& admit);

}  // namespace jyppx::liuding::data::detail

#endif  // JYPPX_LIUDING_DATA_BATCH_MEMORY_DETAIL_HPP
