// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_DATA_BATCH_TRANSFORM_DETAIL_HPP
#define JYPPX_LIUDING_DATA_BATCH_TRANSFORM_DETAIL_HPP

#include <jyppx/liuding/data/batch_memory.hpp>
#include <jyppx/liuding/data/batch_transform.hpp>

#include <optional>

namespace jyppx::liuding::data::detail {

void validate_consumer_transform(
    const std::optional<BatchConsumerTransform>& transform);

BatchMemoryUsage run_consumer_transform(c10::IValue& value,
                                        const BatchConsumerTransform& transform,
                                        const BatchConsumerContext& context,
                                        const BatchMemoryOptions& limits,
                                        std::stop_token stop);

std::shared_ptr<void> run_resource_lease_factory(
    const BatchResourceLeaseFactory& factory, const c10::IValue& value,
    const BatchMemoryMetadata& metadata, std::stop_token stop);

std::shared_ptr<void> run_admission_factory(
    const BatchAdmissionFactory& factory, std::stop_token stop);

}  // namespace jyppx::liuding::data::detail

#endif  // JYPPX_LIUDING_DATA_BATCH_TRANSFORM_DETAIL_HPP
