// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_DATA_IMAGE_PANOPTIC_DETAIL_HPP
#define JYPPX_LIUDING_DATA_IMAGE_PANOPTIC_DETAIL_HPP

#include <jyppx/liuding/data_image/panoptic.hpp>
#include <jyppx/liuding/data_image/resource_budget.hpp>

namespace jyppx::liuding::data::image::detail {

struct PanopticBinding {
  ImageResourceLease output_lease;
  VisionAnnotationRecord record;
};

struct PanopticRecordOwner {
  ImageResourceLease output_lease;
  VisionAnnotationRecord record;
};

std::uint64_t PanopticRecordBytes(const VisionAnnotationRecord& record);

PanopticBinding BindPanopticMask(
    const VisionAnnotationRecord& record, const DecodedMask& mask,
    std::size_t category_count, const PanopticBindOptions& options,
    const std::shared_ptr<ImageResourceBudget>& budget, std::stop_token stop);

}  // namespace jyppx::liuding::data::image::detail

#endif  // JYPPX_LIUDING_DATA_IMAGE_PANOPTIC_DETAIL_HPP
