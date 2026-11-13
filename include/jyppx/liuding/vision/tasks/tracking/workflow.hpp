// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_TRACKING_WORKFLOW_HPP
#define JYPPX_LIUDING_VISION_TASKS_TRACKING_WORKFLOW_HPP
/** @file workflow.hpp @brief Detector, ReID and native sequence composition. */
#include <jyppx/liuding/vision/tasks/tracking/tracker.hpp>
#include <jyppx/liuding/vision/tasks/workflow/suite.hpp>
namespace jyppx::liuding::vision::tasks::tracking {
struct FramePrediction { ImageResult detections;std::vector<TrackResult> tracks; };
/** @brief Keeps low confidence detections for the tracker's second association.
 * Optional ReID crops use original source coordinates and classification
 * preprocessing. Affine GMC is computed from the supplied previous frame;
 * passing no previous frame explicitly selects identity for the first frame.
 * No video codec, model checkpoint or tracker singleton is hidden here. */
FramePrediction track_vision_frame(Tracker& tracker,TaskBinding& detector,
    const data::image::VisionSample& frame,const RecipeSpec& detector_recipe,
    std::int64_t frame_id,const TrackerOptions& options={},
    TaskBinding* reid=nullptr,const RecipeSpec* reid_recipe=nullptr,
    const data::image::VisionSample* previous=nullptr,std::stop_token stop={});
}
#endif
