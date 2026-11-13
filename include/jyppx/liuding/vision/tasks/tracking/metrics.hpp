// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_TRACKING_METRICS_HPP
#define JYPPX_LIUDING_VISION_TASKS_TRACKING_METRICS_HPP
/** @file metrics.hpp @brief Native sequence metrics with global identity matching. */
#include <jyppx/liuding/vision/tasks/tracking/tracker.hpp>
#include <map>
namespace jyppx::liuding::vision::tasks::tracking {
struct TrackingFrame {
  std::int64_t frame_id=0;
  std::vector<TrackResult> ground_truth,predictions;
};
/** @brief One already-filtered sequence: MOTA with continuity matching at .5,
 * IDF1 with global identity assignment, HOTA averaged over .05:.05:.95.
 * Ignored/crowd/distractor regions must be removed before this contract.
 * Bounded to 1024 objects/frame, 4096 IDs/side, 100 million total frame pairs.
 * Empty frames contribute false positives/negatives; no trainable state. */
std::map<std::string,double> evaluate_tracking(std::span<const TrackingFrame> frames,
    std::stop_token stop={});
/** @brief Native MOTChallenge CSV (frame,id,left,top,width,height,score,-1,-1,-1).
 * Input/output coordinates use continuous pixels; frames remain caller-defined. */
std::vector<TrackingFrame> read_mot_sequence(const std::filesystem::path& path,
    bool ground_truth,std::size_t max_rows=1000000);
void write_mot_sequence(const std::filesystem::path& path,
    std::span<const TrackingFrame> frames);
}
#endif
