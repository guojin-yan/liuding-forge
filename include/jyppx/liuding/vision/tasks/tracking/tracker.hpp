// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_TRACKING_TRACKER_HPP
#define JYPPX_LIUDING_VISION_TASKS_TRACKING_TRACKER_HPP
/** @file tracker.hpp @brief Native stateful ByteTrack and appearance association. */
#include <jyppx/liuding/vision/tasks/results.hpp>
namespace jyppx::liuding::vision::tasks::tracking {
enum class TrackerKind { byte_track, bot_sort };
struct TrackerOptions {
  TrackerKind kind=TrackerKind::byte_track;
  double high_threshold=.5,low_threshold=.1,new_threshold=.6;
  double match_threshold=.8,second_match_threshold=.5;
  double proximity_threshold=.5,appearance_threshold=.25;
  std::uint64_t track_buffer=30;
  std::int64_t max_tracks=1024,max_detections=1024;
  bool fuse_score=true;
  bool global_motion_compensation=true;
};
struct TrackResult {
  std::uint64_t track_id=0;std::int64_t category=0;
  std::array<double,4> box{};double confidence=0;
};
/** @brief Consumes detector results before their low confidence boxes are
 * discarded. ByteTrack uses XYAH Kalman prediction, high/low association and
 * lost/unconfirmed track lifecycles. BoT-SORT mode uses XYWH, normalized ReID
 * embeddings, proximity gating and explicitly supplied affine GMC (or identity
 * when disabled). Video decoding is an independent caller-owned boundary.
 * A sequence has strictly increasing frame IDs; reset before another sequence.
 * Snapshot includes full Kalman covariance, appearances, ID allocator and
 * lifecycle state; restore is transactional and config/sequence checked.
 * Instances are not thread safe. Tracking itself has no trainable parameters.
 */
class Tracker final {
 public:
  explicit Tracker(TrackerOptions options={});~Tracker();
  Tracker(Tracker&&)noexcept;Tracker& operator=(Tracker&&)noexcept;
  Tracker(const Tracker&)=delete;Tracker& operator=(const Tracker&)=delete;
  void reset(std::string sequence_id);
  std::vector<TrackResult> step(const ImageResult& detections,std::int64_t frame_id,
      const Tensor& embeddings={},const Tensor& affine_gmc={});
  std::string snapshot()const;
  void restore(std::string_view json);
 private:
  struct Impl;std::unique_ptr<Impl> impl_;
};
/** @brief Bounded native grayscale patch matching and robust affine fit.
 * Inputs RGB float/uint8 CHW at equal extents; output CPU float64 [2,3],
 * identity on insufficient/inconsistent matches. Distinct native GMC profile,
 * no implicit video/OpenCV dependency. */
Tensor estimate_affine_motion(const Tensor& previous,const Tensor& current,
    std::int64_t search_radius=8,std::int64_t grid_step=24);
}
#endif
