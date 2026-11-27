// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "roi_gpu_build.hpp"
#include "yolo_gpu_launch.hpp"
#ifdef JYPPX_VISION_GPU_HIP
#include <hip/hip_runtime.h>
using YoloGpuStream = hipStream_t;
static int yolo_last_error() { return int(hipGetLastError()); }
#else
#include <cuda_runtime.h>
using YoloGpuStream = cudaStream_t;
static int yolo_last_error() { return int(cudaGetLastError()); }
#endif

namespace jyppx::liuding::vision_ops::detail {
using I = std::int64_t;
namespace {

// Retain NaN propagation and separate FP32 operation boundaries. The SDK-only
// kernel is compiled without floating-point contraction or fast-math.
__device__ float minimum(float a, float b) {
  return isnan(a) ? a : isnan(b) ? b : a < b ? a : b;
}
__device__ float maximum(float a, float b) {
  return isnan(a) ? a : isnan(b) ? b : a > b ? a : b;
}
__device__ float positive(float a) { return a < 0.f ? 0.f : a; }
__device__ float power(float value, float exponent) {
  if (exponent == .5f) return sqrtf(value);
  if (exponent == 1.f) return value;
  if (exponent == 2.f) return value * value;
  if (exponent == 3.f) return value * value * value;
  return powf(value, exponent);
}
__global__ void kernel(const float* gt, const float* candidates,
                       const float* centers, const float* boxes,
                       const float* scores, const I* images, const I* classes,
                       const bool* valid, bool* inside, float* overlaps,
                       float* alignment, YoloPairGpuShape s) {
  const I total = s.size * s.count * s.anchors;
  for (I index = I(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
       index += I(gridDim.x) * blockDim.x) {
    const I anchor = index % s.anchors;
    const I row = (index / s.anchors) % s.count;
    const I image_slot = index / s.anchors / s.count;
    const I image = images[image_slot * s.images];
    const auto* a = gt + image_slot * s.gt[0] + row * s.gt[1];
    const auto* candidate =
        candidates + image_slot * s.candidates[0] + row * s.candidates[1];
    const auto* b = boxes + image * s.boxes[0] + anchor * s.boxes[1];
    float l[4], r[4];
    for (int k = 0; k < 4; ++k) {
      l[k] = a[k * s.gt[2]];
      r[k] = b[k * s.boxes[2]];
    }
    const float cx = centers[anchor * s.centers[0]];
    const float cy = centers[anchor * s.centers[0] + s.centers[1]];
    const float dx1 = cx - candidate[0];
    const float dy1 = cy - candidate[s.candidates[2]];
    const float dx2 = candidate[2 * s.candidates[2]] - cx;
    const float dy2 = candidate[3 * s.candidates[2]] - cy;
    const float distance = minimum(minimum(dx1, dy1), minimum(dx2, dy2));
    const bool eligible = distance > float(1e-9) &&
                          valid[image_slot * s.valid[0] + row * s.valid[1]];
    inside[index] = eligible;
    const float lw = l[2] - l[0], lh = (l[3] - l[1]) + float(1e-7);
    const float rw = r[2] - r[0], rh = (r[3] - r[1]) + float(1e-7);
    const float iw = positive(minimum(l[2], r[2]) - maximum(l[0], r[0]));
    const float ih = positive(minimum(l[3], r[3]) - maximum(l[1], r[1]));
    const float intersection = iw * ih;
    const float iou =
        intersection / (((lw * lh + rw * rh) - intersection) + float(1e-7));
    const float cw = maximum(l[2], r[2]) - minimum(l[0], r[0]);
    const float ch = maximum(l[3], r[3]) - minimum(l[1], r[1]);
    const float diagonal = (cw * cw + ch * ch) + float(1e-7);
    const float xdistance = ((r[0] + r[2]) - l[0]) - l[2];
    const float ydistance = ((r[1] + r[3]) - l[1]) - l[3];
    const float center_distance =
        (xdistance * xdistance + ydistance * ydistance) / float(4);
    const float angle = atanf(rw / rh) - atanf(lw / lh);
    const float aspect =
        float(4. / (3.14159265358979323846 * 3.14159265358979323846)) *
        (angle * angle);
    const float alpha = aspect / (((float(1) - iou) + aspect) + float(1e-7));
    const float overlap =
        positive(iou - (center_distance / diagonal + aspect * alpha)) *
        float(eligible);
    overlaps[index] = overlap;
    const I label = classes[image_slot * s.classes[0] + row * s.classes[1]];
    const float confidence = scores[image * s.scores[0] + anchor * s.scores[1] +
                                    label * s.scores[2]];
    alignment[index] =
        (power(confidence, s.alpha) * power(overlap, s.beta)) * float(eligible);
  }
}
}  // namespace
int launch_yolo_pair_gpu(const float* gt, const float* candidates,
                         const float* centers, const float* boxes,
                         const float* scores, const I* images, const I* classes,
                         const bool* valid, bool* inside, float* overlaps,
                         float* alignment, YoloPairGpuShape shape,
                         void* stream) {
  const I total = shape.size * shape.count * shape.anchors;
  const auto blocks =
      unsigned((total + 255) / 256 < 4096 ? (total + 255) / 256 : 4096);
  kernel<<<blocks, 256, 0, static_cast<YoloGpuStream>(stream)>>>(
      gt, candidates, centers, boxes, scores, images, classes, valid, inside,
      overlaps, alignment, shape);
  return yolo_last_error();
}
}  // namespace jyppx::liuding::vision_ops::detail

namespace jyppx::liuding::vision_ops::detail {
namespace {
__global__ void resolve_kernel(bool* positive, const float* overlaps,
                               const float* alignment, float* masked, I size,
                               I count, I anchors) {
  const I total = size * anchors;
  for (I index = I(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
       index += I(gridDim.x) * blockDim.x) {
    const I s = index / anchors, a = index % anchors;
    I positives = 0, best = 0;
    float highest = overlaps[(s * count) * anchors + a];
    for (I g = 0; g < count; ++g) {
      const I offset = (s * count + g) * anchors + a;
      positives += I(positive[offset]);
      const float value = overlaps[offset];
      if ((isnan(value) && !isnan(highest)) || value > highest) {
        highest = value;
        best = g;
      }
    }
    for (I g = 0; g < count; ++g) {
      const I offset = (s * count + g) * anchors + a;
      const bool keep = positives > 1 ? g == best : positive[offset];
      positive[offset] = keep;
      masked[offset] = alignment[offset] * float(keep);
    }
  }
}
__global__ void mask_kernel(const bool* positive, const float* overlaps,
                            const float* alignment, float* masked,
                            float* masked_overlap, I total) {
  for (I i = I(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += I(gridDim.x) * blockDim.x) {
    const float p = float(positive[i]);
    masked[i] = alignment[i] * p;
    masked_overlap[i] = overlaps[i] * p;
  }
}
__global__ void finish_kernel(const bool* positive, const float* masked,
                              const float* best_alignment,
                              const float* best_overlap, const float* gt,
                              const I* classes, const I* images, const I* rows,
                              float* boxes, float* scores, bool* foreground,
                              I* target_ids, YoloPostGpuShape s) {
  const I total = s.size * s.anchors;
  for (I index = I(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
       index += I(gridDim.x) * blockDim.x) {
    const I batch = index / s.anchors, a = index % s.anchors;
    I assigned = 0;
    bool found = false;
    float quality = 0.f;
    for (I g = 0; g < s.count; ++g) {
      const I offset = (batch * s.count + g) * s.anchors + a;
      if (positive[offset] && !found) {
        assigned = g;
        found = true;
      }
      const I norm = batch * s.count + g;
      const float value = (masked[offset] * best_overlap[norm]) /
                          (best_alignment[norm] + float(1e-9));
      quality = g == 0 ? value : maximum(quality, value);
    }
    const I image = images[batch * s.images];
    const I label = classes[batch * s.classes[0] + assigned * s.classes[1]];
    scores[image * s.scores[0] + a * s.scores[1] + label * s.scores[2]] =
        quality;
    for (int k = 0; k < 4; ++k)
      boxes[image * s.boxes[0] + a * s.boxes[1] + k * s.boxes[2]] =
          gt[batch * s.gt[0] + assigned * s.gt[1] + k * s.gt[2]];
    foreground[image * s.foreground[0] + a * s.foreground[1]] = found;
    target_ids[image * s.target_ids[0] + a * s.target_ids[1]] =
        found ? rows[batch * s.rows[0] + assigned * s.rows[1]] : I(-1);
  }
}
unsigned blocks(I total) {
  return unsigned((total + 255) / 256 < 4096 ? (total + 255) / 256 : 4096);
}
}  // namespace
int launch_yolo_conflict_gpu(bool* positive, const float* overlaps,
                             const float* alignment, float* masked, I size,
                             I count, I anchors, void* stream) {
  resolve_kernel<<<blocks(size * anchors), 256, 0,
                   static_cast<YoloGpuStream>(stream)>>>(
      positive, overlaps, alignment, masked, size, count, anchors);
  return yolo_last_error();
}
int launch_yolo_mask_gpu(const bool* positive, const float* overlaps,
                         const float* alignment, float* masked,
                         float* masked_overlap, I size, I count, I anchors,
                         void* stream) {
  const I total = size * count * anchors;
  mask_kernel<<<blocks(total), 256, 0, static_cast<YoloGpuStream>(stream)>>>(
      positive, overlaps, alignment, masked, masked_overlap, total);
  return yolo_last_error();
}
int launch_yolo_finish_gpu(const bool* positive, const float* masked,
                           const float* best_alignment,
                           const float* best_overlap, const float* gt,
                           const I* classes, const I* images, const I* rows,
                           float* boxes, float* scores, bool* foreground,
                           I* target_ids, YoloPostGpuShape shape,
                           void* stream) {
  finish_kernel<<<blocks(shape.size * shape.anchors), 256, 0,
                  static_cast<YoloGpuStream>(stream)>>>(
      positive, masked, best_alignment, best_overlap, gt, classes, images, rows,
      boxes, scores, foreground, target_ids, shape);
  return yolo_last_error();
}
}  // namespace jyppx::liuding::vision_ops::detail
