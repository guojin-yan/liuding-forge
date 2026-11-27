// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "deformable_gpu_launch.hpp"
#include "roi_gpu_build.hpp"
#ifdef JYPPX_VISION_GPU_HIP
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include <climits>
namespace jyppx::liuding::vision_ops::detail {
using I = std::int64_t;
template <class T>
struct Sample {
  I x, y;
  T dx, dy, a, b, c, d;
};
template <class T>
__device__ Sample<T> geometry(T lx, T ly, I h, I w) {
  const T gx = lx * T(2) - T(1), gy = ly * T(2) - T(1);
  // Grid normalization is separate; GPU grid_sample contracts the subsequent
  // unnormalization. Keep these rounding boundaries explicit with fma.
  T x = fma(gx + T(1), T(w), T(-1)) * T(.5),
    y = fma(gy + T(1), T(h), T(-1)) * T(.5);
  // Match GPU grid_sample's safe integer conversion for zeros padding.
  if (!isfinite(x) || x > T(INT_MAX - 1) || x < T(INT_MIN)) x = T(-100);
  if (!isfinite(y) || y > T(INT_MAX - 1) || y < T(INT_MIN)) y = T(-100);
  const I ix = I(floor(x)), iy = I(floor(y));
  const T dx = x - T(ix), dy = y - T(iy);
  return {ix,
          iy,
          dx,
          dy,
          (T(1) - dx) * (T(1) - dy),
          dx * (T(1) - dy),
          (T(1) - dx) * dy,
          dx * dy};
}
__device__ bool inside(I y, I x, I h, I w) {
  return y >= 0 && y < h && x >= 0 && x < w;
}
template <class T>
__device__ T sample(const T* v, I base, I h, I w, const DeformGpuShape& s,
                    Sample<T> g) {
  T result = T(0);
  if (inside(g.y, g.x, h, w))
    result += v[base + (g.y * w + g.x) * s.vs[1]] * g.a;
  if (inside(g.y, g.x + 1, h, w))
    result += v[base + (g.y * w + g.x + 1) * s.vs[1]] * g.b;
  if (inside(g.y + 1, g.x, h, w))
    result += v[base + ((g.y + 1) * w + g.x) * s.vs[1]] * g.c;
  if (inside(g.y + 1, g.x + 1, h, w))
    result += v[base + ((g.y + 1) * w + g.x + 1) * s.vs[1]] * g.d;
  return result;
}
template <class T>
__global__ void forward_kernel(const T* v, const T* loc, const T* weight,
                               T* out, DeformGpuShape s) {
  const I count = s.batch * s.queries * s.heads * s.channels;
  for (I index = I(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
       index += I(gridDim.x) * blockDim.x) {
    I rest = index;
    const I c = rest % s.channels;
    rest /= s.channels;
    const I head = rest % s.heads;
    rest /= s.heads;
    const I q = rest % s.queries, batch = rest / s.queries;
    T result = T(0);
    for (I l = 0; l < s.levels; ++l) {
      const I h = s.spatial[l * 3], w = s.spatial[l * 3 + 1],
              offset = s.spatial[l * 3 + 2];
      for (I p = 0; p < s.points; ++p) {
        const I li = batch * s.ls[0] + q * s.ls[1] + head * s.ls[2] +
                     l * s.ls[3] + p * s.ls[4];
        const I wi = batch * s.ws[0] + q * s.ws[1] + head * s.ws[2] +
                     l * s.ws[3] + p * s.ws[4];
        auto g = geometry(loc[li], loc[li + s.ls[5]], h, w);
        const I base =
            batch * s.vs[0] + offset * s.vs[1] + head * s.vs[2] + c * s.vs[3];
        result += sample(v, base, h, w, s, g) * weight[wi];
      }
    }
    out[index] = result;
  }
}
template <class T>
__global__ void feature_backward_kernel(const T* loc, const T* weight,
                                        const T* grad, T* out,
                                        DeformGpuShape s) {
  const I count = s.batch * s.queries * s.heads * s.channels;
  for (I index = I(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
       index += I(gridDim.x) * blockDim.x) {
    I rest = index;
    const I c = rest % s.channels;
    rest /= s.channels;
    const I head = rest % s.heads;
    rest /= s.heads;
    const I q = rest % s.queries, batch = rest / s.queries;
    for (I l = 0; l < s.levels; ++l) {
      const I h = s.spatial[l * 3], w = s.spatial[l * 3 + 1],
              offset = s.spatial[l * 3 + 2];
      for (I p = 0; p < s.points; ++p) {
        const I li = batch * s.ls[0] + q * s.ls[1] + head * s.ls[2] +
                     l * s.ls[3] + p * s.ls[4];
        const I wi = batch * s.ws[0] + q * s.ws[1] + head * s.ws[2] +
                     l * s.ws[3] + p * s.ws[4];
        auto g = geometry(loc[li], loc[li + s.ls[5]], h, w);
        const T go = grad[index] * weight[wi];
        const I base = (batch * s.tokens + offset) * s.heads * s.channels +
                       head * s.channels + c;
        const I step = s.heads * s.channels;
        if (inside(g.y, g.x, h, w))
          atomicAdd(out + base + (g.y * w + g.x) * step, g.a * go);
        if (inside(g.y, g.x + 1, h, w))
          atomicAdd(out + base + (g.y * w + g.x + 1) * step, g.b * go);
        if (inside(g.y + 1, g.x, h, w))
          atomicAdd(out + base + ((g.y + 1) * w + g.x) * step, g.c * go);
        if (inside(g.y + 1, g.x + 1, h, w))
          atomicAdd(out + base + ((g.y + 1) * w + g.x + 1) * step, g.d * go);
      }
    }
  }
}
template <class T>
__global__ void point_backward_kernel(const T* v, const T* loc, const T* weight,
                                      const T* grad, T* gradloc, T* gradweight,
                                      DeformGpuShape s) {
  const I count = s.batch * s.queries * s.heads * s.levels * s.points;
  for (I index = I(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
       index += I(gridDim.x) * blockDim.x) {
    I rest = index;
    const I p = rest % s.points;
    rest /= s.points;
    const I l = rest % s.levels;
    rest /= s.levels;
    const I head = rest % s.heads;
    rest /= s.heads;
    const I q = rest % s.queries, batch = rest / s.queries;
    const I h = s.spatial[l * 3], w = s.spatial[l * 3 + 1],
            offset = s.spatial[l * 3 + 2];
    const I li = batch * s.ls[0] + q * s.ls[1] + head * s.ls[2] + l * s.ls[3] +
                 p * s.ls[4];
    const I wi = batch * s.ws[0] + q * s.ws[1] + head * s.ws[2] + l * s.ws[3] +
                 p * s.ws[4];
    auto g = geometry(loc[li], loc[li + s.ls[5]], h, w);
    T gx = T(0), gy = T(0), gw = T(0);
    for (I c = 0; c < s.channels; ++c) {
      const T upstream =
          grad[((batch * s.queries + q) * s.heads + head) * s.channels + c];
      const T go = upstream * weight[wi];
      const I base =
          batch * s.vs[0] + offset * s.vs[1] + head * s.vs[2] + c * s.vs[3];
      T sampled = T(0);
      // Preserve grid_sample's channel/neighbor order and contracted location
      // accumulation; ordinary multiply/add changes small location gradients.
      if (inside(g.y, g.x, h, w)) {
        const T value = v[base + (g.y * w + g.x) * s.vs[1]];
        sampled += value * g.a;
        gx = fma(-(value * (T(1) - g.dy)), go, gx);
        gy = fma(-(value * (T(1) - g.dx)), go, gy);
      }
      if (inside(g.y, g.x + 1, h, w)) {
        const T value = v[base + (g.y * w + g.x + 1) * s.vs[1]];
        sampled += value * g.b;
        gx = fma(value * (T(1) - g.dy), go, gx);
        gy = fma(-(value * g.dx), go, gy);
      }
      if (inside(g.y + 1, g.x, h, w)) {
        const T value = v[base + ((g.y + 1) * w + g.x) * s.vs[1]];
        sampled += value * g.c;
        gx = fma(-(value * g.dy), go, gx);
        gy = fma(value * (T(1) - g.dx), go, gy);
      }
      if (inside(g.y + 1, g.x + 1, h, w)) {
        const T value = v[base + ((g.y + 1) * w + g.x + 1) * s.vs[1]];
        sampled += value * g.d;
        gx = fma(value * g.dy, go, gx);
        gy = fma(value * g.dx, go, gy);
      }
      gw += sampled * upstream;
    }
    gradloc[index * 2] = (gx * (T(w) / T(2))) * T(2);
    gradloc[index * 2 + 1] = (gy * (T(h) / T(2))) * T(2);
    gradweight[index] = gw;
  }
}
int launch_deform_gpu_forward(const void* v, const void* l, const void* w,
                              void* out, DeformGpuShape s, bool fp64,
                              void* stream) {
#ifdef JYPPX_VISION_GPU_HIP
  const auto gpu_stream = static_cast<hipStream_t>(stream);
#else
  const auto gpu_stream = static_cast<cudaStream_t>(stream);
#endif
  const I count = s.batch * s.queries * s.heads * s.channels;
  const auto blocks =
      unsigned((count + 255) / 256 < 4096 ? (count + 255) / 256 : 4096);
  if (fp64)
    forward_kernel<double><<<blocks, 256, 0, gpu_stream>>>(
        static_cast<const double*>(v), static_cast<const double*>(l),
        static_cast<const double*>(w), static_cast<double*>(out), s);
  else
    forward_kernel<float><<<blocks, 256, 0, gpu_stream>>>(
        static_cast<const float*>(v), static_cast<const float*>(l),
        static_cast<const float*>(w), static_cast<float*>(out), s);
#ifdef JYPPX_VISION_GPU_HIP
  return int(hipGetLastError());
#else
  return int(cudaGetLastError());
#endif
}
int launch_deform_gpu_backward(const void* v, const void* l, const void* w,
                               const void* grad, void* gv, void* gl, void* gw,
                               DeformGpuShape s, bool fp64, void* stream) {
#ifdef JYPPX_VISION_GPU_HIP
  const auto gpu_stream = static_cast<hipStream_t>(stream);
#else
  const auto gpu_stream = static_cast<cudaStream_t>(stream);
#endif
  const I count = s.batch * s.queries * s.heads * s.channels,
          points = s.batch * s.queries * s.heads * s.levels * s.points;
  const auto blocks = unsigned((count + 255) / 256 < 4096 ? (count + 255) / 256
                                                          : 4096),
             pblocks = unsigned(
                 (points + 255) / 256 < 4096 ? (points + 255) / 256 : 4096);
  if (fp64) {
    feature_backward_kernel<double><<<blocks, 256, 0, gpu_stream>>>(
        static_cast<const double*>(l), static_cast<const double*>(w),
        static_cast<const double*>(grad), static_cast<double*>(gv), s);
    point_backward_kernel<double><<<pblocks, 256, 0, gpu_stream>>>(
        static_cast<const double*>(v), static_cast<const double*>(l),
        static_cast<const double*>(w), static_cast<const double*>(grad),
        static_cast<double*>(gl), static_cast<double*>(gw), s);
  } else {
    feature_backward_kernel<float><<<blocks, 256, 0, gpu_stream>>>(
        static_cast<const float*>(l), static_cast<const float*>(w),
        static_cast<const float*>(grad), static_cast<float*>(gv), s);
    point_backward_kernel<float><<<pblocks, 256, 0, gpu_stream>>>(
        static_cast<const float*>(v), static_cast<const float*>(l),
        static_cast<const float*>(w), static_cast<const float*>(grad),
        static_cast<float*>(gl), static_cast<float*>(gw), s);
  }
#ifdef JYPPX_VISION_GPU_HIP
  return int(hipGetLastError());
#else
  return int(cudaGetLastError());
#endif
}
}  // namespace jyppx::liuding::vision_ops::detail
