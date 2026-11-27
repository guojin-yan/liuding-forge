// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
#include "roi_gpu_build.hpp"
#include "roi_gpu_launch.hpp"
#ifdef JYPPX_VISION_GPU_HIP
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include <cstdio>
#include <cstring>
namespace jyppx::liuding::vision_ops::detail {
namespace {
using I = std::int64_t;
// Preserve ATen's separate coordinate rounding boundaries. Contraction is
// disabled by the device compiler; half/bfloat16 keep the ATen path.
template <class T>
struct Weights {
  I x0, x1, y0, y1;
  T a, b, c, d;
  bool valid;
};
template <class T>
__device__ Weights<T> weights(const double* parameters, I rx, I ry,
                              const RoiGpuShape& s) {
  const T sx = T(T(rx) + T(.5)) * T(parameters[0]);
  const T sy = T(T(ry) + T(.5)) * T(parameters[1]);
  const T x = sx + T(parameters[2]), y = sy + T(parameters[3]);
  const bool valid =
      x >= T(-1) && x <= T(s.width) && y >= T(-1) && y <= T(s.height);
  const T gx = (x + T(.5)) * T(2. / s.width) - T(1);
  const T gy = (y + T(.5)) * T(2. / s.height) - T(1);
  T xx = ((gx + T(1)) * T(s.width) - T(1)) * T(.5);
  T yy = ((gy + T(1)) * T(s.height) - T(1)) * T(.5);
  xx = xx < T(0) ? T(0) : xx > T(s.width - 1) ? T(s.width - 1) : xx;
  yy = yy < T(0) ? T(0) : yy > T(s.height - 1) ? T(s.height - 1) : yy;
  const I x0 = I(floor(xx)), y0 = I(floor(yy));
  const I x1 = x0 + 1, y1 = y0 + 1;
  const T dx = xx - T(x0), dy = yy - T(y0);
  return {x0,
          x1,
          y0,
          y1,
          (T(1) - dx) * (T(1) - dy),
          dx * (T(1) - dy),
          (T(1) - dx) * dy,
          dx * dy,
          valid};
}
template <class T>
__global__ void forward_kernel(const T* input, const double* rows,
                               const I* batches, T* output, RoiGpuShape s) {
  const I total = s.count * s.channels * s.pooled_h * s.pooled_w;
  for (I index = I(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
       index += I(gridDim.x) * blockDim.x) {
    I rest = index;
    const I px = rest % s.pooled_w;
    rest /= s.pooled_w;
    const I py = rest % s.pooled_h;
    rest /= s.pooled_h;
    const I c = rest % s.channels, r = rest / s.channels;
    const auto* source = input + batches[r] * s.sn + c * s.sc;
    T sum = T(0);
    for (I iy = 0; iy < s.ratio; ++iy)
      for (I ix = 0; ix < s.ratio; ++ix) {
        const auto w =
            weights<T>(rows + r * 4, px * s.ratio + ix, py * s.ratio + iy, s);
        T value = source[w.y0 * s.sh + w.x0 * s.sw] * w.a;
        if (w.x1 < s.width) value += source[w.y0 * s.sh + w.x1 * s.sw] * w.b;
        if (w.y1 < s.height) value += source[w.y1 * s.sh + w.x0 * s.sw] * w.c;
        if (w.x1 < s.width && w.y1 < s.height)
          value += source[w.y1 * s.sh + w.x1 * s.sw] * w.d;
        sum += value * T(w.valid);
      }
    output[index] = sum / T(s.ratio * s.ratio);
  }
}
template <class T>
__global__ void backward_kernel(const T* grad, const double* rows,
                                const I* batches, T* output, RoiGpuShape s) {
  const I total = s.count * s.channels * s.pooled_h * s.pooled_w;
  for (I index = I(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
       index += I(gridDim.x) * blockDim.x) {
    I rest = index;
    const I px = rest % s.pooled_w;
    rest /= s.pooled_w;
    const I py = rest % s.pooled_h;
    rest /= s.pooled_h;
    const I c = rest % s.channels, r = rest / s.channels;
    auto* destination =
        output + (batches[r] * s.channels + c) * s.height * s.width;
    const T value = grad[index] / T(s.ratio * s.ratio);
    for (I iy = 0; iy < s.ratio; ++iy)
      for (I ix = 0; ix < s.ratio; ++ix) {
        const auto w =
            weights<T>(rows + r * 4, px * s.ratio + ix, py * s.ratio + iy, s);
        const T masked = value * T(w.valid);
        atomicAdd(destination + w.y0 * s.width + w.x0, masked * w.a);
        if (w.x1 < s.width)
          atomicAdd(destination + w.y0 * s.width + w.x1, masked * w.b);
        if (w.y1 < s.height)
          atomicAdd(destination + w.y1 * s.width + w.x0, masked * w.c);
        if (w.x1 < s.width && w.y1 < s.height)
          atomicAdd(destination + w.y1 * s.width + w.x1, masked * w.d);
      }
  }
}
}  // namespace
bool vision_gpu_supported() {
  int device = 0;
#ifdef JYPPX_VISION_GPU_HIP
  hipDeviceProp_t properties{};
  if (hipGetDevice(&device) != hipSuccess ||
      hipGetDeviceProperties(&properties, device) != hipSuccess)
    return false;
  char name[sizeof(properties.gcnArchName)]{};
  std::memcpy(name, properties.gcnArchName, sizeof(name) - 1);
  if (auto* suffix = std::strchr(name, ':')) *suffix = '\0';
#else
  cudaDeviceProp properties{};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&properties, device) != cudaSuccess)
    return false;
  char name[16]{};
  std::snprintf(name, sizeof(name), "%d",
                properties.major * 10 + properties.minor);
#endif
  const char* architectures[] = {JYPPX_VISION_GPU_ARCHS};
  for (const auto* arch : architectures)
    if (std::strcmp(name, arch) == 0) return true;
  return false;
}
int launch_roi_gpu(const void* input, const double* rows, const I* batches,
                   void* output, RoiGpuShape shape, bool fp64, bool backward,
                   void* stream) {
  const I count =
      shape.count * shape.channels * shape.pooled_h * shape.pooled_w;
  const auto blocks =
      unsigned((count + 255) / 256 < 4096 ? (count + 255) / 256 : 4096);
#ifdef JYPPX_VISION_GPU_HIP
  auto gpu_stream = static_cast<hipStream_t>(stream);
#else
  auto gpu_stream = static_cast<cudaStream_t>(stream);
#endif
  if (backward) {
    if (fp64)
      backward_kernel<double><<<blocks, 256, 0, gpu_stream>>>(
          static_cast<const double*>(input), rows, batches,
          static_cast<double*>(output), shape);
    else
      backward_kernel<float><<<blocks, 256, 0, gpu_stream>>>(
          static_cast<const float*>(input), rows, batches,
          static_cast<float*>(output), shape);
  } else {
    if (fp64)
      forward_kernel<double><<<blocks, 256, 0, gpu_stream>>>(
          static_cast<const double*>(input), rows, batches,
          static_cast<double*>(output), shape);
    else
      forward_kernel<float><<<blocks, 256, 0, gpu_stream>>>(
          static_cast<const float*>(input), rows, batches,
          static_cast<float*>(output), shape);
  }
#ifdef JYPPX_VISION_GPU_HIP
  return int(hipGetLastError());
#else
  return int(cudaGetLastError());
#endif
}
}  // namespace jyppx::liuding::vision_ops::detail
