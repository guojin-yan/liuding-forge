// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// Private official oneDNN/ATen integer-composition implementation.
#ifndef JYPPX_LIUDING_QUANTIZATION_ONEDNN_INTEGER_MATMUL_HPP
#define JYPPX_LIUDING_QUANTIZATION_ONEDNN_INTEGER_MATMUL_HPP

#include <ATen/core/Tensor.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace jyppx::liuding::quantization_onednn::detail {

// Does not set ISA or qEngine. The separately pinned SDK and effective AVX2
// must already be available. This is not an executed-PC attestation.
void require_sdk();

// q = (q & 127) + 128 * (q >> 7), with unchanged original U8 bytes.
// Both reductions are official oneDNN gemm:jit. ATen performs the pointwise
// split, INT64 reconstruction and cached weight compensation reduction.
// There is no handwritten K-dimensional reduction in this implementation.
// Caller serializes all methods. Returned tensors borrow one batch's buffers
// and remain valid only until the next prepare/execute or destruction.
class IntegerMatmul final {
 public:
  IntegerMatmul(std::vector<std::int8_t> weights_nk, std::size_t inputs,
                std::size_t outputs, std::size_t max_batch,
                std::size_t max_io_bytes,
                std::uint64_t external_fixed_bytes = 0,
                std::uint64_t external_row_bytes = 0);
  ~IntegerMatmul();
  IntegerMatmul(const IntegerMatmul&) = delete;
  IntegerMatmul& operator=(const IntegerMatmul&) = delete;

  void prepare_batch(std::int64_t batch);
  const at::Tensor& execute(const at::Tensor& original_u8);
  const at::Tensor& low_s32() const;
  const at::Tensor& high_s32() const;
  std::string selected_implementation() const;
  std::uint64_t logical_memory_bytes() const;
  std::uint64_t packing_reserve_bytes() const;
  std::uint64_t scratchpad_bytes() const;
  std::uint64_t maximum_weight_abs_sum() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace jyppx::liuding::quantization_onednn::detail
#endif
