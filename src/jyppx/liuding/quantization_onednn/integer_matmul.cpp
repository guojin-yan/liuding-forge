// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "integer_matmul.hpp"

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/quantization_onednn/onednn_build_identity.hpp>

#include <ATen/ATen.h>
#include <algorithm>
#include <array>
#include <cfenv>
#include <limits>
#include <oneapi/dnnl/dnnl.hpp>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <utility>

#if DNNL_VERSION_MAJOR != 3 || DNNL_VERSION_MINOR != 12 || \
    DNNL_VERSION_PATCH != 0
#error "The composite profile requires official oneDNN 3.12.0"
#endif
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SEQ || \
    DNNL_GPU_RUNTIME != DNNL_RUNTIME_NONE
#error "The composite profile requires the pinned SEQ CPU / NONE GPU SDK"
#endif

namespace jyppx::liuding::quantization_onednn::detail {
namespace j = jyppx::liuding;
namespace {
std::uint64_t Add(std::uint64_t a, std::uint64_t b) {
  if (b > std::numeric_limits<std::uint64_t>::max() - a)
    throw j::ResourceExhausted("Composite byte accounting overflow");
  return a + b;
}
std::uint64_t Mul(std::uint64_t a, std::uint64_t b) {
  if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a)
    throw j::ResourceExhausted("Composite byte accounting overflow");
  return a * b;
}
std::uint64_t RoundUp(std::uint64_t a, std::uint64_t b) {
  return Mul(Add(a, b - 1U) / b, b);
}
void Budget(std::uint64_t bytes, std::size_t ceiling) {
  if (bytes > ceiling)
    throw j::ResourceExhausted(
        "Composite batch/workspace exceeds max_io_bytes");
}

// Conservative logical reserve for one simultaneous SEQ AVX2 GEMM packing
// allocation, not an allocator/RSS cap. This SDK is BLAS NONE, effective AVX2,
// not AVX512/AMX/VNNI; no packed API, alpha1/beta0. Source 80afa71049cd:
// gemm_info.cpp:245-259; gemm_driver.cpp:188-261,570-630,1848-1851.
// Matmul swaps GEMM dimensions: internal M=N(outputs), internal N=batch.
// Reserve full padded operands, sums/offsets, six alignment pages, matmul's
// compensation vectors and 16KiB additional stack/bookkeeping allowance.
// Both branches run sequentially and reuse the same user scratchpad.
std::uint64_t PackingReserve(std::uint64_t batch, std::uint64_t k,
                             std::uint64_t outputs) {
  const auto mp = RoundUp(std::max<std::uint64_t>(outputs, 16U), 16U);
  const auto np = RoundUp(std::max<std::uint64_t>(batch, 4U), 4U);
  const auto kp =
      std::max<std::uint64_t>(128U, std::min<std::uint64_t>(k, 384U));
  auto bytes = Add(Mul(mp, kp), Mul(kp, np));
  bytes = Add(bytes, Add(Mul(mp, 4U), Mul(np, 8U)));
  bytes = Add(bytes, 16U * 4U + 6U * 4096U + 16U * 1024U);
  return Add(bytes, Mul(Add(batch, outputs), 4U));
}
}  // namespace

void require_sdk() {
  const auto* version = dnnl_version();
  if (!version || version->major != 3 || version->minor != 12 ||
      version->patch != 0 || version->cpu_runtime != DNNL_RUNTIME_SEQ ||
      version->gpu_runtime != DNNL_RUNTIME_NONE ||
      dnnl::get_effective_cpu_isa() != dnnl::cpu_isa::avx2)
    throw j::CompatibilityError(
        "Composite profile requires official 3.12.0 SEQ/NONE effective AVX2");
  const std::string_view runtime_hash(version->hash ? version->hash : "");
  // Archive SDKs may report N/A. An available hash must match the exact pinned
  // source; the immutable compiled archive/configuration SHAs bind either case.
  if (!runtime_hash.empty() && runtime_hash != "N/A" &&
      runtime_hash != build_identity::source_commit)
    throw j::CompatibilityError(
        "Official oneDNN runtime source hash differs from the compiled SDK");
  if (std::fegetround() != FE_TONEAREST)
    throw j::CompatibilityError(
        "Composite profile requires nearest-even rounding");
}

struct IntegerMatmul::Impl {
  using Memory = dnnl::memory;
  using Dims = Memory::dims;
  using Data = Memory::data_type;
  using Tag = Memory::format_tag;
  struct Plan {
    std::int64_t batch;
    dnnl::matmul::primitive_desc descriptor;
    dnnl::matmul operation;
    std::optional<Memory> scratchpad{};
    at::Tensor low{}, high{}, low_s32{}, high_s32{}, low_i64{}, high_i64{},
        result_i64{};
    std::uint64_t logical_bytes = 0, packing_bytes = 0, scratch_bytes = 0;
  };

  Impl(std::vector<std::int8_t> original, std::size_t inputs,
       std::size_t outputs, std::size_t batch_ceiling, std::size_t io_ceiling,
       std::uint64_t client_fixed, std::uint64_t client_row)
      : weights(std::move(original)),
        k(inputs),
        n(outputs),
        max_batch(batch_ceiling),
        max_io(io_ceiling),
        external_fixed(client_fixed),
        external_row(client_row),
        engine(dnnl::engine::kind::cpu, 0),
        stream(engine) {
    require_sdk();
    if (k == 0 || n == 0 || k > 65536U || n > 65536U || max_batch == 0 ||
        max_io == 0 || weights.size() != Mul(k, n))
      throw j::InvalidArgument(
          "Composite original weight shape/budgets are invalid");
    // Includes original weight bytes, deliberately conservative 16 bytes per
    // element for INT64 conversion/absolute temporaries, four N-sized INT64
    // vectors, client metadata and scalar/rounding temporaries.
    const auto load_bytes = Add(Add(Mul(weights.size(), 17U), Mul(n, 32U)),
                                Add(external_fixed, 256U));
    Budget(load_bytes, max_io);
    auto view = at::from_blob(
        weights.data(),
        {static_cast<std::int64_t>(n), static_cast<std::int64_t>(k)},
        at::TensorOptions().device(at::kCPU).dtype(at::kChar));
    auto wide = view.to(at::kLong);
    const std::array<std::int64_t, 1> axis{1};
    compensation = at::sum(wide, axis, false, at::kLong).contiguous();
    wide.abs_();  // int64, so abs(-128) is representable.
    const auto magnitude = at::sum(wide, axis, false, at::kLong);
    const auto signed_maximum = magnitude.max().item<std::int64_t>();
    if (signed_maximum < 0)
      throw j::NumericalError("Composite official weight magnitude overflow");
    max_weight_abs_sum = static_cast<std::uint64_t>(signed_maximum);
    // A selected gemm:jit S32 path may have an internal FP32 postprocessor.
    // Do not assume raw S32 is automatically exact: every possible component
    // prefix is bounded by 127*S <= 2^24, so conversion and identity PP remain
    // exact. This conservative initial contract is deliberately retained.
    constexpr auto i32max =
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
    constexpr auto i64max =
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    if (max_weight_abs_sum > (1ULL << 24U) / 127U ||
        max_weight_abs_sum > i32max / 127U ||
        max_weight_abs_sum > i32max / 128U ||
        max_weight_abs_sum > i64max / 510U)
      throw j::NumericalError(
          "Composite row violates exact-component/INT32/INT64 bounds");
    compensation.mul_(128);  // Official int64 reduction result, cached once.
  }

  Plan make_plan(std::int64_t batch) const {
    require_sdk();
    if (batch <= 0 || static_cast<std::uint64_t>(batch) > max_batch)
      throw j::ResourceExhausted("Composite batch exceeds max_batch");
    const Memory::desc src({batch, static_cast<std::int64_t>(k)}, Data::u8,
                           Tag::ab);
    // Logical [K,N], unchanged original physical [N,K].
    const Memory::desc wei(
        {static_cast<std::int64_t>(k), static_cast<std::int64_t>(n)}, Data::s8,
        Dims{1, static_cast<std::int64_t>(k)});
    const Memory::desc dst({batch, static_cast<std::int64_t>(n)}, Data::s32,
                           Tag::ab);
    dnnl::primitive_attr attributes;
    attributes.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    attributes.set_fpmath_mode(dnnl::fpmath_mode::strict);
    // Default scales=1/zero points=0, no bias, no post-ops. Do not set runtime
    // scale/zero-point masks: these two calls compute only component dots.
    auto descriptor =
        dnnl::matmul::primitive_desc(engine, src, wei, dst, attributes);
    std::size_t visited = 0;
    while (std::string_view(descriptor.impl_info_str()) != "gemm:jit") {
      if (++visited > 256U || !descriptor.next_impl())
        throw j::NotImplemented(
            "Pinned SDK has no gemm:jit component; no fallback");
    }
    if (descriptor.src_desc() != src || descriptor.weights_desc() != wei ||
        descriptor.dst_desc() != dst)
      throw j::BackendError(
          "Composite primitive changed an explicit descriptor");
    const auto count_in = Mul(static_cast<std::uint64_t>(batch), k);
    const auto count_out = Mul(static_cast<std::uint64_t>(batch), n);
    // U8 low/high, two S32 outputs, two INT64 conversions and INT64 result.
    auto bytes = Add(Mul(count_in, 2U), Mul(count_out, 32U));
    bytes = Add(
        bytes, Add(weights.size(), Mul(n, 8U)));  // owned weight + compensation
    bytes = Add(bytes, Add(external_fixed, Mul(batch, external_row)));
    bytes =
        Add(bytes, 256U);  // scalar/epilogue temporaries, not an RSS assertion
    const auto reserve = PackingReserve(batch, k, n);
    const auto scratch =
        static_cast<std::uint64_t>(descriptor.scratchpad_desc().get_size());
    bytes = Add(Add(bytes, reserve), scratch);
    Budget(bytes, max_io);
    Plan result{batch, descriptor, dnnl::matmul(descriptor)};
    result.logical_bytes = bytes;
    result.packing_bytes = reserve;
    result.scratch_bytes = scratch;
    const auto options = at::TensorOptions().device(at::kCPU);
    result.low = at::empty({batch, static_cast<std::int64_t>(k)},
                           options.dtype(at::kByte));
    result.high = at::empty_like(result.low);
    result.low_s32 = at::empty({batch, static_cast<std::int64_t>(n)},
                               options.dtype(at::kInt));
    result.high_s32 = at::empty_like(result.low_s32);
    result.low_i64 = at::empty({batch, static_cast<std::int64_t>(n)},
                               options.dtype(at::kLong));
    result.high_i64 = at::empty_like(result.low_i64);
    result.result_i64 = at::empty_like(result.low_i64);
    if (scratch != 0)
      result.scratchpad.emplace(descriptor.scratchpad_desc(), engine);
    return result;
  }

  void prepare(std::int64_t batch) {
    if (poisoned)
      throw j::NotReady("Composite helper failed; reload the runtime");
    require_sdk();
    if (plan && plan->batch == batch) return;
    try {
      plan.reset();  // Release the old shape before allocating the new one.
      plan.emplace(make_plan(batch));
    } catch (...) {
      poisoned = true;
      throw;
    }
  }

  void dot(Plan& selected, const at::Tensor& input, at::Tensor& output) {
    require_sdk();
    if (std::string_view(selected.descriptor.impl_info_str()) != "gemm:jit")
      throw j::BackendError("Composite component primitive identity changed");
    Memory src(selected.descriptor.src_desc(), engine,
               const_cast<std::uint8_t*>(input.const_data_ptr<std::uint8_t>()));
    Memory wei(selected.descriptor.weights_desc(), engine, weights.data());
    Memory dst(selected.descriptor.dst_desc(), engine,
               output.data_ptr<std::int32_t>());
    std::unordered_map<int, Memory> arguments{
        {DNNL_ARG_SRC, src}, {DNNL_ARG_WEIGHTS, wei}, {DNNL_ARG_DST, dst}};
    if (selected.scratchpad)
      arguments.emplace(DNNL_ARG_SCRATCHPAD, *selected.scratchpad);
    selected.operation.execute(stream, arguments);
    stream.wait();
    require_sdk();
  }

  const at::Tensor& execute(const at::Tensor& input) {
    if (!input.defined() || input.is_quantized() || !input.device().is_cpu() ||
        input.scalar_type() != at::kByte || input.dim() != 2 ||
        input.size(1) != static_cast<std::int64_t>(k) ||
        !input.is_contiguous() || input.layout() != at::kStrided ||
        input.is_conj() || input.is_neg() || input.requires_grad())
      throw j::InvalidArgument(
          "Composite helper requires the original contiguous CPU U8 bytes");
    prepare(input.size(0));
    try {
      auto& selected = *plan;
      at::bitwise_and_out(selected.low, input, at::Scalar(127));
      at::bitwise_right_shift_out(selected.high, input, at::Scalar(7));
      dot(selected, selected.low, selected.low_s32);
      dot(selected, selected.high, selected.high_s32);
      // Official copies convert each S32 component to INT64 before any
      // combination. Signed negative component values are never left-shifted.
      selected.low_i64.copy_(selected.low_s32);
      selected.high_i64.copy_(selected.high_s32);
      at::add_out(selected.result_i64, selected.low_i64, selected.high_i64,
                  at::Scalar(128));
      at::sub_out(selected.result_i64, selected.result_i64, compensation);
      require_sdk();
      return selected.result_i64;
    } catch (...) {
      poisoned = true;
      throw;
    }
  }

  const Plan& current_plan() const {
    if (poisoned || !plan)
      throw j::NotReady(
          "Composite workspace is unavailable; reload the runtime");
    return *plan;
  }

  std::vector<std::int8_t> weights;
  std::size_t k, n, max_batch, max_io;
  std::uint64_t external_fixed, external_row, max_weight_abs_sum = 0;
  at::Tensor compensation;
  dnnl::engine engine;
  dnnl::stream stream;
  std::optional<Plan> plan;
  bool poisoned = false;
};

IntegerMatmul::IntegerMatmul(std::vector<std::int8_t> weights,
                             std::size_t inputs, std::size_t outputs,
                             std::size_t max_batch, std::size_t max_io,
                             std::uint64_t fixed_bytes, std::uint64_t row_bytes)
    : impl_(std::make_unique<Impl>(std::move(weights), inputs, outputs,
                                   max_batch, max_io, fixed_bytes, row_bytes)) {
  impl_->prepare(1);
}
IntegerMatmul::~IntegerMatmul() = default;
void IntegerMatmul::prepare_batch(std::int64_t batch) { impl_->prepare(batch); }
const at::Tensor& IntegerMatmul::execute(const at::Tensor& input) {
  return impl_->execute(input);
}
const at::Tensor& IntegerMatmul::low_s32() const {
  return impl_->current_plan().low_s32;
}
const at::Tensor& IntegerMatmul::high_s32() const {
  return impl_->current_plan().high_s32;
}
std::string IntegerMatmul::selected_implementation() const {
  return impl_->current_plan().descriptor.impl_info_str();
}
std::uint64_t IntegerMatmul::logical_memory_bytes() const {
  return impl_->current_plan().logical_bytes;
}
std::uint64_t IntegerMatmul::packing_reserve_bytes() const {
  return impl_->current_plan().packing_bytes;
}
std::uint64_t IntegerMatmul::scratchpad_bytes() const {
  return impl_->current_plan().scratch_bytes;
}
std::uint64_t IntegerMatmul::maximum_weight_abs_sum() const {
  return impl_->max_weight_abs_sum;
}
}  // namespace jyppx::liuding::quantization_onednn::detail
