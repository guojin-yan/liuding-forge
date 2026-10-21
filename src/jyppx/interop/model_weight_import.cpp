// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/foundation/sha256.hpp>
#include <jyppx/interop/model_weight_import.hpp>
#include <jyppx/nn/model_adapter.hpp>

#include <torch/cuda.h>
#include <torch/types.h>

#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <bit>
#include <c10/core/InferenceMode.h>
#include <cstring>
#include <exception>
#include <iterator>
#include <limits>
#include <map>
#include <new>
#include <set>
#include <string>
#include <utility>

namespace jyppx::interop {
namespace {
struct Entry {
  WeightTensorDescriptor descriptor;
  torch::Tensor tensor;
  c10::Device device;
  const void* storage;
  const void* data;
  std::int64_t storage_offset;
  std::int64_t version;
  bool requires_grad;
};

SafeTensorDtype wire_dtype(c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::Bool:
      return SafeTensorDtype::boolean;
    case c10::ScalarType::Byte:
      return SafeTensorDtype::u8;
    case c10::ScalarType::Char:
      return SafeTensorDtype::i8;
    case c10::ScalarType::Short:
      return SafeTensorDtype::i16;
    case c10::ScalarType::Int:
      return SafeTensorDtype::i32;
    case c10::ScalarType::Long:
      return SafeTensorDtype::i64;
    case c10::ScalarType::Half:
      return SafeTensorDtype::f16;
    case c10::ScalarType::BFloat16:
      return SafeTensorDtype::bf16;
    case c10::ScalarType::Float:
      return SafeTensorDtype::f32;
    case c10::ScalarType::Double:
      return SafeTensorDtype::f64;
    default:
      throw CompatibilityError(
          "Model import does not support this destination dtype");
  }
}

c10::ScalarType torch_dtype(SafeTensorDtype dtype) {
  switch (dtype) {
    case SafeTensorDtype::boolean:
      return c10::ScalarType::Bool;
    case SafeTensorDtype::u8:
      return c10::ScalarType::Byte;
    case SafeTensorDtype::i8:
      return c10::ScalarType::Char;
    case SafeTensorDtype::i16:
      return c10::ScalarType::Short;
    case SafeTensorDtype::i32:
      return c10::ScalarType::Int;
    case SafeTensorDtype::i64:
      return c10::ScalarType::Long;
    case SafeTensorDtype::f16:
      return c10::ScalarType::Half;
    case SafeTensorDtype::bf16:
      return c10::ScalarType::BFloat16;
    case SafeTensorDtype::f32:
      return c10::ScalarType::Float;
    case SafeTensorDtype::f64:
      return c10::ScalarType::Double;
    default:
      throw CompatibilityError(
          "Model import cannot materialize this source dtype");
  }
}

bool floating(SafeTensorDtype dtype) {
  return dtype == SafeTensorDtype::f16 || dtype == SafeTensorDtype::bf16 ||
         dtype == SafeTensorDtype::f32 || dtype == SafeTensorDtype::f64;
}

std::vector<Entry> inventory(nn::ModelStateView& model,
                             const WeightImportLimits& limits) {
  std::vector<Entry> entries;
  std::set<std::string> names;
  std::set<const void*> tensor_impls;
  std::set<const void*> storages;
  using Intervals = std::map<std::uintptr_t, std::uintptr_t>;
  std::map<std::pair<c10::DeviceType, c10::DeviceIndex>, Intervals>
      device_intervals;
  const auto add = [&](const auto& values) {
    for (const auto& item : values) {
      if (entries.size() >= limits.max_tensors)
        throw ResourceExhausted("Model import target count exceeds limit");
      const auto& tensor = item.value();
      if (!tensor.defined() || tensor.layout() != torch::kStrided ||
          !tensor.is_contiguous() || tensor.is_quantized() ||
          (!tensor.is_cpu() && !tensor.is_cuda()) || tensor.is_inference() ||
          tensor.is_conj() || tensor.is_neg() || tensor.is_view() ||
          !tensor.is_leaf() || tensor.grad().defined()) {
        throw CompatibilityError(
            "Model import requires ordinary independent leaf CPU/CUDA state "
            "without views or gradients");
      }
      if (!names.insert(item.key()).second)
        throw CompatibilityError("Model import target keys are ambiguous");
      if (!tensor_impls.insert(tensor.unsafeGetTensorImpl()).second) {
        throw CompatibilityError(
            "Model import rejects repeated destination tensor handles");
      }
      if (tensor.numel() != 0 &&
          !storages.insert(tensor.storage().unsafeGetStorageImpl()).second) {
        throw CompatibilityError(
            "Model import rejects aliased destination state");
      }
      // Independent StorageImpl objects (for example, from_blob wrappers) can
      // still cover the same memory. Compare actual bytes on each device too.
      const auto* data = tensor.data_ptr();
      if (tensor.numel() != 0) {
        const auto begin = reinterpret_cast<std::uintptr_t>(data);
        const auto bytes = tensor.nbytes();
        if (data == nullptr ||
            bytes > (std::numeric_limits<std::uintptr_t>::max)() - begin) {
          throw CompatibilityError(
              "Model import destination has an invalid byte range");
        }
        const auto end = begin + static_cast<std::uintptr_t>(bytes);
        auto& intervals =
            device_intervals[{tensor.device().type(), tensor.device().index()}];
        const auto next = intervals.lower_bound(begin);
        if ((next != intervals.end() && next->first < end) ||
            (next != intervals.begin() && std::prev(next)->second > begin)) {
          throw CompatibilityError(
              "Model import rejects overlapping destination byte ranges");
        }
        intervals.emplace_hint(next, begin, end);
      }
      Entry entry{{.name = item.key(),
                   .dtype = wire_dtype(tensor.scalar_type()),
                   .shape = {}},
                  tensor,
                  tensor.device(),
                  tensor.storage().unsafeGetStorageImpl(),
                  data,
                  tensor.storage_offset(),
                  tensor._version(),
                  tensor.requires_grad()};
      if (static_cast<std::size_t>(tensor.dim()) > limits.max_rank)
        throw ResourceExhausted("Model import target rank exceeds limit");
      for (const auto size : tensor.sizes()) {
        if (size < 0)
          throw CompatibilityError(
              "Model import target has a negative dimension");
        entry.descriptor.shape.push_back(static_cast<std::uint64_t>(size));
      }
      entries.push_back(std::move(entry));
    }
  };
  add(model.named_parameters());
  add(model.named_buffers());
  return entries;
}

void synchronize(const std::vector<Entry>& entries) {
  std::set<c10::DeviceIndex> devices;
  for (const auto& entry : entries)
    if (entry.tensor.is_cuda()) devices.insert(entry.tensor.get_device());
  for (const auto device : devices) {
    try {
      torch::cuda::synchronize(device);
    } catch (const c10::OutOfMemoryError&) {
      throw;
    } catch (const c10::Error&) {
      std::throw_with_nested(DeviceError(
          "Model import device synchronization failed",
          ErrorDetails{.component = "interop.model_weight_import",
                       .operation = "synchronize",
                       .hint = "Inspect the nested backend error and CUDA "
                               "device state before retrying."}));
    }
  }
}

void charge(std::uint64_t& charged, std::uint64_t amount,
            std::uint64_t maximum) {
  if (amount > maximum - charged)
    throw ResourceExhausted("Model import exceeds staging byte limit");
  charged += amount;
}
}  // namespace

ModelWeightImportResult import_model_weights(
    nn::ModelStateView& model, std::span<const std::byte> artifact,
    const ArtifactDigest& expected_digest,
    const ModelWeightImportOptions& options, std::stop_token stop,
    TelemetryDeadline deadline) try {
  check_telemetry_operation(stop, deadline);
  validate_artifact_digest(expected_digest);
  if (c10::InferenceMode::is_enabled())
    throw CompatibilityError(
        "Model import cannot execute inside inference mode");
  if (options.max_staging_bytes == 0U)
    throw InvalidArgument("Model import staging limit must be positive");
  if (std::endian::native != std::endian::little)
    throw CompatibilityError("Model import requires a little-endian host");
  if (artifact.size() > options.tensor_limits.max_file_bytes)
    throw ResourceExhausted("Model import artifact exceeds byte limit");
  Sha256 hasher;
  for (std::size_t offset = 0U; offset < artifact.size();) {
    check_telemetry_operation(stop, deadline);
    const auto count =
        std::min<std::size_t>(artifact.size() - offset, 1024U * 1024U);
    hasher.update(artifact.subspan(offset, count));
    offset += count;
  }
  if (sha256_digest_hex(hasher.finish()) != expected_digest.hex)
    throw SerializationError("Model import source digest mismatch");
  check_telemetry_operation(stop, deadline);
  const SafeTensorsView source(artifact, options.tensor_limits);
  const auto targets = inventory(model, options.mapping_limits);
  std::vector<WeightTensorDescriptor> source_descriptors, target_descriptors;
  for (const auto& tensor : source.directory().tensors)
    source_descriptors.push_back({tensor.name, tensor.dtype, tensor.shape});
  for (const auto& entry : targets)
    target_descriptors.push_back(entry.descriptor);
  const auto plan = plan_weight_import(expected_digest, source_descriptors,
                                       target_descriptors, options.mapping,
                                       options.mapping_limits);
  ModelWeightImportResult result{.plan = plan.report(),
                                 .applied = false,
                                 .applied_tensors = 0U,
                                 .charged_staging_bytes = 0U};
  if (!result.plan.ready) return result;
  for (const auto& assignment : plan.assignments()) {
    const auto& descriptor = source_descriptors[assignment.source_index];
    const auto& target = targets[assignment.target_index];
    if (!assignment.conversion_rule.empty() &&
        (assignment.conversion_rule != "jyppx.float-cast.v1" ||
         !floating(descriptor.dtype) || !floating(target.descriptor.dtype))) {
      throw CompatibilityError(
          "Model import does not implement the requested conversion rule");
    }
    static_cast<void>(torch_dtype(descriptor.dtype));
    charge(result.charged_staging_bytes,
           source.tensor_bytes(descriptor.name).size(),
           options.max_staging_bytes);
    const auto target_bytes =
        static_cast<std::uint64_t>(target.tensor.nbytes());
    charge(result.charged_staging_bytes, target_bytes,
           options.max_staging_bytes);
    charge(result.charged_staging_bytes, target_bytes,
           options.max_staging_bytes);
  }
  std::vector<torch::Tensor> staged, original;
  staged.reserve(plan.assignments().size());
  original.reserve(plan.assignments().size());
  at::NoGradGuard guard;
  for (const auto& assignment : plan.assignments()) {
    check_telemetry_operation(stop, deadline);
    const auto& descriptor = source_descriptors[assignment.source_index];
    const auto& target = targets[assignment.target_index].tensor;
    std::vector<std::int64_t> shape;
    for (const auto dimension : descriptor.shape) {
      if (dimension > static_cast<std::uint64_t>(
                          (std::numeric_limits<std::int64_t>::max)()))
        throw ResourceExhausted(
            "Model import dimension exceeds LibTorch range");
      shape.push_back(static_cast<std::int64_t>(dimension));
    }
    auto decoded = torch::empty(shape, torch::TensorOptions()
                                           .dtype(torch_dtype(descriptor.dtype))
                                           .device(torch::kCPU));
    const auto raw = source.tensor_bytes(descriptor.name);
    if (descriptor.dtype == SafeTensorDtype::boolean &&
        std::any_of(raw.begin(), raw.end(), [](std::byte value) {
          return value != std::byte{0} && value != std::byte{1};
        })) {
      throw SerializationError(
          "Model import BOOL payload must use zero or one bytes");
    }
    if (!raw.empty())
      std::memcpy(decoded.mutable_data_ptr(), raw.data(), raw.size());
    staged.push_back(
        decoded.to(target.device(), target.scalar_type(), false, true));
    original.push_back(target.detach());
  }
  synchronize(targets);
  const auto current = inventory(model, options.mapping_limits);
  if (current.size() != targets.size())
    throw CompatibilityError("Model topology changed while staging import");
  for (std::size_t i = 0U; i < targets.size(); ++i) {
    if (current[i].descriptor.name != targets[i].descriptor.name ||
        current[i].descriptor.dtype != targets[i].descriptor.dtype ||
        current[i].descriptor.shape != targets[i].descriptor.shape ||
        current[i].tensor.unsafeGetTensorImpl() !=
            targets[i].tensor.unsafeGetTensorImpl() ||
        current[i].device != targets[i].device ||
        current[i].storage != targets[i].storage ||
        current[i].data != targets[i].data ||
        current[i].storage_offset != targets[i].storage_offset ||
        current[i].version != targets[i].version ||
        current[i].requires_grad != targets[i].requires_grad) {
      throw CompatibilityError("Model topology changed while staging import");
    }
  }
  check_telemetry_operation(stop, deadline);
  std::size_t attempted = 0U;
  try {
    for (std::size_t i = 0U; i < plan.assignments().size(); ++i) {
      attempted = i + 1U;
      auto tensor = targets[plan.assignments()[i].target_index].tensor;
      tensor.set_data(staged[i]);
    }
  } catch (...) {
    const auto failure = std::current_exception();
    std::size_t restore_failures = 0U;
    // Include the failing assignment: a backend can throw after changing its
    // data handle. A failed restore must not prevent the remaining attempts.
    for (std::size_t remaining = attempted; remaining > 0U; --remaining) {
      const auto i = remaining - 1U;
      try {
        auto tensor = targets[plan.assignments()[i].target_index].tensor;
        tensor.set_data(original[i]);
      } catch (...) {
        ++restore_failures;
      }
    }
    if (restore_failures != 0U) {
      std::throw_with_nested(InternalError(
          "Model import commit failed and rollback was incomplete",
          ErrorDetails{
              .component = "interop.model_weight_import",
              .operation = "rollback",
              .expected = "every attempted data handle restored",
              .actual = "restore_failures=" + std::to_string(restore_failures) +
                        "; attempted=" + std::to_string(attempted),
              .hint = "Discard this model instance and reconstruct it before "
                      "further use. The nested cause is the original commit "
                      "failure."}));
    }
    std::rethrow_exception(failure);
  }
  result.applied = true;
  result.applied_tensors = plan.assignments().size();
  return result;
} catch (const c10::OutOfMemoryError&) {
  std::throw_with_nested(ResourceExhausted(
      "Model import exhausted backend memory",
      ErrorDetails{.component = "interop.model_weight_import",
                   .operation = "import_model_weights",
                   .hint = "Reduce the model or staging requirements and "
                           "inspect the nested backend error."}));
} catch (const c10::Error&) {
  std::throw_with_nested(BackendError(
      "Model import backend operation failed",
      ErrorDetails{
          .component = "interop.model_weight_import",
          .operation = "import_model_weights",
          .hint =
              "Inspect the nested backend error before retrying the import."}));
} catch (const std::bad_alloc&) {
  std::throw_with_nested(ResourceExhausted(
      "Model import exhausted host memory",
      ErrorDetails{.component = "interop.model_weight_import",
                   .operation = "import_model_weights",
                   .hint = "Reduce the artifact, model or staging requirements "
                           "before retrying."}));
}
}  // namespace jyppx::interop
