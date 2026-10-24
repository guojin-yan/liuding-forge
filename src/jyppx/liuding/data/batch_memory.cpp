// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data/batch_memory.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <ATen/core/Dict.h>
#include <ATen/core/List.h>
#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace jyppx::liuding::data {
namespace {

constexpr char kComponent[] = "data.batch_memory";
constexpr std::size_t kMaxNestingDepth = 64;

struct Inspection {
  std::size_t tensor_count = 0;
  std::size_t nonempty_tensor_count = 0;
  std::size_t value_node_count = 0;
  std::uint64_t tensor_bytes = 0;
};

[[noreturn]] void throw_unsupported(std::string path, std::string actual,
                                    std::string hint) {
  throw InvalidArgument(
      "Batch memory stage received an unsupported value",
      ErrorDetails{.component = kComponent,
                   .operation = std::move(path),
                   .expected = "tensor, tuple, list, scalar-key dictionary, "
                               "scalar, string, device, or None",
                   .actual = std::move(actual),
                   .hint = std::move(hint)});
}

void count_node(Inspection& inspection, const BatchMemoryOptions& options,
                std::string_view path) {
  if (inspection.value_node_count == options.max_value_nodes) {
    throw ResourceExhausted(
        "Batch memory value-node limit was exceeded",
        ErrorDetails{.component = kComponent,
                     .operation = std::string(path),
                     .expected = "at most " +
                                 std::to_string(options.max_value_nodes) +
                                 " values",
                     .actual = "additional value",
                     .hint = "Reduce batch container size or raise the "
                             "explicit max_value_nodes limit."});
  }
  ++inspection.value_node_count;
}

bool is_passthrough_leaf(const c10::IValue& value) {
  return value.isNone() || value.isBool() || value.isInt() ||
         value.isDouble() || value.isString() || value.isScalar() ||
         value.isDevice();
}

void inspect_value(const c10::IValue& value, const BatchMemoryOptions& options,
                   Inspection& inspection, std::size_t depth,
                   const std::string& path);

void inspect_dictionary_key(const c10::IValue& key,
                            const BatchMemoryOptions& options,
                            Inspection& inspection, const std::string& path) {
  count_node(inspection, options, path);
  if (!is_passthrough_leaf(key) || key.isNone()) {
    throw_unsupported(path, key.tagKind(),
                      "Use immutable scalar dictionary keys; tensors and "
                      "containers are not valid batch-memory keys.");
  }
}

void inspect_tensor(const Tensor& tensor, const BatchMemoryOptions& options,
                    Inspection& inspection, const std::string& path) {
  if (!tensor.defined()) {
    throw InvalidArgument(
        "Batch memory stage requires defined tensors",
        ErrorDetails{.component = kComponent,
                     .operation = path,
                     .expected = "a defined strided tensor",
                     .actual = "undefined tensor",
                     .hint = "Remove or initialize the tensor leaf."});
  }
  if (tensor.layout() != c10::kStrided) {
    throw InvalidArgument(
        "Batch memory stage does not support this tensor layout",
        ErrorDetails{.component = kComponent,
                     .operation = path,
                     .expected = "strided",
                     .actual = c10::toString(tensor.layout()),
                     .hint = "Densify the batch tensor before staging."});
  }
  if (inspection.tensor_count == options.max_tensors) {
    throw ResourceExhausted(
        "Batch memory tensor-count limit was exceeded",
        ErrorDetails{.component = kComponent,
                     .operation = path,
                     .expected = "at most " +
                                 std::to_string(options.max_tensors) +
                                 " tensor occurrences",
                     .actual = "additional tensor",
                     .hint = "Reduce batch structure or raise the explicit "
                             "max_tensors limit."});
  }
  ++inspection.tensor_count;
  const auto numel = tensor.numel();
  if (numel > 0) ++inspection.nonempty_tensor_count;
  const auto element_size = tensor.element_size();
  if (numel < 0 || static_cast<std::uint64_t>(numel) >
                       std::numeric_limits<std::uint64_t>::max() /
                           std::max<std::size_t>(element_size, 1U)) {
    throw ResourceExhausted("Batch tensor byte count overflowed");
  }
  const auto bytes = static_cast<std::uint64_t>(numel) *
                     static_cast<std::uint64_t>(element_size);
  if (bytes > options.max_bytes - inspection.tensor_bytes) {
    throw ResourceExhausted(
        "Batch memory byte limit was exceeded",
        ErrorDetails{
            .component = kComponent,
            .operation = path,
            .expected =
                "at most " + std::to_string(options.max_bytes) + " bytes",
            .actual = "current=" + std::to_string(inspection.tensor_bytes) +
                      ", next=" + std::to_string(bytes),
            .hint = "Reduce batch size or raise the explicit "
                    "max_bytes limit."});
  }
  inspection.tensor_bytes += bytes;
}

void inspect_value(const c10::IValue& value, const BatchMemoryOptions& options,
                   Inspection& inspection, std::size_t depth,
                   const std::string& path) {
  if (depth > kMaxNestingDepth) {
    throw ResourceExhausted(
        "Batch memory nesting limit was exceeded",
        ErrorDetails{.component = kComponent,
                     .operation = path,
                     .expected = "nesting depth at most " +
                                 std::to_string(kMaxNestingDepth),
                     .actual = std::to_string(depth),
                     .hint = "Flatten the batch container structure."});
  }
  count_node(inspection, options, path);
  if (value.isTensor()) {
    inspect_tensor(value.toTensor(), options, inspection, path);
    return;
  }
  if (value.isTuple()) {
    const auto& elements = value.toTupleRef().elements();
    for (std::size_t index = 0; index < elements.size(); ++index) {
      inspect_value(elements[index], options, inspection, depth + 1,
                    path + "[" + std::to_string(index) + "]");
    }
    return;
  }
  if (value.isList()) {
    const auto list = value.toListRef();
    for (std::size_t index = 0; index < list.size(); ++index) {
      inspect_value(list[index], options, inspection, depth + 1,
                    path + "[" + std::to_string(index) + "]");
    }
    return;
  }
  if (value.isGenericDict()) {
    const auto dictionary = value.toGenericDict();
    std::size_t index = 0;
    for (const auto& entry : dictionary) {
      inspect_dictionary_key(entry.key(), options, inspection,
                             path + ".key[" + std::to_string(index) + "]");
      inspect_value(entry.value(), options, inspection, depth + 1,
                    path + ".value[" + std::to_string(index) + "]");
      ++index;
    }
    return;
  }
  if (!is_passthrough_leaf(value)) {
    throw_unsupported(path, value.tagKind(),
                      "Convert opaque objects to supported IValue containers "
                      "before enabling batch memory staging.");
  }
}

struct TransformContext {
  const BatchMemoryOptions& options;
  BatchMemoryMetadata metadata;
};

enum class TransformPhase : std::uint8_t {
  prepare = 0,
  transfer = 1,
};

Tensor transform_tensor(const Tensor& tensor, TransformContext& context,
                        TransformPhase phase, const std::string& path) {
  try {
    Tensor staged = tensor;
    if (phase == TransformPhase::prepare && context.options.target_device &&
        !staged.device().is_cpu()) {
      throw InvalidArgument(
          "Two-phase batch transfer requires CPU source tensors",
          ErrorDetails{.component = kComponent,
                       .operation = path,
                       .expected = "CPU source tensor",
                       .actual = staged.device().str(),
                       .hint = "Return CPU data from background factories; "
                               "the next() consumer owns device transfer."});
    }
    if (phase == TransformPhase::prepare && context.options.pin_memory) {
      if (!staged.device().is_cpu()) {
        throw InvalidArgument(
            "Batch host pinning requires CPU source tensors",
            ErrorDetails{.component = kComponent,
                         .operation = path,
                         .expected = "CPU source tensor",
                         .actual = staged.device().str(),
                         .hint = "Disable pin_memory for device-resident "
                                 "producer values."});
      }
      // Empty tensors have no bytes to pin; LibTorch may report false even
      // after pin_memory(). Keep their shape/device contract without requiring
      // a nonexistent storage allocation in the transfer phase.
      if (staged.numel() != 0) {
        if (!staged.is_pinned()) staged = staged.pin_memory();
        ++context.metadata.pinned_tensor_count;
      }
    }
    if (phase == TransformPhase::transfer && context.options.pin_memory &&
        staged.device().is_cpu() && staged.numel() != 0 &&
        !staged.is_pinned()) {
      throw CompatibilityError(
          "Prepared batch tensor is not pinned",
          ErrorDetails{.component = kComponent,
                       .operation = path,
                       .expected = "a pinned CPU tensor",
                       .actual = "unpinned CPU tensor",
                       .hint = "Pass the unmodified result of "
                               "prepare_batch_memory to the transfer phase."});
    }
    if (phase == TransformPhase::transfer && context.options.target_device &&
        staged.device() != *context.options.target_device) {
      staged = staged.to(*context.options.target_device, staged.scalar_type(),
                         context.options.non_blocking, false);
      ++context.metadata.transferred_tensor_count;
    }
    return staged;
  } catch (const Error&) {
    throw;
  } catch (const c10::Error& cause) {
    detail::rethrow_backend_error(
        cause, {.component = kComponent,
                .operation = path,
                .failure_kind = detail::BackendFailureKind::device,
                .expected = context.options.target_device
                                ? context.options.target_device->str()
                                : "pinned CPU memory",
                .hint = "Verify CUDA availability, pinned-memory support, "
                        "and device capacity."});
  }
}

c10::IValue transform_value(const c10::IValue& value, TransformContext& context,
                            TransformPhase phase, const std::string& path) {
  if (value.isTensor()) {
    return c10::IValue(
        transform_tensor(value.toTensor(), context, phase, path));
  }
  if (value.isTuple()) {
    const auto tuple = value.toTuple();
    const auto& elements = tuple->elements();
    std::vector<c10::IValue> transformed;
    transformed.reserve(elements.size());
    for (std::size_t index = 0; index < elements.size(); ++index) {
      transformed.push_back(
          transform_value(elements[index], context, phase,
                          path + "[" + std::to_string(index) + "]"));
    }
    return c10::IValue(
        c10::ivalue::Tuple::createNamed(std::move(transformed), tuple->type()));
  }
  if (value.isList()) {
    const auto list = value.toList();
    c10::impl::GenericList transformed(list.elementType());
    transformed.reserve(list.size());
    for (std::size_t index = 0; index < list.size(); ++index) {
      transformed.push_back(
          transform_value(list.get(index), context, phase,
                          path + "[" + std::to_string(index) + "]"));
    }
    return c10::IValue(transformed);
  }
  if (value.isGenericDict()) {
    const auto dictionary = value.toGenericDict();
    c10::impl::GenericDict transformed(dictionary.keyType(),
                                       dictionary.valueType());
    std::size_t index = 0;
    for (const auto& entry : dictionary) {
      transformed.insert(
          entry.key(),
          transform_value(entry.value(), context, phase,
                          path + ".value[" + std::to_string(index) + "]"));
      ++index;
    }
    return c10::IValue(transformed);
  }
  return value;
}

BatchMemoryMetadata make_metadata(const BatchMemoryOptions& options,
                                  const Inspection& inspection) {
  return {.staged = true,
          .pin_memory_requested = options.pin_memory,
          .non_blocking_requested = options.non_blocking,
          .target_device = options.target_device,
          .transfer_stream_policy = BatchTransferStreamPolicy::caller_current,
          .tensor_count = inspection.tensor_count,
          .value_node_count = inspection.value_node_count,
          .tensor_bytes = inspection.tensor_bytes};
}

void validate_prepared_metadata(const BatchMemoryStageResult& prepared,
                                const BatchMemoryOptions& options,
                                const Inspection& inspection) {
  const auto& metadata = prepared.metadata;
  if (!metadata.staged || metadata.pin_memory_requested != options.pin_memory ||
      metadata.non_blocking_requested != options.non_blocking ||
      metadata.target_device != options.target_device ||
      metadata.transfer_stream_policy !=
          BatchTransferStreamPolicy::caller_current ||
      metadata.tensor_count != inspection.tensor_count ||
      metadata.value_node_count != inspection.value_node_count ||
      metadata.tensor_bytes != inspection.tensor_bytes ||
      metadata.transferred_tensor_count != 0U ||
      metadata.pinned_tensor_count !=
          (options.pin_memory ? inspection.nonempty_tensor_count : 0U)) {
    throw CompatibilityError(
        "Prepared batch metadata is incompatible with the transfer policy",
        ErrorDetails{.component = kComponent,
                     .operation = "transfer.metadata",
                     .expected = "metadata emitted by prepare_batch_memory",
                     .actual = "policy or resource summary mismatch",
                     .hint = "Do not modify a prepared value or its metadata "
                             "before device transfer."});
  }
}

}  // namespace

bool batch_memory_stage_enabled(const BatchMemoryOptions& options) noexcept {
  return options.pin_memory || options.target_device.has_value();
}

void validate_batch_memory_options(const BatchMemoryOptions& options) {
  if (options.stage_capacity == 0) {
    throw InvalidArgument("Batch memory stage_capacity must be positive");
  }
  if (options.max_tensors == 0) {
    throw InvalidArgument("Batch memory max_tensors must be positive");
  }
  if (options.max_value_nodes == 0) {
    throw InvalidArgument("Batch memory max_value_nodes must be positive");
  }
  if (options.max_bytes == 0) {
    throw InvalidArgument("Batch memory max_bytes must be positive");
  }
  if (options.non_blocking && !options.target_device) {
    throw InvalidArgument("Batch memory non_blocking requires a target_device");
  }
}

BatchMemoryStageResult prepare_batch_memory(const c10::IValue& value,
                                            const BatchMemoryOptions& options) {
  validate_batch_memory_options(options);
  BatchMemoryStageResult result;
  result.value = value;
  result.metadata.pin_memory_requested = options.pin_memory;
  result.metadata.non_blocking_requested = options.non_blocking;
  result.metadata.target_device = options.target_device;
  result.metadata.transfer_stream_policy =
      BatchTransferStreamPolicy::caller_current;
  if (!batch_memory_stage_enabled(options)) {
    return result;
  }

  Inspection inspection;
  inspect_value(value, options, inspection, 0, "batch");
  TransformContext context{.options = options,
                           .metadata = make_metadata(options, inspection)};
  result.value =
      transform_value(value, context, TransformPhase::prepare, "batch");
  result.metadata = context.metadata;
  return result;
}

BatchMemoryStageResult transfer_batch_memory(
    BatchMemoryStageResult prepared, const BatchMemoryOptions& options) {
  validate_batch_memory_options(options);
  if (!batch_memory_stage_enabled(options)) {
    if (prepared.metadata.staged || prepared.metadata.pin_memory_requested ||
        prepared.metadata.non_blocking_requested ||
        prepared.metadata.target_device) {
      throw CompatibilityError(
          "Prepared batch metadata enables a disabled memory stage");
    }
    return prepared;
  }

  Inspection inspection;
  inspect_value(prepared.value, options, inspection, 0, "batch");
  validate_prepared_metadata(prepared, options, inspection);
  TransformContext context{.options = options, .metadata = prepared.metadata};
  prepared.value = transform_value(prepared.value, context,
                                   TransformPhase::transfer, "batch");
  prepared.metadata = context.metadata;
  return prepared;
}

BatchMemoryStageResult stage_batch_memory(const c10::IValue& value,
                                          const BatchMemoryOptions& options) {
  return transfer_batch_memory(prepare_batch_memory(value, options), options);
}

BatchMemoryUsage inspect_batch_memory(const c10::IValue& value,
                                      const BatchMemoryOptions& limits) {
  validate_batch_memory_options(limits);
  Inspection inspection;
  inspect_value(value, limits, inspection, 0, "batch");
  return {.tensor_count = inspection.tensor_count,
          .value_node_count = inspection.value_node_count,
          .tensor_bytes = inspection.tensor_bytes};
}

}  // namespace jyppx::liuding::data
