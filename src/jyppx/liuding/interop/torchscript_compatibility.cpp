// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/interop/torchscript_compatibility.hpp>

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/cuda.h>
#include <torch/version.h>

#include <ATen/Context.h>
#include <ATen/autocast_mode.h>
#include <ATen/core/Dict.h>
#include <ATen/core/List.h>
#include <ATen/core/jit_type.h>
#include <c10/core/InferenceMode.h>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <sstream>
#include <utility>

namespace jyppx::liuding::interop {
namespace {

constexpr char kComponent[] = "interop.torchscript_compatibility";
using Deadline = std::optional<std::chrono::steady_clock::time_point>;

ErrorDetails Details(std::string operation, std::string expected) {
  return {.component = kComponent,
          .operation = std::move(operation),
          .expected = std::move(expected),
          .actual = {},
          .hint =
              "Use an explicitly trusted, supported compatibility artifact; "
              "use process isolation for hard resource/deadline limits."};
}

[[noreturn]] void Invalid(std::string operation, std::string expected) {
  throw InvalidArgument("Invalid TorchScript runtime boundary",
                        Details(std::move(operation), std::move(expected)));
}

[[noreturn]] void Unsupported(std::string operation, std::string expected) {
  throw CompatibilityError("Unsupported TorchScript runtime boundary",
                           Details(std::move(operation), std::move(expected)));
}

[[noreturn]] void Resource(std::string operation) {
  throw ResourceExhausted(
      "TorchScript boundary resource limit exceeded",
      Details(std::move(operation), "within declared limit"));
}

void Add(std::uint64_t& total, std::uint64_t value, std::uint64_t limit,
         const char* operation) {
  if (value > limit || total > limit - value) Resource(operation);
  total += value;
}

void Boundary(std::stop_token stop, const Deadline& deadline) {
  if (stop.stop_requested())
    throw Cancelled("TorchScript operation cancelled at a cooperative boundary",
                    Details("cancel", "no cancellation requested"));
  if (deadline && std::chrono::steady_clock::now() >= *deadline)
    throw DeadlineExceeded("TorchScript cooperative boundary deadline elapsed",
                           Details("deadline", "a future absolute deadline"));
}

template <typename Function>
decltype(auto) BackendBoundary(const char* operation, Function&& function) {
  try {
    return std::forward<Function>(function)();
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    std::throw_with_nested(
        ResourceExhausted("TorchScript allocation failed",
                          Details(operation, "available memory")));
  } catch (const std::length_error&) {
    std::throw_with_nested(
        ResourceExhausted("TorchScript allocation size failed",
                          Details(operation, "bounded sizes")));
  } catch (const c10::OutOfMemoryError&) {
    std::throw_with_nested(
        ResourceExhausted("TorchScript backend allocation failed",
                          Details(operation, "available backend memory")));
  } catch (const c10::Error&) {
    std::throw_with_nested(BackendError(
        "TorchScript backend operation failed",
        Details(operation, "a supported executable artifact and runtime")));
  } catch (const std::exception&) {
    std::throw_with_nested(BackendError(
        "TorchScript backend operation failed",
        Details(operation, "a supported executable artifact and runtime")));
  }
}

void ValidateLimits(const TorchScriptLimits& limits) {
  if (limits.max_artifact_bytes == 0U ||
      limits.max_artifact_bytes > std::numeric_limits<std::size_t>::max() ||
      limits.max_state_tensor_bytes == 0U ||
      limits.max_input_tensor_bytes == 0U ||
      limits.max_output_tensor_bytes == 0U || limits.max_arguments == 0U ||
      limits.max_arguments > 1024U || limits.max_state_tensors == 0U ||
      limits.max_tensor_rank == 0U || limits.max_tensor_rank > 64U ||
      limits.max_string_bytes == 0U || limits.value_limits.max_depth == 0U ||
      limits.value_limits.max_depth > 64U ||
      limits.value_limits.max_schema_nodes == 0U ||
      limits.value_limits.max_metadata_bytes == 0U ||
      limits.value_limits.max_runtime_nodes == 0U)
    Invalid("limits", "positive bounded argument/rank/depth/byte/node limits");
}

void ValidateDevice(const Device& device) {
  if (device.is_cpu()) {
    if (device.has_index())
      Invalid("device", "cpu without an index, or explicit cuda:N");
    return;
  }
  if (!device.is_cuda() || !device.has_index())
    Unsupported("device", "cpu without an index, or explicit cuda:N");
  const auto index = device.index();
  if (index < 0 || !torch::cuda::is_available() ||
      index >= torch::cuda::device_count())
    throw DeviceError("Requested TorchScript CUDA device is unavailable",
                      Details("device", "an actually available cuda:N device"));
}

void CheckPrecisionContext(const Device& device) {
  if (at::autocast::is_autocast_enabled(device.type()))
    Unsupported("autocast", "autocast disabled on the runtime device");
}

bool SupportedDtype(at::ScalarType dtype) {
  switch (dtype) {
    case at::kBool:
    case at::kByte:
    case at::kChar:
    case at::kShort:
    case at::kInt:
    case at::kLong:
    case at::kHalf:
    case at::kBFloat16:
    case at::kFloat:
    case at::kDouble:
      return true;
    default:
      return false;
  }
}

std::uint64_t TensorBytes(const at::Tensor& tensor, const Device& device,
                          const TorchScriptLimits& limits,
                          bool require_contiguous) {
  if (!tensor.defined() || tensor.layout() != at::kStrided ||
      tensor.is_quantized() || tensor.is_conj() || tensor.is_neg() ||
      !SupportedDtype(tensor.scalar_type()) || tensor.device() != device ||
      (require_contiguous && !tensor.is_contiguous()))
    Unsupported("tensor",
                "defined dense supported-dtype tensors on the exact "
                "device; inputs must be contiguous without conj/neg");
  if (tensor.dim() < 0 ||
      static_cast<std::uint64_t>(tensor.dim()) > limits.max_tensor_rank)
    Resource("tensor.rank");
  const auto elements = tensor.numel();
  const auto element_size = tensor.element_size();
  if (elements < 0 || element_size <= 0) Resource("tensor.bytes");
  const auto width = static_cast<std::uint64_t>(element_size);
  if (static_cast<std::uint64_t>(elements) >
      std::numeric_limits<std::uint64_t>::max() / width)
    Resource("tensor.bytes");
  return static_cast<std::uint64_t>(elements) * width;
}

struct ValueBudget {
  const TorchScriptLimits& limits;
  const Device& device;
  std::uint64_t tensor_limit;
  bool require_contiguous;
  std::uint64_t tensor_bytes = 0U;
  std::uint64_t nodes = 0U;
  std::uint64_t strings = 0U;
};

void CheckValue(const c10::IValue& value, ValueBudget& budget,
                std::size_t depth = 0U) {
  if (depth > budget.limits.value_limits.max_depth) Resource("value.depth");
  Add(budget.nodes, 1U, budget.limits.value_limits.max_runtime_nodes,
      "value.nodes");
  if (value.isTensor()) {
    Add(budget.tensor_bytes,
        TensorBytes(value.toTensor(), budget.device, budget.limits,
                    budget.require_contiguous),
        budget.tensor_limit, "value.tensor_bytes");
  } else if (value.isString()) {
    Add(budget.strings, value.toStringRef().size(),
        budget.limits.max_string_bytes, "value.string_bytes");
  } else if (value.isTuple()) {
    if (value.toTupleRef().type()->name())
      Unsupported("value.tuple", "unnamed tuple values");
    for (const auto& item : value.toTupleRef().elements())
      CheckValue(item, budget, depth + 1U);
  } else if (value.isList()) {
    for (const auto& item : value.toListRef())
      CheckValue(item, budget, depth + 1U);
  } else if (value.isGenericDict()) {
    for (const auto& item : value.toGenericDict()) {
      if (!item.key().isString())
        Unsupported("value.dictionary", "string dictionary keys");
      CheckValue(item.key(), budget, depth + 1U);
      CheckValue(item.value(), budget, depth + 1U);
    }
  } else if (!value.isNone() && !value.isInt() && !value.isDouble() &&
             !value.isBool()) {
    Unsupported("value.kind", "supported tensor/scalar/container values");
  }
}

c10::IValue Snapshot(const c10::IValue& value) {
  if (value.isTensor())
    return value.toTensor().clone(at::MemoryFormat::Contiguous);
  if (value.isTuple()) {
    std::vector<c10::IValue> items;
    items.reserve(value.toTupleRef().elements().size());
    for (const auto& item : value.toTupleRef().elements())
      items.push_back(Snapshot(item));
    return c10::ivalue::Tuple::create(std::move(items));
  }
  if (value.isList()) {
    const auto original = value.toList();
    c10::impl::GenericList items(original.elementType());
    items.reserve(original.size());
    for (const auto& item : original) items.push_back(Snapshot(item));
    return items;
  }
  if (value.isGenericDict()) {
    const auto original = value.toGenericDict();
    c10::impl::GenericDict items(original.keyType(), original.valueType());
    for (const auto& item : original)
      items.insert(item.key(), Snapshot(item.value()));
    return items;
  }
  if (value.isString()) return std::string(value.toStringRef());
  return value;
}

struct TypeBudget {
  const TorchScriptLimits& limits;
  std::uint64_t nodes = 0U;
  std::uint64_t metadata = 0U;
};

void CheckSupportedType(const c10::TypePtr& type, TypeBudget& budget,
                        std::size_t depth = 0U) {
  if (depth > budget.limits.value_limits.max_depth) Resource("type.depth");
  Add(budget.nodes, 1U, budget.limits.value_limits.max_schema_nodes,
      "type.nodes");
  if (const auto optional = type->cast<c10::OptionalType>()) {
    CheckSupportedType(optional->getElementType(), budget, depth + 1U);
  } else if (const auto tuple = type->cast<c10::TupleType>()) {
    if (tuple->name()) Unsupported("type.tuple", "unnamed fixed tuple types");
    for (const auto& item : tuple->elements())
      CheckSupportedType(item, budget, depth + 1U);
  } else if (const auto list = type->cast<c10::ListType>()) {
    CheckSupportedType(list->getElementType(), budget, depth + 1U);
  } else if (const auto dictionary = type->cast<c10::DictType>()) {
    if (dictionary->getKeyType()->kind() != c10::TypeKind::StringType)
      Unsupported("type.dictionary", "Dict[str, SupportedType]");
    CheckSupportedType(dictionary->getValueType(), budget, depth + 1U);
  } else {
    switch (type->kind()) {
      case c10::TypeKind::TensorType:
      case c10::TypeKind::IntType:
      case c10::TypeKind::FloatType:
      case c10::TypeKind::BoolType:
      case c10::TypeKind::StringType:
      case c10::TypeKind::NoneType:
        return;
      default:
        Unsupported("type", "supported exact JIT types; no Any/classes/device");
    }
  }
}

void CheckBinding(const c10::TypePtr& declared, const ValueSchema& schema) {
  auto type = declared;
  const auto optional = type->cast<c10::OptionalType>();
  if (static_cast<bool>(optional) != schema.nullable &&
      type->kind() != c10::TypeKind::NoneType)
    Unsupported("schema.nullable", "nullable exactly for an Optional JIT type");
  if (optional) type = optional->getElementType();
  switch (type->kind()) {
    case c10::TypeKind::TensorType:
      if (schema.kind == ValueKind::tensor) return;
      break;
    case c10::TypeKind::IntType:
      if (schema.kind == ValueKind::integer) return;
      break;
    case c10::TypeKind::FloatType:
      if (schema.kind == ValueKind::floating) return;
      break;
    case c10::TypeKind::BoolType:
      if (schema.kind == ValueKind::boolean) return;
      break;
    case c10::TypeKind::StringType:
      if (schema.kind == ValueKind::string) return;
      break;
    case c10::TypeKind::NoneType:
      if (schema.kind == ValueKind::none) return;
      break;
    case c10::TypeKind::TupleType: {
      const auto tuple = type->cast<c10::TupleType>();
      if (schema.kind != ValueKind::tuple ||
          schema.elements.size() != tuple->elements().size())
        break;
      for (std::size_t i = 0; i < schema.elements.size(); ++i)
        CheckBinding(tuple->elements()[i], schema.elements[i]);
      return;
    }
    case c10::TypeKind::ListType:
      if (schema.kind == ValueKind::list && schema.elements.size() == 1U) {
        CheckBinding(type->cast<c10::ListType>()->getElementType(),
                     schema.elements.front());
        return;
      }
      break;
    case c10::TypeKind::DictType:
      if (schema.kind == ValueKind::dictionary && !schema.allow_extra_fields) {
        for (const auto& [name, field] : schema.fields) {
          static_cast<void>(name);
          CheckBinding(type->cast<c10::DictType>()->getValueType(), field);
        }
        return;
      }
      break;
    default:
      break;
  }
  Unsupported("schema.type", "exact supported JIT and ValueSchema categories");
}

void CheckTensorConstraints(const ValueSchema& schema,
                            const TorchScriptLoadOptions& options) {
  if (schema.tensor_constraints) {
    const auto& tensor = *schema.tensor_constraints;
    if ((tensor.dtype && !SupportedDtype(*tensor.dtype)) ||
        (tensor.device && *tensor.device != options.device))
      Unsupported("schema.tensor",
                  "a supported dtype and exact runtime device");
    if ((tensor.rank && static_cast<std::uint64_t>(*tensor.rank) >
                            options.limits.max_tensor_rank) ||
        (tensor.shape && tensor.shape->size() > options.limits.max_tensor_rank))
      Resource("schema.tensor.rank");
  }
  for (const auto& item : schema.elements)
    CheckTensorConstraints(item, options);
  for (const auto& [name, item] : schema.fields) {
    static_cast<void>(name);
    CheckTensorConstraints(item, options);
  }
}

std::string TypeText(const c10::TypePtr& type, TypeBudget& budget) {
  auto text = type->str();
  Add(budget.metadata, text.size(),
      budget.limits.value_limits.max_metadata_bytes, "type.metadata");
  return text;
}

void Synchronize(const Device& device) {
  if (device.is_cuda()) torch::cuda::synchronize(device.index());
}

std::unique_lock<std::timed_mutex> Lock(std::timed_mutex& mutex,
                                        std::stop_token stop,
                                        const Deadline& deadline) {
  std::unique_lock<std::timed_mutex> lock(mutex, std::defer_lock);
  do {
    Boundary(stop, deadline);
  } while (!lock.try_lock_for(std::chrono::milliseconds(1)));
  Boundary(stop, deadline);
  return lock;
}

}  // namespace

class TorchScriptCompatibilityRuntime::Impl {
 public:
  mutable std::timed_mutex mutex;
  std::optional<torch::jit::Module> module;
  nn::InvocationSchema input_schema;
  ValueSchema output_schema;
  TorchScriptLoadOptions options;
  std::map<std::string, c10::IValue> defaults;
  TorchScriptRuntimeInfo info;

  void Close() {
    module.reset();
    defaults.clear();
    info.open = false;
  }
};

TorchScriptCompatibilityRuntime::TorchScriptCompatibilityRuntime(
    std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
TorchScriptCompatibilityRuntime::~TorchScriptCompatibilityRuntime() = default;
TorchScriptCompatibilityRuntime::TorchScriptCompatibilityRuntime(
    TorchScriptCompatibilityRuntime&&) noexcept = default;
TorchScriptCompatibilityRuntime& TorchScriptCompatibilityRuntime::operator=(
    TorchScriptCompatibilityRuntime&&) noexcept = default;

TorchScriptCompatibilityRuntime TorchScriptCompatibilityRuntime::load(
    std::span<const std::byte> artifact, const ArtifactDigest& expected_digest,
    const nn::InvocationSchema& input_schema, const ValueSchema& output_schema,
    const TorchScriptLoadOptions& options, std::stop_token stop,
    std::optional<std::chrono::steady_clock::time_point> deadline) {
  return BackendBoundary("load", [&] {
    Boundary(stop, deadline);
    ValidateLimits(options.limits);
    if (!options.trust_source)
      Unsupported("trust_source", "explicit approval of executable source");
    validate_artifact_digest(expected_digest);
    if (artifact.empty())
      Invalid("artifact", "nonempty complete archive bytes");
    if (artifact.size() > options.limits.max_artifact_bytes)
      Resource("artifact.bytes");
    if (input_schema.positional.size() > options.limits.max_arguments ||
        input_schema.keyword.size() >
            options.limits.max_arguments - input_schema.positional.size())
      Resource("input.arguments");
    if (input_schema.allow_extra_keywords)
      Unsupported("input.schema", "a complete closed keyword boundary");
    nn::validate_invocation_schema(input_schema);
    for (const auto& field : input_schema.positional) {
      validate_value_schema(field, "input.positional",
                            options.limits.value_limits);
      CheckTensorConstraints(field, options);
    }
    for (const auto& [name, field] : input_schema.keyword) {
      validate_value_schema(field, "input.keyword." + name,
                            options.limits.value_limits);
      CheckTensorConstraints(field, options);
    }
    validate_value_schema(output_schema, "output", options.limits.value_limits);
    CheckTensorConstraints(output_schema, options);
    ValidateDevice(options.device);
    CheckPrecisionContext(options.device);
    // Own and hash the exact same bytes that LibTorch will read. The caller
    // must exclude writes only until this snapshot has been made.
    std::string snapshot(reinterpret_cast<const char*>(artifact.data()),
                         artifact.size());
    if (sha256_hex(std::as_bytes(std::span(
            snapshot.data(), snapshot.size()))) != expected_digest.hex)
      Unsupported("artifact.digest", "the expected exact SHA-256 identity");
    Boundary(stop, deadline);
    auto impl = std::make_unique<Impl>();
    impl->input_schema = input_schema;
    impl->output_schema = output_schema;
    impl->options = options;
    std::istringstream stream(std::move(snapshot),
                              std::ios::in | std::ios::binary);
    c10::InferenceMode inference_guard;
    impl->module = torch::jit::load(stream, options.device, false);
    impl->module->eval();
    Boundary(stop, deadline);
    const auto method = impl->module->find_method("forward");
    if (!method) Unsupported("forward", "one defined forward method");
    const auto& actual = method->function().getSchema();
    if (actual.is_vararg() || actual.is_varret() ||
        actual.arguments().empty() ||
        actual.arguments().front().name() != "self" ||
        actual.returns().size() != 1U)
      Unsupported("forward.schema", "nonvariadic self method with one result");
    const auto count = actual.arguments().size() - 1U;
    const auto prefix = input_schema.positional.size();
    if (count > options.limits.max_arguments) Resource("forward.arguments");
    if (prefix > count || input_schema.keyword.size() != count - prefix)
      Unsupported("input.binding", "one explicit schema per actual argument");
    auto& info = impl->info;
    info.artifact_digest = expected_digest;
    info.artifact_bytes = artifact.size();
    info.libtorch_version = TORCH_VERSION;
    info.device = options.device.str();
    TypeBudget type_budget{options.limits};
    ValueBudget default_budget{options.limits, options.device,
                               options.limits.max_input_tensor_bytes, true};
    for (std::size_t i = 0; i < count; ++i) {
      const auto& argument = actual.arguments()[i + 1U];
      if (argument.kwarg_only())
        Unsupported("forward.argument", "non-keyword-only JIT arguments");
      Add(type_budget.metadata, argument.name().size(),
          options.limits.value_limits.max_metadata_bytes, "forward.metadata");
      const ValueSchema* field = nullptr;
      if (i < prefix) {
        field = &input_schema.positional[i];
        if (!field->required)
          Unsupported("input.positional", "a fixed required positional prefix");
      } else {
        const auto found = input_schema.keyword.find(argument.name());
        if (found == input_schema.keyword.end())
          Unsupported("input.keyword", "exact actual JIT argument names");
        field = &found->second;
        if (!field->required) {
          if (!argument.default_value())
            Unsupported("input.default",
                        "a real artifact default for omission");
          CheckValue(*argument.default_value(), default_budget);
          validate_value(*argument.default_value(), *field, "input.default",
                         options.limits.value_limits);
          impl->defaults.emplace(argument.name(),
                                 Snapshot(*argument.default_value()));
        }
      }
      CheckSupportedType(argument.type(), type_budget);
      CheckBinding(argument.type(), *field);
      info.arguments.push_back(
          {.name = argument.name(),
           .type = TypeText(argument.type(), type_budget),
           .has_default = argument.default_value().has_value(),
           .bound_positionally = i < prefix,
           .required = field->required});
    }
    const auto& result_type = actual.returns().front().type();
    CheckSupportedType(result_type, type_budget);
    CheckBinding(result_type, output_schema);
    info.return_type = TypeText(result_type, type_budget);
    const auto check_state = [&](const auto& entries) {
      for (const auto& entry : entries) {
        if (info.state_tensors >= options.limits.max_state_tensors)
          Resource("state.tensors");
        ++info.state_tensors;
        Add(type_budget.metadata, entry.name.size(),
            options.limits.value_limits.max_metadata_bytes, "state.metadata");
        Add(info.state_tensor_bytes,
            TensorBytes(entry.value, options.device, options.limits, false),
            options.limits.max_state_tensor_bytes, "state.tensor_bytes");
      }
    };
    check_state(impl->module->named_parameters(true));
    check_state(impl->module->named_buffers(true));
    Synchronize(options.device);
    Boundary(stop, deadline);
    info.open = true;
    return TorchScriptCompatibilityRuntime(std::move(impl));
  });
}

nn::InvocationResult TorchScriptCompatibilityRuntime::invoke(
    const nn::Invocation& input, std::stop_token stop,
    std::optional<std::chrono::steady_clock::time_point> deadline) {
  return BackendBoundary("invoke", [&] {
    if (!impl_) throw NotReady("TorchScript runtime ownership was moved away");
    auto lock = Lock(impl_->mutex, stop, deadline);
    if (!impl_->module) throw NotReady("TorchScript runtime is closed");
    const auto& limits = impl_->options.limits;
    CheckPrecisionContext(impl_->options.device);
    ValueBudget input_budget{limits, impl_->options.device,
                             limits.max_input_tensor_bytes, true};
    for (const auto& value : input.positional) CheckValue(value, input_budget);
    for (const auto& [name, value] : input.keyword) {
      Add(input_budget.strings, name.size(), limits.max_string_bytes,
          "input.keyword_bytes");
      CheckValue(value, input_budget);
    }
    nn::validate_invocation(input, impl_->input_schema);
    for (const auto& [name, value] : impl_->defaults)
      if (!input.keyword.contains(name)) CheckValue(value, input_budget);
    const auto& actual_arguments =
        impl_->module->get_method("forward").function().getSchema().arguments();
    for (std::size_t i = 0; i < impl_->info.arguments.size(); ++i) {
      const auto& argument = impl_->info.arguments[i];
      const auto& value = argument.bound_positionally
                              ? input.positional[i]
                              : (input.keyword.contains(argument.name)
                                     ? input.keyword.at(argument.name)
                                     : impl_->defaults.at(argument.name));
      if (!value.type()->isSubtypeOf(actual_arguments[i + 1U].type()))
        Invalid("input.jit_type",
                "the exact typed JIT container/scalar boundary");
    }
    Boundary(stop, deadline);
    c10::InferenceMode inference_guard;
    std::vector<c10::IValue> positional;
    positional.reserve(input.positional.size());
    for (const auto& value : input.positional)
      positional.push_back(Snapshot(value));
    torch::jit::Kwargs keywords;
    for (const auto& [name, value] : input.keyword)
      keywords.emplace(name, Snapshot(value));
    for (const auto& [name, value] : impl_->defaults)
      if (!keywords.contains(name)) keywords.emplace(name, Snapshot(value));
    Boundary(stop, deadline);
    try {
      auto output = impl_->module->forward(std::move(positional), keywords);
      Synchronize(impl_->options.device);
      Boundary(stop, deadline);
      ValueBudget output_budget{limits, impl_->options.device,
                                limits.max_output_tensor_bytes, false};
      CheckValue(output, output_budget);
      const auto& expected_type = impl_->module->get_method("forward")
                                      .function()
                                      .getSchema()
                                      .returns()
                                      .front()
                                      .type();
      if (!output.type()->isSubtypeOf(expected_type))
        Unsupported("output.jit_type", "the actual declared JIT result type");
      validate_value(output, impl_->output_schema, "output",
                     limits.value_limits);
      auto independent_output = Snapshot(output);
      Synchronize(impl_->options.device);
      Boundary(stop, deadline);
      return nn::InvocationResult{.value = std::move(independent_output),
                                  .schema = impl_->output_schema};
    } catch (...) {
      // Forward may have changed private state or enqueued device work. Do not
      // advertise rollback or allow reuse of a partially executed module.
      try {
        Synchronize(impl_->options.device);
      } catch (...) {
        // Preserve the original failure. A failed CUDA context needs recovery
        // by the process owner; this in-process adapter cannot repair it.
      }
      impl_->Close();
      throw;
    }
  });
}

TorchScriptRuntimeInfo TorchScriptCompatibilityRuntime::inspect() const {
  if (!impl_) throw NotReady("TorchScript runtime ownership was moved away");
  const std::lock_guard lock(impl_->mutex);
  return impl_->info;
}

void TorchScriptCompatibilityRuntime::close() {
  if (!impl_) return;
  const std::lock_guard lock(impl_->mutex);
  impl_->Close();
}

}  // namespace jyppx::liuding::interop
