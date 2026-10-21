// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
#include "detail.hpp"

#include <jyppx/core/value_schema.hpp>
#include <jyppx/foundation/sha256.hpp>
#include <jyppx/foundation/version.hpp>
#include <jyppx/interop/model_weight_import.hpp>
#include <jyppx/serving/runtime.hpp>

#include <torch/cuda.h>

#include <ATen/autocast_mode.h>
#include <ATen/core/Dict.h>
#include <ATen/core/List.h>
#include <algorithm>
#include <c10/core/InferenceMode.h>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <limits>
#include <mutex>
#include <utility>
namespace jyppx::serving {
namespace {
void Add(std::uint64_t& total, std::uint64_t value, std::uint64_t limit) {
  if (value > limit || total > limit - value)
    detail::Resource("request.resources");
  total += value;
}
void Limits(const RequestLimits& l) {
  if (!l.max_tensor_bytes || !l.max_tensors || !l.max_rank ||
      l.max_rank > 64U || !l.max_nodes || !l.max_dimension || !l.max_elements ||
      !l.max_string_bytes || (l.minimum && !std::isfinite(*l.minimum)) ||
      (l.maximum && !std::isfinite(*l.maximum)) ||
      (l.minimum && l.maximum && *l.minimum > *l.maximum))
    detail::Invalid(
        "limits",
        "positive bounded traversal and finite ordered numeric range");
}
struct Budget {
  std::uint64_t bytes = 0, tensors = 0, nodes = 0, strings = 0;
};
int CompareIntegerToDouble(std::int64_t value, double bound) {
  constexpr double kSignedBoundary = 0x1p63;
  if (bound >= kSignedBoundary) return -1;
  if (bound < -kSignedBoundary) return 1;
  // The finite bound is now inside the int64 conversion range. Compare its
  // truncated integer part first so a value above 2^53 never loses low bits.
  const auto integral = static_cast<std::int64_t>(bound);
  if (value < integral) return -1;
  if (value > integral) return 1;
  // Truncating a double yields an exactly representable integer: below 2^53
  // every integer is exact, and larger doubles already have no fraction.
  const auto exact = static_cast<double>(integral);
  if (exact < bound) return -1;
  return exact > bound ? 1 : 0;
}
bool IntegerOutsideRange(std::int64_t value, const RequestLimits& limits) {
  return (limits.minimum &&
          CompareIntegerToDouble(value, *limits.minimum) < 0) ||
         (limits.maximum && CompareIntegerToDouble(value, *limits.maximum) > 0);
}
void Value(const c10::IValue& v, const RequestLimits& l,
           const RuntimeDescriptor& runtime, Budget& budget,
           std::size_t depth = 0) {
  if (depth > 32U) detail::Resource("value.depth");
  Add(budget.nodes, 1, l.max_nodes);
  if (v.isTensor()) {
    const auto& t = v.toTensor();
    if (!t.defined() || t.layout() != torch::kStrided || t.is_quantized() ||
        t.is_complex() || t.is_conj() || t.is_neg() ||
        t.device() != runtime.device ||
        std::find(runtime.dtypes.begin(), runtime.dtypes.end(),
                  t.scalar_type()) == runtime.dtypes.end())
      detail::Invalid("tensor", "dense declared dtype on exact runtime device");
    Add(budget.tensors, 1, l.max_tensors);
    if (static_cast<std::size_t>(t.dim()) > l.max_rank)
      detail::Resource("tensor.rank");
    std::uint64_t elements = 1;
    for (const auto extent : t.sizes()) {
      if (extent < 0 || static_cast<std::uint64_t>(extent) > l.max_dimension)
        detail::Resource("tensor.dimension");
      if (extent != 0 &&
          elements > l.max_elements / static_cast<std::uint64_t>(extent))
        detail::Resource("tensor.elements");
      elements *= static_cast<std::uint64_t>(extent);
    }
    if (elements > l.max_elements || t.element_size() == 0 ||
        elements > std::numeric_limits<std::uint64_t>::max() / t.element_size())
      detail::Resource("tensor.elements");
    Add(budget.bytes, elements * t.element_size(), l.max_tensor_bytes);
    if (t.numel() != 0) {
      if (l.require_finite && t.is_floating_point() &&
          !torch::isfinite(t).all().item<bool>())
        throw NumericalError(
            "Inference tensor contains nonfinite values",
            detail::Details("input.range", "finite tensor values"));
      const bool outside =
          t.is_floating_point()
              ? ((l.minimum && t.min().item<double>() < *l.minimum) ||
                 (l.maximum && t.max().item<double>() > *l.maximum))
              : ((l.minimum &&
                  CompareIntegerToDouble(t.min().item<std::int64_t>(),
                                         *l.minimum) < 0) ||
                 (l.maximum &&
                  CompareIntegerToDouble(t.max().item<std::int64_t>(),
                                         *l.maximum) > 0));
      if (outside)
        detail::Invalid("input.range",
                        "values inside the declared numeric range");
    }
  } else if (v.isTuple()) {
    if (v.toTupleRef().type()->name())
      detail::Invalid("value", "unnamed tuple");
    for (const auto& item : v.toTupleRef().elements())
      Value(item, l, runtime, budget, depth + 1);
  } else if (v.isList()) {
    for (const auto& item : v.toListRef())
      Value(item, l, runtime, budget, depth + 1);
  } else if (v.isGenericDict()) {
    for (const auto& item : v.toGenericDict()) {
      if (!item.key().isString()) detail::Invalid("dictionary", "string keys");
      Value(item.key(), l, runtime, budget, depth + 1);
      Value(item.value(), l, runtime, budget, depth + 1);
    }
  } else if (v.isString())
    Add(budget.strings, v.toStringRef().size(), l.max_string_bytes);
  else if (v.isDouble()) {
    const auto x = v.toDouble();
    if ((l.require_finite && !std::isfinite(x)) ||
        (l.minimum && x < *l.minimum) || (l.maximum && x > *l.maximum))
      detail::Invalid("scalar.range", "finite scalar in declared range");
  } else if (v.isInt()) {
    if (IntegerOutsideRange(v.toInt(), l))
      detail::Invalid("scalar.range", "integer inside declared numeric range");
  } else if (!v.isBool() && !v.isNone())
    detail::Invalid("value",
                    "tensor, exact scalar, None or supported closed container");
}
std::string DtypeName(Dtype d);
void Descriptor(const RuntimeDescriptor& d) {
  detail::Text(d.backend_id, true);
  detail::Text(d.backend_version, true);
  detail::Text(d.format, true);
  validate_artifact_digest(d.environment_lock);
  if (d.dtypes.empty() || !d.max_batch_size || d.max_batch_size > 4096U ||
      (!d.tensor_batching && d.max_batch_size != 1U))
    detail::Invalid("runtime.descriptor",
                    "explicit dtype and bounded batch capability");
  std::vector<Dtype> seen;
  for (const auto dtype : d.dtypes) {
    static_cast<void>(DtypeName(dtype));
    if (std::find(seen.begin(), seen.end(), dtype) != seen.end())
      detail::Invalid("runtime.dtype", "unique declared dtypes");
    seen.push_back(dtype);
  }
  const auto libtorch =
      std::find_if(d.dependencies.begin(), d.dependencies.end(),
                   [](const auto& dep) { return dep.name == "libtorch"; });
  if (libtorch == d.dependencies.end() ||
      libtorch->version != build_info().libtorch_version)
    throw CompatibilityError("Serving LibTorch requirement mismatch",
                             detail::Details("runtime.dependencies",
                                             "exact linked libtorch version"));
  if (d.device.is_cpu()) {
    if (d.device.has_index())
      detail::Invalid("runtime.device", "unindexed cpu");
  } else if (!d.device.is_cuda() || !d.device.has_index() ||
             d.device.index() < 0 || !torch::cuda::is_available() ||
             static_cast<std::size_t>(d.device.index()) >=
                 torch::cuda::device_count())
    throw DeviceError(
        "Serving runtime device unavailable",
        detail::Details("runtime.device", "cpu or available indexed cuda:N"));
}
void Sync(const Device& device) {
  if (device.is_cuda()) torch::cuda::synchronize(device.index());
}
void Binding(const DescriptorBinding& binding,
             const interop::BundleDescriptorRef& reference,
             const VerifiedBundle& bundle) {
  const auto& files = bundle.verification.manifest.files;
  const auto found =
      std::find_if(files.begin(), files.end(),
                   [&](const auto& f) { return f.id == reference.file_id; });
  if (found == files.end() || binding.schema_id != reference.schema_id ||
      binding.schema_version != reference.schema_version ||
      binding.digest != found->artifact.digest)
    throw CompatibilityError(
        "Serving owner schema binding mismatch",
        detail::Details("pipeline.binding",
                        "exact versioned verified descriptor"));
}
std::vector<std::byte> Read(const std::filesystem::path& path,
                            const ArtifactRef& ref, std::uint64_t bound) {
  if (ref.size_bytes > bound ||
      ref.size_bytes > std::numeric_limits<std::size_t>::max() ||
      ref.size_bytes > static_cast<std::uint64_t>(
                           std::numeric_limits<std::streamsize>::max()))
    detail::Resource("bundle.snapshot");
  const auto status = std::filesystem::symlink_status(path);
  if (std::filesystem::is_symlink(status) ||
      !std::filesystem::is_regular_file(status))
    throw IoError(
        "Invalid bundle snapshot file",
        detail::Details("bundle.read", "immutable verified regular file"));
  std::ifstream input(path, std::ios::binary);
  std::vector<std::byte> bytes(static_cast<std::size_t>(ref.size_bytes));
  input.read(reinterpret_cast<char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input || input.peek() != std::char_traits<char>::eof() ||
      sha256_hex(bytes) != ref.digest.hex)
    throw SerializationError(
        "Bundle changed during snapshot",
        detail::Details("bundle.read", "exact verified file bytes"));
  return bytes;
}
std::string DtypeName(Dtype d) {
  switch (d) {
    case torch::kFloat32:
      return "float32";
    case torch::kFloat64:
      return "float64";
    case torch::kFloat16:
      return "float16";
    case torch::kBFloat16:
      return "bfloat16";
    case torch::kInt64:
      return "int64";
    case torch::kInt32:
      return "int32";
    case torch::kInt16:
      return "int16";
    case torch::kInt8:
      return "int8";
    case torch::kUInt8:
      return "uint8";
    case torch::kBool:
      return "bool";
    default:
      detail::Invalid("runtime.dtype", "supported dense CPU/CUDA dtype");
  }
}
}  // namespace
namespace detail {
void Context(const RequestContext& c, std::string_view version) {
  for (const auto* value : {&c.request_id, &c.trace_id, &c.project_id,
                            &c.principal_id, &c.deployment_id, &c.model_version,
                            &c.priority_class, &c.deadline_class})
    Text(*value);
  Text(c.request_id, true);
  if (c.model_version != version)
    Invalid("request.model_version", "the loaded immutable model version");
}
c10::IValue Snapshot(const c10::IValue& v) {
  if (v.isTensor()) return v.toTensor().clone(at::MemoryFormat::Contiguous);
  if (v.isTuple()) {
    std::vector<c10::IValue> items;
    for (const auto& x : v.toTupleRef().elements())
      items.push_back(Snapshot(x));
    return c10::ivalue::Tuple::create(std::move(items));
  }
  if (v.isList()) {
    const auto old = v.toList();
    c10::impl::GenericList out(old.elementType());
    out.reserve(old.size());
    for (const auto& x : old) out.push_back(Snapshot(x));
    return out;
  }
  if (v.isGenericDict()) {
    const auto old = v.toGenericDict();
    c10::impl::GenericDict out(old.keyType(), old.valueType());
    for (const auto& x : old) out.insert(x.key(), Snapshot(x.value()));
    return out;
  }
  if (v.isString()) return std::string(v.toStringRef());
  return v;
}
nn::Invocation Snapshot(const nn::Invocation& v) {
  nn::Invocation out;
  for (const auto& x : v.positional) out.positional.push_back(Snapshot(x));
  for (const auto& [key, x] : v.keyword) out.keyword.emplace(key, Snapshot(x));
  return out;
}
bool Equal(const c10::IValue& a, const c10::IValue& b, double rtol,
           double atol) {
  if (a.isTensor() && b.isTensor()) {
    const auto x = a.toTensor(), y = b.toTensor();
    return x.sizes() == y.sizes() && x.scalar_type() == y.scalar_type() &&
           torch::allclose(x, y.to(x.device()), rtol, atol, false);
  }
  if (a.isTuple() && b.isTuple()) {
    const auto& x = a.toTupleRef().elements();
    const auto& y = b.toTupleRef().elements();
    if (x.size() != y.size()) return false;
    for (std::size_t i = 0; i < x.size(); ++i)
      if (!Equal(x[i], y[i], rtol, atol)) return false;
    return true;
  }
  if (a.isList() && b.isList()) {
    const auto x = a.toList(), y = b.toList();
    if (x.size() != y.size()) return false;
    for (std::size_t i = 0; i < x.size(); ++i)
      if (!Equal(x.get(i), y.get(i), rtol, atol)) return false;
    return true;
  }
  if (a.isGenericDict() && b.isGenericDict()) {
    const auto x = a.toGenericDict(), y = b.toGenericDict();
    if (x.size() != y.size()) return false;
    for (const auto& item : x) {
      const auto found = y.find(item.key());
      if (found == y.end() || !Equal(item.value(), found->value(), rtol, atol))
        return false;
    }
    return true;
  }
  if (a.isDouble() && b.isDouble())
    return std::isfinite(a.toDouble()) && std::isfinite(b.toDouble()) &&
           std::abs(a.toDouble() - b.toDouble()) <=
               atol + rtol * std::abs(b.toDouble());
  if (a.isInt() && b.isInt()) return a.toInt() == b.toInt();
  if (a.isBool() && b.isBool()) return a.toBool() == b.toBool();
  if (a.isString() && b.isString()) return a.toStringRef() == b.toStringRef();
  return a.isNone() && b.isNone();
}
}  // namespace detail
std::uint64_t validate_request_resources(const nn::Invocation& input,
                                         const RequestLimits& limits,
                                         const RuntimeDescriptor& runtime) {
  return detail::Backend("request.validate", [&] {
    Limits(limits);
    Budget budget;
    for (const auto& v : input.positional) Value(v, limits, runtime, budget);
    for (const auto& [key, v] : input.keyword) {
      Add(budget.strings, key.size(), limits.max_string_bytes);
      Value(v, limits, runtime, budget);
    }
    return budget.bytes;
  });
}
class Predictor::Impl {
 public:
  mutable std::mutex mutex;
  std::mutex observe_mutex;
  std::condition_variable cv;
  std::timed_mutex execute;
  ModelRuntimeInfo metadata;
  Pipeline pipeline;
  LoadOptions options;
  std::function<nn::InvocationResult(const nn::Invocation&, std::stop_token,
                                     const Deadline&)>
      invoke;
  std::function<void()> release;
  bool accepting = true, releasing = false;
  ModelRuntimeInfo Info() const {
    std::lock_guard lock(mutex);
    return metadata;
  }
  void Observe(const RequestContext& c, const char* event) {
    if (!options.observe) return;
    std::lock_guard observation(observe_mutex);
    try {
      options.observe({.severity = TelemetryLogSeverity::info, .body = event},
                      {.project_id = c.project_id,
                       .run_id = {},
                       .attempt_id = {},
                       .trace_id = c.trace_id,
                       .span_id = {},
                       .parent_span_id = {}},
                      {{"request_id", c.request_id},
                       {"model_version", metadata.model_version},
                       {"deployment_id", c.deployment_id},
                       {"backend", metadata.runtime.backend_id},
                       {"device", metadata.runtime.device.str()}});
    } catch (...) {
      std::lock_guard lock(mutex);
      ++metadata.observation_failures;
    }
  }
  void Warmup(Predictor& owner, std::span<const WarmupCase> cases) {
    if (cases.empty() || cases.size() > options.max_warmup_cases)
      detail::Invalid("warmup", "at least one bounded known-vector case");
    std::size_t i = 0;
    Limits(options.request_limits);
    Limits(options.output_limits);
    for (const auto& c : cases) {
      if (!std::isfinite(c.relative_tolerance) ||
          !std::isfinite(c.absolute_tolerance) || c.relative_tolerance < 0 ||
          c.absolute_tolerance < 0)
        detail::Invalid("warmup.tolerance", "finite nonnegative tolerances");
      detail::Boundary(options.stop, options.deadline);
      validate_request_resources({.positional = {c.expected}, .keyword = {}},
                                 options.output_limits, metadata.runtime);
      validate_value(c.expected, pipeline.response_schema, "warmup.expected");
      RequestContext context;
      context.request_id = "warmup-" + std::to_string(i++);
      context.model_version = metadata.model_version;
      const auto result = owner.predict(
          {.input = c.input, .context = context, .deadline = options.deadline},
          options.stop);
      if (!detail::Equal(result.output.value, c.expected, c.relative_tolerance,
                         c.absolute_tolerance))
        throw NumericalError(
            "Warmup known-vector mismatch",
            detail::Details("warmup", "expected numerical output"));
    }
    std::lock_guard lock(mutex);
    metadata.accepted = metadata.completed = metadata.failed = 0;
  }
};
Predictor::Predictor(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
Predictor::~Predictor() {
  try {
    close();
  } catch (...) {
  }
}
ModelRuntimeInfo Predictor::info() const { return impl_->Info(); }
nn::InvocationSchema Predictor::input_schema() const {
  return impl_->pipeline.request_schema;
}
ValueSchema Predictor::output_schema() const {
  return impl_->pipeline.response_schema;
}
RequestLimits Predictor::request_limits() const {
  return impl_->options.request_limits;
}
PredictRequest Predictor::snapshot_request(
    const PredictRequest& request) const {
  return detail::Backend("request.snapshot", [&] {
    {
      std::lock_guard lock(impl_->mutex);
      if (impl_->metadata.health == HealthStatus::closed ||
          impl_->metadata.health == HealthStatus::degraded)
        throw NotReady(
            "Inference instance unavailable",
            detail::Details("request.snapshot", "retained healthy instance"));
    }
    detail::Context(request.context, impl_->metadata.model_version);
    detail::Boundary({}, request.deadline);
    nn::validate_invocation(request.input, impl_->pipeline.request_schema);
    validate_request_resources(request.input, impl_->options.request_limits,
                               impl_->metadata.runtime);
    c10::InferenceMode guard;
    auto result = request;
    result.input = detail::Snapshot(request.input);
    Sync(impl_->metadata.runtime.device);
    return result;
  });
}
PredictResponse Predictor::predict(const PredictRequest& request,
                                   std::stop_token stop) {
  return detail::Backend("predict", [&] {
    auto impl = impl_;
    detail::Boundary(stop, request.deadline);
    {
      std::lock_guard lock(impl->mutex);
      if (!impl->accepting || impl->metadata.health != HealthStatus::ready)
        throw NotReady("Inference instance is not ready",
                       detail::Details("predict", "ready accepting instance"));
      if (impl->metadata.active >= impl->options.max_active_requests)
        detail::Resource("predict.admission");
      ++impl->metadata.active;
      ++impl->metadata.accepted;
    }
    const auto lease = std::shared_ptr<void>(impl.get(), [impl](void*) {
      std::lock_guard lock(impl->mutex);
      --impl->metadata.active;
      impl->cv.notify_all();
    });
    const auto start = std::chrono::steady_clock::now();
    bool entered = false, invoke_failed = false, retryable = false;
    std::unique_lock lock(impl->execute, std::defer_lock);
    try {
      auto snapshot = snapshot_request(request);
      while (!lock.try_lock_for(std::chrono::milliseconds(1)))
        detail::Boundary(stop, request.deadline);
      detail::Boundary(stop, request.deadline);
      {
        std::lock_guard lifecycle(impl->mutex);
        if (impl->metadata.health == HealthStatus::degraded)
          throw NotReady(
              "Inference instance degraded",
              detail::Details("predict", "healthy existing session"));
      }
      if (at::autocast::is_autocast_enabled(
              impl->metadata.runtime.device.type()))
        detail::Invalid("predict.autocast",
                        "autocast disabled on the explicit runtime device");
      c10::InferenceMode guard;
      TimingBreakdown timing;
      auto stage = std::chrono::steady_clock::now();
      auto processed = impl->pipeline.preprocess
                           ? impl->pipeline.preprocess(snapshot.input)
                           : snapshot.input;
      validate_request_resources(processed, impl->options.request_limits,
                                 impl->metadata.runtime);
      timing.preprocess =
          detail::Seconds(std::chrono::steady_clock::now() - stage);
      detail::Boundary(stop, request.deadline);
      stage = std::chrono::steady_clock::now();
      entered = true;
      nn::InvocationResult result;
      try {
        result = impl->invoke(processed, stop, request.deadline);
        Sync(impl->metadata.runtime.device);
      } catch (const std::exception& error) {
        invoke_failed = true;
        retryable =
            impl->metadata.runtime.oom_retry_safe && detail::IsOom(error);
        if (retryable)
          std::throw_with_nested(RetryableOutOfMemory(
              "Retry-safe native allocation failed",
              detail::Details("predict.forward",
                              "available memory for this batch")));
        throw;
      }
      timing.compute =
          detail::Seconds(std::chrono::steady_clock::now() - stage);
      detail::Boundary(stop, request.deadline);
      stage = std::chrono::steady_clock::now();
      if (impl->pipeline.postprocess)
        result = impl->pipeline.postprocess(result);
      validate_value(result.value, impl->pipeline.response_schema, "response");
      validate_request_resources({.positional = {result.value}, .keyword = {}},
                                 impl->options.output_limits,
                                 impl->metadata.runtime);
      result.value = detail::Snapshot(result.value);
      result.schema = impl->pipeline.response_schema;
      Sync(impl->metadata.runtime.device);
      detail::Boundary(stop, request.deadline);
      timing.postprocess =
          detail::Seconds(std::chrono::steady_clock::now() - stage);
      timing.total = detail::Seconds(std::chrono::steady_clock::now() - start);
      {
        std::lock_guard lifecycle(impl->mutex);
        ++impl->metadata.completed;
      }
      impl->Observe(request.context, "serving.predict.completed");
      return PredictResponse{.output = std::move(result),
                             .context = request.context,
                             .runtime = impl->Info(),
                             .timing = timing};
    } catch (...) {
      // Finish device work before a lease can release request/module storage.
      if (entered) {
        try {
          Sync(impl->metadata.runtime.device);
        } catch (...) {
          retryable = false;
        }
      }
      {
        std::lock_guard lifecycle(impl->mutex);
        ++impl->metadata.failed;
        if (entered && !(invoke_failed && retryable)) {
          impl->metadata.health = HealthStatus::degraded;
          impl->accepting = false;
        }
      }
      impl->Observe(request.context, "serving.predict.failed");
      throw;
    }
  });
}
bool Predictor::close(Deadline deadline, std::stop_token stop) {
  auto impl = impl_;
  std::unique_lock lock(impl->mutex);
  impl->accepting = false;
  if (impl->metadata.health == HealthStatus::closed) return true;
  if (impl->metadata.health == HealthStatus::ready)
    impl->metadata.health = HealthStatus::draining;
  while (impl->metadata.active || impl->releasing) {
    if (stop.stop_requested() ||
        (deadline && std::chrono::steady_clock::now() >= *deadline))
      return false;
    impl->cv.wait_for(lock, std::chrono::milliseconds(1));
    if (impl->metadata.health == HealthStatus::closed) return true;
  }
  impl->releasing = true;
  lock.unlock();
  try {
    if (impl->release) impl->release();
  } catch (...) {
    lock.lock();
    impl->releasing = false;
    impl->metadata.health = HealthStatus::degraded;
    impl->cv.notify_all();
    throw;
  }
  lock.lock();
  impl->invoke = {};
  impl->release = {};
  impl->pipeline.preprocess = {};
  impl->pipeline.postprocess = {};
  impl->metadata.health = HealthStatus::closed;
  impl->releasing = false;
  impl->cv.notify_all();
  return true;
}
NativeStateBackend::NativeStateBackend(RuntimeDescriptor descriptor,
                                       NativeModelFactory factory,
                                       std::string model_id)
    : descriptor_(std::move(descriptor)),
      factory_(std::move(factory)),
      model_id_(std::move(model_id)) {
  Descriptor(descriptor_);
  detail::Text(model_id_, true);
  if (!factory_ || descriptor_.format != "safetensors")
    detail::Invalid("native.backend",
                    "explicit safetensors descriptor and factory");
}
RuntimeDescriptor NativeStateBackend::descriptor() const { return descriptor_; }
std::shared_ptr<Predictor> NativeStateBackend::load(
    const std::filesystem::path& directory,
    const interop::ModelBundleVerifyOptions& verify,
    std::span<const WarmupCase> warmup, const LoadOptions& options) const {
  return detail::Backend("native.load", [&] {
    if (!options.max_snapshot_bytes || !options.max_state_bytes ||
        !options.max_warmup_cases || !options.max_active_requests)
      detail::Invalid("load.limits", "positive state/snapshot/warmup limits");
    detail::Boundary(options.stop, options.deadline);
    Limits(options.request_limits);
    Limits(options.output_limits);
    VerifiedBundle bundle;
    bundle.verification =
        interop::verify_model_bundle_directory(directory, verify);
    const auto& manifest = bundle.verification.manifest;
    const auto& runtime = manifest.runtime;
    if (runtime.backend_id != descriptor_.backend_id ||
        runtime.backend_version != descriptor_.backend_version ||
        runtime.environment_lock != descriptor_.environment_lock ||
        runtime.dynamic_shapes != descriptor_.dynamic_shapes)
      throw CompatibilityError(
          "Native bundle backend requirement mismatch",
          detail::Details("native.capability",
                          "exact declared backend/version/environment/shape"));
    auto dependencies = descriptor_.dependencies;
    const auto order = [](const auto& a, const auto& b) {
      return a.name < b.name;
    };
    auto declared = runtime.dependencies;
    std::sort(dependencies.begin(), dependencies.end(), order);
    std::sort(declared.begin(), declared.end(), order);
    if (dependencies.size() != declared.size())
      throw CompatibilityError("Native dependency mismatch");
    for (std::size_t i = 0; i < dependencies.size(); ++i)
      if (dependencies[i].name != declared[i].name ||
          dependencies[i].version != declared[i].version)
        throw CompatibilityError("Native dependency mismatch");
    const auto platform = serving_platform_id();
    if (std::find(runtime.os_arch.begin(), runtime.os_arch.end(), platform) ==
        runtime.os_arch.end())
      throw CompatibilityError(
          "Native bundle platform requirement mismatch",
          detail::Details(
              "native.platform",
              "actual current build os_arch in declared platforms"));
    const auto family = descriptor_.device.is_cuda() ? "cuda" : "cpu";
    if (std::find(runtime.devices.begin(), runtime.devices.end(), family) ==
        runtime.devices.end())
      throw CompatibilityError("Native bundle device requirement mismatch");
    for (const auto dtype : descriptor_.dtypes)
      if (std::find(runtime.dtypes.begin(), runtime.dtypes.end(),
                    DtypeName(dtype)) == runtime.dtypes.end())
        throw CompatibilityError("Native bundle dtype requirement mismatch");
    std::uint64_t total = 0;
    try {
      for (const auto& file : manifest.files) {
        detail::Boundary(options.stop, options.deadline);
        Add(total, file.artifact.size_bytes, options.max_snapshot_bytes);
        bundle.files.emplace(file.id,
                             Read(directory / file.relative_path, file.artifact,
                                  options.max_snapshot_bytes));
      }
    } catch (const std::filesystem::filesystem_error&) {
      std::throw_with_nested(
          IoError("Cannot snapshot immutable serving bundle"));
    }
    detail::Boundary(options.stop, options.deadline);
    auto native = factory_(bundle, descriptor_);
    detail::Boundary(options.stop, options.deadline);
    if (!native.adapter)
      detail::Invalid("native.factory", "an owning fresh ModelAdapter");
    detail::Text(native.pipeline.version, true);
    nn::validate_invocation_schema(native.pipeline.request_schema);
    validate_value_schema(native.pipeline.response_schema);
    Binding(native.pipeline.input, manifest.input_schema, bundle);
    Binding(native.pipeline.output, manifest.output_schema, bundle);
    Binding(native.pipeline.preprocessing, manifest.preprocessing, bundle);
    Binding(native.pipeline.postprocessing, manifest.postprocessing, bundle);
    const auto model =
        std::find_if(manifest.models.begin(), manifest.models.end(),
                     [&](const auto& m) { return m.id == model_id_; });
    if (model == manifest.models.end() ||
        model->format != interop::BundleModelFormat::safetensors)
      throw CompatibilityError(
          "Selected native-state model variant unavailable");
    std::vector<interop::SafeTensorSource> tensors;
    for (const auto& file_id : model->file_ids) {
      const auto& bytes = bundle.files.at(file_id);
      const interop::SafeTensorsView view(bytes);
      for (const auto& tensor : view.directory().tensors)
        tensors.push_back({.name = tensor.name,
                           .dtype = tensor.dtype,
                           .shape = tensor.shape,
                           .data = view.tensor_bytes(tensor.name)});
    }
    if (total >= options.max_snapshot_bytes)
      detail::Resource("native.state.snapshot");
    interop::SafeTensorsLimits encoded_limits;
    encoded_limits.max_file_bytes = options.max_snapshot_bytes - total;
    auto state = interop::serialize_safetensors(tensors, {}, encoded_limits);
    Add(total, state.size(), options.max_snapshot_bytes);
    if (state.size() > options.max_snapshot_bytes)
      detail::Resource("native.state.bytes");
    auto& module = native.adapter->state().module();
    std::uint64_t state_bytes = 0;
    for (const auto& t : module.parameters()) {
      if (t.device() != descriptor_.device)
        throw CompatibilityError("Factory state device mismatch");
      if (t.numel() < 0 || t.element_size() == 0 ||
          static_cast<std::uint64_t>(t.numel()) >
              options.max_state_bytes / t.element_size())
        detail::Resource("native.state.bytes");
      Add(state_bytes, static_cast<std::uint64_t>(t.numel()) * t.element_size(),
          options.max_state_bytes);
    }
    for (const auto& t : module.buffers()) {
      if (t.device() != descriptor_.device)
        throw CompatibilityError("Factory state device mismatch");
      if (t.numel() < 0 || t.element_size() == 0 ||
          static_cast<std::uint64_t>(t.numel()) >
              options.max_state_bytes / t.element_size())
        detail::Resource("native.state.bytes");
      Add(state_bytes, static_cast<std::uint64_t>(t.numel()) * t.element_size(),
          options.max_state_bytes);
    }
    interop::ModelWeightImportOptions import_options;
    import_options.max_staging_bytes = options.max_state_bytes;
    // Factory code may allocate/initialize on an asynchronous CUDA stream.
    Sync(descriptor_.device);
    const auto imported = interop::import_model_weights(
        native.adapter->state(), state, {.hex = sha256_hex(state)},
        import_options, options.stop, options.deadline);
    if (!imported.applied)
      throw CompatibilityError("Native bundle state mapping rejected");
    native.adapter->state().eval();
    auto adapter = native.adapter;
    auto impl = std::make_shared<Predictor::Impl>();
    impl->options = options;
    impl->pipeline = std::move(native.pipeline);
    impl->metadata = {.model_version = manifest.bundle_id,
                      .pipeline_version = impl->pipeline.version,
                      .runtime = descriptor_,
                      .health = HealthStatus::ready};
    impl->invoke = [adapter](const nn::Invocation& input, std::stop_token stop,
                             const Deadline& deadline) {
      detail::Boundary(stop, deadline);
      return adapter->invoke(input);
    };
    impl->release = [adapter] { static_cast<void>(adapter); };
    auto predictor = std::shared_ptr<Predictor>(new Predictor(impl));
    impl->Warmup(*predictor, warmup);
    return predictor;
  });
}
std::shared_ptr<Predictor> load_torchscript_predictor(
    std::span<const std::byte> archive, const ArtifactDigest& expected,
    const nn::InvocationSchema& input_schema, const ValueSchema& output_schema,
    const interop::TorchScriptLoadOptions& runtime_options,
    RuntimeDescriptor descriptor, std::string model_version,
    std::span<const WarmupCase> warmup, const LoadOptions& options) {
  Descriptor(descriptor);
  detail::Text(model_version, true);
  Limits(options.request_limits);
  Limits(options.output_limits);
  if (!options.max_active_requests || !options.max_warmup_cases ||
      descriptor.device != runtime_options.device ||
      descriptor.tensor_batching || descriptor.oom_retry_safe ||
      descriptor.backend_id != "jyppx.torchscript.compatibility" ||
      descriptor.backend_version != "1" || descriptor.format != "torchscript")
    detail::Invalid("jit.descriptor",
                    "explicit nonbatching compatibility backend, exact device "
                    "and no retry");
  auto runtime = std::make_shared<interop::TorchScriptCompatibilityRuntime>(
      interop::TorchScriptCompatibilityRuntime::load(
          archive, expected, input_schema, output_schema, runtime_options,
          options.stop, options.deadline));
  auto impl = std::make_shared<Predictor::Impl>();
  impl->options = options;
  impl->pipeline.version = "jit-compatibility-v1";
  impl->pipeline.request_schema = input_schema;
  impl->pipeline.response_schema = output_schema;
  impl->metadata.model_version = std::move(model_version);
  impl->metadata.pipeline_version = impl->pipeline.version;
  impl->metadata.runtime = std::move(descriptor);
  impl->metadata.health = HealthStatus::ready;
  impl->invoke = [runtime](const nn::Invocation& input, std::stop_token stop,
                           const Deadline& deadline) {
    return runtime->invoke(input, stop, deadline);
  };
  impl->release = [runtime] { runtime->close(); };
  auto predictor = std::shared_ptr<Predictor>(new Predictor(impl));
  impl->Warmup(*predictor, warmup);
  return predictor;
}
std::string serving_platform_id() {
  auto result = current_build_fingerprint().os_arch;
  for (auto& ch : result)
    if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
  return result;
}
std::string serving_preview_manifest_json() {
  return R"({"schema_id":"jyppx.serving.preview","schema_version":1,"default_enabled":false,"preview":true,"native_state":"safetensors","torchscript":"explicit-trust-compatibility","onnx_runtime":"unavailable","threading":"one-serialized-instance","batching":"single-positional-tensor-single-tensor-result","network_server":false,"reload":false,"qualification":"requires-recorded-build-test-reference-performance-evidence"})";
}
}  // namespace jyppx::serving
