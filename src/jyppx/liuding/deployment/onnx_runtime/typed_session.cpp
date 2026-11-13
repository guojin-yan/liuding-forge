// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/deployment/onnx_runtime/typed_session.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <onnxruntime_cxx_api.h>
#include <set>
#include <type_traits>
namespace jyppx::liuding::onnx_runtime {
namespace {
template <class T>
constexpr TensorElementType Type() {
  if constexpr (std::is_same_v<T, float>)
    return TensorElementType::float32;
  else if constexpr (std::is_same_v<T, std::int32_t>)
    return TensorElementType::int32;
  else
    return TensorElementType::int64;
}
std::uint64_t Width(TensorElementType type) {
  return type == TensorElementType::int64 ? 8 : 4;
}
std::uint64_t Elements(std::span<const std::int64_t> shape,
                       std::uint64_t maximum, bool zero = false) {
  std::uint64_t count = 1;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    const auto n = shape[i];
    if (n < 0 || (!n && (!zero || i)))
      throw InvalidArgument("Typed ORT dimension is invalid");
    if (n && count > maximum / static_cast<std::uint64_t>(n))
      throw ResourceExhausted("Typed ORT tensor exceeds byte capacity");
    count *= n;
  }
  return count;
}
TypedTensorSignature Signature(Ort::Session& session, bool input,
                               std::size_t index) {
  auto type = input ? session.GetInputTypeInfo(index)
                    : session.GetOutputTypeInfo(index);
  if (type.GetONNXType() != ONNX_TYPE_TENSOR)
    throw CompatibilityError("Typed ORT requires dense tensors");
  auto info = type.GetTensorTypeAndShapeInfo();
  const auto dtype = info.GetElementType();
  if (dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
      dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
      dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
    throw CompatibilityError("Typed ORT supports only FP32/int32/int64");
  Ort::AllocatorWithDefaultOptions allocator;
  auto name = input ? session.GetInputNameAllocated(index, allocator)
                    : session.GetOutputNameAllocated(index, allocator);
  TypedTensorSignature result{
      name.get(), static_cast<TensorElementType>(dtype), info.GetShape(), {}};
  if (result.name.empty() || result.name.size() > 4096 ||
      result.shape.empty() || result.shape.size() > 8)
    throw CompatibilityError("Typed ORT signature exceeds bounds");
  auto symbols = info.GetSymbolicDimensions();
  for (std::size_t i = 0; i < result.shape.size(); ++i)
    if (result.shape[i] <= 0) {
      if (i || result.shape[i] != -1 || !symbols[i] || !*symbols[i])
        throw CompatibilityError(
            "Typed ORT requires positive fixed axes or a named leading dynamic "
            "axis");
      result.leading_symbol = symbols[i];
      if (result.leading_symbol.size() > 4096)
        throw ResourceExhausted("Typed ORT symbol exceeds bound");
    }
  return result;
}
}  // namespace
struct TypedSession::Impl {
  TypedSessionOptions options;
  Ort::Env environment{ORT_LOGGING_LEVEL_ERROR, "liuding-typed-ort"};
  Ort::Session session{nullptr};
  std::vector<TypedTensorSignature> inputs, outputs;
  std::vector<const char*> input_names, output_names;
  Impl(std::span<const std::byte> bytes, const TypedSessionOptions& o)
      : options(o) {
    if (std::string_view(OrtGetApiBase()->GetVersionString()) != "1.23.2")
      throw CompatibilityError(
          "Typed ORT runtime version differs from the pinned SDK");
    Ort::SessionOptions settings;
    settings.SetIntraOpNumThreads(static_cast<int>(o.cpu_threads));
    settings.SetInterOpNumThreads(1);
    settings.SetExecutionMode(ORT_SEQUENTIAL);
    settings.DisableMemPattern();
    settings.DisableCpuMemArena();
    settings.AddConfigEntry("session.intra_op.allow_spinning", "0");
    settings.AddConfigEntry("session.load_model_format", "ONNX");
    settings.SetGraphOptimizationLevel(
        o.disable_graph_optimizations ? ORT_DISABLE_ALL : ORT_ENABLE_ALL);
    session = Ort::Session(environment, bytes.data(), bytes.size(), settings);
    const auto ni = session.GetInputCount(), no = session.GetOutputCount();
    if (!ni || !no || ni > o.max_io_tensors || no > o.max_io_tensors - ni ||
        session.GetOverridableInitializerCount())
      throw CompatibilityError("Typed ORT requires bounded immutable I/O");
    std::set<std::string> symbols, used_limits;
    for (bool input : {true, false}) {
      auto remaining = input ? o.max_input_bytes : o.max_output_bytes;
      auto& list = input ? inputs : outputs;
      std::set<std::string> names;
      for (std::size_t i = 0; i < (input ? ni : no); ++i) {
        auto item = Signature(session, input, i);
        if (!names.insert(item.name).second)
          throw CompatibilityError("Duplicate typed ORT I/O name");
        auto shape = item.shape;
        if (!item.leading_symbol.empty()) {
          if (input) {
            symbols.insert(item.leading_symbol);
            shape[0] = 1;
          } else if (symbols.contains(item.leading_symbol))
            shape[0] = 1;
          else {
            auto bound = o.maximum_output_rows.find(item.name);
            if (bound == o.maximum_output_rows.end() || !bound->second ||
                bound->second > 1000000)
              throw CompatibilityError(
                  "Unbound typed output needs maximum_output_rows");
            shape[0] = static_cast<std::int64_t>(bound->second);
            used_limits.insert(item.name);
          }
        }
        auto count = Elements(shape, remaining / Width(item.type));
        remaining -= count * Width(item.type);
        if (input &&
            static_cast<std::uint64_t>(shape.front()) > o.max_batch_size)
          throw ResourceExhausted("Typed ORT fixed batch exceeds bound");
        list.push_back(std::move(item));
      }
    }
    if (used_limits.size() != o.maximum_output_rows.size())
      throw InvalidArgument(
          "Typed ORT output limit refers to a fixed or input-bound output");
    for (const auto& item : inputs) input_names.push_back(item.name.c_str());
    for (const auto& item : outputs) output_names.push_back(item.name.c_str());
  }
};
TypedSession::TypedSession(std::span<const std::byte> bytes,
                           const ArtifactDigest& expected,
                           const TypedSessionOptions& o) {
  validate_artifact_digest(expected);
  if (!o.cpu_threads || o.cpu_threads > 256 || !o.max_io_tensors ||
      o.max_io_tensors > 256 || !o.max_model_bytes ||
      o.max_model_bytes > (4ULL << 30U) || !o.max_input_bytes ||
      o.max_input_bytes > (4ULL << 30U) || !o.max_output_bytes ||
      o.max_output_bytes > (4ULL << 30U) || !o.max_batch_size ||
      o.max_batch_size > 1048576)
    throw InvalidArgument("Invalid typed ORT limits");
  if (bytes.empty() || bytes.size() > o.max_model_bytes)
    throw ResourceExhausted("Typed ORT model byte bound exceeded");
  if (sha256_hex(bytes) != expected.hex)
    throw CompatibilityError("Typed ORT model SHA-256 mismatch");
  try {
    impl_ = std::make_unique<Impl>(bytes, o);
  } catch (const Ort::Exception& error) {
    throw CompatibilityError(std::string("Typed ORT load: ") + error.what());
  }
}
TypedSession::~TypedSession() = default;
const std::vector<TypedTensorSignature>& TypedSession::inputs() const noexcept {
  return impl_->inputs;
}
const std::vector<TypedTensorSignature>& TypedSession::outputs()
    const noexcept {
  return impl_->outputs;
}
std::vector<TypedTensorOutput> TypedSession::run(
    std::span<const TypedTensorInput> inputs, std::stop_token stop) const {
  if (stop.stop_requested()) throw Cancelled("Typed ORT run cancelled");
  if (inputs.size() != impl_->inputs.size())
    throw InvalidArgument("Typed ORT needs every input once");
  try {
    std::vector<const TypedTensorInput*> ordered(inputs.size());
    std::map<std::string, std::int64_t> symbols;
    auto remaining = impl_->options.max_input_bytes;
    for (const auto& input : inputs) {
      auto it =
          std::find_if(impl_->inputs.begin(), impl_->inputs.end(),
                       [&](const auto& s) { return s.name == input.name; });
      if (it == impl_->inputs.end() || ordered[it - impl_->inputs.begin()])
        throw InvalidArgument("Unknown or repeated typed ORT input");
      if (input.shape.size() != it->shape.size() || input.shape.front() < 1 ||
          static_cast<std::uint64_t>(input.shape.front()) >
              impl_->options.max_batch_size)
        throw InvalidArgument("Typed ORT input shape/batch differs");
      for (std::size_t i = 0; i < input.shape.size(); ++i)
        if (it->shape[i] != -1 && input.shape[i] != it->shape[i])
          throw InvalidArgument("Typed ORT fixed input dimension differs");
      if (!it->leading_symbol.empty()) {
        auto [where, inserted] =
            symbols.emplace(it->leading_symbol, input.shape[0]);
        if (!inserted && where->second != input.shape[0])
          throw InvalidArgument("Typed ORT input symbols disagree");
      }
      const auto count = Elements(input.shape, remaining / Width(it->type));
      remaining -= count * Width(it->type);
      std::visit(
          [&](auto values) {
            using T =
                std::remove_const_t<typename decltype(values)::element_type>;
            if (Type<T>() != it->type || values.size() != count)
              throw InvalidArgument("Typed ORT input dtype/count differs");
            if constexpr (std::is_same_v<T, float>)
              for (auto v : values)
                if (!std::isfinite(v))
                  throw InvalidArgument("Typed ORT input is nonfinite");
          },
          input.values);
      ordered[it - impl_->inputs.begin()] = &input;
    }
    // Known input-bound outputs can be charged before ORT allocates them.
    // Unbound dynamic outputs use their explicit maximum and are rechecked
    // against the actual returned rows after Run.
    auto output_remaining = impl_->options.max_output_bytes;
    for (const auto& output : impl_->outputs) {
      auto shape = output.shape;
      if (!output.leading_symbol.empty()) {
        auto bound = symbols.find(output.leading_symbol);
        shape[0] =
            bound != symbols.end()
                ? bound->second
                : static_cast<std::int64_t>(
                      impl_->options.maximum_output_rows.at(output.name));
      }
      const auto count = Elements(shape, output_remaining / Width(output.type));
      output_remaining -= count * Width(output.type);
    }
    auto memory =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> values;
    for (auto input : ordered)
      std::visit(
          [&](auto data) {
            using T =
                std::remove_const_t<typename decltype(data)::element_type>;
            values.push_back(Ort::Value::CreateTensor<T>(
                memory, const_cast<T*>(data.data()), data.size(),
                input->shape.data(), input->shape.size()));
          },
          input->values);
    Ort::RunOptions run;
    std::stop_callback cancellation(stop, [&]() {
      try {
        run.SetTerminate();
      } catch (...) {
      }
    });
    std::vector<Ort::Value> outputs;
    try {
      outputs = impl_->session.Run(
          run, impl_->input_names.data(), values.data(), values.size(),
          impl_->output_names.data(), impl_->output_names.size());
    } catch (const Ort::Exception&) {
      if (stop.stop_requested()) throw Cancelled("Typed ORT run cancelled");
      throw;
    }
    if (stop.stop_requested()) throw Cancelled("Typed ORT run cancelled");
    std::vector<TypedTensorOutput> result;
    remaining = impl_->options.max_output_bytes;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      const auto& expected = impl_->outputs[i];
      auto info = outputs[i].GetTensorTypeAndShapeInfo();
      auto shape = info.GetShape();
      if (static_cast<int>(info.GetElementType()) !=
              static_cast<int>(expected.type) ||
          shape.size() != expected.shape.size())
        throw CompatibilityError("Typed ORT output type/rank changed");
      for (std::size_t axis = 0; axis < shape.size(); ++axis)
        if (expected.shape[axis] != -1 && shape[axis] != expected.shape[axis])
          throw CompatibilityError("Typed ORT output dimension changed");
      if (!expected.leading_symbol.empty()) {
        if (auto it = symbols.find(expected.leading_symbol);
            it != symbols.end()) {
          if (shape[0] != it->second)
            throw CompatibilityError(
                "Typed ORT output symbol differs from input");
        } else if (shape[0] < 0 ||
                   static_cast<std::uint64_t>(shape[0]) >
                       impl_->options.maximum_output_rows.at(expected.name))
          throw ResourceExhausted("Typed ORT dynamic output exceeds row bound");
      }
      const auto count = Elements(shape, remaining / Width(expected.type),
                                  !expected.leading_symbol.empty());
      remaining -= count * Width(expected.type);
      TypedTensorOutput item{expected.name, std::move(shape), {}};
      auto copy = [&]<class T>() {
        std::vector<T> data(count);
        if (count)
          std::memcpy(data.data(), outputs[i].GetTensorData<T>(),
                      count * sizeof(T));
        if constexpr (std::is_same_v<T, float>)
          for (auto v : data)
            if (!std::isfinite(v))
              throw NumericalError("Typed ORT output is nonfinite");
        item.values = std::move(data);
      };
      if (expected.type == TensorElementType::float32)
        copy.operator()<float>();
      else if (expected.type == TensorElementType::int32)
        copy.operator()<std::int32_t>();
      else
        copy.operator()<std::int64_t>();
      result.push_back(std::move(item));
    }
    return result;
  } catch (const Ort::Exception& error) {
    throw BackendError(std::string("Typed ORT run: ") + error.what());
  }
}
}  // namespace jyppx::liuding::onnx_runtime
