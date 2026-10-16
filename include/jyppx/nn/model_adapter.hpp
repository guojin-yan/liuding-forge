// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_NN_MODEL_ADAPTER_HPP
#define JYPPX_NN_MODEL_ADAPTER_HPP

/**
 * @file model_adapter.hpp
 * @brief Defines model state ownership and schema-checked invocation adapters.
 */

#include <jyppx/foundation/error.hpp>
#include <jyppx/nn/invocation.hpp>

#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>

#include <concepts>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace jyppx::nn {

/**
 * @brief Matches concrete implementations derived from `torch::nn::Module`.
 * @tparam Module Candidate concrete module implementation type.
 */
template <typename Module>
concept ModuleImplementation =
    std::derived_from<std::remove_cv_t<Module>, torch::nn::Module>;

/**
 * @brief Matches LibTorch holder types that expose a shared module pointer.
 * @tparam Holder Candidate `TORCH_MODULE` or compatible holder type.
 */
template <typename Holder>
concept ModuleHolderLike = requires(const Holder& holder) {
  typename Holder::ContainedType;
  requires ModuleImplementation<typename Holder::ContainedType>;
  { static_cast<bool>(holder) } -> std::same_as<bool>;
  {
    holder.ptr()
  } -> std::convertible_to<std::shared_ptr<typename Holder::ContainedType>>;
};

/**
 * @brief Matches callables that convert a structured invocation into an
 * IValue-compatible result.
 * @tparam Forward Candidate forwarding callable type.
 * @tparam Module Concrete module implementation passed to the callable.
 */
template <typename Forward, typename Module>
concept InvocationForward =
    ModuleImplementation<Module> &&
    requires(Forward& forward, Module& module, const Invocation& input) {
      c10::IValue(std::invoke(forward, module, input));
    };

/**
 * @brief Matches concrete modules exposing an IValue-compatible
 * `forward(const Tensor&)` operation.
 * @tparam Module Candidate concrete module implementation type.
 */
template <typename Module>
concept TensorForwardModule = ModuleImplementation<Module> &&
                              requires(Module& module, const Tensor& input) {
                                c10::IValue(module.forward(input));
                              };

/**
 * @brief Matches module holders whose implementation satisfies the common
 * tensor-forward contract.
 * @tparam Holder Candidate `TORCH_MODULE` or compatible holder type.
 */
template <typename Holder>
concept TensorForwardHolder =
    ModuleHolderLike<Holder> &&
    TensorForwardModule<typename Holder::ContainedType>;

/**
 * @brief Shared-ownership view of a LibTorch module's mutable state.
 *
 * The view owns a shared pointer, so every tensor and mode operation remains
 * valid for at least the lifetime of the view. It deliberately does not infer
 * or invoke `forward()`, because LibTorch modules have no common virtual
 * forward signature. Concurrent mutation is not safe without caller-provided
 * synchronization.
 */
class ModelStateView final {
 public:
  /**
   * @brief Creates a state view that shares ownership of a module.
   * @param module Non-null module to retain.
   * @throws InvalidArgument If `module` is null.
   */
  explicit ModelStateView(std::shared_ptr<torch::nn::Module> module);

  /**
   * @brief Returns mutable access to the retained module.
   * @return A reference valid for the lifetime of this view.
   */
  torch::nn::Module& module() noexcept;

  /**
   * @brief Returns read-only access to the retained module.
   * @return A reference valid for the lifetime of this view.
   */
  const torch::nn::Module& module() const noexcept;

  /**
   * @brief Returns the shared module handle retained by the view.
   * @return A non-null shared pointer reference.
   */
  const std::shared_ptr<torch::nn::Module>& shared_module() const noexcept;

  /**
   * @brief Returns module parameters in LibTorch traversal order.
   * @param recurse Whether to include parameters from child modules.
   * @return Tensor handles that alias the module's current parameters.
   */
  std::vector<Tensor> parameters(bool recurse = true) const;

  /**
   * @brief Returns named parameters in deterministic LibTorch order.
   * @param recurse Whether to include parameters from child modules.
   * @return An ordered dictionary of stable state keys and tensor handles.
   */
  torch::OrderedDict<std::string, Tensor> named_parameters(
      bool recurse = true) const;

  /**
   * @brief Returns module buffers in LibTorch traversal order.
   * @param recurse Whether to include buffers from child modules.
   * @return Tensor handles that alias the module's current buffers.
   */
  std::vector<Tensor> buffers(bool recurse = true) const;

  /**
   * @brief Returns named buffers in deterministic LibTorch order.
   * @param recurse Whether to include buffers from child modules.
   * @return An ordered dictionary of stable state keys and tensor handles.
   */
  torch::OrderedDict<std::string, Tensor> named_buffers(
      bool recurse = true) const;

  /**
   * @brief Selects training or evaluation mode recursively.
   * @param on `true` for training mode and `false` for evaluation mode.
   */
  void train(bool on = true);

  /** @brief Selects evaluation mode recursively. */
  void eval();

  /**
   * @brief Reports the module's current mode.
   * @return `true` in training mode, otherwise `false`.
   */
  bool is_training() const noexcept;

  /**
   * @brief Moves parameters and buffers to a device.
   * @param device Destination device.
   * @param non_blocking Whether eligible transfers may execute asynchronously.
   */
  void to(const Device& device, bool non_blocking = false);

  /**
   * @brief Converts floating parameters and buffers to a scalar type.
   * @param dtype Destination scalar type.
   * @param non_blocking Whether eligible conversions may execute
   * asynchronously.
   */
  void to(Dtype dtype, bool non_blocking = false);

  /**
   * @brief Moves and converts parameters and buffers in one operation.
   * @param device Destination device.
   * @param dtype Destination scalar type.
   * @param non_blocking Whether eligible transfers may execute asynchronously.
   */
  void to(const Device& device, Dtype dtype, bool non_blocking = false);

 private:
  std::shared_ptr<torch::nn::Module>
      module_;  ///< Shared module whose state and lifetime are exposed.
};

/**
 * @brief Type-erased callable that maps one invocation to one IValue result.
 *
 * The callable must capture or otherwise retain every object required by its
 * forward implementation. Adapter factories additionally retain the state
 * module even when the callable does not capture it.
 */
using ModelInvoker = std::function<c10::IValue(const Invocation&)>;

/**
 * @brief Couples model state, invocation behavior, and validated schemas.
 *
 * The adapter validates all inputs before calling user code and validates the
 * result before returning it. The class provides no internal serialization;
 * concurrent invocation is safe only when the supplied invoker and module are
 * safe for the same access pattern.
 */
class ModelAdapter final {
 public:
  /**
   * @brief Creates a schema-checked model adapter.
   * @param state Non-null shared state view.
   * @param invoker Non-empty callable implementing model execution.
   * @param input_schema Positional and keyword input contract.
   * @param output_schema Result value contract.
   * @throws InvalidArgument If an object is missing or a schema is malformed.
   */
  ModelAdapter(std::shared_ptr<ModelStateView> state, ModelInvoker invoker,
               InvocationSchema input_schema, ValueSchema output_schema);

  /**
   * @brief Returns mutable model state access.
   * @return The retained state view.
   */
  ModelStateView& state() noexcept;

  /**
   * @brief Returns read-only model state access.
   * @return The retained state view.
   */
  const ModelStateView& state() const noexcept;

  /**
   * @brief Validates and executes one model invocation.
   * @param input Positional and keyword arguments.
   * @return The validated result and declared output schema.
   * @throws InvalidArgument If the input or result violates its schema.
   * @throws BackendError If LibTorch raises an unclassified backend failure.
   */
  InvocationResult invoke(const Invocation& input) const;

  /**
   * @brief Returns the adapter's immutable input contract.
   * @return A reference valid for the lifetime of this adapter.
   */
  const InvocationSchema& input_schema() const noexcept;

  /**
   * @brief Returns the adapter's immutable output contract.
   * @return A reference valid for the lifetime of this adapter.
   */
  const ValueSchema& output_schema() const noexcept;

 private:
  std::shared_ptr<ModelStateView>
      state_;             ///< Shared state boundary and module lifetime owner.
  ModelInvoker invoker_;  ///< User-supplied type-erased forward operation.
  InvocationSchema input_schema_;  ///< Validated call input contract.
  ValueSchema output_schema_;      ///< Validated call result contract.
};

/**
 * @brief Adapts a shared base module and a fully type-erased invoker.
 * @param module Non-null module retained by the adapter.
 * @param invoker Non-empty invocation callable.
 * @param input_schema Positional and keyword input contract.
 * @param output_schema Result value contract.
 * @return An owning, schema-checked model adapter.
 * @throws InvalidArgument If the module, callable, or schema is invalid.
 */
ModelAdapter adapt_module(std::shared_ptr<torch::nn::Module> module,
                          ModelInvoker invoker, InvocationSchema input_schema,
                          ValueSchema output_schema);

/**
 * @brief Adapts a concrete shared module with a typed forwarding callable.
 *
 * The callable receives `Module&` followed by the validated invocation. The
 * returned value must be constructible as `c10::IValue`.
 *
 * @tparam Module Concrete LibTorch module implementation type.
 * @tparam Forward Callable type used for model execution.
 * @param module Non-null shared module retained by state and invoker.
 * @param forward Callable implementing the concrete forward contract.
 * @param input_schema Positional and keyword input contract.
 * @param output_schema Result value contract.
 * @return An owning, schema-checked model adapter.
 * @throws InvalidArgument If the module, callable, or schema is invalid.
 */
template <ModuleImplementation Module, InvocationForward<Module> Forward>
ModelAdapter adapt_module(std::shared_ptr<Module> module, Forward forward,
                          InvocationSchema input_schema,
                          ValueSchema output_schema) {
  if (!module) {
    throw InvalidArgument(
        "Cannot adapt a null model",
        ErrorDetails{.component = "nn.model_adapter",
                     .operation = "adapt_module",
                     .expected = "a non-null shared module",
                     .actual = "null",
                     .hint = "Construct the module before adapting it."});
  }
  ModelInvoker invoker =
      [module, forward = std::move(forward)](const Invocation& input) mutable {
        return c10::IValue(std::invoke(forward, *module, input));
      };
  return adapt_module(std::static_pointer_cast<torch::nn::Module>(module),
                      std::move(invoker), std::move(input_schema),
                      std::move(output_schema));
}

/**
 * @brief Adapts a LibTorch holder with a typed forwarding callable.
 * @tparam Holder `TORCH_MODULE` or compatible holder type.
 * @tparam Forward Callable receiving the holder's concrete module type.
 * @param holder Non-empty holder whose shared implementation is retained.
 * @param forward Callable implementing the concrete forward contract.
 * @param input_schema Positional and keyword input contract.
 * @param output_schema Result value contract.
 * @return An owning, schema-checked model adapter.
 * @throws InvalidArgument If the holder is empty or a schema is invalid.
 */
template <ModuleHolderLike Holder,
          InvocationForward<typename Holder::ContainedType> Forward>
ModelAdapter adapt_holder(Holder holder, Forward forward,
                          InvocationSchema input_schema,
                          ValueSchema output_schema) {
  if (!static_cast<bool>(holder)) {
    throw InvalidArgument(
        "Cannot adapt an empty module holder",
        ErrorDetails{.component = "nn.model_adapter",
                     .operation = "adapt_holder",
                     .expected = "a holder containing a module",
                     .actual = "empty holder",
                     .hint = "Initialize the holder before adapting it."});
  }
  return adapt_module(holder.ptr(), std::move(forward), std::move(input_schema),
                      std::move(output_schema));
}

/**
 * @brief Adapts a common `Tensor -> Tensor` concrete module.
 * @tparam Module Concrete module exposing `forward(const Tensor&)`.
 * @param module Non-null shared module retained by the adapter.
 * @param input_constraints Constraints for the sole input tensor.
 * @param output_constraints Constraints for the output tensor.
 * @return An adapter with one positional tensor and no keywords.
 */
template <TensorForwardModule Module>
ModelAdapter adapt_tensor_module(std::shared_ptr<Module> module,
                                 TensorSchema input_constraints = {},
                                 TensorSchema output_constraints = {}) {
  InvocationSchema input_schema;
  input_schema.positional.push_back(
      ValueSchema::tensor(std::move(input_constraints), "input"));
  auto output_schema =
      ValueSchema::tensor(std::move(output_constraints), "output");
  return adapt_module(
      std::move(module),
      [](Module& concrete, const Invocation& input) {
        return concrete.forward(input.positional.front().toTensor());
      },
      std::move(input_schema), std::move(output_schema));
}

/**
 * @brief Adapts a common `Tensor -> Tensor` LibTorch holder.
 * @tparam Holder Holder whose implementation exposes `forward(const Tensor&)`.
 * @param holder Non-empty holder retained through its shared implementation.
 * @param input_constraints Constraints for the sole input tensor.
 * @param output_constraints Constraints for the output tensor.
 * @return An adapter with one positional tensor and no keywords.
 */
template <TensorForwardHolder Holder>
ModelAdapter adapt_tensor_holder(Holder holder,
                                 TensorSchema input_constraints = {},
                                 TensorSchema output_constraints = {}) {
  if (!static_cast<bool>(holder)) {
    throw InvalidArgument(
        "Cannot adapt an empty module holder",
        ErrorDetails{.component = "nn.model_adapter",
                     .operation = "adapt_tensor_holder",
                     .expected = "a holder containing a module",
                     .actual = "empty holder",
                     .hint = "Initialize the holder before adapting it."});
  }
  return adapt_tensor_module(holder.ptr(), std::move(input_constraints),
                             std::move(output_constraints));
}

}  // namespace jyppx::nn

#endif  // JYPPX_NN_MODEL_ADAPTER_HPP
