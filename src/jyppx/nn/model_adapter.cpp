// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/nn/model_adapter.hpp>

#include <utility>

namespace jyppx::nn {
namespace {

constexpr char kComponent[] = "nn.model_adapter";

[[noreturn]] void throw_missing_object(const char* operation,
                                       const char* expected,
                                       const char* actual, const char* hint) {
  throw InvalidArgument(
      "Model adapter construction failed",
      ErrorDetails{.component = kComponent,
                   .operation = operation,
                   .expected = expected,
                   .actual = actual,
                   .hint = hint});
}

}  // namespace

ModelStateView::ModelStateView(std::shared_ptr<torch::nn::Module> module)
    : module_(std::move(module)) {
  if (!module_) {
    throw_missing_object("ModelStateView", "a non-null shared module", "null",
                         "Construct the module before creating a state view.");
  }
}

torch::nn::Module& ModelStateView::module() noexcept { return *module_; }

const torch::nn::Module& ModelStateView::module() const noexcept {
  return *module_;
}

const std::shared_ptr<torch::nn::Module>& ModelStateView::shared_module()
    const noexcept {
  return module_;
}

std::vector<Tensor> ModelStateView::parameters(bool recurse) const {
  return module_->parameters(recurse);
}

torch::OrderedDict<std::string, Tensor> ModelStateView::named_parameters(
    bool recurse) const {
  return module_->named_parameters(recurse);
}

std::vector<Tensor> ModelStateView::buffers(bool recurse) const {
  return module_->buffers(recurse);
}

torch::OrderedDict<std::string, Tensor> ModelStateView::named_buffers(
    bool recurse) const {
  return module_->named_buffers(recurse);
}

void ModelStateView::train(bool on) { module_->train(on); }

void ModelStateView::eval() { module_->eval(); }

bool ModelStateView::is_training() const noexcept {
  return module_->is_training();
}

void ModelStateView::to(const Device& device, bool non_blocking) {
  module_->to(device, non_blocking);
}

void ModelStateView::to(Dtype dtype, bool non_blocking) {
  module_->to(dtype, non_blocking);
}

void ModelStateView::to(const Device& device, Dtype dtype, bool non_blocking) {
  module_->to(device, dtype, non_blocking);
}

ModelAdapter::ModelAdapter(std::shared_ptr<ModelStateView> state,
                           ModelInvoker invoker,
                           InvocationSchema input_schema,
                           ValueSchema output_schema)
    : state_(std::move(state)),
      invoker_(std::move(invoker)),
      input_schema_(std::move(input_schema)),
      output_schema_(std::move(output_schema)) {
  if (!state_) {
    throw_missing_object("state", "a non-null ModelStateView", "null",
                         "Create the state view from the model being invoked.");
  }
  if (!invoker_) {
    throw_missing_object("invoker", "a non-empty ModelInvoker",
                         "empty callable",
                         "Provide the model's explicit forward operation.");
  }
  validate_invocation_schema(input_schema_);
  validate_value_schema(output_schema_, "output");
}

ModelStateView& ModelAdapter::state() noexcept { return *state_; }

const ModelStateView& ModelAdapter::state() const noexcept { return *state_; }

InvocationResult ModelAdapter::invoke(const Invocation& input) const {
  validate_invocation(input, input_schema_);
  try {
    auto value = invoker_(input);
    validate_value(value, output_schema_, "output");
    return InvocationResult{.value = std::move(value),
                            .schema = output_schema_};
  } catch (const Error&) {
    throw;
  } catch (const c10::Error& cause) {
    detail::rethrow_backend_error(
        cause,
        detail::BackendErrorContext{
            .component = kComponent,
            .operation = "model invocation",
            .failure_kind = detail::BackendFailureKind::unknown,
            .expected = "an invocation accepted by the active LibTorch backend",
            .hint = "Inspect the nested backend error and model implementation."});
  }
}

const InvocationSchema& ModelAdapter::input_schema() const noexcept {
  return input_schema_;
}

const ValueSchema& ModelAdapter::output_schema() const noexcept {
  return output_schema_;
}

ModelAdapter adapt_module(std::shared_ptr<torch::nn::Module> module,
                          ModelInvoker invoker,
                          InvocationSchema input_schema,
                          ValueSchema output_schema) {
  if (!module) {
    throw_missing_object("adapt_module", "a non-null shared module", "null",
                         "Construct the module before adapting it.");
  }
  if (!invoker) {
    throw_missing_object("adapt_module", "a non-empty ModelInvoker",
                         "empty callable",
                         "Provide the model's explicit forward operation.");
  }

  auto state = std::make_shared<ModelStateView>(module);
  ModelInvoker retained_invoker =
      [module = std::move(module), invoker = std::move(invoker)](
          const Invocation& input) mutable {
        static_cast<void>(module);
        return invoker(input);
      };
  return ModelAdapter(std::move(state), std::move(retained_invoker),
                      std::move(input_schema), std::move(output_schema));
}

}  // namespace jyppx::nn
