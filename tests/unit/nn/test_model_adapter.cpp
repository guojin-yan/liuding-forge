// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/nn/model_adapter.hpp>

#include <cstdint>
#include <exception>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

class ProbeModelImpl final : public torch::nn::Module {
 public:
  ProbeModelImpl()
      : linear(register_module("linear", torch::nn::Linear(2, 1))),
        offset(register_buffer("offset", torch::zeros({1}))) {}

  torch::Tensor forward(const torch::Tensor& input) {
    return linear->forward(input) + offset;
  }

  torch::nn::Linear linear;
  torch::Tensor offset;
};

TORCH_MODULE(ProbeModel);

static_assert(jyppx::liuding::nn::ModuleImplementation<ProbeModelImpl>);
static_assert(jyppx::liuding::nn::ModuleHolderLike<ProbeModel>);

jyppx::liuding::TensorSchema matrix_schema(
    std::int64_t columns, torch::Dtype dtype = torch::kFloat32) {
  jyppx::liuding::TensorSchema schema;
  schema.dtype = dtype;
  schema.rank = 2;
  schema.shape =
      std::vector<std::optional<std::int64_t>>{std::nullopt, columns};
  return schema;
}

TEST(InvocationSchemaTest, RejectsMalformedTensorAndListSchemas) {
  auto tensor = jyppx::liuding::ValueSchema::tensor();
  tensor.tensor_constraints->rank = 2;
  tensor.tensor_constraints->shape =
      std::vector<std::optional<std::int64_t>>{3};
  EXPECT_THROW(jyppx::liuding::validate_value_schema(tensor),
               jyppx::liuding::InvalidArgument);

  jyppx::liuding::ValueSchema list;
  list.kind = jyppx::liuding::ValueKind::list;
  EXPECT_THROW(jyppx::liuding::validate_value_schema(list),
               jyppx::liuding::InvalidArgument);
}

TEST(InvocationSchemaTest, ValidatesNestedTupleListAndDictionaryValues) {
  auto optional_note = jyppx::liuding::ValueSchema{};
  optional_note.kind = jyppx::liuding::ValueKind::string;
  optional_note.required = false;

  const auto output_schema = jyppx::liuding::ValueSchema::dictionary(
      {{"scores", jyppx::liuding::ValueSchema::tensor(matrix_schema(2))},
       {"note", optional_note}},
      false, "prediction");
  jyppx::liuding::validate_value_schema(output_schema);

  c10::Dict<std::string, torch::Tensor> output;
  output.insert("scores", torch::ones({3, 2}));
  EXPECT_NO_THROW(
      jyppx::liuding::validate_value(c10::IValue(output), output_schema));

  c10::Dict<std::string, torch::Tensor> unexpected;
  unexpected.insert("scores", torch::ones({3, 2}));
  unexpected.insert("extra", torch::zeros({1}));
  EXPECT_THROW(
      jyppx::liuding::validate_value(c10::IValue(unexpected), output_schema),
      jyppx::liuding::InvalidArgument);

  const auto tuple_schema = jyppx::liuding::ValueSchema::tuple(
      {jyppx::liuding::ValueSchema::list(
           jyppx::liuding::ValueSchema::tensor(matrix_schema(2))),
       jyppx::liuding::ValueSchema{.kind = jyppx::liuding::ValueKind::integer}},
      "batched_result");
  c10::impl::GenericList tensors(c10::TensorType::get());
  tensors.push_back(torch::ones({2, 2}));
  tensors.push_back(torch::zeros({1, 2}));
  const auto tuple = c10::ivalue::Tuple::create(
      {c10::IValue(tensors), c10::IValue(std::int64_t{2})});
  EXPECT_NO_THROW(
      jyppx::liuding::validate_value(c10::IValue(tuple), tuple_schema));
}

TEST(InvocationSchemaTest, ChecksArgumentsBeforeCallingTheModel) {
  auto model = std::make_shared<ProbeModelImpl>();
  int calls = 0;
  jyppx::liuding::nn::InvocationSchema input_schema;
  input_schema.positional.push_back(
      jyppx::liuding::ValueSchema::tensor(matrix_schema(2), "features"));
  auto adapter = jyppx::liuding::nn::adapt_module(
      model,
      [&calls](ProbeModelImpl& concrete,
               const jyppx::liuding::nn::Invocation& input) {
        ++calls;
        return concrete.forward(input.positional.front().toTensor());
      },
      input_schema,
      jyppx::liuding::ValueSchema::tensor(matrix_schema(1), "prediction"));

  EXPECT_THROW(adapter.invoke({.positional = {torch::ones({4, 3})}}),
               jyppx::liuding::InvalidArgument);
  EXPECT_EQ(calls, 0);
  EXPECT_THROW(adapter.invoke({.positional = {torch::ones({4, 2})},
                               .keyword = {{"unexpected", 1}}}),
               jyppx::liuding::InvalidArgument);
  EXPECT_EQ(calls, 0);
}

TEST(ModelStateViewTest,
     ExposesStateModeAndConversionWithoutForwardAssumptions) {
  auto model = std::make_shared<ProbeModelImpl>();
  jyppx::liuding::nn::ModelStateView state(model);

  EXPECT_EQ(state.parameters().size(), 2);
  EXPECT_TRUE(state.named_parameters().contains("linear.weight"));
  EXPECT_TRUE(state.named_parameters().contains("linear.bias"));
  EXPECT_TRUE(state.named_buffers().contains("offset"));
  EXPECT_TRUE(state.is_training());
  state.eval();
  EXPECT_FALSE(state.is_training());
  state.train();
  EXPECT_TRUE(state.is_training());

  state.to(torch::kFloat64);
  EXPECT_EQ(state.parameters().front().scalar_type(), torch::kFloat64);
  EXPECT_EQ(state.buffers().front().scalar_type(), torch::kFloat64);
  state.to(torch::Device(torch::kCPU), torch::kFloat32);
  EXPECT_EQ(state.parameters().front().scalar_type(), torch::kFloat32);
}

TEST(ModelAdapterTest, TensorHelperOwnsConcreteModuleAndValidatesResult) {
  auto model = std::make_shared<ProbeModelImpl>();
  const std::weak_ptr<ProbeModelImpl> lifetime = model;
  auto adapter = jyppx::liuding::nn::adapt_tensor_module(
      model, matrix_schema(2), matrix_schema(1));
  model.reset();

  ASSERT_FALSE(lifetime.expired());
  const auto result = adapter.invoke({.positional = {torch::ones({5, 2})}});
  EXPECT_TRUE(result.value.isTensor());
  EXPECT_EQ(result.value.toTensor().sizes().vec(),
            std::vector<std::int64_t>({5, 1}));
  EXPECT_EQ(result.schema.kind, jyppx::liuding::ValueKind::tensor);
}

TEST(ModelAdapterTest, TensorHolderHelperRetainsTheImplementation) {
  ProbeModel holder;
  const std::weak_ptr<ProbeModelImpl> lifetime = holder.ptr();
  auto adapter = jyppx::liuding::nn::adapt_tensor_holder(
      holder, matrix_schema(2), matrix_schema(1));
  holder = nullptr;

  ASSERT_FALSE(lifetime.expired());
  const auto result = adapter.invoke({.positional = {torch::zeros({2, 2})}});
  EXPECT_EQ(result.value.toTensor().sizes().vec(),
            std::vector<std::int64_t>({2, 1}));
}

TEST(ModelAdapterTest, BaseModuleFactoryRetainsStateForARawTypedInvoker) {
  auto concrete = std::make_shared<ProbeModelImpl>();
  const std::weak_ptr<ProbeModelImpl> lifetime = concrete;
  ProbeModelImpl* const raw = concrete.get();
  std::shared_ptr<torch::nn::Module> base = concrete;
  concrete.reset();

  jyppx::liuding::nn::InvocationSchema input_schema;
  input_schema.positional.push_back(
      jyppx::liuding::ValueSchema::tensor(matrix_schema(2)));
  auto adapter = jyppx::liuding::nn::adapt_module(
      std::move(base),
      jyppx::liuding::nn::ModelInvoker(
          [raw](const jyppx::liuding::nn::Invocation& input) {
            return raw->forward(input.positional.front().toTensor());
          }),
      std::move(input_schema),
      jyppx::liuding::ValueSchema::tensor(matrix_schema(1)));

  ASSERT_FALSE(lifetime.expired());
  EXPECT_TRUE(
      adapter.invoke({.positional = {torch::ones({1, 2})}}).value.isTensor());
}

TEST(ModelAdapterTest, SupportsMultipleInputsAndTupleOutputs) {
  auto model = std::make_shared<ProbeModelImpl>();
  jyppx::liuding::nn::InvocationSchema input_schema;
  input_schema.positional = {
      jyppx::liuding::ValueSchema::tensor(matrix_schema(2), "left"),
      jyppx::liuding::ValueSchema::tensor(matrix_schema(2), "right")};
  auto adapter = jyppx::liuding::nn::adapt_module(
      model,
      [](ProbeModelImpl&, const jyppx::liuding::nn::Invocation& input) {
        const auto left = input.positional[0].toTensor();
        const auto right = input.positional[1].toTensor();
        return c10::IValue(c10::ivalue::Tuple::create(
            {c10::IValue(left + right), c10::IValue(left - right)}));
      },
      std::move(input_schema),
      jyppx::liuding::ValueSchema::tuple(
          {jyppx::liuding::ValueSchema::tensor(matrix_schema(2), "sum"),
           jyppx::liuding::ValueSchema::tensor(matrix_schema(2),
                                               "difference")}));

  const auto result = adapter.invoke(
      {.positional = {torch::ones({3, 2}), torch::full({3, 2}, 2.0)}});
  ASSERT_TRUE(result.value.isTuple());
  const auto& values = result.value.toTupleRef().elements();
  EXPECT_TRUE(torch::equal(values[0].toTensor(), torch::full({3, 2}, 3.0)));
  EXPECT_TRUE(torch::equal(values[1].toTensor(), torch::full({3, 2}, -1.0)));
}

TEST(ModelAdapterTest, RejectsAnOutputThatViolatesTheDeclaredSchema) {
  auto model = std::make_shared<ProbeModelImpl>();
  jyppx::liuding::nn::InvocationSchema input_schema;
  input_schema.positional.push_back(
      jyppx::liuding::ValueSchema::tensor(matrix_schema(2)));
  auto adapter = jyppx::liuding::nn::adapt_module(
      model,
      [](ProbeModelImpl&, const jyppx::liuding::nn::Invocation&) {
        return c10::IValue(std::int64_t{7});
      },
      std::move(input_schema),
      jyppx::liuding::ValueSchema::tensor(matrix_schema(1)));

  EXPECT_THROW(adapter.invoke({.positional = {torch::ones({1, 2})}}),
               jyppx::liuding::InvalidArgument);
}

TEST(ModelAdapterTest, PreservesNestedLibTorchFailureContext) {
  auto model = std::make_shared<ProbeModelImpl>();
  auto adapter = jyppx::liuding::nn::adapt_module(
      model,
      [](ProbeModelImpl&, const jyppx::liuding::nn::Invocation&) {
        return c10::IValue(
            torch::matmul(torch::ones({2, 3}), torch::ones({5, 4})));
      },
      jyppx::liuding::nn::InvocationSchema{},
      jyppx::liuding::ValueSchema::tensor());

  try {
    static_cast<void>(adapter.invoke({}));
    FAIL() << "Expected LibTorch matmul to reject incompatible shapes";
  } catch (const jyppx::liuding::BackendError& converted) {
    EXPECT_EQ(converted.details().component, "nn.model_adapter");
    EXPECT_EQ(converted.details().operation, "model invocation");
    EXPECT_THROW(std::rethrow_if_nested(converted), c10::Error);
  }
}

TEST(ModelAdapterTest, RejectsMissingModulesHoldersAndInvokers) {
  EXPECT_THROW(
      jyppx::liuding::nn::adapt_module(
          std::shared_ptr<torch::nn::Module>{},
          [](const jyppx::liuding::nn::Invocation&) { return c10::IValue{}; },
          {}, jyppx::liuding::ValueSchema::any()),
      jyppx::liuding::InvalidArgument);

  auto model = std::make_shared<ProbeModelImpl>();
  EXPECT_THROW(jyppx::liuding::nn::adapt_module(
                   std::static_pointer_cast<torch::nn::Module>(model),
                   jyppx::liuding::nn::ModelInvoker{}, {},
                   jyppx::liuding::ValueSchema::any()),
               jyppx::liuding::InvalidArgument);

  ProbeModel empty(nullptr);
  EXPECT_THROW(jyppx::liuding::nn::adapt_tensor_holder(empty),
               jyppx::liuding::InvalidArgument);
}

}  // namespace
