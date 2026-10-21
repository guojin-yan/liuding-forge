// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/foundation/sha256.hpp>
#include <jyppx/interop/torchscript_compatibility.hpp>

#include <torch/script.h>
#include <torch/torch.h>

#include <ATen/autocast_mode.h>
#include <ATen/core/Dict.h>
#include <ATen/core/List.h>
#include <algorithm>
#include <array>
#include <c10/core/InferenceMode.h>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <iterator>
#include <span>
#include <sstream>
#include <stop_token>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace jyppx;
using namespace jyppx::interop;
using Runtime = TorchScriptCompatibilityRuntime;

constexpr char kIdentity[] = R"JIT(
def forward(self, x: Tensor) -> Tensor:
    return x
)JIT";
constexpr char kArithmetic[] = R"JIT(
def forward(self, x: Tensor, y: Tensor, scale: float = 1.5) -> Tuple[Tensor, Tensor]:
    total = (x + y) * scale
    return (total, total.sum(1))
)JIT";

struct Artifact {
  std::vector<std::byte> bytes;
  ArtifactDigest digest;
};

Artifact Save(torch::jit::Module& module) {
  std::ostringstream stream(std::ios::out | std::ios::binary);
  module.save(stream);
  const auto serialized = stream.str();
  const auto bytes =
      std::as_bytes(std::span(serialized.data(), serialized.size()));
  Artifact artifact{.bytes = {bytes.begin(), bytes.end()}, .digest = {}};
  artifact.digest.hex = sha256_hex(artifact.bytes);
  return artifact;
}

Artifact Compile(const char* source, bool with_bias = false) {
  torch::jit::Module module("CompatibilityFixture");
  if (with_bias)
    module.register_buffer("bias", torch::arange(3, torch::kFloat32));
  module.define(source);
  return Save(module);
}

TorchScriptLoadOptions Trusted(Device device = Device(c10::kCPU)) {
  TorchScriptLoadOptions options;
  options.trust_source = true;
  options.device = device;
  return options;
}

nn::InvocationSchema OneTensor() {
  return {.positional = {ValueSchema::tensor()},
          .keyword = {},
          .allow_extra_keywords = false};
}

ValueSchema Scalar(ValueKind kind, bool required = true) {
  ValueSchema field;
  field.kind = kind;
  field.required = required;
  return field;
}

nn::InvocationSchema ArithmeticSchema(std::size_t prefix = 2U) {
  std::vector<ValueSchema> all{ValueSchema::tensor(), ValueSchema::tensor(),
                               Scalar(ValueKind::floating)};
  constexpr std::array<const char*, 3> names{"x", "y", "scale"};
  nn::InvocationSchema schema;
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (i < prefix) {
      schema.positional.push_back(all[i]);
    } else {
      all[i].required = i != 2U;
      schema.keyword.emplace(names[i], all[i]);
    }
  }
  return schema;
}

ValueSchema ArithmeticOutput() {
  return ValueSchema::tuple({ValueSchema::tensor(), ValueSchema::tensor()});
}

void CheckArithmetic(Runtime& runtime, const Device& device) {
  const auto x = torch::arange(6, torch::kFloat32).reshape({2, 3}).to(device);
  const auto y = torch::full({2, 3}, 2.0, torch::kFloat32).to(device);
  const auto result = runtime.invoke({.positional = {x, y}, .keyword = {}});
  const auto& values = result.value.toTupleRef().elements();
  ASSERT_EQ(values.size(), 2U);
  EXPECT_TRUE(torch::allclose(values[0].toTensor(), (x + y) * 1.5));
  EXPECT_TRUE(torch::allclose(values[1].toTensor(), ((x + y) * 1.5).sum(1)));
  EXPECT_EQ(values[0].toTensor().device(), device);
  const auto explicit_result = runtime.invoke(
      {.positional = {x, y}, .keyword = {{"scale", c10::IValue(2.0)}}});
  EXPECT_TRUE(torch::allclose(
      explicit_result.value.toTupleRef().elements()[0].toTensor(),
      (x + y) * 2.0));
}

TEST(TorchScriptCompatibility,
     ActualJitArchiveLoadsRunsDefaultsAndReportsTypes) {
  const auto artifact = Compile(kArithmetic);
  auto runtime =
      Runtime::load(artifact.bytes, artifact.digest, ArithmeticSchema(),
                    ArithmeticOutput(), Trusted());
  const auto info = runtime.inspect();
  EXPECT_TRUE(info.open);
  EXPECT_TRUE(info.compatibility_only);
  EXPECT_TRUE(info.deprecated);
  EXPECT_EQ(info.artifact_digest, artifact.digest);
  EXPECT_EQ(info.artifact_bytes, artifact.bytes.size());
  EXPECT_FALSE(info.libtorch_version.empty());
  EXPECT_EQ(info.device, "cpu");
  ASSERT_EQ(info.arguments.size(), 3U);
  EXPECT_EQ(info.arguments[2].name, "scale");
  EXPECT_TRUE(info.arguments[2].has_default);
  EXPECT_FALSE(info.arguments[2].required);
  EXPECT_FALSE(info.arguments[2].bound_positionally);
  EXPECT_EQ(info.return_type.find("1.5"), std::string::npos);
  CheckArithmetic(runtime, Device(c10::kCPU));
}

TEST(TorchScriptCompatibility, EveryArgumentMayBeExplicitlyBoundByKeyword) {
  const auto artifact = Compile(kArithmetic);
  auto runtime =
      Runtime::load(artifact.bytes, artifact.digest, ArithmeticSchema(0U),
                    ArithmeticOutput(), Trusted());
  const auto x = torch::ones({2, 3});
  const auto result = runtime.invoke(
      {.positional = {}, .keyword = {{"x", x}, {"y", x}, {"scale", 3.0}}});
  EXPECT_TRUE(torch::equal(result.value.toTupleRef().elements()[0].toTensor(),
                           torch::full({2, 3}, 6.0)));
}

TEST(TorchScriptCompatibility, ThreePositionalsUseTheExplicitScale) {
  const auto artifact = Compile(kArithmetic);
  auto runtime =
      Runtime::load(artifact.bytes, artifact.digest, ArithmeticSchema(3U),
                    ArithmeticOutput(), Trusted());
  const auto result = runtime.invoke(
      {.positional = {torch::ones({2, 3}), torch::ones({2, 3}), 4.0},
       .keyword = {}});
  EXPECT_TRUE(torch::equal(result.value.toTupleRef().elements()[0].toTensor(),
                           torch::full({2, 3}, 8.0)));
}

TEST(TorchScriptCompatibility, SourceTrustIsRequiredBeforeDeserialization) {
  const std::array<std::byte, 1> corrupt{std::byte{0x61}};
  const ArtifactDigest digest{.hex = sha256_hex(corrupt)};
  EXPECT_THROW(
      Runtime::load(corrupt, digest, OneTensor(), ValueSchema::tensor()),
      CompatibilityError);
  EXPECT_THROW(Runtime::load(corrupt, digest, OneTensor(),
                             ValueSchema::tensor(), Trusted()),
               BackendError);
}

TEST(TorchScriptCompatibility, RejectsWrongDigestEmptyAndOversizedArtifacts) {
  const auto artifact = Compile(kIdentity);
  ArtifactDigest wrong{.hex = std::string(64U, '0')};
  EXPECT_THROW(Runtime::load(artifact.bytes, wrong, OneTensor(),
                             ValueSchema::tensor(), Trusted()),
               CompatibilityError);
  EXPECT_THROW(Runtime::load({}, artifact.digest, OneTensor(),
                             ValueSchema::tensor(), Trusted()),
               InvalidArgument);
  auto options = Trusted();
  options.limits.max_artifact_bytes = artifact.bytes.size() - 1U;
  EXPECT_THROW(Runtime::load(artifact.bytes, artifact.digest, OneTensor(),
                             ValueSchema::tensor(), options),
               ResourceExhausted);
  auto malformed_digest = artifact.digest;
  malformed_digest.hex[0] = 'A';
  EXPECT_THROW(Runtime::load(artifact.bytes, malformed_digest, OneTensor(),
                             ValueSchema::tensor(), Trusted()),
               InvalidArgument);
}

TEST(TorchScriptCompatibility,
     InputArtifactAndReturnedTensorsHaveIndependentStorage) {
  auto artifact = Compile(kIdentity);
  auto runtime = Runtime::load(artifact.bytes, artifact.digest, OneTensor(),
                               ValueSchema::tensor(), Trusted());
  std::fill(artifact.bytes.begin(), artifact.bytes.end(), std::byte{0});
  auto input = torch::arange(3, torch::kFloat32);
  auto result = runtime.invoke({.positional = {input}, .keyword = {}});
  EXPECT_NE(result.value.toTensor().data_ptr(), input.data_ptr());
  input.fill_(99.0);
  runtime.close();
  EXPECT_TRUE(
      torch::equal(result.value.toTensor(), torch::arange(3, torch::kFloat32)));
  EXPECT_FALSE(result.value.toTensor().requires_grad());
  EXPECT_TRUE(result.value.toTensor().is_inference());
}

TEST(TorchScriptCompatibility, ScriptedMutationChangesOnlyInputSnapshot) {
  const auto artifact = Compile(R"JIT(
def forward(self, x: Tensor) -> Tensor:
    return x.add_(2.0)
)JIT");
  auto runtime = Runtime::load(artifact.bytes, artifact.digest, OneTensor(),
                               ValueSchema::tensor(), Trusted());
  const auto input = torch::ones({3});
  const auto result = runtime.invoke({.positional = {input}, .keyword = {}});
  EXPECT_TRUE(torch::equal(input, torch::ones({3})));
  EXPECT_TRUE(torch::equal(result.value.toTensor(), torch::full({3}, 3.0)));
}

TEST(TorchScriptCompatibility,
     DenseNoncontiguousOutputIsSnapshottedContiguously) {
  const auto artifact = Compile(R"JIT(
def forward(self, x: Tensor) -> Tensor:
    return x.t()
)JIT");
  auto runtime = Runtime::load(artifact.bytes, artifact.digest, OneTensor(),
                               ValueSchema::tensor(), Trusted());
  const auto input = torch::arange(6, torch::kFloat32).reshape({2, 3});
  const auto result = runtime.invoke({.positional = {input}, .keyword = {}});
  EXPECT_TRUE(result.value.toTensor().is_contiguous());
  EXPECT_TRUE(torch::equal(result.value.toTensor(), input.t()));
}

TEST(TorchScriptCompatibility,
     TensorIdentityPreservesAllTenSupportedCpuDtypes) {
  const auto artifact = Compile(kIdentity);
  auto runtime = Runtime::load(artifact.bytes, artifact.digest, OneTensor(),
                               ValueSchema::tensor(), Trusted());
  const std::array<at::ScalarType, 10> dtypes{
      at::kBool, at::kByte, at::kChar,     at::kShort, at::kInt,
      at::kLong, at::kHalf, at::kBFloat16, at::kFloat, at::kDouble};
  for (const auto dtype : dtypes) {
    const auto input = torch::ones({2}, torch::TensorOptions().dtype(dtype));
    const auto result = runtime.invoke({.positional = {input}, .keyword = {}});
    EXPECT_EQ(result.value.toTensor().scalar_type(), dtype);
    EXPECT_TRUE(torch::equal(input, result.value.toTensor()));
  }
}

TEST(TorchScriptCompatibility,
     TensorTupleListAndFixedDictionaryAreReallyExecuted) {
  const auto artifact = Compile(R"JIT(
def forward(self, xs: List[Tensor], pair: Tuple[Tensor, int], options: Dict[str, Tensor]) -> Dict[str, Tensor]:
    total = xs[0] + xs[1] + pair[0] * pair[1] + options["bias"]
    return {"result": total}
)JIT");
  nn::InvocationSchema schema{
      .positional = {ValueSchema::list(ValueSchema::tensor()),
                     ValueSchema::tuple(
                         {ValueSchema::tensor(), Scalar(ValueKind::integer)}),
                     ValueSchema::dictionary(
                         {{"bias", ValueSchema::tensor()}})},
      .keyword = {}};
  auto runtime = Runtime::load(
      artifact.bytes, artifact.digest, schema,
      ValueSchema::dictionary({{"result", ValueSchema::tensor()}}), Trusted());
  const auto one = torch::ones({3});
  c10::List<at::Tensor> list;
  list.push_back(one);
  list.push_back(one * 2.0);
  c10::Dict<std::string, at::Tensor> dictionary;
  dictionary.insert("bias", one * 4.0);
  const auto pair =
      c10::ivalue::Tuple::create({one, c10::IValue(std::int64_t{3})});
  const auto result =
      runtime.invoke({.positional = {list, pair, dictionary}, .keyword = {}});
  EXPECT_TRUE(torch::equal(result.value.toGenericDict().at("result").toTensor(),
                           one * 10.0));
  dictionary.insert("unexpected", one);
  EXPECT_THROW(
      runtime.invoke({.positional = {list, pair, dictionary}, .keyword = {}}),
      InvalidArgument);
  EXPECT_TRUE(runtime.inspect().open);
}

TEST(TorchScriptCompatibility, ListAndTupleResultsRetainTheirTypedStructure) {
  const auto artifact = Compile(R"JIT(
def forward(self, xs: List[int], pair: Tuple[str, bool]) -> Tuple[List[int], Tuple[str, bool]]:
    return (xs, pair)
)JIT");
  const auto list_schema = ValueSchema::list(Scalar(ValueKind::integer));
  const auto pair_schema = ValueSchema::tuple(
      {Scalar(ValueKind::string), Scalar(ValueKind::boolean)});
  auto runtime =
      Runtime::load(artifact.bytes, artifact.digest,
                    {.positional = {list_schema, pair_schema}, .keyword = {}},
                    ValueSchema::tuple({list_schema, pair_schema}), Trusted());
  c10::List<std::int64_t> list;
  list.push_back(7);
  list.push_back(11);
  const auto pair =
      c10::ivalue::Tuple::create({c10::IValue("ok"), c10::IValue(true)});
  const auto result =
      runtime.invoke({.positional = {list, pair}, .keyword = {}});
  const auto& values = result.value.toTupleRef().elements();
  ASSERT_EQ(values.size(), 2U);
  EXPECT_EQ(values[0].toIntVector(), (std::vector<std::int64_t>{7, 11}));
  EXPECT_EQ(values[1].toTupleRef().elements()[0].toStringRef(), "ok");
  EXPECT_TRUE(values[1].toTupleRef().elements()[1].toBool());
}

TEST(TorchScriptCompatibility,
     EmptyContainersMustHaveTheCorrectJitElementType) {
  const auto artifact = Compile(R"JIT(
def forward(self, xs: List[Tensor]) -> List[Tensor]:
    return xs
)JIT");
  const auto schema = ValueSchema::list(ValueSchema::tensor());
  auto runtime =
      Runtime::load(artifact.bytes, artifact.digest,
                    {.positional = {schema}, .keyword = {}}, schema, Trusted());
  c10::List<at::Tensor> correct;
  EXPECT_TRUE(runtime.invoke({.positional = {correct}, .keyword = {}})
                  .value.toListRef()
                  .empty());
  c10::List<std::int64_t> wrong;
  EXPECT_THROW(runtime.invoke({.positional = {wrong}, .keyword = {}}),
               InvalidArgument);
  EXPECT_TRUE(runtime.inspect().open);
}

TEST(TorchScriptCompatibility,
     OptionalTensorUsesActualNoneDefaultAndExplicitValue) {
  const auto artifact = Compile(R"JIT(
def forward(self, x: Optional[Tensor] = None) -> Tensor:
    if x is None:
        return self.bias
    return x + self.bias
)JIT",
                                true);
  auto field = ValueSchema::tensor();
  field.nullable = true;
  field.required = false;
  auto runtime = Runtime::load(artifact.bytes, artifact.digest,
                               {.positional = {}, .keyword = {{"x", field}}},
                               ValueSchema::tensor(), Trusted());
  const auto default_result = runtime.invoke({});
  EXPECT_TRUE(torch::equal(default_result.value.toTensor(),
                           torch::arange(3, torch::kFloat32)));
  const auto explicit_none =
      runtime.invoke({.positional = {}, .keyword = {{"x", c10::IValue()}}});
  EXPECT_TRUE(torch::equal(explicit_none.value.toTensor(),
                           default_result.value.toTensor()));
  const auto explicit_tensor =
      runtime.invoke({.positional = {}, .keyword = {{"x", torch::ones({3})}}});
  EXPECT_TRUE(torch::equal(explicit_tensor.value.toTensor(),
                           torch::arange(3, torch::kFloat32) + 1.0));
}

TEST(TorchScriptCompatibility, NoneScalarBoundaryIsExplicit) {
  const auto artifact = Compile(R"JIT(
def forward(self) -> None:
    return None
)JIT");
  auto runtime = Runtime::load(artifact.bytes, artifact.digest, {},
                               Scalar(ValueKind::none), Trusted());
  EXPECT_TRUE(runtime.invoke({}).value.isNone());
}

TEST(TorchScriptCompatibility,
     ActualParametersAndBuffersAreAccountedAfterLoad) {
  torch::jit::Module module("StateFixture");
  module.register_parameter("weight", torch::ones({3}), false);
  module.register_buffer("bias", torch::ones({3}));
  module.define(R"JIT(
def forward(self, x: Tensor) -> Tensor:
    return x * self.weight + self.bias
)JIT");
  const auto artifact = Save(module);
  auto runtime = Runtime::load(artifact.bytes, artifact.digest, OneTensor(),
                               ValueSchema::tensor(), Trusted());
  EXPECT_EQ(runtime.inspect().state_tensors, 2U);
  EXPECT_EQ(runtime.inspect().state_tensor_bytes, 24U);
  auto options = Trusted();
  options.limits.max_state_tensor_bytes = 23U;
  EXPECT_THROW(Runtime::load(artifact.bytes, artifact.digest, OneTensor(),
                             ValueSchema::tensor(), options),
               ResourceExhausted);
  options = Trusted();
  options.limits.max_state_tensors = 1U;
  EXPECT_THROW(Runtime::load(artifact.bytes, artifact.digest, OneTensor(),
                             ValueSchema::tensor(), options),
               ResourceExhausted);
}

TEST(TorchScriptCompatibility, RejectsMissingForwardAndUnsupportedJitTypes) {
  torch::jit::Module missing("NoForwardFixture");
  missing.register_buffer("bias", torch::ones({1}));
  const auto absent = Save(missing);
  EXPECT_THROW(Runtime::load(absent.bytes, absent.digest, {},
                             ValueSchema::tensor(), Trusted()),
               CompatibilityError);
  const auto any = Compile(R"JIT(
def forward(self, x: Any) -> Any:
    return x
)JIT");
  EXPECT_THROW(
      Runtime::load(any.bytes, any.digest,
                    {.positional = {ValueSchema::any()}, .keyword = {}},
                    ValueSchema::any(), Trusted()),
      CompatibilityError);
  const auto int_dict = Compile(R"JIT(
def forward(self, x: Dict[int, Tensor]) -> Tensor:
    return x[0]
)JIT");
  EXPECT_THROW(Runtime::load(
                   int_dict.bytes, int_dict.digest,
                   {.positional = {ValueSchema::dictionary({})}, .keyword = {}},
                   ValueSchema::tensor(), Trusted()),
               CompatibilityError);
}

TEST(TorchScriptCompatibility, LoadRejectsIncompleteRenamedOrOpenBinding) {
  const auto artifact = Compile(kArithmetic);
  auto schema = ArithmeticSchema();
  schema.keyword.clear();
  EXPECT_THROW(Runtime::load(artifact.bytes, artifact.digest, schema,
                             ArithmeticOutput(), Trusted()),
               CompatibilityError);
  schema = ArithmeticSchema();
  auto field = schema.keyword.at("scale");
  schema.keyword.clear();
  schema.keyword.emplace("renamed_scale", field);
  EXPECT_THROW(Runtime::load(artifact.bytes, artifact.digest, schema,
                             ArithmeticOutput(), Trusted()),
               CompatibilityError);
  schema = ArithmeticSchema();
  schema.allow_extra_keywords = true;
  EXPECT_THROW(Runtime::load(artifact.bytes, artifact.digest, schema,
                             ArithmeticOutput(), Trusted()),
               CompatibilityError);
}

TEST(TorchScriptCompatibility, CannotInventMissingDefaultsOrNullableSemantics) {
  const auto artifact = Compile(kIdentity);
  auto field = ValueSchema::tensor();
  field.required = false;
  EXPECT_THROW(Runtime::load(artifact.bytes, artifact.digest,
                             {.positional = {}, .keyword = {{"x", field}}},
                             ValueSchema::tensor(), Trusted()),
               CompatibilityError);
  field = ValueSchema::tensor();
  field.nullable = true;
  EXPECT_THROW(Runtime::load(artifact.bytes, artifact.digest,
                             {.positional = {field}, .keyword = {}},
                             ValueSchema::tensor(), Trusted()),
               CompatibilityError);
}

TEST(TorchScriptCompatibility,
     InputDtypeShapeArityAndKeywordErrorsDoNotExecute) {
  const auto artifact = Compile(kArithmetic);
  auto schema = ArithmeticSchema();
  schema.positional[0] = ValueSchema::tensor(
      {.dtype = at::kFloat,
       .device = Device(c10::kCPU),
       .rank = 2,
       .shape = std::vector<std::optional<std::int64_t>>{2, 3}});
  auto runtime = Runtime::load(artifact.bytes, artifact.digest, schema,
                               ArithmeticOutput(), Trusted());
  const auto one = torch::ones({2, 3});
  EXPECT_THROW(runtime.invoke({.positional = {one}, .keyword = {}}),
               InvalidArgument);
  EXPECT_THROW(
      runtime.invoke({.positional = {one.to(at::kDouble), one}, .keyword = {}}),
      InvalidArgument);
  EXPECT_THROW(
      runtime.invoke({.positional = {torch::ones({3, 2}), one}, .keyword = {}}),
      InvalidArgument);
  EXPECT_THROW(
      runtime.invoke({.positional = {one, one}, .keyword = {{"scale", 2}}}),
      InvalidArgument);
  EXPECT_THROW(
      runtime.invoke({.positional = {one, one}, .keyword = {{"x", one}}}),
      InvalidArgument);
  EXPECT_TRUE(runtime.inspect().open);
  CheckArithmetic(runtime, Device(c10::kCPU));
}

TEST(TorchScriptCompatibility, RejectsUnsupportedLayoutDtypeAndInputStride) {
  const auto artifact = Compile(kIdentity);
  auto runtime = Runtime::load(artifact.bytes, artifact.digest, OneTensor(),
                               ValueSchema::tensor(), Trusted());
  EXPECT_THROW(
      runtime.invoke({.positional = {torch::ones({2}, torch::kComplexFloat)},
                      .keyword = {}}),
      CompatibilityError);
  EXPECT_THROW(
      runtime.invoke({.positional = {torch::ones({2, 3}).t()}, .keyword = {}}),
      CompatibilityError);
  EXPECT_THROW(runtime.invoke({.positional = {torch::ones({2, 3}).to_sparse()},
                               .keyword = {}}),
               CompatibilityError);
  EXPECT_TRUE(runtime.inspect().open);
}

TEST(TorchScriptCompatibility,
     InputBudgetRejectsBeforeForwardAndOutputBudgetCloses) {
  const auto artifact = Compile(kIdentity);
  auto options = Trusted();
  options.limits.max_input_tensor_bytes = 7U;
  auto input_limited =
      Runtime::load(artifact.bytes, artifact.digest, OneTensor(),
                    ValueSchema::tensor(), options);
  EXPECT_THROW(
      input_limited.invoke({.positional = {torch::ones({2})}, .keyword = {}}),
      ResourceExhausted);
  EXPECT_TRUE(input_limited.inspect().open);
  options = Trusted();
  options.limits.max_output_tensor_bytes = 7U;
  auto output_limited =
      Runtime::load(artifact.bytes, artifact.digest, OneTensor(),
                    ValueSchema::tensor(), options);
  EXPECT_THROW(
      output_limited.invoke({.positional = {torch::ones({2})}, .keyword = {}}),
      ResourceExhausted);
  EXPECT_FALSE(output_limited.inspect().open);
  EXPECT_THROW(
      output_limited.invoke({.positional = {torch::ones({1})}, .keyword = {}}),
      NotReady);
}

TEST(TorchScriptCompatibility, ValueAndStringBudgetsAreEnforced) {
  const auto artifact = Compile(R"JIT(
def forward(self, xs: List[str]) -> List[str]:
    return xs
)JIT");
  const auto schema = ValueSchema::list(Scalar(ValueKind::string));
  auto options = Trusted();
  options.limits.max_string_bytes = 3U;
  auto limited =
      Runtime::load(artifact.bytes, artifact.digest,
                    {.positional = {schema}, .keyword = {}}, schema, options);
  c10::List<std::string> list;
  list.push_back("long");
  EXPECT_THROW(limited.invoke({.positional = {list}, .keyword = {}}),
               ResourceExhausted);
  options = Trusted();
  options.limits.value_limits.max_runtime_nodes = 2U;
  auto node_limited =
      Runtime::load(artifact.bytes, artifact.digest,
                    {.positional = {schema}, .keyword = {}}, schema, options);
  list.clear();
  list.push_back("a");
  list.push_back("b");
  EXPECT_THROW(node_limited.invoke({.positional = {list}, .keyword = {}}),
               ResourceExhausted);
}

TEST(TorchScriptCompatibility, WrongActualResultClosesRuntime) {
  const auto artifact = Compile(kIdentity);
  auto runtime =
      Runtime::load(artifact.bytes, artifact.digest, OneTensor(),
                    ValueSchema::tensor({.dtype = at::kDouble}), Trusted());
  EXPECT_THROW(
      runtime.invoke(
          {.positional = {torch::ones({2}, torch::kFloat32)}, .keyword = {}}),
      InvalidArgument);
  EXPECT_FALSE(runtime.inspect().open);
}

TEST(TorchScriptCompatibility,
     BackendFailureClosesAndPreservesPrivateNestedCause) {
  const auto artifact = Compile(R"JIT(
def forward(self, x: Tensor) -> Tensor:
    return x.reshape([2, 2])
)JIT");
  auto runtime = Runtime::load(artifact.bytes, artifact.digest, OneTensor(),
                               ValueSchema::tensor(), Trusted());
  try {
    static_cast<void>(
        runtime.invoke({.positional = {torch::ones({1})}, .keyword = {}}));
    FAIL() << "The real JIT method must raise";
  } catch (const BackendError& error) {
    EXPECT_EQ(std::string(error.what()).find("reshape"), std::string::npos);
    bool nested = false;
    try {
      std::rethrow_if_nested(error);
    } catch (const std::exception& cause) {
      nested = true;
      EXPECT_NE(std::string(cause.what()).find("shape"), std::string::npos);
    }
    EXPECT_TRUE(nested);
  }
  EXPECT_FALSE(runtime.inspect().open);
}

TEST(TorchScriptCompatibility,
     CooperativePreCancellationAndDeadlinesKeepReadyState) {
  const auto artifact = Compile(kIdentity);
  std::stop_source stop;
  stop.request_stop();
  EXPECT_THROW(
      Runtime::load(artifact.bytes, artifact.digest, OneTensor(),
                    ValueSchema::tensor(), Trusted(), stop.get_token()),
      Cancelled);
  const auto past = std::chrono::steady_clock::now() - std::chrono::seconds(1);
  EXPECT_THROW(Runtime::load(artifact.bytes, artifact.digest, OneTensor(),
                             ValueSchema::tensor(), Trusted(), {}, past),
               DeadlineExceeded);
  auto runtime = Runtime::load(artifact.bytes, artifact.digest, OneTensor(),
                               ValueSchema::tensor(), Trusted());
  const nn::Invocation input{.positional = {torch::ones({1})}, .keyword = {}};
  EXPECT_THROW(runtime.invoke(input, stop.get_token()), Cancelled);
  EXPECT_THROW(runtime.invoke(input, {}, past), DeadlineExceeded);
  EXPECT_TRUE(runtime.inspect().open);
}

TEST(TorchScriptCompatibility,
     PrivateStateIsSerializedAndPriorResultsRemainSnapshots) {
  torch::jit::Module module("CounterFixture");
  module.register_buffer("count", torch::zeros({1}));
  module.define(R"JIT(
def forward(self, x: Tensor) -> Tensor:
    self.count.add_(1.0)
    return self.count + x
)JIT");
  const auto artifact = Save(module);
  auto runtime = Runtime::load(artifact.bytes, artifact.digest, OneTensor(),
                               ValueSchema::tensor(), Trusted());
  std::promise<void> start;
  const auto started = start.get_future().share();
  std::vector<std::future<float>> calls;
  for (int i = 0; i < 8; ++i)
    calls.push_back(std::async(std::launch::async, [&runtime, started] {
      started.wait();
      return runtime.invoke({.positional = {torch::zeros({1})}, .keyword = {}})
          .value.toTensor()
          .item<float>();
    }));
  start.set_value();
  std::vector<float> observed;
  for (auto& call : calls) observed.push_back(call.get());
  std::sort(observed.begin(), observed.end());
  EXPECT_EQ(observed, (std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}));
  const auto ninth =
      runtime.invoke({.positional = {torch::zeros({1})}, .keyword = {}});
  static_cast<void>(
      runtime.invoke({.positional = {torch::zeros({1})}, .keyword = {}}));
  EXPECT_EQ(ninth.value.toTensor().item<float>(), 9.0F);
}

TEST(TorchScriptCompatibility, CloseAndMoveHaveDefinedOwnership) {
  const auto artifact = Compile(kIdentity);
  auto original = Runtime::load(artifact.bytes, artifact.digest, OneTensor(),
                                ValueSchema::tensor(), Trusted());
  auto moved = std::move(original);
  EXPECT_THROW(original.inspect(), NotReady);
  original.close();
  EXPECT_TRUE(moved.inspect().open);
  moved.close();
  moved.close();
  EXPECT_FALSE(moved.inspect().open);
  EXPECT_EQ(moved.inspect().artifact_digest, artifact.digest);
  EXPECT_THROW(moved.invoke({.positional = {torch::ones({1})}, .keyword = {}}),
               NotReady);
}

TEST(TorchScriptCompatibility,
     InvalidLimitsAndImplicitOrOtherDevicesAreRejected) {
  const auto artifact = Compile(kIdentity);
  auto options = Trusted();
  options.limits.max_arguments = 0U;
  EXPECT_THROW(Runtime::load(artifact.bytes, artifact.digest, OneTensor(),
                             ValueSchema::tensor(), options),
               InvalidArgument);
  options = Trusted(Device(c10::kCUDA));
  EXPECT_THROW(Runtime::load(artifact.bytes, artifact.digest, OneTensor(),
                             ValueSchema::tensor(), options),
               CompatibilityError);
  options = Trusted(Device(c10::kMeta));
  EXPECT_THROW(Runtime::load(artifact.bytes, artifact.digest, OneTensor(),
                             ValueSchema::tensor(), options),
               CompatibilityError);
}

TEST(TorchScriptCompatibility,
     ActiveAutocastIsRejectedWithoutChangingCallerContext) {
  struct AutocastState {
    bool previous = at::autocast::is_autocast_enabled(at::kCPU);
    ~AutocastState() { at::autocast::set_autocast_enabled(at::kCPU, previous); }
  } restore;
  at::autocast::set_autocast_enabled(at::kCPU, false);
  const auto artifact = Compile(kIdentity);
  auto runtime = Runtime::load(artifact.bytes, artifact.digest, OneTensor(),
                               ValueSchema::tensor(), Trusted());
  at::autocast::set_autocast_enabled(at::kCPU, true);
  EXPECT_THROW(
      runtime.invoke({.positional = {torch::ones({1})}, .keyword = {}}),
      CompatibilityError);
  EXPECT_TRUE(at::autocast::is_autocast_enabled(at::kCPU));
  EXPECT_TRUE(runtime.inspect().open);
  EXPECT_THROW(Runtime::load(artifact.bytes, artifact.digest, OneTensor(),
                             ValueSchema::tensor(), Trusted()),
               CompatibilityError);
}

#ifdef JYPPX_TEST_TORCHSCRIPT_CUDA
TEST(TorchScriptCompatibilityCuda,
     ActualLoadRelocatesStateAndExecutesTupleAndDefaults) {
  ASSERT_TRUE(torch::cuda::is_available());
  const Device device(c10::kCUDA, 0);
  const auto artifact = Compile(kArithmetic);
  auto runtime =
      Runtime::load(artifact.bytes, artifact.digest, ArithmeticSchema(),
                    ArithmeticOutput(), Trusted(device));
  CheckArithmetic(runtime, device);
  EXPECT_EQ(runtime.inspect().device, "cuda:0");
  const auto state_artifact = Compile(R"JIT(
def forward(self, x: Tensor) -> Tensor:
    return x + self.bias
)JIT",
                                      true);
  auto state_runtime =
      Runtime::load(state_artifact.bytes, state_artifact.digest, OneTensor(),
                    ValueSchema::tensor(), Trusted(device));
  const auto input = torch::ones({3}, torch::TensorOptions().device(device));
  const auto result =
      state_runtime.invoke({.positional = {input}, .keyword = {}});
  EXPECT_EQ(result.value.toTensor().device(), device);
  EXPECT_TRUE(torch::allclose(result.value.toTensor().cpu(),
                              torch::arange(3, torch::kFloat32) + 1.0));
  EXPECT_THROW(
      state_runtime.invoke({.positional = {torch::ones({3})}, .keyword = {}}),
      CompatibilityError);
  EXPECT_TRUE(state_runtime.inspect().open);
}

TEST(TorchScriptCompatibilityCuda, IdentityPreservesAllTenDeclaredCudaDtypes) {
  ASSERT_TRUE(torch::cuda::is_available());
  const Device device(c10::kCUDA, 0);
  const auto artifact = Compile(kIdentity);
  auto runtime = Runtime::load(artifact.bytes, artifact.digest, OneTensor(),
                               ValueSchema::tensor(), Trusted(device));
  const std::array<at::ScalarType, 10> dtypes{
      at::kBool, at::kByte, at::kChar,     at::kShort, at::kInt,
      at::kLong, at::kHalf, at::kBFloat16, at::kFloat, at::kDouble};
  for (const auto dtype : dtypes) {
    const auto input =
        torch::ones({2}, torch::TensorOptions().dtype(dtype).device(device));
    const auto result = runtime.invoke({.positional = {input}, .keyword = {}});
    EXPECT_EQ(result.value.toTensor().scalar_type(), dtype);
    EXPECT_TRUE(torch::equal(input.cpu(), result.value.toTensor().cpu()));
  }
}
#endif

#ifdef JYPPX_TORCHSCRIPT_PYTHON_INTEROP
TEST(TorchScriptCompatibilityOfficial, PythonScriptArchiveReallyLoadsAndRuns) {
  const auto* fixture = std::getenv("JYPPX_TORCHSCRIPT_FIXTURE");
  const auto* digest = std::getenv("JYPPX_TORCHSCRIPT_EXPECT_DIGEST");
  const auto* version =
      std::getenv("JYPPX_TORCHSCRIPT_EXPECT_LIBTORCH_VERSION");
  ASSERT_NE(fixture, nullptr);
  ASSERT_NE(digest, nullptr);
  ASSERT_NE(version, nullptr);
  std::ifstream stream(fixture, std::ios::in | std::ios::binary);
  ASSERT_TRUE(stream.is_open());
  const std::string serialized((std::istreambuf_iterator<char>(stream)),
                               std::istreambuf_iterator<char>());
  const auto bytes =
      std::as_bytes(std::span(serialized.data(), serialized.size()));
  auto runtime = Runtime::load(bytes, {.hex = digest}, ArithmeticSchema(),
                               ArithmeticOutput(), Trusted());
  EXPECT_EQ(runtime.inspect().libtorch_version, version);
  CheckArithmetic(runtime, Device(c10::kCPU));
  if (std::getenv("JYPPX_TORCHSCRIPT_EXPECT_CUDA") != nullptr) {
#ifdef JYPPX_TEST_TORCHSCRIPT_CUDA
    ASSERT_TRUE(torch::cuda::is_available());
    const Device device(c10::kCUDA, 0);
    auto cuda_runtime =
        Runtime::load(bytes, {.hex = digest}, ArithmeticSchema(),
                      ArithmeticOutput(), Trusted(device));
    CheckArithmetic(cuda_runtime, device);
#else
    FAIL() << "CUDA evidence requires JYPPX_TEST_TORCHSCRIPT_CUDA";
#endif
  }
}
#endif

}  // namespace
