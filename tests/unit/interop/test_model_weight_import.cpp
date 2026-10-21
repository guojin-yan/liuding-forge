// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/foundation/sha256.hpp>
#include <jyppx/interop/model_weight_import.hpp>
#include <jyppx/nn/model_adapter.hpp>

#include <torch/torch.h>

#include <array>
#include <c10/core/InferenceMode.h>
#include <c10/core/TensorImpl.h>
#include <chrono>
#include <exception>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <new>
#include <stop_token>
#include <string>

namespace {
using namespace jyppx;
using namespace jyppx::interop;

struct Model : torch::nn::Module {
  torch::Tensor weight =
      register_parameter("weight", torch::zeros({2}, torch::kFloat64));
  torch::Tensor count =
      register_buffer("count", torch::tensor(7, torch::kInt64));
};

std::vector<std::byte> source(bool include_count = true,
                              SafeTensorDtype dtype = SafeTensorDtype::f64,
                              bool bad_shape = false) {
  const std::array<double, 2> doubles{1.25, -2.5};
  const std::array<float, 2> floats{1.25F, -2.5F};
  const std::int64_t count = 99;
  const auto bytes = dtype == SafeTensorDtype::f32
                         ? std::as_bytes(std::span<const float>(floats))
                         : std::as_bytes(std::span<const double>(doubles));
  std::vector<SafeTensorSource> tensors{
      {.name = "weight",
       .dtype = dtype,
       .shape = bad_shape ? std::vector<std::uint64_t>{1U, 2U}
                          : std::vector<std::uint64_t>{2U},
       .data = bytes}};
  if (include_count)
    tensors.push_back({.name = "count",
                       .dtype = SafeTensorDtype::i64,
                       .shape = {},
                       .data = std::as_bytes(std::span(&count, 1U))});
  return serialize_safetensors(tensors);
}

ArtifactDigest hash(const std::vector<std::byte>& bytes) {
  return {.hex = sha256_hex(bytes)};
}

enum class InjectedFailure { backend, backend_oom, host_oom };

struct CopyFault {
  std::size_t attempt;
  bool after_copy;
  InjectedFailure kind = InjectedFailure::backend;
};

struct CopyFaults {
  std::vector<CopyFault> faults;
  std::size_t attempts = 0U;
};

// Exercise the real set_data transaction boundary without a product test hook.
// The normal CPU tensor's dispatch, storage and metadata remain intact.
struct FaultingTensorImpl final : c10::TensorImpl {
  std::shared_ptr<CopyFaults> faults;

  FaultingTensorImpl(const torch::Tensor& initial,
                     std::shared_ptr<CopyFaults> state)
      : c10::TensorImpl(c10::Storage(initial.storage()),
                        initial.unsafeGetTensorImpl()->key_set(),
                        initial.dtype()),
        faults(std::move(state)) {
    c10::TensorImpl::shallow_copy_from(initial.getIntrusivePtr());
  }

  void shallow_copy_from(
      const c10::intrusive_ptr<c10::TensorImpl>& other) override {
    ++faults->attempts;
    for (const auto& fault : faults->faults) {
      if (fault.attempt != faults->attempts) continue;
      if (fault.after_copy) c10::TensorImpl::shallow_copy_from(other);
      if (fault.kind == InjectedFailure::host_oom) throw std::bad_alloc();
      const c10::SourceLocation location{__func__, __FILE__,
                                         static_cast<std::uint32_t>(__LINE__)};
      if (fault.kind == InjectedFailure::backend_oom) {
        throw c10::OutOfMemoryError(location, "test-only-private-backend-path");
      }
      throw c10::Error(location, "test-only-private-backend-path");
    }
    c10::TensorImpl::shallow_copy_from(other);
  }
};

struct FaultModel : torch::nn::Module {
  torch::Tensor weight;
  torch::Tensor count =
      register_buffer("count", torch::tensor(7, torch::kInt64));

  explicit FaultModel(const std::shared_ptr<CopyFaults>& faults)
      : weight(register_parameter(
            "weight", torch::Tensor(c10::make_intrusive<FaultingTensorImpl>(
                          torch::zeros({2}, torch::kFloat64), faults)))) {}
};

void expect_nested_backend_cause(const Error& error, bool oom = false) {
  EXPECT_EQ(std::string(error.what()).find("test-only-private-backend-path"),
            std::string::npos);
  try {
    std::rethrow_if_nested(error);
    FAIL() << "The backend failure must remain inspectable as a nested cause";
  } catch (const c10::OutOfMemoryError&) {
    EXPECT_TRUE(oom);
  } catch (const c10::Error&) {
    EXPECT_FALSE(oom);
  }
}

TEST(ModelWeightImportTest,
     StrictImportAppliesVerifiedParametersAndBuffersTogether) {
  auto module = std::make_shared<Model>();
  nn::ModelStateView view(module);
  const auto identity = module->weight.unsafeGetTensorImpl();
  const auto old_view = module->weight.detach();
  const auto bytes = source();
  const auto result = import_model_weights(view, bytes, hash(bytes));
  ASSERT_TRUE(result.applied);
  EXPECT_TRUE(result.plan.ready);
  EXPECT_EQ(result.applied_tensors, 2U);
  EXPECT_EQ(module->weight.unsafeGetTensorImpl(), identity);
  EXPECT_TRUE(torch::equal(module->weight,
                           torch::tensor({1.25, -2.5}, torch::kFloat64)));
  EXPECT_EQ(module->count.item<std::int64_t>(), 99);
  EXPECT_TRUE(module->weight.requires_grad());
  EXPECT_TRUE(module->is_training());
  EXPECT_TRUE(torch::equal(old_view, torch::zeros({2}, torch::kFloat64)));
}

TEST(ModelWeightImportTest,
     RejectedStrictMappingLeavesEveryStateValueUnchanged) {
  auto module = std::make_shared<Model>();
  nn::ModelStateView view(module);
  const auto original = module->weight.clone();
  for (const auto bytes : {source(false), source(true, SafeTensorDtype::f32),
                           source(true, SafeTensorDtype::f64, true)}) {
    const auto result = import_model_weights(view, bytes, hash(bytes));
    EXPECT_FALSE(result.applied);
    EXPECT_FALSE(result.plan.ready);
    EXPECT_TRUE(torch::equal(module->weight, original));
    EXPECT_EQ(module->count.item<std::int64_t>(), 7);
  }
}

TEST(ModelWeightImportTest,
     ExplicitPartialImportRetainsMissingStateAndReportsCoverage) {
  auto module = std::make_shared<Model>();
  nn::ModelStateView view(module);
  const auto bytes = source(false);
  ModelWeightImportOptions options;
  options.mapping.mode = WeightImportMode::partial;
  options.mapping.minimum_tensor_coverage = 0.5;
  const auto result = import_model_weights(view, bytes, hash(bytes), options);
  ASSERT_TRUE(result.applied);
  EXPECT_DOUBLE_EQ(result.plan.tensor_coverage, 0.5);
  ASSERT_EQ(result.plan.retained_targets.size(), 1U);
  EXPECT_EQ(result.plan.retained_targets.front().key, "count");
  EXPECT_EQ(module->count.item<std::int64_t>(), 7);
  options.mapping.minimum_tensor_coverage = 1.0;
  EXPECT_FALSE(import_model_weights(view, bytes, hash(bytes), options).applied);
}

TEST(ModelWeightImportTest, PartialCastRequiresExactImplementedPerKeyRule) {
  auto module = std::make_shared<Model>();
  nn::ModelStateView view(module);
  const auto bytes = source(true, SafeTensorDtype::f32);
  ModelWeightImportOptions options;
  options.mapping.mode = WeightImportMode::partial;
  options.mapping.minimum_tensor_coverage = 1.0;
  EXPECT_FALSE(import_model_weights(view, bytes, hash(bytes), options).applied);
  options.mapping.conversions = {{"weight", SafeTensorDtype::f32,
                                  SafeTensorDtype::f64, "unsupported.rule.v1"}};
  EXPECT_THROW(import_model_weights(view, bytes, hash(bytes), options),
               CompatibilityError);
  EXPECT_TRUE(torch::equal(module->weight, torch::zeros({2}, torch::kFloat64)));
  options.mapping.conversions.front().rule_id = "jyppx.float-cast.v1";
  const auto result = import_model_weights(view, bytes, hash(bytes), options);
  EXPECT_TRUE(result.applied);
  ASSERT_EQ(result.plan.conversions.size(), 1U);
  EXPECT_EQ(module->weight.scalar_type(), torch::kFloat64);
  EXPECT_TRUE(torch::equal(module->weight,
                           torch::tensor({1.25, -2.5}, torch::kFloat64)));
}

TEST(ModelWeightImportTest,
     DigestResourceCancellationAndDeadlineFailuresPreserveModel) {
  auto module = std::make_shared<Model>();
  nn::ModelStateView view(module);
  const auto bytes = source();
  auto digest = hash(bytes);
  digest.hex[0] = digest.hex[0] == '0' ? '1' : '0';
  EXPECT_THROW(import_model_weights(view, bytes, digest), SerializationError);
  ModelWeightImportOptions options;
  options.max_staging_bytes = 1U;
  EXPECT_THROW(import_model_weights(view, bytes, hash(bytes), options),
               ResourceExhausted);
  std::stop_source stop;
  stop.request_stop();
  EXPECT_THROW(
      import_model_weights(view, bytes, hash(bytes), {}, stop.get_token()),
      Cancelled);
  EXPECT_THROW(import_model_weights(
                   view, bytes, hash(bytes), {}, {},
                   std::chrono::steady_clock::now() - std::chrono::seconds(1)),
               DeadlineExceeded);
  EXPECT_TRUE(torch::equal(module->weight, torch::zeros({2}, torch::kFloat64)));
  EXPECT_EQ(module->count.item<std::int64_t>(), 7);
}

TEST(ModelWeightImportTest,
     AliasedStateAndExistingGradientsAreRejectedBeforeMutation) {
  auto module = std::make_shared<Model>();
  nn::ModelStateView view(module);
  const auto bytes = source();
  module->weight.mutable_grad() = torch::ones_like(module->weight);
  EXPECT_THROW(import_model_weights(view, bytes, hash(bytes)),
               CompatibilityError);
  module->weight.mutable_grad() = torch::Tensor{};
  module->register_buffer("alias", module->weight.detach().view({2}));
  EXPECT_THROW(import_model_weights(view, bytes, hash(bytes)),
               CompatibilityError);
  EXPECT_TRUE(torch::equal(module->weight, torch::zeros({2}, torch::kFloat64)));
  EXPECT_EQ(module->count.item<std::int64_t>(), 7);
}

TEST(ModelWeightImportTest,
     SeparateStorageWrappersCannotHideFullOrPartialOverlap) {
  for (const std::size_t offset : {0U, 1U}) {
    for (const auto mode :
         {WeightImportMode::strict, WeightImportMode::partial}) {
      SCOPED_TRACE(offset);
      std::array<double, 3> backing{4.0, 5.0, 6.0};
      auto module = std::make_shared<torch::nn::Module>();
      const auto first = module->register_buffer(
          "weight", torch::from_blob(backing.data(), {2}, torch::kFloat64));
      const auto second = module->register_buffer(
          "retained",
          torch::from_blob(backing.data() + offset, {2}, torch::kFloat64));
      ASSERT_NE(first.storage().unsafeGetStorageImpl(),
                second.storage().unsafeGetStorageImpl());
      nn::ModelStateView view(module);
      const auto bytes = source(false);
      ModelWeightImportOptions options;
      options.mapping.mode = mode;
      options.mapping.minimum_tensor_coverage = 0.5;
      EXPECT_THROW(import_model_weights(view, bytes, hash(bytes), options),
                   CompatibilityError);
      EXPECT_EQ(first.data_ptr(), backing.data());
      EXPECT_EQ(second.data_ptr(), backing.data() + offset);
      EXPECT_EQ(backing, (std::array<double, 3>{4.0, 5.0, 6.0}));
    }
  }
}

TEST(ModelWeightImportTest,
     AdjacentIndependentWrappersAreAcceptedWithoutTouchingRetainedBytes) {
  std::array<double, 4> backing{4.0, 5.0, 6.0, 7.0};
  auto module = std::make_shared<torch::nn::Module>();
  const auto weight = module->register_buffer(
      "weight", torch::from_blob(backing.data(), {2}, torch::kFloat64));
  const auto retained = module->register_buffer(
      "retained", torch::from_blob(backing.data() + 2U, {2}, torch::kFloat64));
  nn::ModelStateView view(module);
  const auto bytes = source(false);
  ModelWeightImportOptions options;
  options.mapping.mode = WeightImportMode::partial;
  options.mapping.minimum_tensor_coverage = 0.5;
  ASSERT_TRUE(import_model_weights(view, bytes, hash(bytes), options).applied);
  EXPECT_TRUE(
      torch::equal(weight, torch::tensor({1.25, -2.5}, torch::kFloat64)));
  EXPECT_EQ(retained.data_ptr(), backing.data() + 2U);
  EXPECT_EQ(backing, (std::array<double, 4>{4.0, 5.0, 6.0, 7.0}));
}

TEST(ModelWeightImportTest, RepeatedEmptyTensorHandlesAreRejected) {
  auto module = std::make_shared<torch::nn::Module>();
  const auto empty = torch::empty({0}, torch::kFloat64);
  module->register_buffer("one", empty);
  module->register_buffer("two", empty);
  nn::ModelStateView view(module);
  const auto bytes = source(false);
  EXPECT_THROW(import_model_weights(view, bytes, hash(bytes)),
               CompatibilityError);
}

TEST(ModelWeightImportTest, ViewsNonLeavesInferenceAndLazyStateAreRejected) {
  auto leaf = torch::zeros(
      {2}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  const auto nonleaf = leaf * 2;
  const auto view_tensor = torch::zeros({4}, torch::kFloat64).narrow(0, 1, 2);
  const auto negative = torch::ones({2}, torch::kFloat64)._neg_view().detach();
  const auto conjugate =
      torch::zeros({2}, torch::kComplexDouble).conj().detach();
  torch::Tensor inference;
  {
    c10::InferenceMode mode;
    inference = torch::zeros({2}, torch::kFloat64);
  }
  ASSERT_FALSE(nonleaf.is_leaf());
  ASSERT_TRUE(view_tensor.is_view());
  ASSERT_TRUE(negative.is_neg());
  ASSERT_TRUE(conjugate.is_conj());
  ASSERT_TRUE(inference.is_inference());
  const auto bytes = source(false);
  for (const auto& special :
       {nonleaf, view_tensor, negative, conjugate, inference}) {
    auto module = std::make_shared<torch::nn::Module>();
    const auto state = module->register_buffer("weight", special);
    const auto* data = state.data_ptr();
    nn::ModelStateView view(module);
    EXPECT_THROW(import_model_weights(view, bytes, hash(bytes)),
                 CompatibilityError);
    EXPECT_EQ(state.data_ptr(), data);
  }
  auto module = std::make_shared<Model>();
  nn::ModelStateView view(module);
  c10::InferenceMode mode;
  EXPECT_THROW(import_model_weights(view, bytes, hash(bytes)),
               CompatibilityError);
}

TEST(ModelWeightImportTest,
     BackendCommitFailureRestoresEvenThePartiallyChangedAssignment) {
  const auto faults = std::make_shared<CopyFaults>();
  faults->faults = {{1U, true, InjectedFailure::backend}};
  const auto module = std::make_shared<FaultModel>(faults);
  nn::ModelStateView view(module);
  const auto bytes = source();
  const auto* original_weight = module->weight.data_ptr();
  const auto* original_count = module->count.data_ptr();
  try {
    static_cast<void>(import_model_weights(view, bytes, hash(bytes)));
    FAIL() << "The injected commit failure must be reported";
  } catch (const BackendError& error) {
    EXPECT_EQ(error.details().component, "interop.model_weight_import");
    expect_nested_backend_cause(error);
  }
  EXPECT_EQ(faults->attempts, 2U);
  EXPECT_EQ(module->weight.data_ptr(), original_weight);
  EXPECT_EQ(module->count.data_ptr(), original_count);
  EXPECT_TRUE(torch::equal(module->weight, torch::zeros({2}, torch::kFloat64)));
  EXPECT_EQ(module->count.item<std::int64_t>(), 7);
}

TEST(ModelWeightImportTest,
     RollbackFailureStillRestoresOtherEntriesAndReportsUnusableState) {
  const auto faults = std::make_shared<CopyFaults>();
  faults->faults = {{1U, true, InjectedFailure::backend},
                    {2U, false, InjectedFailure::backend}};
  const auto module = std::make_shared<FaultModel>(faults);
  nn::ModelStateView view(module);
  const auto bytes = source();
  const auto* original_weight = module->weight.data_ptr();
  const auto* original_count = module->count.data_ptr();
  try {
    static_cast<void>(import_model_weights(view, bytes, hash(bytes)));
    FAIL() << "Incomplete rollback must invalidate the model explicitly";
  } catch (const InternalError& error) {
    EXPECT_EQ(error.details().operation, "rollback");
    EXPECT_EQ(error.details().actual, "restore_failures=1; attempted=2");
    EXPECT_NE(error.details().hint.find("Discard this model"),
              std::string::npos);
    expect_nested_backend_cause(error);
  }
  EXPECT_EQ(faults->attempts, 2U);
  EXPECT_NE(module->weight.data_ptr(), original_weight);
  EXPECT_TRUE(torch::equal(module->weight,
                           torch::tensor({1.25, -2.5}, torch::kFloat64)));
  EXPECT_EQ(module->count.data_ptr(), original_count);
  EXPECT_EQ(module->count.item<std::int64_t>(), 7);
}

TEST(ModelWeightImportTest,
     TypedMemoryFailuresUseFrameworkResourceErrorsAfterRollback) {
  for (const auto kind :
       {InjectedFailure::backend_oom, InjectedFailure::host_oom}) {
    const auto faults = std::make_shared<CopyFaults>();
    faults->faults = {{1U, true, kind}};
    const auto module = std::make_shared<FaultModel>(faults);
    nn::ModelStateView view(module);
    const auto bytes = source();
    try {
      static_cast<void>(import_model_weights(view, bytes, hash(bytes)));
      FAIL() << "Typed memory failure must be translated";
    } catch (const ResourceExhausted& error) {
      EXPECT_EQ(error.details().component, "interop.model_weight_import");
      if (kind == InjectedFailure::backend_oom) {
        expect_nested_backend_cause(error, true);
      } else {
        EXPECT_THROW(std::rethrow_if_nested(error), std::bad_alloc);
      }
    }
    EXPECT_TRUE(
        torch::equal(module->weight, torch::zeros({2}, torch::kFloat64)));
    EXPECT_EQ(module->count.item<std::int64_t>(), 7);
  }
}

#ifdef JYPPX_TEST_MODEL_IMPORT_CUDA
TEST(ModelWeightImportTest, SeparateOverlappingCudaStorageWrappersAreRejected) {
  ASSERT_TRUE(torch::cuda::is_available());
  const auto backing = torch::zeros(
      {3}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA));
  auto* data = backing.data_ptr<double>();
  auto module = std::make_shared<torch::nn::Module>();
  const auto options =
      torch::TensorOptions().dtype(torch::kFloat64).device(backing.device());
  const auto first =
      module->register_buffer("weight", torch::from_blob(data, {2}, options));
  const auto second = module->register_buffer(
      "retained", torch::from_blob(data + 1U, {2}, options));
  ASSERT_NE(first.storage().unsafeGetStorageImpl(),
            second.storage().unsafeGetStorageImpl());
  nn::ModelStateView view(module);
  const auto bytes = source(false);
  ModelWeightImportOptions import_options;
  import_options.mapping.mode = WeightImportMode::partial;
  import_options.mapping.minimum_tensor_coverage = 0.5;
  EXPECT_THROW(import_model_weights(view, bytes, hash(bytes), import_options),
               CompatibilityError);
  EXPECT_EQ(first.data_ptr(), data);
  EXPECT_EQ(second.data_ptr(), data + 1U);
  EXPECT_TRUE(torch::equal(backing.cpu(), torch::zeros({3}, torch::kFloat64)));
}

TEST(ModelWeightImportTest, AppliesActualCudaStateAfterCompleteDeviceStaging) {
  ASSERT_TRUE(torch::cuda::is_available());
  auto module = std::make_shared<Model>();
  module->to(torch::kCUDA);
  nn::ModelStateView view(module);
  const auto bytes = source();
  ASSERT_TRUE(import_model_weights(view, bytes, hash(bytes)).applied);
  EXPECT_TRUE(module->weight.is_cuda());
  EXPECT_TRUE(torch::equal(module->weight.cpu(),
                           torch::tensor({1.25, -2.5}, torch::kFloat64)));
  EXPECT_EQ(module->count.item<std::int64_t>(), 99);
}
#endif
}  // namespace
