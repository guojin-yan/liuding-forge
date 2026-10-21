// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
#include "../../../examples/serving/linear_bundle.hpp"

#include <torch/cuda.h>
#include <torch/script.h>

#include <array>
#include <gtest/gtest.h>
#include <sstream>
namespace {
using namespace jyppx;
using namespace jyppx::serving;
namespace example = jyppx::examples::serving;
struct Archive {
  std::vector<std::byte> bytes;
  ArtifactDigest digest;
};
Archive Compile() {
  torch::jit::Module module("ServingKnownVector");
  module.define(R"JIT(
def forward(self, x: Tensor) -> Tensor:
    return x * 2.0 + 1.0
)JIT");
  std::ostringstream output(std::ios::out | std::ios::binary);
  module.save(output);
  const auto serialized = output.str();
  const auto view = example::Bytes(serialized);
  return {.bytes = {view.begin(), view.end()}, .digest = example::Digest(view)};
}
RuntimeDescriptor Descriptor(Device device) {
  auto descriptor = example::Descriptor(device);
  descriptor.backend_id = "jyppx.torchscript.compatibility";
  descriptor.format = "torchscript";
  descriptor.tensor_batching = false;
  descriptor.max_batch_size = 1;
  descriptor.oom_retry_safe = false;
  return descriptor;
}
void Verify(Device device) {
  auto artifact = Compile();
  auto descriptor = Descriptor(device);
  interop::TorchScriptLoadOptions trust;
  trust.device = device;
  trust.trust_source = true;
  auto warmup = example::KnownVector(device);
  warmup.expected =
      torch::full({1, 2}, 3.0F, torch::TensorOptions().device(device));
  const std::array<WarmupCase, 1> cases{warmup};
  auto predictor = load_torchscript_predictor(
      artifact.bytes, artifact.digest, example::Input(device),
      example::Output(device), trust, descriptor, "example-linear-v1", cases);
  auto response = predictor->predict(example::Request(device));
  EXPECT_TRUE(torch::equal(response.output.value.toTensor().cpu(),
                           torch::full({1, 2}, 3.0F)));
  EXPECT_EQ(response.runtime.runtime.format, "torchscript");
  EXPECT_FALSE(response.runtime.runtime.oom_retry_safe);
  EXPECT_EQ(response.runtime.runtime.environment_lock,
            descriptor.environment_lock);
  EXPECT_NE(descriptor.environment_lock, artifact.digest);
  EXPECT_TRUE(predictor->close());
  EXPECT_TRUE(torch::equal(response.output.value.toTensor().cpu(),
                           torch::full({1, 2}, 3.0F)));
  trust.trust_source = false;
  EXPECT_THROW(load_torchscript_predictor(
                   artifact.bytes, artifact.digest, example::Input(device),
                   example::Output(device), trust, descriptor,
                   "example-linear-v1", cases),
               Error);
  trust.trust_source = true;
  descriptor.tensor_batching = true;
  descriptor.max_batch_size = 2;
  EXPECT_THROW(load_torchscript_predictor(
                   artifact.bytes, artifact.digest, example::Input(device),
                   example::Output(device), trust, descriptor,
                   "example-linear-v1", cases),
               InvalidArgument);
}
TEST(ServingTorchScript, RealTrustedCompatibilityLoadWarmupForwardCpu) {
  Verify(Device(c10::kCPU));
}
TEST(ServingTorchScript, RealTrustedCompatibilityLoadWarmupForwardCuda) {
  if (!torch::cuda::is_available())
    GTEST_SKIP() << "CUDA unavailable in this validation environment";
  Verify(Device(c10::kCUDA, 0));
}
}  // namespace
