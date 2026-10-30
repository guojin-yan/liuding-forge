// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "reference_file.hpp"

#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/interop/safetensors.hpp>
#include <jyppx/liuding/models/models.hpp>

#include <torch/optim.h>
#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace jyppx::liuding;
using namespace jyppx::liuding::models;
struct Pin {
  const char* name;
  std::uint64_t state_bytes;
  const char* state_sha;
  std::uint64_t input_bytes;
  const char* input_sha;
};
constexpr Pin kPins[] = {
    {"vit_small_patch14_dinov2", 88240512,
     "cd1aa20613acd6d68210f84d170e1c3b5c16d67968c10c8918881a682f2971a9",
     1505368,
     "0ba7d144d17e1d8a0d97494357b80d8d16d7448339e7755b2b21d45d767a4632"},
    {"dinov3_vits16", 86477024,
     "74412baf39fda638532a3c1a8e469721f66763e3a0ca274854079ca643301e1d", 42080,
     "43ce9bd3c548f763a6e733f7f38e7d118d770feffb82781c6bbe812c559dcbf5"}};

Tensor Input(std::span<const std::byte> bytes) {
  const interop::SafeTensorsView view(bytes);
  if (view.directory().tensors.size() != 1U ||
      view.directory().tensors[0].name != "input" ||
      view.directory().tensors[0].dtype != interop::SafeTensorDtype::f32)
    throw std::runtime_error("Unexpected fixed input contract");
  std::vector<std::int64_t> shape;
  for (const auto extent : view.directory().tensors[0].shape)
    shape.push_back(static_cast<std::int64_t>(extent));
  const auto payload = view.tensor_bytes("input");
  return torch::from_blob(const_cast<std::byte*>(payload.data()), shape,
                          torch::TensorOptions().dtype(torch::kFloat32))
      .clone();
}

std::vector<Tensor> Invoke(nn::ModelAdapter& model, const Tensor& input) {
  const auto tuple = model.invoke({.positional = {input}}).value.toTuple();
  std::vector<Tensor> result;
  for (const auto& value : tuple->elements())
    result.push_back(value.toTensor());
  return result;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 4)
      throw std::runtime_error("Expected architecture fixtures output");
    const Pin* pin = nullptr;
    for (const auto& candidate : kPins)
      if (candidate.name == std::string(argv[1])) pin = &candidate;
    if (!pin)
      throw std::runtime_error(
          "Only two fixed Small CPU references are supported");
    const auto root = std::filesystem::u8path(argv[2]);
    const auto output = std::filesystem::u8path(argv[3]);
    if (std::filesystem::exists(output))
      throw std::runtime_error("Output must be new");
    torch::set_num_threads(2);
    torch::set_num_interop_threads(1);
    const test::MappedReference state(
        root / (std::string(pin->name) + "-state.safetensors"),
        pin->state_bytes);
    const test::MappedReference source(
        root / (std::string(pin->name) + "-input.safetensors"),
        pin->input_bytes);
    if (sha256_hex(state) != pin->state_sha ||
        sha256_hex(source) != pin->input_sha)
      throw std::runtime_error("Fixed source state/input digest differs");
    auto model = create_model(pin->name);
    const WeightRecord record{
        .id = "fixed-intermediate-random-state",
        .model_name = pin->name,
        .source_uri = "fixture://independent-pinned-random-state",
        .license_spdx = "LicenseRef-Generated-Test-Fixture",
        .license_uri = "fixture://models-reference/local-generated-state",
        .availability = WeightAvailability::pinned_external,
        .artifact = ArtifactRef{.digest = ArtifactDigest{.hex = pin->state_sha},
                                .size_bytes = pin->state_bytes,
                                .logical_name = std::string(pin->name) +
                                                "-state.safetensors",
                                .media_type = "application/x-safetensors",
                                .producer = "independent-model-reference"},
        .status_reason = "Random fixture only; no pretrained qualification"};
    if (!load_weights(model, record, state).applied)
      throw std::runtime_error("Strict fixed state import failed");
    model.state().eval();
    auto images = Input(source).requires_grad_(true);
    const DinoIntermediateOptions selection{.block_indices = {8, 2},
                                            .out_indices = {0, 1, 2}};
    const auto spec = model_intermediate_feature_spec(model, selection);
    auto normalized = create_intermediate_feature_extractor(model, selection);
    auto raw_selection = selection;
    raw_selection.normalize = false;
    auto raw = create_intermediate_feature_extractor(model, raw_selection);
    std::vector<Tensor> payloads;
    std::vector<std::string> names;
    {
      torch::NoGradGuard guard;
      const auto values = Invoke(raw, images);
      for (std::size_t index = 0; index < values.size(); ++index) {
        names.push_back("raw." + spec.outputs[index].name);
        payloads.push_back(values[index].contiguous());
      }
    }
    const auto values = Invoke(normalized, images);
    auto loss = torch::zeros({});
    for (std::size_t index = 0; index < values.size(); ++index) {
      const auto coefficients =
          torch::linspace(-0.3, 0.7, values[index].numel())
              .view(values[index].sizes());
      loss = loss + (values[index] * coefficients).mean() *
                        static_cast<double>(index + 1);
      names.push_back("normalized." + spec.outputs[index].name);
      payloads.push_back(values[index].detach().contiguous());
    }
    auto parameters = model.state().named_parameters();
    auto patch = parameters["patch_embed.proj.weight"];
    const auto before = patch.detach().clone();
    loss.backward();
    if (!patch.grad().defined() || !images.grad().defined() ||
        patch.grad().abs().sum().item<double>() <= 0.0)
      throw std::runtime_error("Missing real input/parameter gradient");
    for (const auto& parameter : parameters)
      if ((parameter.key().starts_with("blocks.9.") ||
           parameter.key().starts_with("blocks.10.") ||
           parameter.key().starts_with("blocks.11.")) &&
          parameter.value().grad().defined())
        throw std::runtime_error(
            "A block after the early-stop boundary received a gradient");
    names.push_back("input_gradient");
    payloads.push_back(images.grad().detach().contiguous());
    names.push_back("patch_gradient");
    payloads.push_back(patch.grad().detach().clone().contiguous());
    torch::optim::SGD optimizer(model.state().parameters(),
                                torch::optim::SGDOptions(0.01));
    optimizer.step();
    const auto delta = (patch.detach() - before).contiguous();
    if (delta.abs().sum().item<double>() <= 0.0)
      throw std::runtime_error("No real SGD patch update");
    names.push_back("patch_delta");
    payloads.push_back(delta);
    std::vector<interop::SafeTensorSource> tensors;
    for (std::size_t index = 0; index < payloads.size(); ++index) {
      const auto& tensor = payloads[index];
      if (!torch::isfinite(tensor).all().item<bool>())
        throw std::runtime_error("Nonfinite result");
      std::vector<std::uint64_t> shape;
      for (const auto extent : tensor.sizes())
        shape.push_back(static_cast<std::uint64_t>(extent));
      tensors.push_back(
          {.name = names[index],
           .dtype = interop::SafeTensorDtype::f32,
           .shape = std::move(shape),
           .data = {static_cast<const std::byte*>(tensor.data_ptr()),
                    static_cast<std::size_t>(tensor.numel()) * sizeof(float)}});
    }
    const auto bytes = interop::serialize_safetensors(tensors);
    std::ofstream stream(output, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    stream.close();
    if (!stream) throw std::runtime_error("Cannot write reference result");
    std::cout << "{\"actual_cpp_backward\":true,\"optimizer_steps\":1,"
                 "\"later_block_gradients\":0,\"scope\":\"small_cpu_random_"
                 "reference\"}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
