// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

#include <jyppx/liuding/deployment/onnx_runtime/typed_session.hpp>
#include <jyppx/liuding/vision/tasks/workflow/binding.hpp>

#include <torch/torch.h>

#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <nlohmann/json.hpp>

namespace {
namespace j = jyppx::liuding;
namespace t = j::vision::tasks;
using I = std::int64_t;
using J = nlohmann::json;

std::vector<std::byte> Read(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  const auto size = file.tellg();
  if (size < 1 || size > (4LL << 30))
    throw std::runtime_error("Invalid model size");
  std::vector<std::byte> bytes(static_cast<std::size_t>(size));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(bytes.data()), size);
  if (!file) throw std::runtime_error("Cannot read model");
  return bytes;
}

std::vector<j::Tensor> Native(
    t::TaskBinding& binding, const t::RecipeSpec& recipe,
    const std::vector<j::Tensor>& inputs,
    std::vector<j::Tensor>* diagnostic_raw = nullptr) {
  const auto& name = recipe.model.architecture;
  j::nn::Invocation invocation{{inputs[0]}, {}};
  if (name.starts_with("fast_rcnn_")) {
    const auto size =
        torch::tensor({float(inputs[0].size(2)), float(inputs[0].size(3))})
            .view({1, 2})
            .expand({inputs[0].size(0), 2});
    invocation.positional.insert(invocation.positional.end(),
                                 {inputs[1], torch::empty({0, 6}), size});
  } else if (name.starts_with("rtdetr_")) {
    invocation.positional.push_back(torch::empty({0, 6}));
  } else if (name.starts_with("yoloe_") || name.starts_with("yoloworld_")) {
    const bool visual = name.find("visual") != name.npos;
    const bool free = name.find("prompt_free") != name.npos;
    invocation.positional.push_back(free || visual ? torch::empty({0})
                                                   : inputs[1]);
    invocation.positional.push_back(visual ? inputs[1] : torch::empty({0}));
  } else if (recipe.model.task == t::TaskKind::pretrain_simmim) {
    invocation.positional.push_back(inputs[1].to(torch::kBool));
  }
  auto raw = t::raw_tensors(binding.model->invoke(invocation).value);
  if (diagnostic_raw) *diagnostic_raw = raw;
  if (name.starts_with("fast_rcnn_")) return {raw[3], raw[4], raw[5]};
  if (name.starts_with("rtdetr_")) return {raw[0][-1], raw[1][-1]};
  if (name.starts_with("retinanet_") || name.starts_with("ssd300_"))
    return {raw[0], raw[1], raw[3], raw[4]};
  if (recipe.model.task == t::TaskKind::reid) return {raw[1]};
  if (name.starts_with("yoloe_") || name.starts_with("yoloworld_"))
    raw.resize(recipe.model.task == t::TaskKind::instance_segment ? 2 : 1);
  if (recipe.model.task == t::TaskKind::classify ||
      recipe.model.task == t::TaskKind::classify_multilabel ||
      recipe.model.task == t::TaskKind::semantic_segment ||
      recipe.model.task == t::TaskKind::depth)
    raw.resize(1);
  return raw;
}

J TensorJson(const j::Tensor& tensor) {
  auto values = tensor.to(torch::kCPU, torch::kFloat64).contiguous();
  std::vector<double> data;
  if (values.numel())
    data.assign(values.const_data_ptr<double>(),
                values.const_data_ptr<double>() + values.numel());
  return {{"shape", tensor.sizes().vec()}, {"values", std::move(data)}};
}

void AlignQuerySet(
    std::vector<j::Tensor>& reference,
    const std::vector<j::onnx_runtime::TypedTensorOutput>& actual) {
  if (reference.size() != 2 || actual.size() != 2)
    throw std::runtime_error("RT-DETR output arity mismatch");
  std::vector<j::Tensor> values;
  for (const auto& output : actual) {
    const auto& data = std::get<std::vector<float>>(output.values);
    values.push_back(torch::from_blob(const_cast<float*>(data.data()),
                                      output.shape, torch::kFloat32));
    if (values.back().sizes() != reference[values.size() - 1].sizes())
      throw std::runtime_error("RT-DETR output shape mismatch");
  }
  // TopK ties may permute queries. Require a bijection on the concatenated
  // box and class vector; both outputs share the same permutation and
  // tolerance.
  const auto native = torch::cat(reference, -1);
  const auto runtime = torch::cat(values, -1);
  auto candidates = ((runtime.unsqueeze(2) - native.unsqueeze(1)).abs() <=
                     1e-4 + 1e-4 * native.unsqueeze(1).abs())
                        .all(-1)
                        .contiguous();
  const auto matches = candidates.accessor<bool, 3>();
  std::vector<std::vector<j::Tensor>> aligned(2);
  for (I batch = 0; batch < native.size(0); ++batch) {
    const I count = native.size(1);
    std::vector<I> owner(count, -1), permutation(count, -1);
    std::function<bool(I, std::vector<bool>&)> assign =
        [&](I row, std::vector<bool>& seen) {
          for (I col = 0; col < count; ++col) {
            if (!matches[batch][row][col] || seen[col]) continue;
            seen[col] = true;
            if (owner[col] < 0 || assign(owner[col], seen)) {
              owner[col] = row;
              return true;
            }
          }
          return false;
        };
    for (I row = 0; row < count; ++row) {
      std::vector<bool> seen(count);
      if (!assign(row, seen))
        throw std::runtime_error(
            "RT-DETR query sets have no numerical bijection");
    }
    for (I col = 0; col < count; ++col) permutation[owner[col]] = col;
    const auto indices = torch::tensor(permutation, torch::kInt64);
    for (std::size_t i = 0; i < 2; ++i)
      aligned[i].push_back(reference[i][batch].index_select(0, indices));
  }
  for (std::size_t i = 0; i < 2; ++i) reference[i] = torch::stack(aligned[i]);
}
}  // namespace

int main(int argc, char** argv) try {
  if (argc != 5)
    throw std::runtime_error("Expected recipe, bundle, weights, SHA256");
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);
  auto recipe = t::load_recipe(std::filesystem::u8path(argv[1]));
  recipe.device = j::Device(torch::kCPU);
  recipe.precision = j::training::PrecisionMode::full;
  auto binding = t::bind_task(recipe);
  t::InitializationOptions initialization;
  initialization.weights = std::filesystem::u8path(argv[3]);
  initialization.sha256 = argv[4];
  t::initialize_vision(*binding.model, initialization,
                       recipe.model.architecture);
  binding.model->state().eval();
  const auto root = std::filesystem::u8path(argv[2]);
  std::ifstream file(root / "manifest.json");
  J manifest;
  file >> manifest;
  const auto model = Read(root / "model.onnx");
  j::onnx_runtime::TypedSessionOptions limits;
  limits.max_model_bytes = 4ULL << 30;
  limits.max_input_bytes = limits.max_output_bytes = 1ULL << 30;
  j::onnx_runtime::TypedSession session(
      model, {.hex = manifest.at("sha256").get<std::string>()}, limits);
  const auto* diagnostic_directory = std::getenv("LIUDING_VISION_DIFF_DIR");
  bool failed = false;
  torch::NoGradGuard guard;
  for (const I seed : {17, 29}) {
    torch::manual_seed(seed);
    std::vector<j::Tensor> tensors;
    std::vector<j::onnx_runtime::TypedTensorInput> inputs;
    tensors.reserve(session.inputs().size());
    for (const auto& spec : session.inputs()) {
      auto tensor = torch::randn(spec.shape);
      if (spec.name == "patch_mask")
        tensor = (tensor > 0).to(torch::kInt64);
      else if (spec.name == "visual_masks")
        tensor = tensor.sigmoid();
      else if (spec.name == "proposals") {
        tensor = torch::empty(spec.shape);
        const I h = tensors[0].size(2), w = tensors[0].size(3);
        for (I row = 0; row < tensor.size(0); ++row)
          tensor[row].copy_(
              torch::tensor({0.f, float(row % 4), float(row % 5),
                             float(w - 4 - row % 3), float(h - 4 - row % 2)}));
      }
      tensors.push_back(tensor.contiguous());
      const auto& value = tensors.back();
      j::onnx_runtime::TypedInputValues values;
      if (value.scalar_type() == torch::kInt64)
        values = std::span(value.const_data_ptr<I>(),
                           static_cast<std::size_t>(value.numel()));
      else
        values = std::span(value.const_data_ptr<float>(),
                           static_cast<std::size_t>(value.numel()));
      inputs.push_back({spec.name, values, spec.shape});
    }
    std::vector<j::Tensor> raw;
    auto expected =
        Native(binding, recipe, tensors, diagnostic_directory ? &raw : nullptr);
    const auto actual = session.run(inputs);
    if (recipe.model.architecture.starts_with("rtdetr_"))
      AlignQuerySet(expected, actual);
    J diagnostic;
    if (diagnostic_directory) {
      for (const auto& tensor : tensors)
        diagnostic["inputs"].push_back(TensorJson(tensor));
      for (const auto& tensor : raw)
        diagnostic["raw"].push_back(TensorJson(tensor));
    }
    if (actual.size() != expected.size())
      throw std::runtime_error("Output arity mismatch");
    for (std::size_t i = 0; i < actual.size(); ++i) {
      std::visit(
          [&](const auto& values) {
            using V = typename std::decay_t<decltype(values)>::value_type;
            auto dtype = std::is_same_v<V, float> ? torch::kFloat32
                         : std::is_same_v<V, I>   ? torch::kInt64
                                                  : torch::kInt32;
            auto value = torch::from_blob(const_cast<V*>(values.data()),
                                          actual[i].shape, dtype);
            const auto& reference = expected[i];
            if (diagnostic_directory)
              diagnostic["outputs"].push_back(
                  {{"name", actual[i].name},
                   {"native", TensorJson(reference)},
                   {"ort", TensorJson(value)}});
            if (value.sizes() != reference.sizes() ||
                value.scalar_type() != reference.scalar_type())
              throw std::runtime_error("Output shape/type mismatch: " +
                                       actual[i].name);
            const bool floating = value.is_floating_point();
            if (floating && (!torch::isfinite(value).all().item<bool>() ||
                             !torch::isfinite(reference).all().item<bool>()))
              throw std::runtime_error("Nonfinite output: " + actual[i].name);
            if (floating && value.numel())
              std::cout << seed << ' ' << actual[i].name << " max_abs="
                        << (value - reference).abs().max().item<double>()
                        << '\n';
            if (floating ? !torch::allclose(value, reference, 1e-4, 1e-4)
                         : !torch::equal(value, reference)) {
              std::cerr << "ORT/native values differ: " << actual[i].name
                        << '\n';
              failed = true;
            }
          },
          actual[i].values);
    }
    if (diagnostic_directory) {
      const auto directory = std::filesystem::u8path(diagnostic_directory);
      std::filesystem::create_directories(directory);
      std::ofstream(directory / (recipe.model.architecture + "-" +
                                 std::to_string(seed) + ".json"))
          << diagnostic.dump();
    }
  }
  if (failed)
    throw std::runtime_error("Native/ORT numerical differential failed");
  std::cout << "Trained native/ORT differential passed: "
            << recipe.model.architecture << '\n';
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
