// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

#include <jyppx/liuding/vision/tasks/workflow/binding.hpp>

#include <torch/torch.h>

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

namespace {
namespace j = jyppx::liuding;
namespace t = j::vision::tasks;
using J = nlohmann::json;

j::Tensor Tensor(const J& record) {
  return torch::tensor(record.at("values").get<std::vector<float>>())
      .reshape(record.at("shape").get<std::vector<std::int64_t>>());
}

void Compare(const j::Tensor& actual, const J& record, const std::string& name,
             double relative = 1e-4, double absolute = 1e-5) {
  const auto expected = Tensor(record);
  if (!actual.defined() || actual.sizes() != expected.sizes() ||
      !torch::isfinite(actual).all().item<bool>())
    throw std::runtime_error(name + ": invalid shape or nonfinite result");
  const auto error = (actual - expected).abs().max().item<double>();
  std::cout << name << " max_abs=" << error << '\n';
  if (!torch::allclose(actual, expected, relative, absolute)) {
    const auto index =
        (actual - expected).abs().flatten().argmax().item<std::int64_t>();
    std::cerr << name << " index=" << index
              << " actual=" << actual.flatten()[index].item<double>()
              << " expected=" << expected.flatten()[index].item<double>()
              << '\n';
    throw std::runtime_error(name + ": independent reference differs");
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2 && argc != 3)
      throw std::runtime_error("Expected an independent fixture JSON path");
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);
    std::ifstream file(std::filesystem::u8path(argv[1]));
    J fixture;
    file >> fixture;
    t::RecipeSpec recipe;
    recipe.model.architecture = fixture.at("architecture").get<std::string>();
    const auto task = fixture.at("task").get<std::string>();
    recipe.model.task =
        t::parse_task(task == "instance_segment" ? "segment" : task);
    recipe.model.variant = fixture.value("variant", "p5");
    recipe.model.num_classes = fixture.value("classes", 3);
    recipe.training.source = "__independent_reference__";
    recipe.classification_root = "__independent_reference__";
    auto binding = t::bind_task(recipe);
    const auto state =
        std::filesystem::u8path(fixture.at("state").get<std::string>());
    const auto digest = fixture.at("state_sha256").get<std::string>();
    const auto mapping = J::parse(t::propose_vision_weight_mapping(
        *binding.model, state, digest, recipe.model.architecture));
    if (!mapping.at("complete").get<bool>()) {
      std::cerr << mapping.dump(2) << '\n';
      throw std::runtime_error(
          "Independent state inventory is not a complete match");
    }
    t::InitializationOptions initialization;
    initialization.weights = state;
    initialization.sha256 = digest;
    initialization.key_mapping =
        mapping.at("key_mapping").get<std::map<std::string, std::string>>();
    t::initialize_vision(*binding.model, initialization,
                         recipe.model.architecture);
    std::int64_t elements = 0;
    for (const auto& parameter : binding.model->state().parameters())
      elements += parameter.numel();
    if (elements != fixture.at("parameter_elements").get<std::int64_t>())
      throw std::runtime_error("Independent parameter count differs");
    binding.model->state().eval();
    auto input = Tensor(fixture.at("input")).requires_grad_(true);
    const auto outputs =
        t::raw_tensors(binding.model->invoke({{input}, {}}).value);
    if (outputs.size() != fixture.at("outputs").size())
      throw std::runtime_error("Independent output arity differs");
    auto objective = outputs.front().sum() * 0;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      Compare(outputs[i], fixture.at("outputs").at(i),
              "output-" + std::to_string(i));
      const auto weight =
          fixture.contains("objective_weights")
              ? fixture.at("objective_weights").at(i).get<double>()
              : static_cast<double>(i + 1);
      objective = objective + weight * outputs[i].square().mean();
    }
    objective.backward();
    Compare(input.grad(), fixture.at("input_gradient"), "input-gradient", 1e-3);
    const auto parameters = binding.model->state().named_parameters();
    if (argc == 3) {
      J diagnostic;
      for (const auto& [source_name, record] :
           fixture.at("gradients").items()) {
        const auto value =
            parameters[initialization.key_mapping.at(source_name)]
                .grad()
                .flatten()
                .contiguous();
        diagnostic[source_name] =
            std::vector<float>(value.const_data_ptr<float>(),
                               value.const_data_ptr<float>() + value.numel());
      }
      std::ofstream(std::filesystem::u8path(argv[2])) << diagnostic.dump(2);
    }
    for (const auto& [source_name, gradient] :
         fixture.at("gradients").items()) {
      const auto name = initialization.key_mapping.at(source_name);
      Compare(parameters[name].grad(), gradient, name + ".gradient", 1e-3);
    }
    torch::optim::SGD optimizer(binding.model->state().parameters(),
                                torch::optim::SGDOptions(.001));
    optimizer.step();
    for (const auto& [source_name, expected] : fixture.at("updated").items()) {
      const auto name = initialization.key_mapping.at(source_name);
      Compare(parameters[name], expected, name + ".updated");
    }
    std::cout << "independent state/output/backward/SGD passed: "
              << recipe.model.architecture << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
