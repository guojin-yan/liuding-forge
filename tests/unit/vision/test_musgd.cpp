// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "musgd_reference.hpp"
#include "musgd_separate_reference.hpp"

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/vision/tasks/training/musgd.hpp>
#include <jyppx/liuding/vision/tasks/workflow/recipe.hpp>

#include <torch/torch.h>

#include <gtest/gtest.h>
#include <map>
#include <nlohmann/json.hpp>

namespace jyppx::liuding::vision::tests {
namespace {
namespace t = tasks;
using J = nlohmann::json;

class MuSgdState final : public StateReader, public StateWriter {
 public:
  explicit MuSgdState(const t::MuSgdOptimizer& optimizer)
      : descriptor(optimizer.describe_state()) {
    optimizer.save_state(*this);
  }
  const StateManifest& manifest() const noexcept override { return descriptor; }
  bool contains(std::string_view name) const override {
    return blobs.contains(std::string(name));
  }
  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t maximum) const override {
    const auto& bytes = blobs.at(std::string(name));
    if (bytes.size() > maximum) throw ResourceExhausted("fixture read bound");
    return bytes;
  }
  void write_blob(std::string_view name,
                  std::span<const std::byte> bytes) override {
    blobs.emplace(std::string(name),
                  std::vector<std::byte>(bytes.begin(), bytes.end()));
  }
  StateManifest descriptor;
  std::map<std::string, std::vector<std::byte>> blobs;
};

struct MuSgdDomain {
  std::vector<Tensor> parameters;
  std::unique_ptr<t::MuSgdOptimizer> optimizer;
  explicit MuSgdDomain(bool channels_last = false,
                       t::MuSgdMath math = t::MuSgdMath::bf16_batched,
                       std::string_view reference = musgd_reference,
                       Device device = Device(torch::kCPU)) {
    const auto fixture = J::parse(reference);
    std::vector<optim::SgdParameterGroup> groups(2);
    groups[0].group_id = "matrix";
    groups[1].group_id = "vector";
    for (auto& g : groups)
      g.options = {.learning_rate = .012,
                   .momentum = .83,
                   .weight_decay = .017,
                   .nesterov = true};
    for (std::size_t i = 0; i < fixture["shapes"].size(); ++i) {
      auto p =
          torch::tensor(fixture["initial"][i].get<std::vector<float>>())
              .reshape(fixture["shapes"][i].get<std::vector<std::int64_t>>())
              .to(device);
      if (channels_last && p.dim() == 4)
        p = p.contiguous(torch::MemoryFormat::ChannelsLast);
      p.set_requires_grad(true);
      parameters.push_back(p);
      groups[p.dim() >= 2 ? 0 : 1].parameters.push_back(
          {.parameter_id = "parameter_" + std::to_string(i), .parameter = p});
    }
    optimizer = std::make_unique<t::MuSgdOptimizer>(
        "optimizer.musgd.fixture", std::move(groups),
        t::MuSgdOptions{.muon = .528, .sgd = .674, .math = math});
  }
  void step(const J& row) {
    for (std::size_t i = 0; i < parameters.size(); ++i) {
      if (row["gradients"][i].is_null()) {
        parameters[i].mutable_grad() = Tensor{};
        continue;
      }
      auto g = torch::tensor(row["gradients"][i].get<std::vector<float>>())
                   .reshape(parameters[i].sizes())
                   .to(parameters[i].device());
      if (parameters[i].dim() == 4 && !parameters[i].is_contiguous())
        g = g.contiguous(torch::MemoryFormat::ChannelsLast);
      parameters[i].mutable_grad() = g;
    }
    optimizer->set_learning_rates(row["rates"].get<std::vector<double>>());
    optimizer->step();
  }
};

void CheckSeparateReference(const Device& device) {
  const auto rows =
      J::parse(musgd_separate_reference)["variants"]["contiguous"];
  MuSgdDomain domain(false, t::MuSgdMath::bf16_separate,
                     musgd_separate_reference, device);
  for (const auto& row : rows) {
    domain.step(row);
    for (std::size_t i = 0; i < domain.parameters.size(); ++i) {
      // The fixed CPU convolution reference does not define GPU norm rounding.
      // GPU matrix/vector arithmetic uses the shared portable reference below.
      if (device.is_cuda() && domain.parameters[i].dim() > 2) continue;
      const auto expected =
          torch::tensor(row["parameters"][i].get<std::vector<float>>());
      EXPECT_TRUE(torch::allclose(domain.parameters[i].detach().cpu().flatten(),
                                  expected, 1e-6, 1e-7))
          << i;
    }
  }
  EXPECT_EQ(domain.optimizer->completed_steps(), 3);
}

TEST(MuSgdBf16, SeparateCpuMatchesEarlierOfficialDenseReference) {
  torch::set_num_threads(1);
  CheckSeparateReference(Device(torch::kCPU));
}

TEST(MuSgdBf16, SeparateCudaMatchesEarlierOfficialMatrixAndVectorReference) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA backend unavailable";
  CheckSeparateReference(Device(torch::kCUDA, 0));
}

TEST(MuSgdBf16, SeparateRestoresMomentumAndRejectsOtherProfiles) {
  const auto rows =
      J::parse(musgd_separate_reference)["variants"]["contiguous"];
  MuSgdDomain original(false, t::MuSgdMath::bf16_separate,
                       musgd_separate_reference);
  original.step(rows[0]);
  const MuSgdState saved(*original.optimizer);
  MuSgdDomain restored(false, t::MuSgdMath::bf16_separate,
                       musgd_separate_reference);
  {
    torch::NoGradGuard guard;
    for (std::size_t i = 0; i < original.parameters.size(); ++i)
      restored.parameters[i].copy_(original.parameters[i]);
  }
  restored.optimizer->load_state(saved, RestorePolicy::exact_only);
  for (std::size_t step = 1; step < rows.size(); ++step) {
    original.step(rows[step]);
    restored.step(rows[step]);
    for (std::size_t i = 0; i < original.parameters.size(); ++i)
      EXPECT_TRUE(torch::equal(original.parameters[i], restored.parameters[i]));
    EXPECT_EQ(MuSgdState(*original.optimizer).blobs,
              MuSgdState(*restored.optimizer).blobs);
  }
  for (const auto math :
       {t::MuSgdMath::fp32_native, t::MuSgdMath::bf16_batched}) {
    MuSgdDomain other(false, math, musgd_separate_reference);
    const MuSgdState before(*other.optimizer),
        separate_before(*restored.optimizer);
    EXPECT_THROW(other.optimizer->load_state(saved, RestorePolicy::exact_only),
                 Error);
    EXPECT_EQ(MuSgdState(*other.optimizer).blobs, before.blobs);
    EXPECT_THROW(
        restored.optimizer->load_state(before, RestorePolicy::exact_only),
        Error);
    EXPECT_EQ(MuSgdState(*restored.optimizer).blobs, separate_before.blobs);
  }
}

TEST(MuSgdBf16, SeparateRejectsMatrixWorkspaceAtConstruction) {
  auto p = torch::ones({4, 4}, torch::requires_grad());
  EXPECT_THROW(
      (t::MuSgdOptimizer(
          "optimizer.bounded",
          {{.group_id = "matrix", .parameters = {{"p", p}}}},
          {.max_gram_elements = 10, .math = t::MuSgdMath::bf16_separate})),
      ResourceExhausted);
  EXPECT_TRUE(torch::equal(p, torch::ones_like(p)));
}

TEST(MuSgdBf16, OfficialCpuReferenceAndChannelsLast) {
  torch::set_num_threads(1);
  const auto fixture = J::parse(musgd_reference);
  for (const std::string layout : {"contiguous", "channels_last"}) {
    MuSgdDomain domain(layout == "channels_last");
    for (const auto& row : fixture["variants"][layout]) {
      domain.step(row);
      for (std::size_t i = 0; i < domain.parameters.size(); ++i) {
        const auto expected =
            torch::tensor(row["parameters"][i].get<std::vector<float>>());
        EXPECT_TRUE(torch::allclose(domain.parameters[i].detach().flatten(),
                                    expected, 1e-6, 1e-7))
            << layout << " parameter " << i;
      }
    }
    EXPECT_EQ(domain.optimizer->completed_steps(), 3);
  }
}

TEST(MuSgdBf16, RestoresMomentumRatesAndMissingGradientContinuation) {
  torch::set_num_threads(1);
  const auto rows = J::parse(musgd_reference)["variants"]["contiguous"];
  MuSgdDomain original;
  original.step(rows[0]);
  original.optimizer->set_momentums(
      std::vector<double>(original.optimizer->momentums().size(), .6576));
  const MuSgdState saved(*original.optimizer);
  MuSgdDomain restored;
  {
    torch::NoGradGuard guard;
    for (std::size_t i = 0; i < original.parameters.size(); ++i)
      restored.parameters[i].copy_(original.parameters[i]);
  }
  restored.optimizer->load_state(saved, RestorePolicy::exact_only);
  EXPECT_EQ(
      restored.optimizer->momentums(),
      (std::vector<double>(restored.optimizer->momentums().size(), .6576)));
  for (std::size_t step = 1; step < rows.size(); ++step) {
    original.step(rows[step]);
    restored.step(rows[step]);
    for (std::size_t i = 0; i < original.parameters.size(); ++i)
      EXPECT_TRUE(torch::equal(original.parameters[i], restored.parameters[i]));
    EXPECT_EQ(MuSgdState(*original.optimizer).blobs,
              MuSgdState(*restored.optimizer).blobs);
  }
}

TEST(MuSgdBf16, RejectsOtherMathProfileWithoutChangingState) {
  const auto row = J::parse(musgd_reference)["variants"]["contiguous"][0];
  MuSgdDomain native(false, t::MuSgdMath::fp32_native), bf16;
  native.step(row);
  bf16.step(row);
  const MuSgdState saved_native(*native.optimizer), saved_bf16(*bf16.optimizer);
  EXPECT_THROW(
      bf16.optimizer->load_state(saved_native, RestorePolicy::exact_only),
      Error);
  EXPECT_EQ(MuSgdState(*bf16.optimizer).blobs, saved_bf16.blobs);
  EXPECT_THROW(
      native.optimizer->load_state(saved_bf16, RestorePolicy::exact_only),
      Error);
  EXPECT_EQ(MuSgdState(*native.optimizer).blobs, saved_native.blobs);
}

TEST(MuSgdBf16, RejectsBatchedWorkspaceBeforeChangingMomentumOrParameters) {
  auto a = torch::ones({3, 3}, torch::requires_grad());
  auto b = torch::ones({3, 3}, torch::requires_grad());
  t::MuSgdOptimizer optimizer(
      "optimizer.bounded",
      {{.group_id = "matrix",
        .parameters = {{"a", a}, {"b", b}},
        .options = {.learning_rate = .01, .momentum = .9}}},
      {.max_gram_elements = 10, .math = t::MuSgdMath::bf16_batched});
  a.mutable_grad() = torch::ones_like(a);
  b.mutable_grad() = torch::ones_like(b);
  const MuSgdState before(optimizer);
  EXPECT_THROW(optimizer.step(), ResourceExhausted);
  EXPECT_EQ(MuSgdState(optimizer).blobs, before.blobs);
  EXPECT_TRUE(torch::equal(a, torch::ones_like(a)));
  EXPECT_TRUE(torch::equal(b, torch::ones_like(b)));
  EXPECT_EQ(optimizer.completed_steps(), 0);
}

TEST(MuSgdBf16, RecipeRoundTripAndDefaultIdentityBoundary) {
  t::RecipeSpec recipe;
  recipe.schema_version = 2;
  recipe.extension.training_manifest =
      std::filesystem::absolute("fixture.json");
  recipe.optimizer_name = "musgd";
  const auto original = t::recipe_identity(recipe);
  EXPECT_EQ(t::recipe_json(recipe).find("muon_weight"), std::string::npos);
  recipe.optimizer_name = "musgd_bf16";
  recipe.musgd_muon_weight = .528;
  recipe.musgd_sgd_weight = .674;
  EXPECT_NE(t::recipe_identity(recipe), original);
  const auto restored =
      t::parse_recipe(t::recipe_json(recipe), std::filesystem::current_path());
  EXPECT_EQ(t::recipe_identity(restored), t::recipe_identity(recipe));
  EXPECT_DOUBLE_EQ(restored.musgd_muon_weight, .528);
  const auto batched_identity = t::recipe_identity(recipe);
  recipe.optimizer_name = "musgd_bf16_separate";
  const auto separate_json = t::recipe_json(recipe);
  EXPECT_NE(t::recipe_identity(recipe), batched_identity);
  const auto separate =
      t::parse_recipe(separate_json, std::filesystem::current_path());
  EXPECT_EQ(t::recipe_identity(separate), t::recipe_identity(recipe));
  // The existing batched profile retains its recipe identity.
  recipe.optimizer_name = "musgd_bf16";
  EXPECT_EQ(t::recipe_identity(recipe), batched_identity);
  auto classification = recipe;
  classification.model.architecture = "resnet18";
  classification.model.task = t::TaskKind::classify;
  const auto canonical = t::recipe_json(classification);
  EXPECT_EQ(t::recipe_identity(classification),
            sha256_hex(std::as_bytes(std::span(canonical))));
  recipe.optimizer_name = "adamw";
  EXPECT_THROW(t::check_recipe(recipe), InvalidArgument);
}
}  // namespace
}  // namespace jyppx::liuding::vision::tests
