// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "checkpoint_codec.hpp"

#include <jyppx/liuding/nn/model_checkpoint_participant.hpp>
#include <jyppx/liuding/vision/tasks/training/musgd.hpp>

#include <cmath>
#include <map>
#include <nlohmann/json.hpp>
#include <set>
namespace jyppx::liuding::vision::tasks {
namespace {
class PrefixWriter : public StateWriter {
 public:
  StateWriter& target;
  std::string prefix;
  PrefixWriter(StateWriter& t, std::string p)
      : target(t), prefix(std::move(p)) {}
  void write_blob(std::string_view n, std::span<const std::byte> b) override {
    target.write_blob(prefix + std::string(n), b);
  }
};
class PrefixReader : public StateReader {
 public:
  const StateReader& source;
  StateManifest descriptor;
  std::string prefix;
  PrefixReader(const StateReader& s, StateManifest d, std::string p)
      : source(s), descriptor(std::move(d)), prefix(std::move(p)) {}
  const StateManifest& manifest() const noexcept override { return descriptor; }
  bool contains(std::string_view n) const override {
    return source.contains(prefix + std::string(n));
  }
  std::span<const std::byte> read_blob(std::string_view n,
                                       std::uint64_t max) const override {
    return source.read_blob(prefix + std::string(n), max);
  }
};
class Snapshot : public StateReader, public StateWriter {
 public:
  StateManifest descriptor;
  std::map<std::string, std::vector<std::byte>> blobs;
  std::uint64_t bytes = 0;
  explicit Snapshot(StateManifest d) : descriptor(std::move(d)) {}
  const StateManifest& manifest() const noexcept override { return descriptor; }
  bool contains(std::string_view n) const override {
    return blobs.contains(std::string(n));
  }
  std::span<const std::byte> read_blob(std::string_view n,
                                       std::uint64_t max) const override {
    const auto& v = blobs.at(std::string(n));
    if (v.size() > max) throw ResourceExhausted("MuSGD snapshot read bound");
    return v;
  }
  void write_blob(std::string_view n, std::span<const std::byte> b) override {
    if (b.size() > (4ULL << 30U) - bytes)
      throw ResourceExhausted("MuSGD rollback snapshot exceeds 4 GiB");
    blobs.emplace(n, std::vector<std::byte>(b.begin(), b.end()));
    bytes += b.size();
  }
};
}  // namespace
struct MuSgdOptimizer::Impl {
  std::string id, recipe;
  MuSgdOptions options;
  std::vector<optim::SgdParameterGroup> groups;
  std::shared_ptr<optim::SgdOptimizerAdapter> sgd;
  std::unique_ptr<optim::SgdOptimizerCheckpointParticipant> sgd_checkpoint;
  std::shared_ptr<torch::nn::Module> momentum =
      std::make_shared<torch::nn::Module>();
  std::unique_ptr<nn::ModelCheckpointParticipant> momentum_checkpoint;
  std::vector<Tensor> buffers;
  std::vector<std::string> ids;
  std::vector<double> rates;
  Impl(std::string value, std::vector<optim::SgdParameterGroup> g,
       MuSgdOptions o)
      : id(std::move(value)), options(o), groups(std::move(g)) {
    if (!state::Identifier(id) || !std::isfinite(o.muon) || o.muon < 0 ||
        !std::isfinite(o.sgd) || o.sgd <= 0 || o.newton_schulz_steps < 1 ||
        o.newton_schulz_steps > 16 || o.max_gram_elements < 1)
      throw InvalidArgument("Invalid MuSGD profile");
    // Split matrix and vector parameters so pure SGD vectors retain full LR.
    std::vector<optim::SgdParameterGroup> split;
    for (const auto& group : groups)
      for (bool matrix : {true, false}) {
        auto part = group;
        part.group_id += (matrix ? ".muon" : ".sgd");
        part.parameters.clear();
        for (const auto& p : group.parameters)
          if ((p.parameter.dim() >= 2) == matrix) part.parameters.push_back(p);
        if (part.parameters.empty()) continue;
        split.push_back(std::move(part));
      }
    groups = std::move(split);
    auto native = groups;
    std::size_t index = 0;
    for (auto& group : native) {
      ids.push_back(group.group_id);
      rates.push_back(group.options.learning_rate);
      const bool matrix = group.group_id.ends_with(".muon");
      if (matrix) group.options.learning_rate *= o.sgd;
      for (const auto& binding : group.parameters) {
        if (matrix) {
          if (binding.parameter.scalar_type() != torch::kFloat32 ||
              binding.parameter.layout() != torch::kStrided)
            throw InvalidArgument(
                "MuSGD native profile requires dense FP32 model parameters");
          const auto rows = binding.parameter.size(0),
                     cols = binding.parameter.numel() / rows,
                     minimum = (std::min)(rows, cols);
          if (minimum > o.max_gram_elements / minimum)
            throw ResourceExhausted("MuSGD Gram workspace bound exceeded");
          buffers.push_back(
              momentum->register_buffer("momentum_" + std::to_string(index++),
                                        torch::zeros_like(binding.parameter)));
        } else
          buffers.push_back(Tensor{});
      }
    }
    if (!index) momentum->register_buffer("empty", torch::zeros({1}));
    sgd = std::make_shared<optim::SgdOptimizerAdapter>(id + ".sgd",
                                                       std::move(native));
    sgd_checkpoint =
        std::make_unique<optim::SgdOptimizerCheckpointParticipant>(sgd);
    momentum_checkpoint = std::make_unique<nn::ModelCheckpointParticipant>(
        std::make_shared<nn::ModelStateView>(momentum),
        nn::ModelCheckpointParticipantOptions{.participant_id =
                                                  id + ".momentum"});
    recipe = nlohmann::json{"musgd/fp32-ns/v1", o.muon, o.sgd,
                            o.newton_schulz_steps, o.max_gram_elements}
                 .dump();
  }
};
MuSgdOptimizer::MuSgdOptimizer(std::string id,
                               std::vector<optim::SgdParameterGroup> groups,
                               MuSgdOptions options)
    : impl_(std::make_unique<Impl>(std::move(id), std::move(groups), options)) {
}
MuSgdOptimizer::~MuSgdOptimizer() = default;
Tensor MuSgdOptimizer::step(LossClosure closure) {
  Tensor loss;
  if (closure) loss = closure();
  auto& s = *impl_;
  torch::NoGradGuard guard;
  std::size_t index = 0;
  for (const auto& g : s.groups)
    for (const auto& p : g.parameters) {
      const auto grad = p.parameter.grad();
      if (grad.defined() &&
          (grad.is_sparse() || grad.device() != p.parameter.device() ||
           grad.sizes() != p.parameter.sizes() ||
           grad.scalar_type() != p.parameter.scalar_type()))
        throw InvalidArgument("MuSGD gradient topology mismatch");
    }
  for (std::size_t gi = 0; gi < s.groups.size(); ++gi)
    for (const auto& binding : s.groups[gi].parameters) {
      auto buffer = s.buffers[index++];
      const auto grad = binding.parameter.grad();
      if (!buffer.defined() || !grad.defined()) continue;
      const auto& o = s.groups[gi].options;
      buffer.mul_(o.momentum).add_(grad, 1 - o.momentum);
      auto update =
          o.nesterov ? buffer * o.momentum + grad * (1 - o.momentum) : buffer;
      auto matrix = update.reshape({update.size(0), -1});
      bool transposed = matrix.size(0) > matrix.size(1);
      if (transposed) matrix = matrix.transpose(0, 1);
      matrix = matrix / (matrix.norm() + 1e-7);
      for (std::uint32_t i = 0; i < s.options.newton_schulz_steps; ++i) {
        auto gram = matrix.matmul(matrix.transpose(0, 1));
        matrix = matrix * 3.4445 +
                 (-4.7750 * gram + 2.0315 * gram.matmul(gram)).matmul(matrix);
      }
      if (transposed) matrix = matrix.transpose(0, 1);
      const auto scale = std::sqrt(
          (std::max)(1., static_cast<double>(grad.size(-2)) / grad.size(-1)));
      binding.parameter.add_(matrix.reshape_as(binding.parameter),
                             -s.rates[gi] * s.options.muon * scale);
    }
  s.sgd->step();
  return loss;
}
void MuSgdOptimizer::zero_grad(bool none) { impl_->sgd->zero_grad(none); }
void MuSgdOptimizer::synchronize_parameter_placement() {
  auto& s = *impl_;
  s.sgd->synchronize_parameter_placement();
  std::size_t index = 0;
  for (const auto& g : s.groups)
    for (const auto& p : g.parameters) {
      auto& buffer = s.buffers[index++];
      if (buffer.defined() && buffer.device() != p.parameter.device())
        throw InvalidArgument(
            "Place MuSGD parameters before optimizer construction");
    }
}
std::vector<optim::OptimizerParameterBinding>
MuSgdOptimizer::parameter_bindings() const {
  return impl_->sgd->parameter_bindings();
}
std::size_t MuSgdOptimizer::parameter_count() const noexcept {
  return impl_->sgd->parameter_count();
}
std::uint64_t MuSgdOptimizer::completed_steps() const noexcept {
  return impl_->sgd->completed_steps();
}
const std::string& MuSgdOptimizer::optimizer_id() const noexcept {
  return impl_->id;
}
const std::vector<std::string>& MuSgdOptimizer::parameter_group_ids()
    const noexcept {
  return impl_->ids;
}
std::vector<double> MuSgdOptimizer::learning_rates() const {
  return impl_->rates;
}
void MuSgdOptimizer::set_learning_rates(const std::vector<double>& rates) {
  auto& s = *impl_;
  if (rates.size() != s.groups.size())
    throw InvalidArgument("MuSGD LR group count mismatch");
  auto native = rates;
  for (std::size_t i = 0; i < rates.size(); ++i) {
    if (!std::isfinite(rates[i]) || rates[i] < 0)
      throw InvalidArgument("Invalid MuSGD LR");
    if (s.groups[i].group_id.ends_with(".muon")) native[i] *= s.options.sgd;
  }
  s.sgd->set_learning_rates(native);
  s.rates = rates;
}
std::string MuSgdOptimizer::participant_id() const { return impl_->id; }
StateManifest MuSgdOptimizer::describe_state() const {
  const auto& s = *impl_;
  StateManifest m{.participant_id = s.id,
                  .fields = {{"recipe", "application/json", 65536}}};
  for (auto [participant, prefix] :
       std::vector<std::pair<const CheckpointParticipant*, std::string>>{
           {s.sgd_checkpoint.get(), "sgd_"},
           {s.momentum_checkpoint.get(), "muon_"}})
    for (auto field : participant->describe_state().fields) {
      field.name = prefix + field.name;
      m.fields.push_back(std::move(field));
    }
  return m;
}
void MuSgdOptimizer::save_state(StateWriter& writer) const {
  auto& s = *impl_;
  writer.write_blob("recipe", std::as_bytes(std::span(s.recipe)));
  PrefixWriter a(writer, "sgd_"), b(writer, "muon_");
  s.sgd_checkpoint->save_state(a);
  s.momentum_checkpoint->save_state(b);
}
void MuSgdOptimizer::load_state(const StateReader& reader,
                                RestorePolicy policy) {
  auto& s = *impl_;
  state::VerifyManifest(reader.manifest(), describe_state(), policy);
  state::VerifyText(reader, "recipe", s.recipe, 65536);
  Snapshot old_sgd(s.sgd_checkpoint->describe_state()),
      old_momentum(s.momentum_checkpoint->describe_state());
  s.sgd_checkpoint->save_state(old_sgd);
  s.momentum_checkpoint->save_state(old_momentum);
  PrefixReader a(reader, old_sgd.descriptor, "sgd_"),
      b(reader, old_momentum.descriptor, "muon_");
  try {
    s.sgd_checkpoint->load_state(a, policy);
    s.momentum_checkpoint->load_state(b, policy);
  } catch (...) {
    s.sgd_checkpoint->load_state(old_sgd, RestorePolicy::exact_only);
    s.momentum_checkpoint->load_state(old_momentum, RestorePolicy::exact_only);
    throw;
  }
  auto rates = s.sgd->learning_rates();
  for (std::size_t i = 0; i < rates.size(); ++i)
    if (s.groups[i].group_id.ends_with(".muon")) rates[i] /= s.options.sgd;
  s.rates = std::move(rates);
}
}  // namespace jyppx::liuding::vision::tasks
