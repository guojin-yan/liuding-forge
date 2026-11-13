// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_TRAINING_MUSGD_HPP
#define JYPPX_LIUDING_VISION_TASKS_TRAINING_MUSGD_HPP
/** @file musgd.hpp @brief Native momentum/orthogonalization plus delegated SGD.
 */
#include <jyppx/liuding/optim/sgd_adapter.hpp>
namespace jyppx::liuding::vision::tasks {
struct MuSgdOptions {
  double muon = .5;
  double sgd = .5;
  std::uint32_t newton_schulz_steps = 5;
  std::int64_t max_gram_elements = 16777216;
};
/** @brief Explicit FP32 Newton-Schulz MuSGD profile, with exact native state.
 * Rank>=2 parameters receive orthogonalized EMA momentum followed by weighted
 * SGD; vectors receive standard SGD. SGD decay sees post-Muon parameters.
 * Nesterov and momentum follow each frozen parameter group's recipe. This
 * native FP32 arithmetic profile is distinct from upstream BF16 qualification.
 */
class MuSgdOptimizer final : public optim::OptimizerAdapter,
                             public CheckpointParticipant {
 public:
  MuSgdOptimizer(std::string id, std::vector<optim::SgdParameterGroup> groups,
                 MuSgdOptions options = {});
  ~MuSgdOptimizer() override;
  Tensor step(LossClosure closure = {}) override;
  void zero_grad(bool set_to_none = true) override;
  void synchronize_parameter_placement() override;
  std::vector<optim::OptimizerParameterBinding> parameter_bindings()
      const override;
  std::size_t parameter_count() const noexcept override;
  std::uint64_t completed_steps() const noexcept override;
  const std::string& optimizer_id() const noexcept override;
  const std::vector<std::string>& parameter_group_ids() const noexcept override;
  std::vector<double> learning_rates() const override;
  void set_learning_rates(const std::vector<double>& rates) override;
  std::string participant_id() const override;
  StateManifest describe_state() const override;
  void save_state(StateWriter& writer) const override;
  void load_state(const StateReader& reader, RestorePolicy policy) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace jyppx::liuding::vision::tasks
#endif  // JYPPX_LIUDING_VISION_TASKS_TRAINING_MUSGD_HPP
