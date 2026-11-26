// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_OPTIM_DETAIL_ADAMW_BATCH_STEP_HPP
#define JYPPX_LIUDING_OPTIM_DETAIL_ADAMW_BATCH_STEP_HPP

#include <torch/optim/adamw.h>
#include <torch/torch.h>

#include <ATen/ops/_foreach_add.h>
#include <ATen/ops/_foreach_addcdiv.h>
#include <ATen/ops/_foreach_addcmul.h>
#include <ATen/ops/_foreach_div.h>
#include <ATen/ops/_foreach_mul.h>
#include <ATen/ops/_foreach_sqrt.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace jyppx::liuding::optim::adamw_detail {

// Keep the native optimizer's state, scalar arithmetic, and operation order.
// Only independent pointwise operations on eligible tensors are batched.
inline bool try_batched_step(torch::optim::AdamW& optimizer) {
  torch::NoGradGuard guard;
  struct Group {
    std::vector<torch::Tensor> parameters;
    std::int64_t step = -1;
    double lr, beta1, beta2, epsilon, decay;
  };
  std::vector<Group> groups;
  std::vector<std::pair<std::uintptr_t, std::uintptr_t>> ranges;
  std::optional<torch::Device> device;
  std::size_t active = 0;
  const auto record_range = [&](const torch::Tensor& tensor) {
    const auto begin =
        reinterpret_cast<std::uintptr_t>(tensor.const_data_ptr());
    const auto bytes = static_cast<std::uintptr_t>(tensor.nbytes());
    if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes)
      return false;
    ranges.emplace_back(begin, begin + bytes);
    return true;
  };
  for (const auto& native_group : optimizer.param_groups()) {
    const auto& options =
        static_cast<const torch::optim::AdamWOptions&>(native_group.options());
    if (options.amsgrad()) return false;
    const auto [beta1, beta2] = options.betas();
    Group group{.parameters = {},
                .step = -1,
                .lr = options.lr(),
                .beta1 = beta1,
                .beta2 = beta2,
                .epsilon = options.eps(),
                .decay = options.weight_decay()};
    for (const auto& parameter : native_group.params()) {
      if (!parameter.is_cuda() || parameter.scalar_type() != torch::kFloat32 ||
          !parameter.is_contiguous() || parameter.numel() == 0)
        return false;
      if (!device) device = parameter.device();
      if (parameter.device() != *device || !record_range(parameter))
        return false;
      const auto& gradient = parameter.grad();
      if (!gradient.defined()) continue;
      if (!gradient.is_contiguous() || !record_range(gradient)) return false;
      std::int64_t step = 0;
      const auto found =
          optimizer.state().find(parameter.unsafeGetTensorImpl());
      if (found != optimizer.state().end()) {
        const auto& state =
            static_cast<const torch::optim::AdamWParamState&>(*found->second);
        step = state.step();
        for (const auto& moment : {state.exp_avg(), state.exp_avg_sq()}) {
          if (!moment.defined() || moment.device() != parameter.device() ||
              moment.scalar_type() != parameter.scalar_type() ||
              moment.sizes() != parameter.sizes() || !moment.is_contiguous())
            return false;
        }
      }
      if (step == std::numeric_limits<std::int64_t>::max() ||
          (group.step != -1 && group.step != step))
        return false;
      group.step = step;
      group.parameters.push_back(parameter);
      ++active;
    }
    if (!group.parameters.empty()) groups.push_back(std::move(group));
  }
  if (active < 2) return false;
  // Parameter views or gradients may share storage. Retain serial native
  // semantics whenever any participating memory intervals overlap.
  std::sort(ranges.begin(), ranges.end());
  for (std::size_t index = 1; index < ranges.size(); ++index)
    if (ranges[index].first < ranges[index - 1].second) return false;

  for (const auto& group : groups) {
    std::vector<torch::Tensor> gradients, means, variances;
    gradients.reserve(group.parameters.size());
    means.reserve(group.parameters.size());
    variances.reserve(group.parameters.size());
    for (const auto& parameter : group.parameters) {
      const auto key = parameter.unsafeGetTensorImpl();
      if (optimizer.state().find(key) == optimizer.state().end()) {
        auto state = std::make_unique<torch::optim::AdamWParamState>();
        state->exp_avg(torch::zeros_like(parameter));
        state->exp_avg_sq(torch::zeros_like(parameter));
        optimizer.state()[key] = std::move(state);
      }
      auto& state = static_cast<torch::optim::AdamWParamState&>(
          *optimizer.state().at(key));
      state.step(group.step + 1);
      gradients.push_back(parameter.grad());
      means.push_back(state.exp_avg());
      variances.push_back(state.exp_avg_sq());
    }
    const auto correction1 = 1 - std::pow(group.beta1, group.step + 1);
    const auto correction2 =
        std::sqrt(1 - std::pow(group.beta2, group.step + 1));
    // Bound denominator scratch to 16 MiB or one larger tensor, instead of
    // allocating a model-sized temporary as an unbounded foreach step does.
    constexpr std::size_t kScratchBytes = 16U << 20U;
    for (std::size_t first = 0; first < group.parameters.size();) {
      auto end = first;
      std::size_t bytes = 0;
      while (end < group.parameters.size()) {
        const auto next =
            static_cast<std::size_t>(group.parameters[end].nbytes());
        if (end != first && next > kScratchBytes - bytes) break;
        bytes += next;
        ++end;
        if (bytes >= kScratchBytes) break;
      }
      const auto count = end - first;
      const auto p = at::TensorList(group.parameters).slice(first, count);
      const auto g = at::TensorList(gradients).slice(first, count);
      const auto m = at::TensorList(means).slice(first, count);
      const auto v = at::TensorList(variances).slice(first, count);
      at::_foreach_mul_(p, 1 - group.lr * group.decay);
      at::_foreach_mul_(m, group.beta1);
      at::_foreach_add_(m, g, 1 - group.beta1);
      at::_foreach_mul_(v, group.beta2);
      at::_foreach_addcmul_(v, g, g, 1 - group.beta2);
      auto denominators = at::_foreach_sqrt(v);
      at::_foreach_div_(denominators, correction2);
      at::_foreach_add_(denominators, group.epsilon);
      at::_foreach_addcdiv_(p, m, denominators, -group.lr / correction1);
      first = end;
    }
  }
  return true;
}

class BatchedAdamW final : public torch::optim::AdamW {
 public:
  using torch::optim::AdamW::AdamW;

  torch::Tensor step(LossClosure closure = nullptr) override {
    if (!closure && try_batched_step(*this)) return {};
    return torch::optim::AdamW::step(std::move(closure));
  }
};

}  // namespace jyppx::liuding::optim::adamw_detail
#endif  // JYPPX_LIUDING_OPTIM_DETAIL_ADAMW_BATCH_STEP_HPP
