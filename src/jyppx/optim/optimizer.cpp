#include <jyppx/optim/optimizer.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace jyppx {
namespace optim {

namespace {

constexpr std::string_view kFormatVersion = "__optimizer_format_version";
constexpr std::string_view kParamGroupCount = "__optimizer_param_group_count";
constexpr std::string_view kParameterCount = "__optimizer_parameter_count";

std::string parameter_state_count_key(size_t parameter_index) {
  return "__optimizer_state_count_" + std::to_string(parameter_index);
}

int64_t scalar_int64(const Tensor& value, const char* name) {
  if (!value.defined() || value.numel() != 1 ||
      value.scalar_type() != torch::kInt64) {
    throw std::invalid_argument(std::string("Optimizer ") + name +
                                " must be an int64 scalar tensor");
  }
  const int64_t numeric = value.item<int64_t>();
  if (numeric < 0) {
    throw std::invalid_argument(std::string("Optimizer ") + name +
                                " must be a non-negative integer");
  }
  return numeric;
}

void validate_parameter_topology(
    const TensorList& params,
    std::unordered_set<const c10::TensorImpl*>& seen) {
  for (const auto& parameter : params) {
    if (!parameter.defined()) {
      throw std::invalid_argument(
          "Optimizer parameters must be defined tensors");
    }
    if (!seen.insert(parameter.unsafeGetTensorImpl()).second) {
      throw std::invalid_argument(
          "A parameter cannot appear more than once in an optimizer");
    }
  }
}

}  // namespace

Optimizer::Optimizer(const TensorList& params) {
  if (params.empty()) {
    throw std::invalid_argument("Optimizer requires at least one parameter");
  }
  std::unordered_set<const c10::TensorImpl*> seen;
  validate_parameter_topology(params, seen);
  param_groups_.emplace_back(params);
}

void Optimizer::add_param_group(ParamGroup group) {
  if (group.params.empty()) {
    throw std::invalid_argument(
        "Parameter group requires at least one parameter");
  }

  std::unordered_set<const c10::TensorImpl*> known_parameters;
  for (const auto& existing_group : param_groups_) {
    for (const auto& parameter : existing_group.params) {
      if (parameter.defined()) {
        known_parameters.insert(parameter.unsafeGetTensorImpl());
      }
    }
  }

  std::unordered_set<const c10::TensorImpl*> new_parameters;
  for (const auto& parameter : group.params) {
    if (!parameter.defined()) {
      throw std::invalid_argument(
          "Parameter groups cannot contain undefined tensors");
    }
    const auto* identity = parameter.unsafeGetTensorImpl();
    if (known_parameters.count(identity) != 0 ||
        !new_parameters.insert(identity).second) {
      throw std::invalid_argument(
          "A parameter cannot appear in more than one parameter group");
    }
  }

  if (!param_groups_.empty()) {
    auto merged_options = param_groups_.front().options;
    for (const auto& [name, value] : group.options) {
      if (!std::isfinite(value)) {
        throw std::invalid_argument("Parameter-group options must be finite");
      }
      merged_options[name] = value;
    }
    group.options = std::move(merged_options);
  }
  if (const auto lr = group.options.find("lr");
      lr != group.options.end() && lr->second < 0.0) {
    throw std::invalid_argument("Learning rate must be non-negative");
  }

  param_groups_.push_back(std::move(group));
}

void Optimizer::zero_grad(bool set_to_none) {
  torch::NoGradGuard no_grad;
  for (const auto& group : param_groups_) {
    for (const auto& parameter : group.params) {
      if (!parameter.defined() || !parameter.requires_grad()) {
        continue;
      }

      auto& gradient = parameter.mutable_grad();
      if (!gradient.defined()) {
        continue;
      }

      if (set_to_none) {
        gradient = Tensor{};
      } else {
        gradient.zero_();
      }
    }
  }
}

std::vector<double> Optimizer::get_lrs() const {
  std::vector<double> lrs;
  lrs.reserve(param_groups_.size());
  for (const auto& group : param_groups_) {
    const auto it = group.options.find("lr");
    if (it == group.options.end()) {
      throw std::runtime_error(
          "Parameter group does not define a learning rate");
    }
    lrs.push_back(it->second);
  }
  return lrs;
}

void Optimizer::set_lr(double lr) {
  if (!std::isfinite(lr) || lr < 0.0) {
    throw std::invalid_argument("Learning rate must be non-negative");
  }
  for (auto& group : param_groups_) {
    group.options["lr"] = lr;
  }
}

void Optimizer::set_lrs(const std::vector<double>& lrs) {
  if (lrs.size() != param_groups_.size()) {
    throw std::invalid_argument(
        "Learning-rate count must match parameter-group count");
  }
  for (const double lr : lrs) {
    if (!std::isfinite(lr) || lr < 0.0) {
      throw std::invalid_argument("Learning rate must be non-negative");
    }
  }
  for (size_t i = 0; i < lrs.size(); ++i) {
    param_groups_[i].options["lr"] = lrs[i];
  }
}

std::map<std::string, torch::Tensor> Optimizer::state_dict() const {
  std::map<std::string, torch::Tensor> state;
  state[std::string(kFormatVersion)] = torch::tensor(1, torch::kInt64);
  state[std::string(kParamGroupCount)] =
      torch::tensor(static_cast<int64_t>(param_groups_.size()), torch::kInt64);
  int64_t parameter_count = 0;
  for (const auto& group : param_groups_) {
    parameter_count += static_cast<int64_t>(group.params.size());
  }
  state[std::string(kParameterCount)] =
      torch::tensor(parameter_count, torch::kInt64);
  state["step_count"] = torch::tensor(step_count_, torch::kInt64);
  size_t group_index = 0;
  for (const auto& group : param_groups_) {
    for (const auto& [name, value] : group.options) {
      state["param_group_" + std::to_string(group_index) + "_" + name] =
          // Options are double precision; keep them exact across the
          // round trip (torch::tensor(double) would default to float32).
          torch::tensor(value, torch::TensorOptions().dtype(torch::kFloat64));
    }
    ++group_index;
  }

  // TensorImpl pointers are process-local, so serialize state by the stable
  // parameter order exposed by param_groups_. This lets a newly constructed
  // optimizer restore momentum/Adam buffers without exposing implementation
  // pointers in the serialized keys.
  size_t parameter_index = 0;
  for (const auto& group : param_groups_) {
    for (const auto& parameter : group.params) {
      int64_t serialized_state_count = 0;
      if (parameter.defined()) {
        const auto state_it = state_.find(parameter.unsafeGetTensorImpl());
        if (state_it != state_.end()) {
          for (const auto& [name, value] : state_it->second) {
            if (value.defined()) {
              state["param_" + std::to_string(parameter_index) + "_state_" +
                    name] = value.detach().clone();
              ++serialized_state_count;
            }
          }
        }
      }
      state[parameter_state_count_key(parameter_index)] =
          torch::tensor(serialized_state_count, torch::kInt64);
      ++parameter_index;
    }
  }
  return state;
}

void Optimizer::load_state_dict(
    const std::map<std::string, torch::Tensor>& state) {
  int64_t parameter_count = 0;
  for (const auto& group : param_groups_) {
    parameter_count += static_cast<int64_t>(group.params.size());
  }
  const auto format = state.find(std::string(kFormatVersion));
  if (format == state.end()) {
    throw std::invalid_argument(
        "Optimizer checkpoint format version is missing");
  }
  if (scalar_int64(format->second, "format version") != 1) {
    throw std::invalid_argument("Unsupported optimizer state format version");
  }
  const auto group_count = state.find(std::string(kParamGroupCount));
  const auto saved_parameter_count = state.find(std::string(kParameterCount));
  if (group_count == state.end() || saved_parameter_count == state.end()) {
    throw std::invalid_argument(
        "Optimizer checkpoint topology metadata is incomplete");
  }
  if (scalar_int64(group_count->second, "parameter-group count") !=
          static_cast<int64_t>(param_groups_.size()) ||
      scalar_int64(saved_parameter_count->second, "parameter count") !=
          parameter_count) {
    throw std::invalid_argument(
        "Optimizer checkpoint parameter topology does not match the optimizer");
  }

  int64_t restored_step = step_count_;
  const auto step_it = state.find("step_count");
  if (step_it != state.end()) {
    restored_step = scalar_int64(step_it->second, "step_count");
  } else {
    throw std::invalid_argument("Optimizer checkpoint is missing step_count");
  }

  std::vector<std::map<std::string, double>> restored_options;
  restored_options.reserve(param_groups_.size());
  for (size_t group_index = 0; group_index < param_groups_.size();
       ++group_index) {
    restored_options.push_back(param_groups_[group_index].options);
    const std::string prefix =
        "param_group_" + std::to_string(group_index) + "_";
    for (auto& [name, value] : restored_options.back()) {
      const auto it = state.find(prefix + name);
      if (it == state.end()) {
        throw std::invalid_argument(
            "Optimizer checkpoint is missing parameter-group option " + prefix +
            name);
      }
      if (it != state.end()) {
        if (!it->second.defined() || it->second.numel() != 1) {
          throw std::invalid_argument(
              "Optimizer parameter-group option must be scalar");
        }
        value = it->second.item<double>();
        if (!std::isfinite(value)) {
          throw std::invalid_argument(
              "Optimizer parameter-group options must be finite");
        }
      }
    }
    const auto lr = restored_options.back().find("lr");
    if (lr != restored_options.back().end() && lr->second < 0.0) {
      throw std::invalid_argument(
          "Optimizer checkpoint learning rate must be non-negative");
    }
  }

  std::map<const c10::TensorImpl*, std::map<std::string, Tensor>>
      restored_state;
  size_t parameter_index = 0;
  for (const auto& group : param_groups_) {
    for (const auto& parameter : group.params) {
      const std::string prefix =
          "param_" + std::to_string(parameter_index) + "_state_";
      int64_t restored_state_count = 0;
      for (const auto& [name, value] : state) {
        if (name.rfind(prefix, 0) == 0 && value.defined()) {
          restored_state[parameter.unsafeGetTensorImpl()]
                        [name.substr(prefix.size())] = value.detach().clone();
          ++restored_state_count;
        }
      }
      const auto count = state.find(parameter_state_count_key(parameter_index));
      if (count == state.end() ||
          scalar_int64(count->second, "parameter state count") !=
              restored_state_count) {
        throw std::invalid_argument(
            "Optimizer checkpoint parameter state is incomplete");
      }
      ++parameter_index;
    }
  }

  for (const auto& [name, value] : state) {
    if (name.rfind("param_", 0) != 0) {
      continue;
    }
    const auto marker = name.find("_state_");
    if (marker == std::string::npos) {
      continue;
    }
    const auto index_text = name.substr(6, marker - 6);
    size_t parsed = 0;
    try {
      parsed = static_cast<size_t>(std::stoull(index_text));
    } catch (...) {
      throw std::invalid_argument(
          "Optimizer checkpoint contains an invalid parameter state key");
    }
    if (parsed >= static_cast<size_t>(parameter_count)) {
      throw std::invalid_argument(
          "Optimizer checkpoint contains out-of-range parameter state");
    }
    if (!value.defined()) {
      throw std::invalid_argument(
          "Optimizer parameter state tensors must be defined");
    }
  }

  step_count_ = restored_step;
  for (size_t i = 0; i < restored_options.size(); ++i) {
    param_groups_[i].options = std::move(restored_options[i]);
  }
  // State is replaced, rather than merged, so a resumed optimizer cannot
  // retain buffers belonging to a previous checkpoint or parameter layout.
  state_ = std::move(restored_state);
}

}  // namespace optim
}  // namespace jyppx
