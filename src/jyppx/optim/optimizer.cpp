#include <jyppx/optim/optimizer.hpp>

#include <stdexcept>

namespace jyppx {
namespace optim {

Optimizer::Optimizer(const TensorList& params) {
    if (params.empty()) {
        throw std::invalid_argument("Optimizer requires at least one parameter");
    }

    param_groups_.emplace_back(params);
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
            throw std::runtime_error("Parameter group does not define a learning rate");
        }
        lrs.push_back(it->second);
    }
    return lrs;
}

void Optimizer::set_lr(double lr) {
    if (lr < 0.0) {
        throw std::invalid_argument("Learning rate must be non-negative");
    }
    for (auto& group : param_groups_) {
        group.options["lr"] = lr;
    }
}

void Optimizer::set_lrs(const std::vector<double>& lrs) {
    if (lrs.size() != param_groups_.size()) {
        throw std::invalid_argument("Learning-rate count must match parameter-group count");
    }
    for (size_t i = 0; i < lrs.size(); ++i) {
        if (lrs[i] < 0.0) {
            throw std::invalid_argument("Learning rate must be non-negative");
        }
        param_groups_[i].options["lr"] = lrs[i];
    }
}

std::map<std::string, torch::Tensor> Optimizer::state_dict() const {
    std::map<std::string, torch::Tensor> state;
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
            if (parameter.defined()) {
                const auto state_it = state_.find(parameter.unsafeGetTensorImpl());
                if (state_it != state_.end()) {
                    for (const auto& [name, value] : state_it->second) {
                        if (value.defined()) {
                            state["param_" + std::to_string(parameter_index) + "_state_" + name] =
                                value.detach().clone();
                        }
                    }
                }
            }
            ++parameter_index;
        }
    }
    return state;
}

void Optimizer::load_state_dict(const std::map<std::string, torch::Tensor>& state) {
    const auto step_it = state.find("step_count");
    if (step_it != state.end() && step_it->second.defined() && step_it->second.numel() == 1) {
        step_count_ = step_it->second.item<int64_t>();
    }
    for (size_t group_index = 0; group_index < param_groups_.size(); ++group_index) {
        const std::string prefix = "param_group_" + std::to_string(group_index) + "_";
        for (auto& [name, value] : param_groups_[group_index].options) {
            const auto it = state.find(prefix + name);
            if (it != state.end() && it->second.defined() && it->second.numel() == 1) {
                value = it->second.item<double>();
            }
        }
    }

    size_t parameter_index = 0;
    for (auto& group : param_groups_) {
        for (const auto& parameter : group.params) {
            if (!parameter.defined()) {
                ++parameter_index;
                continue;
            }
            auto& parameter_state = state_[parameter.unsafeGetTensorImpl()];
            const std::string prefix = "param_" + std::to_string(parameter_index) + "_state_";
            for (const auto& [name, value] : state) {
                if (name.rfind(prefix, 0) == 0 && value.defined()) {
                    parameter_state[name.substr(prefix.size())] = value.detach().clone();
                }
            }
            ++parameter_index;
        }
    }
}

} // namespace optim
} // namespace jyppx
