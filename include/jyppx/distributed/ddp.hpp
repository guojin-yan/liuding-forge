#ifndef JYPPX_DISTRIBUTED_DDP_HPP
#define JYPPX_DISTRIBUTED_DISTRIBUTED_HPP

#include <jyppx/distributed/distributed.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace jyppx {
namespace distributed {

/**
 * @brief Data-parallel wrapper for a torch::nn::Module subclass
 *
 * Wraps a typed module (like torch::nn::DistributedDataParallel in spirit):
 * `forward` forwards to the wrapped module, and every parameter gradient is
 * routed through `distributed::average_gradients` via autograd hooks. With
 * the single-process backend the hooks are no-ops, so code written against
 * DDP runs unchanged locally; a c10d backend build turns the hooks into real
 * all-reduces.
 *
 * @tparam ModuleT the concrete module type (must expose forward(...))
 */
template <typename ModuleT>
class DistributedDataParallel {
public:
    explicit DistributedDataParallel(std::shared_ptr<ModuleT> module)
        : module_(std::move(module)) {
        if (!module_) {
            throw std::invalid_argument("DistributedDataParallel module cannot be null");
        }
        install_hooks();
    }

    template <typename... Args>
    auto forward(Args&&... args) {
        return module_->forward(std::forward<Args>(args)...);
    }

    /** @brief The wrapped module (for parameters(), train()/eval(), ...). */
    std::shared_ptr<ModuleT> module() const { return module_; }

private:
    void install_hooks() {
        for (const auto& parameter : module_->parameters()) {
            if (!parameter.defined() || !parameter.requires_grad()) {
                continue;
            }
            hooks_.push_back(parameter.register_hook(
                [](Tensor gradient) { return average_gradients(std::move(gradient)); }));
        }
    }

    std::shared_ptr<ModuleT> module_;
    std::vector<unsigned> hooks_;  // autograd hook handles, kept alive
};

} // namespace distributed
} // namespace jyppx

#endif // JYPPX_DISTRIBUTED_DDP_HPP
