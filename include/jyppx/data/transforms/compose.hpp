#ifndef JYPPX_DATA_TRANSFORMS_COMPOSE_HPP
#define JYPPX_DATA_TRANSFORMS_COMPOSE_HPP

#include <jyppx/data/transforms/transform.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace jyppx {
namespace data {
namespace transforms {

/**
 * @brief Applies a list of transforms in order
 */
class Compose : public TensorTransform {
public:
    explicit Compose(std::vector<std::shared_ptr<TensorTransform>> transforms)
        : transforms_(std::move(transforms)) {
        for (const auto& transform : transforms_) {
            if (!transform) {
                throw std::invalid_argument("Compose transforms cannot be null");
            }
        }
    }

    Tensor operator()(const Tensor& input) const override {
        Tensor result = input;
        for (const auto& transform : transforms_) {
            result = (*transform)(result);
        }
        return result;
    }

    std::string name() const override {
        std::string names = "Compose(";
        for (size_t i = 0; i < transforms_.size(); ++i) {
            if (i > 0) {
                names += ", ";
            }
            names += transforms_[i]->name();
        }
        names += ")";
        return names;
    }

    /** @brief Number of composed transforms. */
    size_t size() const { return transforms_.size(); }

private:
    std::vector<std::shared_ptr<TensorTransform>> transforms_;
};

/**
 * @brief Picks one transform at random per call
 *
 * The draw uses the global torch RNG, so runs are reproducible through
 * torch::manual_seed.
 */
class RandomChoice : public TensorTransform {
public:
    explicit RandomChoice(std::vector<std::shared_ptr<TensorTransform>> transforms)
        : transforms_(std::move(transforms)) {
        if (transforms_.empty()) {
            throw std::invalid_argument("RandomChoice requires at least one transform");
        }
        for (const auto& transform : transforms_) {
            if (!transform) {
                throw std::invalid_argument("RandomChoice transforms cannot be null");
            }
        }
    }

    Tensor operator()(const Tensor& input) const override {
        const auto index = torch::randint(0, static_cast<int64_t>(transforms_.size()), {1})
                               .item<int64_t>();
        return (*transforms_[static_cast<size_t>(index)])(input);
    }

    std::string name() const override { return "RandomChoice"; }

private:
    std::vector<std::shared_ptr<TensorTransform>> transforms_;
};

} // namespace transforms
} // namespace data
} // namespace jyppx

#endif // JYPPX_DATA_TRANSFORMS_COMPOSE_HPP
