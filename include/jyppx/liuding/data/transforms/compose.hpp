// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
#ifndef JYPPX_LIUDING_DATA_TRANSFORMS_COMPOSE_HPP
#define JYPPX_LIUDING_DATA_TRANSFORMS_COMPOSE_HPP
/**
 * @file compose.hpp
 * @brief Defines sequential and randomly selected tensor transforms.
 */

#include <jyppx/liuding/data/detail/error.hpp>
#include <jyppx/liuding/data/transforms/transform.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace jyppx::liuding {
namespace data {
namespace transforms {

/**
 * @brief Applies an ordered list of tensor transforms.
 */
class Compose : public TensorTransform {
 public:
  /**
   * @brief Creates an ordered transform pipeline.
   * @param transforms Transform handles applied from first to last; an empty
   * list is the identity transform.
   * @throws jyppx::liuding::InvalidArgument Any transform handle is null.
   */
  explicit Compose(std::vector<std::shared_ptr<TensorTransform>> transforms)
      : transforms_(std::move(transforms)) {
    for (const auto& transform : transforms_) {
      if (!transform) {
        ::jyppx::liuding::data::detail::throw_invalid(
            "data.transforms.compose", "constructor.transforms",
            "non-null transform handles", "null transform",
            "Remove or replace the null transform handle.");
      }
    }
  }

  /**
   * @brief Applies every configured transform in order.
   * @param input Tensor supplied to the first transform.
   * @return Output from the final transform, or @p input for an empty list.
   */
  Tensor operator()(const Tensor& input) const override {
    Tensor result = input;
    for (const auto& transform : transforms_) {
      result = (*transform)(result);
    }
    return result;
  }

  /**
   * @brief Returns the ordered child names in `Compose(...)` form.
   * @return Human-readable pipeline description.
   */
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

  /**
   * @brief Returns the number of composed transforms.
   * @return Child transform count.
   */
  size_t size() const { return transforms_.size(); }

 private:
  std::vector<std::shared_ptr<TensorTransform>>
      transforms_;  ///< Non-null transforms in application order.
};

/**
 * @brief Picks one transform at random per call
 *
 * The draw uses the global torch RNG, so runs are reproducible through
 * torch::manual_seed.
 */
class RandomChoice : public TensorTransform {
 public:
  /**
   * @brief Creates a uniformly selected transform set.
   * @param transforms Non-empty collection of non-null transform handles.
   * @throws jyppx::liuding::InvalidArgument The collection is empty or contains
   * null.
   */
  explicit RandomChoice(
      std::vector<std::shared_ptr<TensorTransform>> transforms)
      : transforms_(std::move(transforms)) {
    if (transforms_.empty()) {
      ::jyppx::liuding::data::detail::throw_invalid(
          "data.transforms.random_choice", "constructor.transforms",
          "at least one transform", "empty collection",
          "Provide a non-empty transform collection.");
    }
    for (const auto& transform : transforms_) {
      if (!transform) {
        ::jyppx::liuding::data::detail::throw_invalid(
            "data.transforms.random_choice", "constructor.transforms",
            "non-null transform handles", "null transform",
            "Remove or replace the null transform handle.");
      }
    }
  }

  /**
   * @brief Applies one uniformly selected child transform.
   * @param input Tensor supplied to the selected transform.
   * @return Selected transform output.
   */
  Tensor operator()(const Tensor& input) const override {
    const auto index =
        torch::randint(0, static_cast<int64_t>(transforms_.size()), {1})
            .item<int64_t>();
    return (*transforms_[static_cast<size_t>(index)])(input);
  }

  /**
   * @brief Returns the stable transform name.
   * @return `RandomChoice`.
   */
  std::string name() const override { return "RandomChoice"; }

 private:
  std::vector<std::shared_ptr<TensorTransform>>
      transforms_;  ///< Non-empty set of selectable transforms.
};

}  // namespace transforms
}  // namespace data
}  // namespace jyppx::liuding

#endif  // JYPPX_LIUDING_DATA_TRANSFORMS_COMPOSE_HPP