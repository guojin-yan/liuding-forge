#ifndef JYPPX_DATA_CHAIN_DATASET_HPP
#define JYPPX_DATA_CHAIN_DATASET_HPP

/**
 * @file chain_dataset.hpp
 * @brief Defines the map-style chain dataset convenience type.
 */

#include <jyppx/data/concat_dataset.hpp>

namespace jyppx {
namespace data {

/**
 * @brief Chains several same-typed datasets into one
 *
 * For map-style datasets this is functionally identical to ConcatDataset; the
 * distinct name mirrors torch.utils.data.ChainDataset, which in PyTorch
 * chains iterable datasets. Provided so plan-listed API names resolve.
 *
 * @tparam T Sample type
 */
template <typename T = Tensor>
class ChainDataset : public ConcatDataset<T> {
 public:
  /**
   * @brief Creates a chain in caller-provided dataset order.
   * @param datasets Non-empty collection of non-null datasets.
   * @throws jyppx::InvalidArgument The collection is empty or contains null.
   */
  explicit ChainDataset(std::vector<std::shared_ptr<Dataset<T>>> datasets)
      : ConcatDataset<T>(std::move(datasets)) {}
};

}  // namespace data
}  // namespace jyppx

#endif  // JYPPX_DATA_CHAIN_DATASET_HPP
