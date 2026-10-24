// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
#ifndef JYPPX_LIUDING_DATA_SAMPLERS_SAMPLER_HPP
#define JYPPX_LIUDING_DATA_SAMPLERS_SAMPLER_HPP
/**
 * @file sampler.hpp
 * @brief Defines the epoch index-order and checkpoint-state contract.
 */

#include <jyppx/liuding/foundation/error.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace jyppx::liuding {
namespace data {
namespace samplers {

/**
 * @brief Produces one epoch of dataset indices.
 *
 * Implementations that opt into checkpoint state must make save/load reproduce
 * all future index sequences. Samplers are mutable through const generation
 * methods and are not thread-safe unless an implementation states otherwise.
 */
class Sampler {
 public:
  /** @brief Destroys a sampler through the base interface. */
  virtual ~Sampler() = default;

  /**
   * @brief Get all indices
   * @return Vector of indices
   */
  virtual std::vector<size_t> indices() const = 0;

  /**
   * @brief Returns the number of indices produced for one epoch.
   * @return Number of indices returned by `indices()`.
   */
  virtual size_t size() const = 0;

  /**
   * @brief Selects the logical epoch used by deterministic samplers.
   * @param epoch Non-negative epoch number. The default sampler ignores it.
   *
   * DataPipeline calls this method exactly once before materializing an epoch.
   * Implementations that derive ordering from an epoch should override it;
   * stateful random samplers may still use their generator state as the source
   * of continuation entropy.
   */
  virtual void set_epoch(std::uint64_t epoch) { static_cast<void>(epoch); }

  /**
   * @brief Reports whether epoch selection is deterministic and supported.
   * @return True when the same configuration and epoch reproduce the order.
   */
  virtual bool supports_epoch_control() const { return false; }

  /**
   * @brief Returns a stable identity for immutable sampler configuration.
   *
   * The identity must change whenever sampler type or immutable configuration
   * can change future index sequences. An empty identity means a checkpoint
   * consumer cannot prove configuration compatibility and must downgrade the
   * restore guarantee.
   *
   * @return Stable, versioned configuration identity, or empty if unavailable.
   */
  virtual std::string configuration_identity() const { return {}; }

  /**
   * @brief Returns whether state APIs reproduce future index sequences.
   * @return True when `save_state()` and `load_state()` support exact restore.
   */
  virtual bool supports_state() const { return false; }

  /**
   * @brief Serialize mutable sampling state used to produce future epochs.
   *
   * Stateless but reproducible samplers use the empty string and override
   * supports_state(). Stateful implementations must reject malformed state
   * in load_state() rather than silently reseeding.
   *
   * @return Opaque, implementation-owned state representation.
   */
  virtual std::string save_state() const { return {}; }

  /**
   * @brief Restores state previously returned by save_state().
   * @param state Opaque state from the same sampler type and configuration.
   * @throws jyppx::liuding::SerializationError The state is malformed or
   * incompatible.
   */
  virtual void load_state(const std::string& state) { (void)state; }
};

}  // namespace samplers
}  // namespace data
}  // namespace jyppx::liuding

#endif  // JYPPX_LIUDING_DATA_SAMPLERS_SAMPLER_HPP