#ifndef JYPPX_TRAINING_AMP_AUTOCAST_HPP
#define JYPPX_TRAINING_AMP_AUTOCAST_HPP

#include <ATen/autocast_mode.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>

namespace jyppx {
namespace training {
namespace amp {

/**
 * @brief RAII guard enabling autocast for the current scope.
 *
 * Wraps the ATen autocast thread-local state: while the guard is alive, ops
 * dispatched on `device_type` run in `dtype` (e.g. float16 on CUDA). The
 * previous state is restored on destruction.
 *
 * @example
 * @code
 * {
 *     amp::AutocastGuard guard{c10::DeviceType::CUDA, at::kHalf};
 *     Tensor loss = loss_fn(model->forward(input), target);
 * }  // autocast disabled again here
 * loss.backward();
 * @endcode
 */
class AutocastGuard {
public:
    AutocastGuard(c10::DeviceType device_type, at::ScalarType dtype)
        : device_type_(device_type),
          was_enabled_(at::autocast::is_autocast_enabled(device_type)) {
        at::autocast::set_autocast_dtype(device_type, dtype);
        at::autocast::set_autocast_enabled(device_type, true);
        at::autocast::increment_nesting();
    }

    ~AutocastGuard() {
        at::autocast::decrement_nesting();
        at::autocast::set_autocast_enabled(device_type_, was_enabled_);
        if (!was_enabled_) {
            at::autocast::clear_cache();
        }
    }

    AutocastGuard(const AutocastGuard&) = delete;
    AutocastGuard& operator=(const AutocastGuard&) = delete;
    AutocastGuard(AutocastGuard&&) = delete;
    AutocastGuard& operator=(AutocastGuard&&) = delete;

private:
    c10::DeviceType device_type_;
    bool was_enabled_;
};

} // namespace amp
} // namespace training
} // namespace jyppx

#endif // JYPPX_TRAINING_AMP_AUTOCAST_HPP
