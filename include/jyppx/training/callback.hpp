#ifndef JYPPX_TRAINING_CALLBACK_HPP
#define JYPPX_TRAINING_CALLBACK_HPP

#include <jyppx/core/types.hpp>
#include <map>
#include <string>

namespace jyppx {
namespace training {

// Forward declaration
class Trainer;

/**
 * @brief Base class for all callbacks
 */
class Callback {
public:
    virtual ~Callback() = default;

    // Training lifecycle hooks
    virtual void on_fit_start(Trainer* trainer) {}
    virtual void on_fit_end(Trainer* trainer) {}

    virtual void on_epoch_start(Trainer* trainer) {}
    virtual void on_epoch_end(Trainer* trainer) {}

    virtual void on_train_epoch_start(Trainer* trainer) {}
    virtual void on_train_epoch_end(Trainer* trainer) {}

    virtual void on_validation_epoch_start(Trainer* trainer) {}
    virtual void on_validation_epoch_end(Trainer* trainer) {}

    virtual void on_train_batch_start(Trainer* trainer, int batch_idx) {}
    virtual void on_train_batch_end(Trainer* trainer, int batch_idx) {}

    virtual void on_validation_batch_start(Trainer* trainer, int batch_idx) {}
    virtual void on_validation_batch_end(Trainer* trainer, int batch_idx) {}
};

} // namespace training
} // namespace jyppx

#endif // JYPPX_TRAINING_CALLBACK_HPP
