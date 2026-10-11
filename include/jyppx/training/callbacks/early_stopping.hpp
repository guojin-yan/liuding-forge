#ifndef JYPPX_TRAINING_CALLBACKS_EARLY_STOPPING_HPP
#define JYPPX_TRAINING_CALLBACKS_EARLY_STOPPING_HPP

#include <jyppx/training/callback.hpp>
#include <string>

namespace jyppx {
namespace training {
namespace callbacks {

/**
 * @brief Early stopping callback
 */
class EarlyStopping : public Callback {
public:
    EarlyStopping(
        const std::string& monitor = "val_loss",
        int patience = 10,
        double min_delta = 0.0,
        const std::string& mode = "min"
    );

    void on_epoch_end(Trainer* trainer) override;

    bool should_stop() const { return should_stop_; }
    void reset();

private:
    std::string monitor_;
    int patience_;
    double min_delta_;
    std::string mode_;
    int wait_count_;
    double best_score_;
    bool should_stop_;
};

} // namespace callbacks
} // namespace training
} // namespace jyppx

#endif // JYPPX_TRAINING_CALLBACKS_EARLY_STOPPING_HPP
