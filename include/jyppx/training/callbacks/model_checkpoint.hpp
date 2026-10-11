#ifndef JYPPX_TRAINING_CALLBACKS_MODEL_CHECKPOINT_HPP
#define JYPPX_TRAINING_CALLBACKS_MODEL_CHECKPOINT_HPP

#include <jyppx/training/callback.hpp>
#include <string>

namespace jyppx {
namespace training {
namespace callbacks {

/**
 * @brief Model checkpoint callback
 */
class ModelCheckpoint : public Callback {
public:
    ModelCheckpoint(
        const std::string& filepath,
        const std::string& monitor = "val_loss",
        const std::string& mode = "min",
        int save_top_k = 1
    );

    void on_epoch_end(Trainer* trainer) override;
    void reset();

private:
    std::string filepath_;
    std::string monitor_;
    std::string mode_;
    int save_top_k_;
    double best_score_;
};

} // namespace callbacks
} // namespace training
} // namespace jyppx

#endif // JYPPX_TRAINING_CALLBACKS_MODEL_CHECKPOINT_HPP
