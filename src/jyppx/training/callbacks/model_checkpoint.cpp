#include <jyppx/training/callbacks/model_checkpoint.hpp>
#include <jyppx/training/trainer.hpp>

#include <filesystem>
#include <limits>
#include <stdexcept>

namespace jyppx {
namespace training {
namespace callbacks {

ModelCheckpoint::ModelCheckpoint(
    const std::string& filepath,
    const std::string& monitor,
    const std::string& mode,
    int save_top_k)
    : filepath_(filepath),
      monitor_(monitor),
      mode_(mode),
      save_top_k_(save_top_k),
      best_score_(mode == "max"
          ? -std::numeric_limits<double>::infinity()
          : std::numeric_limits<double>::infinity()) {
    if (filepath_.empty()) {
        throw std::invalid_argument("ModelCheckpoint filepath cannot be empty");
    }
    if (monitor_.empty()) {
        throw std::invalid_argument("ModelCheckpoint monitor cannot be empty");
    }
    if (mode_ != "min" && mode_ != "max") {
        throw std::invalid_argument("ModelCheckpoint mode must be 'min' or 'max'");
    }
}

void ModelCheckpoint::on_epoch_end(Trainer* trainer) {
    if (!trainer || save_top_k_ == 0) {
        return;
    }
    double current = 0.0;
    if (monitor_ == "val_loss") {
        if (!trainer->has_validation_loss()) {
            return;
        }
        current = trainer->last_val_loss();
    } else if (monitor_ == "train_loss") {
        current = trainer->last_train_loss();
    } else {
        return;
    }

    const bool improved = mode_ == "min" ? current < best_score_ : current > best_score_;
    if (!improved && save_top_k_ > 0) {
        return;
    }

    const std::filesystem::path path(filepath_);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    trainer->save_checkpoint(filepath_);
    best_score_ = current;
}

void ModelCheckpoint::reset() {
    best_score_ = mode_ == "max"
        ? -std::numeric_limits<double>::infinity()
        : std::numeric_limits<double>::infinity();
}

} // namespace callbacks
} // namespace training
} // namespace jyppx
