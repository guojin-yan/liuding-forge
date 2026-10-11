#include <jyppx/training/callbacks/early_stopping.hpp>
#include <jyppx/training/trainer.hpp>

#include <limits>
#include <stdexcept>

namespace jyppx {
namespace training {
namespace callbacks {

EarlyStopping::EarlyStopping(
    const std::string& monitor,
    int patience,
    double min_delta,
    const std::string& mode)
    : monitor_(monitor),
      patience_(patience),
      min_delta_(min_delta),
      mode_(mode),
      wait_count_(0),
      best_score_(mode == "max"
          ? -std::numeric_limits<double>::infinity()
          : std::numeric_limits<double>::infinity()),
      should_stop_(false) {
    if (monitor_.empty()) {
        throw std::invalid_argument("EarlyStopping monitor cannot be empty");
    }
    if (patience_ < 0) {
        throw std::invalid_argument("EarlyStopping patience must be non-negative");
    }
    if (min_delta_ < 0.0) {
        throw std::invalid_argument("EarlyStopping min_delta must be non-negative");
    }
    if (mode_ != "min" && mode_ != "max") {
        throw std::invalid_argument("EarlyStopping mode must be 'min' or 'max'");
    }
}

void EarlyStopping::on_epoch_end(Trainer* trainer) {
    if (!trainer) {
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

    const bool improved = mode_ == "min"
        ? current < best_score_ - min_delta_
        : current > best_score_ + min_delta_;
    if (improved) {
        best_score_ = current;
        wait_count_ = 0;
        return;
    }

    ++wait_count_;
    if (wait_count_ >= patience_) {
        should_stop_ = true;
        trainer->request_stop();
    }
}

void EarlyStopping::reset() {
    wait_count_ = 0;
    best_score_ = mode_ == "max"
        ? -std::numeric_limits<double>::infinity()
        : std::numeric_limits<double>::infinity();
    should_stop_ = false;
}

} // namespace callbacks
} // namespace training
} // namespace jyppx
