#include <jyppx/training/callbacks/early_stopping.hpp>
#include <jyppx/training/trainer.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace jyppx {
namespace training {
namespace callbacks {

EarlyStopping::EarlyStopping(const std::string& monitor, int patience,
                             double min_delta, const std::string& mode)
    : monitor_(monitor),
      patience_(patience),
      min_delta_(min_delta),
      mode_(mode),
      wait_count_(0),
      best_score_(mode == "max" ? -std::numeric_limits<double>::infinity()
                                : std::numeric_limits<double>::infinity()),
      should_stop_(false) {
  if (monitor_.empty()) {
    throw std::invalid_argument("EarlyStopping monitor cannot be empty");
  }
  if (patience_ < 0) {
    throw std::invalid_argument("EarlyStopping patience must be non-negative");
  }
  if (!std::isfinite(min_delta_) || min_delta_ < 0.0) {
    throw std::invalid_argument(
        "EarlyStopping min_delta must be finite and non-negative");
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
  if (!std::isfinite(current)) {
    throw std::runtime_error("EarlyStopping monitored value must be finite");
  }

  const bool improved = mode_ == "min" ? current < best_score_ - min_delta_
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
  best_score_ = mode_ == "max" ? -std::numeric_limits<double>::infinity()
                               : std::numeric_limits<double>::infinity();
  should_stop_ = false;
}

CallbackState EarlyStopping::state_dict() const {
  return CallbackState{{
                           {"wait_count", static_cast<double>(wait_count_)},
                           {"best_score", best_score_},
                           {"should_stop", should_stop_ ? 1.0 : 0.0},
                           {"patience", static_cast<double>(patience_)},
                           {"min_delta", min_delta_},
                       },
                       {
                           {"monitor", monitor_},
                           {"mode", mode_},
                       }};
}

void EarlyStopping::load_state_dict(const CallbackState& state) {
  const auto wait = state.scalars.find("wait_count");
  const auto best = state.scalars.find("best_score");
  const auto stopped = state.scalars.find("should_stop");
  const auto patience = state.scalars.find("patience");
  const auto min_delta = state.scalars.find("min_delta");
  const auto monitor = state.strings.find("monitor");
  const auto mode = state.strings.find("mode");
  if (wait == state.scalars.end() || best == state.scalars.end() ||
      stopped == state.scalars.end() || patience == state.scalars.end() ||
      min_delta == state.scalars.end() || monitor == state.strings.end() ||
      mode == state.strings.end() || !std::isfinite(wait->second) ||
      std::floor(wait->second) != wait->second || wait->second < 0.0 ||
      wait->second > static_cast<double>(std::numeric_limits<int>::max()) ||
      std::isnan(best->second) ||
      (stopped->second != 0.0 && stopped->second != 1.0) ||
      patience->second != patience_ || min_delta->second != min_delta_ ||
      monitor->second != monitor_ || mode->second != mode_) {
    throw std::invalid_argument("Invalid EarlyStopping checkpoint state");
  }
  wait_count_ = static_cast<int>(wait->second);
  best_score_ = best->second;
  should_stop_ = stopped->second != 0.0;
}

}  // namespace callbacks
}  // namespace training
}  // namespace jyppx
