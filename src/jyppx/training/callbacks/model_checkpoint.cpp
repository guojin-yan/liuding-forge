#include <jyppx/training/callbacks/model_checkpoint.hpp>
#include <jyppx/training/trainer.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace jyppx {
namespace training {
namespace callbacks {

ModelCheckpoint::ModelCheckpoint(const std::string& filepath,
                                 const std::string& monitor,
                                 const std::string& mode, int save_top_k)
    : filepath_(filepath),
      monitor_(monitor),
      mode_(mode),
      save_top_k_(save_top_k),
      best_score_(mode == "max" ? -std::numeric_limits<double>::infinity()
                                : std::numeric_limits<double>::infinity()) {
  if (filepath_.empty()) {
    throw std::invalid_argument("ModelCheckpoint filepath cannot be empty");
  }
  if (monitor_ != "train_loss" && monitor_ != "val_loss") {
    throw std::invalid_argument(
        "ModelCheckpoint monitor must be 'train_loss' or 'val_loss'");
  }
  if (mode_ != "min" && mode_ != "max") {
    throw std::invalid_argument("ModelCheckpoint mode must be 'min' or 'max'");
  }
  if (save_top_k_ < -1) {
    throw std::invalid_argument(
        "ModelCheckpoint save_top_k must be -1, 0, or positive");
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
  if (!std::isfinite(current)) {
    throw std::runtime_error("ModelCheckpoint monitored value must be finite");
  }

  const int64_t previous_epoch_index = epoch_index_;
  ++epoch_index_;
  const bool better_than_best =
      mode_ == "min" ? current < best_score_ : current > best_score_;
  if (save_top_k_ < 0) {
    // save every epoch when requested, using a deterministic suffix
    const std::filesystem::path base(filepath_);
    const auto candidate = base.parent_path() / (base.stem().string() + "-" +
                                                 std::to_string(epoch_index_) +
                                                 base.extension().string());
    const double previous_best = best_score_;
    if (better_than_best) {
      best_score_ = current;
    }
    trainer->defer_epoch_end_action(
        [this, trainer, candidate, previous_best, previous_epoch_index]() {
          try {
            if (candidate.has_parent_path()) {
              std::filesystem::create_directories(candidate.parent_path());
            }
            trainer->save_checkpoint(candidate.string());
          } catch (...) {
            best_score_ = previous_best;
            epoch_index_ = previous_epoch_index;
            throw;
          }
        });
    return;
  }
  if (save_top_k_ > 0 && static_cast<int>(saved_.size()) >= save_top_k_) {
    auto worse = std::min_element(
        saved_.begin(), saved_.end(), [&](const auto& a, const auto& b) {
          return mode_ == "min" ? a.first > b.first : a.first < b.first;
        });
    const bool qualifies =
        mode_ == "min" ? current < worse->first : current > worse->first;
    if (!qualifies) return;
  }
  if (save_top_k_ == 1) {
    if (!better_than_best) return;
  }
  if (save_top_k_ <= 0) return;

  std::filesystem::path candidate_path(filepath_);
  if (save_top_k_ > 1) {
    candidate_path =
        candidate_path.parent_path() /
        (candidate_path.stem().string() + "-" + std::to_string(epoch_index_) +
         candidate_path.extension().string());
  }
  if (candidate_path.empty()) {
    return;
  }

  const auto previous_saved = saved_;
  const double previous_best = best_score_;
  if (save_top_k_ == 1) {
    saved_.clear();
  }
  saved_.emplace_back(current, candidate_path.string());
  auto better = [&](const auto& a, const auto& b) {
    return mode_ == "min" ? a.first < b.first : a.first > b.first;
  };
  std::sort(saved_.begin(), saved_.end(), better);
  std::string removed;
  if (static_cast<int>(saved_.size()) > save_top_k_) {
    removed = saved_.back().second;
    saved_.pop_back();
  }
  best_score_ = saved_.front().first;
  trainer->defer_epoch_end_action([this, trainer, candidate_path,
                                   previous_saved, previous_best,
                                   previous_epoch_index, removed]() {
    try {
      if (candidate_path.has_parent_path()) {
        std::filesystem::create_directories(candidate_path.parent_path());
      }
      trainer->save_checkpoint(candidate_path.string());
    } catch (...) {
      saved_ = previous_saved;
      best_score_ = previous_best;
      epoch_index_ = previous_epoch_index;
      throw;
    }
    if (!removed.empty() && removed != candidate_path.string()) {
      std::error_code error;
      std::filesystem::remove(removed, error);
    }
  });
}

void ModelCheckpoint::reset() {
  best_score_ = mode_ == "max" ? -std::numeric_limits<double>::infinity()
                               : std::numeric_limits<double>::infinity();
  saved_.clear();
  epoch_index_ = 0;
}

CallbackState ModelCheckpoint::state_dict() const {
  CallbackState state;
  state.scalars.emplace("best_score", best_score_);
  state.scalars.emplace("epoch_index", static_cast<double>(epoch_index_));
  state.scalars.emplace("saved_count", static_cast<double>(saved_.size()));
  state.scalars.emplace("save_top_k", static_cast<double>(save_top_k_));
  state.strings.emplace("monitor", monitor_);
  state.strings.emplace("mode", mode_);
  for (size_t i = 0; i < saved_.size(); ++i) {
    state.scalars.emplace("saved_score/" + std::to_string(i), saved_[i].first);
    state.strings.emplace("saved_path/" + std::to_string(i), saved_[i].second);
  }
  return state;
}

void ModelCheckpoint::load_state_dict(const CallbackState& state) {
  const auto best = state.scalars.find("best_score");
  const auto epoch = state.scalars.find("epoch_index");
  const auto count = state.scalars.find("saved_count");
  const auto save_top_k = state.scalars.find("save_top_k");
  const auto monitor = state.strings.find("monitor");
  const auto mode = state.strings.find("mode");
  if (best == state.scalars.end() || epoch == state.scalars.end() ||
      count == state.scalars.end() || save_top_k == state.scalars.end() ||
      monitor == state.strings.end() || mode == state.strings.end() ||
      !std::isfinite(epoch->second) ||
      std::floor(epoch->second) != epoch->second || epoch->second < 0.0 ||
      epoch->second >=
          static_cast<double>(std::numeric_limits<int64_t>::max()) ||
      !std::isfinite(count->second) ||
      std::floor(count->second) != count->second || count->second < 0.0 ||
      count->second > static_cast<double>(state.scalars.size()) ||
      save_top_k->second != save_top_k_ || monitor->second != monitor_ ||
      mode->second != mode_) {
    throw std::invalid_argument("Invalid ModelCheckpoint checkpoint state");
  }
  const auto saved_count = static_cast<size_t>(count->second);
  if ((save_top_k_ >= 0 && saved_count > static_cast<size_t>(save_top_k_)) ||
      epoch->second < count->second) {
    throw std::invalid_argument("Invalid ModelCheckpoint retained-state count");
  }
  std::vector<std::pair<double, std::string>> restored;
  restored.reserve(saved_count);
  for (size_t i = 0; i < saved_count; ++i) {
    const auto score = state.scalars.find("saved_score/" + std::to_string(i));
    const auto path = state.strings.find("saved_path/" + std::to_string(i));
    if (score == state.scalars.end() || path == state.strings.end() ||
        !std::isfinite(score->second) || path->second.empty()) {
      throw std::invalid_argument(
          "Incomplete ModelCheckpoint checkpoint state");
    }
    restored.emplace_back(score->second, path->second);
  }
  best_score_ = best->second;
  epoch_index_ = static_cast<int64_t>(epoch->second);
  saved_ = std::move(restored);
}

}  // namespace callbacks
}  // namespace training
}  // namespace jyppx
