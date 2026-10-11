#include <jyppx/training/callbacks/tensorboard_callback.hpp>

#include <jyppx/training/trainer.hpp>

#include <stdexcept>

namespace jyppx {
namespace training {
namespace callbacks {

TensorBoardCallback::TensorBoardCallback(const std::string& log_dir,
                                         int log_every_n_steps)
    : writer_(std::make_unique<utils::TensorBoardWriter>(log_dir)),
      log_every_n_steps_(log_every_n_steps) {
    if (log_every_n_steps_ <= 0) {
        throw std::invalid_argument(
            "TensorBoardCallback log_every_n_steps must be positive");
    }
}

void TensorBoardCallback::on_fit_start(Trainer* trainer) {
    if (trainer && trainer->optimizer()) {
        const auto lrs = trainer->optimizer()->get_lrs();
        double lr = lrs.empty() ? 0.0 : lrs.front();
        writer_->add_scalar("train/learning_rate", lr, trainer->global_step());
    }
}

void TensorBoardCallback::on_train_batch_end(Trainer* trainer, int batch_idx,
                                             double loss) {
    if (trainer && trainer->global_step() % log_every_n_steps_ == 0) {
        writer_->add_scalar("train/loss", loss, trainer->global_step());
    }
}

void TensorBoardCallback::on_epoch_end(Trainer* trainer) {
    if (!trainer) {
        return;
    }
    const int64_t epoch = trainer->current_epoch();
    writer_->add_scalar("train/epoch_loss", trainer->last_train_loss(), epoch);
    if (trainer->has_validation_loss()) {
        writer_->add_scalar("validation/epoch_loss", trainer->last_val_loss(),
                            epoch);
    }
    if (trainer->optimizer()) {
        const auto lrs = trainer->optimizer()->get_lrs();
        if (!lrs.empty()) {
            writer_->add_scalar("train/learning_rate", lrs.front(), epoch);
        }
    }
    writer_->flush();
}

void TensorBoardCallback::on_fit_end(Trainer*) {
    writer_->flush();
}

} // namespace callbacks
} // namespace training
} // namespace jyppx
