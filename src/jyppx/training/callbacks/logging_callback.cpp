#include <jyppx/training/callbacks/logging_callback.hpp>

#include <jyppx/optim/optimizer.hpp>
#include <jyppx/training/trainer.hpp>
#include <jyppx/utils/logger.hpp>

#include <stdexcept>
#include <string>

namespace jyppx {
namespace training {
namespace callbacks {

namespace {

std::string join_lrs(const optim::Optimizer& optimizer) {
    std::string text = "[";
    const auto lrs = optimizer.get_lrs();
    for (size_t i = 0; i < lrs.size(); ++i) {
        if (i > 0) {
            text += ", ";
        }
        text += std::to_string(lrs[i]);
    }
    text += "]";
    return text;
}

} // namespace

LoggingCallback::LoggingCallback(int log_every_n_steps, std::string log_file)
    : log_every_n_steps_(log_every_n_steps), log_file_(std::move(log_file)) {
    if (log_every_n_steps_ <= 0) {
        throw std::invalid_argument("LoggingCallback log_every_n_steps must be positive");
    }
    if (!log_file_.empty()) {
        jyppx::utils::Logger::instance().set_file(log_file_);
    }
}

void LoggingCallback::on_fit_start(Trainer*) {
    global_step_ = 0;
    LOG_INFO("training started");
}

void LoggingCallback::on_train_batch_end(Trainer* trainer, int batch_idx, double loss) {
    ++global_step_;
    if (trainer && global_step_ % log_every_n_steps_ == 0) {
        LOG_INFO("epoch " + std::to_string(trainer->current_epoch()) + " step " +
                 std::to_string(trainer->global_step()) + " (batch " +
                 std::to_string(batch_idx) + ") loss=" + std::to_string(loss));
    }
}

void LoggingCallback::on_epoch_end(Trainer* trainer) {
    if (!trainer) {
        return;
    }
    std::string line = "epoch " + std::to_string(trainer->current_epoch()) +
                       " finished: train_loss=" +
                       std::to_string(trainer->last_train_loss());
    if (trainer->has_validation_loss()) {
        line += ", val_loss=" + std::to_string(trainer->last_val_loss());
    }
    LOG_INFO(line);
}

void LoggingCallback::on_fit_end(Trainer*) {
    LOG_INFO("training finished after " + std::to_string(global_step_) +
             " logged batches");
}

void LRMonitor::on_epoch_start(Trainer* trainer) {
    if (trainer && trainer->optimizer()) {
        LOG_INFO("epoch " + std::to_string(trainer->current_epoch()) +
                 " learning rate: " + join_lrs(*trainer->optimizer()));
    }
}

} // namespace callbacks
} // namespace training
} // namespace jyppx
