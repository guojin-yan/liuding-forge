#ifndef JYPPX_JYPPX_HPP
#define JYPPX_JYPPX_HPP

/**
 * @file jyppx.hpp
 * @brief Main header file for JYPPX Training Framework
 *
 * Include this file to use all JYPPX functionality.
 */

// Foundation
#include <jyppx/foundation/foundation.hpp>

// Core
#include <jyppx/core/device.hpp>
#include <jyppx/core/tensor_utils.hpp>
#include <jyppx/core/types.hpp>
#include <jyppx/core/value_schema.hpp>

// Optimizers
#include <jyppx/optim/adadelta_adapter.hpp>
#include <jyppx/optim/adagrad_adapter.hpp>
#include <jyppx/optim/adam_adapter.hpp>
#include <jyppx/optim/adamw_adapter.hpp>
#include <jyppx/optim/composite_lr_scheduler.hpp>
#include <jyppx/optim/epoch_lr_scheduler.hpp>
#include <jyppx/optim/event_lr_scheduler.hpp>
#include <jyppx/optim/lamb_adapter.hpp>
#include <jyppx/optim/linear_lr_scheduler.hpp>
#include <jyppx/optim/lr_scheduler.hpp>
#include <jyppx/optim/nadam_adapter.hpp>
#include <jyppx/optim/optimizer_rate_control.hpp>
#include <jyppx/optim/rmsprop_adapter.hpp>
#include <jyppx/optim/sgd_adapter.hpp>
#include <jyppx/optim/step_lr_scheduler.hpp>

// Data
#include <jyppx/data/batch_memory.hpp>
#include <jyppx/data/batch_schema.hpp>
#include <jyppx/data/chain_dataset.hpp>
#include <jyppx/data/concat_dataset.hpp>
#include <jyppx/data/data_pipeline.hpp>
#include <jyppx/data/dataloader.hpp>
#include <jyppx/data/dataset.hpp>
#include <jyppx/data/iterable_data_pipeline.hpp>
#include <jyppx/data/iterable_source.hpp>
#include <jyppx/data/stack_dataset.hpp>
#include <jyppx/data/subset.hpp>
#include <jyppx/data/tensor_dataset.hpp>
#include <jyppx/data/thread_safe_queue.hpp>

// Transforms
#include <jyppx/data/transforms/compose.hpp>
#include <jyppx/data/transforms/transform.hpp>
#include <jyppx/data/transforms/vision.hpp>

// Samplers
#include <jyppx/data/samplers/batch_sampler.hpp>
#include <jyppx/data/samplers/random_sampler.hpp>
#include <jyppx/data/samplers/sampler.hpp>
#include <jyppx/data/samplers/sequential_sampler.hpp>
#include <jyppx/data/samplers/weighted_random_sampler.hpp>

// Training
#include <jyppx/training/callback_dispatcher.hpp>
#include <jyppx/training/callbacks/early_stopping_callback.hpp>
#include <jyppx/training/checkpoint_coordinator.hpp>
#include <jyppx/training/event_sink.hpp>
#include <jyppx/training/scheduler_adapter.hpp>
#include <jyppx/training/single_device_strategy.hpp>
#include <jyppx/training/strategy.hpp>
#include <jyppx/training/trainer.hpp>

// Mixed precision
#include <jyppx/training/amp/autocast.hpp>
#include <jyppx/training/amp/grad_scaler.hpp>
#include <jyppx/training/amp/grad_scaler_checkpoint_participant.hpp>

// Metrics
#include <jyppx/training/metric.hpp>
#include <jyppx/training/metrics/accuracy.hpp>
#include <jyppx/training/metrics/average_meter.hpp>
#include <jyppx/training/metrics/classification.hpp>
#include <jyppx/training/metrics/metric_collection.hpp>
#include <jyppx/training/metrics/regression.hpp>
#include <jyppx/training/metrics/topk_accuracy.hpp>

// Utils
#include <jyppx/utils/logger.hpp>
#include <jyppx/utils/model_summary.hpp>
#include <jyppx/utils/profiler.hpp>
#include <jyppx/utils/progress_bar.hpp>
#include <jyppx/utils/tensorboard.hpp>

// Distributed (Phase 5)
#include <jyppx/distributed/ddp.hpp>
#include <jyppx/distributed/distributed.hpp>

// NN
#include <jyppx/nn/invocation.hpp>
#include <jyppx/nn/loss/contrastive_loss.hpp>
#include <jyppx/nn/loss/dice_loss.hpp>
#include <jyppx/nn/loss/focal_loss.hpp>
#include <jyppx/nn/loss/label_smoothing_cross_entropy.hpp>
#include <jyppx/nn/model_adapter.hpp>
#include <jyppx/nn/model_checkpoint_participant.hpp>

#endif  // JYPPX_JYPPX_HPP
