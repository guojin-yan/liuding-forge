// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_LIUDING_HPP
#define JYPPX_LIUDING_LIUDING_HPP

/**
 * @file liuding.hpp
 * @brief Main header for 六丁炼算 (Liuding Forge), project alias JYPPX Training
 * Framework
 *
 * Include this file to use all JYPPX functionality.
 */

// Foundation
#include <jyppx/liuding/foundation/foundation.hpp>

// Core
#include <jyppx/liuding/core/device.hpp>
#include <jyppx/liuding/core/tensor_utils.hpp>
#include <jyppx/liuding/core/types.hpp>
#include <jyppx/liuding/core/value_schema.hpp>

// Interoperability
#include <jyppx/liuding/interop/checkpoint_manifest.hpp>
#include <jyppx/liuding/interop/file_state_archive.hpp>

// Local experiments
#include <jyppx/liuding/experiment/artifact_store.hpp>
#include <jyppx/liuding/experiment/config.hpp>
#include <jyppx/liuding/experiment/run.hpp>
#include <jyppx/liuding/experiment/run_store.hpp>

// Optimizers
#include <jyppx/liuding/optim/adadelta_adapter.hpp>
#include <jyppx/liuding/optim/adagrad_adapter.hpp>
#include <jyppx/liuding/optim/adam_adapter.hpp>
#include <jyppx/liuding/optim/adamw_adapter.hpp>
#include <jyppx/liuding/optim/composite_lr_scheduler.hpp>
#include <jyppx/liuding/optim/epoch_lr_scheduler.hpp>
#include <jyppx/liuding/optim/event_lr_scheduler.hpp>
#include <jyppx/liuding/optim/lamb_adapter.hpp>
#include <jyppx/liuding/optim/linear_lr_scheduler.hpp>
#include <jyppx/liuding/optim/lr_scheduler.hpp>
#include <jyppx/liuding/optim/nadam_adapter.hpp>
#include <jyppx/liuding/optim/optimizer_rate_control.hpp>
#include <jyppx/liuding/optim/rmsprop_adapter.hpp>
#include <jyppx/liuding/optim/sgd_adapter.hpp>
#include <jyppx/liuding/optim/step_lr_scheduler.hpp>

// Data
#include <jyppx/liuding/data/batch_memory.hpp>
#include <jyppx/liuding/data/batch_schema.hpp>
#include <jyppx/liuding/data/batch_transform.hpp>
#include <jyppx/liuding/data/chain_dataset.hpp>
#include <jyppx/liuding/data/concat_dataset.hpp>
#include <jyppx/liuding/data/data_pipeline.hpp>
#include <jyppx/liuding/data/dataset.hpp>
#include <jyppx/liuding/data/iterable_data_pipeline.hpp>
#include <jyppx/liuding/data/iterable_source.hpp>
#include <jyppx/liuding/data/stack_dataset.hpp>
#include <jyppx/liuding/data/subset.hpp>
#include <jyppx/liuding/data/tensor_dataset.hpp>

// Transforms
#include <jyppx/liuding/data/transforms/compose.hpp>
#include <jyppx/liuding/data/transforms/transform.hpp>
#include <jyppx/liuding/data/transforms/vision.hpp>

// Samplers
#include <jyppx/liuding/data/samplers/batch_sampler.hpp>
#include <jyppx/liuding/data/samplers/random_sampler.hpp>
#include <jyppx/liuding/data/samplers/sampler.hpp>
#include <jyppx/liuding/data/samplers/sequential_sampler.hpp>
#include <jyppx/liuding/data/samplers/weighted_random_sampler.hpp>

// Training
#include <jyppx/liuding/training/callback_dispatcher.hpp>
#include <jyppx/liuding/training/callbacks/early_stopping_callback.hpp>
#include <jyppx/liuding/training/checkpoint_coordinator.hpp>
#include <jyppx/liuding/training/classification_evaluation.hpp>
#include <jyppx/liuding/training/event_sink.hpp>
#include <jyppx/liuding/training/file_checkpoint_store.hpp>
#include <jyppx/liuding/training/scheduler_adapter.hpp>
#include <jyppx/liuding/training/segmentation_evaluation.hpp>
#include <jyppx/liuding/training/single_device_strategy.hpp>
#include <jyppx/liuding/training/strategy.hpp>
#include <jyppx/liuding/training/trainer.hpp>

// Mixed precision
#include <jyppx/liuding/training/amp/autocast.hpp>
#include <jyppx/liuding/training/amp/grad_scaler.hpp>
#include <jyppx/liuding/training/amp/grad_scaler_checkpoint_participant.hpp>

// Metrics
#include <jyppx/liuding/training/metric.hpp>
#include <jyppx/liuding/training/metrics/accuracy.hpp>
#include <jyppx/liuding/training/metrics/average_meter.hpp>
#include <jyppx/liuding/training/metrics/classification.hpp>
#include <jyppx/liuding/training/metrics/metric_collection.hpp>
#include <jyppx/liuding/training/metrics/regression.hpp>
#include <jyppx/liuding/training/metrics/segmentation.hpp>
#include <jyppx/liuding/training/metrics/topk_accuracy.hpp>

// Utils
#include <jyppx/liuding/utils/model_summary.hpp>
#include <jyppx/liuding/utils/profiler.hpp>
#include <jyppx/liuding/utils/progress_bar.hpp>

// NN
#include <jyppx/liuding/nn/invocation.hpp>
#include <jyppx/liuding/nn/loss/contrastive_loss.hpp>
#include <jyppx/liuding/nn/loss/dice_loss.hpp>
#include <jyppx/liuding/nn/loss/focal_loss.hpp>
#include <jyppx/liuding/nn/loss/label_smoothing_cross_entropy.hpp>
#include <jyppx/liuding/nn/model_adapter.hpp>
#include <jyppx/liuding/nn/model_checkpoint_participant.hpp>

#endif  // JYPPX_LIUDING_LIUDING_HPP
