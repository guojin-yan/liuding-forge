# Unit test executables, one per framework module.

jyppx_add_gtest(jyppx_test_foundation
    LINK_TARGET jyppx::foundation
    foundation/test_checkpoint.cpp
    foundation/test_foundation.cpp
    foundation/test_rng_checkpoint_participant.cpp
)

jyppx_add_gtest(jyppx_test_core
    LINK_TARGET jyppx::core
    core/test_core.cpp
)

jyppx_add_gtest(jyppx_test_utils
    utils/test_logger.cpp
)

jyppx_add_gtest(jyppx_test_data
    LINK_TARGET jyppx::data
    data/test_data_pipeline.cpp
    data/test_iterable_data_pipeline.cpp
    data/test_tensor_dataset.cpp
    data/test_samplers.cpp
    data/test_dataloader.cpp
    data/test_thread_safe_queue.cpp
    data/test_dataset_utils.cpp
    data/test_transforms.cpp
)

jyppx_add_gtest(jyppx_test_optim
    optim/test_extreme_params.cpp
)

jyppx_add_gtest(jyppx_test_optim_adagrad_adapter
    LINK_TARGET jyppx::optim
    optim/test_adagrad_adapter.cpp
)

jyppx_add_gtest(jyppx_test_optim_sgd_adapter
    LINK_TARGET jyppx::optim
    optim/test_sgd_adapter.cpp
)

jyppx_add_gtest(jyppx_test_optim_adam_adapter
    LINK_TARGET jyppx::optim
    optim/test_adam_adapter.cpp
)

jyppx_add_gtest(jyppx_test_optim_adamw_adapter
    LINK_TARGET jyppx::optim
    optim/test_adamw_adapter.cpp
)

jyppx_add_gtest(jyppx_test_optim_nadam_adapter
    LINK_TARGET jyppx::optim
    optim/test_nadam_adapter.cpp
)

jyppx_add_gtest(jyppx_test_optim_rmsprop_adapter
    LINK_TARGET jyppx::optim
    optim/test_rmsprop_adapter.cpp
)

jyppx_add_gtest(jyppx_test_optim_step_lr_scheduler
    LINK_TARGET jyppx::optim
    optim/test_step_lr_scheduler.cpp
)

jyppx_add_gtest(jyppx_test_optim_composite_lr_scheduler
    LINK_TARGET jyppx::optim
    optim/test_composite_lr_scheduler.cpp
)

jyppx_add_gtest(jyppx_test_optim_epoch_lr_scheduler
    LINK_TARGET jyppx::optim
    optim/test_epoch_lr_scheduler.cpp
)

jyppx_add_gtest(jyppx_test_optim_event_lr_scheduler
    LINK_TARGET jyppx::optim
    optim/test_event_lr_scheduler.cpp
)

jyppx_add_gtest(jyppx_test_optim_linear_lr_scheduler
    LINK_TARGET jyppx::optim
    optim/test_linear_lr_scheduler.cpp
)

jyppx_add_gtest(jyppx_test_nn_label_smoothing
    LINK_TARGET jyppx::nn
    nn/loss/test_label_smoothing_cross_entropy.cpp
)

jyppx_add_gtest(jyppx_test_nn_contrastive
    LINK_TARGET jyppx::nn
    nn/loss/test_contrastive_loss.cpp
)

jyppx_add_gtest(jyppx_test_nn_dice
    LINK_TARGET jyppx::nn
    nn/loss/test_dice_loss.cpp
)

jyppx_add_gtest(jyppx_test_nn_focal
    LINK_TARGET jyppx::nn
    nn/loss/test_focal_loss.cpp
)

jyppx_add_gtest(jyppx_test_nn_model_adapter
    LINK_TARGET jyppx::nn
    nn/test_model_adapter.cpp
)

jyppx_add_gtest(jyppx_test_nn_model_checkpoint
    LINK_TARGET jyppx::nn
    nn/test_model_checkpoint_participant.cpp
)

jyppx_add_gtest(jyppx_test_training_checkpoint
    LINK_TARGET jyppx::training
    training/test_checkpoint_coordinator.cpp
)

jyppx_add_gtest(jyppx_test_training_scheduler_adapter
    LINK_TARGET jyppx::training
    training/test_scheduler_adapter.cpp
)

jyppx_add_gtest(jyppx_test_training_classification_metric
    LINK_TARGET jyppx::training
    training/test_classification_metric.cpp
)

jyppx_add_gtest(jyppx_test_training_ranking_metric
    LINK_TARGET jyppx::training
    training/test_ranking_metric.cpp
)

jyppx_add_gtest(jyppx_test_training_callback_dispatcher
    LINK_TARGET jyppx::training
    training/test_callback_dispatcher.cpp
)

jyppx_add_gtest(jyppx_test_training_event_sink
    LINK_TARGET jyppx::training
    training/test_event_sink.cpp
)

jyppx_add_gtest(jyppx_test_training_single_device_strategy
    LINK_TARGET jyppx::training
    training/test_single_device_strategy.cpp
)

jyppx_add_gtest(jyppx_test_training_trainer
    LINK_TARGET jyppx::training
    training/test_trainer.cpp
)

jyppx_add_gtest(jyppx_test_training_amp
    LINK_TARGET jyppx::training
    training/test_amp.cpp
    training/test_grad_scaler_checkpoint_participant.cpp
    training/test_nan_inf.cpp
)

jyppx_add_gtest(jyppx_test_plan
    test_plan_completion.cpp
)

jyppx_add_gtest(jyppx_test_training
    training/test_accuracy.cpp
    training/test_metrics.cpp
)

jyppx_add_gtest(jyppx_test_data_exceptions
    LINK_TARGET jyppx::data
    data/test_dataloader_exceptions.cpp
)
