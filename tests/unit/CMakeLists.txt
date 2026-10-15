# Unit test executables, one per framework module.

jyppx_add_gtest(jyppx_test_foundation
    LINK_TARGET jyppx::foundation
    foundation/test_foundation.cpp
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
    data/test_tensor_dataset.cpp
    data/test_samplers.cpp
    data/test_dataloader.cpp
    data/test_thread_safe_queue.cpp
    data/test_dataset_utils.cpp
    data/test_transforms.cpp
)

jyppx_add_gtest(jyppx_test_optim
    optim/test_sgd.cpp
    optim/test_adam.cpp
    optim/test_adamw.cpp
    optim/test_rmsprop_adagrad.cpp
    optim/test_schedulers.cpp
    optim/test_new_schedulers.cpp
    optim/test_p1_schedulers.cpp
    optim/test_composite_schedulers.cpp
    optim/test_extreme_params.cpp
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

jyppx_add_gtest(jyppx_test_plan
    test_plan_completion.cpp
)

jyppx_add_gtest(jyppx_test_training
    training/test_accuracy.cpp
    training/test_amp.cpp
    training/test_callbacks.cpp
    training/test_metrics.cpp
    training/test_trainer.cpp
    training/test_nan_inf.cpp
)

jyppx_add_gtest(jyppx_test_data_exceptions
    LINK_TARGET jyppx::data
    data/test_dataloader_exceptions.cpp
)
