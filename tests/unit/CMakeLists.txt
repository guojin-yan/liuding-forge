# Unit test executables, one per framework module.

jyppx_add_gtest(jyppx_test_core
    core/test_core.cpp
)

jyppx_add_gtest(jyppx_test_utils
    utils/test_logger.cpp
)

jyppx_add_gtest(jyppx_test_data
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
    optim/test_extreme_params.cpp
)

jyppx_add_gtest(jyppx_test_nn
    nn/test_init.cpp
    nn/test_losses.cpp
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
    data/test_dataloader_exceptions.cpp
)
