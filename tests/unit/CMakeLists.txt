# Unit test executables, one per framework module.

if(TARGET jyppx::foundation)
jyppx_add_gtest(jyppx_test_foundation
    LINK_TARGET jyppx::foundation
    foundation/test_checkpoint.cpp
    foundation/test_foundation.cpp
    foundation/test_rng_checkpoint_participant.cpp
    foundation/test_sha256.cpp
    foundation/test_telemetry.cpp
)
endif()

if(TARGET jyppx::core)
jyppx_add_gtest(jyppx_test_core
    LINK_TARGET jyppx::core
    core/test_core.cpp
)
endif()

if(TARGET jyppx::interop)
jyppx_add_gtest(jyppx_test_interop_checkpoint_manifest
    LINK_TARGET jyppx::interop
    interop/test_checkpoint_manifest.cpp
)
jyppx_add_gtest(jyppx_test_interop_file_state_archive
    LINK_TARGET jyppx::interop
    interop/test_file_state_archive.cpp
)
jyppx_add_gtest(jyppx_test_interop_formats
    LINK_TARGET jyppx::interop
    interop/test_safetensors.cpp
    interop/test_weight_import_report.cpp
    interop/test_export_backend.cpp
    interop/test_model_bundle.cpp
)
if(TARGET jyppx::nn)
    jyppx_add_gtest(jyppx_test_torchscript_compatibility
        LINK_TARGET jyppx::interop
        interop/test_torchscript_compatibility.cpp)
    jyppx_add_gtest(jyppx_test_model_weight_import
        LINK_TARGET jyppx::interop
        interop/test_model_weight_import.cpp)
    if(JYPPX_TEST_CLI_CUDA)
        target_compile_definitions(jyppx_test_torchscript_compatibility PRIVATE JYPPX_TEST_TORCHSCRIPT_CUDA=1)
        target_compile_definitions(jyppx_test_model_weight_import PRIVATE JYPPX_TEST_MODEL_IMPORT_CUDA=1)
    endif()
endif()
if(JYPPX_SAFETENSORS_PYTHON)
    if(NOT EXISTS "${JYPPX_SAFETENSORS_PYTHON}")
        message(FATAL_ERROR "JYPPX_SAFETENSORS_PYTHON must name the locked CPython 3.11 executable")
    endif()
    add_executable(jyppx_test_safetensors_official interop/test_safetensors.cpp)
    target_link_libraries(jyppx_test_safetensors_official PRIVATE jyppx::interop GTest::gtest_main)
    target_compile_definitions(jyppx_test_safetensors_official PRIVATE JYPPX_SAFETENSORS_PYTHON_INTEROP=1)
    jyppx_apply_instrumentation(jyppx_test_safetensors_official)
    jyppx_add_torch_runtime_dependency(jyppx_test_safetensors_official)
    add_test(NAME OfficialSafeTensorsInterchange
        COMMAND "${JYPPX_SAFETENSORS_PYTHON}"
            "${PROJECT_SOURCE_DIR}/scripts/interop/verify_safetensors_interop.py"
            --cpp-test-executable $<TARGET_FILE:jyppx_test_safetensors_official>
            --work-dir "${CMAKE_BINARY_DIR}/evidence/safetensors-$<CONFIG>"
            --jyppx-commit "${JYPPX_GIT_COMMIT}")
    set_tests_properties(OfficialSafeTensorsInterchange PROPERTIES LABELS "interop;official;packaging" TIMEOUT 120)
endif()
endif()

if(TARGET jyppx::experiment)
jyppx_add_gtest(jyppx_test_experiment_config
    LINK_TARGET jyppx::experiment
    experiment/test_config.cpp
)
jyppx_add_gtest(jyppx_test_experiment_artifact_store
    LINK_TARGET jyppx::experiment
    experiment/test_artifact_store.cpp
)
jyppx_add_gtest(jyppx_test_experiment_run_store
    LINK_TARGET jyppx::experiment
    experiment/test_run_store.cpp
)
jyppx_add_gtest(jyppx_test_experiment_run_events
    LINK_TARGET jyppx::experiment
    experiment/test_run_events.cpp
)
target_include_directories(jyppx_test_experiment_run_events SYSTEM PRIVATE "${jyppx_sqlite_SOURCE_DIR}")
jyppx_add_gtest(jyppx_test_experiment_telemetry
    LINK_TARGET jyppx::experiment
    experiment/test_telemetry_exporters.cpp
)
target_link_libraries(jyppx_test_experiment_telemetry PRIVATE nlohmann_json::nlohmann_json)
jyppx_add_gtest(jyppx_test_experiment_evaluation
    LINK_TARGET jyppx::experiment
    experiment/test_evaluation.cpp
)
target_include_directories(jyppx_test_experiment_run_store SYSTEM PRIVATE
    "${jyppx_sqlite_SOURCE_DIR}"
)
endif()

if(TARGET jyppx::data)
jyppx_add_gtest(jyppx_test_data
    LINK_TARGET jyppx::data
    data/test_batch_memory.cpp
    data/test_value_schema.cpp
    data/test_data_pipeline.cpp
    data/test_iterable_data_pipeline.cpp
    data/test_tensor_dataset.cpp
    data/test_samplers.cpp
    data/test_dataset_utils.cpp
    data/test_transforms.cpp
)
endif()

if(TARGET jyppx::sdk)
jyppx_add_gtest(jyppx_test_optim
    optim/test_extreme_params.cpp
)
endif()

if(TARGET jyppx::optim)
jyppx_add_gtest(jyppx_test_optim_adadelta_adapter
    LINK_TARGET jyppx::optim
    optim/test_adadelta_adapter.cpp
)

jyppx_add_gtest(jyppx_test_optim_adagrad_adapter
    LINK_TARGET jyppx::optim
    optim/test_adagrad_adapter.cpp
)

jyppx_add_gtest(jyppx_test_optim_lamb_adapter
    LINK_TARGET jyppx::optim
    optim/test_lamb_adapter.cpp
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
endif()

if(TARGET jyppx::nn)
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

jyppx_add_gtest(jyppx_test_nn_loss_reference_matrix
    LINK_TARGET jyppx::nn
    nn/loss/test_loss_reference_matrix.cpp
)

jyppx_add_gtest(jyppx_test_nn_model_adapter
    LINK_TARGET jyppx::nn
    nn/test_model_adapter.cpp
)

jyppx_add_gtest(jyppx_test_nn_model_checkpoint
    LINK_TARGET jyppx::nn
    nn/test_model_checkpoint_participant.cpp
)
endif()

if(TARGET jyppx::training)
jyppx_add_gtest(jyppx_test_training_checkpoint
    LINK_TARGET jyppx::training
    training/test_checkpoint_coordinator.cpp
)
if(TARGET jyppx::interop)
jyppx_add_gtest(jyppx_test_training_durable_checkpoint
    LINK_TARGET jyppx::training
    training/test_file_checkpoint_store.cpp
    training/test_file_checkpoint_n1.cpp
    training/test_durable_training_recovery.cpp
)
endif()

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
    training/test_telemetry_adapter.cpp
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
endif()

if(TARGET jyppx::sdk)
jyppx_add_gtest(jyppx_test_plan
    test_plan_completion.cpp
)
endif()

if(TARGET jyppx::training)
jyppx_add_gtest(jyppx_test_training
    LINK_TARGET jyppx::training
    training/test_accuracy.cpp
    training/test_metrics.cpp
)
endif()
