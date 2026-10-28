# Unit test executables, one per framework module.

if(TARGET jyppx::liuding::foundation)
jyppx_add_gtest(jyppx_test_foundation
    LINK_TARGET jyppx::liuding::foundation
    foundation/test_checkpoint.cpp
    foundation/test_foundation.cpp
    foundation/test_rng_checkpoint_participant.cpp
    foundation/test_sha256.cpp
    foundation/test_telemetry.cpp
)
endif()

if(TARGET jyppx::liuding::core)
jyppx_add_gtest(jyppx_test_core
    LINK_TARGET jyppx::liuding::core
    core/test_core.cpp
)
endif()

if(TARGET jyppx::liuding::interop)
jyppx_add_gtest(jyppx_test_interop_checkpoint_manifest
    LINK_TARGET jyppx::liuding::interop
    interop/test_checkpoint_manifest.cpp
)
jyppx_add_gtest(jyppx_test_interop_file_state_archive
    LINK_TARGET jyppx::liuding::interop
    interop/test_file_state_archive.cpp
)
jyppx_add_gtest(jyppx_test_interop_formats
    LINK_TARGET jyppx::liuding::interop
    interop/test_safetensors.cpp
    interop/test_weight_import_report.cpp
    interop/test_export_backend.cpp
    interop/test_model_bundle.cpp
)
if(TARGET jyppx::liuding::nn)
    jyppx_add_gtest(jyppx_test_torchscript_compatibility
        LINK_TARGET jyppx::liuding::interop
        interop/test_torchscript_compatibility.cpp)
    jyppx_add_gtest(jyppx_test_model_weight_import
        LINK_TARGET jyppx::liuding::interop
        interop/test_model_weight_import.cpp)
    if(JYPPX_LIUDING_TEST_CLI_CUDA)
        target_compile_definitions(jyppx_test_torchscript_compatibility PRIVATE JYPPX_LIUDING_TEST_TORCHSCRIPT_CUDA=1)
        target_compile_definitions(jyppx_test_model_weight_import PRIVATE JYPPX_LIUDING_TEST_MODEL_IMPORT_CUDA=1)
    endif()
endif()
if(JYPPX_SAFETENSORS_PYTHON)
    if(NOT EXISTS "${JYPPX_SAFETENSORS_PYTHON}")
        message(FATAL_ERROR "JYPPX_SAFETENSORS_PYTHON must name the locked CPython 3.11 executable")
    endif()
    add_executable(jyppx_test_safetensors_official interop/test_safetensors.cpp)
    target_link_libraries(jyppx_test_safetensors_official PRIVATE jyppx::liuding::interop GTest::gtest_main)
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

if(TARGET jyppx::liuding::experiment)
jyppx_add_gtest(jyppx_test_experiment_config
    LINK_TARGET jyppx::liuding::experiment
    experiment/test_config.cpp
)
jyppx_add_gtest(jyppx_test_experiment_artifact_store
    LINK_TARGET jyppx::liuding::experiment
    experiment/test_artifact_store.cpp
)
jyppx_add_gtest(jyppx_test_experiment_run_store
    LINK_TARGET jyppx::liuding::experiment
    experiment/test_run_store.cpp
)
jyppx_add_gtest(jyppx_test_experiment_run_events
    LINK_TARGET jyppx::liuding::experiment
    experiment/test_run_events.cpp
)
target_include_directories(jyppx_test_experiment_run_events SYSTEM PRIVATE "${jyppx_sqlite_SOURCE_DIR}")
jyppx_add_gtest(jyppx_test_experiment_telemetry
    LINK_TARGET jyppx::liuding::experiment
    experiment/test_telemetry_exporters.cpp
)
target_link_libraries(jyppx_test_experiment_telemetry PRIVATE nlohmann_json::nlohmann_json)
jyppx_add_gtest(jyppx_test_experiment_evaluation
    LINK_TARGET jyppx::liuding::experiment
    experiment/test_evaluation.cpp
)
target_include_directories(jyppx_test_experiment_run_store SYSTEM PRIVATE
    "${jyppx_sqlite_SOURCE_DIR}"
)
endif()

if(TARGET jyppx::liuding::data)
jyppx_add_gtest(jyppx_test_data
    LINK_TARGET jyppx::liuding::data
    data/test_batch_memory.cpp
    data/test_batch_consumer_transform.cpp
    data/test_value_schema.cpp
    data/test_data_pipeline.cpp
    data/test_iterable_data_pipeline.cpp
    data/test_tensor_dataset.cpp
    data/test_samplers.cpp
    data/test_dataset_utils.cpp
    data/test_transforms.cpp
)
endif()

if(TARGET jyppx::liuding::data_image)
    jyppx_add_gtest(jyppx_test_data_image_annotations
        LINK_TARGET jyppx::liuding::data_image
        data_image/test_annotations.cpp
        data_image/test_extended_annotations.cpp
        data_image/test_manifests.cpp
        data_image/test_decoder.cpp
        data_image/test_encoder.cpp
        data_image/test_output_sink.cpp
        data_image/test_mask_decoder.cpp
        data_image/test_pixel_spec.cpp
        data_image/test_geometry_trace.cpp
        data_image/test_sample_random.cpp
        data_image/test_joint_transform.cpp
        data_image/test_joint_regions.cpp
        data_image/test_image_batch.cpp
        data_image/test_batch_geometry.cpp
        data_image/test_image_dataset.cpp)
    if(TARGET torch_cuda)
        target_compile_definitions(jyppx_test_data_image_annotations PRIVATE
            JYPPX_LIUDING_IMAGE_TEST_CUDA=1)
    endif()
endif()

if(TARGET jyppx::liuding::vision_render)
    jyppx_add_gtest(jyppx_test_vision_render
        LINK_TARGET jyppx::liuding::vision_render
        visualization/test_render.cpp)
endif()

if(TARGET jyppx::liuding::sdk)
jyppx_add_gtest(jyppx_test_optim
    optim/test_extreme_params.cpp
)
endif()

if(TARGET jyppx::liuding::optim)
jyppx_add_gtest(jyppx_test_optim_adadelta_adapter
    LINK_TARGET jyppx::liuding::optim
    optim/test_adadelta_adapter.cpp
)

jyppx_add_gtest(jyppx_test_optim_adagrad_adapter
    LINK_TARGET jyppx::liuding::optim
    optim/test_adagrad_adapter.cpp
)

jyppx_add_gtest(jyppx_test_optim_lamb_adapter
    LINK_TARGET jyppx::liuding::optim
    optim/test_lamb_adapter.cpp
)

jyppx_add_gtest(jyppx_test_optim_sgd_adapter
    LINK_TARGET jyppx::liuding::optim
    optim/test_sgd_adapter.cpp
)

jyppx_add_gtest(jyppx_test_optim_adam_adapter
    LINK_TARGET jyppx::liuding::optim
    optim/test_adam_adapter.cpp
)

jyppx_add_gtest(jyppx_test_optim_adamw_adapter
    LINK_TARGET jyppx::liuding::optim
    optim/test_adamw_adapter.cpp
)

jyppx_add_gtest(jyppx_test_optim_nadam_adapter
    LINK_TARGET jyppx::liuding::optim
    optim/test_nadam_adapter.cpp
)

jyppx_add_gtest(jyppx_test_optim_rmsprop_adapter
    LINK_TARGET jyppx::liuding::optim
    optim/test_rmsprop_adapter.cpp
)

jyppx_add_gtest(jyppx_test_optim_step_lr_scheduler
    LINK_TARGET jyppx::liuding::optim
    optim/test_step_lr_scheduler.cpp
)

jyppx_add_gtest(jyppx_test_optim_composite_lr_scheduler
    LINK_TARGET jyppx::liuding::optim
    optim/test_composite_lr_scheduler.cpp
)

jyppx_add_gtest(jyppx_test_optim_epoch_lr_scheduler
    LINK_TARGET jyppx::liuding::optim
    optim/test_epoch_lr_scheduler.cpp
)

jyppx_add_gtest(jyppx_test_optim_event_lr_scheduler
    LINK_TARGET jyppx::liuding::optim
    optim/test_event_lr_scheduler.cpp
)

jyppx_add_gtest(jyppx_test_optim_linear_lr_scheduler
    LINK_TARGET jyppx::liuding::optim
    optim/test_linear_lr_scheduler.cpp
)
endif()

if(TARGET jyppx::liuding::nn)
jyppx_add_gtest(jyppx_test_nn_label_smoothing
    LINK_TARGET jyppx::liuding::nn
    nn/loss/test_label_smoothing_cross_entropy.cpp
)

jyppx_add_gtest(jyppx_test_nn_contrastive
    LINK_TARGET jyppx::liuding::nn
    nn/loss/test_contrastive_loss.cpp
)

jyppx_add_gtest(jyppx_test_nn_dice
    LINK_TARGET jyppx::liuding::nn
    nn/loss/test_dice_loss.cpp
)

jyppx_add_gtest(jyppx_test_nn_focal
    LINK_TARGET jyppx::liuding::nn
    nn/loss/test_focal_loss.cpp
)

jyppx_add_gtest(jyppx_test_nn_loss_reference_matrix
    LINK_TARGET jyppx::liuding::nn
    nn/loss/test_loss_reference_matrix.cpp
)

jyppx_add_gtest(jyppx_test_nn_model_adapter
    LINK_TARGET jyppx::liuding::nn
    nn/test_model_adapter.cpp
)

jyppx_add_gtest(jyppx_test_nn_model_checkpoint
    LINK_TARGET jyppx::liuding::nn
    nn/test_model_checkpoint_participant.cpp
)
endif()

if(TARGET jyppx::liuding::training)
jyppx_add_gtest(jyppx_test_training_checkpoint
    LINK_TARGET jyppx::liuding::training
    training/test_checkpoint_coordinator.cpp
)
if(TARGET jyppx::liuding::interop)
jyppx_add_gtest(jyppx_test_training_durable_checkpoint
    LINK_TARGET jyppx::liuding::training
    training/test_file_checkpoint_store.cpp
    training/test_file_checkpoint_n1.cpp
    training/test_durable_training_recovery.cpp
)
endif()

jyppx_add_gtest(jyppx_test_training_scheduler_adapter
    LINK_TARGET jyppx::liuding::training
    training/test_scheduler_adapter.cpp
)

jyppx_add_gtest(jyppx_test_training_classification_metric
    LINK_TARGET jyppx::liuding::training
    training/test_classification_metric.cpp
)

jyppx_add_gtest(jyppx_test_training_segmentation_metric
    LINK_TARGET jyppx::liuding::training
    training/test_segmentation_metric.cpp
)
target_link_libraries(jyppx_test_training_segmentation_metric PRIVATE nlohmann_json::nlohmann_json)
jyppx_add_gtest(jyppx_test_training_detection_metric
    LINK_TARGET jyppx::liuding::training
    training/test_detection_metric.cpp)
target_link_libraries(jyppx_test_training_detection_metric PRIVATE nlohmann_json::nlohmann_json)
jyppx_add_gtest(jyppx_test_training_detection_evaluation
    LINK_TARGET jyppx::liuding::training
    training/test_detection_evaluation.cpp)
if(TARGET jyppx::liuding::experiment)
    target_link_libraries(jyppx_test_training_detection_evaluation PRIVATE jyppx::liuding::experiment)
    target_compile_definitions(jyppx_test_training_detection_evaluation PRIVATE JYPPX_LIUDING_EVALUATION_REPORT_TEST=1)
endif()
jyppx_add_gtest(jyppx_test_training_segmentation_evaluation
    LINK_TARGET jyppx::liuding::training
    training/test_segmentation_evaluation.cpp
)
if(TARGET jyppx::liuding::experiment)
    target_link_libraries(jyppx_test_training_segmentation_evaluation PRIVATE jyppx::liuding::experiment)
    target_compile_definitions(jyppx_test_training_segmentation_evaluation PRIVATE
        JYPPX_LIUDING_EVALUATION_REPORT_TEST=1)
endif()

jyppx_add_gtest(jyppx_test_training_spatial_evaluation_sessions
    LINK_TARGET jyppx::liuding::training
    training/test_spatial_evaluation_sessions.cpp)
if(TARGET jyppx::liuding::experiment)
    target_link_libraries(jyppx_test_training_spatial_evaluation_sessions PRIVATE
        jyppx::liuding::experiment)
    target_compile_definitions(jyppx_test_training_spatial_evaluation_sessions PRIVATE
        JYPPX_LIUDING_EVALUATION_REPORT_TEST=1)
endif()

jyppx_add_gtest(jyppx_test_training_classification_evaluation
    LINK_TARGET jyppx::liuding::training
    training/test_classification_evaluation.cpp
    training/test_classification_evaluation_session.cpp
)
if(TARGET jyppx::liuding::experiment)
    target_link_libraries(jyppx_test_training_classification_evaluation PRIVATE
        jyppx::liuding::experiment)
    target_compile_definitions(jyppx_test_training_classification_evaluation PRIVATE
        JYPPX_LIUDING_EVALUATION_REPORT_TEST=1)
endif()

if(TARGET jyppx::liuding::models AND TARGET jyppx::liuding::data_image
        AND TARGET jyppx::liuding::experiment)
    jyppx_add_gtest(jyppx_test_models_classification
        LINK_TARGET jyppx::liuding::models
        models/test_models_classification.cpp)
    target_link_libraries(jyppx_test_models_classification PRIVATE
        jyppx::liuding::training jyppx::liuding::data_image
        jyppx::liuding::experiment)
endif()

jyppx_add_gtest(jyppx_test_training_ranking_metric
    LINK_TARGET jyppx::liuding::training
    training/test_ranking_metric.cpp
)

if(TARGET jyppx::liuding::training AND TARGET jyppx::liuding::data_image AND TARGET jyppx::liuding::interop
        AND TARGET jyppx::liuding::optim AND TARGET jyppx::liuding::experiment)
    jyppx_add_gtest(jyppx_test_training_real_spatial_task
        LINK_TARGET jyppx::liuding::training
        training/test_real_spatial_task.cpp
        ENVIRONMENT "CUBLAS_WORKSPACE_CONFIG=:4096:8")
    target_link_libraries(jyppx_test_training_real_spatial_task PRIVATE
        jyppx::liuding::data_image jyppx::liuding::interop
        jyppx::liuding::optim jyppx::liuding::experiment
        nlohmann_json::nlohmann_json)
    if(TARGET torch_cuda OR TORCH_CUDA_VERSION)
        target_compile_definitions(jyppx_test_training_real_spatial_task PRIVATE
            JYPPX_LIUDING_SPATIAL_CUDA_METRICS=1)
    endif()
    if(WIN32)
        target_link_libraries(jyppx_test_training_real_spatial_task PRIVATE psapi)
    endif()
endif()

jyppx_add_gtest(jyppx_test_training_callback_dispatcher
    LINK_TARGET jyppx::liuding::training
    training/test_callback_dispatcher.cpp
)

jyppx_add_gtest(jyppx_test_training_event_sink
    LINK_TARGET jyppx::liuding::training
    training/test_event_sink.cpp
    training/test_telemetry_adapter.cpp
)

jyppx_add_gtest(jyppx_test_training_single_device_strategy
    LINK_TARGET jyppx::liuding::training
    training/test_single_device_strategy.cpp
)

jyppx_add_gtest(jyppx_test_training_trainer
    LINK_TARGET jyppx::liuding::training
    training/test_trainer.cpp
)

jyppx_add_gtest(jyppx_test_training_diagnostics
    LINK_TARGET jyppx::liuding::training
    training/test_diagnostics.cpp
)
if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(jyppx_test_training_diagnostics PRIVATE nlohmann_json::nlohmann_json)
    target_compile_definitions(jyppx_test_training_diagnostics PRIVATE JYPPX_LIUDING_TEST_DIAGNOSTIC_JSON=1)
endif()
if(TARGET jyppx::liuding::experiment)
    target_link_libraries(jyppx_test_training_diagnostics PRIVATE jyppx::liuding::experiment)
    target_compile_definitions(jyppx_test_training_diagnostics PRIVATE JYPPX_LIUDING_TEST_DIAGNOSTIC_STORE=1)
endif()
if(TARGET jyppx::liuding::interop)
    target_link_libraries(jyppx_test_training_diagnostics PRIVATE jyppx::liuding::interop)
endif()

jyppx_add_gtest(jyppx_test_training_amp
    LINK_TARGET jyppx::liuding::training
    training/test_amp.cpp
    training/test_grad_scaler_checkpoint_participant.cpp
    training/test_nan_inf.cpp
)
endif()

if(TARGET jyppx::liuding::sdk)
jyppx_add_gtest(jyppx_test_plan
    test_plan_completion.cpp
)
endif()

if(TARGET jyppx::liuding::training)
jyppx_add_gtest(jyppx_test_training
    LINK_TARGET jyppx::liuding::training
    training/test_accuracy.cpp
    training/test_metrics.cpp
)
endif()


if(TARGET jyppx::liuding::models)
    set(models_test_environment)
    if(JYPPX_LIUDING_MODELS_REFERENCE_DIR)
        if(JYPPX_LIUDING_MODELS_REFERENCE_DIR MATCHES ";")
            message(FATAL_ERROR "JYPPX_LIUDING_MODELS_REFERENCE_DIR must be one directory")
        endif()
        foreach(reference_architecture IN ITEMS resnet18 resnet50
                mobilenet_v3_small mobilenet_v3_large transformer_encoder
                vit_small_patch14_dinov2 vit_base_patch14_dinov2
                vit_large_patch14_dinov2 vit_giant_patch14_dinov2)
            foreach(reference_role IN ITEMS state input output)
                set(reference_file "${reference_architecture}-${reference_role}.safetensors")
                if(NOT EXISTS "${JYPPX_LIUDING_MODELS_REFERENCE_DIR}/${reference_file}")
                    message(FATAL_ERROR "Missing official models fixture: ${reference_file}")
                endif()
            endforeach()
        endforeach()
        list(APPEND models_test_environment
             "JYPPX_LIUDING_MODELS_REFERENCE_DIR=${JYPPX_LIUDING_MODELS_REFERENCE_DIR}")
    endif()
    jyppx_add_gtest(jyppx_test_models_features
        LINK_TARGET jyppx::liuding::models
        models/test_model_features.cpp
    )
    jyppx_add_gtest(jyppx_test_models_heads
        LINK_TARGET jyppx::liuding::models
        models/test_model_heads.cpp
    )
    jyppx_add_gtest(jyppx_test_models_preprocessing
        LINK_TARGET jyppx::liuding::models
        models/test_model_preprocessing.cpp
    )
    jyppx_add_gtest(jyppx_test_models_pretrained
        LINK_TARGET jyppx::liuding::models
        models/test_models_pretrained.cpp
    )
    if(TARGET torch_cuda OR TORCH_CUDA_VERSION)
        target_compile_definitions(jyppx_test_models_pretrained PRIVATE
            JYPPX_LIUDING_PRETRAINED_CUDA_TESTS=1)
    endif()
    if(TARGET jyppx::liuding::data_image)
        target_link_libraries(jyppx_test_models_pretrained PRIVATE
            jyppx::liuding::data_image)
        target_compile_definitions(jyppx_test_models_pretrained PRIVATE
            JYPPX_LIUDING_PRETRAINED_IMAGE_TESTS=1)
    endif()
    if(TARGET jyppx::liuding::data_image)
        target_link_libraries(jyppx_test_models_preprocessing PRIVATE
            jyppx::liuding::data_image)
        target_compile_definitions(jyppx_test_models_preprocessing PRIVATE
            JYPPX_LIUDING_PREPROCESS_IMAGE_TESTS=1)
        if(TARGET torch_cuda OR TORCH_CUDA_VERSION)
            target_compile_definitions(jyppx_test_models_preprocessing PRIVATE
                JYPPX_LIUDING_PREPROCESS_CUDA_TESTS=1)
        endif()
    endif()
    jyppx_add_gtest(jyppx_test_models
        LINK_TARGET jyppx::liuding::models
        models/test_models.cpp
        models/test_position_interpolation.cpp
        ENVIRONMENT ${models_test_environment} "CUBLAS_WORKSPACE_CONFIG=:4096:8"
    )
    if(TARGET torch_cuda OR TORCH_CUDA_VERSION)
        target_compile_definitions(jyppx_test_models PRIVATE
            JYPPX_LIUDING_MODELS_CUDA_METRICS=1)
    endif()
    if(JYPPX_LIUDING_MODELS_REFERENCE_DIR)
        target_compile_definitions(jyppx_test_models PRIVATE
            JYPPX_LIUDING_MODELS_PYTHON_INTEROP=1)
    endif()
    if(TARGET jyppx::liuding::training)
        jyppx_add_gtest(jyppx_test_models_amp
            LINK_TARGET jyppx::liuding::models
            models/test_models_amp.cpp
        )
        target_link_libraries(jyppx_test_models_amp PRIVATE
            jyppx::liuding::training)
        add_executable(jyppx_test_models_recovery
            models/test_models_recovery.cpp
        )
        target_link_libraries(jyppx_test_models_recovery PRIVATE
            jyppx::liuding::models jyppx::liuding::training GTest::gtest_main)
        jyppx_apply_instrumentation(jyppx_test_models_recovery)
        jyppx_add_torch_runtime_dependency(jyppx_test_models_recovery)
        gtest_discover_tests(jyppx_test_models_recovery
            DISCOVERY_MODE PRE_TEST DISCOVERY_TIMEOUT 30
            TEST_LIST jyppx_dino_recovery_tests
            PROPERTIES RUN_SERIAL TRUE TIMEOUT 180
                ENVIRONMENT "CUBLAS_WORKSPACE_CONFIG=:4096:8")
        # Apply the larger checkpoint I/O timeout only to its explicit profile.
        # TEST_INCLUDE_FILES works with our CMake 3.18 minimum and runs after
        # the preceding discovery include has populated the test list.
        file(GENERATE
            OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/DinoRecoveryLimits.cmake"
            CONTENT [[
foreach(jyppx_dino_recovery_test IN LISTS jyppx_dino_recovery_tests)
    if(jyppx_dino_recovery_test MATCHES "/dinov3_convnext_large_")
        set_tests_properties("${jyppx_dino_recovery_test}" PROPERTIES TIMEOUT 360)
    elseif(jyppx_dino_recovery_test MATCHES "/(dinov3_vitl16(plus)?|vit_large_patch14_dinov2)_")
        set_tests_properties("${jyppx_dino_recovery_test}" PROPERTIES TIMEOUT 600)
    endif()
endforeach()
]])
        set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES
            "${CMAKE_CURRENT_BINARY_DIR}/DinoRecoveryLimits.cmake")
        if(TARGET torch_cuda OR TORCH_CUDA_VERSION)
            target_compile_definitions(jyppx_test_models_amp PRIVATE
                JYPPX_LIUDING_MODELS_CUDA_METRICS=1)
            target_compile_definitions(jyppx_test_models_recovery PRIVATE
                JYPPX_LIUDING_MODELS_CUDA_METRICS=1)
        endif()
    endif()
endif()

if(TARGET jyppx::liuding::quantization)
    jyppx_add_gtest(jyppx_test_quantization
        LINK_TARGET jyppx::liuding::quantization
        quantization/test_quantization.cpp
    )
endif()

if(TARGET jyppx::liuding::serving)
    jyppx_add_gtest(jyppx_test_serving
        LINK_TARGET jyppx::liuding::serving
        serving/test_serving.cpp
    )
    jyppx_add_gtest(jyppx_test_serving_torchscript
        LINK_TARGET jyppx::liuding::serving
        serving/test_serving_torchscript.cpp
    )
endif()
