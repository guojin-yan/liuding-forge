if(TARGET jyppx::liuding::sdk)
add_executable(jyppx_basic_data_pipeline basic_data_pipeline.cpp)
set_target_properties(jyppx_basic_data_pipeline PROPERTIES
    OUTPUT_NAME liuding-forge-example-basic-data-pipeline)
target_link_libraries(jyppx_basic_data_pipeline PRIVATE jyppx::liuding::sdk)

add_executable(jyppx_basic_trainer basic_trainer.cpp)
set_target_properties(jyppx_basic_trainer PROPERTIES
    OUTPUT_NAME liuding-forge-example-basic-trainer)
target_link_libraries(jyppx_basic_trainer PRIVATE jyppx::liuding::sdk)

add_executable(jyppx_mnist mnist/mnist.cpp)
set_target_properties(jyppx_mnist PROPERTIES
    OUTPUT_NAME liuding-forge-example-mnist)
target_link_libraries(jyppx_mnist PRIVATE jyppx::liuding::sdk)
if(JYPPX_LIUDING_BUILD_TESTS AND Python3_Interpreter_FOUND)
    add_test(NAME MnistExampleOffline
        COMMAND ${Python3_EXECUTABLE}
            "${PROJECT_SOURCE_DIR}/tests/scripts/test_mnist_example.py"
            --binary $<TARGET_FILE:jyppx_mnist>)
    set_tests_properties(MnistExampleOffline PROPERTIES
        LABELS "examples;data;training" TIMEOUT 180)
    if(JYPPX_LIUDING_TEST_CLI_CUDA)
        add_test(NAME MnistExampleOfflineCUDA
            COMMAND ${Python3_EXECUTABLE}
                "${PROJECT_SOURCE_DIR}/tests/scripts/test_mnist_example.py"
                --binary $<TARGET_FILE:jyppx_mnist> --device cuda)
        set_tests_properties(MnistExampleOfflineCUDA PROPERTIES
            LABELS "examples;data;training;cuda" TIMEOUT 180)
    endif()
endif()

add_executable(jyppx_distillation distillation.cpp)
set_target_properties(jyppx_distillation PROPERTIES
    OUTPUT_NAME liuding-forge-example-distillation)
target_link_libraries(jyppx_distillation PRIVATE jyppx::liuding::sdk)

# LibTorch's Windows distribution keeps runtime DLLs beside its import
# libraries. Copy the DLLs (not the whole directory, which also holds large
# import libraries) once before the examples build.
foreach(example_target IN ITEMS
        jyppx_basic_data_pipeline jyppx_basic_trainer
        jyppx_mnist jyppx_distillation)
    jyppx_add_torch_runtime_dependency(${example_target})
endforeach()


endif()

if(TARGET jyppx_dataset_preparation)
    add_executable(jyppx_paired_affine_data data/paired_affine.cpp)
    set_target_properties(jyppx_paired_affine_data PROPERTIES
        OUTPUT_NAME liuding-forge-example-paired-affine-data)
    target_link_libraries(jyppx_paired_affine_data PRIVATE
        jyppx_dataset_preparation nlohmann_json::nlohmann_json)
    jyppx_apply_instrumentation(jyppx_paired_affine_data)
    jyppx_add_torch_runtime_dependency(jyppx_paired_affine_data)
endif()

if(JYPPX_LIUDING_PREPARE_FASHION_MNIST AND TARGET jyppx_dataset_preparation
   AND TARGET jyppx::liuding::data_image)
    add_executable(jyppx_fashion_mnist_data data/fashion_mnist.cpp)
    set_target_properties(jyppx_fashion_mnist_data PROPERTIES
        OUTPUT_NAME liuding-forge-example-fashion-mnist-data)
    target_link_libraries(jyppx_fashion_mnist_data PRIVATE
        jyppx_dataset_preparation jyppx::liuding::data_image nlohmann_json::nlohmann_json)
    jyppx_apply_instrumentation(jyppx_fashion_mnist_data)
    jyppx_add_torch_runtime_dependency(jyppx_fashion_mnist_data)
endif()

if(TARGET jyppx::liuding::serving)
    add_executable(jyppx_liuding_serving_reload serving/reload.cpp)
    set_target_properties(jyppx_liuding_serving_reload PROPERTIES
        OUTPUT_NAME liuding-forge-example-serving-reload)
    target_link_libraries(jyppx_liuding_serving_reload PRIVATE jyppx::liuding::serving)
    jyppx_apply_instrumentation(jyppx_liuding_serving_reload)
    jyppx_add_torch_runtime_dependency(jyppx_liuding_serving_reload)
    if(JYPPX_LIUDING_BUILD_TESTS)
        add_test(NAME ServingReloadExample COMMAND jyppx_liuding_serving_reload)
        set_tests_properties(ServingReloadExample PROPERTIES
            LABELS "examples;serving" TIMEOUT 60)
    endif()
    add_executable(jyppx_liuding_serving_jsonl serving/jsonl_host.cpp)
    set_target_properties(jyppx_liuding_serving_jsonl PROPERTIES
        OUTPUT_NAME liuding-forge-example-serving-jsonl)
    target_link_libraries(jyppx_liuding_serving_jsonl PRIVATE
        jyppx::liuding::serving nlohmann_json::nlohmann_json)
    jyppx_apply_instrumentation(jyppx_liuding_serving_jsonl)
    jyppx_add_torch_runtime_dependency(jyppx_liuding_serving_jsonl)
endif()

if(TARGET jyppx::liuding::nn AND TARGET jyppx::liuding::interop
   AND TARGET jyppx::liuding::onnx_export AND TARGET jyppx::liuding::onnx_runtime)
    add_executable(jyppx_liuding_deployment_checkpoint deployment/checkpoint_onnx.cpp)
    set_target_properties(jyppx_liuding_deployment_checkpoint PROPERTIES
        OUTPUT_NAME liuding-forge-example-checkpoint-onnx)
    target_link_libraries(jyppx_liuding_deployment_checkpoint PRIVATE
        jyppx::liuding::nn jyppx::liuding::interop
        jyppx::liuding::onnx_export jyppx::liuding::onnx_runtime)
    jyppx_apply_instrumentation(jyppx_liuding_deployment_checkpoint)
    jyppx_add_torch_runtime_dependency(jyppx_liuding_deployment_checkpoint)
    jyppx_add_onnx_runtime_dependency(jyppx_liuding_deployment_checkpoint)
    if(JYPPX_LIUDING_BUILD_TESTS AND Python3_Interpreter_FOUND)
        add_test(NAME DeploymentCheckpointOnnxExample
            COMMAND ${Python3_EXECUTABLE}
                "${PROJECT_SOURCE_DIR}/tests/scripts/test_deployment_example.py"
                --binary $<TARGET_FILE:jyppx_liuding_deployment_checkpoint>)
        set_tests_properties(DeploymentCheckpointOnnxExample PROPERTIES
            LABELS "examples;deployment;onnx_export;onnx_runtime" TIMEOUT 180)
    endif()
endif()

if(TARGET jyppx::liuding::onnx_serving AND TARGET jyppx::liuding::onnx_export)
    add_executable(jyppx_liuding_serving_onnx serving/onnx.cpp)
    set_target_properties(jyppx_liuding_serving_onnx PROPERTIES
        OUTPUT_NAME liuding-forge-example-serving-onnx)
    target_link_libraries(jyppx_liuding_serving_onnx PRIVATE
        jyppx::liuding::onnx_serving jyppx::liuding::onnx_export
        jyppx::liuding::foundation)
    jyppx_apply_instrumentation(jyppx_liuding_serving_onnx)
    jyppx_add_torch_runtime_dependency(jyppx_liuding_serving_onnx)
    jyppx_add_onnx_runtime_dependency(jyppx_liuding_serving_onnx)
    if(JYPPX_LIUDING_BUILD_TESTS)
        add_test(NAME ServingOnnxExample COMMAND jyppx_liuding_serving_onnx)
        set_tests_properties(ServingOnnxExample PROPERTIES
            LABELS "examples;serving;deployment;onnx_runtime" TIMEOUT 120)
    endif()
endif()

if(TARGET jyppx::liuding::experiment AND TARGET jyppx::liuding::nn
   AND TARGET jyppx::liuding::interop AND TARGET jyppx::liuding::training
   AND TARGET jyppx::liuding::optim)
    add_executable(jyppx_liuding_search_trial experiment/search_trial.cpp)
    set_target_properties(jyppx_liuding_search_trial PROPERTIES
        OUTPUT_NAME liuding-forge-example-search-trial)
    target_link_libraries(jyppx_liuding_search_trial PRIVATE
        jyppx::liuding::experiment jyppx::liuding::nn
        jyppx::liuding::interop jyppx::liuding::training jyppx::liuding::optim
        nlohmann_json::nlohmann_json)
    jyppx_apply_instrumentation(jyppx_liuding_search_trial)
    jyppx_add_torch_runtime_dependency(jyppx_liuding_search_trial)
    if(JYPPX_LIUDING_BUILD_TESTS AND Python3_Interpreter_FOUND
       AND TARGET jyppx_cli_recipe_gate)
        add_test(NAME LocalSearchTrainingExample
            COMMAND ${Python3_EXECUTABLE}
                "${PROJECT_SOURCE_DIR}/tests/scripts/test_search_training_example.py"
                --worker $<TARGET_FILE:jyppx_liuding_search_trial>
                --cli $<TARGET_FILE:jyppx_cli_recipe_gate>)
        set_tests_properties(LocalSearchTrainingExample PROPERTIES
            LABELS "examples;experiment;training;cli" TIMEOUT 240)
    endif()
endif()

if(TARGET jyppx::liuding::bundle_signing)
    add_executable(jyppx_liuding_bundle_signing_example deployment/bundle_signing.cpp)
    set_target_properties(jyppx_liuding_bundle_signing_example PROPERTIES
        OUTPUT_NAME liuding-forge-example-bundle-signing)
    target_link_libraries(jyppx_liuding_bundle_signing_example PRIVATE
        jyppx::liuding::bundle_signing)
    jyppx_apply_instrumentation(jyppx_liuding_bundle_signing_example)
    jyppx_add_torch_runtime_dependency(jyppx_liuding_bundle_signing_example)
    if(JYPPX_LIUDING_BUILD_TESTS)
        add_test(NAME BundleSigningExample
            COMMAND jyppx_liuding_bundle_signing_example)
        set_tests_properties(BundleSigningExample PROPERTIES
            LABELS "examples;deployment;signing" TIMEOUT 60)
    endif()
endif()

if(TARGET jyppx::liuding::quantization AND TARGET jyppx::liuding::onnx_export
   AND TARGET jyppx::liuding::onnx_runtime)
    add_executable(jyppx_liuding_quantization_per_output_channel_example
        quantization/per_output_channel.cpp)
    set_target_properties(jyppx_liuding_quantization_per_output_channel_example PROPERTIES
        OUTPUT_NAME liuding-forge-example-int8-per-output-channel)
    target_link_libraries(jyppx_liuding_quantization_per_output_channel_example PRIVATE
        jyppx::liuding::quantization jyppx::liuding::onnx_export
        jyppx::liuding::onnx_runtime)
    jyppx_apply_instrumentation(jyppx_liuding_quantization_per_output_channel_example)
    jyppx_add_torch_runtime_dependency(jyppx_liuding_quantization_per_output_channel_example)
    jyppx_add_onnx_runtime_dependency(jyppx_liuding_quantization_per_output_channel_example)
    if(JYPPX_LIUDING_BUILD_TESTS)
        add_test(NAME PerOutputChannelInt8Example
            COMMAND jyppx_liuding_quantization_per_output_channel_example)
        set_tests_properties(PerOutputChannelInt8Example PROPERTIES
            LABELS "examples;quantization;deployment;onnx_runtime" TIMEOUT 120)
    endif()
endif()

# Public models/optimizer lesson; local weights are explicit at runtime.
if(TARGET jyppx::liuding::models AND TARGET jyppx::liuding::optim)
    add_executable(jyppx_transfer_group_rates EXCLUDE_FROM_ALL
        transfer_learning/group_rates.cpp)
    set_target_properties(jyppx_transfer_group_rates PROPERTIES
        OUTPUT_NAME liuding-forge-example-transfer-group-rates)
    target_link_libraries(jyppx_transfer_group_rates PRIVATE
        jyppx::liuding::models jyppx::liuding::optim)
    jyppx_apply_instrumentation(jyppx_transfer_group_rates)
    jyppx_add_torch_runtime_dependency(jyppx_transfer_group_rates)
endif()

if(JYPPX_LIUDING_PREPARE_CIFAR10 AND TARGET jyppx_dataset_preparation)
    add_executable(jyppx_cifar10_data data/cifar10.cpp)
    set_target_properties(jyppx_cifar10_data PROPERTIES
        OUTPUT_NAME liuding-forge-example-cifar10-data)
    target_link_libraries(jyppx_cifar10_data PRIVATE
        jyppx_dataset_preparation nlohmann_json::nlohmann_json)
    jyppx_apply_instrumentation(jyppx_cifar10_data)
    jyppx_add_torch_runtime_dependency(jyppx_cifar10_data)
endif()

if(JYPPX_LIUDING_PREPARE_MNIST AND TARGET jyppx_dataset_preparation)
    add_executable(jyppx_mnist_data data/mnist.cpp)
    set_target_properties(jyppx_mnist_data PROPERTIES
        OUTPUT_NAME liuding-forge-example-mnist-data)
    target_link_libraries(jyppx_mnist_data PRIVATE
        jyppx_dataset_preparation nlohmann_json::nlohmann_json)
    jyppx_apply_instrumentation(jyppx_mnist_data)
    jyppx_add_torch_runtime_dependency(jyppx_mnist_data)
endif()


if(TARGET jyppx::liuding::telemetry_native)
    add_executable(jyppx_liuding_native_tensorboard_example experiment/tensorboard.cpp)
    set_target_properties(jyppx_liuding_native_tensorboard_example PROPERTIES
        OUTPUT_NAME liuding-forge-example-native-tensorboard)
    target_link_libraries(jyppx_liuding_native_tensorboard_example PRIVATE
        jyppx::liuding::telemetry_native jyppx::liuding::experiment)
    jyppx_apply_instrumentation(jyppx_liuding_native_tensorboard_example)
    jyppx_add_torch_runtime_dependency(jyppx_liuding_native_tensorboard_example)
    if(JYPPX_LIUDING_BUILD_TESTS AND JYPPX_TELEMETRY_PYTHON)
        add_test(NAME NativeTensorBoardExample
            COMMAND "${JYPPX_TELEMETRY_PYTHON}"
                "${PROJECT_SOURCE_DIR}/tests/scripts/test_native_tensorboard.py"
                --example $<TARGET_FILE:jyppx_liuding_native_tensorboard_example>
                --sidecar "${PROJECT_SOURCE_DIR}/tools/telemetry_sidecar/jyppx_telemetry_sidecar.py")
        set_tests_properties(NativeTensorBoardExample PROPERTIES
            LABELS "examples;experiment;telemetry;tensorboard;official" TIMEOUT 180)
    elseif(JYPPX_LIUDING_BUILD_TESTS AND JYPPX_LIUDING_TEST_OFFICIAL_TELEMETRY)
        message(FATAL_ERROR "Official native telemetry acceptance requires JYPPX_TELEMETRY_PYTHON")
    endif()
endif()

if(TARGET jyppx::liuding::telemetry_native)
    add_executable(jyppx_liuding_otlp_projection_example experiment/otlp_projection.cpp)
    set_target_properties(jyppx_liuding_otlp_projection_example PROPERTIES
        OUTPUT_NAME liuding-forge-example-otlp-projection)
    target_link_libraries(jyppx_liuding_otlp_projection_example PRIVATE
        jyppx::liuding::telemetry_native jyppx::liuding::experiment
        nlohmann_json::nlohmann_json)
    jyppx_apply_instrumentation(jyppx_liuding_otlp_projection_example)
    jyppx_add_torch_runtime_dependency(jyppx_liuding_otlp_projection_example)
    if(JYPPX_LIUDING_BUILD_TESTS AND JYPPX_TELEMETRY_PYTHON)
        add_test(NAME NativeOtlpProjectionExample
            COMMAND "${JYPPX_TELEMETRY_PYTHON}"
                "${PROJECT_SOURCE_DIR}/tests/scripts/test_native_otlp_projection.py"
                --example $<TARGET_FILE:jyppx_liuding_otlp_projection_example>
                --sidecar "${PROJECT_SOURCE_DIR}/tools/telemetry_sidecar/jyppx_telemetry_sidecar.py")
        set_tests_properties(NativeOtlpProjectionExample PROPERTIES
            LABELS "examples;experiment;telemetry;otlp;official" TIMEOUT 180)
    elseif(JYPPX_LIUDING_BUILD_TESTS AND JYPPX_LIUDING_TEST_OFFICIAL_TELEMETRY)
        message(FATAL_ERROR "Official native OTLP acceptance requires JYPPX_TELEMETRY_PYTHON")
    endif()
endif()

if(TARGET jyppx::liuding::telemetry_http)
    add_executable(jyppx_liuding_otlp_http_example experiment/otlp_http.cpp)
    set_target_properties(jyppx_liuding_otlp_http_example PROPERTIES
        OUTPUT_NAME liuding-forge-example-otlp-http)
    target_link_libraries(jyppx_liuding_otlp_http_example PRIVATE
        jyppx::liuding::telemetry_http)
    jyppx_apply_instrumentation(jyppx_liuding_otlp_http_example)
    jyppx_add_torch_runtime_dependency(jyppx_liuding_otlp_http_example)
endif()

# Frozen representation learning lesson; no external inputs by default.
if(TARGET jyppx::liuding::models AND TARGET jyppx::liuding::optim)
    add_executable(jyppx_transfer_linear_probe EXCLUDE_FROM_ALL
        transfer_learning/linear_probe.cpp)
    set_target_properties(jyppx_transfer_linear_probe PROPERTIES
        OUTPUT_NAME liuding-forge-example-transfer-linear-probe)
    target_link_libraries(jyppx_transfer_linear_probe PRIVATE
        jyppx::liuding::models jyppx::liuding::optim)
    jyppx_apply_instrumentation(jyppx_transfer_linear_probe)
    jyppx_add_torch_runtime_dependency(jyppx_transfer_linear_probe)
endif()

# Select shared DINO intermediate layers without retaining later activations.
if(TARGET jyppx::liuding::models)
    add_executable(jyppx_dino_intermediate_features EXCLUDE_FROM_ALL
        transfer_learning/intermediate_features.cpp)
    set_target_properties(jyppx_dino_intermediate_features PROPERTIES
        OUTPUT_NAME liuding-forge-example-dino-intermediate-features)
    target_link_libraries(jyppx_dino_intermediate_features PRIVATE
        jyppx::liuding::models)
    jyppx_apply_instrumentation(jyppx_dino_intermediate_features)
    jyppx_add_torch_runtime_dependency(jyppx_dino_intermediate_features)
endif()
