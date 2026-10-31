file(TO_CMAKE_PATH "${Python3_EXECUTABLE}" jyppx_cli_test_python)

jyppx_add_gtest(jyppx_test_sidecar_transport
    LINK_TARGET jyppx_cli_transport
    test_sidecar_transport.cpp)
target_link_libraries(jyppx_test_sidecar_transport PRIVATE nlohmann_json::nlohmann_json)
target_compile_definitions(jyppx_test_sidecar_transport PRIVATE
    JYPPX_SIDECAR_TEST_PYTHON="${jyppx_cli_test_python}"
    JYPPX_SIDECAR_TEST_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/sidecar_fixture.py")

jyppx_add_gtest(jyppx_test_native_telemetry_protocol
    LINK_TARGET jyppx_cli_transport
    test_native_telemetry_protocol.cpp)
target_link_libraries(jyppx_test_native_telemetry_protocol PRIVATE
    nlohmann_json::nlohmann_json)

if(TARGET jyppx::liuding::telemetry_http)
    add_test(NAME Cli_native_otlp_worker
        COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/test_native_otlp_worker.py"
            --cli $<TARGET_FILE:jyppx_cli>)
    set_tests_properties(Cli_native_otlp_worker PROPERTIES
        LABELS "cli;telemetry_http;integration" TIMEOUT 180)
else()
    add_test(NAME Cli_native_otlp_disabled
        COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/test_native_otlp_worker.py"
            --cli $<TARGET_FILE:jyppx_cli> --disabled)
    set_tests_properties(Cli_native_otlp_disabled PROPERTIES
        LABELS "cli;telemetry_http;integration" TIMEOUT 180)
endif()

if(TARGET jyppx::liuding::telemetry_native)
    add_test(NAME Cli_native_tensorboard_worker
        COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/test_native_tensorboard_worker.py"
            --cli $<TARGET_FILE:jyppx_cli>)
    set_tests_properties(Cli_native_tensorboard_worker PROPERTIES
        LABELS "cli;telemetry_native;tensorboard;integration" TIMEOUT 240)
else()
    add_test(NAME Cli_native_tensorboard_disabled
        COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/test_native_tensorboard_worker.py"
            --cli $<TARGET_FILE:jyppx_cli> --disabled)
    set_tests_properties(Cli_native_tensorboard_disabled PROPERTIES
        LABELS "cli;telemetry_native;tensorboard;integration" TIMEOUT 120)
endif()

jyppx_add_gtest(jyppx_test_local_process
    LINK_TARGET jyppx_cli_transport
    test_local_process.cpp)
target_link_libraries(jyppx_test_local_process PRIVATE nlohmann_json::nlohmann_json)
target_compile_definitions(jyppx_test_local_process PRIVATE
    JYPPX_LOCAL_PROCESS_PYTHON="${jyppx_cli_test_python}"
    JYPPX_LOCAL_PROCESS_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/local_process_fixture.py")

add_test(NAME Cli_local_attempt
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_local_attempt.py"
        --cli $<TARGET_FILE:jyppx_cli_recipe_gate>)
set_tests_properties(Cli_local_attempt PROPERTIES
    LABELS "cli;experiment;integration" TIMEOUT 240)
add_test(NAME Cli_local_search
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_local_search.py"
        --cli $<TARGET_FILE:jyppx_cli_recipe_gate>)
set_tests_properties(Cli_local_search PROPERTIES
    LABELS "cli;experiment;integration" TIMEOUT 600)

set(bundle_signing_test_mode off)
if(TARGET jyppx::liuding::bundle_signing)
    set(bundle_signing_test_mode on)
endif()
add_test(NAME Cli_bundle_signing
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_bundle_signing.py"
        --cli $<TARGET_FILE:jyppx_cli_recipe_gate>
        --bundle-signing ${bundle_signing_test_mode})
set_tests_properties(Cli_bundle_signing PROPERTIES
    LABELS "cli;deployment;signing;integration" TIMEOUT 180)
if(TARGET jyppx::liuding::bundle_signing AND TARGET jyppx::liuding::onnx_export
   AND TARGET jyppx::liuding::onnx_runtime)
    add_test(NAME Cli_bundle_signing_onnx
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_bundle_signing_onnx.py"
            --cli $<TARGET_FILE:jyppx_cli_recipe_gate>
            --exporter $<TARGET_FILE:jyppx_test_onnx_export>)
    set_tests_properties(Cli_bundle_signing_onnx PROPERTIES
        LABELS "cli;deployment;signing;onnx_runtime;integration" TIMEOUT 300)
endif()

foreach(contract IN ITEMS model_checkpoint model_bundle evaluation_data offline_workflow run_telemetry migration)
    add_test(NAME Cli_${contract}
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_${contract}.py"
            --cli $<TARGET_FILE:jyppx_cli>)
    set_tests_properties(Cli_${contract} PROPERTIES LABELS "cli;integration" TIMEOUT 180)
endforeach()

if(TARGET jyppx::liuding::onnx_export)
    add_test(NAME Cli_onnx_delivery
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_onnx_delivery.py"
            --cli $<TARGET_FILE:jyppx_cli_recipe_gate>
            --exporter $<TARGET_FILE:jyppx_test_onnx_export>)
    set_tests_properties(Cli_onnx_delivery PROPERTIES
        LABELS "cli;deployment;onnx_export;integration" TIMEOUT 300)
endif()

if(TARGET jyppx::liuding::onnx_export AND TARGET jyppx::liuding::onnx_runtime)
    add_test(NAME Cli_onnx_verification
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_onnx_verification.py"
            --cli $<TARGET_FILE:jyppx_cli_recipe_gate>
            --exporter $<TARGET_FILE:jyppx_test_onnx_export>)
    set_tests_properties(Cli_onnx_verification PROPERTIES
        LABELS "cli;deployment;onnx_runtime;integration" TIMEOUT 300)
    set(onnx_reconstruction_arguments)
    if(TARGET jyppx::liuding::models AND TARGET jyppx::liuding::data_image)
        list(APPEND onnx_reconstruction_arguments --image-recipe)
    endif()
    add_test(NAME Cli_onnx_reconstruction
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_onnx_reconstruction.py"
            --cli $<TARGET_FILE:jyppx_cli_recipe_gate>
            --exporter $<TARGET_FILE:jyppx_test_onnx_export>
            ${onnx_reconstruction_arguments})
    set_tests_properties(Cli_onnx_reconstruction PROPERTIES
        LABELS "cli;deployment;onnx_export;onnx_runtime;integration" TIMEOUT 600)
endif()

if(JYPPX_LIUDING_TEST_CLI_CUDA)
    foreach(contract IN ITEMS model_checkpoint offline_workflow)
        add_test(NAME Cli_${contract}_CUDA
            COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_${contract}.py"
                --cli $<TARGET_FILE:jyppx_cli> --device cuda)
        set_tests_properties(Cli_${contract}_CUDA PROPERTIES LABELS "cli;integration;cuda" TIMEOUT 180)
    endforeach()
endif()

if(TARGET jyppx::liuding::models AND TARGET jyppx::liuding::data_image)
    add_test(NAME Cli_transfer_configuration
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_transfer_recipe.py"
            --cli $<TARGET_FILE:jyppx_cli> --contracts-only)
    set_tests_properties(Cli_transfer_configuration PROPERTIES
        LABELS "cli;integration;recipe" TIMEOUT 180)
    add_test(NAME Cli_transfer_recipe
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_transfer_recipe.py"
            --cli $<TARGET_FILE:jyppx_cli>)
    set_tests_properties(Cli_transfer_recipe PROPERTIES
        LABELS "cli;integration;recipe;external-weights" TIMEOUT 1200 SKIP_RETURN_CODE 77)
    add_test(NAME Cli_transfer_recipe_group_rates
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_transfer_recipe.py"
            --cli $<TARGET_FILE:jyppx_cli_recipe_gate> --group-rates)
    set_tests_properties(Cli_transfer_recipe_group_rates PROPERTIES
        LABELS "cli;integration;recipe;external-weights" TIMEOUT 1200 SKIP_RETURN_CODE 77)
    if(JYPPX_LIUDING_TEST_CLI_CUDA)
        add_test(NAME Cli_transfer_recipe_CUDA
            COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_transfer_recipe.py"
                --cli $<TARGET_FILE:jyppx_cli> --device cuda)
        set_tests_properties(Cli_transfer_recipe_CUDA PROPERTIES
            LABELS "cli;integration;recipe;external-weights;cuda" TIMEOUT 1200 SKIP_RETURN_CODE 77)
    endif()
    if(JYPPX_LIUDING_TEST_CLI_CUDA)
        add_test(NAME Cli_transfer_recipe_group_rates_CUDA
            COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_transfer_recipe.py"
                --cli $<TARGET_FILE:jyppx_cli_recipe_gate> --device cuda --group-rates)
        set_tests_properties(Cli_transfer_recipe_group_rates_CUDA PROPERTIES
            LABELS "cli;integration;recipe;external-weights;cuda" TIMEOUT 1200 SKIP_RETURN_CODE 77)
    endif()
    add_test(NAME Cli_image_recipe
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_image_recipe.py"
            --cli $<TARGET_FILE:jyppx_cli>)
    set_tests_properties(Cli_image_recipe PROPERTIES LABELS "cli;integration;recipe" TIMEOUT 600)
    add_test(NAME Cli_update_recipe
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_update_recipe.py"
            --cli $<TARGET_FILE:jyppx_cli_recipe_gate>)
    set_tests_properties(Cli_update_recipe PROPERTIES
        LABELS "cli;integration;recipe;scheduler" TIMEOUT 1200)
    add_test(NAME Cli_transfer_update_recipe
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_update_recipe.py"
            --cli $<TARGET_FILE:jyppx_cli_recipe_gate> --transfer-update)
    set_tests_properties(Cli_transfer_update_recipe PROPERTIES
        LABELS "cli;integration;recipe;scheduler;external-weights" TIMEOUT 1200 SKIP_RETURN_CODE 77)
    if(JYPPX_LIUDING_TEST_CLI_CUDA)
        add_test(NAME Cli_image_recipe_CUDA
            COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_image_recipe.py"
                --cli $<TARGET_FILE:jyppx_cli> --device cuda)
        set_tests_properties(Cli_image_recipe_CUDA PROPERTIES LABELS "cli;integration;recipe;cuda" TIMEOUT 600)
        add_test(NAME Cli_update_recipe_CUDA
            COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_update_recipe.py"
                --cli $<TARGET_FILE:jyppx_cli_recipe_gate> --device cuda)
        set_tests_properties(Cli_update_recipe_CUDA PROPERTIES
            LABELS "cli;integration;recipe;scheduler;cuda" TIMEOUT 1200)
        add_test(NAME Cli_transfer_update_recipe_CUDA
            COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_update_recipe.py"
                --cli $<TARGET_FILE:jyppx_cli_recipe_gate> --device cuda --transfer-update)
        set_tests_properties(Cli_transfer_update_recipe_CUDA PROPERTIES
            LABELS "cli;integration;recipe;scheduler;external-weights;cuda" TIMEOUT 1200 SKIP_RETURN_CODE 77)
    endif()
endif()

set(paired_preparation_example_arguments)
if(JYPPX_LIUDING_BUILD_EXAMPLES)
    list(APPEND paired_preparation_example_arguments
        --example $<TARGET_FILE:jyppx_paired_affine_data>)
endif()
add_test(NAME NativePairedAffineDataPreparation
    COMMAND ${Python3_EXECUTABLE}
        "${CMAKE_CURRENT_SOURCE_DIR}/test_paired_data_preparation.py"
        --cli $<TARGET_FILE:jyppx_cli_recipe_gate>
        ${paired_preparation_example_arguments})
set_tests_properties(NativePairedAffineDataPreparation PROPERTIES
    LABELS "data;recipe;cli;examples" TIMEOUT 180)

add_test(NAME Cli_paired_recipe
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_paired_recipe.py"
        --cli $<TARGET_FILE:jyppx_cli_recipe_gate>)
set_tests_properties(Cli_paired_recipe PROPERTIES LABELS "cli;integration;recipe" TIMEOUT 600)
if(JYPPX_LIUDING_TEST_CLI_CUDA)
    add_test(NAME Cli_paired_recipe_CUDA
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_paired_recipe.py"
            --cli $<TARGET_FILE:jyppx_cli_recipe_gate> --device cuda)
    set_tests_properties(Cli_paired_recipe_CUDA PROPERTIES LABELS "cli;integration;recipe;cuda" TIMEOUT 600)
endif()

get_filename_component(JYPPX_CLI_TORCH_RUNTIME "${TORCH_LIBRARY}" DIRECTORY)
add_test(NAME CliInstalledConsumer
    COMMAND ${Python3_EXECUTABLE} "${PROJECT_SOURCE_DIR}/tests/install/cli/run_acceptance.py"
        --cmake "${CMAKE_COMMAND}" --build-dir "${CMAKE_BINARY_DIR}" --config "$<CONFIG>"
        --prefix "${CMAKE_BINARY_DIR}/_cli_install_consumer/$<CONFIG>"
        --torch-runtime "${JYPPX_CLI_TORCH_RUNTIME}"
        --runtime-stager "${PROJECT_SOURCE_DIR}/cmake/StageTorchRuntime.cmake")
set_tests_properties(CliInstalledConsumer PROPERTIES LABELS "cli;packaging"
    RESOURCE_LOCK jyppx_install_consumer TIMEOUT 180)

if(JYPPX_LIUDING_PREPARE_FASHION_MNIST AND TARGET jyppx_test_fashion_mnist_harness)
    add_test(NAME NativeFashionMnistCliIdentity
        COMMAND "${Python3_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/tests/datasets/test_native_fashion_mnist.py"
            --harness $<TARGET_FILE:jyppx_test_fashion_mnist_harness>
            --cli $<TARGET_FILE:jyppx_cli_recipe_gate>
            NativeFashion.test_product_cli_rejects_synthetic_official_identity)
    set_tests_properties(NativeFashionMnistCliIdentity PROPERTIES
        LABELS "data;recipe;cli;native" TIMEOUT 180)
endif()

# Native fixed-weight converter; Python is only a process/reference checker.
if(TARGET jyppx_convert_dinov2_hf)
    jyppx_add_gtest(jyppx_test_dinov2_conversion
        LINK_TARGET jyppx_dinov2_conversion test_dinov2_conversion.cpp)
    if(Python3_Interpreter_FOUND)
        add_test(NAME NativeDinoV2ConversionCli
            COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_dinov2_converter.py"
                --binary $<TARGET_FILE:jyppx_convert_dinov2_hf>)
        set_tests_properties(NativeDinoV2ConversionCli PROPERTIES
            LABELS "tools;models;interop;integration" TIMEOUT 180)
        set(JYPPX_LIUDING_TEST_DINOV2_CONVERSION_SOURCE "" CACHE FILEPATH
            "Explicit pinned official DINOv2-S safetensors; no downloads")
        if(JYPPX_LIUDING_TEST_DINOV2_CONVERSION_SOURCE)
            if(NOT EXISTS "${JYPPX_LIUDING_TEST_DINOV2_CONVERSION_SOURCE}")
                message(FATAL_ERROR "Configured DINOv2 conversion source is missing")
            endif()
            add_test(NAME NativeDinoV2ConversionCanonical
                COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_dinov2_converter.py"
                    --binary $<TARGET_FILE:jyppx_convert_dinov2_hf>
                    --source "${JYPPX_LIUDING_TEST_DINOV2_CONVERSION_SOURCE}"
                    --reference "${PROJECT_SOURCE_DIR}/tests/unit/models/reference/convert_dinov2_hf_reference.py")
            set_tests_properties(NativeDinoV2ConversionCanonical PROPERTIES
                LABELS "tools;models;interop;external-weights;reference" TIMEOUT 600
                RUN_SERIAL TRUE)
        endif()
    endif()
endif()


if(TARGET jyppx::liuding::telemetry_native AND JYPPX_TELEMETRY_PYTHON)
    add_test(NAME Cli_native_tensorboard
        COMMAND "${JYPPX_TELEMETRY_PYTHON}"
            "${PROJECT_SOURCE_DIR}/tests/scripts/test_native_tensorboard.py"
            --cli $<TARGET_FILE:jyppx_cli_recipe_gate>
            --cli-fixture "${CMAKE_CURRENT_SOURCE_DIR}/test_run_telemetry.py"
            --sidecar "${PROJECT_SOURCE_DIR}/tools/telemetry_sidecar/jyppx_telemetry_sidecar.py")
    set_tests_properties(Cli_native_tensorboard PROPERTIES
        LABELS "cli;experiment;telemetry;tensorboard;official;integration" TIMEOUT 240)
    add_test(NAME Cli_native_tensorboard_worker_official
        COMMAND "${JYPPX_TELEMETRY_PYTHON}"
            "${CMAKE_CURRENT_SOURCE_DIR}/test_native_tensorboard_worker.py"
            --cli $<TARGET_FILE:jyppx_cli_recipe_gate> --official
            --sidecar "${PROJECT_SOURCE_DIR}/tools/telemetry_sidecar/jyppx_telemetry_sidecar.py")
    set_tests_properties(Cli_native_tensorboard_worker_official PROPERTIES
        LABELS "cli;telemetry_native;tensorboard;official;integration" TIMEOUT 120)
endif()
