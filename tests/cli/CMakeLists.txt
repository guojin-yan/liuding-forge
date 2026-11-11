file(TO_CMAKE_PATH "${Python3_EXECUTABLE}" jyppx_cli_test_python)
# A Windows venv redirector starts another process; direct-process accounting
# fixtures must run in the interpreter whose process handle is measured.
set(jyppx_local_process_test_python "${jyppx_cli_test_python}")
if(WIN32)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -I -B -c
            "import sys; print(getattr(sys, '_base_executable', sys.executable))"
        RESULT_VARIABLE jyppx_local_process_python_result
        OUTPUT_VARIABLE jyppx_local_process_python_path
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT jyppx_local_process_python_result EQUAL 0 OR
       NOT IS_ABSOLUTE "${jyppx_local_process_python_path}" OR
       NOT EXISTS "${jyppx_local_process_python_path}")
        message(FATAL_ERROR "Direct Python interpreter unavailable for process accounting tests")
    endif()
    file(TO_CMAKE_PATH "${jyppx_local_process_python_path}" jyppx_local_process_test_python)
endif()

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
    JYPPX_LOCAL_PROCESS_PYTHON="${jyppx_local_process_test_python}"
    JYPPX_LOCAL_PROCESS_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/local_process_fixture.py")

foreach(group IN ITEMS core resources environment runtime_files host)
    if(group STREQUAL "core")
        set(local_attempt_test Cli_local_attempt)
        set(local_attempt_timeout 240)
    else()
        set(local_attempt_test Cli_local_attempt_${group})
        set(local_attempt_timeout 120)
    endif()
    add_test(NAME ${local_attempt_test}
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_local_attempt.py"
            --cli $<TARGET_FILE:jyppx_cli_recipe_gate> --contract-group ${group} -v)
    set_tests_properties(${local_attempt_test} PROPERTIES
        LABELS "cli;experiment;integration" TIMEOUT ${local_attempt_timeout})
endforeach()
add_test(NAME Cli_local_cuda_schema
    COMMAND ${Python3_EXECUTABLE} -I -B "${CMAKE_CURRENT_SOURCE_DIR}/test_local_cuda.py"
        --cli $<TARGET_FILE:jyppx_cli_recipe_gate> -v)
set_tests_properties(Cli_local_cuda_schema PROPERTIES LABELS "cli;experiment;integration" TIMEOUT 120)
if(JYPPX_LIUDING_TEST_CLI_CUDA AND JYPPX_ACCELERATOR_BUILD STREQUAL "cuda")
    add_executable(jyppx_local_cuda_fixture local_cuda_fixture.cpp)
    target_include_directories(jyppx_local_cuda_fixture PRIVATE ${CUDA_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(jyppx_local_cuda_fixture PRIVATE torch nlohmann_json::nlohmann_json "${CUDA_CUDART_LIBRARY}")
    jyppx_add_torch_runtime_dependency(jyppx_local_cuda_fixture)
    add_test(NAME Cli_local_cuda_placement
        COMMAND ${Python3_EXECUTABLE} -I -B "${CMAKE_CURRENT_SOURCE_DIR}/test_local_cuda.py"
            --cli $<TARGET_FILE:jyppx_cli_recipe_gate> --fixture $<TARGET_FILE:jyppx_local_cuda_fixture> --hardware -v)
    set_tests_properties(Cli_local_cuda_placement PROPERTIES LABELS "cli;experiment;cuda;integration" TIMEOUT 180)
endif()
add_test(NAME CliLocalAttemptRegistration
    COMMAND ${Python3_EXECUTABLE} -I -B
        "${PROJECT_SOURCE_DIR}/tests/scripts/test_local_attempt_registration.py")
set_tests_properties(CliLocalAttemptRegistration PROPERTIES
    LABELS "cli;hygiene" TIMEOUT 60)
add_test(NAME Cli_local_search
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_local_search.py"
        --cli $<TARGET_FILE:jyppx_cli_recipe_gate>)
set_tests_properties(Cli_local_search PROPERTIES
    LABELS "cli;experiment;integration" TIMEOUT 600)

if(WIN32 OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(paired_memory_devices cpu)
    if(JYPPX_LIUDING_TEST_CLI_CUDA)
        list(APPEND paired_memory_devices cuda)
    endif()
    foreach(device IN LISTS paired_memory_devices)
        add_test(NAME Cli_local_paired_memory_${device}
            COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_local_paired_memory.py"
                --cli $<TARGET_FILE:jyppx_cli_recipe_gate> --device ${device}
                --source-root "${PROJECT_SOURCE_DIR}")
        set_tests_properties(Cli_local_paired_memory_${device} PROPERTIES
            LABELS "cli;experiment;recipe;memory;${device}" TIMEOUT 900
            RUN_SERIAL TRUE SKIP_RETURN_CODE 77)
    endforeach()
endif()

set(local_workspace_recipes paired)
if(TARGET jyppx::liuding::models AND TARGET jyppx::liuding::data_image)
    list(APPEND local_workspace_recipes image)
endif()
foreach(recipe IN LISTS local_workspace_recipes)
    add_test(NAME Cli_local_recipe_workspace_${recipe}
        COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/test_local_recipe_workspace.py"
            --cli $<TARGET_FILE:jyppx_cli_recipe_gate>
            --source-root "${PROJECT_SOURCE_DIR}" --recipe ${recipe})
    set_tests_properties(Cli_local_recipe_workspace_${recipe} PROPERTIES
        LABELS "cli;experiment;recipe;integration" TIMEOUT 1000 RUN_SERIAL TRUE)
endforeach()

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
    add_test(NAME Cli_local_recipe
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_local_recipe.py"
            --cli $<TARGET_FILE:jyppx_cli_recipe_gate>)
    set_tests_properties(Cli_local_recipe PROPERTIES
        LABELS "cli;integration;recipe;local" TIMEOUT 1200)
    add_test(NAME Cli_local_recipe_lifecycle
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_local_recipe_lifecycle.py"
            --cli $<TARGET_FILE:jyppx_cli_recipe_gate>)
    set_tests_properties(Cli_local_recipe_lifecycle PROPERTIES
        LABELS "cli;integration;recipe;local" TIMEOUT 360)
    add_test(NAME Cli_local_recipe_failure_lifecycle
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_local_recipe_failure_lifecycle.py"
            --cli $<TARGET_FILE:jyppx_cli_recipe_gate>)
    set_tests_properties(Cli_local_recipe_failure_lifecycle PROPERTIES
        LABELS "cli;integration;recipe;local" TIMEOUT 420 RUN_SERIAL TRUE)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_test(NAME Cli_local_recipe_memory
            COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_local_recipe_memory.py"
                --cli $<TARGET_FILE:jyppx_cli_recipe_gate>)
        set_tests_properties(Cli_local_recipe_memory PROPERTIES
            LABELS "cli;integration;recipe;local;memory" TIMEOUT 1500 RUN_SERIAL TRUE
            SKIP_RETURN_CODE 77)
        add_test(NAME Cli_local_recipe_parent_death
            COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_local_recipe_parent_death.py"
                --cli $<TARGET_FILE:jyppx_cli_recipe_gate>)
        set_tests_properties(Cli_local_recipe_parent_death PROPERTIES
            LABELS "cli;integration;recipe;local" TIMEOUT 180 RUN_SERIAL TRUE)
    endif()
    if(JYPPX_LIUDING_TEST_CLI_CUDA)
        add_test(NAME Cli_local_recipe_CUDA
            COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_local_recipe.py"
                --cli $<TARGET_FILE:jyppx_cli_recipe_gate> --device cuda)
        set_tests_properties(Cli_local_recipe_CUDA PROPERTIES
            LABELS "cli;integration;recipe;local;cuda" TIMEOUT 1200)
        add_test(NAME Cli_local_recipe_lifecycle_CUDA
            COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_local_recipe_lifecycle.py"
                --cli $<TARGET_FILE:jyppx_cli_recipe_gate> --device cuda)
        set_tests_properties(Cli_local_recipe_lifecycle_CUDA PROPERTIES
            LABELS "cli;integration;recipe;local;cuda" TIMEOUT 360)
        add_test(NAME Cli_local_recipe_failure_lifecycle_CUDA
            COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_local_recipe_failure_lifecycle.py"
                --cli $<TARGET_FILE:jyppx_cli_recipe_gate> --device cuda)
        set_tests_properties(Cli_local_recipe_failure_lifecycle_CUDA PROPERTIES
            LABELS "cli;integration;recipe;local;cuda" TIMEOUT 420 RUN_SERIAL TRUE)
        if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
            add_test(NAME Cli_local_recipe_parent_death_CUDA
                COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_local_recipe_parent_death.py"
                    --cli $<TARGET_FILE:jyppx_cli_recipe_gate> --device cuda)
            set_tests_properties(Cli_local_recipe_parent_death_CUDA PROPERTIES
                LABELS "cli;integration;recipe;local;cuda" TIMEOUT 180 RUN_SERIAL TRUE)
        endif()
    endif()
    add_test(NAME Cli_initial_state_recipe
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_initial_state_recipe.py"
            --cli $<TARGET_FILE:jyppx_cli_recipe_gate>)
    set_tests_properties(Cli_initial_state_recipe PROPERTIES
        LABELS "cli;integration;recipe;initialization" TIMEOUT 600)
    if(JYPPX_LIUDING_TEST_CLI_CUDA)
        add_test(NAME Cli_initial_state_recipe_CUDA
            COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_initial_state_recipe.py"
                --cli $<TARGET_FILE:jyppx_cli_recipe_gate> --device cuda)
        set_tests_properties(Cli_initial_state_recipe_CUDA PROPERTIES
            LABELS "cli;integration;recipe;initialization;cuda" TIMEOUT 600)
    endif()
    add_test(NAME Cli_transfer_configuration
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_transfer_recipe.py"
            --cli $<TARGET_FILE:jyppx_cli> --contracts-only)
    set_tests_properties(Cli_transfer_configuration PROPERTIES
        LABELS "cli;integration;recipe" TIMEOUT 180)
    add_test(NAME Cli_local_search_transfer_configuration
        COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/test_local_search_transfer_recipe.py"
            --cli $<TARGET_FILE:jyppx_cli_recipe_gate> --task contracts)
    set_tests_properties(Cli_local_search_transfer_configuration PROPERTIES
        LABELS "cli;integration;recipe;search;transfer" TIMEOUT 300 RUN_SERIAL TRUE)
    set(JYPPX_LIUDING_TEST_TRANSFER_SEARCH_WEIGHTS "" CACHE FILEPATH
        "Explicit pinned ResNet50 weights for the four-update transfer search task; no downloads")
    if(JYPPX_LIUDING_TEST_TRANSFER_SEARCH_WEIGHTS)
        if(NOT EXISTS "${JYPPX_LIUDING_TEST_TRANSFER_SEARCH_WEIGHTS}")
            message(FATAL_ERROR "Configured transfer search weights are missing")
        endif()
        add_test(NAME Cli_local_search_transfer_recipe
            COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/test_local_search_transfer_recipe.py"
                --cli $<TARGET_FILE:jyppx_cli_recipe_gate> --task fixture
                --weights-file "${JYPPX_LIUDING_TEST_TRANSFER_SEARCH_WEIGHTS}")
        set_tests_properties(Cli_local_search_transfer_recipe PROPERTIES
            LABELS "cli;integration;recipe;search;transfer;external-weights"
            TIMEOUT 1200 RUN_SERIAL TRUE)
    endif()
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
    add_test(NAME Cli_adamw_recipe
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_adamw_recipe.py"
            --cli $<TARGET_FILE:jyppx_cli_recipe_gate>)
    set_tests_properties(Cli_adamw_recipe PROPERTIES
        LABELS "cli;integration;recipe;optimizer;scheduler" TIMEOUT 1200)
    add_test(NAME Cli_transfer_adamw_recipe
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_adamw_recipe.py"
            --cli $<TARGET_FILE:jyppx_cli_recipe_gate> --transfer)
    set_tests_properties(Cli_transfer_adamw_recipe PROPERTIES
        LABELS "cli;integration;recipe;optimizer;scheduler;pretrained"
        TIMEOUT 1200 SKIP_RETURN_CODE 77)
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
        add_test(NAME Cli_adamw_recipe_CUDA
            COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_adamw_recipe.py"
                --cli $<TARGET_FILE:jyppx_cli_recipe_gate> --device cuda)
        set_tests_properties(Cli_adamw_recipe_CUDA PROPERTIES
            LABELS "cli;integration;recipe;optimizer;scheduler;cuda" TIMEOUT 1200)
        add_test(NAME Cli_transfer_adamw_recipe_CUDA
            COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_adamw_recipe.py"
                --cli $<TARGET_FILE:jyppx_cli_recipe_gate> --transfer --device cuda)
        set_tests_properties(Cli_transfer_adamw_recipe_CUDA PROPERTIES
            LABELS "cli;integration;recipe;optimizer;scheduler;pretrained;cuda"
            TIMEOUT 1200 SKIP_RETURN_CODE 77)
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
            --sidecar "${PROJECT_SOURCE_DIR}/tests/reference/telemetry/official_mapping.py")
    set_tests_properties(Cli_native_tensorboard PROPERTIES
        LABELS "cli;experiment;telemetry;tensorboard;official;integration" TIMEOUT 240)
    add_test(NAME Cli_native_tensorboard_worker_official
        COMMAND "${JYPPX_TELEMETRY_PYTHON}"
            "${CMAKE_CURRENT_SOURCE_DIR}/test_native_tensorboard_worker.py"
            --cli $<TARGET_FILE:jyppx_cli_recipe_gate> --official
            --sidecar "${PROJECT_SOURCE_DIR}/tests/reference/telemetry/official_mapping.py")
    set_tests_properties(Cli_native_tensorboard_worker_official PROPERTIES
        LABELS "cli;telemetry_native;tensorboard;official;integration" TIMEOUT 120)
endif()
