jyppx_add_gtest(jyppx_test_sidecar_transport
    LINK_TARGET jyppx_cli_transport
    test_sidecar_transport.cpp)
target_link_libraries(jyppx_test_sidecar_transport PRIVATE nlohmann_json::nlohmann_json)
target_compile_definitions(jyppx_test_sidecar_transport PRIVATE
    JYPPX_SIDECAR_TEST_PYTHON="${Python3_EXECUTABLE}"
    JYPPX_SIDECAR_TEST_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/sidecar_fixture.py")

jyppx_add_gtest(jyppx_test_local_process
    LINK_TARGET jyppx_cli_transport
    test_local_process.cpp)
target_link_libraries(jyppx_test_local_process PRIVATE nlohmann_json::nlohmann_json)
target_compile_definitions(jyppx_test_local_process PRIVATE
    JYPPX_LOCAL_PROCESS_PYTHON="${Python3_EXECUTABLE}"
    JYPPX_LOCAL_PROCESS_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/local_process_fixture.py")

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
    if(JYPPX_LIUDING_TEST_CLI_CUDA)
        add_test(NAME Cli_transfer_recipe_CUDA
            COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_transfer_recipe.py"
                --cli $<TARGET_FILE:jyppx_cli> --device cuda)
        set_tests_properties(Cli_transfer_recipe_CUDA PROPERTIES
            LABELS "cli;integration;recipe;external-weights;cuda" TIMEOUT 1200 SKIP_RETURN_CODE 77)
    endif()
    add_test(NAME Cli_image_recipe
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_image_recipe.py"
            --cli $<TARGET_FILE:jyppx_cli>)
    set_tests_properties(Cli_image_recipe PROPERTIES LABELS "cli;integration;recipe" TIMEOUT 600)
    if(JYPPX_LIUDING_TEST_CLI_CUDA)
        add_test(NAME Cli_image_recipe_CUDA
            COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_image_recipe.py"
                --cli $<TARGET_FILE:jyppx_cli> --device cuda)
        set_tests_properties(Cli_image_recipe_CUDA PROPERTIES LABELS "cli;integration;recipe;cuda" TIMEOUT 600)
    endif()
endif()

add_test(NAME Cli_paired_recipe
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_paired_recipe.py"
        --cli $<TARGET_FILE:jyppx_cli>)
set_tests_properties(Cli_paired_recipe PROPERTIES LABELS "cli;integration;recipe" TIMEOUT 600)
if(JYPPX_LIUDING_TEST_CLI_CUDA)
    add_test(NAME Cli_paired_recipe_CUDA
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_paired_recipe.py"
            --cli $<TARGET_FILE:jyppx_cli> --device cuda)
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
