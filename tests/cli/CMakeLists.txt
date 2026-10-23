jyppx_add_gtest(jyppx_test_sidecar_transport
    LINK_TARGET jyppx_cli_transport
    test_sidecar_transport.cpp)
target_link_libraries(jyppx_test_sidecar_transport PRIVATE nlohmann_json::nlohmann_json)
target_compile_definitions(jyppx_test_sidecar_transport PRIVATE
    JYPPX_SIDECAR_TEST_PYTHON="${Python3_EXECUTABLE}"
    JYPPX_SIDECAR_TEST_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/sidecar_fixture.py")

foreach(contract IN ITEMS model_checkpoint model_bundle evaluation_data offline_workflow run_telemetry migration)
    add_test(NAME Cli_${contract}
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_${contract}.py"
            --cli $<TARGET_FILE:jyppx_cli>)
    set_tests_properties(Cli_${contract} PROPERTIES LABELS "cli;integration" TIMEOUT 180)
endforeach()

if(JYPPX_LIUDING_TEST_CLI_CUDA)
    foreach(contract IN ITEMS model_checkpoint offline_workflow)
        add_test(NAME Cli_${contract}_CUDA
            COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_${contract}.py"
                --cli $<TARGET_FILE:jyppx_cli> --device cuda)
        set_tests_properties(Cli_${contract}_CUDA PROPERTIES LABELS "cli;integration;cuda" TIMEOUT 180)
    endforeach()
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
