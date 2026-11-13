cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(liuding_data_examples LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    find_package(jyppx CONFIG REQUIRED COMPONENTS data interop)
    include(CTest)
endif()

add_executable(jyppx_basic_data_pipeline basic_data_pipeline.cpp)
set_target_properties(jyppx_basic_data_pipeline PROPERTIES
    OUTPUT_NAME liuding-forge-example-basic-data-pipeline)
target_link_libraries(jyppx_basic_data_pipeline PRIVATE
    jyppx::liuding::data jyppx::liuding::interop)
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # Preserve the executable location used before grouping data examples.
    set_target_properties(jyppx_basic_data_pipeline PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/..")
endif()
if(COMMAND jyppx_apply_instrumentation)
    jyppx_apply_instrumentation(jyppx_basic_data_pipeline)
endif()
if(COMMAND jyppx_add_torch_runtime_dependency)
    jyppx_add_torch_runtime_dependency(jyppx_basic_data_pipeline)
endif()

if(JYPPX_LIUDING_BUILD_TESTS OR BUILD_TESTING)
    foreach(topology IN ITEMS inline workers)
        set(state_path "${CMAKE_CURRENT_BINARY_DIR}/pipeline_state/${topology}.bin")
        add_test(NAME DataPipeline-${topology}-Example
            COMMAND jyppx_basic_data_pipeline ${topology})
        add_test(NAME DataPipeline-${topology}-Save-Example
            COMMAND jyppx_basic_data_pipeline save "${state_path}" ${topology})
        set_tests_properties(DataPipeline-${topology}-Save-Example PROPERTIES
            FIXTURES_SETUP data_pipeline_${topology})
        foreach(mode IN ITEMS restore failure)
            add_test(NAME DataPipeline-${topology}-${mode}-Example
                COMMAND jyppx_basic_data_pipeline ${mode} "${state_path}" ${topology})
            set_tests_properties(DataPipeline-${topology}-${mode}-Example PROPERTIES
                FIXTURES_REQUIRED data_pipeline_${topology})
        endforeach()
        set_tests_properties(DataPipeline-${topology}-Example
            DataPipeline-${topology}-Save-Example
            DataPipeline-${topology}-restore-Example
            DataPipeline-${topology}-failure-Example PROPERTIES
            LABELS "examples;data;interop" TIMEOUT 30)
    endforeach()
endif()
