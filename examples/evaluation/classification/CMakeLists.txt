cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(liuding_classification_evaluation_example LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    find_package(jyppx CONFIG REQUIRED COMPONENTS training interop experiment)
    include(CTest)
endif()

add_executable(jyppx_classification_evaluation classification.cpp)
set_target_properties(jyppx_classification_evaluation PROPERTIES
    OUTPUT_NAME liuding-forge-example-classification-evaluation)
target_link_libraries(jyppx_classification_evaluation PRIVATE
    jyppx::liuding::training jyppx::liuding::interop jyppx::liuding::experiment)
if(COMMAND jyppx_apply_instrumentation)
    jyppx_apply_instrumentation(jyppx_classification_evaluation)
endif()
if(COMMAND jyppx_add_torch_runtime_dependency)
    jyppx_add_torch_runtime_dependency(jyppx_classification_evaluation)
endif()

if(JYPPX_LIUDING_BUILD_TESTS OR BUILD_TESTING)
    set(evaluation_example_root "${CMAKE_CURRENT_BINARY_DIR}/classification-output")
    foreach(mode IN ITEMS full pause resume cancel reject-identity)
        if(mode STREQUAL "full")
            set(output_name full)
        elseif(mode STREQUAL "cancel")
            set(output_name cancelled)
        else()
            set(output_name paused)
        endif()
        add_test(NAME ClassificationEvaluation-${mode}-Example
            COMMAND jyppx_classification_evaluation ${mode}
                "${evaluation_example_root}/${output_name}")
        set_tests_properties(ClassificationEvaluation-${mode}-Example PROPERTIES
            LABELS "examples;training;experiment;interop" TIMEOUT 30)
    endforeach()
    set_tests_properties(ClassificationEvaluation-pause-Example PROPERTIES
        FIXTURES_SETUP classification_evaluation_pause)
    set_tests_properties(ClassificationEvaluation-reject-identity-Example PROPERTIES
        FIXTURES_REQUIRED classification_evaluation_pause)
    set_tests_properties(ClassificationEvaluation-resume-Example PROPERTIES
        FIXTURES_REQUIRED classification_evaluation_pause
        FIXTURES_SETUP classification_evaluation_finished)
    set_tests_properties(ClassificationEvaluation-cancel-Example PROPERTIES
        FIXTURES_SETUP classification_evaluation_cancelled)
    add_test(NAME ClassificationEvaluation-replay-Example
        COMMAND jyppx_classification_evaluation resume "${evaluation_example_root}/paused")
    set_tests_properties(ClassificationEvaluation-replay-Example PROPERTIES
        FIXTURES_REQUIRED classification_evaluation_finished
        LABELS "examples;training;experiment;interop" TIMEOUT 30)
    add_test(NAME ClassificationEvaluation-cancel-resume-Example
        COMMAND jyppx_classification_evaluation resume "${evaluation_example_root}/cancelled")
    set_tests_properties(ClassificationEvaluation-cancel-resume-Example PROPERTIES
        FIXTURES_REQUIRED classification_evaluation_cancelled
        FIXTURES_SETUP classification_evaluation_cancel_finished
        LABELS "examples;training;experiment;interop" TIMEOUT 30)
    add_test(NAME ClassificationEvaluation-cancel-replay-Example
        COMMAND jyppx_classification_evaluation resume "${evaluation_example_root}/cancelled")
    set_tests_properties(ClassificationEvaluation-cancel-replay-Example PROPERTIES
        FIXTURES_REQUIRED classification_evaluation_cancel_finished
        LABELS "examples;training;experiment;interop" TIMEOUT 30)
endif()
