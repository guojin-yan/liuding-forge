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
    add_test(NAME MnistDataPreparation
        COMMAND ${Python3_EXECUTABLE}
            "${PROJECT_SOURCE_DIR}/tests/scripts/test_prepare_mnist.py")
    add_test(NAME MnistExampleOffline
        COMMAND ${Python3_EXECUTABLE}
            "${PROJECT_SOURCE_DIR}/tests/scripts/test_mnist_example.py"
            --binary $<TARGET_FILE:jyppx_mnist>)
    set_tests_properties(MnistDataPreparation MnistExampleOffline PROPERTIES
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

if(TARGET jyppx::liuding::serving)
    add_executable(jyppx_liuding_serving_jsonl serving/jsonl_host.cpp)
    set_target_properties(jyppx_liuding_serving_jsonl PROPERTIES
        OUTPUT_NAME liuding-forge-example-serving-jsonl)
    target_link_libraries(jyppx_liuding_serving_jsonl PRIVATE
        jyppx::liuding::serving nlohmann_json::nlohmann_json)
    jyppx_apply_instrumentation(jyppx_liuding_serving_jsonl)
    jyppx_add_torch_runtime_dependency(jyppx_liuding_serving_jsonl)
endif()
