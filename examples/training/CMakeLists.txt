cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# The same source also builds directly against an installed SDK.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(liuding_training_examples LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    find_package(jyppx CONFIG REQUIRED COMPONENTS training interop)
    include(CTest)
endif()

add_executable(jyppx_basic_trainer basic_trainer.cpp)
set_target_properties(jyppx_basic_trainer PROPERTIES
    OUTPUT_NAME liuding-forge-example-basic-trainer)
target_link_libraries(jyppx_basic_trainer PRIVATE jyppx::liuding::training)
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set_target_properties(jyppx_basic_trainer PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/..")
endif()
if(COMMAND jyppx_apply_instrumentation)
    jyppx_apply_instrumentation(jyppx_basic_trainer)
endif()
if(COMMAND jyppx_add_torch_runtime_dependency)
    jyppx_add_torch_runtime_dependency(jyppx_basic_trainer)
endif()

add_executable(jyppx_optimizer_scheduling optimizer_scheduling.cpp)
set_target_properties(jyppx_optimizer_scheduling PROPERTIES
    OUTPUT_NAME liuding-forge-example-optimizer-scheduling)
target_link_libraries(jyppx_optimizer_scheduling PRIVATE
    jyppx::liuding::training jyppx::liuding::interop)
if(COMMAND jyppx_apply_instrumentation)
    jyppx_apply_instrumentation(jyppx_optimizer_scheduling)
endif()
if(COMMAND jyppx_add_torch_runtime_dependency)
    jyppx_add_torch_runtime_dependency(jyppx_optimizer_scheduling)
endif()

add_executable(jyppx_checkpoint_state checkpoint_state.cpp)
set_target_properties(jyppx_checkpoint_state PROPERTIES
    OUTPUT_NAME liuding-forge-example-checkpoint-state)
target_link_libraries(jyppx_checkpoint_state PRIVATE
    jyppx::liuding::training jyppx::liuding::interop)
if(COMMAND jyppx_apply_instrumentation)
    jyppx_apply_instrumentation(jyppx_checkpoint_state)
endif()
if(COMMAND jyppx_add_torch_runtime_dependency)
    jyppx_add_torch_runtime_dependency(jyppx_checkpoint_state)
endif()

add_executable(jyppx_grad_scaler grad_scaler.cpp)
set_target_properties(jyppx_grad_scaler PROPERTIES
    OUTPUT_NAME liuding-forge-example-grad-scaler)
target_link_libraries(jyppx_grad_scaler PRIVATE jyppx::liuding::training)
if(COMMAND jyppx_apply_instrumentation)
    jyppx_apply_instrumentation(jyppx_grad_scaler)
endif()
if(COMMAND jyppx_add_torch_runtime_dependency)
    jyppx_add_torch_runtime_dependency(jyppx_grad_scaler)
endif()

if(JYPPX_LIUDING_BUILD_TESTS OR BUILD_TESTING)
    add_test(NAME BasicTrainer-Example COMMAND jyppx_basic_trainer)
    set_tests_properties(BasicTrainer-Example PROPERTIES
        LABELS "examples;training;optim" TIMEOUT 30)
    foreach(mode IN ITEMS epoch warmup plateau)
        add_test(NAME OptimizerScheduling-${mode}-Example
            COMMAND jyppx_optimizer_scheduling ${mode})
        set_tests_properties(OptimizerScheduling-${mode}-Example PROPERTIES
            LABELS "examples;training;optim" TIMEOUT 30)
    endforeach()
    set(optimizer_state_path
        "${CMAKE_CURRENT_BINARY_DIR}/optimizer_scheduling/state.bin")
    add_test(NAME OptimizerScheduling-save-Example
        COMMAND jyppx_optimizer_scheduling save "${optimizer_state_path}")
    set_tests_properties(OptimizerScheduling-save-Example PROPERTIES
        FIXTURES_SETUP optimizer_scheduling LABELS "examples;training;optim"
        TIMEOUT 30)
    add_test(NAME OptimizerScheduling-restore-Example
        COMMAND jyppx_optimizer_scheduling restore "${optimizer_state_path}")
    set_tests_properties(OptimizerScheduling-restore-Example PROPERTIES
        FIXTURES_REQUIRED optimizer_scheduling LABELS "examples;training;optim"
        TIMEOUT 30)

    foreach(mode IN ITEMS single overflow group group-overflow)
        add_test(NAME GradScaler-${mode}-Example
            COMMAND jyppx_grad_scaler ${mode})
        set_tests_properties(GradScaler-${mode}-Example PROPERTIES
            LABELS "examples;training;optim" TIMEOUT 30)
    endforeach()
    set(checkpoint_example_path
        "${CMAKE_CURRENT_BINARY_DIR}/checkpoint_state_example/state.bin")
    add_test(NAME CheckpointStateSaveExample COMMAND jyppx_checkpoint_state
        save "${checkpoint_example_path}")
    set_tests_properties(CheckpointStateSaveExample PROPERTIES
        FIXTURES_SETUP checkpoint_state_example)
    foreach(mode IN ITEMS restore rollback budget)
        add_test(NAME CheckpointState-${mode}-Example
            COMMAND jyppx_checkpoint_state ${mode} "${checkpoint_example_path}")
        set_tests_properties(CheckpointState-${mode}-Example PROPERTIES
            FIXTURES_REQUIRED checkpoint_state_example)
    endforeach()
    set_tests_properties(CheckpointStateSaveExample
        CheckpointState-restore-Example CheckpointState-rollback-Example
        CheckpointState-budget-Example PROPERTIES
        LABELS "examples;training;interop" TIMEOUT 30)
endif()
