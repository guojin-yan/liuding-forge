cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(jyppx_experiment_consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(jyppx CONFIG REQUIRED COMPONENTS experiment)

if(NOT TARGET jyppx::experiment)
    message(FATAL_ERROR "Installed package does not export jyppx::experiment")
endif()
if(NOT jyppx_experiment_FOUND)
    message(FATAL_ERROR "Installed package did not satisfy component experiment")
endif()
if(TARGET jyppx_training_framework::jyppx_experiment)
    message(FATAL_ERROR "Installed package exports a legacy experiment target")
endif()

add_executable(jyppx_experiment_consumer main.cpp)
target_link_libraries(jyppx_experiment_consumer PRIVATE jyppx::experiment)

if(WIN32)
    if(NOT JYPPX_TORCH_RUNTIME_DIR)
        message(FATAL_ERROR "Windows consumer requires JYPPX_TORCH_RUNTIME_DIR")
    endif()
    file(GLOB JYPPX_CONSUMER_RUNTIME_DLLS "${JYPPX_TORCH_RUNTIME_DIR}/*.dll")
    add_custom_command(TARGET jyppx_experiment_consumer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${JYPPX_CONSUMER_RUNTIME_DLLS}
            "$<TARGET_FILE_DIR:jyppx_experiment_consumer>"
        VERBATIM
    )
endif()

enable_testing()
add_test(NAME Runs COMMAND jyppx_experiment_consumer)
