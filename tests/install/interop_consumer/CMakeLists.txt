cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(jyppx_liuding_interop_consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(jyppx CONFIG REQUIRED COMPONENTS interop)

if(NOT TARGET jyppx::liuding::interop)
    message(FATAL_ERROR "Installed package does not export jyppx::liuding::interop")
endif()
if(NOT jyppx_interop_FOUND)
    message(FATAL_ERROR "Installed package did not satisfy component interop")
endif()
if(TARGET jyppx_liuding_training_framework::jyppx_liuding_interop)
    message(FATAL_ERROR "Installed package exports a legacy interop target")
endif()

add_executable(jyppx_liuding_interop_consumer main.cpp)
target_link_libraries(jyppx_liuding_interop_consumer PRIVATE jyppx::liuding::interop)

if(WIN32)
    if(NOT JYPPX_TORCH_RUNTIME_DIR)
        message(FATAL_ERROR "Windows consumer requires JYPPX_TORCH_RUNTIME_DIR")
    endif()
    file(GLOB JYPPX_CONSUMER_RUNTIME_DLLS
        "${JYPPX_TORCH_RUNTIME_DIR}/*.dll"
    )
    add_custom_command(TARGET jyppx_liuding_interop_consumer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${JYPPX_CONSUMER_RUNTIME_DLLS}
            "$<TARGET_FILE_DIR:jyppx_liuding_interop_consumer>"
        VERBATIM
    )
endif()

enable_testing()
add_test(NAME Runs COMMAND jyppx_liuding_interop_consumer)
