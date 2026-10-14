cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(jyppx_foundation_consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(jyppx_training_framework CONFIG REQUIRED)

if(NOT TARGET jyppx::foundation)
    message(FATAL_ERROR "Installed package does not export jyppx::foundation")
endif()
if(TARGET jyppx_training_framework::jyppx_foundation OR
   TARGET jyppx_training_framework::jyppx_training_framework)
    message(FATAL_ERROR "Installed package still exports a legacy target name")
endif()

add_executable(jyppx_foundation_consumer main.cpp)
target_link_libraries(jyppx_foundation_consumer PRIVATE jyppx::foundation)

if(WIN32)
    if(NOT JYPPX_TORCH_RUNTIME_DIR)
        message(FATAL_ERROR "Windows consumer smoke requires JYPPX_TORCH_RUNTIME_DIR")
    endif()
    file(GLOB JYPPX_CONSUMER_RUNTIME_DLLS
        "${JYPPX_TORCH_RUNTIME_DIR}/*.dll"
    )
    add_custom_command(TARGET jyppx_foundation_consumer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${JYPPX_CONSUMER_RUNTIME_DLLS}
            "$<TARGET_FILE_DIR:jyppx_foundation_consumer>"
        VERBATIM
    )
endif()

enable_testing()
add_test(NAME Runs COMMAND jyppx_foundation_consumer)
