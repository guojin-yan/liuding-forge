cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(jyppx_serving_consumer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(jyppx CONFIG REQUIRED COMPONENTS serving)
if(NOT TARGET jyppx::serving OR NOT jyppx_serving_FOUND)
    message(FATAL_ERROR "Installed package does not satisfy jyppx::serving")
endif()
add_executable(jyppx_serving_consumer main.cpp)
target_link_libraries(jyppx_serving_consumer PRIVATE jyppx::serving)
if(WIN32)
    if(NOT JYPPX_TORCH_RUNTIME_DIR)
        message(FATAL_ERROR "Windows consumer requires JYPPX_TORCH_RUNTIME_DIR")
    endif()
    if(JYPPX_RUNTIME_STAGER AND EXISTS "${JYPPX_RUNTIME_STAGER}")
        add_custom_command(TARGET jyppx_serving_consumer POST_BUILD
            COMMAND ${CMAKE_COMMAND}
                "-DJYPPX_RUNTIME_SOURCE_DIR=${JYPPX_TORCH_RUNTIME_DIR}"
                "-DJYPPX_RUNTIME_OUTPUT_DIR=$<TARGET_FILE_DIR:jyppx_serving_consumer>"
                -P "${JYPPX_RUNTIME_STAGER}" VERBATIM)
    else()
        file(GLOB JYPPX_CONSUMER_RUNTIME_DLLS "${JYPPX_TORCH_RUNTIME_DIR}/*.dll")
        add_custom_command(TARGET jyppx_serving_consumer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${JYPPX_CONSUMER_RUNTIME_DLLS}
                "$<TARGET_FILE_DIR:jyppx_serving_consumer>" VERBATIM)
    endif()
endif()
enable_testing()
add_test(NAME Runs COMMAND jyppx_serving_consumer)
