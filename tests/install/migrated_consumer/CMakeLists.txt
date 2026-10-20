cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(jyppx_migrated_api_consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(jyppx CONFIG REQUIRED COMPONENTS
    foundation core interop data nn training optim experiment sdk
)

foreach(JYPPX_REQUIRED_COMPONENT IN ITEMS
        foundation core interop data nn training optim experiment sdk)
    if(NOT jyppx_${JYPPX_REQUIRED_COMPONENT}_FOUND)
        message(FATAL_ERROR
            "Installed package did not satisfy component ${JYPPX_REQUIRED_COMPONENT}")
    endif()
endforeach()

if(NOT TARGET jyppx::foundation)
    message(FATAL_ERROR "Installed package does not export jyppx::foundation")
endif()
if(NOT TARGET jyppx::core)
    message(FATAL_ERROR "Installed package does not export jyppx::core")
endif()
if(NOT TARGET jyppx::interop)
    message(FATAL_ERROR "Installed package does not export jyppx::interop")
endif()
if(NOT TARGET jyppx::data)
    message(FATAL_ERROR "Installed package does not export jyppx::data")
endif()
if(NOT TARGET jyppx::nn)
    message(FATAL_ERROR "Installed package does not export jyppx::nn")
endif()
if(NOT TARGET jyppx::training)
    message(FATAL_ERROR "Installed package does not export jyppx::training")
endif()
if(NOT TARGET jyppx::optim)
    message(FATAL_ERROR "Installed package does not export jyppx::optim")
endif()
if(NOT TARGET jyppx::experiment)
    message(FATAL_ERROR "Installed package does not export jyppx::experiment")
endif()
if(NOT TARGET jyppx::sdk)
    message(FATAL_ERROR "Installed package does not export jyppx::sdk")
endif()
if(TARGET jyppx_training_framework::jyppx_foundation OR
   TARGET jyppx_training_framework::jyppx_training_framework)
    message(FATAL_ERROR "Installed package still exports a legacy target name")
endif()

add_executable(jyppx_migrated_api_consumer main.cpp)
target_link_libraries(jyppx_migrated_api_consumer
    PRIVATE jyppx::sdk
)

if(WIN32)
    if(NOT JYPPX_TORCH_RUNTIME_DIR)
        message(FATAL_ERROR "Windows consumer smoke requires JYPPX_TORCH_RUNTIME_DIR")
    endif()
    file(GLOB JYPPX_CONSUMER_RUNTIME_DLLS
        "${JYPPX_TORCH_RUNTIME_DIR}/*.dll"
    )
    add_custom_command(TARGET jyppx_migrated_api_consumer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${JYPPX_CONSUMER_RUNTIME_DLLS}
            "$<TARGET_FILE_DIR:jyppx_migrated_api_consumer>"
        VERBATIM
    )
endif()

enable_testing()
add_test(NAME Runs COMMAND jyppx_migrated_api_consumer)
