cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(jyppx_component_consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(JYPPX_SUPPORTED_COMPONENTS
    foundation core interop data nn optim training experiment sdk
)
if(NOT JYPPX_CONSUMER_COMPONENT IN_LIST JYPPX_SUPPORTED_COMPONENTS)
    message(FATAL_ERROR
        "JYPPX_CONSUMER_COMPONENT must name one supported component")
endif()

find_package(jyppx CONFIG REQUIRED COMPONENTS ${JYPPX_CONSUMER_COMPONENT})
if(NOT jyppx_${JYPPX_CONSUMER_COMPONENT}_FOUND)
    message(FATAL_ERROR
        "Installed package did not satisfy component ${JYPPX_CONSUMER_COMPONENT}")
endif()
if(NOT TARGET jyppx::${JYPPX_CONSUMER_COMPONENT})
    message(FATAL_ERROR
        "Installed package does not export jyppx::${JYPPX_CONSUMER_COMPONENT}")
endif()

string(TOUPPER "${JYPPX_CONSUMER_COMPONENT}" JYPPX_COMPONENT_MACRO)
add_executable(jyppx_component_consumer main.cpp)
target_compile_definitions(jyppx_component_consumer
    PRIVATE JYPPX_CONSUMER_${JYPPX_COMPONENT_MACRO}=1
)
target_link_libraries(jyppx_component_consumer
    PRIVATE jyppx::${JYPPX_CONSUMER_COMPONENT}
)

if(WIN32)
    if(NOT JYPPX_TORCH_RUNTIME_DIR)
        message(FATAL_ERROR
            "Windows consumer smoke requires JYPPX_TORCH_RUNTIME_DIR")
    endif()
    file(GLOB JYPPX_CONSUMER_RUNTIME_DLLS
        "${JYPPX_TORCH_RUNTIME_DIR}/*.dll"
    )
    add_custom_command(TARGET jyppx_component_consumer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${JYPPX_CONSUMER_RUNTIME_DLLS}
            "$<TARGET_FILE_DIR:jyppx_component_consumer>"
        VERBATIM
    )
endif()

enable_testing()
add_test(NAME Runs COMMAND jyppx_component_consumer)
