cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(jyppx_component_consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(JYPPX_SUPPORTED_COMPONENTS
    foundation core interop data nn optim training experiment models quantization serving all sdk
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

if(DEFINED JYPPX_EXPECTED_COMPONENTS)
    string(REPLACE "," ";" JYPPX_EXPECTED_COMPONENT_LIST
           "${JYPPX_EXPECTED_COMPONENTS}")
    foreach(JYPPX_EXPECTED_COMPONENT IN LISTS JYPPX_EXPECTED_COMPONENT_LIST)
        if(NOT TARGET jyppx::${JYPPX_EXPECTED_COMPONENT} OR
           NOT jyppx_${JYPPX_EXPECTED_COMPONENT}_FOUND)
            message(FATAL_ERROR
                "Expected installed component is absent: ${JYPPX_EXPECTED_COMPONENT}")
        endif()
    endforeach()
endif()

if(DEFINED JYPPX_ABSENT_COMPONENTS)
    string(REPLACE "," ";" JYPPX_ABSENT_COMPONENT_LIST
           "${JYPPX_ABSENT_COMPONENTS}")
    foreach(JYPPX_ABSENT_COMPONENT IN LISTS JYPPX_ABSENT_COMPONENT_LIST)
        if(TARGET jyppx::${JYPPX_ABSENT_COMPONENT} OR
           jyppx_${JYPPX_ABSENT_COMPONENT}_FOUND)
            message(FATAL_ERROR
                "Disabled installed component is still visible: ${JYPPX_ABSENT_COMPONENT}")
        endif()
    endforeach()
endif()

string(TOUPPER "${JYPPX_CONSUMER_COMPONENT}" JYPPX_COMPONENT_MACRO)
add_executable(jyppx_component_consumer main.cpp)
target_compile_definitions(jyppx_component_consumer
    PRIVATE JYPPX_CONSUMER_${JYPPX_COMPONENT_MACRO}=1
)
foreach(JYPPX_ENTITY_COMPONENT IN ITEMS
        foundation core interop data nn optim training experiment models quantization serving)
    string(TOUPPER "${JYPPX_ENTITY_COMPONENT}" JYPPX_ENTITY_COMPONENT_UPPER)
    if(TARGET jyppx::${JYPPX_ENTITY_COMPONENT})
        target_compile_definitions(jyppx_component_consumer PRIVATE
            JYPPX_EXPECT_COMPONENT_${JYPPX_ENTITY_COMPONENT_UPPER}=1)
    else()
        target_compile_definitions(jyppx_component_consumer PRIVATE
            JYPPX_EXPECT_COMPONENT_${JYPPX_ENTITY_COMPONENT_UPPER}=0)
    endif()
endforeach()
target_link_libraries(jyppx_component_consumer
    PRIVATE jyppx::${JYPPX_CONSUMER_COMPONENT}
)

if(WIN32)
    if(NOT JYPPX_TORCH_RUNTIME_DIR)
        message(FATAL_ERROR
            "Windows consumer smoke requires JYPPX_TORCH_RUNTIME_DIR")
    endif()
    if(JYPPX_RUNTIME_STAGER AND EXISTS "${JYPPX_RUNTIME_STAGER}")
        add_custom_command(TARGET jyppx_component_consumer POST_BUILD
            COMMAND ${CMAKE_COMMAND}
                "-DJYPPX_RUNTIME_SOURCE_DIR=${JYPPX_TORCH_RUNTIME_DIR}"
                "-DJYPPX_RUNTIME_OUTPUT_DIR=$<TARGET_FILE_DIR:jyppx_component_consumer>"
                -P "${JYPPX_RUNTIME_STAGER}" VERBATIM)
    else()
        file(GLOB JYPPX_CONSUMER_RUNTIME_DLLS "${JYPPX_TORCH_RUNTIME_DIR}/*.dll")
        add_custom_command(TARGET jyppx_component_consumer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${JYPPX_CONSUMER_RUNTIME_DLLS}
                "$<TARGET_FILE_DIR:jyppx_component_consumer>" VERBATIM)
    endif()
endif()

enable_testing()
add_test(NAME Runs COMMAND jyppx_component_consumer)
