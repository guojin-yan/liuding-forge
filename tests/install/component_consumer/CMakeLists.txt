cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(jyppx_component_consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(JYPPX_SUPPORTED_COMPONENTS
    foundation core interop data data_image vision_render nn optim training experiment models onnx_export onnx_runtime quantization serving all sdk
)
if(NOT JYPPX_CONSUMER_COMPONENT IN_LIST JYPPX_SUPPORTED_COMPONENTS)
    message(FATAL_ERROR
        "JYPPX_CONSUMER_COMPONENT must name one supported component")
endif()

find_package(jyppx CONFIG REQUIRED COMPONENTS ${JYPPX_CONSUMER_COMPONENT})
if(TARGET jyppx::liuding::sdk)
    # Even when Preview components are installed, SDK consumers must link
    # exactly the eight Core components, without the `all` aggregate.
    set(expected_sdk_links)
    foreach(component IN ITEMS foundation core interop data nn optim training experiment)
        list(APPEND expected_sdk_links jyppx::liuding::${component})
    endforeach()
    get_target_property(actual_sdk_links jyppx::liuding::sdk INTERFACE_LINK_LIBRARIES)
    list(SORT expected_sdk_links)
    list(SORT actual_sdk_links)
    if(NOT actual_sdk_links STREQUAL expected_sdk_links)
        message(FATAL_ERROR "Installed SDK must link exactly eight Core components: ${actual_sdk_links}")
    endif()
endif()
if(NOT jyppx_${JYPPX_CONSUMER_COMPONENT}_FOUND)
    message(FATAL_ERROR
        "Installed package did not satisfy component ${JYPPX_CONSUMER_COMPONENT}")
endif()
if(NOT TARGET jyppx::liuding::${JYPPX_CONSUMER_COMPONENT})
    message(FATAL_ERROR
        "Installed package does not export jyppx::liuding::${JYPPX_CONSUMER_COMPONENT}")
endif()

if(DEFINED JYPPX_EXPECTED_COMPONENTS)
    string(REPLACE "," ";" JYPPX_EXPECTED_COMPONENT_LIST
           "${JYPPX_EXPECTED_COMPONENTS}")
    foreach(JYPPX_EXPECTED_COMPONENT IN LISTS JYPPX_EXPECTED_COMPONENT_LIST)
        if(NOT TARGET jyppx::liuding::${JYPPX_EXPECTED_COMPONENT} OR
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
        if(TARGET jyppx::liuding::${JYPPX_ABSENT_COMPONENT} OR
           jyppx_${JYPPX_ABSENT_COMPONENT}_FOUND)
            message(FATAL_ERROR
                "Disabled installed component is still visible: ${JYPPX_ABSENT_COMPONENT}")
        endif()
    endforeach()
endif()

string(TOUPPER "${JYPPX_CONSUMER_COMPONENT}" JYPPX_LIUDING_COMPONENT_MACRO)
add_executable(jyppx_component_consumer main.cpp)
target_compile_definitions(jyppx_component_consumer
    PRIVATE JYPPX_CONSUMER_${JYPPX_LIUDING_COMPONENT_MACRO}=1
)
foreach(JYPPX_ENTITY_COMPONENT IN ITEMS
        foundation core interop data data_image vision_render nn optim training experiment models onnx_export onnx_runtime quantization serving)
    string(TOUPPER "${JYPPX_ENTITY_COMPONENT}" JYPPX_ENTITY_COMPONENT_UPPER)
    if(TARGET jyppx::liuding::${JYPPX_ENTITY_COMPONENT})
        target_compile_definitions(jyppx_component_consumer PRIVATE
            JYPPX_EXPECT_COMPONENT_${JYPPX_ENTITY_COMPONENT_UPPER}=1)
        # Imported targets keep the stable API but must locate branded files
        # in every installed configuration, including Unix's lib prefix.
        string(REPLACE "_" "-" JYPPX_LIBRARY_COMPONENT
               "${JYPPX_ENTITY_COMPONENT}")
        get_target_property(JYPPX_IMPORTED_CONFIGURATIONS
            jyppx::liuding::${JYPPX_ENTITY_COMPONENT} IMPORTED_CONFIGURATIONS)
        foreach(JYPPX_IMPORTED_CONFIG IN LISTS JYPPX_IMPORTED_CONFIGURATIONS)
            get_target_property(JYPPX_IMPORTED_LOCATION
                jyppx::liuding::${JYPPX_ENTITY_COMPONENT}
                IMPORTED_LOCATION_${JYPPX_IMPORTED_CONFIG})
            get_filename_component(JYPPX_IMPORTED_FILENAME
                "${JYPPX_IMPORTED_LOCATION}" NAME)
            if(NOT EXISTS "${JYPPX_IMPORTED_LOCATION}" OR
               NOT JYPPX_IMPORTED_FILENAME MATCHES
                   "^(lib)?liuding-forge-${JYPPX_LIBRARY_COMPONENT}")
                message(FATAL_ERROR
                    "Installed target has an invalid branded artifact: ${JYPPX_IMPORTED_LOCATION}")
            endif()
        endforeach()
    else()
        target_compile_definitions(jyppx_component_consumer PRIVATE
            JYPPX_EXPECT_COMPONENT_${JYPPX_ENTITY_COMPONENT_UPPER}=0)
    endif()
endforeach()
target_link_libraries(jyppx_component_consumer
    PRIVATE jyppx::liuding::${JYPPX_CONSUMER_COMPONENT}
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

if(WIN32 AND JYPPX_CONSUMER_COMPONENT STREQUAL "onnx_runtime")
    get_filename_component(runtime_prefix "${jyppx_DIR}/../../.." ABSOLUTE)
    file(GLOB runtime_dlls "${runtime_prefix}/bin/onnxruntime*.dll")
    if(NOT EXISTS "${runtime_prefix}/bin/onnxruntime.dll")
        message(FATAL_ERROR "Installed ONNX Runtime DLL is missing")
    endif()
    foreach(runtime_dll IN LISTS runtime_dlls)
        get_filename_component(runtime_name "${runtime_dll}" NAME)
        add_custom_command(TARGET jyppx_component_consumer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E rm -f
                "$<TARGET_FILE_DIR:jyppx_component_consumer>/${runtime_name}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${runtime_dll}"
                "$<TARGET_FILE_DIR:jyppx_component_consumer>" VERBATIM)
    endforeach()
endif()

enable_testing()
add_test(NAME Runs COMMAND jyppx_component_consumer)
