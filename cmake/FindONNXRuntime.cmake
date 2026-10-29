# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.

# This optional adapter uses an explicitly staged official SDK. Providers
# are selected at runtime; neither a filename nor provider discovery qualifies
# an execution path. No PATH search or configure-time download is allowed.
set(JYPPX_LIUDING_ONNX_RUNTIME_ROOT "" CACHE PATH
    "Explicit official ONNX Runtime 1.23.2 SDK root")
if(NOT IS_DIRECTORY "${JYPPX_LIUDING_ONNX_RUNTIME_ROOT}")
    message(FATAL_ERROR
        "ONNX Runtime requires JYPPX_LIUDING_ONNX_RUNTIME_ROOT pointing to a staged 1.23.2 SDK")
endif()
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8 OR
   NOT (WIN32 OR CMAKE_SYSTEM_NAME STREQUAL "Linux" OR APPLE))
    message(FATAL_ERROR "The ONNX Runtime adapter requires a compatible 64-bit Windows/Linux/macOS SDK")
endif()

get_filename_component(ONNXRuntime_ROOT "${JYPPX_LIUDING_ONNX_RUNTIME_ROOT}" REALPATH)
set(ONNXRuntime_INCLUDE_DIR "${ONNXRuntime_ROOT}/include")
foreach(required IN ITEMS VERSION_NUMBER LICENSE ThirdPartyNotices.txt
        include/onnxruntime_cxx_api.h include/onnxruntime_c_api.h)
    if(NOT EXISTS "${ONNXRuntime_ROOT}/${required}")
        message(FATAL_ERROR "Incomplete ONNX Runtime SDK: ${required}")
    endif()
endforeach()
file(STRINGS "${ONNXRuntime_ROOT}/VERSION_NUMBER" ONNXRuntime_VERSION LIMIT_COUNT 1)
string(STRIP "${ONNXRuntime_VERSION}" ONNXRuntime_VERSION)
if(NOT ONNXRuntime_VERSION STREQUAL "1.23.2")
    message(FATAL_ERROR "ONNX Runtime requires SDK 1.23.2, received ${ONNXRuntime_VERSION}")
endif()
file(STRINGS "${ONNXRuntime_ROOT}/include/onnxruntime_c_api.h" onnx_api_line
    REGEX "^#define ORT_API_VERSION ")
if(NOT onnx_api_line MATCHES "ORT_API_VERSION +23$")
    message(FATAL_ERROR "ONNX Runtime requires ORT C API version 23")
endif()

if(WIN32)
    set(ONNXRuntime_LIBRARY "${ONNXRuntime_ROOT}/lib/onnxruntime.lib")
    set(ONNXRuntime_RUNTIME "${ONNXRuntime_ROOT}/lib/onnxruntime.dll")
    file(GLOB ONNXRuntime_PROVIDER_LIBRARIES "${ONNXRuntime_ROOT}/lib/onnxruntime_providers_*.dll")
    set(JYPPX_ORT_INSTALL_LIBRARY "lib/onnxruntime.lib")
    set(JYPPX_ORT_INSTALL_RUNTIME "bin/onnxruntime.dll")
elseif(APPLE)
    set(ONNXRuntime_LIBRARY "${ONNXRuntime_ROOT}/lib/libonnxruntime.1.23.2.dylib")
    set(ONNXRuntime_RUNTIME "${ONNXRuntime_LIBRARY}")
    set(JYPPX_ORT_INSTALL_LIBRARY "lib/libonnxruntime.1.23.2.dylib")
    set(JYPPX_ORT_INSTALL_RUNTIME "${JYPPX_ORT_INSTALL_LIBRARY}")
else()
    set(ONNXRuntime_LIBRARY "${ONNXRuntime_ROOT}/lib/libonnxruntime.so.1.23.2")
    set(ONNXRuntime_RUNTIME "${ONNXRuntime_LIBRARY}")
    set(JYPPX_ORT_INSTALL_LIBRARY "lib/libonnxruntime.so.1.23.2")
    set(JYPPX_ORT_INSTALL_RUNTIME "${JYPPX_ORT_INSTALL_LIBRARY}")
endif()
foreach(required IN ITEMS "${ONNXRuntime_LIBRARY}" "${ONNXRuntime_RUNTIME}")
    if(NOT EXISTS "${required}")
        message(FATAL_ERROR "Incomplete ONNX Runtime SDK binary: ${required}")
    endif()
endforeach()
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime
    REQUIRED_VARS ONNXRuntime_INCLUDE_DIR ONNXRuntime_LIBRARY ONNXRuntime_RUNTIME
    VERSION_VAR ONNXRuntime_VERSION)

if(NOT TARGET ONNXRuntime::onnxruntime)
    add_library(ONNXRuntime::onnxruntime SHARED IMPORTED GLOBAL)
    set_target_properties(ONNXRuntime::onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRuntime_RUNTIME}"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRuntime_INCLUDE_DIR}")
    if(WIN32)
        set_property(TARGET ONNXRuntime::onnxruntime PROPERTY
            IMPORTED_IMPLIB "${ONNXRuntime_LIBRARY}")
    endif()
endif()

function(jyppx_add_onnx_runtime_dependency target)
    if(WIN32)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRuntime_RUNTIME}" ${ONNXRuntime_PROVIDER_LIBRARIES}
                "$<TARGET_FILE_DIR:${target}>" VERBATIM)
    endif()
endfunction()

function(jyppx_install_onnx_runtime)
    if(WIN32)
        install(FILES "${ONNXRuntime_LIBRARY}" DESTINATION lib)
        install(FILES "${ONNXRuntime_RUNTIME}" DESTINATION bin)
        if(ONNXRuntime_PROVIDER_LIBRARIES)
            install(FILES ${ONNXRuntime_PROVIDER_LIBRARIES} DESTINATION bin)
        endif()
    elseif(APPLE)
        file(GLOB runtime_files "${ONNXRuntime_ROOT}/lib/libonnxruntime*.dylib")
        install(FILES ${runtime_files} DESTINATION lib)
    else()
        # Preserve the upstream SONAME links as well as the actual binary.
        foreach(runtime_name IN ITEMS libonnxruntime.so libonnxruntime.so.1
                libonnxruntime.so.1.23.2)
            if(NOT EXISTS "${ONNXRuntime_ROOT}/lib/${runtime_name}")
                message(FATAL_ERROR "Incomplete ONNX Runtime SONAME chain: ${runtime_name}")
            endif()
            install(FILES "${ONNXRuntime_ROOT}/lib/${runtime_name}" DESTINATION lib)
        endforeach()
        file(GLOB provider_files "${ONNXRuntime_ROOT}/lib/libonnxruntime_providers_*.so")
        if(provider_files)
            install(FILES ${provider_files} DESTINATION lib)
        endif()
    endif()
    install(FILES "${ONNXRuntime_ROOT}/LICENSE"
        "${ONNXRuntime_ROOT}/ThirdPartyNotices.txt"
        "${ONNXRuntime_ROOT}/VERSION_NUMBER"
        DESTINATION share/doc/jyppx/licenses/onnxruntime COMPONENT licenses)
endfunction()
