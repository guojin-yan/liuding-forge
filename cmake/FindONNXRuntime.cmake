# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.

# This optional adapter uses an explicitly staged official SDK. Providers
# are selected at runtime; neither a filename nor provider discovery qualifies
# an execution path. No PATH search or configure-time download is allowed.
set(JYPPX_LIUDING_ONNX_RUNTIME_ROOT "" CACHE PATH
    "Explicit official ONNX Runtime SDK root matching the selected exact profile")
set(_jyppx_ort_version "1.23.2")
set(_jyppx_ort_api "23")
if(JYPPX_LIUDING_ONNX_RUNTIME_DIRECTML)
    set(_jyppx_ort_version "1.24.4")
    set(_jyppx_ort_api "24")
endif()
if(NOT IS_DIRECTORY "${JYPPX_LIUDING_ONNX_RUNTIME_ROOT}")
    message(FATAL_ERROR
        "ONNX Runtime requires JYPPX_LIUDING_ONNX_RUNTIME_ROOT pointing to a staged ${_jyppx_ort_version} SDK")
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
if(NOT ONNXRuntime_VERSION STREQUAL _jyppx_ort_version)
    message(FATAL_ERROR "ONNX Runtime requires SDK ${_jyppx_ort_version}, received ${ONNXRuntime_VERSION}")
endif()
file(STRINGS "${ONNXRuntime_ROOT}/include/onnxruntime_c_api.h" onnx_api_line
    REGEX "^#define ORT_API_VERSION ")
if(NOT onnx_api_line MATCHES "ORT_API_VERSION +${_jyppx_ort_api}$")
    message(FATAL_ERROR "ONNX Runtime requires ORT C API version ${_jyppx_ort_api}")
endif()

if(JYPPX_LIUDING_ONNX_RUNTIME_DIRECTML)
    if(NOT WIN32)
        message(FATAL_ERROR "The DirectML profile requires Windows x64")
    endif()
    if(CMAKE_CXX_COMPILER_ARCHITECTURE_ID AND
       NOT CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "x64")
        message(FATAL_ERROR "The pinned DirectML payload requires an x64 target")
    endif()
    set(JYPPX_LIUDING_DIRECTML_ROOT "" CACHE PATH
        "Explicit Microsoft.AI.DirectML 1.15.4 Windows x64 release SDK")
    if(NOT IS_ABSOLUTE "${JYPPX_LIUDING_DIRECTML_ROOT}" OR
       NOT EXISTS "${ONNXRuntime_INCLUDE_DIR}/dml_provider_factory.h")
        message(FATAL_ERROR "DirectML requires its explicit SDK root and the ORT DirectML provider header")
    endif()
    # Pin the official NuGet payload as well as its package version. Do not
    # accept a system DirectML DLL, debug payload, Xbox build or another ABI.
    if(NOT EXISTS "${JYPPX_LIUDING_DIRECTML_ROOT}/VERSION_NUMBER")
        message(FATAL_ERROR "DirectML requires a VERSION_NUMBER file containing 1.15.4")
    endif()
    file(STRINGS "${JYPPX_LIUDING_DIRECTML_ROOT}/VERSION_NUMBER" _jyppx_dml_version)
    if(NOT _jyppx_dml_version STREQUAL "1.15.4")
        message(FATAL_ERROR "The DirectML profile requires exactly DirectML 1.15.4")
    endif()
    set(_jyppx_dml_files
        "LICENSE.txt|a05138e3a085ff60a44881eedfa58dccb03ecc1d7b1f6ae888418e8c2fec4b8d"
        "include/DirectML.h|3daeed73aa64905c07b6f5394c7b45eabf0ac17bbca54b1584c871e03e4f54ad"
        "include/DirectMLConfig.h|2363f2f6cf6b4a8fbebd266d295e09ba43d6d4722c981afe7454917d1e927d7a"
        "bin/DirectML.dll|9c9e6d822561c6c41b90e6994b3e8857cf1d66dbfb1e0c4c799c7c89b4e92da1")
    foreach(entry IN LISTS _jyppx_dml_files)
        string(REPLACE "|" ";" pair "${entry}")
        list(GET pair 0 name)
        list(GET pair 1 expected)
        if(NOT EXISTS "${JYPPX_LIUDING_DIRECTML_ROOT}/${name}")
            message(FATAL_ERROR "Incomplete DirectML 1.15.4 SDK: ${name}")
        endif()
        file(SHA256 "${JYPPX_LIUDING_DIRECTML_ROOT}/${name}" actual)
        if(NOT actual STREQUAL expected)
            message(FATAL_ERROR "DirectML 1.15.4 official payload mismatch: ${name}")
        endif()
    endforeach()
    set(_jyppx_ort_dml_files
        "include/dml_provider_factory.h|e0bf35c3a45df1e41aa3659d8502ca525a669134c0f537a549ff555ae8c12261"
        "include/onnxruntime_c_api.h|7f03541b0233c53b8c91855910c9382c4fb4a26b8fe099965c7c3cad1af2ced6"
        "lib/onnxruntime.dll|e7eedec6a6f26dc39dc948276a75ef6d2bee3fff944d874ceed0bbd3b97bff40"
        "lib/onnxruntime.lib|281d55c95107ad75ac060569c005031baf1bb7fe1011bdc07a6bef1dee8792b5"
        "lib/onnxruntime_providers_shared.dll|265c8daf29637cb259cac8be9f08f2cd45f3883f0f0e4949cbfddd5b4cbec3b6")
    foreach(entry IN LISTS _jyppx_ort_dml_files)
        string(REPLACE "|" ";" pair "${entry}")
        list(GET pair 0 name)
        list(GET pair 1 expected)
        if(NOT EXISTS "${ONNXRuntime_ROOT}/${name}")
            message(FATAL_ERROR "Incomplete ONNX Runtime DirectML SDK: ${name}")
        endif()
        file(SHA256 "${ONNXRuntime_ROOT}/${name}" actual)
        if(NOT actual STREQUAL expected)
            message(FATAL_ERROR "ONNX Runtime 1.24.4 DirectML official payload mismatch: ${name}")
        endif()
    endforeach()
endif()

if(WIN32)
    set(ONNXRuntime_LIBRARY "${ONNXRuntime_ROOT}/lib/onnxruntime.lib")
    set(ONNXRuntime_RUNTIME "${ONNXRuntime_ROOT}/lib/onnxruntime.dll")
    file(GLOB ONNXRuntime_PROVIDER_LIBRARIES "${ONNXRuntime_ROOT}/lib/onnxruntime_providers_*.dll")
    if(JYPPX_LIUDING_ONNX_RUNTIME_DIRECTML)
        list(APPEND ONNXRuntime_PROVIDER_LIBRARIES
            "${JYPPX_LIUDING_DIRECTML_ROOT}/bin/DirectML.dll")
    endif()
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
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # Installed executables resolve the SDK's bundled ORT without the
        # original build tree or an external ONNX Runtime loader path.
        set_property(TARGET ${target} APPEND PROPERTY
            INSTALL_RPATH "$ORIGIN/../lib")
    endif()
endfunction()

function(jyppx_install_onnx_runtime)
    if(JYPPX_LIUDING_ONNX_RUNTIME_DIRECTML)
        install(FILES "${JYPPX_LIUDING_DIRECTML_ROOT}/LICENSE.txt"
            "${JYPPX_LIUDING_DIRECTML_ROOT}/VERSION_NUMBER"
            DESTINATION share/doc/jyppx/licenses/directml COMPONENT licenses)
    endif()
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
