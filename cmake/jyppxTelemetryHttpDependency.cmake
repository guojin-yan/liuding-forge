# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.

# The same selector is used by the source build and installed consumer. The
# product never vendors curl/TLS or records the builder's absolute SDK path in
# its exports. Each consumer explicitly selects its own system SDK prefix.
function(jyppx_require_telemetry_http_curl)
    set(JYPPX_LIUDING_TELEMETRY_HTTP_CURL_ROOT "" CACHE PATH
        "Explicit system shared curl >= 7.81 SDK prefix; never downloaded")
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
       NOT IS_ABSOLUTE "${JYPPX_LIUDING_TELEMETRY_HTTP_CURL_ROOT}" OR
       NOT IS_DIRECTORY "${JYPPX_LIUDING_TELEMETRY_HTTP_CURL_ROOT}")
        message(FATAL_ERROR "TELEMETRY_HTTP requires an explicit absolute JYPPX_LIUDING_TELEMETRY_HTTP_CURL_ROOT on Linux")
    endif()
    get_filename_component(_http_curl_root
        "${JYPPX_LIUDING_TELEMETRY_HTTP_CURL_ROOT}" REALPATH)
    unset(_http_curl_include CACHE)
    unset(_http_curl_library CACHE)
    unset(_http_curl_include)
    unset(_http_curl_library)
    find_path(_http_curl_include curl/curl.h NO_DEFAULT_PATH
        PATHS "${_http_curl_root}/include"
              "${_http_curl_root}/include/${CMAKE_LIBRARY_ARCHITECTURE}")
    # Require an actual shared object, never a static archive selected by a
    # consumer's CMAKE_FIND_LIBRARY_SUFFIXES override.
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".so")
    find_library(_http_curl_library NAMES curl NO_DEFAULT_PATH
        PATHS "${_http_curl_root}/lib" "${_http_curl_root}/lib64"
              "${_http_curl_root}/lib/${CMAKE_LIBRARY_ARCHITECTURE}")
    if(NOT _http_curl_include OR NOT _http_curl_library)
        message(FATAL_ERROR "TELEMETRY_HTTP requires the explicitly selected shared system curl SDK")
    endif()
    get_filename_component(_http_curl_real "${_http_curl_library}" REALPATH)
    get_filename_component(_http_curl_headers_real "${_http_curl_include}" REALPATH)
    foreach(_http_curl_path IN ITEMS "${_http_curl_real}" "${_http_curl_headers_real}")
        file(RELATIVE_PATH _http_curl_relative "${_http_curl_root}" "${_http_curl_path}")
        if(IS_ABSOLUTE "${_http_curl_relative}" OR
           _http_curl_relative STREQUAL ".." OR
           _http_curl_relative MATCHES "^\\.\\./")
            message(FATAL_ERROR "TELEMETRY_HTTP curl SDK resolves outside its explicit prefix")
        endif()
    endforeach()
    if(NOT _http_curl_real MATCHES "\\.so(\\.[0-9]+)*$" OR
       NOT EXISTS "${_http_curl_include}/curl/curlver.h")
        message(FATAL_ERROR "TELEMETRY_HTTP needs shared libcurl and its versioned headers")
    endif()
    file(STRINGS "${_http_curl_include}/curl/curlver.h" _http_curl_version_line
        REGEX "^#define LIBCURL_VERSION \"")
    string(REGEX REPLACE "^#define LIBCURL_VERSION \"([^\"]+)\".*" "\\1"
        _http_curl_version "${_http_curl_version_line}")
    if(NOT _http_curl_version MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+" OR
       _http_curl_version VERSION_LESS "7.81.0")
        message(FATAL_ERROR "TELEMETRY_HTTP requires curl headers >= 7.81.0")
    endif()
    if(TARGET jyppx_http_private::system_curl)
        get_target_property(_http_curl_existing jyppx_http_private::system_curl IMPORTED_LOCATION)
        if(NOT _http_curl_existing STREQUAL _http_curl_real)
            message(FATAL_ERROR "TELEMETRY_HTTP curl target already selects a different shared SDK")
        endif()
    else()
        add_library(jyppx_http_private::system_curl SHARED IMPORTED GLOBAL)
        set_target_properties(jyppx_http_private::system_curl PROPERTIES
            IMPORTED_LOCATION "${_http_curl_real}"
            INTERFACE_INCLUDE_DIRECTORIES "${_http_curl_include}")
    endif()
    file(SHA256 "${_http_curl_real}" _http_curl_library_sha)
    file(SHA256 "${_http_curl_include}/curl/curlver.h" _http_curl_header_sha)
    set(JYPPX_LIUDING_TELEMETRY_HTTP_CURL_SELECTED_LIBRARY "${_http_curl_real}"
        CACHE INTERNAL "Selected OTLP system shared curl" FORCE)
    set(JYPPX_LIUDING_TELEMETRY_HTTP_CURL_SELECTED_VERSION "${_http_curl_version}"
        CACHE INTERNAL "Selected OTLP curl header version" FORCE)
    set(JYPPX_LIUDING_TELEMETRY_HTTP_CURL_LIBRARY_SHA256 "${_http_curl_library_sha}"
        CACHE INTERNAL "Selected OTLP shared curl digest" FORCE)
    set(JYPPX_LIUDING_TELEMETRY_HTTP_CURL_HEADER_SHA256 "${_http_curl_header_sha}"
        CACHE INTERNAL "Selected OTLP curl header digest" FORCE)
endfunction()
