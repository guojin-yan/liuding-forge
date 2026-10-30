# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.

if(NOT TARGET jyppx_liuding_telemetry_native)
    message(FATAL_ERROR "TELEMETRY_HTTP requires explicit TELEMETRY_NATIVE=ON")
endif()
add_library(jyppx_liuding_telemetry_http STATIC
    src/jyppx/liuding/telemetry_http/otlp_http.cpp)
add_library(jyppx::liuding::telemetry_http ALIAS jyppx_liuding_telemetry_http)
set_target_properties(jyppx_liuding_telemetry_http PROPERTIES
    OUTPUT_NAME liuding-forge-telemetry-http EXPORT_NAME telemetry_http)
target_include_directories(jyppx_liuding_telemetry_http PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_include_directories(jyppx_liuding_telemetry_http PRIVATE
    "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(jyppx_liuding_telemetry_http PUBLIC
    jyppx_liuding_telemetry_native jyppx_liuding_foundation)
if(WIN32)
    target_sources(jyppx_liuding_telemetry_http PRIVATE
        src/jyppx/liuding/telemetry_http/http_winhttp.cpp)
    target_link_libraries(jyppx_liuding_telemetry_http PRIVATE winhttp ws2_32)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    include("${CMAKE_CURRENT_LIST_DIR}/jyppxTelemetryHttpDependency.cmake")
    jyppx_require_telemetry_http_curl()
    target_sources(jyppx_liuding_telemetry_http PRIVATE
        src/jyppx/liuding/telemetry_http/http_curl.cpp)
    target_link_libraries(jyppx_liuding_telemetry_http PRIVATE
        jyppx_http_private::system_curl)
    install(FILES "${CMAKE_CURRENT_LIST_DIR}/jyppxTelemetryHttpDependency.cmake"
        DESTINATION lib/cmake/jyppx)
else()
    message(FATAL_ERROR "TELEMETRY_HTTP supports Windows WinHTTP and Linux shared system curl")
endif()
list(APPEND JYPPX_ENABLED_TARGETS jyppx_liuding_telemetry_http)
list(APPEND JYPPX_ENABLED_COMPONENTS telemetry_http)
list(APPEND JYPPX_ENABLED_HEADER_DIRS telemetry_http)
