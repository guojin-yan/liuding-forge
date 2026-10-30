cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(jyppx_liuding_bundle_signing_consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(jyppx CONFIG REQUIRED COMPONENTS bundle_signing)
if(NOT TARGET jyppx::liuding::bundle_signing OR NOT jyppx_bundle_signing_FOUND)
    message(FATAL_ERROR "Installed package does not satisfy component bundle_signing")
endif()

add_executable(jyppx_liuding_bundle_signing_consumer main.cpp)
target_link_libraries(jyppx_liuding_bundle_signing_consumer PRIVATE
    jyppx::liuding::bundle_signing)

# Explicit opt-in regression only; normal installed-consumer runs stay offline.
set(JYPPX_TEST_SYSTEM_CURL_ROOT "" CACHE PATH
    "Explicit Linux shared system curl SDK for the signing/TLS co-use regression")
if(NOT JYPPX_TEST_SYSTEM_CURL_ROOT STREQUAL "")
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
       NOT IS_ABSOLUTE "${JYPPX_TEST_SYSTEM_CURL_ROOT}" OR
       NOT IS_DIRECTORY "${JYPPX_TEST_SYSTEM_CURL_ROOT}")
        message(FATAL_ERROR "Signing/TLS co-use needs an explicit absolute Linux shared curl SDK prefix")
    endif()
    unset(_test_curl_include CACHE)
    unset(_test_curl_library CACHE)
    unset(_test_curl_include)
    unset(_test_curl_library)
    find_path(_test_curl_include curl/curl.h
        PATHS "${JYPPX_TEST_SYSTEM_CURL_ROOT}/include"
              "${JYPPX_TEST_SYSTEM_CURL_ROOT}/include/${CMAKE_LIBRARY_ARCHITECTURE}"
        NO_DEFAULT_PATH)
    find_library(_test_curl_library NAMES curl
        PATHS "${JYPPX_TEST_SYSTEM_CURL_ROOT}/lib"
              "${JYPPX_TEST_SYSTEM_CURL_ROOT}/lib64"
              "${JYPPX_TEST_SYSTEM_CURL_ROOT}/lib/${CMAKE_LIBRARY_ARCHITECTURE}"
        NO_DEFAULT_PATH)
    if(NOT _test_curl_include OR NOT _test_curl_library OR
       _test_curl_library MATCHES "\\.a$")
        message(FATAL_ERROR "Signing/TLS co-use requires shared libcurl and its matching headers")
    endif()
    get_filename_component(_test_curl_root "${JYPPX_TEST_SYSTEM_CURL_ROOT}" REALPATH)
    foreach(_test_curl_selected IN ITEMS _test_curl_include _test_curl_library)
        get_filename_component(_test_curl_real "${${_test_curl_selected}}" REALPATH)
        file(RELATIVE_PATH _test_curl_relative "${_test_curl_root}" "${_test_curl_real}")
        if(IS_ABSOLUTE "${_test_curl_relative}" OR _test_curl_relative MATCHES "^\\.\\.")
            message(FATAL_ERROR "Signing/TLS co-use curl resolves outside the explicit SDK prefix")
        endif()
    endforeach()
    file(STRINGS "${_test_curl_include}/curl/curlver.h" _test_curl_version_line
        REGEX "^#define LIBCURL_VERSION \"")
    string(REGEX REPLACE "^#define LIBCURL_VERSION \"([^\"]+)\".*" "\\1"
        _test_curl_version "${_test_curl_version_line}")
    if(NOT _test_curl_version MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+" OR
       _test_curl_version VERSION_LESS "7.81.0")
        message(FATAL_ERROR "Signing/TLS co-use requires curl headers >= 7.81.0")
    endif()
    file(SHA256 "${_test_curl_library}" _test_curl_library_sha256)
    file(SHA256 "${_test_curl_include}/curl/curlver.h" _test_curl_header_sha256)
    add_library(jyppx_test_system_curl SHARED IMPORTED)
    set_target_properties(jyppx_test_system_curl PROPERTIES
        IMPORTED_LOCATION "${_test_curl_library}"
        INTERFACE_INCLUDE_DIRECTORIES "${_test_curl_include}")
    target_link_libraries(jyppx_liuding_bundle_signing_consumer PRIVATE
        jyppx_test_system_curl)
    target_compile_definitions(jyppx_liuding_bundle_signing_consumer PRIVATE
        JYPPX_TEST_WITH_SYSTEM_CURL=1)
    message(STATUS "Signing/TLS co-use enabled: curl ${_test_curl_version}; library SHA256=${_test_curl_library_sha256}; header SHA256=${_test_curl_header_sha256}")
endif()

# Use the caller's installed-runtime environment; do not copy SDK DLL trees.
enable_testing()
add_test(NAME Runs COMMAND jyppx_liuding_bundle_signing_consumer)
set_tests_properties(Runs PROPERTIES TIMEOUT 60)
if(NOT JYPPX_TEST_SYSTEM_CURL_ROOT STREQUAL "")
    set_tests_properties(Runs PROPERTIES TIMEOUT 90)
endif()
