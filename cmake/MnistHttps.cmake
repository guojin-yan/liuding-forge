# Optional product-tool transport. Never fetch dependencies or affect offline
# builds. Windows uses the existing system WinHTTP SDK/runtime. Linux accepts
# an explicitly selected shared system libcurl SDK only; its TLS dependencies
# remain OS/package-manager managed, not vendored or installed by this project.
if(WIN32)
    target_sources(jyppx_dataset_preparation PRIVATE mnist_download_winhttp.cpp)
    target_link_libraries(jyppx_dataset_preparation PRIVATE winhttp)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(JYPPX_LIUDING_MNIST_CURL_ROOT "" CACHE PATH
        "Explicit system shared libcurl >= 7.81 SDK prefix, e.g. /usr; never downloaded")
    if(NOT IS_ABSOLUTE "${JYPPX_LIUDING_MNIST_CURL_ROOT}" OR
       NOT IS_DIRECTORY "${JYPPX_LIUDING_MNIST_CURL_ROOT}")
        message(FATAL_ERROR "MNIST HTTPS needs an explicit absolute JYPPX_LIUDING_MNIST_CURL_ROOT; offline preparation needs no curl")
    endif()
    # Local non-cache variables prevent a different prior SDK selection from
    # silently satisfying this configuration.
    unset(_mnist_curl_include CACHE)
    unset(_mnist_curl_library CACHE)
    unset(_mnist_curl_include)
    unset(_mnist_curl_library)
    find_path(_mnist_curl_include curl/curl.h
        PATHS "${JYPPX_LIUDING_MNIST_CURL_ROOT}/include"
              "${JYPPX_LIUDING_MNIST_CURL_ROOT}/include/${CMAKE_LIBRARY_ARCHITECTURE}"
        NO_DEFAULT_PATH)
    find_library(_mnist_curl_library NAMES curl
        PATHS "${JYPPX_LIUDING_MNIST_CURL_ROOT}/lib"
              "${JYPPX_LIUDING_MNIST_CURL_ROOT}/lib64"
              "${JYPPX_LIUDING_MNIST_CURL_ROOT}/lib/${CMAKE_LIBRARY_ARCHITECTURE}"
        NO_DEFAULT_PATH)
    if(NOT _mnist_curl_include OR NOT _mnist_curl_library OR
       _mnist_curl_library MATCHES "\\.a$")
        message(FATAL_ERROR "MNIST HTTPS requires the explicitly selected shared system curl SDK; static/transitive dependency assembly is unsupported")
    endif()
    get_filename_component(_mnist_curl_real "${_mnist_curl_library}" REALPATH)
    get_filename_component(_mnist_curl_root "${JYPPX_LIUDING_MNIST_CURL_ROOT}" REALPATH)
    file(RELATIVE_PATH _mnist_curl_relative "${_mnist_curl_root}" "${_mnist_curl_real}")
    if(IS_ABSOLUTE "${_mnist_curl_relative}" OR _mnist_curl_relative MATCHES "^\\.\\.")
        message(FATAL_ERROR "MNIST curl library resolves outside the explicit SDK prefix")
    endif()
    get_filename_component(_mnist_curl_include_real "${_mnist_curl_include}" REALPATH)
    file(RELATIVE_PATH _mnist_curl_include_relative "${_mnist_curl_root}" "${_mnist_curl_include_real}")
    if(IS_ABSOLUTE "${_mnist_curl_include_relative}" OR _mnist_curl_include_relative MATCHES "^\\.\\.")
        message(FATAL_ERROR "MNIST curl headers resolve outside the explicit SDK prefix")
    endif()
    if(NOT EXISTS "${_mnist_curl_include}/curl/curlver.h")
        message(FATAL_ERROR "MNIST curl SDK lacks curlver.h")
    endif()
    file(STRINGS "${_mnist_curl_include}/curl/curlver.h" _mnist_curl_version_line
        REGEX "^#define LIBCURL_VERSION \"")
    string(REGEX REPLACE "^#define LIBCURL_VERSION \"([^\"]+)\".*" "\\1"
        _mnist_curl_version "${_mnist_curl_version_line}")
    if(NOT _mnist_curl_version MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+" OR
       _mnist_curl_version VERSION_LESS "7.81.0")
        message(FATAL_ERROR "MNIST HTTPS requires curl headers >= 7.81.0")
    endif()
    file(SHA256 "${_mnist_curl_real}" JYPPX_LIUDING_MNIST_CURL_LIBRARY_SHA256)
    file(SHA256 "${_mnist_curl_include}/curl/curlver.h" JYPPX_LIUDING_MNIST_CURL_HEADER_SHA256)
    set(JYPPX_LIUDING_MNIST_CURL_SELECTED_LIBRARY "${_mnist_curl_real}"
        CACHE INTERNAL "Selected MNIST HTTPS system curl shared library" FORCE)
    set(JYPPX_LIUDING_MNIST_CURL_SELECTED_HEADER "${_mnist_curl_include}/curl/curlver.h"
        CACHE INTERNAL "Selected MNIST HTTPS system curl version header" FORCE)
    set(JYPPX_LIUDING_MNIST_CURL_SELECTED_VERSION "${_mnist_curl_version}"
        CACHE INTERNAL "Selected MNIST HTTPS system curl header version" FORCE)
    set(JYPPX_LIUDING_MNIST_CURL_LIBRARY_SHA256 "${JYPPX_LIUDING_MNIST_CURL_LIBRARY_SHA256}"
        CACHE INTERNAL "Selected MNIST HTTPS system curl library digest" FORCE)
    set(JYPPX_LIUDING_MNIST_CURL_HEADER_SHA256 "${JYPPX_LIUDING_MNIST_CURL_HEADER_SHA256}"
        CACHE INTERNAL "Selected MNIST HTTPS system curl header digest" FORCE)
    add_library(jyppx_mnist_system_curl SHARED IMPORTED GLOBAL)
    set_target_properties(jyppx_mnist_system_curl PROPERTIES
        IMPORTED_LOCATION "${_mnist_curl_real}"
        INTERFACE_INCLUDE_DIRECTORIES "${_mnist_curl_include}")
    target_sources(jyppx_dataset_preparation PRIVATE mnist_download_curl.cpp)
    target_link_libraries(jyppx_dataset_preparation PRIVATE jyppx_mnist_system_curl)
    message(STATUS "MNIST HTTPS system curl ${_mnist_curl_version}: ${_mnist_curl_real}; SHA256=${JYPPX_LIUDING_MNIST_CURL_LIBRARY_SHA256}")
else()
    message(FATAL_ERROR "Optional native MNIST HTTPS currently supports Windows WinHTTP and Linux system shared libcurl")
endif()
