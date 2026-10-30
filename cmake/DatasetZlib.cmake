# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0.

include_guard(GLOBAL)

# Private static SDK only: no ambient find_package, Torch ABI, or network.
# An SDK must be staged and independently reviewed before enabling this option.
set(JYPPX_LIUDING_DATASET_ZLIB_ROOT "" CACHE PATH "Explicit zlib 1.3.2 static SDK")
set(JYPPX_LIUDING_DATASET_ZLIB_SOURCE_SHA256 "" CACHE STRING "Reviewed zlib-1.3.2.tar.gz SHA-256")
set(JYPPX_LIUDING_DATASET_ZLIB_LIBRARY_SHA256 "" CACHE STRING "Reviewed static zlib archive SHA-256")
set(JYPPX_LIUDING_DATASET_ZLIB_LICENSE_SHA256 "" CACHE STRING "Reviewed exact zlib LICENSE SHA-256")
if(NOT IS_ABSOLUTE "${JYPPX_LIUDING_DATASET_ZLIB_ROOT}")
    message(FATAL_ERROR "Native dataset preparation requires an explicit absolute DATASET_ZLIB_ROOT; no dependencies are downloaded")
endif()
set(_dataset_zlib_include "${JYPPX_LIUDING_DATASET_ZLIB_ROOT}/include")
if(NOT EXISTS "${_dataset_zlib_include}/zlib.h" OR NOT EXISTS "${_dataset_zlib_include}/zconf.h")
    message(FATAL_ERROR "Explicit dataset zlib SDK requires include/zlib.h and zconf.h")
endif()
file(READ "${_dataset_zlib_include}/zlib.h" _dataset_zlib_header)
if(NOT _dataset_zlib_header MATCHES "#[ \t]*define[ \t]+ZLIB_VERSION[ \t]+\"1\\.3\\.2\"")
    message(FATAL_ERROR "Dataset preparation requires exact final zlib 1.3.2 headers")
endif()
if(WIN32)
    set(_dataset_zlib_library "${JYPPX_LIUDING_DATASET_ZLIB_ROOT}/lib/zlibstatic.lib")
else()
    set(_dataset_zlib_library "${JYPPX_LIUDING_DATASET_ZLIB_ROOT}/lib/libz.a")
endif()
set(_dataset_zlib_source "${JYPPX_LIUDING_DATASET_ZLIB_ROOT}/source/zlib-1.3.2.tar.gz")
set(_dataset_zlib_license "${JYPPX_LIUDING_DATASET_ZLIB_ROOT}/share/licenses/zlib/LICENSE")
foreach(_dataset_zlib_kind IN ITEMS SOURCE LIBRARY LICENSE)
    string(TOLOWER "${_dataset_zlib_kind}" _dataset_zlib_lower)
    set(_dataset_zlib_file "${_dataset_zlib_${_dataset_zlib_lower}}")
    set(_dataset_zlib_expected "${JYPPX_LIUDING_DATASET_ZLIB_${_dataset_zlib_kind}_SHA256}")
    string(LENGTH "${_dataset_zlib_expected}" _dataset_zlib_hash_length)
    if(NOT _dataset_zlib_hash_length EQUAL 64 OR NOT _dataset_zlib_expected MATCHES "^[0-9a-f]+$")
        message(FATAL_ERROR "Explicit reviewed lowercase SHA-256 required for dataset zlib ${_dataset_zlib_kind}")
    endif()
    if(NOT EXISTS "${_dataset_zlib_file}")
        message(FATAL_ERROR "Explicit dataset zlib SDK is missing ${_dataset_zlib_file}")
    endif()
    file(SHA256 "${_dataset_zlib_file}" _dataset_zlib_actual)
    if(NOT _dataset_zlib_actual STREQUAL _dataset_zlib_expected)
        message(FATAL_ERROR "Dataset zlib ${_dataset_zlib_kind} SHA-256 mismatch")
    endif()
endforeach()
add_library(jyppx_dataset_zlib STATIC IMPORTED GLOBAL)
set_target_properties(jyppx_dataset_zlib PROPERTIES
    IMPORTED_LOCATION "${_dataset_zlib_library}"
    INTERFACE_INCLUDE_DIRECTORIES "${_dataset_zlib_include}"
    INTERFACE_COMPILE_DEFINITIONS Z_SOLO)
install(FILES "${_dataset_zlib_license}"
    DESTINATION share/doc/jyppx/licenses RENAME zlib-1.3.2-LICENSE.txt COMPONENT licenses)
