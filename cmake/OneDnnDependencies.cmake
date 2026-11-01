# The optional adapter uses an explicitly staged official SDK. Never discover
# another oneDNN through LibTorch, the system search path, or the network.
set(JYPPX_LIUDING_ONEDNN_ROOT "" CACHE PATH "Explicit official oneDNN 3.12.0 static SDK")
set(JYPPX_LIUDING_ONEDNN_BUILD_CACHE "" CACHE FILEPATH "CMakeCache.txt that produced the explicit oneDNN SDK")
set(JYPPX_LIUDING_ONEDNN_LIBRARY_SHA256 "" CACHE STRING "Expected SHA256 of the explicit oneDNN static archive")
if(NOT Torch_VERSION STREQUAL "2.14.1")
    message(FATAL_ERROR "quantization_onednn requires the declared LibTorch 2.14.1 baseline")
endif()
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8 OR
   NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64|X86_64)$" OR
   NOT (WIN32 OR CMAKE_SYSTEM_NAME STREQUAL "Linux"))
    message(FATAL_ERROR "quantization_onednn currently requires Windows or Linux x86_64")
endif()
if(NOT IS_DIRECTORY "${JYPPX_LIUDING_ONEDNN_ROOT}" OR
   NOT EXISTS "${JYPPX_LIUDING_ONEDNN_BUILD_CACHE}")
    message(FATAL_ERROR "quantization_onednn requires explicit JYPPX_LIUDING_ONEDNN_ROOT and JYPPX_LIUDING_ONEDNN_BUILD_CACHE")
endif()

# Read values as data; do not execute a caller's cache or import a global DNNL
# target that could also be owned by a consumer's LibTorch installation.
file(STRINGS "${JYPPX_LIUDING_ONEDNN_BUILD_CACHE}" _jyppx_onednn_cache_lines)
function(jyppx_onednn_require_cache key expected)
    set(found FALSE)
    foreach(line IN LISTS _jyppx_onednn_cache_lines)
        if(line MATCHES "^${key}:[^=]+=(.*)$")
            set(found TRUE)
            if(NOT CMAKE_MATCH_1 STREQUAL "${expected}")
                message(FATAL_ERROR "quantization_onednn SDK requires ${key}=${expected}")
            endif()
        endif()
    endforeach()
    if(NOT found)
        message(FATAL_ERROR "quantization_onednn SDK cache is missing ${key}")
    endif()
endfunction()
jyppx_onednn_require_cache(DNNL_LIBRARY_TYPE STATIC)
jyppx_onednn_require_cache(DNNL_TARGET_ARCH X64)
jyppx_onednn_require_cache(DNNL_CPU_RUNTIME SEQ)
jyppx_onednn_require_cache(DNNL_GPU_RUNTIME NONE)
jyppx_onednn_require_cache(DNNL_BLAS_VENDOR NONE)
jyppx_onednn_require_cache(DNNL_ENABLE_PRIMITIVE_CPU_ISA AVX2)
jyppx_onednn_require_cache(ONEDNN_ENABLE_GEMM_KERNELS_ISA AVX2)
jyppx_onednn_require_cache(DNNL_ENABLE_MAX_CPU_ISA ON)
jyppx_onednn_require_cache(DNNL_ENABLE_PRIMITIVE "MATMUL;REORDER")
if(WIN32)
    jyppx_onednn_require_cache(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
else()
    jyppx_onednn_require_cache(CMAKE_POSITION_INDEPENDENT_CODE ON)
    jyppx_onednn_require_cache(CMAKE_C_VISIBILITY_PRESET hidden)
    jyppx_onednn_require_cache(CMAKE_CXX_VISIBILITY_PRESET hidden)
    jyppx_onednn_require_cache(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
endif()

set(JYPPX_ONEDNN_INCLUDE_DIR "${JYPPX_LIUDING_ONEDNN_ROOT}/include")
set(JYPPX_ONEDNN_LIBRARY_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}dnnl${CMAKE_STATIC_LIBRARY_SUFFIX}")
set(JYPPX_ONEDNN_LIBRARY "${JYPPX_LIUDING_ONEDNN_ROOT}/lib/${JYPPX_ONEDNN_LIBRARY_NAME}")
foreach(required IN ITEMS
        "${JYPPX_ONEDNN_LIBRARY}"
        "${JYPPX_ONEDNN_INCLUDE_DIR}/oneapi/dnnl/dnnl.hpp"
        "${JYPPX_ONEDNN_INCLUDE_DIR}/oneapi/dnnl/dnnl_version.h"
        "${JYPPX_ONEDNN_INCLUDE_DIR}/oneapi/dnnl/dnnl_config.h"
        "${JYPPX_LIUDING_ONEDNN_ROOT}/share/doc/dnnl/LICENSE")
    if(NOT EXISTS "${required}")
        message(FATAL_ERROR "quantization_onednn SDK is missing required file: ${required}")
    endif()
endforeach()
file(READ "${JYPPX_ONEDNN_INCLUDE_DIR}/oneapi/dnnl/dnnl_version.h" _jyppx_onednn_version)
foreach(pair IN ITEMS "MAJOR=3" "MINOR=12" "PATCH=0")
    string(REPLACE "=" ";" parts "${pair}")
    list(GET parts 0 key)
    list(GET parts 1 value)
    if(NOT _jyppx_onednn_version MATCHES "#define DNNL_VERSION_${key} ${value}([\r\n]|$)")
        message(FATAL_ERROR "quantization_onednn requires official oneDNN 3.12.0 headers")
    endif()
endforeach()
file(READ "${JYPPX_ONEDNN_INCLUDE_DIR}/oneapi/dnnl/dnnl_config.h" _jyppx_onednn_config)
if(NOT _jyppx_onednn_config MATCHES "#define DNNL_CPU_RUNTIME DNNL_RUNTIME_SEQ([\r\n]|$)" OR
   NOT _jyppx_onednn_config MATCHES "#define DNNL_GPU_RUNTIME DNNL_RUNTIME_NONE([\r\n]|$)")
    message(FATAL_ERROR "quantization_onednn SDK headers require CPU SEQ and GPU NONE")
endif()
string(LENGTH "${JYPPX_LIUDING_ONEDNN_LIBRARY_SHA256}" _jyppx_onednn_digest_length)
if(NOT _jyppx_onednn_digest_length EQUAL 64 OR
   NOT JYPPX_LIUDING_ONEDNN_LIBRARY_SHA256 MATCHES "^[0-9a-fA-F]+$")
    message(FATAL_ERROR "quantization_onednn requires an explicit 64-digit JYPPX_LIUDING_ONEDNN_LIBRARY_SHA256")
endif()
file(SHA256 "${JYPPX_ONEDNN_LIBRARY}" JYPPX_ONEDNN_LIBRARY_SHA256)
string(TOLOWER "${JYPPX_LIUDING_ONEDNN_LIBRARY_SHA256}" _jyppx_onednn_expected_sha256)
if(NOT JYPPX_ONEDNN_LIBRARY_SHA256 STREQUAL _jyppx_onednn_expected_sha256)
    message(FATAL_ERROR "quantization_onednn SDK static archive SHA256 mismatch")
endif()
file(SHA256 "${JYPPX_LIUDING_ONEDNN_BUILD_CACHE}" JYPPX_ONEDNN_BUILD_CONFIGURATION_SHA256)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${JYPPX_ONEDNN_LIBRARY}" "${JYPPX_LIUDING_ONEDNN_BUILD_CACHE}"
    "${JYPPX_ONEDNN_INCLUDE_DIR}/oneapi/dnnl/dnnl_version.h"
    "${JYPPX_ONEDNN_INCLUDE_DIR}/oneapi/dnnl/dnnl_config.h")
set(JYPPX_ONEDNN_SOURCE_COMMIT "80afa71049cd69a3df32adcccb623b12cd7baa22")
set(JYPPX_ONEDNN_SOURCE_ARCHIVE_SHA256 "f13ce92168cae7bd25f5efc43f00b37ff14bcc62ffdea3d85220b1c50c104f17")
add_library(LiudingOneDNN::dnnl STATIC IMPORTED GLOBAL)
set_target_properties(LiudingOneDNN::dnnl PROPERTIES
    IMPORTED_LOCATION "${JYPPX_ONEDNN_LIBRARY}")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Keep the private archive's C/C++ symbols local in final ELF consumers so
    # it cannot interpose LibTorch's separately built oneDNN (or vice versa).
    set_property(TARGET LiudingOneDNN::dnnl PROPERTY
        INTERFACE_LINK_OPTIONS "LINKER:--exclude-libs,${JYPPX_ONEDNN_LIBRARY_NAME}")
endif()
