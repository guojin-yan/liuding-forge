# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0.

# Deliberately avoid find_package(OpenSSL): an ambient Python, Torch or PATH
# installation must not silently choose the signing backend.
set(JYPPX_LIUDING_OPENSSL_ROOT "" CACHE PATH
    "Explicit OpenSSL 3.5.9 static SDK prefix for native bundle signing")
if(NOT IS_ABSOLUTE "${JYPPX_LIUDING_OPENSSL_ROOT}")
    message(FATAL_ERROR "Bundle signing requires an explicit absolute JYPPX_LIUDING_OPENSSL_ROOT (OpenSSL 3.5.9 static SDK)")
endif()
set(_jyppx_crypto_include "${JYPPX_LIUDING_OPENSSL_ROOT}/include")
if(NOT EXISTS "${_jyppx_crypto_include}/openssl/opensslv.h")
    message(FATAL_ERROR "Explicit bundle signing SDK is missing openssl/opensslv.h")
endif()
file(READ "${_jyppx_crypto_include}/openssl/opensslv.h" _jyppx_crypto_version)
foreach(_jyppx_version_part IN ITEMS MAJOR MINOR PATCH)
    if(_jyppx_version_part STREQUAL "MAJOR")
        set(_jyppx_expected_part 3)
    elseif(_jyppx_version_part STREQUAL "MINOR")
        set(_jyppx_expected_part 5)
    else()
        set(_jyppx_expected_part 9)
    endif()
    if(NOT _jyppx_crypto_version MATCHES
       "#[ \t]*define[ \t]+OPENSSL_VERSION_${_jyppx_version_part}[ \t]+${_jyppx_expected_part}([^0-9]|$)")
        message(FATAL_ERROR "Bundle signing requires exact OpenSSL 3.5.9 headers")
    endif()
endforeach()
if(NOT _jyppx_crypto_version MATCHES "OPENSSL_VERSION_STR[ \t]+\"3\\.5\\.9\"")
    message(FATAL_ERROR "Bundle signing requires final OpenSSL 3.5.9, without prerelease suffix")
endif()
if(WIN32)
    set(_jyppx_crypto_archive_name libcrypto.lib)
else()
    set(_jyppx_crypto_archive_name libcrypto.a)
endif()
set(JYPPX_BUNDLE_CRYPTO_LIBRARY "")
foreach(_jyppx_crypto_libdir IN ITEMS lib lib64)
    if(EXISTS "${JYPPX_LIUDING_OPENSSL_ROOT}/${_jyppx_crypto_libdir}/${_jyppx_crypto_archive_name}")
        set(JYPPX_BUNDLE_CRYPTO_LIBRARY
            "${JYPPX_LIUDING_OPENSSL_ROOT}/${_jyppx_crypto_libdir}/${_jyppx_crypto_archive_name}")
        break()
    endif()
endforeach()
if(NOT JYPPX_BUNDLE_CRYPTO_LIBRARY)
    message(FATAL_ERROR "Explicit bundle signing SDK is missing its static Crypto archive")
endif()
if(TARGET OpenSSL::Crypto)
    message(FATAL_ERROR "Bundle signing refuses a pre-existing ambient OpenSSL::Crypto target; configure with only the explicit SDK")
endif()
find_package(Threads REQUIRED)
add_library(OpenSSL::Crypto STATIC IMPORTED GLOBAL)
set_target_properties(OpenSSL::Crypto PROPERTIES
    IMPORTED_LOCATION "${JYPPX_BUNDLE_CRYPTO_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${_jyppx_crypto_include}")
if(WIN32)
    set_property(TARGET OpenSSL::Crypto PROPERTY INTERFACE_LINK_LIBRARIES
        "ws2_32;crypt32;advapi32;user32;gdi32;bcrypt")
else()
    set_property(TARGET OpenSSL::Crypto PROPERTY INTERFACE_LINK_LIBRARIES
        "Threads::Threads;${CMAKE_DL_LIBS}")
endif()
file(SHA256 "${JYPPX_BUNDLE_CRYPTO_LIBRARY}" JYPPX_BUNDLE_CRYPTO_SHA256)
set(JYPPX_BUNDLE_CRYPTO_FILENAME "${_jyppx_crypto_archive_name}")
configure_file("${PROJECT_SOURCE_DIR}/cmake/jyppxBundleCrypto.cmake.in"
    "${PROJECT_BINARY_DIR}/jyppxBundleCrypto.cmake" @ONLY)
install(FILES "${JYPPX_BUNDLE_CRYPTO_LIBRARY}"
    DESTINATION lib/jyppx/openssl-3.5.9)
install(FILES "${PROJECT_BINARY_DIR}/jyppxBundleCrypto.cmake"
    DESTINATION lib/cmake/jyppx)
install(FILES "${PROJECT_SOURCE_DIR}/third_party/licenses/openssl-3.5.9-LICENSE.txt"
    DESTINATION share/doc/jyppx/licenses COMPONENT licenses)
