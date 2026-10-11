cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(jyppx_training_framework VERSION 0.1.0 LANGUAGES CXX)

# ============================================
# Project Options
# ============================================
option(JYPPX_BUILD_TESTS "Build tests" ON)
option(JYPPX_BUILD_EXAMPLES "Build examples" ON)
option(JYPPX_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(JYPPX_BUILD_DOCS "Build documentation" OFF)
option(JYPPX_ENABLE_COVERAGE "Enable code coverage" OFF)

# ============================================
# C++ Standard
# ============================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================
# Build Type
# ============================================
get_property(JYPPX_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT JYPPX_MULTI_CONFIG AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")

# ============================================
# CMake Modules
# ============================================
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# ============================================
# Find LibTorch
# ============================================
set(JYPPX_BUNDLED_LIBTORCH "${PROJECT_SOURCE_DIR}/third_party/install/libtorch")
if(EXISTS "${JYPPX_BUNDLED_LIBTORCH}/share/cmake/Torch/TorchConfig.cmake"
   AND NOT Torch_DIR AND NOT CMAKE_PREFIX_PATH)
    list(PREPEND CMAKE_PREFIX_PATH "${JYPPX_BUNDLED_LIBTORCH}")
    message(STATUS "Using bundled LibTorch: ${JYPPX_BUNDLED_LIBTORCH}")
endif()
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

if(MSVC)
    # LibTorch's official Windows binaries are Release-built with /MD. Every
    # config of this project must use the same CRT, otherwise STL objects and
    # heap blocks crossing the torch DLL boundary corrupt the heap
    # (STATUS_HEAP_CORRUPTION / access violations in Debug builds).
    # CMP0091 moves runtime selection out of CMAKE_CXX_FLAGS_<CONFIG>, so set
    # the abstraction variable instead of string-replacing flags.
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    # CMake still defines _DEBUG for Debug configs, which flips the STL back
    # to iterator-debug level 2 (different std layout than the DLLs use).
    # Pin level 0 so both sides agree.
    string(APPEND CMAKE_CXX_FLAGS_DEBUG " /D_ITERATOR_DEBUG_LEVEL=0")
endif()

message(STATUS "LibTorch version: ${Torch_VERSION}")
message(STATUS "LibTorch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "LibTorch libraries: ${TORCH_LIBRARIES}")

# ============================================
# JYPPX Library
# ============================================
add_library(jyppx_training_framework STATIC)
add_library(jyppx_training_framework::jyppx_training_framework ALIAS jyppx_training_framework)
set_target_properties(jyppx_training_framework PROPERTIES
    OUTPUT_NAME jyppx_training_framework
)

# Include directories
target_include_directories(jyppx_training_framework
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${TORCH_INCLUDE_DIRS}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link LibTorch
target_link_libraries(jyppx_training_framework
    PUBLIC
        ${TORCH_LIBRARIES}
)

# Compiler options must be applied after the target has been created.
include(cmake/CompilerOptions.cmake)

# Sources will be added via subdirectories
add_subdirectory(src)

# ============================================
# Tests
# ============================================
if(JYPPX_BUILD_TESTS)
    enable_testing()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
        add_subdirectory(tests)
    else()
        message(STATUS "Skipping tests: tests/CMakeLists.txt is not present yet")
    endif()
endif()

# ============================================
# Examples
# ============================================
if(JYPPX_BUILD_EXAMPLES)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt")
        add_subdirectory(examples)
    else()
        message(STATUS "Skipping examples: examples/CMakeLists.txt is not present yet")
    endif()
endif()

# ============================================
# Documentation
# ============================================
if(JYPPX_BUILD_DOCS)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        add_subdirectory(docs)
    else()
        message(WARNING "Doxygen not found, documentation will not be built")
    endif()
endif()

# ============================================
# Installation
# ============================================
install(TARGETS jyppx_training_framework
    EXPORT jyppx_training_frameworkTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/jyppx
    DESTINATION include
)

install(EXPORT jyppx_training_frameworkTargets
    FILE jyppx_training_frameworkTargets.cmake
    NAMESPACE jyppx_training_framework::
    DESTINATION lib/cmake/jyppx_training_framework
)

# ============================================
# Package Configuration
# ============================================
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/jyppx_training_frameworkConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/jyppx_training_frameworkConfig.cmake
    INSTALL_DESTINATION lib/cmake/jyppx_training_framework
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/jyppx_training_frameworkConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/jyppx_training_frameworkConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/jyppx_training_frameworkConfigVersion.cmake
    DESTINATION lib/cmake/jyppx_training_framework
)

# ============================================
# Summary
# ============================================
message(STATUS "")
message(STATUS "JYPPX Training Framework Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build tests: ${JYPPX_BUILD_TESTS}")
message(STATUS "  Build examples: ${JYPPX_BUILD_EXAMPLES}")
message(STATUS "  Build benchmarks: ${JYPPX_BUILD_BENCHMARKS}")
message(STATUS "  Build docs: ${JYPPX_BUILD_DOCS}")
message(STATUS "")
