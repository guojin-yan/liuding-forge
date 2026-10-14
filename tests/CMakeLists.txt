# JYPPX Training Framework tests.
#
# GoogleTest is fetched from the local third_party/downloads cache when
# available, falling back to the network. Tests are registered with CTest
# through gtest_discover_tests (PRE_TEST keeps discovery at ctest time, after
# the POST_BUILD step has copied the LibTorch runtime DLLs next to the test
# executables).

include(FetchContent)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(
        NAME CompatibilitySource
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/compatibility.py
                --file ${CMAKE_SOURCE_DIR}/compatibility.json --validate
    )
    set_tests_properties(CompatibilitySource PROPERTIES LABELS "hygiene;compatibility")
endif()

set(JYPPX_GTEST_VERSION 1.14.0)
set(JYPPX_GTEST_ZIP "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/downloads/googletest-${JYPPX_GTEST_VERSION}.zip")

if(EXISTS "${JYPPX_GTEST_ZIP}")
    FetchContent_Declare(googletest
        URL "${JYPPX_GTEST_ZIP}"
        URL_HASH SHA256=1f357c27ca988c3f7c6b4bf68a9395005ac6761f034046e9dde0896e3aba00e4
    )
else()
    message(STATUS "googletest zip not found in third_party/downloads; fetching from GitHub")
    FetchContent_Declare(googletest
        URL "https://github.com/google/googletest/archive/refs/tags/v${JYPPX_GTEST_VERSION}.zip"
    )
endif()

# Build gtest against the shared CRT the project already forces (see the root
# CMakeLists MSVC block), and keep it out of the install/export set.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

add_test(
    NAME RepositoryHygiene
    COMMAND ${CMAKE_COMMAND}
        -DJYPPX_SOURCE_DIR=${CMAKE_SOURCE_DIR}
        -P ${CMAKE_SOURCE_DIR}/cmake/CheckRepositoryHygiene.cmake
)
set_tests_properties(RepositoryHygiene PROPERTIES LABELS "hygiene")

# Set before add_subdirectory(unit) so jyppx_add_gtest can locate test_utils/.
set(JYPPX_TESTS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# LibTorch's Windows distribution keeps runtime DLLs beside its import
# libraries; tests must find them to run.
if(WIN32)
    get_filename_component(JYPPX_TORCH_RUNTIME_DIR "${TORCH_LIBRARY}" DIRECTORY)
    file(GLOB JYPPX_TORCH_RUNTIME_DLLS "${JYPPX_TORCH_RUNTIME_DIR}/*.dll")
endif()

add_test(
    NAME FoundationInstallConsumer
    COMMAND ${CMAKE_COMMAND}
        -DJYPPX_BUILD_DIR=${CMAKE_BINARY_DIR}
        -DJYPPX_CONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/install/foundation_consumer
        -DJYPPX_TEST_ROOT=${CMAKE_BINARY_DIR}/_fc/$<CONFIG>
        -DJYPPX_TEST_CONFIG=$<CONFIG>
        -DJYPPX_TORCH_DIR=${Torch_DIR}
        -DJYPPX_TORCH_RUNTIME_DIR=${JYPPX_TORCH_RUNTIME_DIR}
        -P ${CMAKE_SOURCE_DIR}/cmake/TestInstalledFoundation.cmake
)
set_tests_properties(FoundationInstallConsumer PROPERTIES
    LABELS "foundation;packaging"
    RESOURCE_LOCK jyppx_install_consumer
)

# Adds a GoogleTest executable linked against the framework and registered
# with CTest. Sources are passed as extra arguments.
function(jyppx_add_gtest test_name)
    cmake_parse_arguments(JYPPX_TEST "" "LINK_TARGET" "" ${ARGN})
    if(NOT JYPPX_TEST_LINK_TARGET)
        set(JYPPX_TEST_LINK_TARGET jyppx_training_framework)
    endif()
    add_executable(${test_name} ${JYPPX_TEST_UNPARSED_ARGUMENTS})
    target_link_libraries(${test_name} PRIVATE ${JYPPX_TEST_LINK_TARGET} GTest::gtest_main)
    # Expose tests/test_utils/ to test sources via "test_utils/..." includes.
    target_include_directories(${test_name} PRIVATE "${JYPPX_TESTS_SOURCE_DIR}")

    if(WIN32 AND JYPPX_TORCH_RUNTIME_DLLS)
        add_custom_command(TARGET ${test_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${JYPPX_TORCH_RUNTIME_DLLS}
                "$<TARGET_FILE_DIR:${test_name}>")
    endif()

    gtest_discover_tests(${test_name} DISCOVERY_MODE PRE_TEST)
endfunction()

add_subdirectory(unit)
add_subdirectory(integration)

if(JYPPX_BUILD_BENCHMARKS)
    # Benchmarks are executables linked like tests but not registered with
    # CTest; they are run manually.
    function(jyppx_add_bench bench_name)
        add_executable(${bench_name} ${ARGN})
        target_link_libraries(${bench_name} PRIVATE jyppx_training_framework)
        if(WIN32 AND JYPPX_TORCH_RUNTIME_DLLS)
            add_custom_command(TARGET ${bench_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${JYPPX_TORCH_RUNTIME_DLLS}
                    "$<TARGET_FILE_DIR:${bench_name}>")
        endif()
    endfunction()

    add_subdirectory(benchmarks)
endif()
