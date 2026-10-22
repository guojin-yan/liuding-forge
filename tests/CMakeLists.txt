# JYPPX Training Framework tests.
#
# GoogleTest is fetched from the local third_party/downloads cache when
# available, falling back to the network. Tests are registered with CTest
# through gtest_discover_tests (PRE_TEST keeps discovery at ctest time, after
# the shared runtime-copy dependency has placed the LibTorch DLLs next to the
# test executables).

include(FetchContent)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(
        NAME CompatibilitySource
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/validation/compatibility.py
                --file ${CMAKE_SOURCE_DIR}/compatibility.json --validate
    )
    set_tests_properties(CompatibilitySource PROPERTIES LABELS "hygiene;compatibility")
    add_test(
        NAME CompatibilityToolUnitTests
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/tests/scripts/test_compatibility.py
    )
    set_tests_properties(CompatibilityToolUnitTests PROPERTIES
        LABELS "hygiene;compatibility"
    )
    add_test(
        NAME DependencyManifest
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/validation/dependency_manifest.py
                --file ${CMAKE_SOURCE_DIR}/dependency-manifest.json
    )
    set_tests_properties(DependencyManifest PROPERTIES LABELS "hygiene;dependency")
    add_test(
        NAME DependencyManifestToolUnitTests
        COMMAND ${Python3_EXECUTABLE} -m unittest
                ${CMAKE_SOURCE_DIR}/tests/scripts/test_dependency_manifest.py
    )
    set_tests_properties(DependencyManifestToolUnitTests PROPERTIES
        LABELS "hygiene;dependency"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    add_test(
        NAME ComponentDependencyGraph
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/scripts/validation/component_graph.py
                --manifest ${CMAKE_SOURCE_DIR}/component-dependencies.json
                --source-root ${CMAKE_SOURCE_DIR}
                --target-graph ${JYPPX_COMPONENT_TARGET_GRAPH}
                --output ${CMAKE_CURRENT_BINARY_DIR}/component-dependency-report.json
    )
    set_tests_properties(ComponentDependencyGraph PROPERTIES
        LABELS "hygiene;dependency;architecture"
    )
    add_test(
        NAME ComponentDependencyGraphToolUnitTests
        COMMAND ${Python3_EXECUTABLE} -m unittest
                ${CMAKE_SOURCE_DIR}/tests/scripts/test_component_graph.py
    )
    set_tests_properties(ComponentDependencyGraphToolUnitTests PROPERTIES
        LABELS "hygiene;dependency;architecture"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    add_test(
        NAME CheckpointFixtureIntegrity
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/scripts/validation/verify_checkpoint_fixtures.py
                --root ${CMAKE_SOURCE_DIR}/tests/fixtures/checkpoint
    )
    set_tests_properties(CheckpointFixtureIntegrity PROPERTIES
        LABELS "hygiene;checkpoint;compatibility"
    )
    add_test(
        NAME CoverageSummaryTool
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/validation/coverage_summary.py
                ${CMAKE_SOURCE_DIR}/tests/fixtures/coverage/summary.json
    )
    set_tests_properties(CoverageSummaryTool PROPERTIES
        LABELS "hygiene;coverage"
        PASS_REGULAR_EXPRESSION "Function: 8/10 [(]80[.]00%[)]"
    )
    add_test(
        NAME CoverageSummaryRejectsMissingFunction
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/validation/coverage_summary.py
                ${CMAKE_SOURCE_DIR}/tests/fixtures/coverage/missing-function.json
    )
    set_tests_properties(CoverageSummaryRejectsMissingFunction PROPERTIES
        LABELS "hygiene;coverage"
        WILL_FAIL TRUE
    )
    add_test(
        NAME CoverageTargetGate
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/validation/coverage_gate.py
                ${CMAKE_SOURCE_DIR}/tests/fixtures/coverage/target-summary.json
                ${CMAKE_SOURCE_DIR}/tests/fixtures/coverage/target-policy.json
                --output ${CMAKE_CURRENT_BINARY_DIR}/coverage-targets.json
    )
    set_tests_properties(CoverageTargetGate PROPERTIES
        LABELS "hygiene;coverage"
        PASS_REGULAR_EXPRESSION "training: line=18/20 [(]90[.]00%[)]"
    )
    if(UNIX)
        add_test(
            NAME CoverageLaneProtocolUnitTests
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_SOURCE_DIR}/tests/scripts/test_coverage_lane.py
        )
        set_tests_properties(CoverageLaneProtocolUnitTests PROPERTIES
            LABELS "hygiene;coverage"
        )
    endif()
    add_test(
        NAME TSanLaneProtocolUnitTests
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/tests/scripts/test_tsan_lane.py
    )
    set_tests_properties(TSanLaneProtocolUnitTests PROPERTIES
        LABELS "hygiene;sanitizer"
    )
    add_test(
        NAME CoverageTargetGateRejectsRegression
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/validation/coverage_gate.py
                ${CMAKE_SOURCE_DIR}/tests/fixtures/coverage/target-summary-regressed.json
                ${CMAKE_SOURCE_DIR}/tests/fixtures/coverage/target-policy.json
    )
    set_tests_properties(CoverageTargetGateRejectsRegression PROPERTIES
        LABELS "hygiene;coverage"
        WILL_FAIL TRUE
    )
    add_test(
        NAME CoverageTargetGateRejectsDenominatorShrink
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/validation/coverage_gate.py
                ${CMAKE_SOURCE_DIR}/tests/fixtures/coverage/target-summary-shrunk.json
                ${CMAKE_SOURCE_DIR}/tests/fixtures/coverage/target-policy.json
    )
    set_tests_properties(CoverageTargetGateRejectsDenominatorShrink PROPERTIES
        LABELS "hygiene;coverage"
        WILL_FAIL TRUE
    )
    add_test(
        NAME CoverageTargetGateRejectsUnownedFile
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/validation/coverage_gate.py
                ${CMAKE_SOURCE_DIR}/tests/fixtures/coverage/target-summary-unowned.json
                ${CMAKE_SOURCE_DIR}/tests/fixtures/coverage/target-policy.json
    )
    set_tests_properties(CoverageTargetGateRejectsUnownedFile PROPERTIES
        LABELS "hygiene;coverage"
        WILL_FAIL TRUE
    )
    add_test(
        NAME BaselineStatisticsTool
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/benchmarks/baseline_stats.py
                ${CMAKE_SOURCE_DIR}/tests/fixtures/performance/stable-samples.json
                --output ${CMAKE_CURRENT_BINARY_DIR}/baseline-statistics.json
                --reference ${CMAKE_SOURCE_DIR}/tests/fixtures/performance/reference.json
    )
    set_tests_properties(BaselineStatisticsTool PROPERTIES
        LABELS "hygiene;performance"
        PASS_REGULAR_EXPRESSION "trainer[.]throughput[.]batch_32: median=101[.]000"
    )
    add_test(
        NAME BaselineStatisticsRejectsInstability
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/benchmarks/baseline_stats.py
                ${CMAKE_SOURCE_DIR}/tests/fixtures/performance/unstable-samples.json
                --output ${CMAKE_CURRENT_BINARY_DIR}/unstable-statistics.json
    )
    set_tests_properties(BaselineStatisticsRejectsInstability PROPERTIES
        LABELS "hygiene;performance"
        WILL_FAIL TRUE
    )
    add_test(
        NAME BaselineStatisticsRejectsRegression
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/benchmarks/baseline_stats.py
                ${CMAKE_SOURCE_DIR}/tests/fixtures/performance/regressed-samples.json
                --output ${CMAKE_CURRENT_BINARY_DIR}/regressed-statistics.json
                --reference ${CMAKE_SOURCE_DIR}/tests/fixtures/performance/reference.json
    )
    set_tests_properties(BaselineStatisticsRejectsRegression PROPERTIES
        LABELS "hygiene;performance"
        WILL_FAIL TRUE
    )
    add_test(
        NAME CudaBaselineRunnerUnitTests
        COMMAND ${Python3_EXECUTABLE} -m unittest
                ${CMAKE_SOURCE_DIR}/tests/scripts/test_cuda_baseline.py
    )
    set_tests_properties(CudaBaselineRunnerUnitTests PROPERTIES
        LABELS "hygiene;performance;cuda"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    add_test(NAME ResourceBaselineRunnerUnitTests
        COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_SOURCE_DIR}/tests/scripts/test_resource_baseline.py")
    set_tests_properties(ResourceBaselineRunnerUnitTests PROPERTIES
        LABELS "hygiene;performance" TIMEOUT 60)
    add_test(
        NAME PublicHeaderGuards
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/validation/header_guards.py
                --root ${CMAKE_SOURCE_DIR}/include/jyppx
    )
    set_tests_properties(PublicHeaderGuards PROPERTIES LABELS "hygiene;headers")
    add_test(NAME ReleaseToolUnitTests
        COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_SOURCE_DIR}/tests/scripts/test_release.py")
    set_tests_properties(ReleaseToolUnitTests PROPERTIES
        LABELS "hygiene;release;packaging" TIMEOUT 180)
endif()

add_test(
    NAME LegacySurface
    COMMAND ${CMAKE_COMMAND}
        -DJYPPX_SOURCE_DIR=${CMAKE_SOURCE_DIR}
        -P ${CMAKE_SOURCE_DIR}/cmake/CheckLegacySurface.cmake
)
set_tests_properties(LegacySurface PROPERTIES LABELS "hygiene;migration")

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
endif()

if(TARGET jyppx::sdk)
add_test(
    NAME MigratedInstallConsumer
    COMMAND ${CMAKE_COMMAND}
        -DJYPPX_BUILD_DIR=${CMAKE_BINARY_DIR}
        -DJYPPX_CONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/install/migrated_consumer
        -DJYPPX_TEST_ROOT=${CMAKE_BINARY_DIR}/_install_consumer/$<CONFIG>
        -DJYPPX_TEST_CONFIG=$<CONFIG>
        -DJYPPX_TORCH_DIR=${Torch_DIR}
        -DJYPPX_TORCH_RUNTIME_DIR=${JYPPX_TORCH_RUNTIME_DIR}
        -DJYPPX_CUDA_STANDARD=${CMAKE_CUDA_STANDARD}
        -DJYPPX_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}
        -DJYPPX_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
        -DJYPPX_CUDA_TOOLKIT_ROOT=${CUDAToolkit_ROOT}
        -DJYPPX_TORCH_CUDA_ARCH_LIST=$ENV{TORCH_CUDA_ARCH_LIST}
        -P ${CMAKE_SOURCE_DIR}/cmake/TestInstalledTargets.cmake
)
set_tests_properties(MigratedInstallConsumer PROPERTIES
    LABELS "core;data;experiment;foundation;interop;nn;optim;training;packaging"
    RESOURCE_LOCK jyppx_install_consumer
)

add_test(
    NAME InteropInstallConsumer
    COMMAND ${CMAKE_COMMAND}
        -DJYPPX_BUILD_DIR=${CMAKE_BINARY_DIR}
        -DJYPPX_CONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/install/interop_consumer
        -DJYPPX_TEST_ROOT=${CMAKE_BINARY_DIR}/_interop_install_consumer/$<CONFIG>
        -DJYPPX_TEST_CONFIG=$<CONFIG>
        -DJYPPX_TORCH_DIR=${Torch_DIR}
        -DJYPPX_TORCH_RUNTIME_DIR=${JYPPX_TORCH_RUNTIME_DIR}
        -DJYPPX_CUDA_STANDARD=${CMAKE_CUDA_STANDARD}
        -DJYPPX_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}
        -DJYPPX_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
        -DJYPPX_CUDA_TOOLKIT_ROOT=${CUDAToolkit_ROOT}
        -DJYPPX_TORCH_CUDA_ARCH_LIST=$ENV{TORCH_CUDA_ARCH_LIST}
        -P ${CMAKE_SOURCE_DIR}/cmake/TestInstalledTargets.cmake
)
set_tests_properties(InteropInstallConsumer PROPERTIES
    LABELS "foundation;interop;packaging"
    RESOURCE_LOCK jyppx_install_consumer
)

add_test(
    NAME ExperimentInstallConsumer
    COMMAND ${CMAKE_COMMAND}
        -DJYPPX_BUILD_DIR=${CMAKE_BINARY_DIR}
        -DJYPPX_CONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/install/experiment_consumer
        -DJYPPX_TEST_ROOT=${CMAKE_BINARY_DIR}/_experiment_install_consumer/$<CONFIG>
        -DJYPPX_TEST_CONFIG=$<CONFIG>
        -DJYPPX_TORCH_DIR=${Torch_DIR}
        -DJYPPX_TORCH_RUNTIME_DIR=${JYPPX_TORCH_RUNTIME_DIR}
        -DJYPPX_CUDA_STANDARD=${CMAKE_CUDA_STANDARD}
        -DJYPPX_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}
        -DJYPPX_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
        -DJYPPX_CUDA_TOOLKIT_ROOT=${CUDAToolkit_ROOT}
        -DJYPPX_TORCH_CUDA_ARCH_LIST=$ENV{TORCH_CUDA_ARCH_LIST}
        -P ${CMAKE_SOURCE_DIR}/cmake/TestInstalledTargets.cmake
)
set_tests_properties(ExperimentInstallConsumer PROPERTIES
    LABELS "core;experiment;foundation;packaging"
    RESOURCE_LOCK jyppx_install_consumer
)

add_test(
    NAME PackageComponentConsumer
    COMMAND ${CMAKE_COMMAND}
        -DJYPPX_BUILD_DIR=${CMAKE_BINARY_DIR}
        -DJYPPX_CONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/install/package_component_consumer
        -DJYPPX_TEST_ROOT=${CMAKE_BINARY_DIR}/_package_component_consumer/$<CONFIG>
        -DJYPPX_TEST_CONFIG=$<CONFIG>
        -DJYPPX_TORCH_DIR=${Torch_DIR}
        -DJYPPX_TORCH_RUNTIME_DIR=${JYPPX_TORCH_RUNTIME_DIR}
        -DJYPPX_CUDA_STANDARD=${CMAKE_CUDA_STANDARD}
        -DJYPPX_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}
        -DJYPPX_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
        -DJYPPX_CUDA_TOOLKIT_ROOT=${CUDAToolkit_ROOT}
        -DJYPPX_TORCH_CUDA_ARCH_LIST=$ENV{TORCH_CUDA_ARCH_LIST}
        -P ${CMAKE_SOURCE_DIR}/cmake/TestInstalledTargets.cmake
)
set_tests_properties(PackageComponentConsumer PROPERTIES
    LABELS "packaging"
    RESOURCE_LOCK jyppx_install_consumer
)

add_test(
    NAME ComponentInstallMatrixConsumer
    COMMAND ${CMAKE_COMMAND}
        -DJYPPX_SOURCE_DIR=${CMAKE_SOURCE_DIR}
        -DJYPPX_BUILD_DIR=${CMAKE_BINARY_DIR}
        -DJYPPX_CONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/install/component_consumer
        -DJYPPX_TEST_ROOT=${CMAKE_BINARY_DIR}/_component_install_consumer/$<CONFIG>
        -DJYPPX_TEST_CONFIG=$<CONFIG>
        -DJYPPX_TORCH_DIR=${Torch_DIR}
        -DJYPPX_TORCH_RUNTIME_DIR=${JYPPX_TORCH_RUNTIME_DIR}
        -DJYPPX_CUDA_STANDARD=${CMAKE_CUDA_STANDARD}
        -DJYPPX_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}
        -DJYPPX_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
        -DJYPPX_CUDA_TOOLKIT_ROOT=${CUDAToolkit_ROOT}
        -DJYPPX_TORCH_CUDA_ARCH_LIST=$ENV{TORCH_CUDA_ARCH_LIST}
        -P ${CMAKE_SOURCE_DIR}/cmake/TestInstalledComponentMatrix.cmake
)
set_tests_properties(ComponentInstallMatrixConsumer PROPERTIES
    LABELS "core;data;experiment;foundation;interop;nn;optim;training;packaging"
    RESOURCE_LOCK jyppx_install_consumer
)

add_test(
    NAME OptionalComponentMatrix
    COMMAND ${CMAKE_COMMAND}
        -DJYPPX_SOURCE_DIR=${CMAKE_SOURCE_DIR}
        -DJYPPX_CONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/install/component_consumer
        -DJYPPX_TEST_ROOT=${CMAKE_SOURCE_DIR}/out/component-matrix/$<CONFIG>
        -DJYPPX_TEST_CONFIG=$<CONFIG>
        -DJYPPX_TORCH_DIR=${Torch_DIR}
        -DJYPPX_TORCH_RUNTIME_DIR=${JYPPX_TORCH_RUNTIME_DIR}
        -DJYPPX_CMAKE_GENERATOR=${CMAKE_GENERATOR}
        -DJYPPX_CMAKE_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}
        -DJYPPX_CMAKE_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}
        -DJYPPX_CUDA_STANDARD=${CMAKE_CUDA_STANDARD}
        -DJYPPX_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}
        -DJYPPX_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
        -DJYPPX_CUDA_TOOLKIT_ROOT=${CUDAToolkit_ROOT}
        -DJYPPX_TORCH_CUDA_ARCH_LIST=$ENV{TORCH_CUDA_ARCH_LIST}
        -P ${CMAKE_SOURCE_DIR}/cmake/TestOptionalComponentMatrix.cmake
)
set_tests_properties(OptionalComponentMatrix PROPERTIES
    LABELS "core;data;experiment;foundation;interop;nn;optim;training;packaging"
    RESOURCE_LOCK jyppx_component_profile_matrix
    TIMEOUT 3600
)
endif()


# Preview consumers are independent of the stable SDK aggregate.
foreach(preview_component IN ITEMS models quantization serving)
    if(TARGET jyppx::${preview_component})
        if(preview_component STREQUAL "models")
            set(preview_test_name ModelsInstallConsumer)
        elseif(preview_component STREQUAL "quantization")
            set(preview_test_name QuantizationInstallConsumer)
        else()
            set(preview_test_name ServingInstallConsumer)
        endif()
        add_test(
            NAME ${preview_test_name}
            COMMAND ${CMAKE_COMMAND}
                -DJYPPX_BUILD_DIR=${CMAKE_BINARY_DIR}
                -DJYPPX_CONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/install/${preview_component}_consumer
                -DJYPPX_TEST_ROOT=${CMAKE_BINARY_DIR}/_${preview_component}_install_consumer/$<CONFIG>
                -DJYPPX_TEST_CONFIG=$<CONFIG>
                -DJYPPX_TORCH_DIR=${Torch_DIR}
                -DJYPPX_TORCH_RUNTIME_DIR=${JYPPX_TORCH_RUNTIME_DIR}
                -DJYPPX_CUDA_STANDARD=${CMAKE_CUDA_STANDARD}
                -DJYPPX_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}
                -DJYPPX_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
                -DJYPPX_CUDA_TOOLKIT_ROOT=${CUDAToolkit_ROOT}
                -DJYPPX_TORCH_CUDA_ARCH_LIST=$ENV{TORCH_CUDA_ARCH_LIST}
                -P ${CMAKE_SOURCE_DIR}/cmake/TestInstalledTargets.cmake
        )
        set_tests_properties(${preview_test_name} PROPERTIES
            LABELS "${preview_component};preview;packaging"
            RESOURCE_LOCK jyppx_install_consumer
            TIMEOUT 900
        )
    endif()
endforeach()

if(JYPPX_BUILD_MODELS OR JYPPX_BUILD_QUANTIZATION OR JYPPX_BUILD_SERVING)
    add_test(
        NAME PreviewComponentMatrix
        COMMAND ${CMAKE_COMMAND}
            -DJYPPX_SOURCE_DIR=${CMAKE_SOURCE_DIR}
            -DJYPPX_CONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/install/component_consumer
            -DJYPPX_TEST_ROOT=${CMAKE_SOURCE_DIR}/out/preview-component-matrix/$<CONFIG>
            -DJYPPX_TEST_CONFIG=$<CONFIG>
            -DJYPPX_TORCH_DIR=${Torch_DIR}
            -DJYPPX_TORCH_RUNTIME_DIR=${JYPPX_TORCH_RUNTIME_DIR}
            -DJYPPX_CMAKE_GENERATOR=${CMAKE_GENERATOR}
            -DJYPPX_CMAKE_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}
            -DJYPPX_CMAKE_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}
            -DJYPPX_CUDA_STANDARD=${CMAKE_CUDA_STANDARD}
            -DJYPPX_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}
            -DJYPPX_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
            -DJYPPX_CUDA_TOOLKIT_ROOT=${CUDAToolkit_ROOT}
            -DJYPPX_TORCH_CUDA_ARCH_LIST=$ENV{TORCH_CUDA_ARCH_LIST}
            -DJYPPX_NM=${CMAKE_NM}
            -DJYPPX_LINKER=${CMAKE_LINKER}
            -P ${CMAKE_SOURCE_DIR}/cmake/tests/TestPreviewComponentMatrix.cmake
    )
    set_tests_properties(PreviewComponentMatrix PROPERTIES
        LABELS "models;quantization;serving;preview;packaging"
        RESOURCE_LOCK jyppx_component_profile_matrix
        TIMEOUT 10800
    )
endif()

# Adds a GoogleTest executable linked against the framework and registered
# with CTest. Sources are passed as extra arguments.
function(jyppx_add_gtest test_name)
    cmake_parse_arguments(JYPPX_TEST "" "LINK_TARGET" "ENVIRONMENT" ${ARGN})
    if(NOT JYPPX_TEST_LINK_TARGET)
        set(JYPPX_TEST_LINK_TARGET jyppx::sdk)
    endif()
    add_executable(${test_name} ${JYPPX_TEST_UNPARSED_ARGUMENTS})
    target_link_libraries(${test_name} PRIVATE ${JYPPX_TEST_LINK_TARGET} GTest::gtest_main)
    jyppx_apply_instrumentation(${test_name})
    # Expose tests/test_utils/ to test sources via "test_utils/..." includes.
    target_include_directories(${test_name} PRIVATE "${JYPPX_TESTS_SOURCE_DIR}")
    target_compile_definitions(${test_name} PRIVATE
        JYPPX_TEST_FIXTURE_ROOT="${CMAKE_SOURCE_DIR}/tests/fixtures"
    )

    jyppx_add_torch_runtime_dependency(${test_name})

    set(discovery_properties)
    if(JYPPX_TEST_ENVIRONMENT)
        list(APPEND discovery_properties PROPERTIES
             ENVIRONMENT "${JYPPX_TEST_ENVIRONMENT}")
    endif()
    gtest_discover_tests(${test_name} DISCOVERY_MODE PRE_TEST DISCOVERY_TIMEOUT 30
        ${discovery_properties})
endfunction()

add_subdirectory(unit)
if(TARGET jyppx_cli)
    add_subdirectory(cli)
endif()
if(JYPPX_TEST_OFFICIAL_TELEMETRY)
    if(NOT TARGET jyppx_cli OR NOT EXISTS "${JYPPX_TELEMETRY_PYTHON}" OR
       NOT EXISTS "${JYPPX_OTELCOL_EXECUTABLE}")
        message(FATAL_ERROR "Official telemetry acceptance requires product CLI and locked Python/collector executable paths")
    endif()
    add_test(NAME OfficialTelemetryAcceptance
        COMMAND ${CMAKE_COMMAND} -E env "JYPPX_TEST_OTELCOL=${JYPPX_OTELCOL_EXECUTABLE}"
            "JYPPX_TEST_CLI=$<TARGET_FILE:jyppx_cli>"
            "${JYPPX_TELEMETRY_PYTHON}"
            "${CMAKE_CURRENT_SOURCE_DIR}/integration/telemetry_sidecar/run_acceptance.py")
    set_tests_properties(OfficialTelemetryAcceptance PROPERTIES LABELS "experiment;official;integration" TIMEOUT 300)
endif()
if(TARGET jyppx::sdk)
    add_subdirectory(integration)
endif()

if(JYPPX_BUILD_FUZZERS)
    if(NOT TARGET jyppx::sdk)
        message(FATAL_ERROR "JYPPX_BUILD_FUZZERS=ON requires all Core components")
    endif()
    add_subdirectory(fuzz)
endif()

if(JYPPX_BUILD_BENCHMARKS)
    if(NOT TARGET jyppx::sdk)
        message(FATAL_ERROR "JYPPX_BUILD_BENCHMARKS=ON requires all Core components")
    endif()
    # Benchmarks are executables linked like tests but not registered with
    # CTest; they are run manually.
    function(jyppx_add_bench bench_name)
        cmake_parse_arguments(JYPPX_BENCH "" "LINK_TARGET" "" ${ARGN})
        if(NOT JYPPX_BENCH_LINK_TARGET)
            set(JYPPX_BENCH_LINK_TARGET jyppx::sdk)
        endif()
        add_executable(${bench_name} ${JYPPX_BENCH_UNPARSED_ARGUMENTS})
        target_link_libraries(${bench_name} PRIVATE ${JYPPX_BENCH_LINK_TARGET})
        jyppx_apply_instrumentation(${bench_name})
        jyppx_add_torch_runtime_dependency(${bench_name})
    endfunction()

    add_subdirectory(benchmarks)
endif()
