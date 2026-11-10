# Integration tests exercise worker pools and explicit pipeline epochs.

jyppx_add_gtest(jyppx_test_large_dataset test_large_dataset.cpp)
jyppx_add_gtest(jyppx_test_concurrent_stress test_concurrent_stress.cpp)

if(TARGET jyppx::liuding::data_image AND TARGET jyppx::liuding::experiment)
    jyppx_add_gtest(jyppx_test_image_output_evidence
        LINK_TARGET jyppx::liuding::data_image
        test_image_output_evidence.cpp)
    target_link_libraries(jyppx_test_image_output_evidence PRIVATE
        jyppx::liuding::experiment)
    if(Python3_Interpreter_FOUND)
        add_executable(jyppx_image_output_recovery_child
            experiment/image_output_recovery_child.cpp)
        target_link_libraries(jyppx_image_output_recovery_child PRIVATE
            jyppx::liuding::data_image jyppx::liuding::experiment
            nlohmann_json::nlohmann_json)
        jyppx_apply_instrumentation(jyppx_image_output_recovery_child)
        jyppx_add_torch_runtime_dependency(jyppx_image_output_recovery_child)
        if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
            target_sources(jyppx_image_output_recovery_child PRIVATE
                experiment/image_output_native_hooks.cpp)
            target_compile_definitions(jyppx_image_output_recovery_child PRIVATE
                LIUDING_OUTPUT_NATIVE_FAULTS=1)
            target_link_options(jyppx_image_output_recovery_child PRIVATE
                -Wl,--wrap=open -Wl,--wrap=write -Wl,--wrap=fsync
                -Wl,--wrap=linkat -Wl,--wrap=link -Wl,--wrap=close)
            add_test(NAME ImageOutputStagingRecovery
                COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/experiment/test_image_output_recovery.py"
                    --child $<TARGET_FILE:jyppx_image_output_recovery_child>
                    --work-dir "${CMAKE_BINARY_DIR}/_image_recovery/$<CONFIG>"
                    --suite staging)
            add_test(NAME ImageOutputStagingFaults
                COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/experiment/test_image_output_recovery.py"
                    --child $<TARGET_FILE:jyppx_image_output_recovery_child>
                    --work-dir "${CMAKE_BINARY_DIR}/_image_recovery/$<CONFIG>"
                    --suite faults)
            set_tests_properties(ImageOutputStagingRecovery ImageOutputStagingFaults
                PROPERTIES LABELS "data_image;experiment;integration;process" TIMEOUT 120)
        elseif(WIN32 AND MSVC AND CMAKE_SIZEOF_VOID_P EQUAL 8)
            add_library(jyppx_image_output_staging_test_impl OBJECT
                "${PROJECT_SOURCE_DIR}/src/jyppx/liuding/data_image/image_output_sink.cpp")
            target_link_libraries(jyppx_image_output_staging_test_impl PRIVATE
                jyppx::liuding::data_image)
            target_compile_options(jyppx_image_output_staging_test_impl PRIVATE
                "/FI${CMAKE_CURRENT_SOURCE_DIR}/experiment/image_output_windows_native_hooks.hpp")
            jyppx_apply_instrumentation(jyppx_image_output_staging_test_impl)
            target_sources(jyppx_image_output_recovery_child PRIVATE
                experiment/image_output_windows_native_hooks.cpp
                $<TARGET_OBJECTS:jyppx_image_output_staging_test_impl>)
            target_compile_definitions(jyppx_image_output_recovery_child PRIVATE
                LIUDING_OUTPUT_NATIVE_FAULTS=1)
            add_test(NAME ImageOutputStagingRecovery
                COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/experiment/test_image_output_recovery.py"
                    --child $<TARGET_FILE:jyppx_image_output_recovery_child>
                    --work-dir "${CMAKE_BINARY_DIR}/_image_recovery/$<CONFIG>"
                    --suite staging)
            add_test(NAME ImageOutputStagingFaults
                COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/experiment/test_image_output_recovery.py"
                    --child $<TARGET_FILE:jyppx_image_output_recovery_child>
                    --work-dir "${CMAKE_BINARY_DIR}/_image_recovery/$<CONFIG>"
                    --suite faults)
            set_tests_properties(ImageOutputStagingRecovery ImageOutputStagingFaults
                PROPERTIES LABELS "data_image;experiment;integration;process" TIMEOUT 120)
        endif()
        add_test(NAME ImageOutputProcessRecovery
            COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/experiment/test_image_output_recovery.py"
                --child $<TARGET_FILE:jyppx_image_output_recovery_child>
                --work-dir "${CMAKE_BINARY_DIR}/_image_recovery/$<CONFIG>")
        set_tests_properties(ImageOutputProcessRecovery PROPERTIES
            LABELS "data_image;experiment;integration;process" TIMEOUT 120)
    endif()
endif()

# A real process boundary is required to validate durable outbox replay after
# SQLite commits but the producer has not received its receipt.
if(Python3_Interpreter_FOUND)
    add_executable(jyppx_liuding_experiment_recovery_child experiment/recovery_child.cpp)
    target_link_libraries(jyppx_liuding_experiment_recovery_child PRIVATE
        jyppx::liuding::training jyppx::liuding::experiment jyppx::liuding::interop
        nlohmann_json::nlohmann_json)
    jyppx_apply_instrumentation(jyppx_liuding_experiment_recovery_child)
    jyppx_add_torch_runtime_dependency(jyppx_liuding_experiment_recovery_child)
    add_test(NAME ExperimentProcessRecovery
        COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/experiment/test_process_recovery.py"
            --child $<TARGET_FILE:jyppx_liuding_experiment_recovery_child>
            --work-dir "${CMAKE_CURRENT_BINARY_DIR}/process-recovery/$<CONFIG>")
    set_tests_properties(ExperimentProcessRecovery PROPERTIES
        LABELS "training;experiment;checkpoint;integration;process"
        TIMEOUT 300)

    add_executable(jyppx_storage_crash_child experiment/storage_crash_child.cpp)
    target_link_libraries(jyppx_storage_crash_child PRIVATE
        jyppx::liuding::experiment nlohmann_json::nlohmann_json)
    # experiment embeds the pinned amalgamation. Only add its private header;
    # linking a second SQLite runtime would invalidate connection hooks.
    target_include_directories(jyppx_storage_crash_child SYSTEM PRIVATE
        "${jyppx_sqlite_SOURCE_DIR}")
    jyppx_apply_instrumentation(jyppx_storage_crash_child)
    jyppx_add_torch_runtime_dependency(jyppx_storage_crash_child)
    # The scenarios add Unicode directories and SHA-256 object names. Keep
    # staging shallow so a fresh validation tree fits Windows MAX_PATH.
    add_test(NAME ExperimentStorageCrashBoundaries
        COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/experiment/test_storage_crash_boundaries.py"
            --child $<TARGET_FILE:jyppx_storage_crash_child>
            --work-dir "${CMAKE_BINARY_DIR}/_storage_crash/$<CONFIG>")
    set_tests_properties(ExperimentStorageCrashBoundaries PROPERTIES
        LABELS "experiment;checkpoint;integration;process" TIMEOUT 600)

    get_target_property(experiment_library_type jyppx_liuding_experiment TYPE)
    if(NOT experiment_library_type STREQUAL "STATIC_LIBRARY")
        message(FATAL_ERROR "CAS crash supervision requires the static experiment archive")
    endif()
    add_executable(jyppx_cas_crash_child experiment/cas_crash_child.cpp)
    target_link_libraries(jyppx_cas_crash_child PRIVATE
        jyppx::liuding::experiment nlohmann_json::nlohmann_json)
    target_include_directories(jyppx_cas_crash_child PRIVATE "${PROJECT_SOURCE_DIR}/src")
    jyppx_apply_instrumentation(jyppx_cas_crash_child)
    jyppx_add_torch_runtime_dependency(jyppx_cas_crash_child)
    add_test(NAME ExperimentCasCrashBoundaries
        COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/experiment/test_cas_crash_boundaries.py"
            --child $<TARGET_FILE:jyppx_cas_crash_child>
            --work-dir "${CMAKE_BINARY_DIR}/_cas_crash/$<CONFIG>")
    set_tests_properties(ExperimentCasCrashBoundaries PROPERTIES
        LABELS "experiment;checkpoint;integration;process" TIMEOUT 300)

    add_executable(jyppx_published_cas_recovery_child
        experiment/published_cas_recovery_child.cpp)
    target_link_libraries(jyppx_published_cas_recovery_child PRIVATE
        jyppx::liuding::experiment nlohmann_json::nlohmann_json)
    target_include_directories(jyppx_published_cas_recovery_child PRIVATE
        "${PROJECT_SOURCE_DIR}/src")
    target_include_directories(jyppx_published_cas_recovery_child SYSTEM PRIVATE
        "${jyppx_sqlite_SOURCE_DIR}")
    jyppx_apply_instrumentation(jyppx_published_cas_recovery_child)
    jyppx_add_torch_runtime_dependency(jyppx_published_cas_recovery_child)
    add_test(NAME PublishedCasProcessRecovery
        COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/experiment/test_published_cas_recovery.py"
            --child $<TARGET_FILE:jyppx_published_cas_recovery_child>
            --work-dir "${CMAKE_BINARY_DIR}/_published_cas/$<CONFIG>")
    set_tests_properties(PublishedCasProcessRecovery PROPERTIES
        LABELS "experiment;integration;process" TIMEOUT 180)
endif()
