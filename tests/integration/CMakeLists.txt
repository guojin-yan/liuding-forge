# Integration tests exercise worker pools and explicit pipeline epochs.

jyppx_add_gtest(jyppx_test_large_dataset test_large_dataset.cpp)
jyppx_add_gtest(jyppx_test_concurrent_stress test_concurrent_stress.cpp)

# A real process boundary is required to validate durable outbox replay after
# SQLite commits but the producer has not received its receipt.
if(Python3_Interpreter_FOUND)
    add_executable(jyppx_experiment_recovery_child experiment/recovery_child.cpp)
    target_link_libraries(jyppx_experiment_recovery_child PRIVATE
        jyppx::training jyppx::experiment jyppx::interop
        nlohmann_json::nlohmann_json)
    jyppx_apply_instrumentation(jyppx_experiment_recovery_child)
    jyppx_add_torch_runtime_dependency(jyppx_experiment_recovery_child)
    add_test(NAME ExperimentProcessRecovery
        COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/experiment/test_process_recovery.py"
            --child $<TARGET_FILE:jyppx_experiment_recovery_child>
            --work-dir "${CMAKE_CURRENT_BINARY_DIR}/process-recovery/$<CONFIG>")
    set_tests_properties(ExperimentProcessRecovery PROPERTIES
        LABELS "training;experiment;checkpoint;integration;process"
        TIMEOUT 300)

    add_executable(jyppx_storage_crash_child experiment/storage_crash_child.cpp)
    target_link_libraries(jyppx_storage_crash_child PRIVATE
        jyppx::experiment nlohmann_json::nlohmann_json)
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

    get_target_property(experiment_library_type jyppx_experiment TYPE)
    if(NOT experiment_library_type STREQUAL "STATIC_LIBRARY")
        message(FATAL_ERROR "CAS crash supervision requires the static experiment archive")
    endif()
    add_executable(jyppx_cas_crash_child experiment/cas_crash_child.cpp)
    target_link_libraries(jyppx_cas_crash_child PRIVATE
        jyppx::experiment nlohmann_json::nlohmann_json)
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
endif()
