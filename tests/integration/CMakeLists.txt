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
endif()
