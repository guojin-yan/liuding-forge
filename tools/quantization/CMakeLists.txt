# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.

include(GNUInstallDirs)

if(JYPPX_LIUDING_BUILD_TOOLS)
    add_executable(jyppx_int8_prepare prepare_static_int8_linear.cpp)
    set_target_properties(jyppx_int8_prepare PROPERTIES
        OUTPUT_NAME liuding-forge-int8-prepare)
    target_compile_features(jyppx_int8_prepare PRIVATE cxx_std_20)
    target_link_libraries(jyppx_int8_prepare PRIVATE jyppx::liuding::quantization)

    install(TARGETS jyppx_int8_prepare RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    if(TARGET jyppx::liuding::onnx_export)
        add_executable(jyppx_int8_export export_static_int8_linear.cpp)
        set_target_properties(jyppx_int8_export PROPERTIES
            OUTPUT_NAME liuding-forge-int8-export)
        target_compile_features(jyppx_int8_export PRIVATE cxx_std_20)
        target_link_libraries(jyppx_int8_export PRIVATE
            jyppx::liuding::onnx_export jyppx::liuding::quantization)
        jyppx_add_torch_runtime_dependency(jyppx_int8_export)
        install(TARGETS jyppx_int8_export RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
endif()
if(JYPPX_LIUDING_BUILD_TESTS)
    add_executable(jyppx_int8_fashion_fixture
        ${PROJECT_SOURCE_DIR}/tests/quantization/prepare_fashion_linear_fixture.cpp)
    set_target_properties(jyppx_int8_fashion_fixture PROPERTIES
        OUTPUT_NAME liuding-forge-int8-fashion-fixture)
    target_compile_features(jyppx_int8_fashion_fixture PRIVATE cxx_std_20)
    target_link_libraries(jyppx_int8_fashion_fixture PRIVATE jyppx::liuding::quantization)
    if(TARGET jyppx::liuding::onnx_runtime)
        add_executable(jyppx_int8_task
            ${PROJECT_SOURCE_DIR}/tests/quantization/run_static_int8_task.cpp)
        set_target_properties(jyppx_int8_task PROPERTIES
            OUTPUT_NAME liuding-forge-int8-task)
        target_compile_features(jyppx_int8_task PRIVATE cxx_std_20)
        target_link_libraries(jyppx_int8_task PRIVATE jyppx::liuding::onnx_runtime)
        jyppx_add_torch_runtime_dependency(jyppx_int8_task)
        jyppx_add_onnx_runtime_dependency(jyppx_int8_task)
    endif()
endif()

# Test-only native FP32/INT8 comparison; the external driver owns full sampling.
if(JYPPX_LIUDING_BUILD_TESTS AND TARGET jyppx::liuding::onnx_export
   AND TARGET jyppx::liuding::onnx_runtime)
    add_executable(jyppx_int8_frozen_compare
        "${PROJECT_SOURCE_DIR}/tests/quantization/benchmark_frozen_linear.cpp")
    set_target_properties(jyppx_int8_frozen_compare PROPERTIES
        OUTPUT_NAME liuding-forge-int8-frozen-compare)
    target_compile_features(jyppx_int8_frozen_compare PRIVATE cxx_std_20)
    target_include_directories(jyppx_int8_frozen_compare PRIVATE
        "${PROJECT_SOURCE_DIR}/tools/quantization")
    target_link_libraries(jyppx_int8_frozen_compare PRIVATE
        jyppx::liuding::onnx_export jyppx::liuding::onnx_runtime
        nlohmann_json::nlohmann_json)
    if(WIN32)
        target_link_libraries(jyppx_int8_frozen_compare PRIVATE Psapi)
    endif()
    jyppx_add_torch_runtime_dependency(jyppx_int8_frozen_compare)
    jyppx_add_onnx_runtime_dependency(jyppx_int8_frozen_compare)
endif()
