# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.

include(GNUInstallDirs)

# Private native tool implementation; no Python or model execution.
add_library(jyppx_dinov2_conversion STATIC dinov2_conversion.cpp)
set_target_properties(jyppx_dinov2_conversion PROPERTIES
    OUTPUT_NAME liuding-forge-dinov2-conversion)
target_compile_features(jyppx_dinov2_conversion PUBLIC cxx_std_20)
target_include_directories(jyppx_dinov2_conversion PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(jyppx_dinov2_conversion PUBLIC
    jyppx::liuding::interop jyppx::liuding::foundation nlohmann_json::nlohmann_json)

add_executable(jyppx_convert_dinov2_hf convert_dinov2_hf.cpp)
set_target_properties(jyppx_convert_dinov2_hf PROPERTIES
    OUTPUT_NAME liuding-forge-convert-dinov2-hf)
target_link_libraries(jyppx_convert_dinov2_hf PRIVATE jyppx_dinov2_conversion)
jyppx_add_torch_runtime_dependency(jyppx_convert_dinov2_hf)
install(TARGETS jyppx_convert_dinov2_hf RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
