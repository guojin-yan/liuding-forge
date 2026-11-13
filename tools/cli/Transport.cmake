# Copyright 2026 JYPPX Training Framework Contributors
# Licensed under the Apache License, Version 2.0.
if(NOT TARGET jyppx_cli_transport)
    find_package(Threads REQUIRED)
    add_library(jyppx_cli_transport STATIC
        "${CMAKE_CURRENT_LIST_DIR}/sidecar_process.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/native_telemetry_protocol.cpp")
    set_target_properties(jyppx_cli_transport PROPERTIES OUTPUT_NAME liuding-forge-cli-transport)
    target_include_directories(jyppx_cli_transport PUBLIC "${CMAKE_CURRENT_LIST_DIR}")
    target_link_libraries(jyppx_cli_transport PUBLIC jyppx::liuding::experiment Threads::Threads)
    target_link_libraries(jyppx_cli_transport PRIVATE nlohmann_json::nlohmann_json)
endif()
