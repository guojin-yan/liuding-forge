# Product tools are composed here, outside the library component graph.
find_package(Threads REQUIRED)
add_library(jyppx_cli_transport STATIC sidecar_process.cpp)
set_target_properties(jyppx_cli_transport PROPERTIES
    OUTPUT_NAME liuding-forge-cli-transport)
target_include_directories(jyppx_cli_transport PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(jyppx_cli_transport PUBLIC jyppx::liuding::experiment Threads::Threads)
target_link_libraries(jyppx_cli_transport PRIVATE nlohmann_json::nlohmann_json)

add_executable(jyppx_cli
    cli_main.cpp
    api_migration.cpp
    config_migration.cpp
    doctor_config.cpp
    evaluation_data.cpp
    model_checkpoint.cpp
    offline_workflow.cpp
    image_recipe.cpp
    run_commands.cpp
    telemetry_commands.cpp
)
set_target_properties(jyppx_cli PROPERTIES OUTPUT_NAME liuding-forge)
target_link_libraries(jyppx_cli PRIVATE
    jyppx::liuding::sdk jyppx_cli_transport nlohmann_json::nlohmann_json Threads::Threads)
jyppx_add_torch_runtime_dependency(jyppx_cli)
if(TARGET jyppx::liuding::models AND TARGET jyppx::liuding::data_image)
    target_link_libraries(jyppx_cli PRIVATE jyppx::liuding::models jyppx::liuding::data_image)
    target_compile_definitions(jyppx_cli PRIVATE JYPPX_LIUDING_CLI_IMAGE_RECIPE=1)
    install(FILES
        "${PROJECT_SOURCE_DIR}/recipes/image-classification/resnet18.json"
        "${PROJECT_SOURCE_DIR}/recipes/image-classification/cifar10-resnet18.json"
        DESTINATION share/jyppx/recipes/image-classification)
endif()
install(TARGETS jyppx_cli RUNTIME DESTINATION bin)

if(JYPPX_LIUDING_INSTALL_TELEMETRY_SIDECAR)
    install(FILES
        "${PROJECT_SOURCE_DIR}/tools/telemetry_sidecar/jyppx_telemetry_sidecar.py"
        "${PROJECT_SOURCE_DIR}/tools/telemetry_sidecar/requirements.lock"
        "${PROJECT_SOURCE_DIR}/tools/telemetry_sidecar/dependency-metadata.json"
        "${PROJECT_SOURCE_DIR}/tools/telemetry_sidecar/README.txt"
        DESTINATION share/jyppx/telemetry_sidecar)
endif()
