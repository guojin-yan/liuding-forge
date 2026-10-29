# Product tools are composed here, outside the library component graph.
find_package(Threads REQUIRED)
add_library(jyppx_cli_transport STATIC sidecar_process.cpp)
set_target_properties(jyppx_cli_transport PROPERTIES
    OUTPUT_NAME liuding-forge-cli-transport)
target_include_directories(jyppx_cli_transport PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(jyppx_cli_transport PUBLIC jyppx::liuding::experiment Threads::Threads)
target_link_libraries(jyppx_cli_transport PRIVATE nlohmann_json::nlohmann_json)

add_library(jyppx_cli_objects OBJECT
    cli_main.cpp
    api_migration.cpp
    config_migration.cpp
    doctor_config.cpp
    evaluation_data.cpp
    model_checkpoint.cpp
    onnx_delivery.cpp
    onnx_verification.cpp
    onnx_reconstruction.cpp
    local_attempt.cpp
    local_search.cpp
    offline_workflow.cpp
    image_recipe.cpp
    transfer_recipe.cpp
    paired_recipe.cpp
    run_commands.cpp
    telemetry_commands.cpp
)
add_executable(jyppx_cli)
set_target_properties(jyppx_cli PROPERTIES OUTPUT_NAME liuding-forge)
target_link_libraries(jyppx_cli_objects PUBLIC
    jyppx::liuding::sdk jyppx_cli_transport nlohmann_json::nlohmann_json Threads::Threads)
target_link_libraries(jyppx_cli PRIVATE jyppx_cli_objects)
jyppx_add_torch_runtime_dependency(jyppx_cli)
if(TARGET jyppx::liuding::onnx_export)
    target_link_libraries(jyppx_cli_objects PUBLIC jyppx::liuding::onnx_export)
    target_compile_definitions(jyppx_cli_objects PRIVATE JYPPX_LIUDING_CLI_ONNX_EXPORT=1)
endif()
if(TARGET jyppx::liuding::onnx_runtime)
    target_link_libraries(jyppx_cli_objects PUBLIC jyppx::liuding::onnx_runtime)
    set_property(SOURCE onnx_verification.cpp onnx_reconstruction.cpp APPEND PROPERTY
        COMPILE_DEFINITIONS JYPPX_LIUDING_CLI_ONNX_RUNTIME=1)
    jyppx_add_onnx_runtime_dependency(jyppx_cli)
endif()
if(TARGET jyppx::liuding::models)
    target_link_libraries(jyppx_cli_objects PUBLIC jyppx::liuding::models)
    set_property(SOURCE onnx_reconstruction.cpp APPEND PROPERTY
        COMPILE_DEFINITIONS JYPPX_LIUDING_CLI_ONNX_MODELS=1)
endif()
if(TARGET jyppx::liuding::models AND TARGET jyppx::liuding::data_image)
    target_link_libraries(jyppx_cli_objects PUBLIC jyppx::liuding::models jyppx::liuding::data_image)
    target_compile_definitions(jyppx_cli_objects PRIVATE JYPPX_LIUDING_CLI_IMAGE_RECIPE=1)
    install(FILES
        "${PROJECT_SOURCE_DIR}/recipes/image-classification/resnet18.json"
        "${PROJECT_SOURCE_DIR}/recipes/image-classification/cifar10-resnet18.json"
        DESTINATION share/jyppx/recipes/image-classification)
    install(FILES
        "${PROJECT_SOURCE_DIR}/recipes/transfer-learning/resnet50.json"
        "${PROJECT_SOURCE_DIR}/recipes/transfer-learning/dinov2-small.json"
        "${PROJECT_SOURCE_DIR}/recipes/transfer-learning/resnet50-a1.weights.json"
        "${PROJECT_SOURCE_DIR}/recipes/transfer-learning/dinov2-small.weights.json"
        DESTINATION share/jyppx/recipes/transfer-learning)
endif()
if(JYPPX_LIUDING_BUILD_TESTS)
    add_executable(jyppx_cli_recipe_gate EXCLUDE_FROM_ALL)
    set_target_properties(jyppx_cli_recipe_gate PROPERTIES OUTPUT_NAME liuding-forge-recipe-gate)
    target_link_libraries(jyppx_cli_recipe_gate PRIVATE jyppx_cli_objects)
    jyppx_add_torch_runtime_dependency(jyppx_cli_recipe_gate)
    if(TARGET jyppx::liuding::onnx_runtime)
        jyppx_add_onnx_runtime_dependency(jyppx_cli_recipe_gate)
    endif()
endif()
install(FILES "${PROJECT_SOURCE_DIR}/recipes/multi-io/paired-affine.json"
    DESTINATION share/jyppx/recipes/multi-io)
install(TARGETS jyppx_cli RUNTIME DESTINATION bin)

if(JYPPX_LIUDING_INSTALL_TELEMETRY_SIDECAR)
    install(FILES
        "${PROJECT_SOURCE_DIR}/tools/telemetry_sidecar/jyppx_telemetry_sidecar.py"
        "${PROJECT_SOURCE_DIR}/tools/telemetry_sidecar/requirements.lock"
        "${PROJECT_SOURCE_DIR}/tools/telemetry_sidecar/dependency-metadata.json"
        "${PROJECT_SOURCE_DIR}/tools/telemetry_sidecar/README.txt"
        DESTINATION share/jyppx/telemetry_sidecar)
endif()
