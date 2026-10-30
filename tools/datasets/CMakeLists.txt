# Product tools remain outside the public library component graph.
add_library(jyppx_dataset_preparation STATIC paired_affine.cpp)
set_target_properties(jyppx_dataset_preparation PROPERTIES
    OUTPUT_NAME liuding-forge-dataset-preparation)
target_compile_features(jyppx_dataset_preparation PRIVATE cxx_std_20)
target_compile_definitions(jyppx_dataset_preparation PRIVATE
    JYPPX_LIUDING_SOURCE_ROOT="${PROJECT_SOURCE_DIR}")
target_link_libraries(jyppx_dataset_preparation PUBLIC jyppx::liuding::foundation)
target_link_libraries(jyppx_dataset_preparation PRIVATE nlohmann_json::nlohmann_json)
