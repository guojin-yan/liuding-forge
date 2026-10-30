# Product tools remain outside the public library component graph.
add_library(jyppx_dataset_preparation STATIC paired_affine.cpp preparation_support.cpp)
set_target_properties(jyppx_dataset_preparation PROPERTIES
    OUTPUT_NAME liuding-forge-dataset-preparation)
target_compile_features(jyppx_dataset_preparation PRIVATE cxx_std_20)
target_compile_definitions(jyppx_dataset_preparation PRIVATE
    JYPPX_LIUDING_SOURCE_ROOT="${PROJECT_SOURCE_DIR}")
target_link_libraries(jyppx_dataset_preparation PUBLIC jyppx::liuding::foundation)
target_link_libraries(jyppx_dataset_preparation PRIVATE nlohmann_json::nlohmann_json)

option(JYPPX_LIUDING_PREPARE_FASHION_MNIST
    "Enable offline native Fashion-MNIST preparation (explicit dataset codec SDK)" OFF)
option(JYPPX_LIUDING_PREPARE_CIFAR10
    "Enable offline native CIFAR-10 preparation (explicit dataset codec SDK)" OFF)
option(JYPPX_LIUDING_PREPARE_MNIST
    "Enable offline native MNIST IDX preparation (explicit dataset codec SDK)" OFF)
if(JYPPX_LIUDING_PREPARE_FASHION_MNIST OR JYPPX_LIUDING_PREPARE_CIFAR10 OR
   JYPPX_LIUDING_PREPARE_MNIST)
    include("${PROJECT_SOURCE_DIR}/cmake/DatasetZlib.cmake")
    target_sources(jyppx_dataset_preparation PRIVATE codec_support.cpp)
    target_link_libraries(jyppx_dataset_preparation PRIVATE jyppx_dataset_zlib)
endif()
if(JYPPX_LIUDING_PREPARE_FASHION_MNIST)
    target_sources(jyppx_dataset_preparation PRIVATE fashion_mnist.cpp)
    target_compile_definitions(jyppx_dataset_preparation PRIVATE
        JYPPX_LIUDING_FASHION_ZLIB_SOURCE_SHA256="${JYPPX_LIUDING_DATASET_ZLIB_SOURCE_SHA256}"
        JYPPX_LIUDING_FASHION_ZLIB_LIBRARY_SHA256="${JYPPX_LIUDING_DATASET_ZLIB_LIBRARY_SHA256}")
else()
    target_sources(jyppx_dataset_preparation PRIVATE fashion_mnist_unavailable.cpp)
endif()
if(JYPPX_LIUDING_PREPARE_CIFAR10)
    target_sources(jyppx_dataset_preparation PRIVATE cifar10.cpp)
else()
    target_sources(jyppx_dataset_preparation PRIVATE cifar10_unavailable.cpp)
endif()

if(JYPPX_LIUDING_PREPARE_MNIST)
    target_sources(jyppx_dataset_preparation PRIVATE mnist.cpp)
else()
    target_sources(jyppx_dataset_preparation PRIVATE mnist_unavailable.cpp)
endif()


option(JYPPX_LIUDING_DOWNLOAD_MNIST
    "Enable explicit native MNIST HTTPS downloads using the system TLS SDK" OFF)
if(JYPPX_LIUDING_DOWNLOAD_MNIST)
    if(NOT JYPPX_LIUDING_PREPARE_MNIST)
        message(FATAL_ERROR "MNIST HTTPS download requires JYPPX_LIUDING_PREPARE_MNIST=ON")
    endif()
    target_sources(jyppx_dataset_preparation PRIVATE mnist_download.cpp)
    include("${PROJECT_SOURCE_DIR}/cmake/MnistHttps.cmake")
else()
    target_sources(jyppx_dataset_preparation PRIVATE mnist_download_unavailable.cpp)
endif()
