cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(jyppx_liuding_bundle_signing_consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(jyppx CONFIG REQUIRED COMPONENTS bundle_signing)
if(NOT TARGET jyppx::liuding::bundle_signing OR NOT jyppx_bundle_signing_FOUND)
    message(FATAL_ERROR "Installed package does not satisfy component bundle_signing")
endif()

add_executable(jyppx_liuding_bundle_signing_consumer main.cpp)
target_link_libraries(jyppx_liuding_bundle_signing_consumer PRIVATE
    jyppx::liuding::bundle_signing)

# Use the caller's installed-runtime environment; do not copy SDK DLL trees.
enable_testing()
add_test(NAME Runs COMMAND jyppx_liuding_bundle_signing_consumer)
set_tests_properties(Runs PROPERTIES TIMEOUT 60)
