cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(jyppx_liuding_data_image_consumer LANGUAGES CXX)

set(JYPPX_CONSUMER_COMPONENT data_image)
enable_testing()
add_subdirectory(../component_consumer component)
