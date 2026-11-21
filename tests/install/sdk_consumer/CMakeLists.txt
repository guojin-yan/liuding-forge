cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(jyppx_sdk_consumer LANGUAGES CXX)

enable_testing()

set(JYPPX_CONSUMER_COMPONENT sdk)
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../component_consumer" component)
