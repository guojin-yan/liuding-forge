cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(jyppx_liuding_vision_render_consumer LANGUAGES CXX)

set(JYPPX_CONSUMER_COMPONENT vision_render)
enable_testing()
add_subdirectory(../component_consumer component)
