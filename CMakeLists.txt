cmake_minimum_required(VERSION 3.0)

set(PROJECT_NAME onnxtest)

project(${PROJECT_NAME})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 17)

add_definitions(-D NOMINMAX)

add_subdirectory("FBAMatting")
add_subdirectory("EnlightenGAN")
add_subdirectory("DeepLabv3Plus")

include_directories(
    $ENV{DEV_LIB_DIR}/cxxopts/include
    $ENV{DEV_LIB_DIR}/spdlog/include
    $ENV{DEV_LIB_DIR}/opencv/include
    $ENV{DEV_LIB_DIR}/onnxruntime/include
    $ENV{DEV_LIB_DIR}/tensorflow/include
    $ENV{DEV_LIB_DIR}/tensorflow/include/src
)

link_directories(
    $ENV{DEV_LIB_DIR}/spdlog/lib
    $ENV{DEV_LIB_DIR}/opencv/lib
    $ENV{DEV_LIB_DIR}/onnxruntime/lib
    $ENV{DEV_LIB_DIR}/tensorflow/lib
)

link_libraries(
    spdlog
    opencv_world347
    onnxruntime
    tensorflow_cc
    FBAMatting
    EnlightenGAN
    DeepLabv3Plus
)

add_executable(${PROJECT_NAME} main.cpp)