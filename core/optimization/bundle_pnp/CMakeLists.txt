cmake_minimum_required(VERSION 3.0)
project(BoardExperiment)

find_package(Ceres REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(optimizer optimizer.cpp optimizer.h)
target_link_libraries(optimizer PUBLIC ceres)

add_executable(board_experiment board_experiment.cpp)
target_link_libraries(board_experiment PRIVATE optimizer ceres)
