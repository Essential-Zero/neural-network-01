# CMake dependencies
cmake_minimum_required(VERSION 3.22.1)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Project definition
project(neural-network-01)

add_subdirectory(data_handler)
add_subdirectory(algorithms/k-nearest-neighbour)
