# CMake dependencies
cmake_minimum_required(VERSION 3.22.1)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# File dependency
file(GLOB HEADERS "./includes/*.hpp")
file(GLOB SOURCES "./sources/*.cpp")

# Project binary type
add_executable(k-nearest-neighbour.app ${HEADERS} ${SOURCES})

# Project compilation flags
target_compile_options(k-nearest-neighbour.app PRIVATE -Wall -Wextra -Werror -Wpedantic -g)

# Project preporcessor definitions
add_compile_definitions(EUCLID)

# Project library linking flag
target_link_libraries(k-nearest-neighbour.app ${CMAKE_SOURCE_DIR}/build/data_handler/libdata-handler.app.a)

# Project dependencies
target_include_directories(k-nearest-neighbour.app PRIVATE includes)
