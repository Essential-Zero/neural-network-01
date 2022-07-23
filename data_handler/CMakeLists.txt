# CMake dependencies
cmake_minimum_required(VERSION 3.22.1)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# File dependency
file(GLOB HEADERS "./includes/*.hpp")
file(GLOB SOURCES "./sources/*.cpp")

# Project binary type
add_library(data-handler.app STATIC ${HEADERS} ${SOURCES})

# Project compilation flags
target_compile_options(data-handler.app PRIVATE -Wall -Wextra -Werror -Wpedantic -g)

# Project library linking flag
# target_link_libraries(data-handler.app -lpthread)

# Project dependencies
target_include_directories(data-handler.app PRIVATE includes)
