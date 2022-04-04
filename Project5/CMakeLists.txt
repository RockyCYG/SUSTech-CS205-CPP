cmake_minimum_required(VERSION 3.12)
project(Project5)

set(CMAKE_CXX_STANDARD 20)

set(OpenCV_DIR D:\\opencv\\mingw-build)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(Project5 main.cpp)
target_link_libraries(Project5 ${OpenCV_LIBS})