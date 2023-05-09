cmake_minimum_required(VERSION 3.15)
project(hello)
add_executable(hello hello.cpp)
target_include_directories(hello PRIVATE ${CMAKE_SOURCE_DIR}/../include)
