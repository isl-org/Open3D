project(NewLogger CXX)
cmake_minimum_required(VERSION 3.15)

add_executable(NewLogger)
target_sources(NewLogger PRIVATE
    Main.cpp
)
