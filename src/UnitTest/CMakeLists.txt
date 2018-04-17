
cmake_minimum_required(VERSION 3.0)

# Locate GTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Specify the unit tests
add_executable(unitTests
               main.cpp
               Image_unittest.cpp)

# Link with GTest and pthread libraries
target_link_libraries(unitTests ${GTEST_LIBRARIES} pthread Core)
