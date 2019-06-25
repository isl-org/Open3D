
cmake_minimum_required(VERSION 3.0)

include_directories(".")

file(GLOB_RECURSE UNIT_TESTS "*.cpp")
add_executable(unitTests ${UNIT_TESTS})
add_definitions(-DTEST_DATA_DIR="${PROJECT_SOURCE_DIR}/examples/TestData")

# If gpu not available, add "DISABLED_" to the gpu test names
if(BUILD_CUDA_MODULE)
    add_definitions(-DGPU_CONDITIONAL_TEST_STR=) # Empty string
else()
    add_definitions(-DGPU_CONDITIONAL_TEST_STR=DISABLED_)
endif()

target_link_libraries(unitTests pthread ${CMAKE_PROJECT_NAME})
