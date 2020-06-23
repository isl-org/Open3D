cmake_minimum_required(VERSION 3.0)
# Overide policy for hidden symbol visibility
if(POLICY CMP0063)
    cmake_policy(SET CMP0063 NEW)
endif()

################################################################################
#
# build benchmark dependency here so we can pass the _GLIBCXX_USE_CXX11_ABI
# definition
#

# turn off installing and testing of the benchmark lib
set(BENCHMARK_ENABLE_INSTALL  OFF CACHE BOOL "This should be OFF. Enables installing the benchmark lib")
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "This should be OFF. Enables gtest framework for the benchmark lib")
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "This should be OFF. Enables tests for the benchmark lib")

if(GLIBCXX_USE_CXX11_ABI)
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
else()
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
endif()
add_subdirectory(${Open3D_3RDPARTY_DIR}/benchmark ${CMAKE_BINARY_DIR}/3rdparty/benchmark2)
# set the cache vars introduced by the benchmark lib as advanced to not
# clutter the cmake interfaces
mark_as_advanced(
    BENCHMARK_ENABLE_INSTALL
    BENCHMARK_ENABLE_GTEST_TESTS
    BENCHMARK_ENABLE_TESTING
    BENCHMARK_ENABLE_ASSEMBLY_TESTS
    BENCHMARK_DOWNLOAD_DEPENDENCIES
    BENCHMARK_BUILD_32_BITS
    BENCHMARK_ENABLE_EXCEPTIONS
    BENCHMARK_ENABLE_LTO
    BENCHMARK_USE_LIBCXX
)
################################################################################


set(BENCHMARK_SOURCE_FILES
    geometry/KDTreeFlann.cpp
    geometry/SamplePoints.cpp
    core/Reduction.cpp
    io/PointCloudIO.cpp
)

add_executable(benchmarks ${BENCHMARK_SOURCE_FILES})
add_definitions(-DTEST_DATA_DIR="${PROJECT_SOURCE_DIR}/examples/test_data")
add_definitions(-DBENCHMARK_DATA_DIR="${PROJECT_SOURCE_DIR}/data/Benchmark")

target_link_libraries(benchmarks ${CMAKE_PROJECT_NAME} benchmark::benchmark benchmark::benchmark_main)
open3d_show_and_abort_on_warning(benchmarks)
open3d_set_global_properties(benchmarks)

if (BUILD_CUDA_MODULE)
    target_include_directories(benchmarks SYSTEM PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
