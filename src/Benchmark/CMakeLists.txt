cmake_minimum_required(VERSION 3.0)

set(BENCHMARK_SOURCE_FILES
    Geometry/KDTreeFlann.cpp
    Geometry/SamplePoints.cpp
    Core/Reduction.cpp
)

add_executable(benchmarks ${BENCHMARK_SOURCE_FILES})
add_definitions(-DTEST_DATA_DIR="${PROJECT_SOURCE_DIR}/examples/TestData")
add_definitions(-DBENCHMARK_DATA_DIR="${PROJECT_SOURCE_DIR}/data/Benchmark")

target_link_libraries(benchmarks ${CMAKE_PROJECT_NAME} benchmark::benchmark benchmark::benchmark_main)
open3d_show_and_abort_on_warning(benchmarks)

if (BUILD_CUDA_MODULE)
    target_include_directories(benchmarks PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
