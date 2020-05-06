file(GLOB_RECURSE BENCHMARK_SOURCE_FILES "*.cpp")

add_executable(benchmarks ${BENCHMARK_SOURCE_FILES})
add_definitions(-DTEST_DATA_DIR="${PROJECT_SOURCE_DIR}/examples/TestData")
add_definitions(-DBENCHMARK_DATA_DIR="${PROJECT_SOURCE_DIR}/data/Benchmark")

target_link_libraries(benchmarks benchmark::benchmark benchmark::benchmark_main ${CMAKE_PROJECT_NAME})
ShowAndAbortOnWarning(benchmarks)
