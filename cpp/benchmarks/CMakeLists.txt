add_executable(benchmarks)

add_subdirectory(core)
add_subdirectory(geometry)
add_subdirectory(io)
add_subdirectory(pipelines)
add_subdirectory(t/geometry)
add_subdirectory(t/pipelines)

target_compile_definitions(benchmarks PRIVATE TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/examples/test_data")
target_compile_definitions(benchmarks PRIVATE BENCHMARK_DATA_DIR="${PROJECT_SOURCE_DIR}/data/Benchmark")

target_link_libraries(benchmarks PRIVATE Open3D::Open3D)
open3d_show_and_abort_on_warning(benchmarks)
open3d_set_global_properties(benchmarks)
open3d_link_3rdparty_libraries(benchmarks)

if (BUILD_CUDA_MODULE)
    target_include_directories(benchmarks SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

if (WITH_IPPICV)
    target_compile_definitions(benchmarks PRIVATE IPP_CONDITIONAL_TEST_STR=) # Empty string (test not disabled)
else()
    target_compile_definitions(benchmarks PRIVATE IPP_CONDITIONAL_TEST_STR=DISABLED_)
endif()

target_link_libraries(benchmarks PRIVATE benchmark::benchmark benchmark::benchmark_main)

