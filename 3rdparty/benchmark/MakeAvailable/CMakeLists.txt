# Turn off installing and testing of the benchmark lib
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "This should be OFF. Enables installing the benchmark lib" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "This should be OFF. Enables gtest framework for the benchmark lib" FORCE)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "This should be OFF. Enables tests for the benchmark lib" FORCE)

if(GLIBCXX_USE_CXX11_ABI)
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)
else()
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
endif()

# IntelLLVM (SYCL) compiler defaults to fast math, causing NaN comparison
# code compilation error.
add_compile_options($<$<CXX_COMPILER_ID:IntelLLVM>:-ffp-contract=on>)
add_compile_options($<$<CXX_COMPILER_ID:IntelLLVM>:-fno-fast-math>)

FetchContent_MakeAvailable(ext_benchmark)

# Set the cache vars introduced by the benchmark lib as advanced to not
# clutter the CMake interfaces
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
