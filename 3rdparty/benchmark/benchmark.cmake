include(FetchContent)

FetchContent_Declare(
    ext_benchmark
    PREFIX benchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.5.5.tar.gz
    URL_HASH SHA256=3bff5f237c317ddfd8d5a9b96b3eede7c0802e799db520d38ce756a2a46a18a0
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/benchmark"
)

# Use an additional directory scope to set GLIBCXX_USE_CXX11_ABI
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/MakeAvailable)
