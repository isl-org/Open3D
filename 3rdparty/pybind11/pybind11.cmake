include(FetchContent)

FetchContent_Declare(
    ext_pybind11
    PREFIX pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v2.5.0.tar.gz
    URL_HASH SHA256=97504db65640570f32d3fdf701c25a340c8643037c3b69aec469c10c93dc8504
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/pybind11"
)

FetchContent_MakeAvailable(ext_pybind11)
