include(FetchContent)

FetchContent_Declare(
    ext_pybind11
    PREFIX pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v3.0.0.tar.gz
    URL_HASH SHA256=453b1a3e2b266c3ae9da872411cadb6d693ac18063bd73226d96cfb7015a200c
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/pybind11"
)

FetchContent_MakeAvailable(ext_pybind11)
