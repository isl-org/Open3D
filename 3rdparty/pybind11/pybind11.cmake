include(FetchContent)

FetchContent_Declare(
    ext_pybind11
    PREFIX pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz
    URL_HASH SHA256=e08cb87f4773da97fa7b5f035de8763abc656d87d5773e62f6da0587d1f0ec20
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/pybind11"
)

FetchContent_MakeAvailable(ext_pybind11)
