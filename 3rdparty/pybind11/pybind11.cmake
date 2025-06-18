include(FetchContent)

FetchContent_Declare(
    ext_pybind11
    PREFIX pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v3.0.0rc1.tar.gz
    URL_HASH SHA256=385a60c19e40458fd63509ccdb21c3f3317a59944d98dab093b53dbcf2aaac80
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/pybind11"
)

FetchContent_MakeAvailable(ext_pybind11)
