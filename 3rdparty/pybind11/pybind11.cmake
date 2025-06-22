include(FetchContent)

FetchContent_Declare(
    ext_pybind11
    PREFIX pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v3.0.0rc4.tar.gz
    URL_HASH SHA256=c6d5d337697d7390d268ded56235fdfb925614be4139c2fb4671a7c2e8828c4b
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/pybind11"
)

FetchContent_MakeAvailable(ext_pybind11)
