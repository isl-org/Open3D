include(FetchContent)

FetchContent_Declare(
    ext_pybind11
    PREFIX pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG d28904f12e906dc43f72b549a6b26bceb9c2caff
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/pybind11"
)

FetchContent_MakeAvailable(ext_pybind11)
