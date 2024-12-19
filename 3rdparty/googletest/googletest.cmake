include(FetchContent)

FetchContent_Declare(
    ext_googletest
    PREFIX googletest
    URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.tar.gz
    URL_HASH SHA256=b4870bf121ff7795ba20d20bcdd8627b8e088f2d1dab299a031c1034eddc93d5
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/googletest"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

FetchContent_MakeAvailable(ext_googletest)
FetchContent_GetProperties(ext_googletest SOURCE_DIR GOOGLETEST_SOURCE_DIR)
