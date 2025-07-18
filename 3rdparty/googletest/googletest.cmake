include(FetchContent)

FetchContent_Declare(
    ext_googletest
    PREFIX googletest
    URL https://github.com/google/googletest/releases/download/v1.16.0/googletest-1.16.0.tar.gz
    URL_HASH SHA256=78c676fc63881529bf97bf9d45948d905a66833fbfa5318ea2cd7478cb98f399
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/googletest"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

FetchContent_MakeAvailable(ext_googletest)
FetchContent_GetProperties(ext_googletest SOURCE_DIR GOOGLETEST_SOURCE_DIR)
