include(FetchContent)

FetchContent_Declare(
    ext_imgui
    PREFIX imgui
    URL https://github.com/ocornut/imgui/archive/refs/tags/v1.88.tar.gz
    URL_HASH SHA256=9f14c788aee15b777051e48f868c5d4d959bd679fc5050e3d2a29de80d8fd32e
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/imgui"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

FetchContent_MakeAvailable(ext_imgui)
FetchContent_GetProperties(ext_imgui SOURCE_DIR IMGUI_SOURCE_DIR)
