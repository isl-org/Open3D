include(ExternalProject)

ExternalProject_Add(
    ext_imgui
    PREFIX imgui
    URL https://github.com/ocornut/imgui/archive/refs/tags/v1.74.tar.gz
    URL_HASH SHA256=2f5f2b789edb00260aa71f03189da5f21cf4b5617c4fbba709e9fbcfc76a2f1e
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/imgui"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_imgui SOURCE_DIR)
set(IMGUI_SOURCE_DIR ${SOURCE_DIR})
