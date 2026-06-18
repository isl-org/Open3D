include(ExternalProject)

ExternalProject_Add(
    ext_vulkan_headers
    PREFIX vulkan_headers
    URL https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/v1.4.349.tar.gz
    URL_HASH SHA256=d7b84712f8469657baa37a436d1a23efbf0a6354fc8835b6758ef036e15dcc14
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/vulkan_headers"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_vulkan_headers SOURCE_DIR)
set(VULKAN_HEADERS_INCLUDE_DIRS ${SOURCE_DIR}/include/) # "/" is critical.