include(ExternalProject)

set(VMA_SOURCE_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/vkmemalloc")

ExternalProject_Add(
    ext_vkmemalloc
    PREFIX vkmemalloc
    URL https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/raw/refs/tags/v3.3.0/include/vk_mem_alloc.h
    URL_HASH SHA256=90ce12fc4a2466235a09ae02905dd0c13aee80c1bbf11b331ab61230c2ceb112
    DOWNLOAD_NAME vk_mem_alloc.h
    DOWNLOAD_NO_EXTRACT TRUE
    SOURCE_DIR "${VMA_SOURCE_DIR}"
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/vkmemalloc"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

set(VMA_INCLUDE_DIRS ${VMA_SOURCE_DIR}/) # "/" is critical.