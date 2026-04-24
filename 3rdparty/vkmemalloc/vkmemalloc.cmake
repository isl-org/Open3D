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

# VulkanMemoryAllocator-Hpp: C++ bindings for VMA (header-only, CC0 license).
# Provides vma::functionsFromDispatcher() and vma::raii:: RAII wrappers.
# Pinned to v3.3.0+3, matching the VMA C header above.
set(VMA_HPP_SOURCE_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/vkmemalloc_hpp")
ExternalProject_Add(
    ext_vmahpp
    PREFIX vkmemalloc_hpp
    URL https://github.com/YaaZ/VulkanMemoryAllocator-Hpp/releases/download/v3.3.0%2B3/VulkanMemoryAllocator-Hpp-3.3.0.tar.gz
    URL_HASH SHA256=8c2a0573babe1f86f3241d12b4c9df40bb944649b7f4239c1c611797f8af6cd5
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/vkmemalloc_hpp"
    SOURCE_DIR "${VMA_HPP_SOURCE_DIR}/src"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    DEPENDS ext_vkmemalloc
)

set(VMA_INCLUDE_DIRS ${VMA_SOURCE_DIR}/ ${VMA_HPP_SOURCE_DIR}/src/) # "/" is critical.