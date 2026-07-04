// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/DevicePluginLoader.h"

#include <memory>
#include <string>

#include "open3d/Open3DConfig.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <limits.h>
#include <unistd.h>
#endif

namespace open3d {
namespace core {
namespace {

bool g_cuda_loaded = false;
bool g_xpu_loaded = false;

struct DevicePluginInitializer {
    DevicePluginInitializer() {
#if defined(BUILD_SHARED_LIBS)
        const std::string host_dir =
                open3d::utility::filesystem::GetSelfBinaryDirectory();
        if (host_dir.empty()) {
            return;
        }

        const std::string cuda_lib_path = host_dir +
#ifdef _WIN32
                                          "/Open3D_cuda.dll"
#else
                                          "/libOpen3D_cuda."
                                          "so." OPEN3D_ABI_VERSION_STRING
#endif
                ;
#ifdef _WIN32
        if (LoadLibraryA(cuda_lib_path.c_str()) != nullptr) {
#else
        if (dlopen(cuda_lib_path.c_str(), RTLD_NOW | RTLD_GLOBAL) != nullptr) {
#endif
            utility::LogDebug("Loaded CUDA device library: {}", cuda_lib_path);
            g_cuda_loaded = true;
        }

        const std::string xpu_lib_path = host_dir +
#ifdef _WIN32
                                         "/Open3D_xpu.dll"
#else
                                         "/libOpen3D_xpu."
                                         "so." OPEN3D_ABI_VERSION_STRING
#endif
                ;
#ifdef _WIN32
        if (LoadLibraryA(xpu_lib_path.c_str()) != nullptr) {
#else
        if (dlopen(xpu_lib_path.c_str(), RTLD_NOW | RTLD_GLOBAL) != nullptr) {
#endif
            utility::LogDebug("Loaded SYCL device library: {}", xpu_lib_path);
            g_xpu_loaded = true;
        }

#else  // Static link: device code is in the same binary.
#if defined(BUILD_CUDA_MODULE)
        g_cuda_loaded = true;
#endif
#if defined(BUILD_SYCL_MODULE)
        g_xpu_loaded = true;
#endif
#endif
    }
};

void EnsureDevicePluginsLoaded() { static DevicePluginInitializer initializer; }

}  // namespace

bool IsCudaDeviceLibraryLoaded() {
    EnsureDevicePluginsLoaded();
    return g_cuda_loaded;
}

bool IsXpuDeviceLibraryLoaded() {
    EnsureDevicePluginsLoaded();
    return g_xpu_loaded;
}

#if defined(BUILD_SHARED_LIBS)
std::shared_ptr<MemoryManagerDevice> CreateCudaMemoryManagerDevice() {
    using FactoryFn = void (*)(std::shared_ptr<MemoryManagerDevice>*);
    static FactoryFn factory = nullptr;
    if (factory == nullptr) {
#ifdef _WIN32
        factory = reinterpret_cast<FactoryFn>(
                GetProcAddress(GetModuleHandleA("Open3D_cuda.dll"),
                               "open3d_create_cuda_memory_manager"));
#else
        factory = reinterpret_cast<FactoryFn>(
                dlsym(RTLD_DEFAULT, "open3d_create_cuda_memory_manager"));
#endif
        if (factory == nullptr) {
            utility::LogError(
                    "open3d_create_cuda_memory_manager not found in the CUDA "
                    "device library.");
        }
    }
    std::shared_ptr<MemoryManagerDevice> result;
    factory(&result);
    return result;
}
#endif

}  // namespace core
}  // namespace open3d
