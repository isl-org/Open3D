// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#if defined(_WIN32)
#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR 1
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <dxgi.h>
#pragma comment(lib, "dxgi.lib")
#endif

#include "open3d/visualization/rendering/GpuAdapterSelection.h"

#if !defined(__APPLE__)

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan_raii.hpp>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#if defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#endif

#include <GL/glew.h>

#include <cstdlib>
#include <cstring>
#include <iterator>

#include <fmt/format.h>

#include "open3d/utility/Logging.h"

namespace open3d {
namespace visualization {
namespace rendering {

#if defined(_WIN32)

namespace {

/// Finds the DXGI adapter identified by `info` and returns the desktop rect
/// of its first output, if it drives one at all.
bool FindMonitorRectForAdapter(const GpuAdapterInfo& info, RECT* out_rect) {
    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1),
                                  reinterpret_cast<void**>(&factory))) ||
        !factory) {
        return false;
    }

    bool found = false;
    for (UINT i = 0;; ++i) {
        IDXGIAdapter1* adapter = nullptr;
        if (factory->EnumAdapters1(i, &adapter) == DXGI_ERROR_NOT_FOUND) {
            break;
        }
        if (!adapter) continue;
        DXGI_ADAPTER_DESC1 adesc{};
        if (SUCCEEDED(adapter->GetDesc1(&adesc)) &&
            std::memcmp(&adesc.AdapterLuid, info.luid,
                        sizeof(info.luid)) == 0) {
            IDXGIOutput* output = nullptr;
            if (adapter->EnumOutputs(0, &output) != DXGI_ERROR_NOT_FOUND &&
                output) {
                DXGI_OUTPUT_DESC odesc{};
                if (SUCCEEDED(output->GetDesc(&odesc))) {
                    *out_rect = odesc.DesktopCoordinates;
                    found = true;
                }
                output->Release();
            }
        }
        adapter->Release();
        if (found) break;
    }
    factory->Release();
    return found;
}

}  // namespace

GpuAdapterInfo GetAdapterInfo(VkPhysicalDevice physical_device) {
    GpuAdapterInfo info;
    vk::PhysicalDevice pd(physical_device);
    const auto chain =
            pd.getProperties2<vk::PhysicalDeviceProperties2,
                              vk::PhysicalDeviceIDProperties>();
    info.device_name = chain.get<vk::PhysicalDeviceProperties2>()
                                .properties.deviceName.data();
    const auto& id = chain.get<vk::PhysicalDeviceIDProperties>();
    if (id.deviceLUIDValid) {
        std::memcpy(info.luid, id.deviceLUID.data(), sizeof(info.luid));
        info.valid = true;
    }
    return info;
}

bool SteerNextGLContextToAdapter(const GpuAdapterInfo& info) {
    if (!info.valid) return false;

    RECT rect{};
    if (!FindMonitorRectForAdapter(info, &rect)) {
        utility::LogWarning(
                "GpuAdapterSelection: could not find a monitor driven by "
                "adapter '{}'; GL context will use its default adapter.",
                info.device_name);
        return false;
    }

    // GLFW binds a WGL context to whatever adapter drives the monitor the
    // window is created on, so positioning the (still hidden) window on
    // that monitor before glfwCreateWindow() steers the context there.
    glfwWindowHint(GLFW_POSITION_X, rect.left);
    glfwWindowHint(GLFW_POSITION_Y, rect.top);
    return true;
}

GpuAdapterInfo GetAdapterInfoForWindow(void* glfw_window) {
    if (!glfw_window) return GpuAdapterInfo();
    HWND hwnd = glfwGetWin32Window(static_cast<GLFWwindow*>(glfw_window));
    if (!hwnd) return GpuAdapterInfo();
    HMONITOR monitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTOPRIMARY);

    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1),
                                  reinterpret_cast<void**>(&factory))) ||
        !factory) {
        return GpuAdapterInfo();
    }

    GpuAdapterInfo info;
    for (UINT i = 0;; ++i) {
        IDXGIAdapter1* adapter = nullptr;
        if (factory->EnumAdapters1(i, &adapter) == DXGI_ERROR_NOT_FOUND) {
            break;
        }
        if (!adapter) continue;
        bool found = false;
        for (UINT j = 0;; ++j) {
            IDXGIOutput* output = nullptr;
            if (adapter->EnumOutputs(j, &output) == DXGI_ERROR_NOT_FOUND) {
                break;
            }
            if (!output) continue;
            DXGI_OUTPUT_DESC output_desc{};
            if (SUCCEEDED(output->GetDesc(&output_desc)) &&
                output_desc.Monitor == monitor) {
                DXGI_ADAPTER_DESC1 adapter_desc{};
                if (SUCCEEDED(adapter->GetDesc1(&adapter_desc))) {
                    std::memcpy(info.luid, &adapter_desc.AdapterLuid,
                                sizeof(info.luid));
                    const std::wstring name(adapter_desc.Description);
                    info.device_name.assign(name.begin(), name.end());
                    info.valid = true;
                    found = true;
                }
            }
            output->Release();
            if (found) break;
        }
        adapter->Release();
        if (found) break;
    }
    factory->Release();
    return info;
}

#else  // !_WIN32

GpuAdapterInfo GetAdapterInfo(VkPhysicalDevice physical_device) {
    GpuAdapterInfo info;
    vk::PhysicalDevice pd(physical_device);
    const auto base_props = pd.getProperties();
    info.device_name = base_props.deviceName.data();

    bool has_pci_ext = false;
    for (const auto& ext : pd.enumerateDeviceExtensionProperties()) {
        if (std::strcmp(ext.extensionName,
                        VK_EXT_PCI_BUS_INFO_EXTENSION_NAME) == 0) {
            has_pci_ext = true;
            break;
        }
    }
    if (!has_pci_ext) return info;  // info.valid stays false

    const auto chain =
            pd.getProperties2<vk::PhysicalDeviceProperties2,
                              vk::PhysicalDevicePCIBusInfoPropertiesEXT>();
    const auto& pci =
            chain.get<vk::PhysicalDevicePCIBusInfoPropertiesEXT>();
    info.pci_domain = pci.pciDomain;
    info.pci_bus = pci.pciBus;
    info.pci_device = pci.pciDevice;
    info.pci_function = pci.pciFunction;

    const auto driver_chain =
            pd.getProperties2<vk::PhysicalDeviceProperties2,
                              vk::PhysicalDeviceDriverProperties>();
    info.is_nvidia = driver_chain.get<vk::PhysicalDeviceDriverProperties>()
                              .driverID == vk::DriverId::eNvidiaProprietary;
    info.valid = true;
    return info;
}

bool SteerNextGLContextToAdapter(const GpuAdapterInfo& info) {
    if (!info.valid) return false;

        const std::string pci_id = fmt::format(
            "pci-{:04x}_{:02x}_{:02x}_{:01x}", info.pci_domain,
            info.pci_bus, info.pci_device, info.pci_function);

    // EXPERIMENTAL: no portable GLX API exists to select a specific GPU, so
    // this relies on the Mesa/NVIDIA PRIME-offload env var convention, which
    // only takes effect if set before the first GL/GLX context is created
    // in this process.
    setenv("DRI_PRIME", pci_id, 1);
    if (info.is_nvidia) {
        setenv("__NV_PRIME_RENDER_OFFLOAD", "1", 1);
        setenv("__GLX_VENDOR_LIBRARY_NAME", "nvidia", 1);
    }
    utility::LogDebug(
            "GpuAdapterSelection: best-effort steering next GL context to "
            "'{}' ({}) via DRI_PRIME{}. This is experimental; verify with "
            "GetCurrentGLAdapterUUID().",
            info.device_name, pci_id, info.is_nvidia ? "/NVIDIA PRIME" : "");
    return true;
}

#endif  // _WIN32

#if !defined(_WIN32)
GpuAdapterInfo GetAdapterInfoForWindow(void* /*glfw_window*/) {
    // No portable GLX/EGL API exists to query the GPU adapter backing an
    // existing context, so verifying SteerNextGLContextToAdapter() actually
    // took effect is not currently supported on this platform.
    return GpuAdapterInfo();
}

#endif

bool SameAdapter(const GpuAdapterInfo& a, const GpuAdapterInfo& b) {
    if (!a.valid || !b.valid) return false;
#if defined(_WIN32)
    return std::memcmp(a.luid, b.luid, sizeof(a.luid)) == 0;
#else
    return a.pci_domain == b.pci_domain && a.pci_bus == b.pci_bus &&
           a.pci_device == b.pci_device && a.pci_function == b.pci_function;
#endif
}

std::string GetCurrentGLAdapterUUID() {
    if (!GLEW_EXT_memory_object) return {};
    GLint num_uuids = 0;
    glGetIntegerv(GL_NUM_DEVICE_UUIDS_EXT, &num_uuids);
    if (num_uuids < 1) return {};
    GLubyte uuid[16] = {};
    glGetUnsignedBytei_vEXT(GL_DEVICE_UUID_EXT, 0, uuid);
    if (glGetError() != GL_NO_ERROR) return {};
    return std::string(reinterpret_cast<const char*>(uuid), sizeof(uuid));
}

std::string HexEncode(const std::string& raw_bytes) {
    std::string out;
    out.reserve(raw_bytes.size() * 2);
    for (unsigned char b : raw_bytes) {
        fmt::format_to(std::back_inserter(out), "{:02x}", b);
    }
    return out;
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // !defined(__APPLE__)
