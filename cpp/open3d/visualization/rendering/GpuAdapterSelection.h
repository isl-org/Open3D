// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// General-purpose (non-Gaussian-Splat-specific) helpers for keeping a Vulkan
// device and an OpenGL context on the *same* physical GPU adapter. Needed
// because GL_EXT_memory_object cross-adapter texture import silently fails
// (GL_OUT_OF_MEMORY) on multi-GPU (hybrid graphics) systems.
//
// Architecture: Vulkan selects its physical device first (see
// GaussianSplatVulkanInteropContext), then GetAdapterInfo() + the
// SteerNextGLContextToAdapter() helpers below let the *following* OpenGL
// context creation be steered onto that same adapter.

#pragma once

#include <cstdint>
#include <string>

#if !defined(__APPLE__)

typedef struct VkPhysicalDevice_T* VkPhysicalDevice;

namespace open3d {
namespace visualization {
namespace rendering {

/// Identifies the physical GPU adapter backing a Vulkan device.
struct GpuAdapterInfo {
    bool valid = false;
#if defined(_WIN32)
    std::uint8_t luid[8] = {};
#else
    std::uint32_t pci_domain = 0;
    std::uint32_t pci_bus = 0;
    std::uint32_t pci_device = 0;
    std::uint32_t pci_function = 0;
    bool is_nvidia = false;  ///< driverID == eNvidiaProprietary
#endif
    std::string device_name;  ///< For logging only.
};

/// Extracts the adapter identity from a Vulkan physical device: the 8-byte
/// DXGI LUID on Windows (via VkPhysicalDeviceIDProperties), or the PCI bus
/// address on other platforms (via the optional VK_EXT_pci_bus_info device
/// extension). Returns GpuAdapterInfo::valid == false if unavailable.
GpuAdapterInfo GetAdapterInfo(VkPhysicalDevice physical_device);

/// Best-effort: steers the *next* OpenGL context created in this process
/// onto the physical GPU described by `info`. No-op (returns false) if
/// `info.valid` is false.
///   Windows: sets GLFW window-position hints so the next glfwCreateWindow()
///            lands on the monitor driven by the matching DXGI adapter,
///            causing its WGL context to bind to that adapter. Must be
///            called before glfwCreateWindow().
///   Other platforms: EXPERIMENTAL. Sets Mesa/NVIDIA PRIME-offload
///            environment variables (DRI_PRIME / __NV_PRIME_RENDER_OFFLOAD)
///            matching the PCI bus address. No portable GLX API exists to
///            force adapter selection, so this only has a chance of taking
///            effect if called before *any* GL context has been created in
///            this process (Mesa/GLVND cache the driver choice on first
///            load), and success is not guaranteed. Verify after the fact
///            with GetCurrentGLAdapterUUID().
bool SteerNextGLContextToAdapter(const GpuAdapterInfo& info);

/// Reverse lookup: identifies the physical GPU adapter actually backing an
/// already-created OpenGL context, given its GLFW window handle. Used to
/// verify SteerNextGLContextToAdapter() actually took effect, since it can
/// silently fail (e.g. the target adapter drives no monitor at all, which
/// happens on some hybrid-graphics laptops where the discrete GPU is
/// render-only).
///   Windows: looks up the DXGI adapter driving the monitor the window is
///            on. Always succeeds if the window has a monitor, regardless
///            of whether that adapter was the intended steering target.
///   Other platforms: not implemented (returns GpuAdapterInfo::valid ==
///            false) — no portable way to query the GPU behind an existing
///            GLX context.
GpuAdapterInfo GetAdapterInfoForWindow(void* glfw_window);

/// True if `a` and `b` identify the same physical GPU adapter. False if
/// either is invalid.
bool SameAdapter(const GpuAdapterInfo& a, const GpuAdapterInfo& b);

/// Diagnostic-only: returns the current GL context's GL_DEVICE_UUID_EXT (16
/// raw bytes), or an empty string if unavailable/unsupported. Requires a
/// current GL context with GLEW already initialized. Never used to gate
/// device-selection behavior: some drivers advertise this extension but
/// fail the query (observed: Intel Iris Xe/Arc hybrid Windows driver).
std::string GetCurrentGLAdapterUUID();

/// Hex-encodes raw bytes for log messages (e.g. GetCurrentGLAdapterUUID()).
std::string HexEncode(const std::string& raw_bytes);

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // !defined(__APPLE__)
