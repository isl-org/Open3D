// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// GL context management for OpenGL compute-based Gaussian splatting.
// Uses a hidden GLFW window with a GL 4.6 core-profile context and exposes
// the native handle (GLXContext on Linux/X11/XWayland, HGLRC on Windows) so
// Filament can create its own context sharing the same GL object namespace.
//
// Offscreen rendering on Linux now requires an X11 or XWayland server. EGL is
// intentionally not used because Filament's Linux OpenGL path is GLX-only.

#pragma once

#include <cstddef>
#include <cstdint>

#if !defined(__APPLE__)

namespace open3d {
namespace visualization {
namespace rendering {

/// Manages a shared OpenGL 4.6 context for Gaussian splatting compute.
/// GLFW handles platform-specific context creation (GLX on Linux/X11/XWayland,
/// WGL on Windows).
///
/// Lifecycle:
///   1. Call InitializeStandalone() before Filament Engine::create().
///   2. Pass GetNativeContext() to Engine::create() as sharedGLContext.
///   3. For compute work: MakeCurrent() → dispatch → ReleaseCurrent().
///   4. Call Shutdown() when done.
class GaussianSplatOpenGLContext {
public:
    static GaussianSplatOpenGLContext& GetInstance();

    /// Creates a hidden GLFW window carrying a GL 4.6 core-profile context.
    /// Must be called before Filament Engine::create() so the returned native
    /// context can be passed as sharedGLContext for zero-copy sharing.
    bool InitializeStandalone();

    /// Safety wrapper used from render paths. Late initialization cannot
    /// establish zero-copy sharing with Filament, so this only succeeds when
    /// InitializeStandalone() already ran.
    bool Initialize();

    /// Returns the native context handle suitable for Filament's
    /// sharedGLContext parameter:
    ///   Linux   -> GLXContext
    ///   Windows -> HGLRC
    void* GetNativeContext() const;

    /// Queries an identifier for the GPU adapter this context is bound to,
    /// so callers (Vulkan interop) can select a *matching* physical device.
    /// GL_EXT_memory_object cross-adapter import silently fails, so this
    /// match is required on multi-GPU (hybrid graphics) systems.
    ///   Windows -> DXGI adapter LUID (8 bytes), via the monitor the
    ///              context's window is associated with. More reliable than
    ///              GL_DEVICE_UUID_EXT, which some drivers advertise but do
    ///              not actually implement (observed: Intel Iris Xe/Arc
    ///              hybrid driver returns GL_INVALID_OPERATION for it).
    ///   Other platforms -> GL_DEVICE_UUID_EXT (16 bytes).
    /// Returns false (leaving out_id untouched) if unavailable.
    bool GetAdapterId(std::uint8_t out_id[16], std::size_t& out_size) const;

    /// Destroys the context and associated resources.
    void Shutdown();

    /// Returns true if the context is initialized and valid.
    bool IsValid() const;

    /// Makes this context current on the calling thread.
    bool MakeCurrent();

    /// Releases the context (no context current on this thread).
    void ReleaseCurrent();

private:
    GaussianSplatOpenGLContext() = default;
    ~GaussianSplatOpenGLContext();

    GaussianSplatOpenGLContext(const GaussianSplatOpenGLContext&) = delete;
    GaussianSplatOpenGLContext& operator=(const GaussianSplatOpenGLContext&) =
            delete;

private:
    // Stored as void* (== GLFWwindow*) to avoid a GLFW header dependency.
    void* glfw_window_ = nullptr;

    bool initialized_ = false;
    bool gl_logged_ = false;
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // !defined(__APPLE__)
