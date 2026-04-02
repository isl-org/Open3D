// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// GL context management for OpenGL compute-based Gaussian splatting.
// On X11 creates a GLX context; on Wayland creates an EGL context; on Windows
// creates a WGL context.  All use a headless pbuffer / hidden-window surface
// for offscreen compute dispatch.

#pragma once

#if !defined(__APPLE__)

namespace open3d {
namespace visualization {
namespace rendering {

/// Manages a GL context for OpenGL compute operations.
/// Uses GLX on X11 (matching Filament's PlatformGLX) and EGL on Wayland
/// (matching Filament's PlatformEGL).
///
/// Lifecycle:
///   1. Call Initialize() before first compute dispatch.
///   2. For compute work: MakeCurrent() → dispatch → Finish() →
///   ReleaseCurrent().
///   3. Call Shutdown() when done.
class GaussianComputeOpenGLContext {
public:
    static GaussianComputeOpenGLContext& GetInstance();

    /// Creates the GL context (GLX on X11, EGL on Wayland).
    /// Returns true on success.
    bool Initialize();

    /// Creates a standalone (non-shared) GL context, meant to be called
    /// BEFORE the Filament engine so it can be passed to Engine::create()
    /// as the shared context.  When Filament starts with this as the
    /// shared context it creates its own context sharing our GL namespace.
    bool InitializeStandalone();

    /// Returns the native GL context handle (GLXContext* or EGLContext*),
    /// suitable for passing to EngineInstance::SetSharedContext().
    void* GetNativeContext() const;

    /// Destroys the context and associated resources.
    void Shutdown();

    /// Returns true if the context is initialized and valid.
    bool IsValid() const;

    /// Makes this context current on the calling thread.
    bool MakeCurrent();

    /// Releases the context (no context current on this thread).
    void ReleaseCurrent();

    /// Calls glFinish() to ensure all compute commands complete.
    void Finish();

private:
    GaussianComputeOpenGLContext() = default;
    ~GaussianComputeOpenGLContext();

    GaussianComputeOpenGLContext(const GaussianComputeOpenGLContext&) = delete;
    GaussianComputeOpenGLContext& operator=(
            const GaussianComputeOpenGLContext&) = delete;

    // Made public for use by DetectBackend() helper.
public:
    enum class Backend { kNone, kGLX, kEGL, kWGL };

private:
    bool InitializeGLX();
    bool InitializeGLXStandalone();  // creates context without sharing
    bool InitializeEGL();
    bool InitializeWGL();
    bool InitializeWGLStandalone();  // creates context without sharing

    Backend backend_ = Backend::kNone;

    // Helper window for context creation (GLFW-owned, Linux/Windows only).
    // Stored as void* (== GLFWwindow*) to avoid GLFW header dependency.
    void* glfw_helper_window_ = nullptr;

    // GLX state (X11). GLXContext is a pointer; Window is XID (ulong).
    void* x_display_ = nullptr;
    void* glx_context_ = nullptr;
    unsigned long glx_drawable_ = 0;

    // EGL state (Wayland). All stored as void* to avoid header pollution.
    void* egl_display_ = nullptr;
    void* egl_context_ = nullptr;
    void* egl_surface_ = nullptr;

    bool initialized_ = false;
    bool gl_logged_ = false;
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // !defined(__APPLE__)
