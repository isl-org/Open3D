// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// EGL surfaceless/pbuffer offscreen rendering is only used on Linux, where it
// replaces the legacy OSMesa (software) headless path with GPU-accelerated
// rendering that ships in the standard Open3D binary.
#if defined(__linux__)

#include <EGL/egl.h>

#include <memory>

namespace open3d {
namespace visualization {

/// \class EGLOffscreenContext
///
/// \brief GPU-accelerated off-screen OpenGL context created via EGL.
///
/// Used by Visualizer::CreateVisualizerWindow() as a fallback when no
/// windowing system display is available (e.g. no DISPLAY/WAYLAND_DISPLAY),
/// so that headless rendering runs on the GPU instead of a software
/// rasterizer, and does not require a special headless build of Open3D.
class EGLOffscreenContext {
public:
    /// \brief Create an EGL pbuffer-backed OpenGL context of the given size.
    /// \return nullptr if EGL is unavailable or context creation fails.
    static std::unique_ptr<EGLOffscreenContext> Create(int width, int height);

    ~EGLOffscreenContext();
    EGLOffscreenContext(const EGLOffscreenContext &) = delete;
    EGLOffscreenContext &operator=(const EGLOffscreenContext &) = delete;

    /// \brief Make this context current on the calling thread.
    bool MakeCurrent();

    /// \brief Release the context from the calling thread.
    void ReleaseCurrent();

    int GetWidth() const { return width_; }
    int GetHeight() const { return height_; }

private:
    EGLOffscreenContext() = default;
    bool Initialize(int width, int height);

    EGLDisplay display_ = EGL_NO_DISPLAY;
    EGLContext context_ = EGL_NO_CONTEXT;
    EGLSurface surface_ = EGL_NO_SURFACE;
    int width_ = 0;
    int height_ = 0;
};

}  // namespace visualization
}  // namespace open3d

#endif  // defined(__linux__)
