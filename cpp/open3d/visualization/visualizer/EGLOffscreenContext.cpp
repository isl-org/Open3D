// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/visualizer/EGLOffscreenContext.h"

#if defined(__linux__)

#include <EGL/eglext.h>

#include <algorithm>

#include "open3d/utility/Logging.h"

namespace open3d {
namespace visualization {

namespace {

// Prefer a display bound to a physical GPU device (EGL_EXT_device_base /
// EGL_EXT_platform_device) so that offscreen rendering runs on the GPU. Falls
// back to the explicit "surfaceless" platform (EGL_MESA_platform_surfaceless)
// when no device is found, e.g. in a container with no /dev/dri node. We
// deliberately avoid the implicit eglGetDisplay(EGL_DEFAULT_DISPLAY) platform
// auto-detection here: on systems without a GPU or windowing server, Mesa's
// platform probing (Wayland/X11/DRM) can crash instead of falling back
// gracefully, whereas requesting the surfaceless platform directly is safe
// and always available with Mesa software rendering (llvmpipe).
EGLDisplay GetHardwareDisplay() {
    auto eglQueryDevicesEXT = reinterpret_cast<PFNEGLQUERYDEVICESEXTPROC>(
            eglGetProcAddress("eglQueryDevicesEXT"));
    auto eglGetPlatformDisplayEXT =
            reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
                    eglGetProcAddress("eglGetPlatformDisplayEXT"));
    if (eglQueryDevicesEXT && eglGetPlatformDisplayEXT) {
        constexpr EGLint kMaxDevices = 16;
        EGLDeviceEXT devices[kMaxDevices];
        EGLint num_devices = 0;
        if (eglQueryDevicesEXT(kMaxDevices, devices, &num_devices) ==
                    EGL_TRUE &&
            num_devices > 0) {
            for (EGLint i = 0; i < num_devices; ++i) {
                EGLDisplay display = eglGetPlatformDisplayEXT(
                        EGL_PLATFORM_DEVICE_EXT, devices[i], nullptr);
                if (display != EGL_NO_DISPLAY) {
                    return display;
                }
            }
        }
    }
    if (eglGetPlatformDisplayEXT) {
        utility::LogDebug(
                "EGLOffscreenContext: no EGL device found, falling back to "
                "the surfaceless platform (software rendering).");
        EGLDisplay display =
                eglGetPlatformDisplayEXT(EGL_PLATFORM_SURFACELESS_MESA,
                                         /*native_display=*/nullptr, nullptr);
        if (display != EGL_NO_DISPLAY) {
            return display;
        }
    }
    utility::LogDebug(
            "EGLOffscreenContext: EGL_EXT_platform_device and "
            "EGL_MESA_platform_surfaceless not available, using default "
            "EGL display.");
    return eglGetDisplay(EGL_DEFAULT_DISPLAY);
}

}  // namespace

std::unique_ptr<EGLOffscreenContext> EGLOffscreenContext::Create(int width,
                                                                 int height) {
    // Constructor is private; use a raw new/reset via a helper deleter-free
    // pattern since std::make_unique cannot access it.
    std::unique_ptr<EGLOffscreenContext> ctx(new EGLOffscreenContext());
    if (!ctx->Initialize(width, height)) {
        return nullptr;
    }
    return ctx;
}

bool EGLOffscreenContext::Initialize(int width, int height) {
    width_ = std::max(width, 1);
    height_ = std::max(height, 1);

    display_ = GetHardwareDisplay();
    if (display_ == EGL_NO_DISPLAY) {
        utility::LogWarning("EGLOffscreenContext: no EGL display found.");
        return false;
    }

    EGLint major = 0, minor = 0;
    if (!eglInitialize(display_, &major, &minor)) {
        utility::LogWarning("EGLOffscreenContext: eglInitialize() failed.");
        display_ = EGL_NO_DISPLAY;
        return false;
    }

    if (!eglBindAPI(EGL_OPENGL_API)) {
        utility::LogWarning(
                "EGLOffscreenContext: eglBindAPI(EGL_OPENGL_API) failed.");
        return false;
    }

    const EGLint config_attribs[] = {EGL_SURFACE_TYPE,
                                     EGL_PBUFFER_BIT,
                                     EGL_RENDERABLE_TYPE,
                                     EGL_OPENGL_BIT,
                                     EGL_RED_SIZE,
                                     8,
                                     EGL_GREEN_SIZE,
                                     8,
                                     EGL_BLUE_SIZE,
                                     8,
                                     EGL_ALPHA_SIZE,
                                     8,
                                     EGL_DEPTH_SIZE,
                                     24,
                                     EGL_STENCIL_SIZE,
                                     8,
                                     EGL_NONE};
    EGLConfig config;
    EGLint num_configs = 0;
    if (!eglChooseConfig(display_, config_attribs, &config, 1, &num_configs) ||
        num_configs == 0) {
        utility::LogWarning("EGLOffscreenContext: eglChooseConfig() failed.");
        return false;
    }

    const EGLint pbuffer_attribs[] = {EGL_WIDTH, width_, EGL_HEIGHT, height_,
                                      EGL_NONE};
    surface_ = eglCreatePbufferSurface(display_, config, pbuffer_attribs);
    if (surface_ == EGL_NO_SURFACE) {
        utility::LogWarning(
                "EGLOffscreenContext: eglCreatePbufferSurface() failed.");
        return false;
    }

    // Request a GL 3.3 core-profile context, matching the windowed GLFW path
    // in Visualizer::CreateVisualizerWindow().
    const EGLint context_attribs[] = {EGL_CONTEXT_MAJOR_VERSION,
                                      3,
                                      EGL_CONTEXT_MINOR_VERSION,
                                      3,
                                      EGL_CONTEXT_OPENGL_PROFILE_MASK,
                                      EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
                                      EGL_NONE};
    context_ =
            eglCreateContext(display_, config, EGL_NO_CONTEXT, context_attribs);
    if (context_ == EGL_NO_CONTEXT) {
        utility::LogWarning("EGLOffscreenContext: eglCreateContext() failed.");
        return false;
    }

    if (!MakeCurrent()) {
        utility::LogWarning("EGLOffscreenContext: eglMakeCurrent() failed.");
        return false;
    }

    utility::LogInfo(
            "EGLOffscreenContext: created {}x{} offscreen GPU context "
            "(EGL {}.{}, vendor: {}).",
            width_, height_, major, minor,
            eglQueryString(display_, EGL_VENDOR));
    return true;
}

bool EGLOffscreenContext::MakeCurrent() {
    if (display_ == EGL_NO_DISPLAY || context_ == EGL_NO_CONTEXT) {
        return false;
    }
    return eglMakeCurrent(display_, surface_, surface_, context_) == EGL_TRUE;
}

bool EGLOffscreenContext::SwapBuffers() {
    if (display_ == EGL_NO_DISPLAY || surface_ == EGL_NO_SURFACE) {
        return false;
    }
    return eglSwapBuffers(display_, surface_) == EGL_TRUE;
}

void EGLOffscreenContext::ReleaseCurrent() {
    if (display_ != EGL_NO_DISPLAY) {
        eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE,
                       EGL_NO_CONTEXT);
    }
}

EGLOffscreenContext::~EGLOffscreenContext() {
    if (display_ == EGL_NO_DISPLAY) {
        return;
    }
    ReleaseCurrent();
    if (context_ != EGL_NO_CONTEXT) {
        eglDestroyContext(display_, context_);
    }
    if (surface_ != EGL_NO_SURFACE) {
        eglDestroySurface(display_, surface_);
    }
    eglTerminate(display_);
}

}  // namespace visualization
}  // namespace open3d

#endif  // defined(__linux__)
