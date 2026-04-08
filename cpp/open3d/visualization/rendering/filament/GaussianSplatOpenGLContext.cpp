// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// OpenGL context management for Gaussian splatting compute.
//
// A hidden GLFW window owns a GL 4.6 core-profile context. Its native handle
// (GLXContext on Linux/X11/XWayland, HGLRC on Windows) is passed to Filament's
// Engine::create() as sharedGLContext so both contexts share the same GL object
// namespace for zero-copy texture import.
//
// Linux intentionally uses GLFW's X11 platform only; EGL is not used because
// Filament's Linux OpenGL backend is PlatformGLX. Offscreen rendering on Linux
// therefore requires an X11 or XWayland server.

#include "open3d/visualization/rendering/filament/GaussianSplatOpenGLContext.h"

#if !defined(__APPLE__)

#include <cstdlib>
#include <cstring>

// GLFW for cross-platform hidden window and context creation.
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

// GLEW must be included before any GL header pulled in by glfw3native.h.
#include <GL/glew.h>

#if defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WGL
#else
#define GLFW_EXPOSE_NATIVE_X11
#define GLFW_EXPOSE_NATIVE_GLX
#endif
#include <GLFW/glfw3native.h>

#include "open3d/utility/Logging.h"

namespace open3d {
namespace visualization {
namespace rendering {

namespace {

const char* GetSessionType() {
#if defined(_WIN32)
    return "windows";
#else
    const char* session = std::getenv("XDG_SESSION_TYPE");
    return session ? session : "unknown";
#endif
}

const char* GetNativeBackendName() {
#if defined(_WIN32)
    return "WGL";
#else
    return "GLX";
#endif
}

}  // namespace

GaussianSplatOpenGLContext& GaussianSplatOpenGLContext::GetInstance() {
    static GaussianSplatOpenGLContext instance;
    return instance;
}

GaussianSplatOpenGLContext::~GaussianSplatOpenGLContext() { Shutdown(); }

bool GaussianSplatOpenGLContext::InitializeStandalone() {
    if (initialized_) {
        return true;
    }

#if !defined(_WIN32)
    if (std::strcmp(GetSessionType(), "wayland") == 0) {
        utility::LogInfo(
                "GaussianSplatOpenGLContext: Wayland session detected; "
                "using X11/GLX via XWayland for Filament compatibility.");
    }
#endif

    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_FALSE);

    GLFWwindow* window =
            glfwCreateWindow(1, 1, "O3D_GS_Helper", nullptr, nullptr);
    glfwDefaultWindowHints();
    if (!window) {
        utility::LogWarning(
                "GaussianSplatOpenGLContext: glfwCreateWindow failed. "
                "Linux offscreen rendering now requires an X11/XWayland "
                "server.");
        return false;
    }

    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    const GLenum glew_err = glewInit();
    while (glGetError() != GL_NO_ERROR) {
    }
    if (glew_err != GLEW_OK) {
        utility::LogWarning(
                "GaussianSplatOpenGLContext: glewInit warning: {}",
                reinterpret_cast<const char*>(glewGetErrorString(glew_err)));
    }

    glfw_window_ = window;
    initialized_ = true;

    const char* vendor = reinterpret_cast<const char*>(glGetString(GL_VENDOR));
    const char* renderer =
            reinterpret_cast<const char*>(glGetString(GL_RENDERER));
    const char* version =
            reinterpret_cast<const char*>(glGetString(GL_VERSION));
    utility::LogDebug(
            "GaussianSplatOpenGLContext: Created standalone {} context "
            "for session={} native={:p}",
            GetNativeBackendName(), GetSessionType(), GetNativeContext());
    utility::LogDebug(
            "GaussianSplatOpenGLContext: GL vendor={} renderer={} "
            "version={}",
            vendor ? vendor : "?", renderer ? renderer : "?",
            version ? version : "?");

    glfwMakeContextCurrent(nullptr);
    return true;
}

bool GaussianSplatOpenGLContext::Initialize() {
    if (initialized_) {
        return true;
    }

    utility::LogWarning(
            "GaussianSplatOpenGLContext: late initialization is not "
            "supported. InitializeStandalone() must run before Filament "
            "Engine::create() so zero-copy sharedGLContext setup succeeds.");
    return false;
}

void* GaussianSplatOpenGLContext::GetNativeContext() const {
    if (!glfw_window_) {
        return nullptr;
    }

    GLFWwindow* window = static_cast<GLFWwindow*>(glfw_window_);
#if defined(_WIN32)
    return reinterpret_cast<void*>(glfwGetWGLContext(window));
#else
    return reinterpret_cast<void*>(glfwGetGLXContext(window));
#endif
}

void GaussianSplatOpenGLContext::Shutdown() {
    if (!initialized_) {
        return;
    }

    glfwMakeContextCurrent(nullptr);
    glfwDestroyWindow(static_cast<GLFWwindow*>(glfw_window_));
    glfw_window_ = nullptr;
    initialized_ = false;
    gl_logged_ = false;
    utility::LogDebug("GaussianSplatOpenGLContext: Shut down.");
}

bool GaussianSplatOpenGLContext::IsValid() const { return initialized_; }

bool GaussianSplatOpenGLContext::MakeCurrent() {
    if (!initialized_) {
        return false;
    }

    GLFWwindow* window = static_cast<GLFWwindow*>(glfw_window_);
    glfwMakeContextCurrent(window);
    const bool ok = (glfwGetCurrentContext() == window);
    if (!ok) {
        utility::LogWarning(
                "GaussianSplatOpenGLContext: glfwMakeContextCurrent "
                "failed.");
        return false;
    }

    if (!gl_logged_) {
        gl_logged_ = true;
        const char* version =
                reinterpret_cast<const char*>(glGetString(GL_VERSION));
        const char* renderer =
                reinterpret_cast<const char*>(glGetString(GL_RENDERER));
        utility::LogDebug(
                "GaussianSplatOpenGLContext: {} active - GL {} on {}",
                GetNativeBackendName(), version ? version : "?",
                renderer ? renderer : "?");
    }

    return true;
}

void GaussianSplatOpenGLContext::ReleaseCurrent() {
    glfwMakeContextCurrent(nullptr);
}

void GaussianSplatOpenGLContext::Finish() { glFinish(); }

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // !defined(__APPLE__)
