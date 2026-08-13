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

#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatOpenGLContext.h"

#if !defined(__APPLE__)

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatVulkanInteropContext.h"

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

#if defined(_WIN32)
#include <dxgi.h>
#pragma comment(lib, "dxgi.lib")
#endif

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
        utility::LogDebug(
                "GaussianSplatOpenGLContext: Wayland session detected; "
                "using X11/GLX via XWayland.");
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
                "GaussianSplatOpenGLContext: GS helper window failed. "
                "Gaussian Splat rendering is not available.");
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

    // After glewInit(), probe GL interop extensions for the Vulkan interop
    // context (EXT_memory_object_fd, etc.).
    // This must happen while the GL context is current.
    auto& vk_ctx = GaussianSplatVulkanInteropContext::GetInstance();
    if (vk_ctx.IsValid() && !vk_ctx.AreGLExtensionsReady()) {
        vk_ctx.ProbeGLExtensions();
    }

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

bool GaussianSplatOpenGLContext::GetAdapterId(std::uint8_t out_id[16],
                                              std::size_t& out_size) const {
    if (!initialized_ || !glfw_window_) {
        return false;
    }
#if defined(_WIN32)
    // Find the DXGI adapter driving the monitor this (hidden) window is
    // associated with, and return its 8-byte LUID. This is used to match
    // against Vulkan's VkPhysicalDeviceIDProperties::deviceLUID so the
    // Vulkan device selected for GL_EXT_memory_object interop is guaranteed
    // to be the *same* physical GPU as this GL context — required because
    // cross-adapter memory import silently fails (GL_OUT_OF_MEMORY).
    HWND hwnd = glfwGetWin32Window(static_cast<GLFWwindow*>(glfw_window_));
    if (!hwnd) return false;
    HMONITOR mon = MonitorFromWindow(hwnd, MONITOR_DEFAULTTOPRIMARY);

    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1),
                                  reinterpret_cast<void**>(&factory))) ||
        !factory) {
        return false;
    }

    bool found = false;
    LUID luid{};
    for (UINT i = 0;; ++i) {
        IDXGIAdapter1* adapter = nullptr;
        if (factory->EnumAdapters1(i, &adapter) == DXGI_ERROR_NOT_FOUND) {
            break;
        }
        if (!adapter) continue;
        for (UINT j = 0;; ++j) {
            IDXGIOutput* output = nullptr;
            if (adapter->EnumOutputs(j, &output) == DXGI_ERROR_NOT_FOUND) {
                break;
            }
            if (!output) continue;
            DXGI_OUTPUT_DESC odesc{};
            if (SUCCEEDED(output->GetDesc(&odesc)) &&
                odesc.Monitor == mon) {
                DXGI_ADAPTER_DESC1 adesc{};
                if (SUCCEEDED(adapter->GetDesc1(&adesc))) {
                    luid = adesc.AdapterLuid;
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

    if (!found) return false;
    std::memcpy(out_id, &luid, sizeof(luid));
    out_size = sizeof(luid);
    return true;
#else
    if (GLEW_EXT_memory_object == 0) {
        return false;
    }
    GLint num_uuids = 0;
    glGetIntegerv(GL_NUM_DEVICE_UUIDS_EXT, &num_uuids);
    if (num_uuids < 1) {
        return false;
    }
    glGetUnsignedBytei_vEXT(GL_DEVICE_UUID_EXT, 0,
                            reinterpret_cast<GLubyte*>(out_id));
    if (glGetError() != GL_NO_ERROR) return false;
    out_size = 16;
    return true;
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
        utility::LogDebug("GaussianSplatOpenGLContext: {} active - GL {} on {}",
                          GetNativeBackendName(), version ? version : "?",
                          renderer ? renderer : "?");
    }

    return true;
}

void GaussianSplatOpenGLContext::ReleaseCurrent() {
    glfwMakeContextCurrent(nullptr);
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // !defined(__APPLE__)
