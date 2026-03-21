// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/filament/GaussianComputeOpenGLContext.h"

#if !defined(__APPLE__)

#include <cstdlib>
#include <cstring>

// GL function prototypes — works with both GLX and EGL contexts.
// Must be defined before any GL header is included.
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>

// GLX (X11)
#include <GL/glx.h>
#include <X11/Xlib.h>

// GLX_ARB_create_context constants (from glxext.h / system headers).
#ifndef GLX_CONTEXT_MAJOR_VERSION_ARB
#define GLX_CONTEXT_MAJOR_VERSION_ARB 0x2091
#endif
#ifndef GLX_CONTEXT_MINOR_VERSION_ARB
#define GLX_CONTEXT_MINOR_VERSION_ARB 0x2092
#endif
#ifndef GLX_CONTEXT_PROFILE_MASK_ARB
#define GLX_CONTEXT_PROFILE_MASK_ARB 0x9126
#endif
#ifndef GLX_CONTEXT_CORE_PROFILE_BIT_ARB
#define GLX_CONTEXT_CORE_PROFILE_BIT_ARB 0x00000001
#endif

// EGL (Wayland)
#include <EGL/egl.h>

#include "open3d/utility/Logging.h"

namespace open3d {
namespace visualization {
namespace rendering {

namespace {

/// Detect whether to use GLX (X11) or EGL (Wayland) at runtime.
GaussianComputeOpenGLContext::Backend DetectBackend() {
    // Prefer XDG_SESSION_TYPE (most reliable on modern Linux).
    const char* session = std::getenv("XDG_SESSION_TYPE");
    if (session) {
        if (std::strcmp(session, "wayland") == 0) {
            return GaussianComputeOpenGLContext::Backend::kEGL;
        }
        if (std::strcmp(session, "x11") == 0) {
            return GaussianComputeOpenGLContext::Backend::kGLX;
        }
    }
    // Fallback to display environment variables.
    if (std::getenv("WAYLAND_DISPLAY")) {
        return GaussianComputeOpenGLContext::Backend::kEGL;
    }
    if (std::getenv("DISPLAY")) {
        return GaussianComputeOpenGLContext::Backend::kGLX;
    }
    // Default to EGL (more portable).
    return GaussianComputeOpenGLContext::Backend::kEGL;
}

}  // namespace

GaussianComputeOpenGLContext& GaussianComputeOpenGLContext::GetInstance() {
    static GaussianComputeOpenGLContext instance;
    return instance;
}

GaussianComputeOpenGLContext::~GaussianComputeOpenGLContext() { Shutdown(); }

bool GaussianComputeOpenGLContext::Initialize() {
    if (initialized_) {
        return true;
    }

    backend_ = DetectBackend();
    if (backend_ == Backend::kGLX) {
        if (InitializeGLX()) {
            return true;
        }
        utility::LogWarning(
                "GaussianComputeOpenGLContext: GLX init failed, trying EGL.");
        backend_ = Backend::kEGL;
    }
    return InitializeEGL();
}

// ---------------------------------------------------------------------------
// GLX backend (X11)
// ---------------------------------------------------------------------------
bool GaussianComputeOpenGLContext::InitializeGLX() {
    Display* dpy = XOpenDisplay(nullptr);
    if (!dpy) {
        utility::LogWarning(
                "GaussianComputeOpenGLContext: XOpenDisplay failed.");
        return false;
    }

    // Choose an FBConfig that supports pbuffer + RGBA.
    // clang-format off
    const int fb_attribs[] = {
        GLX_DRAWABLE_TYPE, GLX_PBUFFER_BIT,
        GLX_RENDER_TYPE,   GLX_RGBA_BIT,
        GLX_RED_SIZE,      8,
        GLX_GREEN_SIZE,    8,
        GLX_BLUE_SIZE,     8,
        None
    };
    // clang-format on

    int fb_count = 0;
    GLXFBConfig* fbc =
            glXChooseFBConfig(dpy, DefaultScreen(dpy), fb_attribs, &fb_count);
    if (!fbc || fb_count == 0) {
        utility::LogWarning(
                "GaussianComputeOpenGLContext: glXChooseFBConfig failed.");
        XCloseDisplay(dpy);
        return false;
    }

    // Load glXCreateContextAttribsARB (GLX_ARB_create_context extension).
    using CreateContextAttribsARB = GLXContext (*)(Display*, GLXFBConfig,
                                                   GLXContext, Bool,
                                                   const int*);
    auto glXCreateContextAttribsARB =
            reinterpret_cast<CreateContextAttribsARB>(glXGetProcAddressARB(
                    reinterpret_cast<const GLubyte*>(
                            "glXCreateContextAttribsARB")));
    if (!glXCreateContextAttribsARB) {
        utility::LogWarning(
                "GaussianComputeOpenGLContext: "
                "glXCreateContextAttribsARB not available.");
        XFree(fbc);
        XCloseDisplay(dpy);
        return false;
    }

    // Request an OpenGL 4.5 core profile context.
    // clang-format off
    const int ctx_attribs[] = {
        GLX_CONTEXT_MAJOR_VERSION_ARB, 4,
        GLX_CONTEXT_MINOR_VERSION_ARB, 5,
        GLX_CONTEXT_PROFILE_MASK_ARB,  GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
        None
    };
    // clang-format on

    GLXContext ctx = glXCreateContextAttribsARB(dpy, fbc[0], nullptr, True,
                                                ctx_attribs);
    if (!ctx) {
        utility::LogWarning(
                "GaussianComputeOpenGLContext: GLX context creation failed.");
        XFree(fbc);
        XCloseDisplay(dpy);
        return false;
    }

    // Create a 1×1 GLX pbuffer (needed to make context current).
    const int pbuf_attribs[] = {GLX_PBUFFER_WIDTH, 1, GLX_PBUFFER_HEIGHT, 1,
                                None};
    GLXPbuffer pbuf = glXCreatePbuffer(dpy, fbc[0], pbuf_attribs);
    XFree(fbc);

    if (!pbuf) {
        utility::LogWarning(
                "GaussianComputeOpenGLContext: glXCreatePbuffer failed.");
        glXDestroyContext(dpy, ctx);
        XCloseDisplay(dpy);
        return false;
    }

    x_display_ = dpy;
    glx_context_ = ctx;
    glx_drawable_ = pbuf;
    initialized_ = true;

    utility::LogDebug(
            "GaussianComputeOpenGLContext: Created GLX context (X11).");
    return true;
}

// ---------------------------------------------------------------------------
// EGL backend (Wayland)
// ---------------------------------------------------------------------------
bool GaussianComputeOpenGLContext::InitializeEGL() {
    EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (display == EGL_NO_DISPLAY) {
        utility::LogWarning(
                "GaussianComputeOpenGLContext: eglGetDisplay failed.");
        return false;
    }

    EGLint major = 0, minor = 0;
    if (!eglInitialize(display, &major, &minor)) {
        utility::LogWarning(
                "GaussianComputeOpenGLContext: eglInitialize failed.");
        return false;
    }

    if (!eglBindAPI(EGL_OPENGL_API)) {
        utility::LogWarning(
                "GaussianComputeOpenGLContext: eglBindAPI(EGL_OPENGL_API) "
                "failed.");
        return false;
    }

    // clang-format off
    const EGLint config_attribs[] = {
        EGL_SURFACE_TYPE,    EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_RED_SIZE,        8,
        EGL_GREEN_SIZE,      8,
        EGL_BLUE_SIZE,       8,
        EGL_NONE
    };
    // clang-format on

    EGLConfig config = nullptr;
    EGLint num_configs = 0;
    if (!eglChooseConfig(display, config_attribs, &config, 1, &num_configs) ||
        num_configs == 0) {
        utility::LogWarning(
                "GaussianComputeOpenGLContext: eglChooseConfig failed.");
        return false;
    }

    const EGLint pbuffer_attribs[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
    EGLSurface surface =
            eglCreatePbufferSurface(display, config, pbuffer_attribs);
    if (surface == EGL_NO_SURFACE) {
        utility::LogWarning(
                "GaussianComputeOpenGLContext: eglCreatePbufferSurface "
                "failed.");
        return false;
    }

    // clang-format off
    const EGLint context_attribs[] = {
        EGL_CONTEXT_MAJOR_VERSION,       4,
        EGL_CONTEXT_MINOR_VERSION,       5,
        EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
        EGL_NONE
    };
    // clang-format on

    EGLContext context =
            eglCreateContext(display, config, EGL_NO_CONTEXT, context_attribs);
    if (context == EGL_NO_CONTEXT) {
        utility::LogWarning(
                "GaussianComputeOpenGLContext: eglCreateContext failed. "
                "EGL error: 0x{:04X}.",
                eglGetError());
        eglDestroySurface(display, surface);
        return false;
    }

    egl_display_ = display;
    egl_context_ = context;
    egl_surface_ = surface;
    initialized_ = true;

    utility::LogDebug(
            "GaussianComputeOpenGLContext: Created EGL context (EGL {}.{}).",
            major, minor);
    return true;
}

// ---------------------------------------------------------------------------
// Shutdown
// ---------------------------------------------------------------------------
void GaussianComputeOpenGLContext::Shutdown() {
    if (!initialized_) {
        return;
    }

    if (backend_ == Backend::kGLX) {
        auto dpy = static_cast<Display*>(x_display_);
        glXMakeContextCurrent(dpy, None, None, nullptr);
        if (glx_drawable_) {
            glXDestroyPbuffer(dpy, glx_drawable_);
            glx_drawable_ = 0;
        }
        if (glx_context_) {
            glXDestroyContext(dpy, static_cast<GLXContext>(glx_context_));
            glx_context_ = nullptr;
        }
        if (dpy) {
            XCloseDisplay(dpy);
            x_display_ = nullptr;
        }
    } else {
        auto display = static_cast<EGLDisplay>(egl_display_);
        eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE,
                        EGL_NO_CONTEXT);
        if (egl_context_) {
            eglDestroyContext(display, static_cast<EGLContext>(egl_context_));
            egl_context_ = nullptr;
        }
        if (egl_surface_) {
            eglDestroySurface(display, static_cast<EGLSurface>(egl_surface_));
            egl_surface_ = nullptr;
        }
        egl_display_ = nullptr;
    }

    backend_ = Backend::kNone;
    initialized_ = false;
    gl_logged_ = false;
    utility::LogDebug("GaussianComputeOpenGLContext: Shut down.");
}

bool GaussianComputeOpenGLContext::IsValid() const { return initialized_; }

// ---------------------------------------------------------------------------
// MakeCurrent / ReleaseCurrent / Finish
// ---------------------------------------------------------------------------
bool GaussianComputeOpenGLContext::MakeCurrent() {
    if (!initialized_) {
        return false;
    }

    bool ok = false;
    if (backend_ == Backend::kGLX) {
        auto dpy = static_cast<Display*>(x_display_);
        auto ctx = static_cast<GLXContext>(glx_context_);
        ok = glXMakeContextCurrent(dpy, glx_drawable_, glx_drawable_, ctx);
        if (!ok) {
            utility::LogWarning(
                    "GaussianComputeOpenGLContext: glXMakeContextCurrent "
                    "failed.");
        }
    } else {
        auto display = static_cast<EGLDisplay>(egl_display_);
        auto surface = static_cast<EGLSurface>(egl_surface_);
        auto context = static_cast<EGLContext>(egl_context_);
        ok = eglMakeCurrent(display, surface, surface, context);
        if (!ok) {
            utility::LogWarning(
                    "GaussianComputeOpenGLContext: eglMakeCurrent failed. "
                    "EGL error: 0x{:04X}.",
                    eglGetError());
        }
    }

    if (ok && !gl_logged_) {
        gl_logged_ = true;
        const char* version =
                reinterpret_cast<const char*>(glGetString(GL_VERSION));
        const char* renderer =
                reinterpret_cast<const char*>(glGetString(GL_RENDERER));
        utility::LogDebug(
                "GaussianComputeOpenGLContext: {} — GL {} on {}",
                backend_ == Backend::kGLX ? "GLX" : "EGL",
                version ? version : "?", renderer ? renderer : "?");
    }

    return ok;
}

void GaussianComputeOpenGLContext::ReleaseCurrent() {
    if (!initialized_) {
        return;
    }

    if (backend_ == Backend::kGLX) {
        auto dpy = static_cast<Display*>(x_display_);
        glXMakeContextCurrent(dpy, None, None, nullptr);
    } else {
        auto display = static_cast<EGLDisplay>(egl_display_);
        eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE,
                        EGL_NO_CONTEXT);
    }
}

void GaussianComputeOpenGLContext::Finish() { glFinish(); }

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // !defined(__APPLE__)
