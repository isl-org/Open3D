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

    // Called from RenderGeometryStage (after Filament engine exists).
    // InitializeStandalone() should already have been called from
    // FilamentEngine.cpp *before* Engine::create(), making this a no-op.
    // If not (e.g. non-GL backend path), fall back to GLX/EGL independent
    // init for compute-only work.
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

bool GaussianComputeOpenGLContext::InitializeStandalone() {
    // Creates a standalone GL context (no Filament sharing required), meant
    // to be called BEFORE Filament's Engine::create() so the context can be
    // passed as the sharedGLContext.  Filament will then create its own
    // context sharing our GL namespace (same GLX_FBCONFIG_ID, same Display).
    if (initialized_) {
        return true;
    }
    backend_ = DetectBackend();
    if (backend_ == Backend::kGLX) {
        if (InitializeGLXStandalone()) {
            return true;
        }
        utility::LogWarning(
                "GaussianComputeOpenGLContext: GLX standalone init failed, "
                "trying EGL.");
        backend_ = Backend::kEGL;
    }
    return InitializeEGL();
}

void* GaussianComputeOpenGLContext::GetNativeContext() const {
    if (!initialized_) {
        return nullptr;
    }
    // Return the underlying native context handle so it can be passed to
    // Filament's Engine::create() as the shared GL context.
    if (backend_ == Backend::kGLX) {
        return glx_context_;  // GLXContext (opaque pointer)
    } else {
        return egl_context_;  // EGLContext (opaque pointer)
    }
}

// ---------------------------------------------------------------------------
// GLX backend (X11)
// ---------------------------------------------------------------------------
bool GaussianComputeOpenGLContext::InitializeGLX() {
    // Create a GLX context that SHARES Filament's GL object namespace.
    // This is required so that GL textures created in our context are
    // visible to Filament (for zero-copy import()) and vice-versa.
    //
    // Strategy: Filament's context must be current when this is called
    // (after engine creation + flushAndWait).  We query its FBConfig ID,
    // find the matching GLXFBConfig, and create our context sharing with it.
    // This is exactly what Filament's own PlatformGLX does when it receives
    // a sharedGLContext argument.

    // 1. Get Filament's current context and display.
    GLXContext filament_ctx = glXGetCurrentContext();
    Display* dpy = glXGetCurrentDisplay();
    if (!filament_ctx || !dpy) {
        utility::LogWarning(
                "GaussianComputeOpenGLContext: No current GLX context — "
                "Filament must be initialized and its context current.");
        return false;
    }

    // 2. Query the FBConfig ID that Filament's context uses.
    int used_fb_id = -1;
    if (glXQueryContext(dpy, filament_ctx, GLX_FBCONFIG_ID, &used_fb_id) !=
                0 ||
        used_fb_id < 0) {
        utility::LogWarning(
                "GaussianComputeOpenGLContext: Failed to query "
                "GLX_FBCONFIG_ID from Filament's context.");
        return false;
    }

    // 3. Find the matching FBConfig from all available configs.
    int num_configs = 0;
    GLXFBConfig* all_configs =
            glXGetFBConfigs(dpy, DefaultScreen(dpy), &num_configs);
    if (!all_configs || num_configs == 0) {
        utility::LogWarning(
                "GaussianComputeOpenGLContext: glXGetFBConfigs failed.");
        return false;
    }

    GLXFBConfig matched_config = nullptr;
    for (int i = 0; i < num_configs; ++i) {
        int fb_id = -1;
        if (glXGetFBConfigAttrib(dpy, all_configs[i], GLX_FBCONFIG_ID,
                                 &fb_id) == 0 &&
            fb_id == used_fb_id) {
            matched_config = all_configs[i];
            break;
        }
    }
    XFree(all_configs);

    if (!matched_config) {
        utility::LogWarning(
                "GaussianComputeOpenGLContext: No FBConfig matches "
                "Filament's GLX_FBCONFIG_ID={}.",
                used_fb_id);
        return false;
    }

    // Log the matched FBConfig's capabilities.
    {
        int depth_size = 0, drawable_type = 0;
        glXGetFBConfigAttrib(dpy, matched_config, GLX_DEPTH_SIZE, &depth_size);
        glXGetFBConfigAttrib(dpy, matched_config, GLX_DRAWABLE_TYPE,
                             &drawable_type);
        utility::LogDebug(
                "GaussianComputeOpenGLContext: Matched FBConfig id={} "
                "depth={} drawable_type=0x{:X}",
                used_fb_id, depth_size, drawable_type);
    }

    // 4. Load glXCreateContextAttribsARB.
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
                "glXCreateContextAttribsARB unavailable.");
        return false;
    }

    // 5. Create an OpenGL 4.5 core-profile context SHARED with Filament.
    // clang-format off
    const int ctx_attribs[] = {
        GLX_CONTEXT_MAJOR_VERSION_ARB, 4,
        GLX_CONTEXT_MINOR_VERSION_ARB, 5,
        GLX_CONTEXT_PROFILE_MASK_ARB,  GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
        None
    };
    // clang-format on

    GLXContext ctx = glXCreateContextAttribsARB(
            dpy, matched_config, filament_ctx, True, ctx_attribs);
    if (!ctx) {
        utility::LogWarning(
                "GaussianComputeOpenGLContext: Shared GLX context "
                "creation failed.");
        return false;
    }

    // 6. Create a small offscreen X11 window as the GLX drawable.
    // Using a window (not a deprecated PBuffer) with the matched FBConfig's
    // visual.  The window is never mapped/displayed; it only provides a
    // valid GLX drawable for glXMakeContextCurrent.
    XVisualInfo* vis = glXGetVisualFromFBConfig(dpy, matched_config);
    Window drawable = 0;
    if (vis) {
        XSetWindowAttributes swa;
        swa.colormap = XCreateColormap(dpy, RootWindow(dpy, vis->screen),
                                       vis->visual, AllocNone);
        swa.border_pixel = 0;
        drawable = XCreateWindow(
                dpy, RootWindow(dpy, vis->screen),
                0, 0, 1, 1, 0,
                vis->depth, InputOutput, vis->visual,
                CWColormap | CWBorderPixel, &swa);
        XFree(vis);
    }

    if (!drawable) {
        utility::LogWarning(
                "GaussianComputeOpenGLContext: "
                "Failed to create offscreen X11 window.");
        glXDestroyContext(dpy, ctx);
        return false;
    }

    // We borrow the Display* from Filament/GLFW — do NOT XCloseDisplay it.
    x_display_ = dpy;
    glx_context_ = ctx;
    glx_drawable_ = drawable;
    owns_display_ = false;
    initialized_ = true;

    utility::LogDebug(
            "GaussianComputeOpenGLContext: Created shared GLX context "
            "(FBConfig ID={}).",
            used_fb_id);
    return true;
}

bool GaussianComputeOpenGLContext::InitializeGLXStandalone() {
    // Called BEFORE Filament's Engine::create().  Creates our compute context
    // independently (no shareContext), then passes it to Filament so Filament
    // creates its own context sharing ours — establishing a shared GL object
    // namespace for zero-copy texture import().
    //
    // FBConfig requirements:
    //  • GLX_WINDOW_BIT  – Filament renders to the GLFW window; the FBConfig
    //    must be window-compatible so glXMakeContextCurrent(…, window, …)
    //    succeeds after Filament matches our config.
    //  • GLX_PBUFFER_BIT – Filament creates a dummy PBuffer for its idle
    //    surface; PlatformGLX::createDriver requires this.
    //  • DOUBLEBUFFER, DEPTH_SIZE 24 – standard rendering requirements that
    //    match Filament's own defaults.

    Display* dpy = XOpenDisplay(nullptr);
    if (!dpy) {
        utility::LogWarning(
                "GaussianComputeOpenGLContext: XOpenDisplay failed "
                "(standalone GLX init).");
        return false;
    }

    // clang-format off
    const int fb_attribs[] = {
        GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT | GLX_PBUFFER_BIT,
        GLX_RENDER_TYPE,   GLX_RGBA_BIT,
        // GLX_DOUBLEBUFFER,  True,  // Slows down rendering significantly
        GLX_RED_SIZE,      8,
        GLX_GREEN_SIZE,    8,
        GLX_BLUE_SIZE,     8,
        GLX_DEPTH_SIZE,    24,
        None
    };
    // clang-format on

    int num_configs = 0;
    GLXFBConfig* configs = glXChooseFBConfig(dpy, DefaultScreen(dpy),
                                             fb_attribs, &num_configs);
    if (!configs || num_configs == 0) {
        utility::LogWarning(
                "GaussianComputeOpenGLContext: glXChooseFBConfig failed "
                "(standalone GLX init).");
        XCloseDisplay(dpy);
        return false;
    }
    GLXFBConfig chosen_config = configs[0];

    // Log the config ID so we can verify PlatformGLX matches it.
    int fb_id = -1;
    glXGetFBConfigAttrib(dpy, chosen_config, GLX_FBCONFIG_ID, &fb_id);
    utility::LogDebug(
            "GaussianComputeOpenGLContext: Standalone GLX FBConfig ID={}.",
            fb_id);
    XFree(configs);

    // Load glXCreateContextAttribsARB.
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
                "glXCreateContextAttribsARB unavailable "
                "(standalone GLX init).");
        XCloseDisplay(dpy);
        return false;
    }

    // Create an OpenGL 4.5 core-profile context — no sharing yet.
    // Filament will create ITS context sharing with THIS one.
    // clang-format off
    const int ctx_attribs[] = {
        GLX_CONTEXT_MAJOR_VERSION_ARB, 4,
        GLX_CONTEXT_MINOR_VERSION_ARB, 5,
        GLX_CONTEXT_PROFILE_MASK_ARB,  GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
        None
    };
    // clang-format on

    GLXContext ctx = glXCreateContextAttribsARB(dpy, chosen_config,
                                                nullptr,  // no share context
                                                True, ctx_attribs);
    if (!ctx) {
        utility::LogWarning(
                "GaussianComputeOpenGLContext: GL 4.5 context creation "
                "failed (standalone GLX init).");
        XCloseDisplay(dpy);
        return false;
    }

    // Create a 1×1 PBuffer as our offscreen drawable.
    const int pbuf_attribs[] = {GLX_PBUFFER_WIDTH, 1, GLX_PBUFFER_HEIGHT, 1,
                                 None};
    GLXPbuffer pbuf =
            glXCreatePbuffer(dpy, chosen_config, pbuf_attribs);
    if (!pbuf) {
        utility::LogWarning(
                "GaussianComputeOpenGLContext: PBuffer creation failed "
                "(standalone GLX init).");
        glXDestroyContext(dpy, ctx);
        XCloseDisplay(dpy);
        return false;
    }

    x_display_ = dpy;
    glx_context_ = ctx;
    glx_drawable_ = pbuf;
    owns_display_ = true;  // we opened this display, close it on Shutdown
    initialized_ = true;

    utility::LogDebug(
            "GaussianComputeOpenGLContext: Standalone GLX context created "
            "(FBConfig ID={}).  Pass to Filament as sharedContext.",
            fb_id);
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
            if (owns_display_) {
                // Standalone init: drawable is a PBuffer.
                glXDestroyPbuffer(dpy,
                                  static_cast<GLXPbuffer>(glx_drawable_));
            } else {
                // Shared init: drawable is an offscreen X11 window.
                XDestroyWindow(dpy, static_cast<Window>(glx_drawable_));
            }
            glx_drawable_ = 0;
        }
        if (glx_context_) {
            glXDestroyContext(dpy, static_cast<GLXContext>(glx_context_));
            glx_context_ = nullptr;
        }
        // Only close the display if we opened it ourselves.
        if (dpy && owns_display_) {
            XCloseDisplay(dpy);
        }
        x_display_ = nullptr;
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
