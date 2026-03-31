// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/filament/GaussianComputeOpenGLContext.h"

#if !defined(__APPLE__)

#include <cstdlib>
#include <cstring>

#if !defined(_WIN32)
// GLEW provides GL 4.x function pointers on Linux (GLX/EGL) the same way it
// does on Windows (WGL).  Must be included before any other GL header.
#include <GL/glew.h>

// GLX (X11) — included after glew.h so it doesn't pull in raw <GL/gl.h>.
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

#else  // _WIN32

// GLEW must be included before any other GL header on Windows.
// It provides all GL 4.x function pointers and WGL extension functions.
#include <GL/glew.h>
#include <GL/wglew.h>
#include <windows.h>

#endif  // _WIN32

#include "open3d/utility/Logging.h"

namespace open3d {
namespace visualization {
namespace rendering {

namespace {
// (Anonymous namespace reserved for future local helpers.)
}  // namespace

GaussianComputeOpenGLContext& GaussianComputeOpenGLContext::GetInstance() {
    static GaussianComputeOpenGLContext instance;
    return instance;
}

GaussianComputeOpenGLContext::~GaussianComputeOpenGLContext() { Shutdown(); }

// ---------------------------------------------------------------------------
// Platform-specific context implementations
// ---------------------------------------------------------------------------

#if !defined(_WIN32)

bool GaussianComputeOpenGLContext::Initialize() {
    if (initialized_) {
        return true;
    }

    // Called from RenderGeometryStage (after Filament engine exists).
    // InitializeStandalone() should already have been called from
    // FilamentEngine.cpp *before* Engine::create(), making this a no-op.
    // If not (e.g. non-GL backend path), try to create a compute context now.
    //
    // Filament v1.54.0 uses PlatformGLX on Linux regardless of the active
    // display server. Always try GLX first so our context shares Filament's
    // GL object namespace. On Wayland the X11/GLX connection uses XWayland.
    backend_ = Backend::kGLX;
    if (InitializeGLX()) {
        return true;
    }
    utility::LogWarning(
            "GaussianComputeOpenGLContext: GLX init failed, trying EGL.");
    backend_ = Backend::kEGL;
    return InitializeEGL();
}

bool GaussianComputeOpenGLContext::InitializeStandalone() {
    // Creates a standalone GL context meant to be called BEFORE Filament's
    // Engine::create() so the context can be passed as the sharedGLContext.
    // Filament then creates its own context sharing our GL namespace
    // (same GLX_FBCONFIG_ID, same Display).
    if (initialized_) {
        return true;
    }
    // Filament v1.54.0 uses PlatformGLX on Linux regardless of the active
    // display server (compile-time selection in PlatformFactory.cpp). Always
    // create a GLX context so the handle passed to Engine::create() matches
    // what PlatformGLX expects. On Wayland compositors, XWayland provides the
    // required X11/GLX connection.
    backend_ = Backend::kGLX;
    if (InitializeGLXStandalone()) {
        const char* session = std::getenv("XDG_SESSION_TYPE");
        if (session && std::strcmp(session, "wayland") == 0) {
            utility::LogInfo(
                    "GaussianComputeOpenGLContext: Wayland session detected; "
                    "using GLX via XWayland for Filament compatibility.");
        }
        return true;
    }
    utility::LogWarning(
            "GaussianComputeOpenGLContext: GLX standalone init failed, "
            "falling back to EGL (no Filament context sharing).");
    backend_ = Backend::kEGL;
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
    if (glXQueryContext(dpy, filament_ctx, GLX_FBCONFIG_ID, &used_fb_id) != 0 ||
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
    using CreateContextAttribsARB =
            GLXContext (*)(Display*, GLXFBConfig, GLXContext, Bool, const int*);
    auto glXCreateContextAttribsARB = reinterpret_cast<CreateContextAttribsARB>(
            glXGetProcAddressARB(reinterpret_cast<const GLubyte*>(
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
        drawable = XCreateWindow(dpy, RootWindow(dpy, vis->screen), 0, 0, 1, 1,
                                 0, vis->depth, InputOutput, vis->visual,
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
    using CreateContextAttribsARB =
            GLXContext (*)(Display*, GLXFBConfig, GLXContext, Bool, const int*);
    auto glXCreateContextAttribsARB = reinterpret_cast<CreateContextAttribsARB>(
            glXGetProcAddressARB(reinterpret_cast<const GLubyte*>(
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
    GLXPbuffer pbuf = glXCreatePbuffer(dpy, chosen_config, pbuf_attribs);
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
                glXDestroyPbuffer(dpy, static_cast<GLXPbuffer>(glx_drawable_));
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
        eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
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

// ---------------------------------------------------------------------------
// MakeCurrent / ReleaseCurrent (Linux: GLX / EGL)
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
        // Initialize GLEW on first use so all GL 4.x function pointers are
        // loaded before any compute shader dispatch.  glewExperimental=GL_TRUE
        // ensures core-profile extensions are also exposed.
        glewExperimental = GL_TRUE;
        GLenum glew_err = glewInit();
        if (glew_err != GLEW_OK) {
            utility::LogWarning(
                    "GaussianComputeOpenGLContext: glewInit warning: {}",
                    reinterpret_cast<const char*>(
                            glewGetErrorString(glew_err)));
        }
        const char* version =
                reinterpret_cast<const char*>(glGetString(GL_VERSION));
        const char* renderer =
                reinterpret_cast<const char*>(glGetString(GL_RENDERER));
        utility::LogDebug("GaussianComputeOpenGLContext: {} — GL {} on {}",
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
        eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    }
}

#else  // _WIN32

// ---------------------------------------------------------------------------
// Windows WGL backend
// ---------------------------------------------------------------------------

// WGL_ARB_create_context constants (from wglext.h / system headers).
#ifndef WGL_CONTEXT_MAJOR_VERSION_ARB
#define WGL_CONTEXT_MAJOR_VERSION_ARB 0x2091
#endif
#ifndef WGL_CONTEXT_MINOR_VERSION_ARB
#define WGL_CONTEXT_MINOR_VERSION_ARB 0x2092
#endif
#ifndef WGL_CONTEXT_PROFILE_MASK_ARB
#define WGL_CONTEXT_PROFILE_MASK_ARB 0x9126
#endif
#ifndef WGL_CONTEXT_CORE_PROFILE_BIT_ARB
#define WGL_CONTEXT_CORE_PROFILE_BIT_ARB 0x00000001
#endif

namespace {

/// Create a minimal hidden 1×1 window to obtain a valid HDC for pixel format
/// selection and WGL context creation.
HWND CreateHelperWindow() {
    WNDCLASSA wc = {};
    wc.style = CS_OWNDC;
    wc.lpfnWndProc = DefWindowProcA;
    wc.hInstance = GetModuleHandleA(nullptr);
    wc.lpszClassName = "Open3D_GS_GL_Helper";
    RegisterClassA(&wc);  // Ignore failure: class may already be registered.
    return CreateWindowExA(0, wc.lpszClassName, "", WS_POPUP, 0, 0, 1, 1,
                           nullptr, nullptr, wc.hInstance, nullptr);
}

}  // namespace

bool GaussianComputeOpenGLContext::Initialize() {
    if (initialized_) {
        return true;
    }
    // Called from RenderGeometryStage after Filament is up.
    // InitializeStandalone() should already have run from FilamentEngine.cpp
    // before Engine::create(), so this is typically a no-op.
    backend_ = Backend::kWGL;
    if (InitializeWGL()) {
        return true;
    }
    utility::LogWarning("GaussianComputeOpenGLContext: WGL init failed.");
    return false;
}

bool GaussianComputeOpenGLContext::InitializeStandalone() {
    if (initialized_) {
        return true;
    }
    // Create our GL context BEFORE Filament so it can be passed as the
    // sharedGLContext.  Filament's PlatformWGL then creates its own context
    // sharing our GL object namespace, enabling zero-copy texture import().
    backend_ = Backend::kWGL;
    if (InitializeWGLStandalone()) {
        return true;
    }
    utility::LogWarning(
            "GaussianComputeOpenGLContext: WGL standalone init failed.");
    return false;
}

void* GaussianComputeOpenGLContext::GetNativeContext() const {
    if (!initialized_) {
        return nullptr;
    }
    return glx_context_;  // HGLRC stored in glx_context_
}

// ---------------------------------------------------------------------------
// InitializeWGL — shared context (queries Filament's current HGLRC).
// ---------------------------------------------------------------------------
bool GaussianComputeOpenGLContext::InitializeWGL() {
    // Filament's context must be current on this thread when this is called
    // (after engine creation + flushAndWait).
    HGLRC filament_ctx = wglGetCurrentContext();
    HDC filament_dc = wglGetCurrentDC();
    if (!filament_ctx || !filament_dc) {
        utility::LogWarning(
                "GaussianComputeOpenGLContext: No current WGL context — "
                "Filament must be initialized and its context current.");
        return false;
    }

    using wglCreateContextAttribsARB_t = HGLRC(WINAPI*)(HDC, HGLRC, const int*);
    auto create_ctx = reinterpret_cast<wglCreateContextAttribsARB_t>(
            wglGetProcAddress("wglCreateContextAttribsARB"));
    if (!create_ctx) {
        utility::LogWarning(
                "GaussianComputeOpenGLContext: "
                "wglCreateContextAttribsARB unavailable.");
        return false;
    }

    // clang-format off
    const int ctx_attribs[] = {
        WGL_CONTEXT_MAJOR_VERSION_ARB, 4,
        WGL_CONTEXT_MINOR_VERSION_ARB, 5,
        WGL_CONTEXT_PROFILE_MASK_ARB,  WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
        0
    };
    // clang-format on

    // Create a GL 4.5 core context that SHARES Filament's object namespace.
    HGLRC ctx = create_ctx(filament_dc, filament_ctx, ctx_attribs);
    if (!ctx) {
        utility::LogWarning(
                "GaussianComputeOpenGLContext: Shared WGL context creation "
                "failed (error=0x{:08X}).",
                static_cast<unsigned>(GetLastError()));
        return false;
    }

    // Borrow the DC from Filament — do NOT ReleaseDC/DestroyWindow it.
    x_display_ = filament_dc;    // HDC
    glx_context_ = ctx;          // HGLRC
    egl_surface_ = filament_dc;  // drawable DC for MakeCurrent
    owns_display_ = false;
    initialized_ = true;
    utility::LogDebug(
            "GaussianComputeOpenGLContext: Created shared WGL context.");
    return true;
}

// ---------------------------------------------------------------------------
// InitializeWGLStandalone — create context before Filament for sharing.
// ---------------------------------------------------------------------------
bool GaussianComputeOpenGLContext::InitializeWGLStandalone() {
    // 1. Create a hidden 1×1 window to obtain a valid HDC.
    HWND hwnd = CreateHelperWindow();
    if (!hwnd) {
        utility::LogWarning(
                "GaussianComputeOpenGLContext: Failed to create helper window "
                "(WGL standalone).");
        return false;
    }
    HDC dc = GetDC(hwnd);
    if (!dc) {
        utility::LogWarning(
                "GaussianComputeOpenGLContext: GetDC failed "
                "(WGL standalone).");
        DestroyWindow(hwnd);
        return false;
    }

    // 2. Set pixel format — standard double-buffered RGBA with depth,
    //    matching what Filament expects from a shared context.
    PIXELFORMATDESCRIPTOR pfd = {};
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cDepthBits = 24;
    pfd.iLayerType = PFD_MAIN_PLANE;
    int pf = ChoosePixelFormat(dc, &pfd);
    if (!pf || !SetPixelFormat(dc, pf, &pfd)) {
        utility::LogWarning(
                "GaussianComputeOpenGLContext: SetPixelFormat failed "
                "(WGL standalone).");
        ReleaseDC(hwnd, dc);
        DestroyWindow(hwnd);
        return false;
    }

    // 3. Bootstrap: create a legacy GL context so wglGetProcAddress is
    //    available for wglCreateContextAttribsARB, and run glewInit() to
    //    load all GL 4.x + WGL extension function pointers.
    HGLRC temp_ctx = wglCreateContext(dc);
    if (!temp_ctx) {
        utility::LogWarning(
                "GaussianComputeOpenGLContext: wglCreateContext (bootstrap) "
                "failed (WGL standalone).");
        ReleaseDC(hwnd, dc);
        DestroyWindow(hwnd);
        return false;
    }
    wglMakeCurrent(dc, temp_ctx);

    // 4. Initialize GLEW — loads all GL 4.x and WGL extension pointers.
    //    Must be done with a current GL context.
    glewExperimental = GL_TRUE;
    GLenum glew_err = glewInit();
    if (glew_err != GLEW_OK) {
        // Non-fatal: missing extensions we don't use are expected.  GL core
        // functions up to 4.5 are loaded regardless.
        utility::LogWarning(
                "GaussianComputeOpenGLContext: glewInit warning: {} "
                "(WGL standalone).",
                reinterpret_cast<const char*>(glewGetErrorString(glew_err)));
    }

    using wglCreateContextAttribsARB_t = HGLRC(WINAPI*)(HDC, HGLRC, const int*);
    auto create_ctx = reinterpret_cast<wglCreateContextAttribsARB_t>(
            wglGetProcAddress("wglCreateContextAttribsARB"));

    wglMakeCurrent(nullptr, nullptr);
    wglDeleteContext(temp_ctx);

    if (!create_ctx) {
        utility::LogWarning(
                "GaussianComputeOpenGLContext: "
                "wglCreateContextAttribsARB unavailable (WGL standalone).");
        ReleaseDC(hwnd, dc);
        DestroyWindow(hwnd);
        return false;
    }

    // 5. Create the final GL 4.5 core-profile context.
    //    No sharing here — Filament creates its context sharing with ours.
    // clang-format off
    const int ctx_attribs[] = {
        WGL_CONTEXT_MAJOR_VERSION_ARB, 4,
        WGL_CONTEXT_MINOR_VERSION_ARB, 5,
        WGL_CONTEXT_PROFILE_MASK_ARB,  WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
        0
    };
    // clang-format on

    HGLRC ctx = create_ctx(dc, nullptr, ctx_attribs);
    if (!ctx) {
        utility::LogWarning(
                "GaussianComputeOpenGLContext: GL 4.5 context creation failed "
                "(WGL standalone, error=0x{:08X}).",
                static_cast<unsigned>(GetLastError()));
        ReleaseDC(hwnd, dc);
        DestroyWindow(hwnd);
        return false;
    }

    x_display_ = dc;      // HDC (our window's DC)
    glx_context_ = ctx;   // HGLRC
    egl_surface_ = hwnd;  // HWND (for cleanup on Shutdown)
    owns_display_ = true;
    initialized_ = true;
    utility::LogDebug(
            "GaussianComputeOpenGLContext: Standalone WGL GL 4.5 context "
            "created.  Pass to Filament as sharedContext.");
    return true;
}

// ---------------------------------------------------------------------------
// Shutdown / MakeCurrent / ReleaseCurrent (Windows WGL)
// ---------------------------------------------------------------------------
void GaussianComputeOpenGLContext::Shutdown() {
    if (!initialized_) {
        return;
    }

    auto ctx = static_cast<HGLRC>(glx_context_);
    auto dc = static_cast<HDC>(x_display_);
    wglMakeCurrent(nullptr, nullptr);
    if (ctx) {
        wglDeleteContext(ctx);
        glx_context_ = nullptr;
    }
    if (owns_display_) {
        auto hwnd = static_cast<HWND>(egl_surface_);
        if (dc && hwnd) {
            ReleaseDC(hwnd, dc);
        }
        if (hwnd) {
            DestroyWindow(hwnd);
        }
        egl_surface_ = nullptr;
    }
    x_display_ = nullptr;
    backend_ = Backend::kNone;
    initialized_ = false;
    gl_logged_ = false;
    utility::LogDebug("GaussianComputeOpenGLContext: Shut down (WGL).");
}

bool GaussianComputeOpenGLContext::MakeCurrent() {
    if (!initialized_) {
        return false;
    }
    auto dc = static_cast<HDC>(x_display_);
    auto ctx = static_cast<HGLRC>(glx_context_);
    bool ok = (wglMakeCurrent(dc, ctx) != FALSE);
    if (!ok) {
        utility::LogWarning(
                "GaussianComputeOpenGLContext: wglMakeCurrent failed "
                "(error=0x{:08X}).",
                static_cast<unsigned>(GetLastError()));
        return false;
    }
    if (!gl_logged_) {
        gl_logged_ = true;
        const char* version =
                reinterpret_cast<const char*>(glGetString(GL_VERSION));
        const char* renderer =
                reinterpret_cast<const char*>(glGetString(GL_RENDERER));
        utility::LogDebug("GaussianComputeOpenGLContext: WGL — GL {} on {}",
                          version ? version : "?", renderer ? renderer : "?");
    }
    return true;
}

void GaussianComputeOpenGLContext::ReleaseCurrent() {
    if (!initialized_) {
        return;
    }
    wglMakeCurrent(nullptr, nullptr);
}

#endif  // _WIN32

// ---------------------------------------------------------------------------
// Shared across all platforms
// ---------------------------------------------------------------------------

bool GaussianComputeOpenGLContext::IsValid() const { return initialized_; }

void GaussianComputeOpenGLContext::Finish() { glFinish(); }

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // !defined(__APPLE__)
