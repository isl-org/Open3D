// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/Open3D.h"

#if defined(__linux__)
#include <EGL/egl.h>
#endif

void GLFWErrorCallback(int error, const char *description) {
    open3d::utility::LogWarning("GLFW Error: {}", description);
}

#if defined(__linux__)
// Minimal EGL offscreen context used only to report GPU info when no
// windowing system display is available (mirrors the fallback used by
// open3d::visualization::Visualizer::CreateVisualizerWindow()).
void ReportEGLGLInfo() {
    using namespace open3d;

    EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    EGLint major = 0, minor = 0;
    if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor)) {
        utility::LogWarning(
                "GLInfo: no display and no EGL device available; cannot "
                "query GPU info.");
        return;
    }
    utility::LogInfo("GLInfo: using EGL {}.{} offscreen context", major, minor);

    eglBindAPI(EGL_OPENGL_API);
    const EGLint config_attribs[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                                     EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
                                     EGL_NONE};
    EGLConfig config;
    EGLint num_configs = 0;
    if (!eglChooseConfig(display, config_attribs, &config, 1, &num_configs) ||
        num_configs == 0) {
        utility::LogWarning("GLInfo: eglChooseConfig failed.");
        return;
    }
    const EGLint pbuffer_attribs[] = {EGL_WIDTH, 640, EGL_HEIGHT, 480,
                                      EGL_NONE};
    EGLSurface surface =
            eglCreatePbufferSurface(display, config, pbuffer_attribs);
    const EGLint context_attribs[] = {EGL_CONTEXT_MAJOR_VERSION, 3,
                                      EGL_CONTEXT_MINOR_VERSION, 3, EGL_NONE};
    EGLContext context =
            eglCreateContext(display, config, EGL_NO_CONTEXT, context_attribs);
    if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT ||
        !eglMakeCurrent(display, surface, surface, context)) {
        utility::LogWarning("GLInfo: failed to create/activate EGL context.");
        return;
    }

    utility::LogInfo("GL_VERSION:\t{}",
                     reinterpret_cast<const char *>(glGetString(GL_VERSION)));
    utility::LogInfo("GL_RENDERER:\t{}",
                     reinterpret_cast<const char *>(glGetString(GL_RENDERER)));
    utility::LogInfo("GL_VENDOR:\t{}",
                     reinterpret_cast<const char *>(glGetString(GL_VENDOR)));

    eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroyContext(display, context);
    eglDestroySurface(display, surface);
    eglTerminate(display);
}
#endif  // defined(__linux__)

// Returns false if GLFW could not create a window (e.g. no display), in
// which case the caller should fall back to an offscreen EGL context.
bool TryGLVersion(int major,
                  int minor,
                  bool forwardCompat,
                  bool setProfile,
                  int profileId) {
    using namespace open3d;
    using namespace visualization;

    std::string forwardCompatStr =
            (forwardCompat ? "GLFW_OPENGL_FORWARD_COMPAT " : "");
    std::string profileStr = "UnknownProfile";
    // Some versions of GCC reports Wstringop-overflow error if string storage
    // is not reserved. This might be related to a bug reported here:
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=96963
    profileStr.reserve(32);
#define OPEN3D_CHECK_PROFILESTR(p) \
    if (profileId == p) {          \
        profileStr = #p;           \
    }
    OPEN3D_CHECK_PROFILESTR(GLFW_OPENGL_CORE_PROFILE);
    OPEN3D_CHECK_PROFILESTR(GLFW_OPENGL_COMPAT_PROFILE);
    OPEN3D_CHECK_PROFILESTR(GLFW_OPENGL_ANY_PROFILE);
#undef OPEN3D_CHECK_PROFILESTR

    utility::LogInfo("TryGLVersion: {:d}.{:d} {}{}", major, minor,
                     forwardCompatStr, profileStr);

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    glfwSetErrorCallback(GLFWErrorCallback);
    if (!glfwInit()) {
        utility::LogDebug("Failed to initialize GLFW (no display available)");
        return false;
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, major);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, minor);
    if (forwardCompat) glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    if (setProfile) glfwWindowHint(GLFW_OPENGL_PROFILE, profileId);
    glfwWindowHint(GLFW_VISIBLE, 0);

    GLFWwindow *window_ = glfwCreateWindow(640, 480, "GLInfo", NULL, NULL);
    if (!window_) {
        utility::LogDebug("Failed to create window");
        glfwTerminate();
        return true;  // GLFW initialized fine; this GL version is just
                      // unsupported.
    } else {
        glfwMakeContextCurrent(window_);
    }

    auto reportGlStringFunc = [](GLenum id, std::string name) {
// Note: with GLFW 3.3.9 it appears that OpenGL entry points are no longer auto
// loaded? The else part crashes on Apple with a null pointer.
#ifdef __APPLE__
        PFNGLGETSTRINGIPROC _glGetString =
                (PFNGLGETSTRINGIPROC)glfwGetProcAddress("glGetString");
        const auto r = _glGetString(id, 0);
#else
        const auto r = glGetString(id);
#endif
        if (!r) {
            utility::LogWarning("Unable to get info on {} id {:d}", name, id);
        } else {
            utility::LogDebug("{}:\t{}", name,
                              reinterpret_cast<const char *>(r));
        }
    };
#define OPEN3D_REPORT_GL_STRING(n) reportGlStringFunc(n, #n)
    OPEN3D_REPORT_GL_STRING(GL_VERSION);
    OPEN3D_REPORT_GL_STRING(GL_RENDERER);
    OPEN3D_REPORT_GL_STRING(GL_VENDOR);
    OPEN3D_REPORT_GL_STRING(GL_SHADING_LANGUAGE_VERSION);
    // OPEN3D_REPORT_GL_STRING(GL_EXTENSIONS);
#undef OPEN3D_REPORT_GL_STRING

    if (window_) glfwDestroyWindow(window_);
    glfwTerminate();
    return true;
}

int main(int argc, char **argv) {
    bool has_display =
            TryGLVersion(1, 0, false, false, GLFW_OPENGL_ANY_PROFILE);
    if (has_display) {
        TryGLVersion(3, 2, true, true, GLFW_OPENGL_CORE_PROFILE);
        TryGLVersion(4, 1, false, false, GLFW_OPENGL_ANY_PROFILE);
        TryGLVersion(3, 3, false, true, GLFW_OPENGL_CORE_PROFILE);
        TryGLVersion(3, 3, true, true, GLFW_OPENGL_CORE_PROFILE);
        TryGLVersion(3, 3, false, true, GLFW_OPENGL_COMPAT_PROFILE);
        TryGLVersion(3, 3, false, true, GLFW_OPENGL_ANY_PROFILE);
        TryGLVersion(1, 0, false, true, GLFW_OPENGL_ANY_PROFILE);
    } else {
#if defined(__linux__)
        // No windowing system display; report GPU info via an offscreen EGL
        // context instead (used e.g. in CI to validate headless rendering).
        ReportEGLGLInfo();
#else
        open3d::utility::LogWarning(
                "GLInfo: no display available and no offscreen fallback on "
                "this platform.");
#endif
    }
    return 0;
}
