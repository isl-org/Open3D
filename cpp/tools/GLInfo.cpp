// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/Open3D.h"

#if defined(__linux__)
#include "open3d/visualization/visualizer/EGLOffscreenContext.h"
#endif

void GLFWErrorCallback(int error, const char *description) {
    open3d::utility::LogWarning("GLFW Error: {}", description);
}

#if defined(__linux__)
// Report GPU info via the same offscreen EGL context that
// open3d::visualization::Visualizer::CreateVisualizerWindow() falls back to
// when no windowing system display is available, so this reflects what the
// Visualizer will actually use.
void ReportEGLGLInfo() {
    using namespace open3d;

    auto context = visualization::EGLOffscreenContext::Create(640, 480);
    if (!context) {
        utility::LogWarning(
                "GLInfo: no display and no EGL device available; cannot "
                "query GPU info.");
        return;
    }

    utility::LogInfo("GL_VERSION:\t{}",
                     reinterpret_cast<const char *>(glGetString(GL_VERSION)));
    utility::LogInfo("GL_RENDERER:\t{}",
                     reinterpret_cast<const char *>(glGetString(GL_RENDERER)));
    utility::LogInfo("GL_VENDOR:\t{}",
                     reinterpret_cast<const char *>(glGetString(GL_VENDOR)));
}
#endif  // defined(__linux__)

// Returns false only if GLFW itself failed to initialize (e.g. no display),
// in which case the caller should fall back to an offscreen EGL context.
// Returns true if GLFW initialized successfully, even if glfwCreateWindow()
// failed for this specific GL version/profile combination (unsupported by
// the driver) -- that is a per-version failure, not a "no display" signal.
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
