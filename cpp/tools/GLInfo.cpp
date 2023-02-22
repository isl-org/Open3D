// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/Open3D.h"

void GLFWErrorCallback(int error, const char *description) {
    open3d::utility::LogWarning("GLFW Error: {}", description);
}

void TryGLVersion(int major,
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
        utility::LogError("Failed to initialize GLFW");
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
        return;
    } else {
        glfwMakeContextCurrent(window_);
    }

    auto reportGlStringFunc = [](GLenum id, std::string name) {
        const auto r = glGetString(id);
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
}

int main(int argc, char **argv) {
    TryGLVersion(1, 0, false, false, GLFW_OPENGL_ANY_PROFILE);
    TryGLVersion(3, 2, true, true, GLFW_OPENGL_CORE_PROFILE);
    TryGLVersion(4, 1, false, false, GLFW_OPENGL_ANY_PROFILE);
    TryGLVersion(3, 3, false, true, GLFW_OPENGL_CORE_PROFILE);
    TryGLVersion(3, 3, true, true, GLFW_OPENGL_CORE_PROFILE);
    TryGLVersion(3, 3, false, true, GLFW_OPENGL_COMPAT_PROFILE);
    TryGLVersion(3, 3, false, true, GLFW_OPENGL_ANY_PROFILE);
    TryGLVersion(1, 0, false, true, GLFW_OPENGL_ANY_PROFILE);
    return 0;
}
