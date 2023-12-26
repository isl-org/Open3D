// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/shader/ShaderWrapper.h"

#include "open3d/geometry/Geometry.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace visualization {

namespace glsl {

bool ShaderWrapper::Render(const geometry::Geometry &geometry,
                           const RenderOption &option,
                           const ViewControl &view) {
    if (!compiled_) {
        Compile();
    }
    if (!bound_) {
        BindGeometry(geometry, option, view);
    }
    if (!compiled_ || !bound_) {
        PrintShaderWarning("Something is wrong in compiling or binding.");
        return false;
    }
    return RenderGeometry(geometry, option, view);
}

void ShaderWrapper::InvalidateGeometry() {
    if (bound_) {
        UnbindGeometry();
    }
}

void ShaderWrapper::PrintShaderWarning(const std::string &message) const {
    utility::LogWarning("[{}] {}", GetShaderName(), message);
}

bool ShaderWrapper::CompileShaders(const char *const vertex_shader_code,
                                   const char *const geometry_shader_code,
                                   const char *const fragment_shader_code) {
    if (compiled_) {
        return true;
    }

    if (vertex_shader_code != NULL) {
        vertex_shader_ = glCreateShader(GL_VERTEX_SHADER);
        const GLchar *vertex_shader_code_buffer = vertex_shader_code;
        glShaderSource(vertex_shader_, 1, &vertex_shader_code_buffer, NULL);
        glCompileShader(vertex_shader_);
        if (!ValidateShader(vertex_shader_)) {
            return false;
        }
    }

    if (geometry_shader_code != NULL) {
        geometry_shader_ = glCreateShader(GL_GEOMETRY_SHADER);
        const GLchar *geometry_shader_code_buffer = geometry_shader_code;
        glShaderSource(geometry_shader_, 1, &geometry_shader_code_buffer, NULL);
        glCompileShader(geometry_shader_);
        if (!ValidateShader(geometry_shader_)) {
            return false;
        }
    }

    if (fragment_shader_code != NULL) {
        fragment_shader_ = glCreateShader(GL_FRAGMENT_SHADER);
        const GLchar *fragment_shader_code_buffer = fragment_shader_code;
        glShaderSource(fragment_shader_, 1, &fragment_shader_code_buffer, NULL);
        glCompileShader(fragment_shader_);
        if (!ValidateShader(fragment_shader_)) {
            return false;
        }
    }

    program_ = glCreateProgram();
    if (vertex_shader_code != NULL) {
        glAttachShader(program_, vertex_shader_);
    }
    if (geometry_shader_code != NULL) {
        glAttachShader(program_, geometry_shader_);
    }
    if (fragment_shader_code != NULL) {
        glAttachShader(program_, fragment_shader_);
    }
    glLinkProgram(program_);
    if (!ValidateProgram(program_)) {
        return false;
    }

    // Mark shader objects as deletable.
    // They will be released as soon as program is deleted.
    if (vertex_shader_code != NULL) {
        glDeleteShader(vertex_shader_);
    }
    if (geometry_shader_code != NULL) {
        glDeleteShader(geometry_shader_);
    }
    if (fragment_shader_code != NULL) {
        glDeleteShader(fragment_shader_);
    }

    compiled_ = true;
    return true;
}

void ShaderWrapper::ReleaseProgram() {
    if (compiled_) {
        glDeleteProgram(program_);
        compiled_ = false;
    }
}

bool ShaderWrapper::ValidateShader(GLuint shader_index) {
    GLint result = GL_FALSE;
    int info_log_length;
    glGetShaderiv(shader_index, GL_COMPILE_STATUS, &result);
    if (result == GL_FALSE) {
        glGetShaderiv(shader_index, GL_INFO_LOG_LENGTH, &info_log_length);
        if (info_log_length > 0) {
            std::vector<char> error_message(info_log_length + 1);
            glGetShaderInfoLog(shader_index, info_log_length, NULL,
                               &error_message[0]);
            utility::LogWarning("Shader error: {}", &error_message[0]);
        }
        return false;
    }
    return true;
}

bool ShaderWrapper::ValidateProgram(GLuint program_index) {
    GLint result = GL_FALSE;
    int info_log_length;
    glGetProgramiv(program_index, GL_LINK_STATUS, &result);
    if (result == GL_FALSE) {
        glGetProgramiv(program_index, GL_INFO_LOG_LENGTH, &info_log_length);
        if (info_log_length > 0) {
            std::vector<char> error_message(info_log_length + 1);
            glGetShaderInfoLog(program_index, info_log_length, NULL,
                               &error_message[0]);
            utility::LogWarning("Shader error: {}", &error_message[0]);
        }
        return false;
    }
    return true;
}

}  // namespace glsl

}  // namespace visualization
}  // namespace open3d
