// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/filament/GaussianComputeOpenGLPipeline.h"

#if !defined(__APPLE__)

#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>

#include <algorithm>
#include <cstring>

#include "open3d/utility/Logging.h"

namespace open3d {
namespace visualization {
namespace rendering {

namespace {

/// Check for GL errors and log them. Returns true if an error occurred.
bool CheckGLError(const char* operation) {
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        utility::LogWarning("GL error in {}: 0x{:04X}", operation, err);
        return true;
    }
    return false;
}

}  // namespace

// --- Shader compilation ---

GLComputeProgram CompileGLComputeProgram(const std::string& source,
                                         const std::string& debug_name) {
    GLComputeProgram result;

    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    if (shader == 0) {
        utility::LogWarning("Failed to create compute shader for {}",
                            debug_name);
        return result;
    }

    const char* src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint compiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        GLint log_len = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_len);
        std::string log(std::max(0, log_len - 1), '\0');
        if (log_len > 0) {
            glGetShaderInfoLog(shader, log_len, nullptr, log.data());
        }
        utility::LogWarning("Compute shader compile error ({}): {}", debug_name,
                            log);
        glDeleteShader(shader);
        return result;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);
    glDeleteShader(shader);  // Detach after link.

    GLint linked = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
        GLint log_len = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_len);
        std::string log(std::max(0, log_len - 1), '\0');
        if (log_len > 0) {
            glGetProgramInfoLog(program, log_len, nullptr, log.data());
        }
        utility::LogWarning("Compute program link error ({}): {}", debug_name,
                            log);
        glDeleteProgram(program);
        return result;
    }

    result.id = program;
    result.valid = true;
    return result;
}

GLComputeProgram LoadGLComputeProgramSPIRV(
        const std::vector<std::uint8_t>& spirv,
        const std::string& debug_name,
        const std::string& entry_point) {
    GLComputeProgram result;

    if (spirv.empty()) {
        utility::LogWarning("Empty SPIR-V binary for {}", debug_name);
        return result;
    }

    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    if (shader == 0) {
        utility::LogWarning("Failed to create compute shader for {}",
                            debug_name);
        return result;
    }

    // GL_SHADER_BINARY_FORMAT_SPIR_V = 0x9551 (GL 4.6 / ARB_gl_spirv).
    glShaderBinary(1, &shader, 0x9551, spirv.data(),
                   static_cast<GLsizei>(spirv.size()));
    if (CheckGLError("glShaderBinary")) {
        utility::LogWarning("glShaderBinary failed for {}", debug_name);
        glDeleteShader(shader);
        return result;
    }

    glSpecializeShader(shader, entry_point.c_str(), 0, nullptr, nullptr);
    if (CheckGLError("glSpecializeShader")) {
        utility::LogWarning("glSpecializeShader failed for {}", debug_name);
        glDeleteShader(shader);
        return result;
    }

    GLint compiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        GLint log_len = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_len);
        std::string log(std::max(0, log_len - 1), '\0');
        if (log_len > 0) {
            glGetShaderInfoLog(shader, log_len, nullptr, log.data());
        }
        utility::LogWarning("SPIR-V specialization error ({}): {}", debug_name,
                            log);
        glDeleteShader(shader);
        return result;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);
    glDeleteShader(shader);

    GLint linked = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
        GLint log_len = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_len);
        std::string log(std::max(0, log_len - 1), '\0');
        if (log_len > 0) {
            glGetProgramInfoLog(program, log_len, nullptr, log.data());
        }
        utility::LogWarning("SPIR-V program link error ({}): {}", debug_name,
                            log);
        glDeleteProgram(program);
        return result;
    }

    result.id = program;
    result.valid = true;
    return result;
}

void DestroyGLComputeProgram(GLComputeProgram& program) {
    if (program.valid && program.id != 0) {
        glDeleteProgram(program.id);
    }
    program.id = 0;
    program.valid = false;
}

// --- Buffer management ---

GLBufferHandle CreateGLBuffer(std::size_t size, const void* initial_data) {
    GLBufferHandle result;
    GLuint buf = 0;
    glGenBuffers(1, &buf);
    if (buf == 0 || CheckGLError("CreateGLBuffer/gen")) {
        return result;
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
    // Use GL_DYNAMIC_DRAW: the buffer is updated from CPU and read by GPU
    // compute shaders. Some buffers are updated every frame (view params),
    // others only when the scene changes.
    glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(size),
                 initial_data, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    if (CheckGLError("CreateGLBuffer/data")) {
        glDeleteBuffers(1, &buf);
        return result;
    }

    result.id = buf;
    result.size = size;
    result.valid = true;
    return result;
}

GLBufferHandle ResizeGLBuffer(GLBufferHandle buffer, std::size_t new_size) {
    if (buffer.valid && buffer.size >= new_size) {
        return buffer;
    }
    DestroyGLBuffer(buffer);
    return CreateGLBuffer(new_size, nullptr);
}

void UploadGLBuffer(const GLBufferHandle& buffer,
                    const void* data,
                    std::size_t size,
                    std::size_t offset) {
    if (!buffer.valid || size == 0) return;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer.id);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, static_cast<GLintptr>(offset),
                    static_cast<GLsizeiptr>(size), data);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void DownloadGLBuffer(const GLBufferHandle& buffer,
                      void* data,
                      std::size_t size,
                      std::size_t offset) {
    if (!buffer.valid || size == 0) return;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer.id);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, static_cast<GLintptr>(offset),
                       static_cast<GLsizeiptr>(size), data);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void ClearGLBuffer(const GLBufferHandle& buffer) {
    if (!buffer.valid) return;
    GLuint zero = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer.id);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER,
                      GL_UNSIGNED_INT, &zero);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void DestroyGLBuffer(GLBufferHandle& buffer) {
    if (buffer.valid && buffer.id != 0) {
        glDeleteBuffers(1, &buffer.id);
    }
    buffer.id = 0;
    buffer.size = 0;
    buffer.valid = false;
}

// --- Texture management ---

GLTextureHandle CreateGLTexture2D(std::uint32_t width,
                                  std::uint32_t height,
                                  std::uint32_t format) {
    GLTextureHandle result;
    GLuint tex = 0;
    glGenTextures(1, &tex);
    if (tex == 0 || CheckGLError("CreateGLTexture2D/gen")) {
        return result;
    }

    glBindTexture(GL_TEXTURE_2D, tex);
    // Immutable storage: 1 level, no mipmaps.
    glTexStorage2D(GL_TEXTURE_2D, 1, format, static_cast<GLsizei>(width),
                   static_cast<GLsizei>(height));
    // Linear filtering for Filament sampling.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    if (CheckGLError("CreateGLTexture2D/storage")) {
        glDeleteTextures(1, &tex);
        return result;
    }

    result.id = tex;
    result.width = width;
    result.height = height;
    result.valid = true;
    return result;
}

GLTextureHandle ResizeGLTexture2D(GLTextureHandle texture,
                                  std::uint32_t width,
                                  std::uint32_t height,
                                  std::uint32_t format) {
    if (texture.valid && texture.width == width && texture.height == height) {
        return texture;
    }
    DestroyGLTexture(texture);
    return CreateGLTexture2D(width, height, format);
}

void DestroyGLTexture(GLTextureHandle& texture) {
    if (texture.valid && texture.id != 0) {
        glDeleteTextures(1, &texture.id);
    }
    texture.id = 0;
    texture.width = 0;
    texture.height = 0;
    texture.valid = false;
}

void DownloadGLTexture2D(const GLTextureHandle& texture,
                         void* data,
                         std::uint32_t format) {
    if (!texture.valid || !data) return;
    glBindTexture(GL_TEXTURE_2D, texture.id);
    glGetTexImage(GL_TEXTURE_2D, 0, format, GL_FLOAT, data);
    glBindTexture(GL_TEXTURE_2D, 0);
}

// --- Compute dispatch ---

void BindSSBO(std::uint32_t binding, const GLBufferHandle& buffer) {
    if (!buffer.valid) return;
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, buffer.id);
}

void BindUBO(std::uint32_t binding, const GLBufferHandle& buffer) {
    if (!buffer.valid) return;
    glBindBufferBase(GL_UNIFORM_BUFFER, binding, buffer.id);
}

void BindImage(std::uint32_t binding,
               const GLTextureHandle& texture,
               std::uint32_t format,
               std::uint32_t access) {
    if (!texture.valid) return;
    glBindImageTexture(binding, texture.id, 0, GL_FALSE, 0, access, format);
}

void BindSamplerTexture(std::uint32_t unit, const GLTextureHandle& texture) {
    if (!texture.valid) return;
    // GL 4.5 direct state access: binds texture to texture unit.
    glBindTextureUnit(unit, texture.id);
}

void UseProgram(const GLComputeProgram& program) {
    if (program.valid) {
        glUseProgram(program.id);
    }
}

void DispatchCompute(std::uint32_t num_groups_x,
                     std::uint32_t num_groups_y,
                     std::uint32_t num_groups_z) {
    glDispatchCompute(num_groups_x, num_groups_y, num_groups_z);
}

void GLComputeBarrier(std::uint32_t bits) {
    glMemoryBarrier(static_cast<GLbitfield>(bits));
}

void GLComputeFullBarrier() { glMemoryBarrier(GL_ALL_BARRIER_BITS); }

std::uint32_t DrainGLErrors(const char* context) {
    GLenum first_err = GL_NO_ERROR;
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        if (first_err == GL_NO_ERROR) first_err = err;
        utility::LogWarning("GL error in {}: 0x{:04X}", context, err);
    }
    return static_cast<std::uint32_t>(first_err);
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // !defined(__APPLE__)
