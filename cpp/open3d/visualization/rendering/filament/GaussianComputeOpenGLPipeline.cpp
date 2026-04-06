// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/filament/GaussianComputeOpenGLPipeline.h"

#if !defined(__APPLE__)

// GLEW provides GL 4.x function pointers on all platforms (Windows via WGL,
// Linux via GLX/EGL).  Must be included before any other GL header.
#include <GL/glew.h>

#include <algorithm>
#include <cstring>

#include "open3d/utility/Logging.h"

namespace open3d {
namespace visualization {
namespace rendering {

namespace {

/// Apply a KHR_debug object label to any GL object (GL_BUFFER, GL_TEXTURE,
/// GL_PROGRAM, etc.).  No-op when id == 0, label is empty, or KHR_debug is
/// absent.
void LabelGLObject(GLenum type, GLuint id, const char* label) {
    if (id == 0 || label == nullptr || label[0] == '\0' || !GLEW_KHR_debug) {
        return;
    }

    // If the object already has a non-empty label, don't overwrite it.
    // Query current label with glGetObjectLabel (KHR_debug) and skip when
    // a label exists to avoid stomping existing debug annotations.
    GLsizei existing_length = 0;
    glGetObjectLabel(type, id, 0, &existing_length, nullptr);
    if (existing_length != 0) {
        return;
    }

    glObjectLabel(type, id, -1, label);
}

/// Drain all pending GL errors and return the first one (0 = none).
GLenum DrainErrors(const char* context) {
    GLenum first = GL_NO_ERROR;
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        if (first == GL_NO_ERROR) {
            first = err;
        }
        utility::LogDebug("GL error drained before {}: 0x{:04X}", context,
                          static_cast<unsigned>(err));
    }
    return first;
}

/// Check for GL errors and log them. Returns true if an error occurred.
bool CheckGLError(const char* operation) {
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        utility::LogWarning("GL error in {}: 0x{:04X}", operation,
                            static_cast<unsigned>(err));
        // Drain any additional errors so subsequent checks are clean.
        GLenum extra;
        while ((extra = glGetError()) != GL_NO_ERROR) {
            utility::LogWarning("  (additional GL error): 0x{:04X}",
                                static_cast<unsigned>(extra));
        }
        return true;
    }
    return false;
}

}  // namespace

// --- Shader compilation ---

GLComputeProgram LoadGLComputeProgramSPIRV(
        const std::vector<std::uint8_t>& spirv,
        const std::string& debug_name,
        const std::string& entry_point) {
    GLComputeProgram result;

    if (spirv.empty()) {
        utility::LogWarning("Empty SPIR-V binary for {}", debug_name);
        return result;
    }

    // Log binary size and magic bytes to confirm the right file is loaded.
    // SPIR-V magic number: 0x07230203 (little-endian: 03 02 23 07).
    bool magic_ok = spirv.size() >= 4 && spirv[0] == 0x03 && spirv[1] == 0x02 &&
                    spirv[2] == 0x23 && spirv[3] == 0x07;
    utility::LogInfo("LoadGLComputeProgramSPIRV: {} — {} bytes, magic {}",
                     debug_name, spirv.size(), magic_ok ? "OK" : "INVALID");
    if (!magic_ok) {
        utility::LogWarning(
                "SPIR-V magic mismatch for {} (first 4 bytes: {:02X} {:02X} "
                "{:02X} {:02X}) — file may be stale or not recompiled.",
                debug_name, spirv.size() > 0 ? spirv[0] : 0,
                spirv.size() > 1 ? spirv[1] : 0,
                spirv.size() > 2 ? spirv[2] : 0,
                spirv.size() > 3 ? spirv[3] : 0);
        return result;
    }

    // Check that the driver exposes ARB_gl_spirv and that GLEW loaded the
    // glSpecializeShader entry point.  Both are required; a missing entry
    // point silently returns 0 and causes a bogus GL_INVALID_OPERATION.
    if (!GLEW_ARB_gl_spirv) {
        utility::LogWarning(
                "GL_ARB_gl_spirv not supported by this driver — cannot load "
                "SPIR-V shader {}. "
                "OpenGL 4.6 or the ARB_gl_spirv extension is required.",
                debug_name);
        return result;
    }
    if (!glSpecializeShader) {
        utility::LogWarning(
                "glSpecializeShader entry point not loaded (GLEW did not "
                "resolve it) for {}.",
                debug_name);
        return result;
    }

    // Drain any pre-existing errors so subsequent CheckGLError calls are
    // unambiguous.
    DrainErrors("LoadGLComputeProgramSPIRV/pre");

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
                            log.empty() ? "(empty log)" : log);
        glDeleteShader(shader);
        return result;
    }
    utility::LogDebug("SPIR-V specialization succeeded for {}", debug_name);

    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);

    // Drain GL errors generated during linking before reading link status,
    // so the link-status query itself is not tainted.
    GLenum link_gl_err = DrainErrors("glLinkProgram");

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
        utility::LogWarning(
                "SPIR-V program link error ({}):{}{} GL error during link: "
                "0x{:04X}",
                debug_name, log.empty() ? " (empty driver log)" : " ",
                log.empty() ? "" : log, static_cast<unsigned>(link_gl_err));
        glDeleteProgram(program);
        return result;
    }

    utility::LogDebug("SPIR-V program link succeeded for {}", debug_name);
    result.id = program;
    result.valid = true;
    LabelGLObject(GL_PROGRAM, program, debug_name.c_str());
    return result;
}

GLComputeProgram LoadGLComputeProgramGLSL(const std::string& source,
                                          const std::string& debug_name) {
    GLComputeProgram result;

    if (source.empty()) {
        utility::LogWarning("Empty GLSL source for {}", debug_name);
        return result;
    }

    DrainErrors("LoadGLComputeProgramGLSL/pre");

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
        utility::LogWarning("GLSL compilation error ({}): {}", debug_name,
                            log.empty() ? "(empty log)" : log);
        glDeleteShader(shader);
        return result;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);

    GLenum link_gl_err = DrainErrors("glLinkProgram");
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
        utility::LogWarning(
                "GLSL program link error ({}):{}{} GL error during link: "
                "0x{:04X}",
                debug_name, log.empty() ? " (empty driver log)" : " ",
                log.empty() ? "" : log, static_cast<unsigned>(link_gl_err));
        glDeleteProgram(program);
        return result;
    }

    utility::LogDebug("GLSL program link succeeded for {}", debug_name);
    result.id = program;
    result.valid = true;
    LabelGLObject(GL_PROGRAM, program, debug_name.c_str());
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

GLBufferHandle CreateGLBuffer(std::size_t size,
                              const void* initial_data,
                              const char* label) {
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
    LabelGLObject(GL_BUFFER, buf, label);
    return result;
}

GLBufferHandle CreateGLPrivateBuffer(std::size_t size, const char* label) {
    GLBufferHandle result;
    if (size == 0) {
        return result;
    }
    GLuint buf = 0;
    glGenBuffers(1, &buf);
    if (buf == 0 || CheckGLError("CreateGLPrivateBuffer/gen")) {
        return result;
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
    // GL_DYNAMIC_COPY: modified repeatedly by reading from GL (GPU compute
    // writes it), used for subsequent compute or drawing (GPU reads it).
    // On discrete NVIDIA/AMD hardware this biases allocation toward GPU-local
    // VRAM, avoiding unnecessary CPU-visible coherency overhead.
    glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(size),
                 nullptr, GL_DYNAMIC_COPY);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    if (CheckGLError("CreateGLPrivateBuffer/data")) {
        glDeleteBuffers(1, &buf);
        return result;
    }
    result.id = buf;
    result.size = size;
    result.valid = true;
    LabelGLObject(GL_BUFFER, buf, label);
    return result;
}

GLBufferHandle ResizeGLBuffer(GLBufferHandle buffer,
                              std::size_t new_size,
                              const char* label) {
    if (buffer.valid && buffer.size >= new_size) {
        LabelGLObject(GL_BUFFER, buffer.id, label);
        return buffer;
    }
    DestroyGLBuffer(buffer);
    return CreateGLBuffer(new_size, nullptr, label);
}

GLBufferHandle ResizeGLPrivateBuffer(GLBufferHandle buffer,
                                     std::size_t new_size,
                                     const char* label) {
    if (buffer.valid && buffer.size >= new_size) {
        LabelGLObject(GL_BUFFER, buffer.id, label);
        return buffer;
    }
    DestroyGLBuffer(buffer);
    return CreateGLPrivateBuffer(new_size, label);
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

bool DownloadGLBuffer(const GLBufferHandle& buffer,
                      void* dst,
                      std::size_t size,
                      std::size_t offset) {
    if (!buffer.valid || dst == nullptr || size == 0) {
        return false;
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer.id);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, static_cast<GLintptr>(offset),
                       static_cast<GLsizeiptr>(size), dst);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    return !CheckGLError("DownloadGLBuffer");
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
                                  std::uint32_t format,
                                  const char* label) {
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
    LabelGLObject(GL_TEXTURE, tex, label);
    return result;
}

GLTextureHandle ResizeGLTexture2D(GLTextureHandle texture,
                                  std::uint32_t width,
                                  std::uint32_t height,
                                  std::uint32_t format,
                                  const char* label) {
    if (texture.valid && texture.width == width && texture.height == height) {
        // Re-apply label in case the handle was reused with a different name.
        LabelGLObject(GL_TEXTURE, texture.id, label);
        return texture;
    }
    DestroyGLTexture(texture);
    return CreateGLTexture2D(width, height, format, label);
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

void DispatchComputeIndirect(const GLBufferHandle& buffer,
                             std::size_t byte_offset) {
    if (!buffer.valid) return;
    // GL_DISPATCH_INDIRECT_BUFFER = 0x90EE
    glBindBuffer(0x90EE, buffer.id);
    glDispatchComputeIndirect(static_cast<GLintptr>(byte_offset));
    glBindBuffer(0x90EE, 0);
}

void BindUBORange(std::uint32_t binding,
                  const GLBufferHandle& buffer,
                  std::size_t offset,
                  std::size_t size) {
    if (!buffer.valid) return;
    glBindBufferRange(GL_UNIFORM_BUFFER, binding, buffer.id,
                      static_cast<GLintptr>(offset),
                      static_cast<GLsizeiptr>(size));
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
