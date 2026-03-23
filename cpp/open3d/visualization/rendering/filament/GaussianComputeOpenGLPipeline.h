// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// OpenGL compute pipeline helpers for Gaussian splatting.
// Provides thin wrappers around GL 4.5 compute APIs: shader compilation,
// SSBO/UBO management, image textures, dispatch, and synchronization.

#pragma once

#if !defined(__APPLE__)

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace open3d {
namespace visualization {
namespace rendering {

// GL constants used by the pipeline API, defined here so callers don't
// need to include GL headers directly.
// clang-format off
constexpr std::uint32_t kGL_RGBA16F             = 0x881A;
constexpr std::uint32_t kGL_R32F                = 0x822E;
constexpr std::uint32_t kGL_RGBA8               = 0x8058;
constexpr std::uint32_t kGL_RGBA                = 0x1908;
constexpr std::uint32_t kGL_RED                 = 0x1903;
constexpr std::uint32_t kGL_DEPTH_COMPONENT32F  = 0x8CAC;
constexpr std::uint32_t kGL_DEPTH_COMPONENT     = 0x1902;
constexpr std::uint32_t kGL_TEXTURE_2D          = 0x0DE1;
constexpr std::uint32_t kGL_NEAREST             = 0x2600;
constexpr std::uint32_t kGL_CLAMP_TO_EDGE       = 0x812F;
constexpr std::uint32_t kGL_READ_ONLY           = 0x88B8;
constexpr std::uint32_t kGL_WRITE_ONLY          = 0x88B9;
constexpr std::uint32_t kGL_READ_WRITE          = 0x88BA;
// clang-format on

// --- Handle types (wrapping GLuint) ---

struct GLComputeProgram {
    std::uint32_t id = 0;
    bool valid = false;
};

struct GLBufferHandle {
    std::uint32_t id = 0;
    std::size_t size = 0;
    bool valid = false;
};

struct GLTextureHandle {
    std::uint32_t id = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    bool valid = false;
};

// --- Shader compilation ---

/// Compile a compute shader from GLSL source and link into a program.
/// Returns an invalid handle on failure (logs warnings).
GLComputeProgram CompileGLComputeProgram(const std::string& source,
                                         const std::string& debug_name);

/// Load a compute shader from a SPIR-V binary and link into a program.
/// Uses glShaderBinary(GL_SHADER_BINARY_FORMAT_SPIR_V) + glSpecializeShader.
/// Returns an invalid handle on failure (logs warnings).
GLComputeProgram LoadGLComputeProgramSPIRV(
        const std::vector<std::uint8_t>& spirv,
        const std::string& debug_name,
        const std::string& entry_point = "main");

/// Destroy a compute program.
void DestroyGLComputeProgram(GLComputeProgram& program);

// --- Buffer management ---

/// Create a GL buffer (for SSBO or UBO use).
/// If initial_data is non-null, the buffer is initialized with it.
GLBufferHandle CreateGLBuffer(std::size_t size, const void* initial_data);

/// Resize a GL buffer. If the buffer is already large enough, does nothing.
/// Otherwise creates a new buffer (old data is NOT preserved).
/// Returns the (possibly new) buffer handle.
GLBufferHandle ResizeGLBuffer(GLBufferHandle buffer, std::size_t new_size);

/// Upload data to a region of a buffer.
void UploadGLBuffer(const GLBufferHandle& buffer,
                    const void* data,
                    std::size_t size,
                    std::size_t offset);

/// Download data from a region of a buffer to CPU memory.
void DownloadGLBuffer(const GLBufferHandle& buffer,
                      void* data,
                      std::size_t size,
                      std::size_t offset);

/// Fill a buffer with zeros.
void ClearGLBuffer(const GLBufferHandle& buffer);

/// Destroy a buffer.
void DestroyGLBuffer(GLBufferHandle& buffer);

// --- Texture management ---

/// Create a 2D texture with immutable storage.
/// format: GL_RGBA16F, GL_R32F, etc.
GLTextureHandle CreateGLTexture2D(std::uint32_t width,
                                  std::uint32_t height,
                                  std::uint32_t format);

/// Resize a texture. If dimensions match, does nothing.
/// Otherwise destroys and recreates (old data is NOT preserved).
GLTextureHandle ResizeGLTexture2D(GLTextureHandle texture,
                                  std::uint32_t width,
                                  std::uint32_t height,
                                  std::uint32_t format);

/// Destroy a texture.
void DestroyGLTexture(GLTextureHandle& texture);

/// Download texture pixels to CPU memory (RGBA float32).
/// The output buffer must have room for width * height * 4 floats.
void DownloadGLTexture2D(const GLTextureHandle& texture,
                        void* data,
                        std::uint32_t format);

// --- Compute dispatch ---

/// Bind a buffer to a shader storage buffer binding point.
void BindSSBO(std::uint32_t binding, const GLBufferHandle& buffer);

/// Bind a buffer to a uniform buffer binding point.
void BindUBO(std::uint32_t binding, const GLBufferHandle& buffer);

/// Bind a texture as an image unit for compute shader access.
/// access: GL_READ_ONLY, GL_WRITE_ONLY, or GL_READ_WRITE.
void BindImage(std::uint32_t binding,
               const GLTextureHandle& texture,
               std::uint32_t format,
               std::uint32_t access);

/// Bind a GL texture to a texture unit for sampler access in a compute shader.
/// For SPIR-V shaders, `layout(binding = N)` maps to texture unit N.
void BindSamplerTexture(std::uint32_t unit, const GLTextureHandle& texture);

/// Use a compute program.
void UseProgram(const GLComputeProgram& program);

/// Dispatch compute work groups.
void DispatchCompute(std::uint32_t num_groups_x,
                     std::uint32_t num_groups_y,
                     std::uint32_t num_groups_z);

/// Insert a memory barrier. bits should be a combination of
/// GL_SHADER_STORAGE_BARRIER_BIT, GL_SHADER_IMAGE_ACCESS_BARRIER_BIT, etc.
void GLComputeBarrier(std::uint32_t bits);

/// Full barrier for all buffer and texture writes.
void GLComputeFullBarrier();

/// Drain and return all pending GL errors (0 = GL_NO_ERROR = success).
/// Returns the first error code encountered, or 0 if no errors.
std::uint32_t DrainGLErrors(const char* context);

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // !defined(__APPLE__)
