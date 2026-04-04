// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// OpenGL 4.6 + SPIR-V implementation of GaussianComputeGpuContext.
// Compiled only on non-Apple platforms.

#if !defined(__APPLE__)

#include <memory>
#include <unordered_map>
#include <vector>

// GLEW for GL extension flags (GLEW_ARB_gl_spirv etc.) and glewGetString.
// Must be included before any other GL header.
#include <GL/glew.h>

#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"
#include "open3d/visualization/rendering/filament/ComputeGPU.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/GaussianComputeOpenGLContext.h"
#include "open3d/visualization/rendering/filament/GaussianComputeOpenGLPipeline.h"

namespace open3d {
namespace visualization {
namespace rendering {

namespace {
GLBufferHandle ToGLBuffer(
        std::uintptr_t id,
        const std::unordered_map<std::uintptr_t, std::size_t>& sizes) {
    if (id == 0) {
        return {};
    }
    auto it = sizes.find(id);
    std::size_t sz = it != sizes.end() ? it->second : 0;
    GLBufferHandle h;
    h.id = static_cast<std::uint32_t>(id);
    h.size = sz;
    h.valid = true;
    return h;
}

GLTextureHandle ToGLTexture(
        std::uintptr_t id,
        const std::unordered_map<std::uintptr_t,
                                 std::pair<std::uint32_t, std::uint32_t>>&
                tex_sizes) {
    if (id == 0) {
        return {};
    }
    auto it = tex_sizes.find(id);
    std::uint32_t w = 0;
    std::uint32_t h = 0;
    if (it != tex_sizes.end()) {
        w = it->second.first;
        h = it->second.second;
    }
    GLTextureHandle t;
    t.id = static_cast<std::uint32_t>(id);
    t.width = w;
    t.height = h;
    t.valid = true;
    return t;
}

}  // namespace

class GaussianComputeGpuContextOpenGL final : public GaussianComputeGpuContext {
public:
    GaussianComputeGpuContextOpenGL() = default;

    ~GaussianComputeGpuContextOpenGL() override { CleanupPrograms(); }

    bool EnsureProgramsLoaded() override {
        if (programs_loaded_) {
            return programs_valid_;
        }
        programs_loaded_ = true;
        programs_valid_ = false;

        // Log driver identity so error reports are unambiguous.
        utility::LogInfo("GaussianComputeGpuContext: GL vendor:   {}",
                         reinterpret_cast<const char*>(glGetString(GL_VENDOR)));
        utility::LogInfo(
                "GaussianComputeGpuContext: GL renderer: {}",
                reinterpret_cast<const char*>(glGetString(GL_RENDERER)));
        utility::LogInfo(
                "GaussianComputeGpuContext: GL version:  {}",
                reinterpret_cast<const char*>(glGetString(GL_VERSION)));
        utility::LogInfo(
                "GaussianComputeGpuContext: GLEW version: {}",
                reinterpret_cast<const char*>(glewGetString(GLEW_VERSION)));
        utility::LogInfo(
                "GaussianComputeGpuContext: ARB_gl_spirv supported: {}",
                GLEW_ARB_gl_spirv ? "yes" : "no");
        utility::LogInfo(
                "GaussianComputeGpuContext: ARB_compute_shader supported: {}",
                GLEW_ARB_compute_shader ? "yes" : "no");

        const std::string shader_root =
                EngineInstance::GetResourcePath() + "/gaussian_compute/";
        for (int i = 0; i < static_cast<int>(ComputeProgramId::kCount); ++i) {
            const std::string base = shader_root + kGsShaderNames[i];
            const std::string spv_name =
                    std::string(kGsShaderNames[i]) + ".spv";
            const std::string comp_name =
                    std::string(kGsShaderNames[i]) + ".comp";

            // --- Primary path: SPIR-V binary ---
            std::vector<std::uint8_t> spirv;
            {
                std::vector<char> bytes;
                std::string error;
                if (utility::filesystem::FReadToBuffer(base + ".spv", bytes,
                                                       &error)) {
                    spirv.assign(bytes.begin(), bytes.end());
                    programs_[i] =
                            LoadGLComputeProgramSPIRV(spirv, spv_name.c_str());
                }
            }
            if (programs_[i].valid) {
                utility::LogDebug("GaussianCompute: loaded {} via SPIR-V",
                                  spv_name);
                continue;
            }

            // --- Fallback path: runtime GLSL compilation ---
            // Reached when the .spv file is missing, has an invalid magic,
            // the driver lacks ARB_gl_spirv, or specialization fails.
            utility::LogWarning(
                    "GaussianCompute: SPIR-V path failed for {} — "
                    "trying GLSL source compilation.",
                    spv_name);

            {
                std::vector<char> bytes;
                std::string error;
                if (!utility::filesystem::FReadToBuffer(base + ".comp", bytes,
                                                        &error)) {
                    utility::LogWarning(
                            "GaussianCompute: GLSL fallback source not found: "
                            "{}",
                            base + ".comp");
                    return false;
                }
                std::string glsl_source(bytes.begin(), bytes.end());
                programs_[i] = LoadGLComputeProgramGLSL(glsl_source,
                                                        comp_name.c_str());
            }
            if (!programs_[i].valid) {
                // Compile/link errors already logged by
                // LoadGLComputeProgramGLSL.
                utility::LogWarning(
                        "GaussianCompute: GLSL compilation also failed for {}",
                        comp_name);
                return false;
            }
            utility::LogInfo("GaussianCompute: loaded {} via GLSL fallback.",
                             comp_name);
        }

        programs_valid_ = true;
        return true;
    }

    std::uintptr_t CreateBuffer(std::size_t size) override {
        return AllocGLBuffer(size, /*priv=*/false);
    }

    /// GPU-private buffer: GL_DYNAMIC_COPY biases placement toward GPU-local
    /// VRAM on discrete GPUs.
    std::uintptr_t CreatePrivateBuffer(std::size_t size) override {
        return AllocGLBuffer(size, /*priv=*/true);
    }

    void DestroyBuffer(std::uintptr_t buf) override {
        if (buf == 0) {
            return;
        }
        GLBufferHandle h = ToGLBuffer(buf, buffer_sizes_);
        DestroyGLBuffer(h);
        buffer_sizes_.erase(buf);
    }

    std::uintptr_t ResizeBuffer(std::uintptr_t buf,
                                std::size_t new_size) override {
        if (buf == 0) {
            return CreateBuffer(new_size);
        }
        GLBufferHandle nh =
                ResizeGLBuffer(ToGLBuffer(buf, buffer_sizes_), new_size);
        buffer_sizes_.erase(buf);
        if (!nh.valid) {
            return 0;
        }
        std::uintptr_t nk = static_cast<std::uintptr_t>(nh.id);
        buffer_sizes_[nk] = nh.size;
        return nk;
    }

    std::uintptr_t ResizePrivateBuffer(std::uintptr_t buf,
                                       std::size_t new_size) override {
        if (new_size == 0) {
            DestroyBuffer(buf);
            return 0;
        }
        GLBufferHandle nh =
                ResizeGLPrivateBuffer(ToGLBuffer(buf, buffer_sizes_), new_size);
        buffer_sizes_.erase(buf);
        if (!nh.valid) {
            return 0;
        }
        std::uintptr_t nk = static_cast<std::uintptr_t>(nh.id);
        buffer_sizes_[nk] = nh.size;
        return nk;
    }

    void UploadBuffer(std::uintptr_t buf,
                      const void* data,
                      std::size_t size,
                      std::size_t offset) override {
        UploadGLBuffer(ToGLBuffer(buf, buffer_sizes_), data, size, offset);
    }

    void ClearBufferUInt32Zero(std::uintptr_t buf) override {
        ClearGLBuffer(ToGLBuffer(buf, buffer_sizes_));
    }

    void BindSSBO(std::uint32_t binding, std::uintptr_t buf) override {
        // Qualify to bypass MSVC name-hiding: member hides free function.
        ::open3d::visualization::rendering::BindSSBO(
                binding, ToGLBuffer(buf, buffer_sizes_));
    }

    void BindUBO(std::uint32_t binding, std::uintptr_t buf) override {
        ::open3d::visualization::rendering::BindUBO(
                binding, ToGLBuffer(buf, buffer_sizes_));
    }

    void BindUBORange(std::uint32_t binding,
                      std::uintptr_t buf,
                      std::size_t offset,
                      std::size_t range_size) override {
        ::open3d::visualization::rendering::BindUBORange(
                binding, ToGLBuffer(buf, buffer_sizes_), offset, range_size);
    }

    void UseProgram(ComputeProgramId id) override {
        int i = static_cast<int>(id);
        if (i < 0 || i >= static_cast<int>(ComputeProgramId::kCount)) {
            return;
        }
        // Qualify to bypass MSVC name-hiding: member hides free function.
        ::open3d::visualization::rendering::UseProgram(programs_[i]);
    }

    void Dispatch(std::uint32_t groups_x,
                  std::uint32_t groups_y,
                  std::uint32_t groups_z) override {
        DispatchCompute(groups_x, groups_y, groups_z);
    }

    void DispatchIndirect(std::uintptr_t indirect_buf,
                          std::size_t byte_offset) override {
        DispatchComputeIndirect(ToGLBuffer(indirect_buf, buffer_sizes_),
                                byte_offset);
    }

    void FullBarrier() override { GLComputeFullBarrier(); }

    void PushDebugGroup(const char* label) override {
        if (GLEW_KHR_debug) {
            glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, -1, label);
        }
    }
    void PopDebugGroup() override {
        if (GLEW_KHR_debug) {
            glPopDebugGroup();
        }
    }

    std::uintptr_t CreateTexture2DR32F(std::uint32_t width,
                                       std::uint32_t height) override {
        GLTextureHandle t = CreateGLTexture2D(width, height, kGL_R32F);
        if (!t.valid) {
            return 0;
        }
        std::uintptr_t k = static_cast<std::uintptr_t>(t.id);
        texture_sizes_[k] = {width, height};
        return k;
    }

    void DestroyTexture(std::uintptr_t tex) override {
        if (tex == 0) {
            return;
        }
        GLTextureHandle h = ToGLTexture(tex, texture_sizes_);
        DestroyGLTexture(h);
        texture_sizes_.erase(tex);
    }

    std::uintptr_t ResizeTexture2DR32F(std::uintptr_t tex,
                                       std::uint32_t width,
                                       std::uint32_t height) override {
        if (tex == 0) {
            return CreateTexture2DR32F(width, height);
        }
        GLTextureHandle nh = ResizeGLTexture2D(ToGLTexture(tex, texture_sizes_),
                                               width, height, kGL_R32F);
        texture_sizes_.erase(tex);
        if (!nh.valid) {
            return 0;
        }
        std::uintptr_t nk = static_cast<std::uintptr_t>(nh.id);
        texture_sizes_[nk] = {width, height};
        return nk;
    }

    void BindImage(std::uint32_t binding,
                   std::uintptr_t tex,
                   std::uint32_t width,
                   std::uint32_t height,
                   ImageFormat fmt) override {
        GLTextureHandle h = ToGLTexture(tex, texture_sizes_);
        if (width > 0 && height > 0) {
            h.width = width;
            h.height = height;
            h.valid = true;
        }
        const std::uint32_t gl_fmt =
                (fmt == ImageFormat::kRGBA16F) ? kGL_RGBA16F : kGL_R32F;
        ::open3d::visualization::rendering::BindImage(binding, h, gl_fmt,
                                                      kGL_WRITE_ONLY);
    }

    void BindSamplerTexture(std::uint32_t unit,
                            std::uintptr_t tex,
                            std::uint32_t width,
                            std::uint32_t height) override {
        GLTextureHandle h = ToGLTexture(tex, texture_sizes_);
        if (width > 0 && height > 0) {
            h.width = width;
            h.height = height;
            h.valid = true;
        }
        // Qualify to bypass MSVC name-hiding: member hides free function.
        ::open3d::visualization::rendering::BindSamplerTexture(unit, h);
    }

    void FinishGpuWork() override {
        GaussianComputeOpenGLContext::GetInstance().Finish();
    }

private:
    std::uintptr_t AllocGLBuffer(std::size_t size, bool priv) {
        GLBufferHandle h = priv ? CreateGLPrivateBuffer(size)
                                : CreateGLBuffer(size, nullptr);
        if (!h.valid) {
            return 0;
        }
        std::uintptr_t k = static_cast<std::uintptr_t>(h.id);
        buffer_sizes_[k] = h.size;
        return k;
    }

    void CleanupPrograms() {
        for (auto& p : programs_) {
            DestroyGLComputeProgram(p);
        }
        programs_loaded_ = false;
        programs_valid_ = false;
    }

    GLComputeProgram programs_[static_cast<int>(ComputeProgramId::kCount)] = {};
    bool programs_loaded_ = false;
    bool programs_valid_ = false;
    std::unordered_map<std::uintptr_t, std::size_t> buffer_sizes_;
    std::unordered_map<std::uintptr_t, std::pair<std::uint32_t, std::uint32_t>>
            texture_sizes_;
};

std::unique_ptr<GaussianComputeGpuContext> CreateComputeGpuContextGL() {
    return std::make_unique<GaussianComputeGpuContextOpenGL>();
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // !defined(__APPLE__)
