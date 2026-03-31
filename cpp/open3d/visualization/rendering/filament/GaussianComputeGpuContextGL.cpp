// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#if !defined(__APPLE__)

#include <memory>
#include <unordered_map>
#include <vector>

// GLEW for GL extension flags (GLEW_ARB_gl_spirv etc.) and glewGetString.
// Must be included before any other GL header.
#include <GL/glew.h>

#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/GaussianComputeGpuContext.h"
#include "open3d/visualization/rendering/filament/GaussianComputeOpenGLContext.h"
#include "open3d/visualization/rendering/filament/GaussianComputeOpenGLPipeline.h"

namespace open3d {
namespace visualization {
namespace rendering {

namespace {

bool ReadSPIRVFile(const std::string& path,
                   std::vector<std::uint8_t>* contents) {
    std::vector<char> bytes;
    std::string error;
    if (!utility::filesystem::FReadToBuffer(path, bytes, &error)) {
        utility::LogWarning("Failed to read SPIR-V shader {}: {}", path, error);
        return false;
    }
    contents->assign(bytes.begin(), bytes.end());
    return true;
}

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

/// OpenGL 4.6 + SPIR-V implementation of GaussianComputeGpuContext.
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
        utility::LogInfo(
                "GaussianComputeGpuContext: GL vendor:   {}",
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

        static const char* kShaderSPVFiles[] = {
                "gaussian_project.spv",
                "gaussian_prefix_sum.spv",
                "gaussian_scatter.spv",
                "gaussian_composite.spv",
                "gaussian_radix_sort_keygen.spv",
                "gaussian_radix_sort_histograms.spv",
                "gaussian_radix_sort.spv",
                "gaussian_radix_sort_payload.spv",
                "gaussian_compute_dispatch_args.spv",
        };
        static_assert(
                sizeof(kShaderSPVFiles) / sizeof(kShaderSPVFiles[0]) ==
                        static_cast<std::size_t>(
                                GaussianComputeProgramId::kCount),
                "Shader file list must match GaussianComputeProgramId::kCount");

        const std::string shader_root =
                EngineInstance::GetResourcePath() + "/gaussian_compute/";
        for (int i = 0; i < static_cast<int>(GaussianComputeProgramId::kCount);
             ++i) {
            std::vector<std::uint8_t> spirv;
            std::string path = shader_root + kShaderSPVFiles[i];
            if (!ReadSPIRVFile(path, &spirv)) {
                utility::LogWarning(
                        "Gaussian compute OpenGL: Failed to read SPIR-V shader "
                        "{}",
                        path);
                return false;
            }
            programs_[i] = LoadGLComputeProgramSPIRV(spirv, kShaderSPVFiles[i]);
            if (!programs_[i].valid) {
                return false;
            }
        }

        programs_valid_ = true;
        return true;
    }

    std::uintptr_t CreateBuffer(std::size_t size) override {
        GLBufferHandle h = CreateGLBuffer(size, nullptr);
        if (!h.valid) {
            return 0;
        }
        std::uintptr_t k = static_cast<std::uintptr_t>(h.id);
        buffer_sizes_[k] = h.size;
        return k;
    }

    /// GPU-private buffer: uses GL_DYNAMIC_COPY to tell the driver the buffer
    /// is written by GPU operations and read back by the GPU.  On discrete
    /// GPUs this biases placement toward GPU-local VRAM.
    std::uintptr_t CreatePrivateBuffer(std::size_t size) override {
        GLBufferHandle h = CreateGLPrivateBuffer(size);
        if (!h.valid) {
            return 0;
        }
        std::uintptr_t k = static_cast<std::uintptr_t>(h.id);
        buffer_sizes_[k] = h.size;
        return k;
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
        GLBufferHandle h = ToGLBuffer(buf, buffer_sizes_);
        GLBufferHandle nh = ResizeGLBuffer(h, new_size);
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
        GLBufferHandle h = ToGLBuffer(buf, buffer_sizes_);
        GLBufferHandle nh = ResizeGLPrivateBuffer(h, new_size);
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
        GLBufferHandle h = ToGLBuffer(buf, buffer_sizes_);
        UploadGLBuffer(h, data, size, offset);
    }

    void ClearBufferUInt32Zero(std::uintptr_t buf) override {
        GLBufferHandle h = ToGLBuffer(buf, buffer_sizes_);
        ClearGLBuffer(h);
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

    void UseProgram(GaussianComputeProgramId id) override {
        int i = static_cast<int>(id);
        if (i < 0 || i >= static_cast<int>(GaussianComputeProgramId::kCount)) {
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
        GLTextureHandle h = ToGLTexture(tex, texture_sizes_);
        GLTextureHandle nh = ResizeGLTexture2D(h, width, height, kGL_R32F);
        texture_sizes_.erase(tex);
        if (!nh.valid) {
            return 0;
        }
        std::uintptr_t nk = static_cast<std::uintptr_t>(nh.id);
        texture_sizes_[nk] = {width, height};
        return nk;
    }

    void BindImageRGBA16FWrite(std::uint32_t binding,
                               std::uintptr_t tex,
                               std::uint32_t width,
                               std::uint32_t height) override {
        GLTextureHandle h = ToGLTexture(tex, texture_sizes_);
        if (width > 0 && height > 0) {
            h.width = width;
            h.height = height;
            h.valid = true;
        }
        BindImage(binding, h, kGL_RGBA16F, kGL_WRITE_ONLY);
    }

    void BindImageR32FWrite(std::uint32_t binding,
                            std::uintptr_t tex,
                            std::uint32_t width,
                            std::uint32_t height) override {
        GLTextureHandle h = ToGLTexture(tex, texture_sizes_);
        if (width > 0 && height > 0) {
            h.width = width;
            h.height = height;
            h.valid = true;
        }
        BindImage(binding, h, kGL_R32F, kGL_WRITE_ONLY);
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
        auto& gl_ctx = GaussianComputeOpenGLContext::GetInstance();
        gl_ctx.Finish();
    }

    void BeginGeometryPass() override {}
    void EndGeometryPass() override {}
    void BeginCompositePass() override {}
    void EndCompositePass() override {}

private:
    void CleanupPrograms() {
        for (auto& p : programs_) {
            DestroyGLComputeProgram(p);
        }
        programs_loaded_ = false;
        programs_valid_ = false;
    }

    GLComputeProgram
            programs_[static_cast<int>(GaussianComputeProgramId::kCount)] = {};
    bool programs_loaded_ = false;
    bool programs_valid_ = false;
    std::unordered_map<std::uintptr_t, std::size_t> buffer_sizes_;
    std::unordered_map<std::uintptr_t, std::pair<std::uint32_t, std::uint32_t>>
            texture_sizes_;
};

std::unique_ptr<GaussianComputeGpuContext>
CreateGaussianComputeGpuContextOpenGL() {
    return std::make_unique<GaussianComputeGpuContextOpenGL>();
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // !defined(__APPLE__)
