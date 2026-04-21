// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// OpenGL 4.6 + SPIR-V implementation of GaussianSplatGpuContext.
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
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/gaussian_splat/ComputeGPU.h"
#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatOpenGLContext.h"
#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatOpenGLPipeline.h"

namespace open3d {
namespace visualization {
namespace rendering {

namespace {

// Returns true when the driver reports all subgroup features required by the
// OneSweep shaders:
//   gaussian_onesweep_global_hist_subgroup.comp — arithmetic (subgroupAdd)
//   gaussian_onesweep_digit_pass_subgroup.comp  — ballot (WaveMatch8),
//                                                  shuffle (subgroupShuffle),
//                                                  vote (subgroupAll)
// Also validates subgroup size <= 32 (WaveMatch8 uses only ballot .x bits).
bool GlSubgroupArithmeticAndBallotAvailable() {
    if (!GLEW_KHR_shader_subgroup) {
        return false;
    }
    GLint features = 0;
    glGetIntegerv(GL_SUBGROUP_SUPPORTED_FEATURES_KHR, &features);
    const GLint kRequired = GL_SUBGROUP_FEATURE_ARITHMETIC_BIT_KHR |
                            GL_SUBGROUP_FEATURE_BALLOT_BIT_KHR |
                            GL_SUBGROUP_FEATURE_SHUFFLE_BIT_KHR |
                            GL_SUBGROUP_FEATURE_VOTE_BIT_KHR;
    const bool features_ok = (features & kRequired) == kRequired;

    // WaveMatch8 uses subgroupBallot(...).x (bits 0-31 only); subgroup size
    // must be <= 32 for correctness.
    GLint max_sg = 0;
    glGetIntegerv(GL_SUBGROUP_SIZE_KHR, &max_sg);
    const bool size_ok = (max_sg > 0) && (max_sg <= 32);

    const bool ok = features_ok && size_ok;
    utility::LogDebug(
            "GaussianSplat: GL subgroup features=0x{:x} max_size={}, "
            "onesweep={}",
            features, max_sg, ok ? "yes" : "no");
    return ok;
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

/// Load one shader program.
/// Shaders named with the "_subgroup" suffix are the subgroup-arithmetic
/// variant.  When use_subgroups=false and the name ends in "_subgroup", the
/// suffix is stripped to load the portable no-subgroup version instead.
/// When use_precompiled=true, the SPIR-V binary is tried first, then the
/// GLSL source as a fallback.
static bool LoadOneProgram(GLComputeProgram& out,
                           const std::string& name,
                           const std::string& shader_root,
                           bool use_subgroups,
                           bool use_precompiled) {
    // Resolve file base: strip "_subgroup" suffix when subgroups are off.
    std::string file_base = name;
    constexpr std::string_view kSubgroupSuffix = "_subgroup";
    const bool is_subgroup =
            name.size() > kSubgroupSuffix.size() &&
            name.compare(name.size() - kSubgroupSuffix.size(),
                         kSubgroupSuffix.size(), kSubgroupSuffix) == 0;
    if (is_subgroup && !use_subgroups) {
        file_base = name.substr(0, name.size() - kSubgroupSuffix.size());
    }

    const std::string path = shader_root + file_base;
    const std::string spv_name = file_base + ".spv";
    const std::string comp_name = file_base + ".comp";

    // SPIR-V path (skipped when precompiled=off).
    if (use_precompiled) {
        std::vector<char> bytes;
        std::string error;
        if (utility::filesystem::FReadToBuffer(path + ".spv", bytes, &error)) {
            std::vector<std::uint8_t> spirv(bytes.begin(), bytes.end());
            out = LoadGLComputeProgramSPIRV(spirv, spv_name.c_str());
            if (out.valid) {
                utility::LogDebug("GaussianSplat: loaded {} via SPIR-V",
                                  spv_name);
                return true;
            }
        }
        utility::LogWarning(
                "GaussianSplat: SPIR-V path failed for {} — "
                "falling back to GLSL source compilation.",
                spv_name);
    }

    // Online GLSL path.
    {
        std::vector<char> bytes;
        std::string error;
        if (!utility::filesystem::FReadToBuffer(path + ".comp", bytes,
                                                &error)) {
            utility::LogWarning("GaussianSplat: GLSL source not found: {}.comp",
                                path);
            return false;
        }
        std::string glsl_source(bytes.begin(), bytes.end());
        out = LoadGLComputeProgramGLSL(glsl_source, comp_name.c_str());
        if (out.valid) {
            utility::LogDebug("GaussianSplat: loaded {} via online GLSL.",
                              comp_name);
            return true;
        }
        utility::LogWarning("GaussianSplat: GLSL compilation failed for {}.",
                            comp_name);
        return false;
    }
}

class GaussianSplatGpuContextOpenGL final : public GaussianSplatGpuContext {
public:
    /// use_subgroups: load _subgroup shader variants (faster on Linux/macOS).
    /// use_precompiled: try SPIR-V before online GLSL compilation.
    GaussianSplatGpuContextOpenGL(bool use_subgroups, bool use_precompiled)
        : use_subgroups_(use_subgroups), use_precompiled_(use_precompiled) {}

    ~GaussianSplatGpuContextOpenGL() override { CleanupPrograms(); }

    bool EnsureProgramsLoaded() override {
        if (programs_loaded_) {
            return programs_valid_;
        }
        programs_loaded_ = true;
        programs_valid_ = false;

        // Log driver identity for troubleshooting.
        utility::LogDebug(
                "GaussianSplat GL: vendor={}  renderer={}  version={}",
                reinterpret_cast<const char*>(glGetString(GL_VENDOR)),
                reinterpret_cast<const char*>(glGetString(GL_RENDERER)),
                reinterpret_cast<const char*>(glGetString(GL_VERSION)));
        utility::LogDebug("GaussianSplat GL: ARB_gl_spirv={}",
                          GLEW_ARB_gl_spirv ? "yes" : "no");

        const std::string shader_root =
                EngineInstance::GetResourcePath() + "/gaussian_splat/";

        // Phase 1: Load required base shaders [0, kGsFirstOneSweepProgram).
        // Attempt under the primary policy; retry once with the safe fallback.
        // OneSweep programs are loaded in Phase 2 below.
        bool cur_subgroups = use_subgroups_;
        bool cur_precompiled = use_precompiled_;
        for (int attempt = 0; attempt < 2; ++attempt) {
            bool all_ok = true;
            for (int i = 0; i < kGsFirstOneSweepProgram; ++i) {
                CleanupProgram(programs_[i]);
                if (!LoadOneProgram(programs_[i], kGsShaderNames[i],
                                    shader_root, cur_subgroups,
                                    cur_precompiled)) {
                    all_ok = false;
                    break;
                }
            }
            if (all_ok) {
                programs_valid_ = true;
                break;
            }

            if (attempt == 0) {
                for (int i = 0; i < kGsFirstOneSweepProgram; ++i) {
                    CleanupProgram(programs_[i]);
                }
                utility::LogWarning(
                        "GaussianSplat: primary shader policy failed. "
                        "Retrying with safe fallback "
                        "(subgroups=off, precompiled=off).");
                cur_subgroups = false;
                cur_precompiled = false;
            }
        }

        if (!programs_valid_) {
            utility::LogWarning(
                    "GaussianSplat: all shader load attempts failed.");
            return false;
        }

        // Phase 2: Optionally load OneSweep programs (failure is non-fatal).
        // OneSweep shaders require subgroup arithmetic + ballot (for both the
        // global-histogram SCAN mode and the digit-pass lookback).
        if (!GlSubgroupArithmeticAndBallotAvailable()) {
            onesweep_programs_valid_ = false;
            utility::LogDebug(
                    "GaussianSplat: OneSweep disabled — subgroup "
                    "arithmetic+ballot not available "
                    "(subgroup-basic radix sort still active).");
            return true;
        }

        onesweep_programs_valid_ = true;
        for (int i = kGsFirstOneSweepProgram;
             i < static_cast<int>(ComputeProgramId::kCount); ++i) {
            CleanupProgram(programs_[i]);
            if (!LoadOneProgram(programs_[i], kGsShaderNames[i], shader_root,
                                use_subgroups_, use_precompiled_)) {
                onesweep_programs_valid_ = false;
                // Clean up partially-loaded OneSweep programs.
                for (int j = kGsFirstOneSweepProgram; j <= i; ++j) {
                    CleanupProgram(programs_[j]);
                }
                utility::LogDebug(
                        "GaussianSplat: OneSweep programs not available "
                        "(falling back to 4-pass radix sort).");
                break;
            }
        }
        if (onesweep_programs_valid_) {
            utility::LogDebug("GaussianSplat: OneSweep programs loaded.");
        }

        return true;
    }

    bool AreOneSweepProgramsLoaded() const override {
        return onesweep_programs_valid_;
    }

    std::uintptr_t CreateBuffer(std::size_t size,
                                const char* label = nullptr) override {
        return AllocGLBuffer(size, /*priv=*/false, label);
    }

    /// GPU-private buffer: GL_DYNAMIC_COPY biases placement toward GPU-local
    /// VRAM on discrete GPUs.
    std::uintptr_t CreatePrivateBuffer(std::size_t size,
                                       const char* label = nullptr) override {
        return AllocGLBuffer(size, /*priv=*/true, label);
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
                                std::size_t new_size,
                                const char* label = nullptr) override {
        if (buf == 0) {
            return CreateBuffer(new_size, label);
        }
        GLBufferHandle nh =
                ResizeGLBuffer(ToGLBuffer(buf, buffer_sizes_), new_size, label);
        buffer_sizes_.erase(buf);
        if (!nh.valid) {
            return 0;
        }
        std::uintptr_t nk = static_cast<std::uintptr_t>(nh.id);
        buffer_sizes_[nk] = nh.size;
        return nk;
    }

    std::uintptr_t ResizePrivateBuffer(std::uintptr_t buf,
                                       std::size_t new_size,
                                       const char* label = nullptr) override {
        if (new_size == 0) {
            DestroyBuffer(buf);
            return 0;
        }
        GLBufferHandle nh = ResizeGLPrivateBuffer(
                ToGLBuffer(buf, buffer_sizes_), new_size, label);
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

    bool DownloadBuffer(std::uintptr_t buf,
                        void* dst,
                        std::size_t size,
                        std::size_t offset) override {
        return DownloadGLBuffer(ToGLBuffer(buf, buffer_sizes_), dst, size,
                                offset);
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
                                       std::uint32_t height,
                                       const char* label = nullptr) override {
        GLTextureHandle t = CreateGLTexture2D(width, height, kGL_R32F, label);
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
                                       std::uint32_t height,
                                       const char* label = nullptr) override {
        if (tex == 0) {
            return CreateTexture2DR32F(width, height, label);
        }
        GLTextureHandle nh = ResizeGLTexture2D(ToGLTexture(tex, texture_sizes_),
                                               width, height, kGL_R32F, label);
        texture_sizes_.erase(tex);
        if (!nh.valid) {
            return 0;
        }
        std::uintptr_t nk = static_cast<std::uintptr_t>(nh.id);
        texture_sizes_[nk] = {width, height};
        return nk;
    }

    std::uintptr_t ResizeTexture2DR16UI(std::uintptr_t tex,
                                        std::uint32_t width,
                                        std::uint32_t height,
                                        const char* label = nullptr) override {
        if (tex == 0) {
            GLTextureHandle t =
                    CreateGLTexture2D(width, height, kGL_R16UI, label);
            if (!t.valid) return 0;
            std::uintptr_t k = static_cast<std::uintptr_t>(t.id);
            texture_sizes_[k] = {width, height};
            return k;
        }
        GLTextureHandle nh = ResizeGLTexture2D(ToGLTexture(tex, texture_sizes_),
                                               width, height, kGL_R16UI, label);
        texture_sizes_.erase(tex);
        if (!nh.valid) return 0;
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
        std::uint32_t gl_fmt = kGL_R32F;
        if (fmt == ImageFormat::kRGBA16F)
            gl_fmt = kGL_RGBA16F;
        else if (fmt == ImageFormat::kR16UI)
            gl_fmt = kGL_R16UI;
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

    bool DownloadTextureR32F(std::uintptr_t tex,
                             std::uint32_t width,
                             std::uint32_t height,
                             std::vector<float>& out) override {
        if (tex == 0 || width == 0 || height == 0) return false;
        out.resize(static_cast<std::size_t>(width) * height);
        const GLuint id = static_cast<GLuint>(tex);
        glBindTexture(GL_TEXTURE_2D, id);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, out.data());
        glBindTexture(GL_TEXTURE_2D, 0);
        if (DrainGLErrors("DownloadTextureR32F") != 0) return false;
        // glGetTexImage returns bottom-up (OpenGL origin). Flip to top-down
        // so callers match the orientation of Filament readPixels output.
        for (std::uint32_t row = 0; row < height / 2; ++row) {
            std::swap_ranges(out.begin() + row * width,
                             out.begin() + row * width + width,
                             out.begin() + (height - 1 - row) * width);
        }
        return true;
    }

    bool DownloadTextureR16UI(std::uintptr_t tex,
                              std::uint32_t width,
                              std::uint32_t height,
                              std::vector<std::uint16_t>& out) override {
        if (tex == 0 || width == 0 || height == 0) return false;
        out.resize(static_cast<std::size_t>(width) * height);
        const GLuint id = static_cast<GLuint>(tex);
        glBindTexture(GL_TEXTURE_2D, id);
        // GL_RED_INTEGER + GL_UNSIGNED_SHORT matches the R16UI internal format.
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RED_INTEGER, GL_UNSIGNED_SHORT,
                      out.data());
        glBindTexture(GL_TEXTURE_2D, 0);
        if (DrainGLErrors("DownloadTextureR16UI") != 0) return false;
        // glGetTexImage returns bottom-up (OpenGL origin). Flip to top-down
        // so callers match the orientation of Filament readPixels output.
        for (std::uint32_t row = 0; row < height / 2; ++row) {
            std::swap_ranges(out.begin() + row * width,
                             out.begin() + row * width + width,
                             out.begin() + (height - 1 - row) * width);
        }
        return true;
    }

    void FinishGpuWork() override {
        GaussianSplatOpenGLContext::GetInstance().Finish();
    }

private:
    std::uintptr_t AllocGLBuffer(std::size_t size,
                                 bool priv,
                                 const char* label) {
        GLBufferHandle h = priv ? CreateGLPrivateBuffer(size, label)
                                : CreateGLBuffer(size, nullptr, label);
        if (!h.valid) {
            return 0;
        }
        std::uintptr_t k = static_cast<std::uintptr_t>(h.id);
        buffer_sizes_[k] = h.size;
        return k;
    }

    void CleanupProgram(GLComputeProgram& p) { DestroyGLComputeProgram(p); }

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
    bool onesweep_programs_valid_ = false;  ///< true if OneSweep loaded OK
    bool use_subgroups_;
    bool use_precompiled_;
    std::unordered_map<std::uintptr_t, std::size_t> buffer_sizes_;
    std::unordered_map<std::uintptr_t, std::pair<std::uint32_t, std::uint32_t>>
            texture_sizes_;
};

std::unique_ptr<GaussianSplatGpuContext> CreateComputeGpuContextGL(
        bool use_subgroups, bool use_precompiled) {
    return std::make_unique<GaussianSplatGpuContextOpenGL>(use_subgroups,
                                                           use_precompiled);
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // !defined(__APPLE__)
