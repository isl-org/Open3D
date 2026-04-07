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

#include <cstdlib>
#include <memory>
#include <unordered_map>
#include <unordered_set>
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

// ---------------------------------------------------------------------------
// Shader load policy
// ---------------------------------------------------------------------------

/// Which shader variant + loader to use for each program.
/// Controlled by platform defaults, overridable via env vars:
///   OPEN3D_SHADER_SUBGROUPS=0|1   — 0: no-subgroup variant (safe everywhere)
///                                    1: _subgroup variant (faster on good drivers)
///   OPEN3D_SHADER_PRECOMPILED=0|1 — 0: online GLSL compilation
///                                    1: SPIR-V binary
///
/// Platform defaults:
///   Windows → subgroups=off, precompiled=off  (Intel/AMD WGL GL driver issues)
///   Linux/macOS → subgroups=on, precompiled=on
struct ShaderLoadPolicy {
    bool use_subgroups;   // true: load gaussian_*_subgroup.{spv,comp}
    bool use_precompiled; // true: prefer .spv; false: online GLSL only
};

/// Shader base names that have a _subgroup variant.
/// All other shaders are unaffected by use_subgroups.
static const std::unordered_set<std::string> kSubgroupCapableShaders = {
        "gaussian_prefix_sum",
        "gaussian_radix_sort",
};

/// Read an optional boolean env var.  Returns the default if unset or invalid.
static bool ReadBoolEnv(const char* name, bool default_val) {
    const char* val = std::getenv(name);
    if (val == nullptr) return default_val;
    if (val[0] == '1' && val[1] == '\0') return true;
    if (val[0] == '0' && val[1] == '\0') return false;
    utility::LogWarning(
            "GaussianCompute: env var {} has unexpected value '{}'; "
            "expected 0 or 1. Using default ({}).",
            name, val, default_val ? 1 : 0);
    return default_val;
}

/// Determine the shader load policy from platform defaults and env var overrides.
static ShaderLoadPolicy GetShaderLoadPolicy() {
#if defined(_WIN32)
    // Windows OpenGL drivers (especially Intel/AMD) miscompile subgroup
    // arithmetic both via SPIR-V (GL_ARB_gl_spirv) and online GLSL.
    // Safe default: no subgroups + online GLSL compilation.
    bool default_subgroups   = false;
    bool default_precompiled = false;
#else
    // Linux and macOS (via XWayland/GLX or Metal transpilation) handle
    // subgroup intrinsics and SPIR-V loading correctly.
    bool default_subgroups   = true;
    bool default_precompiled = true;
#endif

    bool use_subgroups   = ReadBoolEnv("OPEN3D_SHADER_SUBGROUPS",   default_subgroups);
    bool use_precompiled = ReadBoolEnv("OPEN3D_SHADER_PRECOMPILED",  default_precompiled);

    const char* sg_source  = (std::getenv("OPEN3D_SHADER_SUBGROUPS")   != nullptr)
                                     ? "env OPEN3D_SHADER_SUBGROUPS"   : "platform default";
    const char* pre_source = (std::getenv("OPEN3D_SHADER_PRECOMPILED") != nullptr)
                                     ? "env OPEN3D_SHADER_PRECOMPILED" : "platform default";

    utility::LogInfo(
            "GaussianCompute: shader policy: subgroups={} ({}), "
            "precompiled={} ({}).",
            use_subgroups ? "on" : "off", sg_source,
            use_precompiled ? "on" : "off", pre_source);

    return {use_subgroups, use_precompiled};
}

/// Load one shader program according to policy.
/// Returns true on success; logs warnings on failure.
/// base      = canonical name, e.g. "gaussian_prefix_sum"
/// shader_root = filesystem directory (with trailing slash)
static bool LoadOneProgram(GLComputeProgram& out,
                           const std::string& base,
                           const std::string& shader_root,
                           const ShaderLoadPolicy& policy) {
    // Choose the file base name: _subgroup variant if policy requests it and
    // the shader has one; otherwise the standard name.
    std::string file_base = base;
    if (policy.use_subgroups && kSubgroupCapableShaders.count(base)) {
        file_base = base + "_subgroup";
    }

    const std::string path     = shader_root + file_base;
    const std::string spv_name = file_base + ".spv";
    const std::string comp_name= file_base + ".comp";

    // --- SPIR-V path (skipped when precompiled=off) ---
    if (policy.use_precompiled) {
        std::vector<char> bytes;
        std::string error;
        if (utility::filesystem::FReadToBuffer(path + ".spv", bytes, &error)) {
            std::vector<std::uint8_t> spirv(bytes.begin(), bytes.end());
            out = LoadGLComputeProgramSPIRV(spirv, spv_name.c_str());
            if (out.valid) {
                utility::LogDebug("GaussianCompute: loaded {} via SPIR-V",
                                  spv_name);
                return true;
            }
        }
        utility::LogWarning(
                "GaussianCompute: SPIR-V path failed for {} — "
                "falling back to GLSL source compilation.",
                spv_name);
    }

    // --- Online GLSL path ---
    {
        std::vector<char> bytes;
        std::string error;
        if (!utility::filesystem::FReadToBuffer(path + ".comp", bytes, &error)) {
            utility::LogWarning(
                    "GaussianCompute: GLSL source not found: {}.comp",
                    path);
            return false;
        }
        std::string glsl_source(bytes.begin(), bytes.end());
        out = LoadGLComputeProgramGLSL(glsl_source, comp_name.c_str());
        if (out.valid) {
            utility::LogInfo("GaussianCompute: loaded {} via online GLSL.",
                             comp_name);
            return true;
        }
        utility::LogWarning(
                "GaussianCompute: GLSL compilation failed for {}.", comp_name);
        return false;
    }
}

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

        ShaderLoadPolicy policy = GetShaderLoadPolicy();

        // Attempt to load all shaders under the primary policy.
        // On failure, retry once with the safe fallback policy
        // (no subgroups + online GLSL), which works on all known GPUs.
        for (int attempt = 0; attempt < 2; ++attempt) {
            bool all_ok = true;
            for (int i = 0; i < static_cast<int>(ComputeProgramId::kCount);
                 ++i) {
                CleanupProgram(programs_[i]);
                if (!LoadOneProgram(programs_[i], kGsShaderNames[i],
                                    shader_root, policy)) {
                    all_ok = false;
                    break;
                }
            }
            if (all_ok) {
                programs_valid_ = true;
                return true;
            }

            if (attempt == 0) {
                // Primary policy failed.  Clean up partial state and retry
                // with the universally safe fallback.
                for (auto& p : programs_) CleanupProgram(p);
                utility::LogWarning(
                        "GaussianCompute: primary shader policy failed. "
                        "Retrying with safe fallback "
                        "(subgroups=off, precompiled=off).");
                policy = {/*use_subgroups=*/false,
                          /*use_precompiled=*/false};
            }
        }

        utility::LogWarning(
                "GaussianCompute: all shader load attempts failed.");
        return false;
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
                ResizeGLBuffer(ToGLBuffer(buf, buffer_sizes_), new_size,
                               label);
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
        GLBufferHandle nh =
                ResizeGLPrivateBuffer(ToGLBuffer(buf, buffer_sizes_), new_size,
                                      label);
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

    void CleanupProgram(GLComputeProgram& p) {
        DestroyGLComputeProgram(p);
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
