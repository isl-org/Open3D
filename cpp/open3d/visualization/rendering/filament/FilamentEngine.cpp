// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/filament/FilamentEngine.h"

#include "open3d/utility/Logging.h"

// 4068: Filament has some clang-specific vectorizing pragma's that MSVC flags
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4068)
#endif  // _MSC_VER

#include <filament/Engine.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif  // _MSC_VER

#include <cstddef>  // <filament/Engine> recursive includes needs this, std::size_t especially
#include <cstdint>

#include "open3d/utility/FileSystem.h"
#include "open3d/visualization/rendering/filament/FilamentResourceManager.h"
#if !defined(__APPLE__)
#include "open3d/visualization/rendering/GpuAdapterSelection.h"
#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatOpenGLContext.h"
#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatVulkanInteropContext.h"
#endif

namespace open3d {
namespace visualization {
namespace rendering {

namespace {
static std::shared_ptr<EngineInstance> g_instance = nullptr;
}  // namespace

RenderingType EngineInstance::type_ = RenderingType::kDefault;
std::string EngineInstance::resource_path_ = "";
void* EngineInstance::shared_context_ = nullptr;

void EngineInstance::SelectBackend(RenderingType type) { type_ = type; }

void EngineInstance::SetResourcePath(const std::string& resource_path) {
    resource_path_ = resource_path;
    if (!utility::filesystem::DirectoryExists(resource_path_)) {
        utility::LogError(
                ("Can't find resource directory: " + resource_path_).c_str());
    }
}

const std::string& EngineInstance::GetResourcePath() { return resource_path_; }

void EngineInstance::SetSharedContext(void* shared_context) {
    shared_context_ = shared_context;
}

void* EngineInstance::GetSharedContext() { return shared_context_; }

filament::Engine& EngineInstance::GetInstance() { return *Get().engine_; }

FilamentResourceManager& EngineInstance::GetResourceManager() {
    return *Get().resource_manager_;
}

filament::backend::Platform* EngineInstance::GetPlatform() {
    return Get().engine_->getPlatform();
}

EngineInstance::~EngineInstance() {
    resource_manager_->DestroyAll();
    delete resource_manager_;
    resource_manager_ = nullptr;

    filament::Engine::destroy(engine_);
    engine_ = nullptr;

#if !defined(__APPLE__)
    GaussianSplatOpenGLContext::GetInstance().Shutdown();
    GaussianSplatVulkanInteropContext::GetInstance().Shutdown();
    // The GLX context handle is now destroyed; clear the cached pointer so
    // that the next EngineInstance creation re-initialises the compute
    // context and passes a fresh handle to Filament's Engine::create().
    shared_context_ = nullptr;
#endif
}

EngineInstance& EngineInstance::Get() {
    if (!g_instance) {
        g_instance = std::shared_ptr<EngineInstance>(new EngineInstance());
    }
    return *g_instance;
}

void EngineInstance::DestroyInstance() { g_instance.reset(); }

EngineInstance::EngineInstance() {
    filament::backend::Backend backend = filament::backend::Backend::DEFAULT;
    switch (type_) {
        case RenderingType::kDefault:
            backend = filament::backend::Backend::DEFAULT;
            break;
        case RenderingType::kOpenGL:
            backend = filament::backend::Backend::OPENGL;
            break;
        case RenderingType::kVulkan:
            backend = filament::backend::Backend::VULKAN;
            break;
        case RenderingType::kMetal:
            backend = filament::backend::Backend::METAL;
            break;
    }

#if !defined(__APPLE__)
    // Filament's DEFAULT backend on Windows and Linux resolves to Vulkan (or
    // the Vulkan D3D12 emulation layer on Windows), which conflicts with our
    // OpenGL-based compute context sharing for Gaussian splatting.  Filament
    // only supports zero copy buffer sharing on Metal and OpenGL. Also, on
    // Windows, Vulkan sometimes defaults to the D3D12 emulated Vulkan GPU,
    // which does not support triple buffering and causes a crash at startup.
    // Force OpenGL unconditionally.
    if (backend == filament::backend::Backend::DEFAULT) {
        backend = filament::backend::Backend::OPENGL;
    }

    // Vulkan selects its physical device first (discrete-GPU-preferred,
    // emulated/software devices down-weighted — see ScoreDevice()), honoring
    // any loader-level device reordering (e.g. VK_LOADER_DEVICE_SELECT on
    // Linux) for free. The compute GL context is then steered onto that same
    // adapter before creation: GL_EXT_memory_object cross-adapter import is
    // not supported and silently fails (GL_OUT_OF_MEMORY) on multi-GPU
    // (hybrid graphics) systems if Vulkan and GL end up on different GPUs.
    auto& vk_ctx = GaussianSplatVulkanInteropContext::GetInstance();
    if (!vk_ctx.IsValid() && !vk_ctx.Initialize()) {
        utility::LogWarning(
                "EngineInstance: Vulkan interop context init failed: {}",
                vk_ctx.GetLastError());
    }

    auto& gl_ctx = GaussianSplatOpenGLContext::GetInstance();
    if ((backend == filament::backend::Backend::OPENGL ||
         backend == filament::backend::Backend::DEFAULT) &&
        !shared_context_) {
        if (!gl_ctx.IsValid()) {
            GpuAdapterInfo vk_adapter_info;
            if (vk_ctx.IsValid()) {
                vk_adapter_info = GetAdapterInfo(vk_ctx.GetPhysicalDevice());
                SteerNextGLContextToAdapter(vk_adapter_info);
            }
            gl_ctx.InitializeStandalone();

            // Safety net only: GaussianSplatVulkanInteropContext::Initialize()
            // already avoids picking a monitor-less adapter when it can, so
            // steering above should normally succeed. This still guards
            // against rarer cases (e.g. the window manager not honoring the
            // position hint) by making Vulkan follow GL if they still
            // disagree — guaranteeing the two share a GPU (required for
            // GL_EXT_memory_object import) matters more than which GPU is
            // used.
            if (gl_ctx.IsValid() && vk_adapter_info.valid) {
                const GpuAdapterInfo gl_actual =
                        GetAdapterInfoForWindow(gl_ctx.GetNativeWindowHandle());
                if (gl_actual.valid &&
                    !SameAdapter(gl_actual, vk_adapter_info)) {
                    utility::LogWarning(
                            "EngineInstance: GL landed on adapter '{}' but "
                            "Vulkan selected '{}'; reinitializing Vulkan to "
                            "match GL so compute interop works correctly.",
                            gl_actual.device_name, vk_adapter_info.device_name);
                    vk_ctx.Shutdown();
                    if (!vk_ctx.Initialize(&gl_actual)) {
                        utility::LogWarning(
                                "EngineInstance: Vulkan reinit to match GL "
                                "adapter failed: {}",
                                vk_ctx.GetLastError());
                    }
                }
            }
        }
        if (gl_ctx.IsValid()) {
            shared_context_ = gl_ctx.GetNativeContext();
            utility::LogDebug(
                    "EngineInstance: passing GS compute context to Filament "
                    "as sharedGLContext ({:p}).",
                    shared_context_);
        }
    }

    if (vk_ctx.IsValid() && gl_ctx.IsValid() &&
        !vk_ctx.AreGLExtensionsReady()) {
        if (gl_ctx.MakeCurrent()) {
            vk_ctx.ProbeGLExtensions();
            gl_ctx.ReleaseCurrent();
        }
    }
    // Diagnostic-only verification that GL and Vulkan ended up on the same
    // adapter; never gates behavior (some drivers fail this query, see
    // GetCurrentGLAdapterUUID()'s doc comment).
    if (gl_ctx.IsValid() && gl_ctx.MakeCurrent()) {
        utility::LogDebug("EngineInstance: GL adapter UUID = {}",
                          HexEncode(GetCurrentGLAdapterUUID()));
        gl_ctx.ReleaseCurrent();
    }
#endif

    filament::Engine::Config fmcfg;
    fmcfg.stereoscopicType = filament::Engine::StereoscopicType::INSTANCED;
    fmcfg.stereoscopicEyeCount = 1;  // We do not support stereo.
    engine_ =
            filament::Engine::create(backend, nullptr, shared_context_, &fmcfg);
    if (!engine_) {
        utility::LogError("Failed to create Filament engine.");
    }

    resource_manager_ = new FilamentResourceManager(*engine_);
    // Query and record the backend selected by filament for future use (e.g.
    // for ImGui)
    switch (engine_->getBackend()) {
        case filament::backend::Backend::OPENGL:
            type_ = RenderingType::kOpenGL;
            break;
        case filament::backend::Backend::VULKAN:
            type_ = RenderingType::kVulkan;
            break;
        case filament::backend::Backend::METAL:
            type_ = RenderingType::kMetal;
            break;
        default:;  // no update
    }
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
