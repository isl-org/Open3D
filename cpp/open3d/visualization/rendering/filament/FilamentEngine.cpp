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
#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatVulkanContext.h"
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
    // Shutdown must happen AFTER Engine::destroy() because Filament does
    // NOT destroy a shared VkDevice (VulkanPlatform::terminate() guards
    // with mSharedContext).  The device is owned by Open3D.
    GaussianSplatVulkanContext::GetInstance().Shutdown();
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
    // Default to VULKAN on Linux/Windows now that the GS compute pipeline
    // runs on the same VkDevice (no GL interop needed).  Use
    // RenderingType::kOpenGL for a legacy OpenGL-only path (GS disabled).
    if (backend == filament::backend::Backend::DEFAULT) {
        backend = filament::backend::Backend::VULKAN;
    }

    if (backend == filament::backend::Backend::VULKAN) {
        auto& vk_ctx = GaussianSplatVulkanContext::GetInstance();
        if (!vk_ctx.IsValid() && !vk_ctx.Initialize()) {
            utility::LogWarning(
                    "EngineInstance: Vulkan context init failed: {}",
                    vk_ctx.GetLastError());
        }
        if (vk_ctx.IsValid() && !vk_ctx.IsSoftwareDevice()) {
            shared_context_ = vk_ctx.GetVulkanSharedContext();
        } else if (vk_ctx.IsSoftwareDevice()) {
            utility::LogInfo(
                    "EngineInstance: Vulkan software device detected; using "
                    "Filament's OpenGL backend");
            backend = filament::backend::Backend::OPENGL;
        }
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
