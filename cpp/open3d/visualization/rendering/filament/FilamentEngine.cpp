// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/filament/FilamentEngine.h"

#include <memory>

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

#if !defined(__APPLE__) && __has_include(<backend/platforms/VulkanPlatform.h>)
#include <backend/platforms/VulkanPlatform.h>
#define OPEN3D_HAS_FILAMENT_VULKAN_PLATFORM 1
#endif

#if __has_include(<backend/platforms/PlatformMetal.h>)
#include <backend/platforms/PlatformMetal.h>
#define OPEN3D_HAS_FILAMENT_METAL_PLATFORM 1
#endif

#include <cstddef>  // <filament/Engine> recursive includes needs this, std::size_t especially

#include "open3d/utility/FileSystem.h"
#include "open3d/visualization/rendering/filament/FilamentResourceManager.h"

namespace open3d {
namespace visualization {
namespace rendering {

namespace {
static std::shared_ptr<EngineInstance> g_instance = nullptr;

std::unique_ptr<filament::backend::Platform> CreateBackendPlatform(
        RenderingType type) {
    if (type == RenderingType::kDefault) {
#if defined(__APPLE__) && defined(OPEN3D_HAS_FILAMENT_METAL_PLATFORM)
        return std::unique_ptr<filament::backend::Platform>(
                new filament::backend::PlatformMetal());
#elif defined(OPEN3D_HAS_FILAMENT_VULKAN_PLATFORM)
        return std::unique_ptr<filament::backend::Platform>(
                new filament::backend::VulkanPlatform());
#else
        return nullptr;
#endif
    }

    switch (type) {
        case RenderingType::kDefault:
            break;
#if defined(__APPLE__)
        case RenderingType::kVulkan:
            utility::LogWarning(
                    "Filament Vulkan backend is disabled on macOS while "
                    "Gaussian "
                    "compute uses Metal-only interop. Falling back to Metal.");
#if defined(OPEN3D_HAS_FILAMENT_METAL_PLATFORM)
            return std::unique_ptr<filament::backend::Platform>(
                    new filament::backend::PlatformMetal());
#else
            return nullptr;
#endif
#endif
#if defined(OPEN3D_HAS_FILAMENT_VULKAN_PLATFORM)
        case RenderingType::kVulkan:
            return std::unique_ptr<filament::backend::Platform>(
                    new filament::backend::VulkanPlatform());
#endif
#if defined(OPEN3D_HAS_FILAMENT_METAL_PLATFORM)
        case RenderingType::kMetal:
            return std::unique_ptr<filament::backend::Platform>(
                    new filament::backend::PlatformMetal());
#endif
        case RenderingType::kOpenGL:
            break;
    }
    return nullptr;
}
}  // namespace

#ifdef _WIN32
// Default for Windows is Vulkan, but this sometimes selects the Direct3D12
// emulated backend, which is not fully functional. Force OpenGL instead.
RenderingType EngineInstance::type_ = RenderingType::kOpenGL;
#else
RenderingType EngineInstance::type_ = RenderingType::kDefault;
#endif
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
    return Get().platform_.get();
}

EngineInstance::~EngineInstance() {
    resource_manager_->DestroyAll();
    delete resource_manager_;
    resource_manager_ = nullptr;

    filament::Engine::destroy(engine_);
    engine_ = nullptr;
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
#if defined(__APPLE__) && defined(OPEN3D_HAS_FILAMENT_METAL_PLATFORM)
            backend = filament::backend::Backend::METAL;
#else
            backend = filament::backend::Backend::DEFAULT;
#endif
            break;
        case RenderingType::kOpenGL:
            backend = filament::backend::Backend::OPENGL;
            break;
        case RenderingType::kVulkan:
#if defined(__APPLE__) && defined(OPEN3D_HAS_FILAMENT_METAL_PLATFORM)
            utility::LogWarning(
                    "Filament Vulkan backend is disabled on macOS while "
                    "Gaussian compute uses Metal-only interop. Falling back "
                    "to Metal.");
            type_ = RenderingType::kMetal;
            backend = filament::backend::Backend::METAL;
#else
            backend = filament::backend::Backend::VULKAN;
#endif
            break;
        case RenderingType::kMetal:
            backend = filament::backend::Backend::METAL;
            break;
    }

    platform_ = CreateBackendPlatform(type_);

    filament::Engine::Builder builder;
    builder.backend(backend);
    if (platform_) {
        builder.platform(platform_.get());
    }
    if (shared_context_) {
        builder.sharedContext(shared_context_);
    }

    engine_ = builder.build();
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
