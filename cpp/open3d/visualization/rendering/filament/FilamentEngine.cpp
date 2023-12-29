// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/filament/FilamentEngine.h"

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

#include "open3d/utility/FileSystem.h"
#include "open3d/visualization/rendering/filament/FilamentResourceManager.h"

namespace open3d {
namespace visualization {
namespace rendering {

namespace {
static std::shared_ptr<EngineInstance> g_instance = nullptr;
}  // namespace

EngineInstance::RenderingType EngineInstance::type_ = RenderingType::kDefault;
bool EngineInstance::is_headless_ = false;
std::string EngineInstance::resource_path_ = "";

void EngineInstance::SelectBackend(RenderingType type) { type_ = type; }

void EngineInstance::EnableHeadless() { is_headless_ = true; }

void EngineInstance::SetResourcePath(const std::string& resource_path) {
    resource_path_ = resource_path;
    if (!utility::filesystem::DirectoryExists(resource_path_)) {
        utility::LogError(
                ("Can't find resource directory: " + resource_path_).c_str());
    }
}

const std::string& EngineInstance::GetResourcePath() { return resource_path_; }

filament::Engine& EngineInstance::GetInstance() { return *Get().engine_; }

FilamentResourceManager& EngineInstance::GetResourceManager() {
    return *Get().resource_manager_;
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

/// external function defined in custom Filament EGL backend for headless
/// rendering
extern "C" filament::backend::Platform* CreateEGLHeadlessPlatform();

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

    filament::backend::Platform* custom_platform = nullptr;
    if (is_headless_) {
#ifdef __linux__
        utility::LogInfo("EGL headless mode enabled.");
        custom_platform = CreateEGLHeadlessPlatform();
#else
        utility::LogError("EGL Headless is not supported on this platform.");
#endif
    }

    engine_ = filament::Engine::create(backend, custom_platform);
    resource_manager_ = new FilamentResourceManager(*engine_);
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
