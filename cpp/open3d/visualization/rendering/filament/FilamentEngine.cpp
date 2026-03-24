// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
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

#include "open3d/utility/FileSystem.h"
#include "open3d/visualization/rendering/filament/FilamentResourceManager.h"
#if !defined(__APPLE__)
#include "open3d/visualization/rendering/filament/GaussianComputeOpenGLContext.h"
#endif

namespace open3d {
namespace visualization {
namespace rendering {

namespace {
static std::shared_ptr<EngineInstance> g_instance = nullptr;
}  // namespace

#if defined(_WIN32) || defined(__linux__)
// Default for Windows/Linux is Vulkan, but this sometimes selects the
// Direct3D12 emulated backend on Windows or has issues with Vulkan on Linux.
// Force OpenGL instead. OpenGL also enables compute-based Gaussian splatting
// with zero-copy output via Filament's import() API.
RenderingType EngineInstance::type_ = RenderingType::kOpenGL;
#else
// macOS uses Metal via kDefault.
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
    return Get().engine_->getPlatform();
}

EngineInstance::~EngineInstance() {
    resource_manager_->DestroyAll();
    delete resource_manager_;
    resource_manager_ = nullptr;

    filament::Engine::destroy(engine_);
    engine_ = nullptr;

#if !defined(__APPLE__)
    GaussianComputeOpenGLContext::GetInstance().Shutdown();
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

        // On Linux (X11/GLX), create our compute GL context BEFORE the Filament
        // engine so we can pass it as the sharedGLContext.  Filament's
        // PlatformGLX will then create its own context sharing our GL
        // namespace, enabling zero-copy texture import() between the two
        // contexts. This must happen before Engine::create() because GLX
        // context sharing can only be established at context creation time.
#if !defined(__APPLE__) && !defined(_WIN32)
    if ((backend == filament::backend::Backend::OPENGL ||
         backend == filament::backend::Backend::DEFAULT) &&
        !shared_context_) {
        auto& gl_ctx = GaussianComputeOpenGLContext::GetInstance();
        if (!gl_ctx.IsValid()) {
            gl_ctx.InitializeStandalone();
        }
        if (gl_ctx.IsValid()) {
            shared_context_ = gl_ctx.GetNativeContext();
            utility::LogDebug(
                    "EngineInstance: passing GS compute context to Filament "
                    "as sharedGLContext ({:p}).",
                    shared_context_);
        }
    }
#endif

    engine_ = filament::Engine::create(backend, nullptr, shared_context_);
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
