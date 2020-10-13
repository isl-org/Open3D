// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
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
std::string EngineInstance::resource_path_ = "";

void EngineInstance::SelectBackend(RenderingType type) { type_ = type; }

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

    engine_ = filament::Engine::create(backend);
    resource_manager_ = new FilamentResourceManager(*engine_);
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
