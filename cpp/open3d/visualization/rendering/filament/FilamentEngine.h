// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

namespace filament {
class Engine;
}

namespace open3d {
namespace visualization {
namespace rendering {

class FilamentResourceManager;

class EngineInstance {
public:
    enum class RenderingType { kDefault, kOpenGL, kVulkan, kMetal };

    // Selects backend to use.
    // Should be called before instance usage.
    // If not called, platform available default backend will be used.
    static void SelectBackend(RenderingType type);

    // Specifies path to load shaders and skyboxes from. Must be called before
    // instance usage, or default path will be used.
    static void SetResourcePath(const std::string& resource_path);
    static const std::string& GetResourcePath();

    static filament::Engine& GetInstance();
    static FilamentResourceManager& GetResourceManager();

    /// Destroys the singleton instance, to force Filament cleanup at a
    /// specific time. Calling GetInstance() after this will re-create
    /// the instance.
    static void DestroyInstance();

    ~EngineInstance();

private:
    static EngineInstance& Get();

    EngineInstance();

    static RenderingType type_;
    static std::string resource_path_;
    filament::Engine* engine_;
    FilamentResourceManager* resource_manager_;
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
