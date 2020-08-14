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

#pragma once

// 4068: Filament has some clang-specific vectorizing pragma's that MSVC flags
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4068)
#endif  // _MSC_VER

#include <filament/Engine.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif  // _MSC_VER

namespace open3d {
namespace visualization {
namespace rendering {

class FilamentResourceManager;

class EngineInstance {
public:
    // Selects backend to use.
    // Should be called before instance usage.
    // If not called, platform available default backend will be used.
    static void SelectBackend(filament::backend::Backend backend);

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

    static filament::backend::Backend backend_;
    filament::Engine* engine_;
    FilamentResourceManager* resource_manager_;
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
