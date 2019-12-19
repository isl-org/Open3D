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

#include "Open3D/Visualization/Rendering/View.h"

#include <memory>

#include <filament/Color.h>

namespace filament {
class Engine;
class Scene;
class View;
class Viewport;
}  // namespace filament

namespace open3d {
namespace visualization {

class FilamentCamera;

class FilamentView : public View {
public:
    FilamentView(filament::Engine& engine, filament::Scene& scene);
    ~FilamentView() override;

    void SetViewport(std::int32_t x,
                     std::int32_t y,
                     std::uint32_t w,
                     std::uint32_t h) override;
    void SetClearColor(const Eigen::Vector3f& color) override;

    Camera* GetCamera() const override;

    filament::View* GetNativeView() const { return view_; }

private:
    std::unique_ptr<FilamentCamera> camera_;

    filament::Engine& engine_;
    filament::Scene& scene_;
    filament::View* view_ = nullptr;
};

}  // namespace visualization
}  // namespace open3d