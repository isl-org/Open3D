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

#include <Eigen/Geometry>

namespace open3d {
namespace visualization {
namespace rendering {

class Scene;
class Camera;

class View {
public:
    enum class TargetBuffers : std::uint8_t {
        None = 0u,
        Color = 1u,
        Depth = 2u,
        Stencil = 4u,

        ColorAndDepth = Color | Depth,
        ColorAndStencil = Color | Stencil,
        DepthAndStencil = Depth | Stencil,
        All = Color | Depth | Stencil
    };

    enum class Mode : std::uint8_t {
        Color = 0u,
        Depth,
        Normals,
        // This three modes always stay at end
        ColorMapX,
        ColorMapY,
        ColorMapZ
    };

    virtual ~View() {}

    virtual void SetDiscardBuffers(const TargetBuffers& buffers) = 0;
    virtual Mode GetMode() const = 0;
    virtual void SetMode(Mode mode) = 0;

    virtual void SetSampleCount(int n) = 0;
    virtual int GetSampleCount() const = 0;

    virtual void SetViewport(std::int32_t x,
                             std::int32_t y,
                             std::uint32_t w,
                             std::uint32_t h) = 0;
    virtual std::array<int, 4> GetViewport() const = 0;

    virtual void SetSSAOEnabled(bool enabled) = 0;

    virtual Camera* GetCamera() const = 0;
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
