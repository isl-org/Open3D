// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include <Eigen/Core>
#include <map>
#include <vector>

#include "open3d/visualization/rendering/RendererHandle.h"

namespace open3d {
namespace visualization {
namespace rendering {

class Renderer;

/// Manages a gradient for the unlitGradient shader.
/// In gradient mode, the array of points specifies points along the gradient,
/// from 0 to 1 (inclusive). These do need to be evenly spaced.
/// Simple greyscale:
///     [ { 0.0, black },
///       { 1.0, white } ]
/// Rainbow (note the gaps around green):
///     [ { 0.000, blue },
///       { 0.125, cornflower blue },
///       { 0.250, cyan },
///       { 0.500, green },
///       { 0.750, yellow },
///       { 0.875, orange },
///       { 1.000, red } ]
/// The gradient will generate a largish texture, so it should be fairly
/// smooth, but the boundaries may not be exactly as specified due to
/// quanitization imposed by the fixed size of the texture.
///   The points *must* be sorted from the smallest value to the largest.
/// The values must be in the range [0, 1].
class Gradient {
public:
    struct Point {
        float value;
        Eigen::Vector4f color;
    };

    Gradient();
    Gradient(const std::vector<Gradient::Point>& points);
    virtual ~Gradient();

    const std::vector<Gradient::Point>& GetPoints() const;
    void SetPoints(const std::vector<Gradient::Point>& points);

    enum class Mode {
        kGradient,  /// Normal gradient mode
        kLUT        /// Point.value will be ignored and the colors will be
                    /// assumed to be evenly spaced. The texture will have only
                    /// as many pixels as there are points.
    };

    Mode GetMode() const;
    void SetMode(Mode mode);

    TextureHandle GetTextureHandle(Renderer& renderer);

private:
    std::vector<Gradient::Point> points_;
    Mode mode_ = Mode::kGradient;
    std::map<Renderer*, TextureHandle> textures_;
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
