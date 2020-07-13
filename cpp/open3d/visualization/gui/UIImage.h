// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include "open3d/visualization/rendering/RendererHandle.h"

namespace open3d {
namespace visualization {

namespace rendering {
class Renderer;
}

namespace gui {

class UIImage {
public:
    /// Uses image from the specified path. Each ImageLabel will use one
    /// draw call.
    explicit UIImage(const char* image_path);
    /// Uses an existing texture, using texture coordinates
    /// (u0, v0) to (u1, v1). Does not deallocate texture on destruction.
    /// This is useful for using an icon atlas to reduce draw calls.
    explicit UIImage(visualization::rendering::TextureHandle texture_id,
                     float u0 = 0.0f,
                     float v0 = 0.0f,
                     float u1 = 1.0f,
                     float v1 = 1.0f);
    ~UIImage();

    enum class Scaling {
        NONE,   /// No scaling, fixed size
        ANY,    /// Scales to any size and aspect ratio
        ASPECT  /// Scales to any size, but fixed aspect ratio (default)
    };
    void SetScaling(Scaling scaling);
    Scaling GetScaling() const;

    Size CalcPreferredSize(const Theme& theme) const;

    struct DrawParams {
        // Default values are to make GCC happy and contented,
        // pos and size don't have reasonable defaults.
        float pos_x = 0.0f;
        float pos_y = 0.0f;
        float width = 0.0f;
        float height = 0.0f;
        float u0 = 0.0f;
        float v0 = 0.0f;
        float u1 = 1.0f;
        float v1 = 1.0f;
        visualization::rendering::TextureHandle texture;
    };
    DrawParams CalcDrawParams(visualization::rendering::Renderer& renderer,
                              const Rect& frame) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
