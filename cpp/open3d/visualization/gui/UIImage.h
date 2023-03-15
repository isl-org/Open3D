// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/visualization/gui/Widget.h"
#include "open3d/visualization/rendering/RendererHandle.h"

namespace open3d {

namespace t {
namespace geometry {
class Image;
}
}

namespace geometry {
class Image;
}  // namespace geometry

namespace visualization {

namespace rendering {
class Renderer;
}

namespace gui {

class UIImage {
public:
    explicit UIImage(const char* image_path);
    explicit UIImage(std::shared_ptr<geometry::Image> image);
    explicit UIImage(std::shared_ptr<t::geometry::Image> image);
    /// Uses an existing texture, using texture coordinates
    /// (u0, v0) to (u1, v1). Does not deallocate texture on destruction.
    /// This is useful for using an icon atlas to reduce draw calls.
    explicit UIImage(visualization::rendering::TextureHandle texture_id,
                     float u0 = 0.0f,
                     float v0 = 0.0f,
                     float u1 = 1.0f,
                     float v1 = 1.0f);
    ~UIImage();

    /// Updates the contents of the texture. If the image is a different
    /// size from the original, a new texture will be created.
    void UpdateImage(std::shared_ptr<geometry::Image> image);

    /// Updates the contents of the texture. If the image is a different
    /// size from the original, a new texture will be created.
    void UpdateImage(std::shared_ptr<t::geometry::Image> image);

    enum class Scaling {
        NONE,   /// No scaling, fixed size
        ANY,    /// Scales to any size and aspect ratio
        ASPECT  /// Scales to any size, but fixed aspect ratio (default)
    };
    void SetScaling(Scaling scaling);
    Scaling GetScaling() const;

    Size CalcPreferredSize(const LayoutContext& context,
                           const Widget::Constraints& constraints) const;

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
        bool image_size_changed = false;
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
