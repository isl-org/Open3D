// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/visualization/gui/Widget.h"

#include "open3d/visualization/gui/UIImage.h"

namespace open3d {
namespace visualization {
namespace gui {

class ImageWidget : public Widget {
    using Super = Widget;

public:
    ImageWidget();
    /// Uses image from the specified path. Each ImageWidget will use one
    /// draw call.
    explicit ImageWidget(const char* image_path);
    /// Uses existing image. Each ImageWidget will use one draw call.
    explicit ImageWidget(std::shared_ptr<geometry::Image> image);
    /// Uses existing image. Each ImageWidget will use one draw call.
    explicit ImageWidget(std::shared_ptr<t::geometry::Image> image);
    /// Uses an existing texture, using texture coordinates
    /// (u0, v0) to (u1, v1). Does not deallocate texture on destruction.
    /// This is useful for using an icon atlas to reduce draw calls.
    explicit ImageWidget(visualization::rendering::TextureHandle texture_id,
                         float u0 = 0.0f,
                         float v0 = 0.0f,
                         float u1 = 1.0f,
                         float v1 = 1.0f);
    ImageWidget(std::shared_ptr<UIImage> image);
    ~ImageWidget();

    /// Mostly a convenience function for GetUIImage()->UpdateImage().
    /// If 'image' is the same size as the current image, will update the
    /// texture with the contents of 'image'. This is the fastest path for
    /// setting an image, and is recommended if you are displaying video.
    /// If 'image' is a different size, it will allocate a new texture,
    /// which is essentially the same as creating a new UIImage and calling
    /// SetUIImage(). This is the slow path, and may eventually exhaust internal
    /// texture resources.
    void UpdateImage(std::shared_ptr<geometry::Image> image);
    void UpdateImage(std::shared_ptr<t::geometry::Image> image);

    std::shared_ptr<UIImage> GetUIImage() const;
    void SetUIImage(std::shared_ptr<UIImage> image);

    Size CalcPreferredSize(const LayoutContext& context,
                           const Constraints& constraints) const override;

    void Layout(const LayoutContext& context) override;

    DrawResult Draw(const DrawContext& context) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
