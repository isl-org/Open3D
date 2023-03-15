// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>

#include "open3d/visualization/gui/Color.h"
namespace open3d {
namespace visualization {
namespace gui {

// Label3D is a helper class for labels (like UI Labels) at 3D points as opposed
// to screen points. It is NOT a UI widget but is instead used via Open3DScene
// class. See Open3DScene::AddLabel/RemoveLabel.
class Label3D {
public:
    /// Copies text
    explicit Label3D(const Eigen::Vector3f& pos, const char* text = nullptr);
    ~Label3D();

    const char* GetText() const;
    /// Sets the text of the label (copies text)
    void SetText(const char* text);

    Eigen::Vector3f GetPosition() const;
    void SetPosition(const Eigen::Vector3f& pos);

    /// Returns the color with which the text will be drawn
    Color GetTextColor() const;

    /// Set the color with which the text will be drawn
    void SetTextColor(const Color& color);

    /// Get the current scale. See not below on meaning of scale.
    float GetTextScale() const;

    /// Sets the scale factor for the text sprite. It does not change the
    /// underlying font size but scales how large it appears. Warning: large
    /// scale factors may result in blurry text.
    void SetTextScale(float scale);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
