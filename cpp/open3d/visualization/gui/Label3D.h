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
