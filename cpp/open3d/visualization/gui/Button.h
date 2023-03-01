// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/visualization/gui/Widget.h"

#include <functional>

#include "open3d/visualization/gui/UIImage.h"

namespace open3d {
namespace visualization {
namespace gui {

class Button : public Widget {
public:
    explicit Button(const char* title);
    explicit Button(std::shared_ptr<UIImage> image);
    ~Button();

    /// Returns the text of the button. If the button is an image, will return
    /// "".
    const char* GetText() const;
    /// Sets the text of the button. Do not call if this is an image button.
    void SetText(const char* text);

    /// Returns the padding, in units of ems
    float GetHorizontalPaddingEm() const;
    float GetVerticalPaddingEm() const;
    /// Sets the padding, in units of ems. Note that for text buttons, a
    /// padding of (0, 0) will not actually give a padding of 0, there will
    /// be a small padding because having zero padding looks horrible and
    /// because this way a vertical padding of zero is exactly the same
    /// size as a text row, which means the button is the same size as the
    /// the other text-based widgets, so it will look nice with them.
    void SetPaddingEm(float horiz_ems, float vert_ems);

    bool GetIsToggleable() const;
    void SetToggleable(bool toggles);

    bool GetIsOn() const;
    void SetOn(bool is_on);

    Size CalcPreferredSize(const LayoutContext& context,
                           const Constraints& constraints) const override;

    DrawResult Draw(const DrawContext& context) override;

    void SetOnClicked(std::function<void()> on_clicked);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
