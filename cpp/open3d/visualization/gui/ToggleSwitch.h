// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <functional>

#include "open3d/visualization/gui/Widget.h"

namespace open3d {
namespace visualization {
namespace gui {

class ToggleSwitch : public Widget {
public:
    explicit ToggleSwitch(const char* title);
    ~ToggleSwitch();

    /// Returns the text of the toggle slider.
    const char* GetText() const;
    /// Sets the text of the toggle slider.
    void SetText(const char* text);

    bool GetIsOn() const;
    void SetOn(bool is_on);

    Size CalcPreferredSize(const LayoutContext& context,
                           const Constraints& constraints) const override;

    DrawResult Draw(const DrawContext& context) override;

    /// Sets a function that will be called when the switch is clicked on to
    /// change state. The boolean argument is true if the switch is now on
    /// and false otherwise.
    void SetOnClicked(std::function<void(bool)> on_clicked);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
