// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/visualization/gui/Widget.h"

namespace open3d {
namespace visualization {
namespace gui {

/// Stacks its children on top of each other, with only the selected child
/// showing. It is like a tab control without the tabs.
class StackedWidget : public Widget {
    using Super = Widget;

public:
    StackedWidget();
    virtual ~StackedWidget();

    /// Sets the index of the child to draw.
    void SetSelectedIndex(int index);
    /// Returns the index of the selected child.
    int GetSelectedIndex() const;

    Size CalcPreferredSize(const LayoutContext& context,
                           const Constraints& constraints) const override;

    void Layout(const LayoutContext& context) override;

    Widget::DrawResult Draw(const DrawContext& context) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
