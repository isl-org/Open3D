// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Geometry>

#include "open3d/visualization/gui/Widget.h"

namespace open3d {
namespace visualization {
namespace gui {

class VectorEdit : public Widget {
    using Super = Widget;

public:
    VectorEdit();
    ~VectorEdit();

    Eigen::Vector3f GetValue() const;
    /// Sets the value of the widget. Does not call onValueChanged.
    void SetValue(const Eigen::Vector3f& val);

    /// Sets the function that is called whenever the value in the widget
    /// changes because of user behavior
    void SetOnValueChanged(
            std::function<void(const Eigen::Vector3f&)> on_changed);

    Size CalcPreferredSize(const LayoutContext& context,
                           const Constraints& constraints) const override;
    Widget::DrawResult Draw(const DrawContext& context) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
