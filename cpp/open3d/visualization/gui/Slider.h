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

class Slider : public Widget {
public:
    enum Type { INT, DOUBLE };
    /// The only difference between INT and DOUBLE is that INT increments by
    /// 1.0 and coerces value to whole numbers.
    explicit Slider(Type type);
    ~Slider();

    /// Returns the value of the control as an integer
    int GetIntValue() const;
    /// Returns the value of the control as a double.
    double GetDoubleValue() const;
    /// Sets the value of the control. Will not call onValueChanged, but the
    /// value will be clamped to [min, max].
    void SetValue(double val);

    double GetMinimumValue() const;
    double GetMaximumValue() const;
    /// Sets the bounds for valid values of the widget. Values will be clamped
    /// to be within [minValue, maxValue].
    void SetLimits(double min_value, double max_value);

    Size CalcPreferredSize(const LayoutContext& theme,
                           const Constraints& constraints) const override;

    DrawResult Draw(const DrawContext& context) override;

    /// Sets a function to call when the value changes because of user action.
    void SetOnValueChanged(std::function<void(double)> on_value_changed);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
