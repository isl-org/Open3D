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

class NumberEdit : public Widget {
    using Super = Widget;

public:
    enum Type { INT, DOUBLE };
    explicit NumberEdit(Type type);
    ~NumberEdit();

    int GetIntValue() const;
    double GetDoubleValue() const;
    void SetValue(double val);

    double GetMinimumValue() const;
    double GetMaximumValue() const;
    void SetLimits(double min_value, double max_value);

    int GetDecimalPrecision();
    void SetDecimalPrecision(int num_digits);

    void SetPreferredWidth(int width);

    void SetOnValueChanged(std::function<void(double)> on_changed);

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
