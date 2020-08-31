// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

    void SetOnValueChanged(std::function<void(double)> on_changed);

    Size CalcPreferredSize(const Theme& theme) const override;
    Widget::DrawResult Draw(const DrawContext& context) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
