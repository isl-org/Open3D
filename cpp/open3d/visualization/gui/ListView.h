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

#include <functional>
#include <string>

#include "open3d/visualization/gui/Widget.h"

namespace open3d {
namespace visualization {
namespace gui {

class ListView : public Widget {
    using Super = Widget;

public:
    ListView();
    virtual ~ListView();

    void SetItems(const std::vector<std::string>& items);

    /// Returns the currently selected item in the list.
    int GetSelectedIndex() const;
    /// Returns the value of the currently selected item in the list.
    const char* GetSelectedValue() const;
    /// Selects the indicated row of the list. Does not call onValueChanged.
    void SetSelectedIndex(int index);
    /// Limit the max visible items shown to user.
    /// Set to negative number will make list extends vertically as much
    /// as possible, otherwise the list will at least show 3 items and
    /// at most show \ref num items.
    void SetMaxVisibleItems(int num);

    Size CalcPreferredSize(const LayoutContext& context,
                           const Constraints& constraints) const override;

    Size CalcMinimumSize(const LayoutContext& context) const override;

    DrawResult Draw(const DrawContext& context) override;

    /// Calls onValueChanged(const char *selectedText, bool isDoubleClick)
    /// when the list selection changes because of user action.
    void SetOnValueChanged(
            std::function<void(const char*, bool)> on_value_changed);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
