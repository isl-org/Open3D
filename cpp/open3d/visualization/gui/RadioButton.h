// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <functional>
#include <string>

#include "open3d/visualization/gui/Widget.h"

namespace open3d {
namespace visualization {
namespace gui {

class RadioButton : public Widget {
public:
    /// VERT radio buttons will be layout vertically, each item takes a line.
    /// HORIZ radio buttons will be layout horizontally, all items will
    /// be in the same line.
    enum Type { VERT, HORIZ };

    explicit RadioButton(Type type);
    ~RadioButton() override;

    void SetItems(const std::vector<std::string>& items);
    int GetSelectedIndex() const;
    const char* GetSelectedValue() const;
    void SetSelectedIndex(int index);

    /// callback to be invoked while selected index is changed.
    /// A SetSelectedIndex will not trigger this callback.
    void SetOnSelectionChanged(std::function<void(int)> callback);

    Size CalcPreferredSize(const LayoutContext& context,
                           const Constraints& constraints) const override;

    DrawResult Draw(const DrawContext& context) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
