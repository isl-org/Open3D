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

class TabControl : public Widget {
    using Super = Widget;

public:
    TabControl();
    ~TabControl();

    void AddTab(const char* name, std::shared_ptr<Widget> panel);

    Size CalcPreferredSize(const LayoutContext& context,
                           const Constraints& constraints) const override;
    void Layout(const LayoutContext& context) override;

    DrawResult Draw(const DrawContext& context) override;

    void SetOnSelectedTabChanged(std::function<void(int)> on_changed);
    void SetSelectedTabIndex(int index);
    int GetSelectedTabIndex();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
