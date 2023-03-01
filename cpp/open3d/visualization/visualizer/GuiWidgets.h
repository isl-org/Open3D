// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/visualization/gui/Button.h"

namespace open3d {
namespace visualization {

class SmallButton : public gui::Button {
    using Super = Button;

public:
    explicit SmallButton(const char *title);

    gui::Size CalcPreferredSize(const gui::LayoutContext &context,
                                const Constraints &constraints) const override;
};

class SmallToggleButton : public SmallButton {
    using Super = SmallButton;

public:
    explicit SmallToggleButton(const char *title);
};

}  // namespace visualization
}  // namespace open3d
