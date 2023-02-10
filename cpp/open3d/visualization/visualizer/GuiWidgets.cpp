// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/visualizer/GuiWidgets.h"

#include "open3d/visualization/gui/Theme.h"

namespace open3d {
namespace visualization {

SmallButton::SmallButton(const char *title) : gui::Button(title) {}

gui::Size SmallButton::CalcPreferredSize(const gui::LayoutContext &context,
                                         const Constraints &constraints) const {
    auto em = context.theme.font_size;
    auto size = Super::CalcPreferredSize(context, constraints);
    return gui::Size(size.width - em, int(std::round(1.2 * em)));
}

SmallToggleButton::SmallToggleButton(const char *title) : SmallButton(title) {
    SetToggleable(true);
}

}  // namespace visualization
}  // namespace open3d
