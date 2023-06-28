// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
// These are internal helper functions

#include <imgui.h>

#include <cstdint>
#include <string>
#include <vector>

#include "open3d/visualization/gui/Gui.h"

namespace open3d {
namespace visualization {
namespace gui {

class Color;

// These functions are here, because ImVec4 requires imgui.h, and can't be
// forward-declared because we need to know the size, since it is a return
// value. Since imgui.h is an implementation detail, we can't put this function
// in Color or it would pull in imgui.h pretty much everywhere that gui is used.
ImVec4 colorToImgui(const Color& color);
uint32_t colorToImguiRGBA(const Color& color);

std::string FindFontPath(std::string font, FontStyle style);

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
