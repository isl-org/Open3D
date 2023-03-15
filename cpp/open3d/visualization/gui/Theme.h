// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

#include "open3d/visualization/gui/Color.h"
#include "open3d/visualization/gui/Gui.h"
#include "open3d/visualization/gui/Widget.h"

namespace open3d {
namespace visualization {
namespace gui {

struct Theme {
    Color background_color;

    std::string font_path;
    std::string font_bold_path;
    std::string font_italic_path;
    std::string font_bold_italic_path;
    std::string font_mono_path;
    int font_size;
    int default_margin;
    int default_layout_spacing;
    Color text_color;

    int border_width;
    int border_radius;
    Color border_color;

    Color menubar_border_color;

    Color button_color;
    Color button_hover_color;
    Color button_active_color;
    Color button_on_color;
    Color button_on_hover_color;
    Color button_on_active_color;
    Color button_on_text_color;

    Color checkbox_background_off_color;
    Color checkbox_background_on_color;
    Color checkbox_background_hover_off_color;
    Color checkbox_background_hover_on_color;
    Color checkbox_check_color;

    Color radiobtn_background_off_color;
    Color radiobtn_background_on_color;
    Color radiobtn_background_hover_off_color;
    Color radiobtn_background_hover_on_color;

    Color toggle_background_off_color;
    Color toggle_background_on_color;
    Color toggle_background_hover_off_color;
    Color toggle_background_hover_on_color;
    Color toggle_thumb_color;

    Color combobox_background_color;
    Color combobox_hover_color;
    Color combobox_arrow_background_color;

    Color slider_grab_color;

    Color text_edit_background_color;

    Color list_background_color;
    Color list_hover_color;
    Color list_selected_color;

    Color tree_background_color;
    Color tree_selected_color;

    Color tab_inactive_color;
    Color tab_hover_color;
    Color tab_active_color;

    int dialog_border_width;
    int dialog_border_radius;
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
