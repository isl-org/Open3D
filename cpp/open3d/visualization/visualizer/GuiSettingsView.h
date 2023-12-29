// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <functional>
#include <memory>
#include <string>

#include "open3d/visualization/gui/Layout.h"

namespace open3d {

namespace visualization {

namespace gui {
class Button;
class Checkbox;
class Combobox;
class ColorEdit;
class CollapsableVert;
class Slider;
class VectorEdit;
}  // namespace gui

class GuiSettingsModel;

class GuiSettingsView : public gui::Vert {
public:
    GuiSettingsView(GuiSettingsModel& model,
                    const gui::Theme& theme,
                    const std::string& resource_path,
                    std::function<void(const char*)> on_load_ibl);

    void ShowFileMaterialEntry(bool show);
    void EnableEstimateNormals(bool enable);
    void Update();

private:
    GuiSettingsModel& model_;
    std::function<void(const char*)> on_load_ibl_;

    std::shared_ptr<gui::Combobox> lighting_profile_;
    std::shared_ptr<gui::Checkbox> show_axes_;
    std::shared_ptr<gui::Checkbox> show_ground_;
    std::shared_ptr<gui::ColorEdit> bg_color_;
    std::shared_ptr<gui::Checkbox> show_skybox_;

    std::shared_ptr<gui::CollapsableVert> advanced_;
    std::shared_ptr<gui::Checkbox> ibl_enabled_;
    std::shared_ptr<gui::Checkbox> sun_enabled_;
    std::shared_ptr<gui::Combobox> ibls_;
    std::shared_ptr<gui::Slider> ibl_intensity_;
    std::shared_ptr<gui::Slider> sun_intensity_;
    std::shared_ptr<gui::VectorEdit> sun_dir_;
    std::shared_ptr<gui::Checkbox> sun_follows_camera_;
    std::shared_ptr<gui::ColorEdit> sun_color_;

    std::shared_ptr<gui::Combobox> material_type_;
    std::shared_ptr<gui::Combobox> prefab_material_;
    std::shared_ptr<gui::ColorEdit> material_color_;
    std::shared_ptr<gui::Button> reset_material_color_;
    std::shared_ptr<gui::Slider> point_size_;
    std::shared_ptr<gui::Button> generate_normals_;
    std::shared_ptr<gui::Checkbox> basic_mode_;
    std::shared_ptr<gui::Checkbox> wireframe_mode_;

    bool sun_follows_cam_was_on_ = true;
    void UpdateUIForBasicMode(bool enable);
};

}  // namespace visualization
}  // namespace open3d
