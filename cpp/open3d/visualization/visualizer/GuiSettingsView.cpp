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

#include "open3d/visualization/visualizer/GuiSettingsView.h"

#include <cmath>

#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"
#include "open3d/visualization/gui/Checkbox.h"
#include "open3d/visualization/gui/ColorEdit.h"
#include "open3d/visualization/gui/Combobox.h"
#include "open3d/visualization/gui/Label.h"
#include "open3d/visualization/gui/Slider.h"
#include "open3d/visualization/gui/Theme.h"
#include "open3d/visualization/gui/VectorEdit.h"
#include "open3d/visualization/visualizer/GuiSettingsModel.h"
#include "open3d/visualization/visualizer/GuiWidgets.h"

namespace open3d {
namespace visualization {

static const char *CUSTOM_LIGHTING = "Custom";

std::shared_ptr<gui::Slider> MakeSlider(const gui::Slider::Type type,
                                        const double min,
                                        const double max,
                                        const double value) {
    auto slider = std::make_shared<gui::Slider>(type);
    slider->SetLimits(min, max);
    slider->SetValue(value);
    return slider;
}

GuiSettingsView::GuiSettingsView(GuiSettingsModel &model,
                                 const gui::Theme &theme,
                                 const std::string &resource_path,
                                 std::function<void(const char *)> on_load_ibl)
    : model_(model), on_load_ibl_(on_load_ibl) {
    const auto em = theme.font_size;
    const int lm = int(std::ceil(0.5 * em));
    const int grid_spacing = int(std::ceil(0.25 * em));

    const int separation_height = int(std::ceil(0.75 * em));
    // (we don't want as much left margin because the twisty arrow is the
    // only thing there, and visually it looks larger than the right.)
    const gui::Margins base_margins(int(std::ceil(0.5 * lm)), lm, lm, lm);
    SetMargins(base_margins);

    gui::Margins indent(em, 0, 0, 0);
    auto view_ctrls =
            std::make_shared<gui::CollapsableVert>("Scene controls", 0, indent);

    // Background
    show_skybox_ = std::make_shared<gui::Checkbox>("Show skymap");
    show_skybox_->SetOnChecked(
            [this](bool checked) { model_.SetShowSkybox(checked); });

    bg_color_ = std::make_shared<gui::ColorEdit>();
    bg_color_->SetOnValueChanged([this](const gui::Color &newColor) {
        model_.SetBackgroundColor(
                {newColor.GetRed(), newColor.GetGreen(), newColor.GetBlue()});
    });

    auto bg_layout = std::make_shared<gui::VGrid>(2, grid_spacing);
    bg_layout->AddChild(std::make_shared<gui::Label>("BG Color"));
    bg_layout->AddChild(bg_color_);

    view_ctrls->AddChild(show_skybox_);
    view_ctrls->AddFixed(int(std::ceil(0.25 * em)));
    view_ctrls->AddChild(bg_layout);

    // Show axes
    show_axes_ = std::make_shared<gui::Checkbox>("Show axes");
    show_axes_->SetOnChecked(
            [this](bool is_checked) { model_.SetShowAxes(is_checked); });
    view_ctrls->AddFixed(separation_height);
    view_ctrls->AddChild(show_axes_);

    // Show ground plane
    show_ground_ = std::make_shared<gui::Checkbox>("Show ground");
    show_ground_->SetOnChecked(
            [this](bool is_checked) { model_.SetShowGround(is_checked); });
    view_ctrls->AddFixed(separation_height);
    view_ctrls->AddChild(show_ground_);

    // Lighting profiles
    lighting_profile_ = std::make_shared<gui::Combobox>();
    for (auto &lp : GuiSettingsModel::lighting_profiles_) {
        lighting_profile_->AddItem(lp.name.c_str());
    }
    lighting_profile_->AddItem(CUSTOM_LIGHTING);
    lighting_profile_->SetOnValueChanged([this](const char *, int index) {
        if (index < int(GuiSettingsModel::lighting_profiles_.size())) {
            sun_dir_->SetEnabled(true);
            model_.SetSunFollowsCamera(false);
            model_.SetLightingProfile(
                    GuiSettingsModel::lighting_profiles_[index]);
            if (GuiSettingsModel::lighting_profiles_[index].use_default_ibl) {
                ibls_->SetSelectedValue(GuiSettingsModel::DEFAULT_IBL);
            }
        }
    });

    auto profile_layout = std::make_shared<gui::Vert>();
    profile_layout->AddChild(std::make_shared<gui::Label>("Lighting profiles"));
    profile_layout->AddChild(lighting_profile_);
    view_ctrls->AddFixed(separation_height);
    view_ctrls->AddChild(profile_layout);

    AddChild(view_ctrls);
    AddFixed(separation_height);

    // Advanced lighting
    advanced_ = std::make_shared<gui::CollapsableVert>("Advanced lighting", 0,
                                                       indent);
    advanced_->SetIsOpen(false);
    AddChild(advanced_);

    // ... lighting on/off
    advanced_->AddChild(std::make_shared<gui::Label>("Light sources"));
    auto checkboxes = std::make_shared<gui::Horiz>();
    ibl_enabled_ = std::make_shared<gui::Checkbox>("HDR map");
    ibl_enabled_->SetOnChecked([this](bool checked) {
        auto lighting = model_.GetLighting();  // copy
        lighting.ibl_enabled = checked;
        model_.SetCustomLighting(lighting);
    });
    checkboxes->AddChild(ibl_enabled_);

    sun_enabled_ = std::make_shared<gui::Checkbox>("Sun");
    sun_enabled_->SetOnChecked([this](bool checked) {
        auto lighting = model_.GetLighting();  // copy
        lighting.sun_enabled = checked;
        model_.SetCustomLighting(lighting);
    });
    checkboxes->AddChild(sun_enabled_);
    advanced_->AddChild(checkboxes);

    advanced_->AddFixed(separation_height);

    // ... IBL
    ibls_ = std::make_shared<gui::Combobox>();
    std::vector<std::string> resource_files;
    utility::filesystem::ListFilesInDirectory(resource_path, resource_files);
    std::sort(resource_files.begin(), resource_files.end());
    int n = 0;
    for (auto &f : resource_files) {
        if (f.find("_ibl.ktx") == f.size() - 8) {
            auto name = utility::filesystem::GetFileNameWithoutDirectory(f);
            name = name.substr(0, name.size() - 8);
            ibls_->AddItem(name.c_str());
            if (name == GuiSettingsModel::DEFAULT_IBL) {
                ibls_->SetSelectedIndex(n);
            }
            n++;
        }
    }
    ibls_->AddItem(GuiSettingsModel::CUSTOM_IBL);
    ibls_->SetOnValueChanged(
            [this](const char *name, int) { on_load_ibl_(name); });

    ibl_intensity_ = MakeSlider(gui::Slider::INT, 0.0, 150000.0,
                                /*lighting_profile.ibl_intensity*/ 45000);
    ibl_intensity_->SetOnValueChanged([this](double new_value) {
        auto lighting = model_.GetLighting();  // copy
        lighting.ibl_intensity = new_value;
        model_.SetCustomLighting(lighting);
    });

    auto ambient_layout = std::make_shared<gui::VGrid>(2, grid_spacing);
    ambient_layout->AddChild(std::make_shared<gui::Label>("HDR map"));
    ambient_layout->AddChild(ibls_);
    ambient_layout->AddChild(std::make_shared<gui::Label>("Intensity"));
    ambient_layout->AddChild(ibl_intensity_);

    advanced_->AddChild(std::make_shared<gui::Label>("Environment"));
    advanced_->AddChild(ambient_layout);
    advanced_->AddFixed(separation_height);

    // ... directional light (sun)
    sun_intensity_ = MakeSlider(gui::Slider::INT, 0.0, 500000.0,
                                /*lighting_profile.sun_intensity*/ 45000);
    sun_intensity_->SetOnValueChanged([this](double new_value) {
        auto lighting = model_.GetLighting();  // copy
        lighting.sun_intensity = new_value;
        model_.SetCustomLighting(lighting);
    });
    sun_dir_ = std::make_shared<gui::VectorEdit>();
    sun_dir_->SetOnValueChanged([this](const Eigen::Vector3f &dir) {
        auto lighting = model_.GetLighting();  // copy
        lighting.sun_dir = dir.normalized();
        model_.SetCustomLighting(lighting);
    });

    sun_follows_camera_ = std::make_shared<gui::Checkbox>(" ");
    sun_follows_camera_->SetChecked(true);
    sun_follows_camera_->SetOnChecked([this](bool checked) {
        sun_dir_->SetEnabled(!checked);
        model_.SetSunFollowsCamera(checked);
    });

    sun_color_ = std::make_shared<gui::ColorEdit>();
    sun_color_->SetOnValueChanged([this](const gui::Color &new_color) {
        auto lighting = model_.GetLighting();  // copy
        lighting.sun_color = {new_color.GetRed(), new_color.GetGreen(),
                              new_color.GetBlue()};
        model_.SetCustomLighting(lighting);
    });

    auto sun_layout = std::make_shared<gui::VGrid>(2, grid_spacing);
    sun_layout->AddChild(std::make_shared<gui::Label>("Intensity"));
    sun_layout->AddChild(sun_intensity_);
    sun_layout->AddChild(std::make_shared<gui::Label>("Direction"));
    sun_layout->AddChild(sun_dir_);
    sun_layout->AddChild(sun_follows_camera_);
    sun_layout->AddChild(std::make_shared<gui::Label>("Sun Follows Camera"));
    sun_layout->AddChild(std::make_shared<gui::Label>("Color"));
    sun_layout->AddChild(sun_color_);

    advanced_->AddChild(
            std::make_shared<gui::Label>("Sun (Directional light)"));
    advanced_->AddChild(sun_layout);

    // Materials
    auto materials = std::make_shared<gui::CollapsableVert>("Material settings",
                                                            0, indent);

    auto mat_grid = std::make_shared<gui::VGrid>(2, grid_spacing);
    mat_grid->AddChild(std::make_shared<gui::Label>("Type"));
    // If edit order of items, change Update()
    material_type_.reset(
            new gui::Combobox({"Lit", "Unlit", "Normal map", "Depth"}));
    material_type_->SetOnValueChanged([this](const char *, int selected_idx) {
        switch (selected_idx) {
            default:  // fall through
            case 0:
                model_.SetMaterialType(GuiSettingsModel::MaterialType::LIT);
                break;
            case 1:
                model_.SetMaterialType(GuiSettingsModel::MaterialType::UNLIT);
                break;
            case 2:
                model_.SetMaterialType(
                        GuiSettingsModel::MaterialType::NORMAL_MAP);
                break;
            case 3:
                model_.SetMaterialType(GuiSettingsModel::MaterialType::DEPTH);
                break;
        }
    });
    mat_grid->AddChild(material_type_);

    prefab_material_ = std::make_shared<gui::Combobox>();
    for (auto &prefab : GuiSettingsModel::prefab_materials_) {
        prefab_material_->AddItem(prefab.first.c_str());
    }
    prefab_material_->SetOnValueChanged([this](const char *name, int) {
        auto mat = GuiSettingsModel::prefab_materials_.find(name);
        if (mat != GuiSettingsModel::prefab_materials_.end()) {
            model_.SetLitMaterial(mat->second, name);
        } else {
            // If it's not a preset it must be the material from the file.
            // Set the name so that GuiVisualizer::Impl can pick it up.
            model_.SetLitMaterial(model_.GetCurrentMaterials().lit, name);
        }
    });
    mat_grid->AddChild(std::make_shared<gui::Label>("Material"));
    mat_grid->AddChild(prefab_material_);

    material_color_ = std::make_shared<gui::ColorEdit>();
    material_color_->SetOnValueChanged([this](const gui::Color &color) {
        model_.SetCurrentMaterialColor(
                {color.GetRed(), color.GetGreen(), color.GetBlue()});
    });
    reset_material_color_ = std::make_shared<SmallButton>("Reset");
    reset_material_color_->SetOnClicked([this]() { model_.ResetColors(); });

    mat_grid->AddChild(std::make_shared<gui::Label>("Color"));
    auto color_layout = std::make_shared<gui::Horiz>();
    color_layout->AddChild(material_color_);
    color_layout->AddFixed(int(std::ceil(0.25 * em)));
    color_layout->AddChild(reset_material_color_);
    mat_grid->AddChild(color_layout);

    mat_grid->AddChild(std::make_shared<gui::Label>("Point size"));
    point_size_ = MakeSlider(gui::Slider::INT, 1.0, 10.0, 3);
    point_size_->SetOnValueChanged([this](double value) {
        model_.SetPointSize(int(std::round(value)));
    });
    mat_grid->AddChild(point_size_);

    mat_grid->AddChild(std::make_shared<gui::Label>(""));
    generate_normals_ = std::make_shared<SmallButton>("Estimate PCD Normals");
    generate_normals_->SetOnClicked(
            [this]() { model_.EstimateNormalsClicked(); });
    generate_normals_->SetEnabled(false);
    mat_grid->AddChild(generate_normals_);
    mat_grid->AddChild(std::make_shared<gui::Label>("Raw Mode"));
    basic_mode_ = std::make_shared<gui::Checkbox>("");
    basic_mode_->SetOnChecked([this](bool checked) {
        UpdateUIForBasicMode(checked);
        model_.SetBasicMode(checked);
    });
    mat_grid->AddChild(basic_mode_);

    materials->AddChild(mat_grid);

    AddFixed(separation_height);
    AddChild(materials);

    Update();
}

void GuiSettingsView::ShowFileMaterialEntry(bool show) {
    if (show) {
        prefab_material_->AddItem(GuiSettingsModel::MATERIAL_FROM_FILE_NAME);
        prefab_material_->ChangeItem(
                (std::string(GuiSettingsModel::DEFAULT_MATERIAL_NAME) +
                 " [default]")
                        .c_str(),
                GuiSettingsModel::DEFAULT_MATERIAL_NAME);
    } else {
        prefab_material_->RemoveItem(GuiSettingsModel::MATERIAL_FROM_FILE_NAME);
        prefab_material_->ChangeItem(
                GuiSettingsModel::DEFAULT_MATERIAL_NAME,
                (std::string(GuiSettingsModel::DEFAULT_MATERIAL_NAME) +
                 " [default]")
                        .c_str());
    }
}

void GuiSettingsView::EnableEstimateNormals(bool enable) {
    generate_normals_->SetEnabled(enable);
}

void GuiSettingsView::Update() {
    show_skybox_->SetChecked(model_.GetShowSkybox());
    show_axes_->SetChecked(model_.GetShowAxes());
    bg_color_->SetValue({model_.GetBackgroundColor().x(),
                         model_.GetBackgroundColor().y(),
                         model_.GetBackgroundColor().z()});
    auto &lighting = model_.GetLighting();
    if (model_.GetUserHasCustomizedLighting()) {
        lighting_profile_->SetSelectedValue(CUSTOM_LIGHTING);
    } else {
        if (!lighting_profile_->SetSelectedValue(lighting.name.c_str())) {
            utility::LogWarning(
                    "Internal Error: lighting profile '{}' is not in combobox",
                    lighting.name.c_str());
            lighting_profile_->SetSelectedValue(CUSTOM_LIGHTING);
        }
    }
    ibl_enabled_->SetChecked(lighting.ibl_enabled);
    sun_enabled_->SetChecked(lighting.sun_enabled);
    ibl_intensity_->SetValue(lighting.ibl_intensity);
    sun_intensity_->SetValue(lighting.sun_intensity);
    sun_dir_->SetValue(lighting.sun_dir);
    sun_color_->SetValue({lighting.sun_color.x(), lighting.sun_color.y(),
                          lighting.sun_color.z()});
    auto &materials = model_.GetCurrentMaterials();
    if (!prefab_material_->SetSelectedValue(materials.lit_name.c_str())) {
        if (materials.lit_name.find(GuiSettingsModel::DEFAULT_MATERIAL_NAME) ==
            0) {
            // if we didn't find the default material, it must be appended
            // " [default]".
            for (int i = 0; i < prefab_material_->GetNumberOfItems(); ++i) {
                if (materials.lit_name.find(prefab_material_->GetItem(i)) ==
                    0) {
                    prefab_material_->SetSelectedIndex(i);
                    break;
                }
            }
        } else {
            utility::LogWarning("Unknown prefab material '{}'",
                                materials.lit_name);
            prefab_material_->SetSelectedValue(
                    GuiSettingsModel::DEFAULT_MATERIAL_NAME);
        }
    }
    switch (model_.GetMaterialType()) {
        case GuiSettingsModel::MaterialType::LIT:
            material_type_->SetSelectedIndex(0);
            prefab_material_->SetEnabled(true);
            material_color_->SetEnabled(true);
            material_color_->SetValue({materials.lit.base_color.x(),
                                       materials.lit.base_color.y(),
                                       materials.lit.base_color.z()});
            point_size_->SetValue(materials.point_size);
            break;
        case GuiSettingsModel::MaterialType::UNLIT:
            material_type_->SetSelectedIndex(1);
            prefab_material_->SetEnabled(false);
            material_color_->SetEnabled(true);
            material_color_->SetValue({materials.unlit.base_color.x(),
                                       materials.unlit.base_color.y(),
                                       materials.unlit.base_color.z()});
            point_size_->SetValue(materials.point_size);
            break;
        case GuiSettingsModel::MaterialType::NORMAL_MAP:
            material_type_->SetSelectedIndex(2);
            prefab_material_->SetEnabled(false);
            material_color_->SetEnabled(false);
            material_color_->SetValue({1.0f, 1.0f, 1.0f});
            break;
        case GuiSettingsModel::MaterialType::DEPTH:
            material_type_->SetSelectedIndex(3);
            prefab_material_->SetEnabled(false);
            material_color_->SetEnabled(false);
            material_color_->SetValue({1.0f, 1.0f, 1.0f});
            break;
    }
    reset_material_color_->SetEnabled(
            model_.GetUserHasChangedColor() &&
            (model_.GetMaterialType() == GuiSettingsModel::MaterialType::LIT ||
             model_.GetMaterialType() ==
                     GuiSettingsModel::MaterialType::UNLIT));
    point_size_->SetEnabled(model_.GetDisplayingPointClouds());
}

void GuiSettingsView::UpdateUIForBasicMode(bool enable) {
    // Enable/disable UI elements
    show_skybox_->SetEnabled(!enable);
    lighting_profile_->SetEnabled(!enable);
    ibls_->SetEnabled(!enable);
    ibl_enabled_->SetEnabled(!enable);
    ibl_intensity_->SetEnabled(!enable);
    sun_enabled_->SetEnabled(!enable);
    sun_dir_->SetEnabled(!enable);
    sun_color_->SetEnabled(!enable);
    sun_follows_camera_->SetEnabled(!enable);
    material_color_->SetEnabled(!enable);
    prefab_material_->SetEnabled(!enable);

    // Set lighting environment for basic/non-basic mode
    auto lighting = model_.GetLighting();  // copy
    if (enable) {
        sun_follows_cam_was_on_ = sun_follows_camera_->IsChecked();
        lighting.ibl_enabled = !enable;
        lighting.sun_enabled = enable;
        lighting.sun_intensity = 160000.f;
        sun_enabled_->SetChecked(true);
        ibl_enabled_->SetChecked(false);
        sun_intensity_->SetValue(160000.0);
        model_.SetCustomLighting(lighting);
        model_.SetSunFollowsCamera(true);
        sun_follows_camera_->SetChecked(true);
    } else {
        model_.SetLightingProfile(GuiSettingsModel::lighting_profiles_[0]);
        if (!sun_follows_cam_was_on_) {
            sun_follows_camera_->SetChecked(false);
            model_.SetSunFollowsCamera(false);
        }
    }
}

}  // namespace visualization
}  // namespace open3d
