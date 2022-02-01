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

#include <map>

#include "open3d/visualization/rendering/Scene.h"

namespace open3d {
namespace visualization {

class GuiSettingsModel {
public:
    static constexpr const char* DEFAULT_IBL = "default";
    static constexpr const char* CUSTOM_IBL = "Custom KTX file...";
    static constexpr const char* DEFAULT_MATERIAL_NAME = "Polished ceramic";
    static constexpr const char* MATERIAL_FROM_FILE_NAME =
            "Material from file [default]";
    static constexpr const char* POINT_CLOUD_PROFILE_NAME =
            "Cloudy day (no direct sun)";

    struct LightingProfile {
        std::string name;
        double ibl_intensity;
        double sun_intensity;
        Eigen::Vector3f sun_dir;
        Eigen::Vector3f sun_color = {1.0f, 1.0f, 1.0f};
        rendering::Scene::Transform ibl_rotation =
                rendering::Scene::Transform::Identity();
        bool ibl_enabled = true;
        bool use_default_ibl = false;
        bool sun_enabled = true;
    };

    enum MaterialType {
        LIT = 0,
        UNLIT,
        NORMAL_MAP,
        DEPTH,
    };

    struct LitMaterial {
        Eigen::Vector3f base_color = {0.9f, 0.9f, 0.9f};
        float metallic = 0.f;
        float roughness = 0.7f;
        float reflectance = 0.5f;
        float clear_coat = 0.2f;
        float clear_coat_roughness = 0.2f;
        float anisotropy = 0.f;
    };

    struct UnlitMaterial {
        // The base color should NOT be {1, 1, 1}, because then the
        // model will be invisible against the default white background.
        Eigen::Vector3f base_color = {0.9f, 0.9f, 0.9f};
    };

    struct Materials {
        LitMaterial lit;
        UnlitMaterial unlit;
        float point_size = 3.0f;
        // 'name' is only used to keep the UI in sync. It is set by
        // Set...Material[s]() and should not be set manually.
        std::string lit_name;
    };

    static const std::vector<LightingProfile> lighting_profiles_;
    static const std::map<std::string, const LitMaterial> prefab_materials_;
    static const LightingProfile& GetDefaultLightingProfile();
    static const LightingProfile& GetDefaultPointCloudLightingProfile();
    static const LitMaterial& GetDefaultLitMaterial();

    GuiSettingsModel();

    bool GetShowSkybox() const;
    void SetShowSkybox(bool show);

    bool GetShowAxes() const;
    void SetShowAxes(bool show);

    bool GetShowGround() const;
    void SetShowGround(bool show);

    bool GetSunFollowsCamera() const;
    void SetSunFollowsCamera(bool follow);

    const Eigen::Vector3f& GetBackgroundColor() const;
    void SetBackgroundColor(const Eigen::Vector3f& color);

    const LightingProfile& GetLighting() const;
    // Should be from lighting_profiles_
    void SetLightingProfile(const LightingProfile& profile);
    void SetCustomLighting(const LightingProfile& profile);

    MaterialType GetMaterialType() const;
    void SetMaterialType(MaterialType type);

    // TODO: Get/SetMaterial
    const Materials& GetCurrentMaterials() const;
    Materials& GetCurrentMaterials();
    void SetLitMaterial(const LitMaterial& material, const std::string& name);
    void SetCurrentMaterials(const Materials& materials,
                             const std::string& name);
    void SetCurrentMaterials(const std::string& name);
    void SetMaterialsToDefault();

    const Eigen::Vector3f& GetCurrentMaterialColor() const;
    void SetCurrentMaterialColor(const Eigen::Vector3f& color);
    void ResetColors();
    void SetCustomDefaultColor(const Eigen::Vector3f color);
    void UnsetCustomDefaultColor();

    int GetPointSize() const;
    void SetPointSize(int size);

    bool GetBasicMode() const;
    void SetBasicMode(bool enable);

    bool GetUserWantsEstimateNormals();
    void EstimateNormalsClicked();

    bool GetDisplayingPointClouds() const;
    /// If true, enables point size
    void SetDisplayingPointClouds(bool displaying);

    bool GetUserHasChangedLightingProfile() const;
    bool GetUserHasCustomizedLighting() const;

    bool GetUserHasChangedColor() const;

    void SetOnChanged(std::function<void(bool)> on_changed);

private:
    Eigen::Vector3f bg_color_ = {1.0f, 1.0f, 1.0f};
    bool show_skybox_ = false;
    bool show_axes_ = false;
    bool show_ground_ = false;
    bool sun_follows_cam_ = false;
    LightingProfile lighting_;
    MaterialType current_type_ = LIT;
    Materials current_materials_;
    Eigen::Vector3f custom_default_color = {-1.0f, -1.0f, 1.0f};
    bool user_has_changed_color_ = false;
    bool user_has_changed_lighting_profile_ = false;
    bool user_has_customized_lighting_ = false;
    bool displaying_point_clouds_ = false;
    bool user_wants_estimate_normals_ = false;
    bool basic_mode_enabled_ = false;

    std::function<void(bool)> on_changed_;

    void NotifyChanged(bool material_changed = false);
};

}  // namespace visualization
}  // namespace open3d
