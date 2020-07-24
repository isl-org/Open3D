// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include "open3d/visualization/visualizer/GuiVisualizer.h"

#include "open3d/Open3DConfig.h"
#include "open3d/geometry/BoundingVolume.h"
#include "open3d/geometry/Image.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/io/FileFormatIO.h"
#include "open3d/io/ImageIO.h"
#include "open3d/io/PointCloudIO.h"
#include "open3d/io/TriangleMeshIO.h"
#include "open3d/utility/Console.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/visualization/gui/Application.h"
#include "open3d/visualization/gui/Button.h"
#include "open3d/visualization/gui/Checkbox.h"
#include "open3d/visualization/gui/Color.h"
#include "open3d/visualization/gui/ColorEdit.h"
#include "open3d/visualization/gui/Combobox.h"
#include "open3d/visualization/gui/Dialog.h"
#include "open3d/visualization/gui/FileDialog.h"
#include "open3d/visualization/gui/Label.h"
#include "open3d/visualization/gui/Layout.h"
#include "open3d/visualization/gui/ProgressBar.h"
#include "open3d/visualization/gui/SceneWidget.h"
#include "open3d/visualization/gui/Slider.h"
#include "open3d/visualization/gui/Theme.h"
#include "open3d/visualization/gui/VectorEdit.h"
#include "open3d/visualization/rendering/Camera.h"
#include "open3d/visualization/rendering/RenderToBuffer.h"
#include "open3d/visualization/rendering/RendererStructs.h"
#include "open3d/visualization/rendering/Scene.h"
#include "open3d/visualization/rendering/filament/FilamentResourceManager.h"

#define LOAD_IN_NEW_WINDOW 0

namespace open3d {
namespace visualization {

namespace {

std::shared_ptr<gui::Dialog> CreateAboutDialog(gui::Window *window) {
    auto &theme = window->GetTheme();
    auto dlg = std::make_shared<gui::Dialog>("About");

    auto title = std::make_shared<gui::Label>(
            (std::string("Open3D ") + OPEN3D_VERSION).c_str());
    auto text = std::make_shared<gui::Label>(
            "The MIT License (MIT)\n"
            "Copyright (c) 2018 - 2020 www.open3d.org\n\n"

            "Permission is hereby granted, free of charge, to any person "
            "obtaining "
            "a copy of this software and associated documentation files (the "
            "\"Software\"), to deal in the Software without restriction, "
            "including "
            "without limitation the rights to use, copy, modify, merge, "
            "publish, "
            "distribute, sublicense, and/or sell copies of the Software, and "
            "to "
            "permit persons to whom the Software is furnished to do so, "
            "subject to "
            "the following conditions:\n\n"

            "The above copyright notice and this permission notice shall be "
            "included in all copies or substantial portions of the "
            "Software.\n\n"

            "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, "
            "EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES "
            "OF "
            "MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND "
            "NONINFRINGEMENT. "
            "IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR "
            "ANY "
            "CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF "
            "CONTRACT, "
            "TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE "
            "SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.");
    auto ok = std::make_shared<gui::Button>("OK");
    ok->SetOnClicked([window]() { window->CloseDialog(); });

    gui::Margins margins(theme.font_size);
    auto layout = std::make_shared<gui::Vert>(0, margins);
    layout->AddChild(gui::Horiz::MakeCentered(title));
    layout->AddFixed(theme.font_size);
    layout->AddChild(text);
    layout->AddFixed(theme.font_size);
    layout->AddChild(gui::Horiz::MakeCentered(ok));
    dlg->AddChild(layout);

    return dlg;
}

std::shared_ptr<gui::VGrid> CreateHelpDisplay(gui::Window *window) {
    auto &theme = window->GetTheme();

    gui::Margins margins(theme.font_size);
    auto layout = std::make_shared<gui::VGrid>(2, 0, margins);
    layout->SetBackgroundColor(gui::Color(0, 0, 0, 0.5));

    auto AddLabel = [layout](const char *text) {
        auto label = std::make_shared<gui::Label>(text);
        label->SetTextColor(gui::Color(1, 1, 1));
        layout->AddChild(label);
    };
    auto AddRow = [layout, &AddLabel](const char *left, const char *right) {
        AddLabel(left);
        AddLabel(right);
    };

    AddRow("Arcball mode", " ");
    AddRow("Left-drag", "Rotate camera");
    AddRow("Shift + left-drag", "Forward/backward");

#if defined(__APPLE__)
    AddLabel("Cmd + left-drag");
#else
    AddLabel("Ctrl + left-drag");
#endif  // __APPLE__
    AddLabel("Pan camera");

#if defined(__APPLE__)
    AddLabel("Opt + left-drag (up/down)  ");
#else
    AddLabel("Win + left-drag (up/down)  ");
#endif  // __APPLE__
    AddLabel("Rotate around forward axis");

    // GNOME3 uses Win/Meta as a shortcut to move windows around, so we
    // need another way to rotate around the forward axis.
    AddLabel("Ctrl + Shift + left-drag");
    AddLabel("Rotate around forward axis");

#if defined(__APPLE__)
    AddLabel("Ctrl + left-drag");
#else
    AddLabel("Alt + left-drag");
#endif  // __APPLE__
    AddLabel("Rotate directional light");

    AddRow("Right-drag", "Pan camera");
    AddRow("Middle-drag", "Rotate directional light");
    AddRow("Wheel", "Forward/backward");
    AddRow("Shift + Wheel", "Change field of view");
    AddRow("", "");

    AddRow("Fly mode", " ");
    AddRow("Left-drag", "Rotate camera");
#if defined(__APPLE__)
    AddLabel("Opt + left-drag");
#else
    AddLabel("Win + left-drag");
#endif  // __APPLE__
    AddLabel("Rotate around forward axis");
    AddRow("W", "Forward");
    AddRow("S", "Backward");
    AddRow("A", "Step left");
    AddRow("D", "Step right");
    AddRow("Q", "Step up");
    AddRow("Z", "Step down");
    AddRow("E", "Roll left");
    AddRow("R", "Roll right");
    AddRow("Up", "Look up");
    AddRow("Down", "Look down");
    AddRow("Left", "Look left");
    AddRow("Right", "Look right");

    return layout;
}

std::shared_ptr<gui::VGrid> CreateCameraDisplay(gui::Window *window) {
    auto &theme = window->GetTheme();

    gui::Margins margins(theme.font_size);
    auto layout = std::make_shared<gui::VGrid>(2, 0, margins);
    layout->SetBackgroundColor(gui::Color(0, 0, 0, 0.5));

    auto AddLabel = [layout](const char *text) {
        auto label = std::make_shared<gui::Label>(text);
        label->SetTextColor(gui::Color(1, 1, 1));
        layout->AddChild(label);
    };
    auto AddRow = [layout, &AddLabel](const char *left, const char *right) {
        AddLabel(left);
        AddLabel(right);
    };

    AddRow("Position:", "[0 0 0]");
    AddRow("Forward:", "[0 0 0]");
    AddRow("Left:", "[0 0 0]");
    AddRow("Up:", "[0 0 0]");

    return layout;
}

std::shared_ptr<gui::Dialog> CreateContactDialog(gui::Window *window) {
    auto &theme = window->GetTheme();
    auto em = theme.font_size;
    auto dlg = std::make_shared<gui::Dialog>("Contact Us");

    auto title = std::make_shared<gui::Label>("Contact Us");
    auto left_col = std::make_shared<gui::Label>(
            "Web site:\n"
            "Code:\n"
            "Mailing list:\n"
            "Discord channel:");
    auto right_col = std::make_shared<gui::Label>(
            "http://www.open3d.org\n"
            "http://github.org/intel-isl/Open3D\n"
            "http://www.open3d.org/index.php/subscribe/\n"
            "https://discord.gg/D35BGvn");
    auto ok = std::make_shared<gui::Button>("OK");
    ok->SetOnClicked([window]() { window->CloseDialog(); });

    gui::Margins margins(em);
    auto layout = std::make_shared<gui::Vert>(0, margins);
    layout->AddChild(gui::Horiz::MakeCentered(title));
    layout->AddFixed(em);

    auto columns = std::make_shared<gui::Horiz>(em, gui::Margins());
    columns->AddChild(left_col);
    columns->AddChild(right_col);
    layout->AddChild(columns);

    layout->AddFixed(em);
    layout->AddChild(gui::Horiz::MakeCentered(ok));
    dlg->AddChild(layout);

    return dlg;
}

std::shared_ptr<geometry::TriangleMesh> CreateAxes(double axis_length) {
    const double sphere_radius = 0.005 * axis_length;
    const double cyl_radius = 0.0025 * axis_length;
    const double cone_radius = 0.0075 * axis_length;
    const double cyl_height = 0.975 * axis_length;
    const double cone_height = 0.025 * axis_length;

    auto mesh_frame = geometry::TriangleMesh::CreateSphere(sphere_radius);
    mesh_frame->ComputeVertexNormals();
    mesh_frame->PaintUniformColor(Eigen::Vector3d(0.5, 0.5, 0.5));

    std::shared_ptr<geometry::TriangleMesh> mesh_arrow;
    Eigen::Matrix4d transformation;

    mesh_arrow = geometry::TriangleMesh::CreateArrow(cyl_radius, cone_radius,
                                                     cyl_height, cone_height);
    mesh_arrow->ComputeVertexNormals();
    mesh_arrow->PaintUniformColor(Eigen::Vector3d(1.0, 0.0, 0.0));
    transformation << 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1;
    mesh_arrow->Transform(transformation);
    *mesh_frame += *mesh_arrow;

    mesh_arrow = geometry::TriangleMesh::CreateArrow(cyl_radius, cone_radius,
                                                     cyl_height, cone_height);
    mesh_arrow->ComputeVertexNormals();
    mesh_arrow->PaintUniformColor(Eigen::Vector3d(0.0, 1.0, 0.0));
    transformation << 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1;
    mesh_arrow->Transform(transformation);
    *mesh_frame += *mesh_arrow;

    mesh_arrow = geometry::TriangleMesh::CreateArrow(cyl_radius, cone_radius,
                                                     cyl_height, cone_height);
    mesh_arrow->ComputeVertexNormals();
    mesh_arrow->PaintUniformColor(Eigen::Vector3d(0.0, 0.0, 1.0));
    transformation << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
    mesh_arrow->Transform(transformation);
    *mesh_frame += *mesh_arrow;

    // Add UVs because material shader for axes expects them
    mesh_frame->triangle_uvs_.resize(mesh_frame->triangles_.size() * 3,
                                     {0.0, 0.0});

    return mesh_frame;
}

bool ColorArrayIsUniform(const std::vector<Eigen::Vector3d> &colors) {
    static const double e = 1.0 / 255.0;
    static const double SQ_EPSILON = Eigen::Vector3d(e, e, e).squaredNorm();
    const auto &color = colors[0];

    for (const auto &c : colors) {
        if ((color - c).squaredNorm() > SQ_EPSILON) {
            return false;
        }
    }

    return true;
}

bool PointCloudHasUniformColor(const geometry::PointCloud &pcd) {
    if (!pcd.HasColors()) {
        return true;
    }
    return ColorArrayIsUniform(pcd.colors_);
};

bool MeshHasUniformColor(const geometry::MeshBase &mesh) {
    if (!mesh.HasVertexColors()) {
        return true;
    }
    return ColorArrayIsUniform(mesh.vertex_colors_);
};

std::shared_ptr<gui::Slider> MakeSlider(const gui::Slider::Type type,
                                        const double min,
                                        const double max,
                                        const double value) {
    auto slider = std::make_shared<gui::Slider>(type);
    slider->SetLimits(min, max);
    slider->SetValue(value);
    return slider;
}

//----
class DrawTimeLabel : public gui::Label {
    using Super = Label;

public:
    DrawTimeLabel(gui::Window *w) : Label("0.0 ms") { window_ = w; }

    gui::Size CalcPreferredSize(const gui::Theme &theme) const override {
        auto h = Super::CalcPreferredSize(theme).height;
        return gui::Size(theme.font_size * 5, h);
    }

    DrawResult Draw(const gui::DrawContext &context) override {
        char text[64];
        // double ms = window_->GetLastFrameTimeSeconds() * 1000.0;
        double ms = 0.0;
        snprintf(text, sizeof(text) - 1, "%.1f ms", ms);
        SetText(text);

        return Super::Draw(context);
    }

private:
    gui::Window *window_;
};

//----
class SmallButton : public gui::Button {
    using Super = Button;

public:
    explicit SmallButton(const char *title) : Button(title) {
        SetPaddingEm(0.5f, 0.0f);
    }
};

//----
class SmallToggleButton : public SmallButton {
    using Super = SmallButton;

public:
    explicit SmallToggleButton(const char *title) : SmallButton(title) {
        SetToggleable(true);
    }
};

}  // namespace

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

static const std::string DEFAULT_IBL = "default";
static const std::string DEFAULT_MATERIAL_NAME = "Polished ceramic";
static const std::string MATERIAL_FROM_FILE_NAME =
        "Material from file [default]";
static const std::string POINT_CLOUD_PROFILE_NAME =
        "Cloudy day (no direct sun)";
static const bool DEFAULT_SHOW_SKYBOX = false;
static const bool DEFAULT_SHOW_AXES = false;

static const std::vector<LightingProfile> g_lighting_profiles = {
        {.name = "Bright day with sun at +Y [default]",
         .ibl_intensity = 45000,
         .sun_intensity = 45000,
         .sun_dir = {0.577f, -0.577f, -0.577f}},
        {.name = "Bright day with sun at -Y",
         .ibl_intensity = 45000,
         .sun_intensity = 45000,
         .sun_dir = {0.577f, 0.577f, 0.577f},
         .sun_color = {1.0f, 1.0f, 1.0f},
         .ibl_rotation = rendering::Scene::Transform(
                 Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitX()))},
        {.name = "Bright day with sun at +Z",
         .ibl_intensity = 45000,
         .sun_intensity = 45000,
         .sun_dir = {0.577f, 0.577f, -0.577f}},
        {.name = "Less bright day with sun at +Y",
         .ibl_intensity = 35000,
         .sun_intensity = 50000,
         .sun_dir = {0.577f, -0.577f, -0.577f}},
        {.name = "Less bright day with sun at -Y",
         .ibl_intensity = 35000,
         .sun_intensity = 50000,
         .sun_dir = {0.577f, 0.577f, 0.577f},
         .sun_color = {1.0f, 1.0f, 1.0f},
         .ibl_rotation = rendering::Scene::Transform(
                 Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitX()))},
        {.name = "Less bright day with sun at +Z",
         .ibl_intensity = 35000,
         .sun_intensity = 50000,
         .sun_dir = {0.577f, 0.577f, -0.577f}},
        {.name = POINT_CLOUD_PROFILE_NAME,
         .ibl_intensity = 60000,
         .sun_intensity = 50000,
         .sun_dir = {0.577f, -0.577f, -0.577f},
         .sun_color = {1.0f, 1.0f, 1.0f},
         .ibl_rotation = rendering::Scene::Transform::Identity(),
         .ibl_enabled = true,
         .use_default_ibl = true,
         .sun_enabled = false}};

enum MenuId {
    FILE_OPEN,
    FILE_EXPORT_RGB,
    FILE_QUIT,
    SETTINGS_LIGHT_AND_MATERIALS,
    HELP_KEYS,
    HELP_CAMERA,
    HELP_ABOUT,
    HELP_CONTACT
};

struct GuiVisualizer::Impl {
    std::vector<rendering::GeometryHandle> geometry_handles_;

    std::shared_ptr<gui::SceneWidget> scene_;
    std::shared_ptr<gui::VGrid> help_keys_;
    std::shared_ptr<gui::VGrid> help_camera_;

    struct LitMaterial {
        rendering::MaterialInstanceHandle handle;
        Eigen::Vector3f base_color = {0.9f, 0.9f, 0.9f};
        float metallic = 0.f;
        float roughness = 0.7;
        float reflectance = 0.5f;
        float clear_coat = 0.2f;
        float clear_coat_roughness = 0.2f;
        float anisotropy = 0.f;
        float point_size = 5.f;
    };

    struct UnlitMaterial {
        rendering::MaterialInstanceHandle handle;
        // The base color should NOT be {1, 1, 1}, because then the
        // model will be invisible against the default white background.
        Eigen::Vector3f base_color = {0.9f, 0.9f, 0.9f};
        float point_size = 5.f;
    };

    struct TextureMaps {
        rendering::TextureHandle albedo_map =
                rendering::FilamentResourceManager::kDefaultTexture;
        rendering::TextureHandle normal_map =
                rendering::FilamentResourceManager::kDefaultNormalMap;
        rendering::TextureHandle ambient_occlusion_map =
                rendering::FilamentResourceManager::kDefaultTexture;
        rendering::TextureHandle roughness_map =
                rendering::FilamentResourceManager::kDefaultTexture;
        rendering::TextureHandle metallic_map =
                rendering::FilamentResourceManager::kDefaultTexture;
        rendering::TextureHandle reflectance_map =
                rendering::FilamentResourceManager::kDefaultTexture;
        rendering::TextureHandle clear_coat_map =
                rendering::FilamentResourceManager::kDefaultTexture;
        rendering::TextureHandle clear_coat_roughness_map =
                rendering::FilamentResourceManager::kDefaultTexture;
        rendering::TextureHandle anisotropy_map =
                rendering::FilamentResourceManager::kDefaultTexture;
    };

    std::map<std::string, LitMaterial> prefab_materials_ = {
            {DEFAULT_MATERIAL_NAME, {}},
            {"Metal (rougher)",
             {rendering::MaterialInstanceHandle::kBad,
              {1.0f, 1.0f, 1.0f},
              1.0f,
              0.5f,
              0.9f,
              0.0f,
              0.0f,
              0.0f,
              3.0f}},
            {"Metal (smoother)",
             {rendering::MaterialInstanceHandle::kBad,
              {1.0f, 1.0f, 1.0f},
              1.0f,
              0.3f,
              0.9f,
              0.0f,
              0.0f,
              0.0f,
              3.0f}},
            {"Plastic",
             {rendering::MaterialInstanceHandle::kBad,
              {1.0f, 1.0f, 1.0f},
              0.0f,
              0.5f,
              0.5f,
              0.5f,
              0.2f,
              0.0f,
              3.0f}},
            {"Glazed ceramic",
             {rendering::MaterialInstanceHandle::kBad,
              {1.0f, 1.0f, 1.0f},
              0.0f,
              0.5f,
              0.9f,
              1.0f,
              0.1f,
              0.0f,
              3.0f}},
            {"Clay",
             {rendering::MaterialInstanceHandle::kBad,
              {0.7725f, 0.7725f, 0.7725f},
              0.0f,
              1.0f,
              0.5f,
              0.1f,
              0.287f,
              0.0f,
              3.0f}},
    };

    struct Materials {
        LitMaterial lit;
        UnlitMaterial unlit;
        TextureMaps maps;
    };
    rendering::MaterialHandle lit_material_;
    rendering::MaterialHandle unlit_material_;

    struct Settings {
        rendering::IndirectLightHandle ibl;
        rendering::SkyboxHandle sky;
        rendering::LightHandle directional_light;
        rendering::GeometryHandle axes;

        std::shared_ptr<gui::Vert> wgt_base;
        std::shared_ptr<gui::Checkbox> wgt_show_axes;
        std::shared_ptr<gui::ColorEdit> wgt_bg_color;
        std::shared_ptr<gui::Button> wgt_mouse_arcball;
        std::shared_ptr<gui::Button> wgt_mouse_fly;
        std::shared_ptr<gui::Button> wgt_mouse_sun;
        std::shared_ptr<gui::Button> wgt_mouse_ibl;
        std::shared_ptr<gui::Button> wgt_mouse_model;
        std::shared_ptr<gui::Combobox> wgt_lighting_profile;
        std::shared_ptr<gui::CollapsableVert> wgtAdvanced;
        std::shared_ptr<gui::Checkbox> wgt_ibl_enabled;
        std::shared_ptr<gui::Checkbox> wgt_sky_enabled;
        std::shared_ptr<gui::Checkbox> wgt_directional_enabled;
        std::shared_ptr<gui::Combobox> wgt_ibls;
        std::shared_ptr<gui::Slider> wgt_ibl_intensity;
        std::shared_ptr<gui::Slider> wgt_sun_intensity;
        std::shared_ptr<gui::VectorEdit> wgt_sun_dir;
        std::shared_ptr<gui::ColorEdit> wgt_sun_color;

        enum MaterialType {
            LIT = 0,
            UNLIT,
            NORMAL_MAP,
            DEPTH,
        };

        MaterialType selected_type = LIT;
        Materials current_materials;
        // geometry index -> material  (entry exists if mesh HasMaterials())
        std::map<int, LitMaterial> loaded_materials_;
        std::shared_ptr<gui::Combobox> wgt_material_type;

        std::shared_ptr<gui::Combobox> wgt_prefab_material;
        std::shared_ptr<gui::ColorEdit> wgt_material_color;
        std::shared_ptr<gui::Button> wgt_reset_material_color;
        std::shared_ptr<gui::Slider> wgt_point_size;

        bool user_has_changed_color = false;
        bool user_has_changed_lighting = false;

        void SetCustomProfile() {
            wgt_lighting_profile->SetSelectedIndex(g_lighting_profiles.size());
            user_has_changed_lighting = true;
        }
    } settings_;

    int app_menu_custom_items_index_ = -1;
    std::shared_ptr<gui::Menu> app_menu_;

    void SetMaterialsToDefault(rendering::Renderer &renderer) {
        settings_.loaded_materials_.clear();
        if (settings_.wgt_prefab_material) {
            settings_.wgt_prefab_material->RemoveItem(
                    MATERIAL_FROM_FILE_NAME.c_str());
        }

        Materials defaults;
        if (settings_.user_has_changed_color) {
            defaults.unlit.base_color =
                    settings_.current_materials.unlit.base_color;
            defaults.lit.base_color =
                    settings_.current_materials.lit.base_color;
        }
        defaults.maps.albedo_map =
                rendering::FilamentResourceManager::kDefaultTexture;
        defaults.maps.normal_map =
                rendering::FilamentResourceManager::kDefaultNormalMap;
        defaults.maps.ambient_occlusion_map =
                rendering::FilamentResourceManager::kDefaultTexture;
        defaults.maps.roughness_map =
                rendering::FilamentResourceManager::kDefaultTexture;
        defaults.maps.metallic_map =
                rendering::FilamentResourceManager::kDefaultTexture;
        defaults.maps.reflectance_map =
                rendering::FilamentResourceManager::kDefaultTexture;
        defaults.maps.clear_coat_map =
                rendering::FilamentResourceManager::kDefaultTexture;
        defaults.maps.clear_coat_roughness_map =
                rendering::FilamentResourceManager::kDefaultTexture;
        defaults.maps.anisotropy_map =
                rendering::FilamentResourceManager::kDefaultTexture;
        settings_.current_materials = defaults;

        auto lit_handle = renderer.AddMaterialInstance(lit_material_);
        settings_.current_materials.lit.handle =
                renderer.ModifyMaterial(lit_handle)
                        .SetColor("baseColor", defaults.lit.base_color)
                        .SetParameter("baseRoughness", defaults.lit.roughness)
                        .SetParameter("baseMetallic", defaults.lit.metallic)
                        .SetParameter("reflectance", defaults.lit.reflectance)
                        .SetParameter("clearCoat", defaults.lit.clear_coat)
                        .SetParameter("clearCoatRoughness",
                                      defaults.lit.clear_coat_roughness)
                        .SetParameter("anisotropy", defaults.lit.anisotropy)
                        .SetParameter("pointSize", defaults.lit.point_size)
                        .SetTexture(
                                "albedo", defaults.maps.albedo_map,
                                rendering::TextureSamplerParameters::Pretty())
                        .SetTexture(
                                "normalMap", defaults.maps.normal_map,
                                rendering::TextureSamplerParameters::Pretty())
                        .SetTexture(
                                "ambientOcclusionMap",
                                defaults.maps.ambient_occlusion_map,
                                rendering::TextureSamplerParameters::Pretty())
                        .SetTexture(
                                "roughnessMap", defaults.maps.roughness_map,
                                rendering::TextureSamplerParameters::Pretty())
                        .SetTexture(
                                "metallicMap", defaults.maps.metallic_map,
                                rendering::TextureSamplerParameters::Pretty())
                        .SetTexture(
                                "reflectanceMap", defaults.maps.reflectance_map,
                                rendering::TextureSamplerParameters::Pretty())
                        .SetTexture(
                                "clearCoatMap", defaults.maps.clear_coat_map,
                                rendering::TextureSamplerParameters::Pretty())
                        .SetTexture(
                                "clearCoatRoughnessMap",
                                defaults.maps.clear_coat_roughness_map,
                                rendering::TextureSamplerParameters::Pretty())
                        .SetTexture(
                                "anisotropyMap", defaults.maps.anisotropy_map,
                                rendering::TextureSamplerParameters::Pretty())
                        .Finish();

        auto unlit_handle = renderer.AddMaterialInstance(unlit_material_);
        settings_.current_materials.unlit.handle =
                renderer.ModifyMaterial(unlit_handle)
                        .SetColor("baseColor", defaults.unlit.base_color)
                        .SetParameter("pointSize", defaults.unlit.point_size)
                        .SetTexture(
                                "albedo", defaults.maps.albedo_map,
                                rendering::TextureSamplerParameters::Pretty())
                        .Finish();

        if (settings_.wgt_prefab_material) {
            settings_.wgt_prefab_material->SetSelectedValue(
                    DEFAULT_MATERIAL_NAME.c_str());
        }
        if (settings_.wgt_material_color) {
            Eigen::Vector3f color =
                    (settings_.selected_type == Settings::MaterialType::LIT
                             ? defaults.lit.base_color
                             : defaults.unlit.base_color);
            settings_.wgt_material_color->SetValue(color.x(), color.y(),
                                                   color.z());
        }
    }

    void SetMaterialsToCurrentSettings(rendering::Renderer &renderer,
                                       LitMaterial material,
                                       TextureMaps maps) {
        // Update the material settings
        settings_.current_materials.lit.base_color = material.base_color;
        settings_.current_materials.lit.roughness = material.roughness;
        settings_.current_materials.lit.metallic = material.metallic;
        settings_.current_materials.lit.reflectance = material.reflectance;
        settings_.current_materials.lit.clear_coat = material.clear_coat;
        settings_.current_materials.lit.clear_coat_roughness =
                material.clear_coat_roughness;
        settings_.current_materials.lit.anisotropy = material.anisotropy;
        settings_.current_materials.unlit.base_color = material.base_color;

        // Update maps
        settings_.current_materials.maps = maps;

        // update materials
        settings_.current_materials.lit.handle =
                renderer.ModifyMaterial(settings_.current_materials.lit.handle)
                        .SetColor("baseColor", material.base_color)
                        .SetParameter("baseRoughness", material.roughness)
                        .SetParameter("baseMetallic", material.metallic)
                        .SetParameter("reflectance", material.reflectance)
                        .SetParameter("clearCoat", material.clear_coat)
                        .SetParameter("clearCoatRoughness",
                                      material.clear_coat_roughness)
                        .SetParameter("anisotropy", material.anisotropy)
                        .SetTexture(
                                "albedo", maps.albedo_map,
                                rendering::TextureSamplerParameters::Pretty())
                        .SetTexture(
                                "normalMap", maps.normal_map,
                                rendering::TextureSamplerParameters::Pretty())
                        .SetTexture(
                                "ambientOcclusionMap",
                                maps.ambient_occlusion_map,
                                rendering::TextureSamplerParameters::Pretty())
                        .SetTexture(
                                "roughnessMap", maps.roughness_map,
                                rendering::TextureSamplerParameters::Pretty())
                        .SetTexture(
                                "metallicMap", maps.metallic_map,
                                rendering::TextureSamplerParameters::Pretty())
                        .SetTexture(
                                "reflectanceMap", maps.reflectance_map,
                                rendering::TextureSamplerParameters::Pretty())
                        .SetTexture(
                                "clearCoatMap", maps.clear_coat_map,
                                rendering::TextureSamplerParameters::Pretty())
                        .SetTexture(
                                "clearCoatRoughnessMap",
                                maps.clear_coat_roughness_map,
                                rendering::TextureSamplerParameters::Pretty())
                        .SetTexture(
                                "anisotropyMap", maps.anisotropy_map,
                                rendering::TextureSamplerParameters::Pretty())
                        .Finish();
        settings_.current_materials.unlit.handle =
                renderer.ModifyMaterial(
                                settings_.current_materials.unlit.handle)
                        .SetColor("baseColor", material.base_color)
                        .SetTexture(
                                "albedo", maps.albedo_map,
                                rendering::TextureSamplerParameters::Pretty())
                        .Finish();
    }

    void SetMaterialType(Impl::Settings::MaterialType type) {
        using MaterialType = Impl::Settings::MaterialType;
        using ViewMode = rendering::View::Mode;

        auto render_scene = scene_->GetScene();
        auto view = scene_->GetView();
        settings_.selected_type = type;
        settings_.wgt_material_type->SetSelectedIndex(int(type));

        bool is_lit = (type == MaterialType::LIT);
        settings_.wgt_prefab_material->SetEnabled(is_lit);

        switch (type) {
            case MaterialType::LIT: {
                view->SetMode(ViewMode::Color);
                for (const auto &handle : geometry_handles_) {
                    auto mat = settings_.current_materials.lit.handle;
                    render_scene->AssignMaterial(handle, mat);
                }
                settings_.wgt_material_color->SetEnabled(true);
                gui::Color color(
                        settings_.current_materials.lit.base_color.x(),
                        settings_.current_materials.lit.base_color.y(),
                        settings_.current_materials.lit.base_color.z());
                settings_.wgt_material_color->SetValue(color);
                settings_.wgt_reset_material_color->SetEnabled(
                        settings_.user_has_changed_color);
                break;
            }
            case MaterialType::UNLIT: {
                view->SetMode(ViewMode::Color);
                for (const auto &handle : geometry_handles_) {
                    auto mat = settings_.current_materials.unlit.handle;
                    render_scene->AssignMaterial(handle, mat);
                }
                settings_.wgt_material_color->SetEnabled(true);
                gui::Color color(
                        settings_.current_materials.unlit.base_color.x(),
                        settings_.current_materials.unlit.base_color.y(),
                        settings_.current_materials.unlit.base_color.z());
                settings_.wgt_material_color->SetValue(color);
                settings_.wgt_reset_material_color->SetEnabled(
                        settings_.user_has_changed_color);
                break;
            }
            case MaterialType::NORMAL_MAP:
                view->SetMode(ViewMode::Normals);
                settings_.wgt_material_color->SetEnabled(false);
                settings_.wgt_reset_material_color->SetEnabled(false);
                break;
            case MaterialType::DEPTH:
                view->SetMode(ViewMode::Depth);
                settings_.wgt_material_color->SetEnabled(false);
                settings_.wgt_reset_material_color->SetEnabled(false);
                break;
        }
    }

    rendering::MaterialInstanceHandle CreateUnlitMaterial(
            rendering::Renderer &renderer,
            rendering::MaterialInstanceHandle mat) {
        auto color = settings_.wgt_material_color->GetValue();
        Eigen::Vector3f color3(color.GetRed(), color.GetGreen(),
                               color.GetBlue());
        float point_size = settings_.wgt_point_size->GetDoubleValue();
        return renderer.ModifyMaterial(mat)
                .SetColor("baseColor", color3)
                .SetParameter("pointSize", point_size)
                .Finish();
    }

    rendering::MaterialInstanceHandle CreateLitMaterial(
            rendering::Renderer &renderer,
            rendering::MaterialInstanceHandle mat,
            const LitMaterial &prefab) {
        Eigen::Vector3f color;
        if (settings_.user_has_changed_color) {
            auto c = settings_.wgt_material_color->GetValue();
            color = Eigen::Vector3f(c.GetRed(), c.GetGreen(), c.GetBlue());
        } else {
            color = prefab.base_color;
        }
        float point_size = settings_.wgt_point_size->GetDoubleValue();
        return renderer.ModifyMaterial(mat)
                .SetColor("baseColor", color)
                .SetParameter("baseRoughness", prefab.roughness)
                .SetParameter("baseMetallic", prefab.metallic)
                .SetParameter("reflectance", prefab.reflectance)
                .SetParameter("clearCoat", prefab.clear_coat)
                .SetParameter("clearCoatRoughness", prefab.clear_coat_roughness)
                .SetParameter("anisotropy", prefab.anisotropy)
                .SetParameter("pointSize", point_size)
                .Finish();
    }

    void SetMaterialByName(rendering::Renderer &renderer,
                           const std::string &name) {
        if (name == MATERIAL_FROM_FILE_NAME) {
            for (size_t i = 0; i < geometry_handles_.size(); ++i) {
                auto mat_desc = settings_.loaded_materials_.find(i);
                if (mat_desc == settings_.loaded_materials_.end()) {
                    continue;
                }
                auto mat = settings_.current_materials.lit.handle;
                mat = this->CreateLitMaterial(renderer, mat, mat_desc->second);
                scene_->GetScene()->AssignMaterial(geometry_handles_[i], mat);
                settings_.current_materials.lit.handle = mat;
            }
            if (!settings_.user_has_changed_color &&
                settings_.loaded_materials_.size() == 1) {
                auto color =
                        settings_.loaded_materials_.begin()->second.base_color;
                settings_.wgt_material_color->SetValue(color.x(), color.y(),
                                                       color.z());
                settings_.current_materials.lit.base_color = color;
            }
        } else {
            auto prefab_it = prefab_materials_.find(name);
            // DEFAULT_MATERIAL_NAME may have "[default]" appended, if the model
            // doesn't have its own material, so search again if that happened.
            if (prefab_it == prefab_materials_.end() &&
                name.find(DEFAULT_MATERIAL_NAME) == 0) {
                prefab_it = prefab_materials_.find(DEFAULT_MATERIAL_NAME);
            }
            if (prefab_it != prefab_materials_.end()) {
                auto &prefab = prefab_it->second;
                if (!settings_.user_has_changed_color) {
                    settings_.current_materials.lit.base_color =
                            prefab.base_color;
                    settings_.wgt_material_color->SetValue(
                            prefab.base_color.x(), prefab.base_color.y(),
                            prefab.base_color.z());
                }
                auto mat = settings_.current_materials.lit.handle;
                mat = this->CreateLitMaterial(renderer, mat, prefab);
                for (const auto &handle : geometry_handles_) {
                    scene_->GetScene()->AssignMaterial(handle, mat);
                }
                settings_.current_materials.lit.handle = mat;
            }
        }
    }

    void SetLightingProfile(rendering::Renderer &renderer,
                            const std::string &name) {
        for (size_t i = 0; i < g_lighting_profiles.size(); ++i) {
            if (g_lighting_profiles[i].name == name) {
                SetLightingProfile(renderer, g_lighting_profiles[i]);
                settings_.wgt_lighting_profile->SetSelectedValue(name.c_str());
                return;
            }
        }
        utility::LogWarning("Could not find lighting profile '{}'", name);
    }

    void SetLightingProfile(rendering::Renderer &renderer,
                            const LightingProfile &profile) {
        auto *render_scene = scene_->GetScene();
        if (profile.use_default_ibl) {
            this->SetIBL(renderer, nullptr);
            settings_.wgt_ibls->SetSelectedValue(DEFAULT_IBL.c_str());
        }
        if (profile.ibl_enabled) {
            render_scene->SetIndirectLight(settings_.ibl);
        } else {
            render_scene->SetIndirectLight(rendering::IndirectLightHandle());
        }
        render_scene->SetIndirectLightIntensity(profile.ibl_intensity);
        render_scene->SetIndirectLightRotation(profile.ibl_rotation);
        render_scene->SetSkybox(rendering::SkyboxHandle());
        render_scene->SetEntityEnabled(settings_.directional_light,
                                       profile.sun_enabled);
        render_scene->SetLightIntensity(settings_.directional_light,
                                        profile.sun_intensity);
        render_scene->SetLightDirection(settings_.directional_light,
                                        profile.sun_dir);
        render_scene->SetLightColor(settings_.directional_light,
                                    profile.sun_color);
        settings_.wgt_ibl_enabled->SetChecked(profile.ibl_enabled);
        settings_.wgt_sky_enabled->SetChecked(false);
        settings_.wgt_directional_enabled->SetChecked(profile.sun_enabled);
        settings_.wgt_ibls->SetSelectedValue(DEFAULT_IBL.c_str());
        settings_.wgt_ibl_intensity->SetValue(profile.ibl_intensity);
        settings_.wgt_sun_intensity->SetValue(profile.sun_intensity);
        settings_.wgt_sun_dir->SetValue(profile.sun_dir);
        settings_.wgt_sun_color->SetValue(gui::Color(profile.sun_color[0],
                                                     profile.sun_color[1],
                                                     profile.sun_color[2]));
    }

    bool SetIBL(rendering::Renderer &renderer, const char *path) {
        rendering::IndirectLightHandle new_ibl;
        std::string ibl_path;
        if (path) {
            new_ibl = renderer.AddIndirectLight(
                    rendering::ResourceLoadRequest(path));
            ibl_path = path;
        } else {
            ibl_path =
                    std::string(
                            gui::Application::GetInstance().GetResourcePath()) +
                    "/" + DEFAULT_IBL + "_ibl.ktx";
            new_ibl = renderer.AddIndirectLight(
                    rendering::ResourceLoadRequest(ibl_path.c_str()));
        }
        if (new_ibl) {
            auto *render_scene = scene_->GetScene();
            settings_.ibl = new_ibl;
            auto intensity = render_scene->GetIndirectLightIntensity();
            render_scene->SetIndirectLight(new_ibl);
            render_scene->SetIndirectLightIntensity(intensity);

            auto skybox_path = std::string(ibl_path);
            if (skybox_path.find("_ibl.ktx") != std::string::npos) {
                skybox_path = skybox_path.substr(0, skybox_path.size() - 8);
                skybox_path += "_skybox.ktx";
                settings_.sky = renderer.AddSkybox(
                        rendering::ResourceLoadRequest(skybox_path.c_str()));
                if (!settings_.sky) {
                    settings_.sky = renderer.AddSkybox(
                            rendering::ResourceLoadRequest(ibl_path.c_str()));
                }
                bool is_on = settings_.wgt_sky_enabled->IsChecked();
                if (is_on) {
                    scene_->GetScene()->SetSkybox(settings_.sky);
                }
                scene_->SetSkyboxHandle(settings_.sky, is_on);
            }
            return true;
        }
        return false;
    }

    void SetMouseControls(gui::Window &window,
                          gui::SceneWidget::Controls mode) {
        using Controls = gui::SceneWidget::Controls;
        scene_->SetViewControls(mode);
        window.SetFocusWidget(scene_.get());
        settings_.wgt_mouse_arcball->SetOn(mode == Controls::ROTATE_OBJ);
        settings_.wgt_mouse_fly->SetOn(mode == Controls::FLY);
        settings_.wgt_mouse_model->SetOn(mode == Controls::ROTATE_MODEL);
        settings_.wgt_mouse_sun->SetOn(mode == Controls::ROTATE_SUN);
        settings_.wgt_mouse_ibl->SetOn(mode == Controls::ROTATE_IBL);
    }
};

GuiVisualizer::GuiVisualizer(const std::string &title, int width, int height)
    : gui::Window(title, width, height), impl_(new GuiVisualizer::Impl()) {
    Init();
}

GuiVisualizer::GuiVisualizer(
        const std::vector<std::shared_ptr<const geometry::Geometry>>
                &geometries,
        const std::string &title,
        int width,
        int height,
        int left,
        int top)
    : gui::Window(title, left, top, width, height),
      impl_(new GuiVisualizer::Impl()) {
    Init();
    SetGeometry(geometries);  // also updates the camera
}

void GuiVisualizer::Init() {
    auto &app = gui::Application::GetInstance();
    auto &theme = GetTheme();

    // Create menu
    if (!gui::Application::GetInstance().GetMenubar()) {
        auto menu = std::make_shared<gui::Menu>();
#if defined(__APPLE__)
        // The first menu item to be added on macOS becomes the application
        // menu (no matter its name)
        auto app_menu = std::make_shared<gui::Menu>();
        app_menu->AddItem("About", HELP_ABOUT);
        app_menu->AddSeparator();
        impl_->app_menu_custom_items_index_ = app_menu->GetNumberOfItems();
        app_menu->AddItem("Quit", FILE_QUIT, gui::KEY_Q);
        menu->AddMenu("Open3D", app_menu);
        impl_->app_menu_ = app_menu;
#endif  // __APPLE__
        auto file_menu = std::make_shared<gui::Menu>();
        file_menu->AddItem("Open...", FILE_OPEN, gui::KEY_O);
        file_menu->AddItem("Export Current Image...", FILE_EXPORT_RGB);
        file_menu->AddSeparator();
#if WIN32
        file_menu->AddItem("Exit", FILE_QUIT);
#elif !defined(__APPLE__)  // quit goes in app menu on macOS
        file_menu->AddItem("Quit", FILE_QUIT, gui::KEY_Q);
#endif
        menu->AddMenu("File", file_menu);

        auto settings_menu = std::make_shared<gui::Menu>();
        settings_menu->AddItem("Lighting & Materials",
                               SETTINGS_LIGHT_AND_MATERIALS);
        settings_menu->SetChecked(SETTINGS_LIGHT_AND_MATERIALS, true);
        menu->AddMenu("Settings", settings_menu);

        auto help_menu = std::make_shared<gui::Menu>();
        help_menu->AddItem("Show Controls", HELP_KEYS);
        help_menu->AddItem("Show Camera Info", HELP_CAMERA);
        help_menu->AddSeparator();
        help_menu->AddItem("About", HELP_ABOUT);
        help_menu->AddItem("Contact", HELP_CONTACT);
#if defined(__APPLE__) && GUI_USE_NATIVE_MENUS
        // macOS adds a special search item to menus named "Help",
        // so add a space to avoid that.
        menu->AddMenu("Help ", help_menu);
#else
        menu->AddMenu("Help", help_menu);
#endif

        gui::Application::GetInstance().SetMenubar(menu);
    }

    // Create scene
    auto scene_id = GetRenderer().CreateScene();
    auto scene = std::make_shared<gui::SceneWidget>(
            *GetRenderer().GetScene(scene_id));
    auto render_scene = scene->GetScene();
    impl_->scene_ = scene;
    scene->SetBackgroundColor(gui::Color(1.0, 1.0, 1.0));

    // Create light
    const int default_lighting_profile_idx = 0;
    auto &lighting_profile = g_lighting_profiles[default_lighting_profile_idx];
    rendering::LightDescription light_description;
    light_description.intensity = lighting_profile.sun_intensity;
    light_description.direction = lighting_profile.sun_dir;
    light_description.cast_shadows = true;
    light_description.custom_attributes["custom_type"] = "SUN";

    impl_->settings_.directional_light =
            scene->GetScene()->AddLight(light_description);

    auto &settings = impl_->settings_;
    std::string resource_path = app.GetResourcePath();
    auto ibl_path = resource_path + "/default_ibl.ktx";
    settings.ibl = GetRenderer().AddIndirectLight(
            rendering::ResourceLoadRequest(ibl_path.data()));
    render_scene->SetIndirectLight(settings.ibl);
    render_scene->SetIndirectLightIntensity(lighting_profile.ibl_intensity);
    render_scene->SetIndirectLightRotation(lighting_profile.ibl_rotation);

    auto sky_path = resource_path + "/" + DEFAULT_IBL + "_skybox.ktx";
    settings.sky = GetRenderer().AddSkybox(
            rendering::ResourceLoadRequest(sky_path.data()));
    scene->SetSkyboxHandle(settings.sky, DEFAULT_SHOW_SKYBOX);

    // Create materials
    auto lit_path = resource_path + "/defaultLit.filamat";
    impl_->lit_material_ = GetRenderer().AddMaterial(
            rendering::ResourceLoadRequest(lit_path.data()));

    auto unlit_path = resource_path + "/defaultUnlit.filamat";
    impl_->unlit_material_ = GetRenderer().AddMaterial(
            rendering::ResourceLoadRequest(unlit_path.data()));

    impl_->SetMaterialsToDefault(GetRenderer());

    // Setup UI
    const auto em = theme.font_size;
    const int lm = std::ceil(0.5 * em);
    const int grid_spacing = std::ceil(0.25 * em);

    AddChild(scene);

    // Add settings widget
    const int separation_height = std::ceil(0.75 * em);
    // (we don't want as much left margin because the twisty arrow is the
    // only thing there, and visually it looks larger than the right.)
    const gui::Margins base_margins(0.5 * lm, lm, lm, lm);
    settings.wgt_base = std::make_shared<gui::Vert>(0, base_margins);

    gui::Margins indent(em, 0, 0, 0);
    auto view_ctrls =
            std::make_shared<gui::CollapsableVert>("View controls", 0, indent);

    // ... view manipulator buttons
    settings.wgt_mouse_arcball = std::make_shared<SmallToggleButton>("Arcball");
    impl_->settings_.wgt_mouse_arcball->SetOn(true);
    settings.wgt_mouse_arcball->SetOnClicked([this]() {
        impl_->SetMouseControls(*this, gui::SceneWidget::Controls::ROTATE_OBJ);
    });
    settings.wgt_mouse_fly = std::make_shared<SmallToggleButton>("Fly");
    settings.wgt_mouse_fly->SetOnClicked([this]() {
        impl_->SetMouseControls(*this, gui::SceneWidget::Controls::FLY);
    });
    settings.wgt_mouse_model = std::make_shared<SmallToggleButton>("Model");
    settings.wgt_mouse_model->SetOnClicked([this]() {
        impl_->SetMouseControls(*this,
                                gui::SceneWidget::Controls::ROTATE_MODEL);
    });
    settings.wgt_mouse_sun = std::make_shared<SmallToggleButton>("Sun");
    settings.wgt_mouse_sun->SetOnClicked([this]() {
        impl_->SetMouseControls(*this, gui::SceneWidget::Controls::ROTATE_SUN);
    });
    settings.wgt_mouse_ibl = std::make_shared<SmallToggleButton>("Environment");
    settings.wgt_mouse_ibl->SetOnClicked([this]() {
        impl_->SetMouseControls(*this, gui::SceneWidget::Controls::ROTATE_IBL);
    });

    auto reset_camera = std::make_shared<SmallButton>("Reset camera");
    reset_camera->SetOnClicked([this]() {
        impl_->scene_->GoToCameraPreset(gui::SceneWidget::CameraPreset::PLUS_Z);
    });

    auto camera_controls1 = std::make_shared<gui::Horiz>(grid_spacing);
    camera_controls1->AddStretch();
    camera_controls1->AddChild(settings.wgt_mouse_arcball);
    camera_controls1->AddChild(settings.wgt_mouse_fly);
    camera_controls1->AddChild(settings.wgt_mouse_model);
    camera_controls1->AddStretch();
    auto camera_controls2 = std::make_shared<gui::Horiz>(grid_spacing);
    camera_controls2->AddStretch();
    camera_controls2->AddChild(settings.wgt_mouse_sun);
    camera_controls2->AddChild(settings.wgt_mouse_ibl);
    camera_controls2->AddStretch();
    view_ctrls->AddChild(std::make_shared<gui::Label>("Mouse Controls"));
    view_ctrls->AddChild(camera_controls1);
    view_ctrls->AddFixed(0.25 * em);
    view_ctrls->AddChild(camera_controls2);
    view_ctrls->AddFixed(separation_height);
    view_ctrls->AddChild(gui::Horiz::MakeCentered(reset_camera));

    // ... background
    settings.wgt_sky_enabled = std::make_shared<gui::Checkbox>("Show skymap");
    settings.wgt_sky_enabled->SetChecked(DEFAULT_SHOW_SKYBOX);
    settings.wgt_sky_enabled->SetOnChecked([this, render_scene](bool checked) {
        if (checked) {
            render_scene->SetSkybox(impl_->settings_.sky);
        } else {
            render_scene->SetSkybox(rendering::SkyboxHandle());
        }
        impl_->scene_->SetSkyboxHandle(impl_->settings_.sky, checked);
        impl_->settings_.wgt_bg_color->SetEnabled(!checked);
    });

    impl_->settings_.wgt_bg_color = std::make_shared<gui::ColorEdit>();
    impl_->settings_.wgt_bg_color->SetValue({1, 1, 1});
    impl_->settings_.wgt_bg_color->SetOnValueChanged(
            [scene](const gui::Color &newColor) {
                scene->SetBackgroundColor(newColor);
            });
    auto bg_layout = std::make_shared<gui::VGrid>(2, grid_spacing);
    bg_layout->AddChild(std::make_shared<gui::Label>("BG Color"));
    bg_layout->AddChild(impl_->settings_.wgt_bg_color);

    view_ctrls->AddFixed(separation_height);
    view_ctrls->AddChild(settings.wgt_sky_enabled);
    view_ctrls->AddFixed(0.25 * em);
    view_ctrls->AddChild(bg_layout);

    // ... show axes
    settings.wgt_show_axes = std::make_shared<gui::Checkbox>("Show axes");
    settings.wgt_show_axes->SetChecked(DEFAULT_SHOW_AXES);
    settings.wgt_show_axes->SetOnChecked([this, render_scene](bool isChecked) {
        render_scene->SetEntityEnabled(this->impl_->settings_.axes, isChecked);
    });
    view_ctrls->AddFixed(separation_height);
    view_ctrls->AddChild(settings.wgt_show_axes);

    // ... lighting profiles
    settings.wgt_lighting_profile = std::make_shared<gui::Combobox>();
    for (size_t i = 0; i < g_lighting_profiles.size(); ++i) {
        settings.wgt_lighting_profile->AddItem(
                g_lighting_profiles[i].name.c_str());
    }
    settings.wgt_lighting_profile->AddItem("Custom");
    settings.wgt_lighting_profile->SetSelectedIndex(
            default_lighting_profile_idx);
    settings.wgt_lighting_profile->SetOnValueChanged(
            [this](const char *, int index) {
                if (index < int(g_lighting_profiles.size())) {
                    this->impl_->SetLightingProfile(this->GetRenderer(),
                                                    g_lighting_profiles[index]);
                    this->impl_->settings_.user_has_changed_lighting = true;
                } else {
                    this->impl_->settings_.wgtAdvanced->SetIsOpen(true);
                    this->SetNeedsLayout();
                }
            });

    auto profile_layout = std::make_shared<gui::Vert>();
    profile_layout->AddChild(std::make_shared<gui::Label>("Lighting profiles"));
    profile_layout->AddChild(settings.wgt_lighting_profile);
    view_ctrls->AddFixed(separation_height);
    view_ctrls->AddChild(profile_layout);

    settings.wgt_base->AddChild(view_ctrls);
    settings.wgt_base->AddFixed(separation_height);

    // ... advanced lighting
    settings.wgtAdvanced = std::make_shared<gui::CollapsableVert>(
            "Advanced lighting", 0, indent);
    settings.wgtAdvanced->SetIsOpen(false);
    settings.wgt_base->AddChild(settings.wgtAdvanced);

    // ....... lighting on/off
    settings.wgtAdvanced->AddChild(
            std::make_shared<gui::Label>("Light sources"));
    auto checkboxes = std::make_shared<gui::Horiz>();
    settings.wgt_ibl_enabled = std::make_shared<gui::Checkbox>("HDR map");
    settings.wgt_ibl_enabled->SetChecked(true);
    settings.wgt_ibl_enabled->SetOnChecked([this, render_scene](bool checked) {
        impl_->settings_.SetCustomProfile();
        if (checked) {
            render_scene->SetIndirectLight(impl_->settings_.ibl);
            render_scene->SetIndirectLightIntensity(
                    impl_->settings_.wgt_ibl_intensity->GetDoubleValue());
        } else {
            render_scene->SetIndirectLight(rendering::IndirectLightHandle());
        }
        this->impl_->settings_.user_has_changed_lighting = true;
    });
    checkboxes->AddChild(settings.wgt_ibl_enabled);
    settings.wgt_directional_enabled = std::make_shared<gui::Checkbox>("Sun");
    settings.wgt_directional_enabled->SetChecked(true);
    settings.wgt_directional_enabled->SetOnChecked(
            [this, render_scene](bool checked) {
                impl_->settings_.SetCustomProfile();
                render_scene->SetEntityEnabled(
                        impl_->settings_.directional_light, checked);
            });
    checkboxes->AddChild(settings.wgt_directional_enabled);
    settings.wgtAdvanced->AddChild(checkboxes);

    settings.wgtAdvanced->AddFixed(separation_height);

    // ....... IBL
    settings.wgt_ibls = std::make_shared<gui::Combobox>();
    std::vector<std::string> resource_files;
    utility::filesystem::ListFilesInDirectory(resource_path, resource_files);
    std::sort(resource_files.begin(), resource_files.end());
    int n = 0;
    for (auto &f : resource_files) {
        if (f.find("_ibl.ktx") == f.size() - 8) {
            auto name = utility::filesystem::GetFileNameWithoutDirectory(f);
            name = name.substr(0, name.size() - 8);
            settings.wgt_ibls->AddItem(name.c_str());
            if (name == DEFAULT_IBL) {
                settings.wgt_ibls->SetSelectedIndex(n);
            }
            n++;
        }
    }
    settings.wgt_ibls->AddItem("Custom KTX file...");
    settings.wgt_ibls->SetOnValueChanged([this](const char *name, int) {
        std::string path = gui::Application::GetInstance().GetResourcePath();
        path += std::string("/") + name + "_ibl.ktx";
        if (!this->SetIBL(path.c_str())) {
            // must be the "Custom..." option
            auto dlg = std::make_shared<gui::FileDialog>(
                    gui::FileDialog::Mode::OPEN, "Open HDR Map", GetTheme());
            dlg->AddFilter(".ktx", "Khronos Texture (.ktx)");
            dlg->SetOnCancel([this]() { this->CloseDialog(); });
            dlg->SetOnDone([this](const char *path) {
                this->CloseDialog();
                this->SetIBL(path);
                this->impl_->settings_.SetCustomProfile();
            });
            ShowDialog(dlg);
        }
    });

    settings.wgt_ibl_intensity = MakeSlider(gui::Slider::INT, 0.0, 150000.0,
                                            lighting_profile.ibl_intensity);
    settings.wgt_ibl_intensity->SetOnValueChanged(
            [this, render_scene](double newValue) {
                render_scene->SetIndirectLightIntensity(newValue);
                this->impl_->settings_.SetCustomProfile();
            });

    auto ambient_layout = std::make_shared<gui::VGrid>(2, grid_spacing);
    ambient_layout->AddChild(std::make_shared<gui::Label>("HDR map"));
    ambient_layout->AddChild(settings.wgt_ibls);
    ambient_layout->AddChild(std::make_shared<gui::Label>("Intensity"));
    ambient_layout->AddChild(settings.wgt_ibl_intensity);

    settings.wgtAdvanced->AddChild(std::make_shared<gui::Label>("Environment"));
    settings.wgtAdvanced->AddChild(ambient_layout);
    settings.wgtAdvanced->AddFixed(separation_height);

    // ... directional light (sun)
    settings.wgt_sun_intensity = MakeSlider(gui::Slider::INT, 0.0, 500000.0,
                                            lighting_profile.sun_intensity);
    settings.wgt_sun_intensity->SetOnValueChanged(
            [this, render_scene](double new_value) {
                render_scene->SetLightIntensity(
                        impl_->settings_.directional_light, new_value);
                this->impl_->settings_.SetCustomProfile();
            });

    auto SetSunDir = [this, render_scene](const Eigen::Vector3f &dir) {
        this->impl_->settings_.wgt_sun_dir->SetValue(dir);
        render_scene->SetLightDirection(impl_->settings_.directional_light,
                                        dir.normalized());
        this->impl_->settings_.SetCustomProfile();
    };

    this->impl_->scene_->SelectDirectionalLight(
            settings.directional_light, [this](const Eigen::Vector3f &new_dir) {
                impl_->settings_.wgt_sun_dir->SetValue(new_dir);
                this->impl_->settings_.SetCustomProfile();
            });

    settings.wgt_sun_dir = std::make_shared<gui::VectorEdit>();
    settings.wgt_sun_dir->SetValue(light_description.direction);
    settings.wgt_sun_dir->SetOnValueChanged(SetSunDir);

    settings.wgt_sun_color = std::make_shared<gui::ColorEdit>();
    settings.wgt_sun_color->SetValue({1, 1, 1});
    settings.wgt_sun_color->SetOnValueChanged(
            [this, render_scene](const gui::Color &new_color) {
                this->impl_->settings_.SetCustomProfile();
                render_scene->SetLightColor(
                        impl_->settings_.directional_light,
                        {new_color.GetRed(), new_color.GetGreen(),
                         new_color.GetBlue()});
            });

    auto sun_layout = std::make_shared<gui::VGrid>(2, grid_spacing);
    sun_layout->AddChild(std::make_shared<gui::Label>("Intensity"));
    sun_layout->AddChild(settings.wgt_sun_intensity);
    sun_layout->AddChild(std::make_shared<gui::Label>("Direction"));
    sun_layout->AddChild(settings.wgt_sun_dir);
    sun_layout->AddChild(std::make_shared<gui::Label>("Color"));
    sun_layout->AddChild(settings.wgt_sun_color);

    settings.wgtAdvanced->AddChild(
            std::make_shared<gui::Label>("Sun (Directional light)"));
    settings.wgtAdvanced->AddChild(sun_layout);

    // materials settings
    auto materials = std::make_shared<gui::CollapsableVert>("Material settings",
                                                            0, indent);

    auto mat_grid = std::make_shared<gui::VGrid>(2, grid_spacing);
    mat_grid->AddChild(std::make_shared<gui::Label>("Type"));
    settings.wgt_material_type.reset(
            new gui::Combobox({"Lit", "Unlit", "Normal map", "Depth"}));
    settings.wgt_material_type->SetOnValueChanged([this](const char *,
                                                         int selected_idx) {
        impl_->SetMaterialType(Impl::Settings::MaterialType(selected_idx));
    });
    mat_grid->AddChild(settings.wgt_material_type);

    settings.wgt_prefab_material = std::make_shared<gui::Combobox>();
    for (auto &prefab : impl_->prefab_materials_) {
        settings.wgt_prefab_material->AddItem(prefab.first.c_str());
    }
    settings.wgt_prefab_material->SetSelectedValue(
            DEFAULT_MATERIAL_NAME.c_str());
    settings.wgt_prefab_material->SetOnValueChanged(
            [this](const char *name, int) {
                auto &renderer = this->GetRenderer();
                impl_->SetMaterialByName(renderer, name);
            });

    mat_grid->AddChild(std::make_shared<gui::Label>("Material"));
    mat_grid->AddChild(settings.wgt_prefab_material);

    settings.wgt_material_color = std::make_shared<gui::ColorEdit>();
    settings.wgt_material_color->SetOnValueChanged(
            [this, render_scene](const gui::Color &color) {
                auto &renderer = this->GetRenderer();
                auto &settings = impl_->settings_;
                Eigen::Vector3f color3(color.GetRed(), color.GetGreen(),
                                       color.GetBlue());
                if (settings.selected_type == Impl::Settings::LIT) {
                    settings.current_materials.lit.base_color = color3;
                } else {
                    settings.current_materials.unlit.base_color = color3;
                }
                settings.user_has_changed_color = true;
                settings.wgt_reset_material_color->SetEnabled(true);

                rendering::MaterialInstanceHandle mat;
                if (settings.selected_type == Impl::Settings::UNLIT) {
                    mat = settings.current_materials.unlit.handle;
                } else {
                    mat = settings.current_materials.lit.handle;
                }
                mat = renderer.ModifyMaterial(mat)
                              .SetColor("baseColor", color3)
                              .Finish();
                for (const auto &handle : impl_->geometry_handles_) {
                    render_scene->AssignMaterial(handle, mat);
                }
            });
    settings.wgt_reset_material_color = std::make_shared<SmallButton>("Reset");
    settings.wgt_reset_material_color->SetEnabled(
            impl_->settings_.user_has_changed_color);
    settings.wgt_reset_material_color->SetOnClicked([this]() {
        auto &renderer = this->GetRenderer();
        impl_->settings_.user_has_changed_color = false;
        impl_->settings_.wgt_reset_material_color->SetEnabled(false);
        impl_->SetMaterialByName(
                renderer,
                impl_->settings_.wgt_prefab_material->GetSelectedValue());
    });

    mat_grid->AddChild(std::make_shared<gui::Label>("Color"));
    auto color_layout = std::make_shared<gui::Horiz>();
    color_layout->AddChild(settings.wgt_material_color);
    color_layout->AddFixed(0.25 * em);
    color_layout->AddChild(impl_->settings_.wgt_reset_material_color);
    mat_grid->AddChild(color_layout);

    mat_grid->AddChild(std::make_shared<gui::Label>("Point size"));
    settings.wgt_point_size = MakeSlider(gui::Slider::INT, 1.0, 10.0, 3);
    settings.wgt_point_size->SetOnValueChanged([this](double value) {
        float size = float(value);
        impl_->settings_.current_materials.unlit.point_size = size;
        auto &renderer = GetRenderer();
        renderer.ModifyMaterial(impl_->settings_.current_materials.lit.handle)
                .SetParameter("pointSize", size)
                .Finish();
        renderer.ModifyMaterial(impl_->settings_.current_materials.unlit.handle)
                .SetParameter("pointSize", size)
                .Finish();
        renderer.ModifyMaterial(
                        rendering::FilamentResourceManager::kDepthMaterial)
                .SetParameter("pointSize", size)
                .Finish();
        renderer.ModifyMaterial(
                        rendering::FilamentResourceManager::kNormalsMaterial)
                .SetParameter("pointSize", size)
                .Finish();
    });
    mat_grid->AddChild(settings.wgt_point_size);
    materials->AddChild(mat_grid);

    settings.wgt_base->AddFixed(separation_height);
    settings.wgt_base->AddChild(materials);

    AddChild(settings.wgt_base);

    // Other items
    impl_->help_keys_ = CreateHelpDisplay(this);
    impl_->help_keys_->SetVisible(false);
    AddChild(impl_->help_keys_);
    impl_->help_camera_ = CreateCameraDisplay(this);
    impl_->help_camera_->SetVisible(false);
    AddChild(impl_->help_camera_);
}

GuiVisualizer::~GuiVisualizer() {}

void GuiVisualizer::SetTitle(const std::string &title) {
    Super::SetTitle(title.c_str());
}

void GuiVisualizer::AddItemsToAppMenu(
        const std::vector<std::pair<std::string, gui::Menu::ItemId>> &items) {
#if !defined(__APPLE__)
    return;  // application menu only exists on macOS
#endif

    if (impl_->app_menu_ && impl_->app_menu_custom_items_index_ >= 0) {
        for (auto &it : items) {
            impl_->app_menu_->InsertItem(impl_->app_menu_custom_items_index_++,
                                         it.first.c_str(), it.second);
        }
        impl_->app_menu_->InsertSeparator(
                impl_->app_menu_custom_items_index_++);
    }
}

void GuiVisualizer::SetGeometry(
        const std::vector<std::shared_ptr<const geometry::Geometry>>
                &geometries) {
    const std::size_t MIN_POINT_CLOUD_POINTS_FOR_DECIMATION = 6000000;

    gui::SceneWidget::ModelDescription desc;

    auto *scene3d = impl_->scene_->GetScene();
    if (impl_->settings_.axes) {
        scene3d->RemoveGeometry(impl_->settings_.axes);
    }
    for (auto &h : impl_->geometry_handles_) {
        scene3d->RemoveGeometry(h);
    }
    impl_->geometry_handles_.clear();

    impl_->SetMaterialsToDefault(GetRenderer());

    std::size_t num_point_clouds = 0;
    std::size_t num_point_cloud_points = 0;
    for (auto &g : geometries) {
        if (g->GetGeometryType() ==
            geometry::Geometry::GeometryType::PointCloud) {
            num_point_clouds++;
            auto cloud =
                    std::static_pointer_cast<const geometry::PointCloud>(g);
            num_point_cloud_points += cloud->points_.size();
        }
    }

    geometry::AxisAlignedBoundingBox bounds;
    std::size_t num_unlit = 0;
    for (size_t i = 0; i < geometries.size(); ++i) {
        std::shared_ptr<const geometry::Geometry> g = geometries[i];
        Impl::Materials materials = impl_->settings_.current_materials;

        rendering::MaterialInstanceHandle selected_material;

        // If a point cloud or mesh has no vertex colors or a single uniform
        // color (usually white), then we want to display it normally, that is,
        // lit. But if the cloud/mesh has differing vertex colors, then we
        // assume that the vertex colors have the lighting value baked in
        // (for example, fountain.ply at http://qianyi.info/scenedata.html)
        switch (g->GetGeometryType()) {
            case geometry::Geometry::GeometryType::PointCloud: {
                auto pcd =
                        std::static_pointer_cast<const geometry::PointCloud>(g);

                if (pcd->HasColors() && !PointCloudHasUniformColor(*pcd)) {
                    selected_material = materials.unlit.handle;
                    num_unlit += 1;
                } else {
                    selected_material = materials.lit.handle;
                }
            } break;
            case geometry::Geometry::GeometryType::LineSet: {
                selected_material = materials.unlit.handle;
                num_unlit += 1;
            } break;
            case geometry::Geometry::GeometryType::TriangleMesh: {
                auto mesh =
                        std::static_pointer_cast<const geometry::TriangleMesh>(
                                g);

                bool albedo_only = true;
                if (mesh->HasMaterials()) {
                    auto mesh_material = mesh->materials_.begin()->second;
                    Impl::LitMaterial material;
                    Impl::TextureMaps maps;
                    material.base_color.x() = mesh_material.baseColor.r();
                    material.base_color.y() = mesh_material.baseColor.g();
                    material.base_color.z() = mesh_material.baseColor.b();
                    material.roughness = mesh_material.baseRoughness;
                    material.reflectance = mesh_material.baseReflectance;
                    material.clear_coat = mesh_material.baseClearCoat;
                    material.clear_coat_roughness =
                            mesh_material.baseClearCoatRoughness;
                    material.anisotropy = mesh_material.baseAnisotropy;

                    auto is_map_valid =
                            [](std::shared_ptr<geometry::Image> map) -> bool {
                        return map && map->HasData();
                    };

                    if (is_map_valid(mesh_material.albedo)) {
                        maps.albedo_map =
                                GetRenderer().AddTexture(mesh_material.albedo);
                    }
                    if (is_map_valid(mesh_material.normalMap)) {
                        maps.normal_map = GetRenderer().AddTexture(
                                mesh_material.normalMap);
                        albedo_only = false;
                    }
                    if (is_map_valid(mesh_material.ambientOcclusion)) {
                        maps.ambient_occlusion_map = GetRenderer().AddTexture(
                                mesh_material.ambientOcclusion);
                        albedo_only = false;
                    }
                    if (is_map_valid(mesh_material.roughness)) {
                        maps.roughness_map = GetRenderer().AddTexture(
                                mesh_material.roughness);
                        albedo_only = false;
                    }
                    if (is_map_valid(mesh_material.metallic)) {
                        material.metallic = 1.f;
                        maps.metallic_map = GetRenderer().AddTexture(
                                mesh_material.metallic);
                        albedo_only = false;
                    } else {
                        material.metallic = 0.f;
                    }
                    if (is_map_valid(mesh_material.reflectance)) {
                        maps.reflectance_map = GetRenderer().AddTexture(
                                mesh_material.reflectance);
                        albedo_only = false;
                    }
                    if (is_map_valid(mesh_material.clearCoat)) {
                        maps.clear_coat_map = GetRenderer().AddTexture(
                                mesh_material.clearCoat);
                        albedo_only = false;
                    }
                    if (is_map_valid(mesh_material.clearCoatRoughness)) {
                        maps.clear_coat_roughness_map =
                                GetRenderer().AddTexture(
                                        mesh_material.clearCoatRoughness);
                        albedo_only = false;
                    }
                    if (is_map_valid(mesh_material.anisotropy)) {
                        maps.anisotropy_map = GetRenderer().AddTexture(
                                mesh_material.anisotropy);
                        albedo_only = false;
                    }
                    impl_->SetMaterialsToCurrentSettings(GetRenderer(),
                                                         material, maps);
                    impl_->settings_.loaded_materials_[i] = material;
                }

                if ((mesh->HasVertexColors() && !MeshHasUniformColor(*mesh)) ||
                    (mesh->HasMaterials() && albedo_only)) {
                    selected_material = materials.unlit.handle;
                    num_unlit += 1;
                } else {
                    selected_material = materials.lit.handle;
                }
            } break;
            default:
                utility::LogWarning("Geometry type {} not supported!",
                                    (int)g->GetGeometryType());
                break;
        }

        auto g3 = std::static_pointer_cast<const geometry::Geometry3D>(g);
        auto handle = scene3d->AddGeometry(*g3, selected_material);
        bounds += scene3d->GetEntityBoundingBox(handle);
        impl_->geometry_handles_.push_back(handle);

        if (g->GetGeometryType() ==
            geometry::Geometry::GeometryType::PointCloud) {
            desc.point_clouds.push_back(handle);
            auto pcd = std::static_pointer_cast<const geometry::PointCloud>(g);
            if (num_point_cloud_points >
                MIN_POINT_CLOUD_POINTS_FOR_DECIMATION) {
                int sample_rate = num_point_cloud_points /
                                  (MIN_POINT_CLOUD_POINTS_FOR_DECIMATION / 2);
                auto small = pcd->UniformDownSample(sample_rate);
                handle = scene3d->AddGeometry(*small, selected_material);
                desc.fast_point_clouds.push_back(handle);
                impl_->geometry_handles_.push_back(handle);
            }
        } else {
            desc.meshes.push_back(handle);
        }
    }

    if (!geometries.empty()) {
        auto view_mode = impl_->scene_->GetView()->GetMode();
        if (view_mode == rendering::View::Mode::Normals) {
            impl_->SetMaterialType(Impl::Settings::NORMAL_MAP);
        } else if (view_mode == rendering::View::Mode::Depth) {
            impl_->SetMaterialType(Impl::Settings::DEPTH);
        } else {
            if (num_unlit == geometries.size()) {
                impl_->SetMaterialType(Impl::Settings::UNLIT);
            } else {
                impl_->SetMaterialType(Impl::Settings::LIT);
            }
        }

        if (num_point_clouds == geometries.size() &&
            !impl_->settings_.user_has_changed_lighting) {
            impl_->SetLightingProfile(GetRenderer(), POINT_CLOUD_PROFILE_NAME);
        }
        impl_->settings_.wgt_point_size->SetEnabled(num_point_clouds > 0);
    }

    if (!impl_->settings_.loaded_materials_.empty()) {
        if (impl_->settings_.loaded_materials_.size() == 1) {
            auto color = impl_->settings_.loaded_materials_.begin()
                                 ->second.base_color;
            impl_->settings_.wgt_material_color->SetValue(color.x(), color.y(),
                                                          color.z());
        }
        int resetIdx = impl_->settings_.wgt_prefab_material->AddItem(
                MATERIAL_FROM_FILE_NAME.c_str());
        impl_->settings_.wgt_prefab_material->SetSelectedIndex(resetIdx);
        impl_->settings_.wgt_prefab_material->ChangeItem(
                (DEFAULT_MATERIAL_NAME + " [default]").c_str(),
                DEFAULT_MATERIAL_NAME.c_str());
    } else {
        impl_->settings_.wgt_prefab_material->ChangeItem(
                DEFAULT_MATERIAL_NAME.c_str(),
                (DEFAULT_MATERIAL_NAME + " [default]").c_str());
    }

    // Add axes. Axes length should be the longer of the bounds extent
    // or 25% of the distance from the origin. The latter is necessary
    // so that the axis is big enough to be visible even if the object
    // is far from the origin. See caterpillar.ply from Tanks & Temples.
    auto axis_length = bounds.GetMaxExtent();
    if (axis_length < 0.001) {  // avoid div by zero errors in CreateAxes()
        axis_length = 1.0;
    }
    axis_length = std::max(axis_length, 0.25 * bounds.GetCenter().norm());
    auto axes = CreateAxes(axis_length);
    impl_->settings_.axes = scene3d->AddGeometry(*axes);
    scene3d->SetGeometryShadows(impl_->settings_.axes, false, false);
    scene3d->SetEntityEnabled(impl_->settings_.axes,
                              impl_->settings_.wgt_show_axes->IsChecked());
    desc.axes = impl_->settings_.axes;
    impl_->scene_->SetModel(desc);

    impl_->scene_->SetupCamera(60.0, bounds, bounds.GetCenter().cast<float>());
}

void GuiVisualizer::Layout(const gui::Theme &theme) {
    auto r = GetContentRect();
    const auto em = theme.font_size;
    impl_->scene_->SetFrame(r);

    // Draw help keys HUD in upper left
    const auto pref = impl_->help_keys_->CalcPreferredSize(theme);
    impl_->help_keys_->SetFrame(gui::Rect(0, r.y, pref.width, pref.height));
    impl_->help_keys_->Layout(theme);

    // Draw camera HUD in lower left
    const auto prefcam = impl_->help_camera_->CalcPreferredSize(theme);
    impl_->help_camera_->SetFrame(gui::Rect(0, r.height + r.y - prefcam.height,
                                            prefcam.width, prefcam.height));
    impl_->help_camera_->Layout(theme);

    // Settings in upper right
    const auto LIGHT_SETTINGS_WIDTH = 18 * em;
    auto light_settings_size =
            impl_->settings_.wgt_base->CalcPreferredSize(theme);
    gui::Rect lightSettingsRect(r.width - LIGHT_SETTINGS_WIDTH, r.y,
                                LIGHT_SETTINGS_WIDTH,
                                std::min(r.height, light_settings_size.height));
    impl_->settings_.wgt_base->SetFrame(lightSettingsRect);

    Super::Layout(theme);
}

bool GuiVisualizer::SetIBL(const char *path) {
    auto result = impl_->SetIBL(GetRenderer(), path);
    PostRedraw();
    return result;
}

void GuiVisualizer::LoadGeometry(const std::string &path) {
    auto progressbar = std::make_shared<gui::ProgressBar>();
    gui::Application::GetInstance().PostToMainThread(this, [this, path,
                                                            progressbar]() {
        auto &theme = GetTheme();
        auto loading_dlg = std::make_shared<gui::Dialog>("Loading");
        auto vert =
                std::make_shared<gui::Vert>(0, gui::Margins(theme.font_size));
        auto loading_text = std::string("Loading ") + path;
        vert->AddChild(std::make_shared<gui::Label>(loading_text.c_str()));
        vert->AddFixed(theme.font_size);
        vert->AddChild(progressbar);
        loading_dlg->AddChild(vert);
        ShowDialog(loading_dlg);
    });

    gui::Application::GetInstance().RunInThread([this, path, progressbar]() {
        auto UpdateProgress = [this, progressbar](float value) {
            gui::Application::GetInstance().PostToMainThread(
                    this,
                    [progressbar, value]() { progressbar->SetValue(value); });
        };

        auto geometry = std::shared_ptr<geometry::Geometry3D>();

        auto geometry_type = io::ReadFileGeometryType(path);

        auto mesh = std::make_shared<geometry::TriangleMesh>();
        bool mesh_success = false;
        if (geometry_type & io::CONTAINS_TRIANGLES) {
            try {
                mesh_success = io::ReadTriangleMesh(path, *mesh);
            } catch (...) {
                mesh_success = false;
            }
        }
        if (mesh_success) {
            if (mesh->triangles_.size() == 0) {
                utility::LogWarning(
                        "Contains 0 triangles, will read as point cloud");
                mesh.reset();
            } else {
                UpdateProgress(0.5);
                mesh->ComputeVertexNormals();
                if (mesh->vertex_colors_.empty()) {
                    mesh->PaintUniformColor({1, 1, 1});
                }
                UpdateProgress(0.666);
                geometry = mesh;
            }
            // Make sure the mesh has texture coordinates
            if (!mesh->HasTriangleUvs()) {
                mesh->triangle_uvs_.resize(mesh->triangles_.size() * 3,
                                           {0.0, 0.0});
            }
        } else {
            // LogError throws an exception, which we don't want, because this
            // might be a point cloud.
            utility::LogInfo("{} appears to be a point cloud", path.c_str());
            mesh.reset();
        }

        if (!geometry) {
            auto cloud = std::make_shared<geometry::PointCloud>();
            bool success = false;
            const float ioProgressAmount = 0.5f;
            try {
                io::ReadPointCloudOption opt;
                opt.update_progress = [ioProgressAmount,
                                       UpdateProgress](double percent) -> bool {
                    UpdateProgress(ioProgressAmount * percent / 100.0);
                    return true;
                };
                success = io::ReadPointCloud(path, *cloud, opt);
            } catch (...) {
                success = false;
            }
            if (success) {
                utility::LogInfo("Successfully read {}", path.c_str());
                UpdateProgress(ioProgressAmount);
                if (!cloud->HasNormals()) {
                    cloud->EstimateNormals();
                }
                UpdateProgress(0.666);
                cloud->NormalizeNormals();
                UpdateProgress(0.75);
                geometry = cloud;
            } else {
                utility::LogWarning("Failed to read points {}", path.c_str());
                cloud.reset();
            }
        }

        if (geometry) {
            gui::Application::GetInstance().PostToMainThread(
                    this, [this, geometry]() {
                        SetGeometry({geometry});
                        CloseDialog();
                    });
        } else {
            gui::Application::GetInstance().PostToMainThread(this, [this,
                                                                    path]() {
                CloseDialog();
                auto msg = std::string("Could not load '") + path + "'.";
                ShowMessageBox("Error", msg.c_str());
            });
        }
    });
}

void GuiVisualizer::ExportCurrentImage(int width,
                                       int height,
                                       const std::string &path) {
    GetRenderer().RenderToImage(
            width, height, impl_->scene_->GetView(), impl_->scene_->GetScene(),
            [this, path](std::shared_ptr<geometry::Image> image) mutable {
                if (!io::WriteImage(path, *image)) {
                    this->ShowMessageBox(
                            "Error", (std::string("Could not write image to ") +
                                      path + ".")
                                             .c_str());
                }
            });
}

void GuiVisualizer::OnMenuItemSelected(gui::Menu::ItemId item_id) {
    auto menu_id = MenuId(item_id);
    switch (menu_id) {
        case FILE_OPEN: {
            auto dlg = std::make_shared<gui::FileDialog>(
                    gui::FileDialog::Mode::OPEN, "Open Geometry", GetTheme());
            dlg->AddFilter(".ply .stl .obj .off .gltf .glb",
                           "Triangle mesh files (.ply, .stl, .obj, .off, "
                           ".gltf, .glb)");
            dlg->AddFilter(".xyz .xyzn .xyzrgb .ply .pcd .pts",
                           "Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, "
                           ".pcd, .pts)");
            dlg->AddFilter(".ply", "Polygon files (.ply)");
            dlg->AddFilter(".stl", "Stereolithography files (.stl)");
            dlg->AddFilter(".obj", "Wavefront OBJ files (.obj)");
            dlg->AddFilter(".off", "Object file format (.off)");
            dlg->AddFilter(".gltf", "OpenGL transfer files (.gltf)");
            dlg->AddFilter(".glb", "OpenGL binary transfer files (.glb)");
            dlg->AddFilter(".xyz", "ASCII point cloud files (.xyz)");
            dlg->AddFilter(".xyzn", "ASCII point cloud with normals (.xyzn)");
            dlg->AddFilter(".xyzrgb",
                           "ASCII point cloud files with colors (.xyzrgb)");
            dlg->AddFilter(".pcd", "Point Cloud Data files (.pcd)");
            dlg->AddFilter(".pts", "3D Points files (.pts)");
            dlg->AddFilter("", "All files");
            dlg->SetOnCancel([this]() { this->CloseDialog(); });
            dlg->SetOnDone([this](const char *path) {
                this->CloseDialog();
                OnDragDropped(path);
            });
            ShowDialog(dlg);
            break;
        }
        case FILE_EXPORT_RGB: {
            auto dlg = std::make_shared<gui::FileDialog>(
                    gui::FileDialog::Mode::SAVE, "Save File", GetTheme());
            dlg->AddFilter(".png", "PNG images (.png)");
            dlg->AddFilter("", "All files");
            dlg->SetOnCancel([this]() { this->CloseDialog(); });
            dlg->SetOnDone([this](const char *path) {
                this->CloseDialog();
                auto r = GetContentRect();
                this->ExportCurrentImage(r.width, r.height, path);
            });
            ShowDialog(dlg);
            break;
        }
        case FILE_QUIT:
            gui::Application::GetInstance().Quit();
            break;
        case SETTINGS_LIGHT_AND_MATERIALS: {
            auto visibility = !impl_->settings_.wgt_base->IsVisible();
            impl_->settings_.wgt_base->SetVisible(visibility);
            auto menubar = gui::Application::GetInstance().GetMenubar();
            menubar->SetChecked(SETTINGS_LIGHT_AND_MATERIALS, visibility);

            // We need relayout because materials settings pos depends on light
            // settings visibility
            Layout(GetTheme());

            break;
        }
        case HELP_KEYS: {
            bool is_visible = !impl_->help_keys_->IsVisible();
            impl_->help_keys_->SetVisible(is_visible);
            auto menubar = gui::Application::GetInstance().GetMenubar();
            menubar->SetChecked(HELP_KEYS, is_visible);
            break;
        }
        case HELP_CAMERA: {
            bool is_visible = !impl_->help_camera_->IsVisible();
            impl_->help_camera_->SetVisible(is_visible);
            auto menubar = gui::Application::GetInstance().GetMenubar();
            menubar->SetChecked(HELP_CAMERA, is_visible);
            if (is_visible) {
                impl_->scene_->SetCameraChangedCallback([this](rendering::Camera
                                                                       *cam) {
                    auto children = this->impl_->help_camera_->GetChildren();
                    auto set_text = [](const Eigen::Vector3f &v,
                                       std::shared_ptr<gui::Widget> label) {
                        auto l = std::dynamic_pointer_cast<gui::Label>(label);
                        l->SetText(fmt::format("[{:.2f} {:.2f} "
                                               "{:.2f}]",
                                               v.x(), v.y(), v.z())
                                           .c_str());
                    };
                    set_text(cam->GetPosition(), children[1]);
                    set_text(cam->GetForwardVector(), children[3]);
                    set_text(cam->GetLeftVector(), children[5]);
                    set_text(cam->GetUpVector(), children[7]);
                    this->SetNeedsLayout();
                });
            } else {
                impl_->scene_->SetCameraChangedCallback(
                        std::function<void(rendering::Camera *)>());
            }
            break;
        }
        case HELP_ABOUT: {
            auto dlg = CreateAboutDialog(this);
            ShowDialog(dlg);
            break;
        }
        case HELP_CONTACT: {
            auto dlg = CreateContactDialog(this);
            ShowDialog(dlg);
            break;
        }
    }
}

void GuiVisualizer::OnDragDropped(const char *path) {
    auto title = std::string("Open3D - ") + path;
#if LOAD_IN_NEW_WINDOW
    auto frame = this->GetFrame();
    std::vector<std::shared_ptr<const geometry::Geometry>> nothing;
    auto vis = std::make_shared<GuiVisualizer>(nothing, title.c_str(),
                                               frame.width, frame.height,
                                               frame.x + 20, frame.y + 20);
    gui::Application::GetInstance().AddWindow(vis);
#else
    this->SetTitle(title);
    auto vis = this;
#endif  // LOAD_IN_NEW_WINDOW
    vis->LoadGeometry(path);
}

}  // namespace visualization
}  // namespace open3d
