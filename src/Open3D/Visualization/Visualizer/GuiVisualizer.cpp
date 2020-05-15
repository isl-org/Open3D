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

#include "GuiVisualizer.h"

#include "Open3D/GUI/Application.h"
#include "Open3D/GUI/Button.h"
#include "Open3D/GUI/Checkbox.h"
#include "Open3D/GUI/Color.h"
#include "Open3D/GUI/ColorEdit.h"
#include "Open3D/GUI/Combobox.h"
#include "Open3D/GUI/Dialog.h"
#include "Open3D/GUI/FileDialog.h"
#include "Open3D/GUI/Label.h"
#include "Open3D/GUI/Layout.h"
#include "Open3D/GUI/SceneWidget.h"
#include "Open3D/GUI/Slider.h"
#include "Open3D/GUI/Theme.h"
#include "Open3D/GUI/VectorEdit.h"
#include "Open3D/Geometry/BoundingVolume.h"
#include "Open3D/Geometry/Image.h"
#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Geometry/TriangleMesh.h"
#include "Open3D/IO/ClassIO/FileFormatIO.h"
#include "Open3D/IO/ClassIO/ImageIO.h"
#include "Open3D/IO/ClassIO/PointCloudIO.h"
#include "Open3D/IO/ClassIO/TriangleMeshIO.h"
#include "Open3D/Open3DConfig.h"
#include "Open3D/Utility/Console.h"
#include "Open3D/Utility/FileSystem.h"
#include "Open3D/Visualization/Rendering/Camera.h"
#include "Open3D/Visualization/Rendering/Filament/FilamentResourceManager.h"
#include "Open3D/Visualization/Rendering/RenderToBuffer.h"
#include "Open3D/Visualization/Rendering/RendererStructs.h"
#include "Open3D/Visualization/Rendering/Scene.h"

#define LOAD_IN_NEW_WINDOW 0

namespace open3d {
namespace visualization {

namespace {

std::shared_ptr<gui::Dialog> createAboutDialog(gui::Window *window) {
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

    gui::Margins margins(theme.fontSize);
    auto layout = std::make_shared<gui::Vert>(0, margins);
    layout->AddChild(gui::Horiz::MakeCentered(title));
    layout->AddFixed(theme.fontSize);
    layout->AddChild(text);
    layout->AddFixed(theme.fontSize);
    layout->AddChild(gui::Horiz::MakeCentered(ok));
    dlg->AddChild(layout);

    return dlg;
}

std::shared_ptr<gui::VGrid> createHelpDisplay(gui::Window *window) {
    auto &theme = window->GetTheme();

    gui::Margins margins(theme.fontSize);
    auto layout = std::make_shared<gui::VGrid>(2, 0, margins);
    layout->SetBackgroundColor(gui::Color(0, 0, 0, 0.5));

    auto addLabel = [layout](const char *text) {
        auto label = std::make_shared<gui::Label>(text);
        label->SetTextColor(gui::Color(1, 1, 1));
        layout->AddChild(label);
    };
    auto addRow = [layout, &addLabel](const char *left, const char *right) {
        addLabel(left);
        addLabel(right);
    };

    addRow("Arcball mode", " ");
    addRow("Left-drag", "Rotate camera");
    addRow("Shift + left-drag", "Forward/backward");

#if defined(__APPLE__)
    addLabel("Cmd + left-drag");
#else
    addLabel("Ctrl + left-drag");
#endif  // __APPLE__
    addLabel("Pan camera");

#if defined(__APPLE__)
    addLabel("Opt + left-drag (up/down)  ");
#else
    addLabel("Win + left-drag (up/down)  ");
#endif  // __APPLE__
    addLabel("Rotate around forward axis");

    // GNOME3 uses Win/Meta as a shortcut to move windows around, so we
    // need another way to rotate around the forward axis.
    addLabel("Ctrl + Shift + left-drag");
    addLabel("Rotate around forward axis");

#if defined(__APPLE__)
    addLabel("Ctrl + left-drag");
#else
    addLabel("Alt + left-drag");
#endif  // __APPLE__
    addLabel("Rotate directional light");

    addRow("Right-drag", "Pan camera");
    addRow("Middle-drag", "Rotate directional light");
    addRow("Wheel", "Forward/backward");
    addRow("Shift + Wheel", "Change field of view");
    addRow("", "");

    addRow("Fly mode", " ");
    addRow("Left-drag", "Rotate camera");
#if defined(__APPLE__)
    addLabel("Opt + left-drag");
#else
    addLabel("Win + left-drag");
#endif  // __APPLE__
    addLabel("Rotate around forward axis");
    addRow("W", "Forward");
    addRow("S", "Backward");
    addRow("A", "Step left");
    addRow("D", "Step right");
    addRow("Q", "Step up");
    addRow("Z", "Step down");
    addRow("E", "Roll left");
    addRow("R", "Roll right");
    addRow("Up", "Look up");
    addRow("Down", "Look down");
    addRow("Left", "Look left");
    addRow("Right", "Look right");

    return layout;
}

std::shared_ptr<gui::VGrid> createCameraDisplay(gui::Window *window) {
    auto &theme = window->GetTheme();

    gui::Margins margins(theme.fontSize);
    auto layout = std::make_shared<gui::VGrid>(2, 0, margins);
    layout->SetBackgroundColor(gui::Color(0, 0, 0, 0.5));

    auto addLabel = [layout](const char *text) {
        auto label = std::make_shared<gui::Label>(text);
        label->SetTextColor(gui::Color(1, 1, 1));
        layout->AddChild(label);
    };
    auto addRow = [layout, &addLabel](const char *left, const char *right) {
        addLabel(left);
        addLabel(right);
    };

    addRow("Position:", "[0 0 0]");
    addRow("Forward:", "[0 0 0]");
    addRow("Left:", "[0 0 0]");
    addRow("Up:", "[0 0 0]");

    return layout;
}

std::shared_ptr<gui::Dialog> createContactDialog(gui::Window *window) {
    auto &theme = window->GetTheme();
    auto em = theme.fontSize;
    auto dlg = std::make_shared<gui::Dialog>("Contact Us");

    auto title = std::make_shared<gui::Label>("Contact Us");
    auto leftCol = std::make_shared<gui::Label>(
            "Web site:\n"
            "Code:\n"
            "Mailing list:\n"
            "Discord channel:");
    auto rightCol = std::make_shared<gui::Label>(
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
    columns->AddChild(leftCol);
    columns->AddChild(rightCol);
    layout->AddChild(columns);

    layout->AddFixed(em);
    layout->AddChild(gui::Horiz::MakeCentered(ok));
    dlg->AddChild(layout);

    return dlg;
}

std::shared_ptr<geometry::TriangleMesh> CreateAxes(double axisLength) {
    const double sphereRadius = 0.005 * axisLength;
    const double cylRadius = 0.0025 * axisLength;
    const double coneRadius = 0.0075 * axisLength;
    const double cylHeight = 0.975 * axisLength;
    const double coneHeight = 0.025 * axisLength;

    auto mesh_frame = geometry::TriangleMesh::CreateSphere(sphereRadius);
    mesh_frame->ComputeVertexNormals();
    mesh_frame->PaintUniformColor(Eigen::Vector3d(0.5, 0.5, 0.5));

    std::shared_ptr<geometry::TriangleMesh> mesh_arrow;
    Eigen::Matrix4d transformation;

    mesh_arrow = geometry::TriangleMesh::CreateArrow(cylRadius, coneRadius,
                                                     cylHeight, coneHeight);
    mesh_arrow->ComputeVertexNormals();
    mesh_arrow->PaintUniformColor(Eigen::Vector3d(1.0, 0.0, 0.0));
    transformation << 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1;
    mesh_arrow->Transform(transformation);
    *mesh_frame += *mesh_arrow;

    mesh_arrow = geometry::TriangleMesh::CreateArrow(cylRadius, coneRadius,
                                                     cylHeight, coneHeight);
    mesh_arrow->ComputeVertexNormals();
    mesh_arrow->PaintUniformColor(Eigen::Vector3d(0.0, 1.0, 0.0));
    transformation << 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1;
    mesh_arrow->Transform(transformation);
    *mesh_frame += *mesh_arrow;

    mesh_arrow = geometry::TriangleMesh::CreateArrow(cylRadius, coneRadius,
                                                     cylHeight, coneHeight);
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
    static const double kSqEpsilon = Eigen::Vector3d(e, e, e).squaredNorm();
    const auto &color = colors[0];

    for (const auto &c : colors) {
        if ((color - c).squaredNorm() > kSqEpsilon) {
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
        return gui::Size(theme.fontSize * 5, h);
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
    explicit SmallButton(const char *title) : Button(title) {}

    gui::Size CalcPreferredSize(const gui::Theme &theme) const override {
        auto em = theme.fontSize;
        auto size = Super::CalcPreferredSize(theme);
        return gui::Size(size.width - em, 1.2 * em);
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
    double iblIntensity;
    double sunIntensity;
    Eigen::Vector3f sunDir;
    Eigen::Vector3f sunColor = {1.0f, 1.0f, 1.0f};
    Scene::Transform iblRotation = Scene::Transform::Identity();
    bool iblEnabled = true;
    bool useDefaultIBL = false;
    bool sunEnabled = true;
};

static const std::string kDefaultIBL = "default";
static const std::string kDefaultMaterialName = "Polished ceramic [default]";
static const std::string kPointCloudProfileName = "Cloudy day (no direct sun)";
static const bool kDefaultShowSkybox = false;
static const bool kDefaultShowAxes = false;

static const std::vector<LightingProfile> gLightingProfiles = {
        {.name = "Bright day with sun at +Y [default]",
         .iblIntensity = 45000,
         .sunIntensity = 45000,
         .sunDir = {0.577f, -0.577f, -0.577f}},
        {.name = "Bright day with sun at -Y",
         .iblIntensity = 45000,
         .sunIntensity = 45000,
         .sunDir = {0.577f, 0.577f, 0.577f},
         .sunColor = {1.0f, 1.0f, 1.0f},
         .iblRotation = Scene::Transform(
                 Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitX()))},
        {.name = "Bright day with sun at +Z",
         .iblIntensity = 45000,
         .sunIntensity = 45000,
         .sunDir = {0.577f, 0.577f, -0.577f}},
        {.name = "Less bright day with sun at +Y",
         .iblIntensity = 35000,
         .sunIntensity = 50000,
         .sunDir = {0.577f, -0.577f, -0.577f}},
        {.name = "Less bright day with sun at -Y",
         .iblIntensity = 35000,
         .sunIntensity = 50000,
         .sunDir = {0.577f, 0.577f, 0.577f},
         .sunColor = {1.0f, 1.0f, 1.0f},
         .iblRotation = Scene::Transform(
                 Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitX()))},
        {.name = "Less bright day with sun at +Z",
         .iblIntensity = 35000,
         .sunIntensity = 50000,
         .sunDir = {0.577f, 0.577f, -0.577f}},
        {.name = kPointCloudProfileName,
         .iblIntensity = 60000,
         .sunIntensity = 50000,
         .sunDir = {0.577f, -0.577f, -0.577f},
         .sunColor = {1.0f, 1.0f, 1.0f},
         .iblRotation = Scene::Transform::Identity(),
         .iblEnabled = true,
         .useDefaultIBL = true,
         .sunEnabled = false}};

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
    std::vector<visualization::GeometryHandle> geometryHandles;

    std::shared_ptr<gui::SceneWidget> scene;
    std::shared_ptr<gui::VGrid> helpKeys;
    std::shared_ptr<gui::VGrid> helpCamera;

    struct LitMaterial {
        visualization::MaterialInstanceHandle handle;
        Eigen::Vector3f baseColor = {0.9f, 0.9f, 0.9f};
        float metallic = 0.f;
        float roughness = 0.7;
        float reflectance = 0.5f;
        float clearCoat = 0.2f;
        float clearCoatRoughness = 0.2f;
        float anisotropy = 0.f;
        float pointSize = 5.f;
    };

    struct UnlitMaterial {
        visualization::MaterialInstanceHandle handle;
        // The base color should NOT be {1, 1, 1}, because then the
        // model will be invisible against the default white background.
        Eigen::Vector3f baseColor = {0.9f, 0.9f, 0.9f};
        float pointSize = 5.f;
    };

    struct TextureMaps {
        TextureHandle albedoMap = FilamentResourceManager::kDefaultTexture;
        TextureHandle normalMap = FilamentResourceManager::kDefaultNormalMap;
        TextureHandle ambientOcclusionMap =
                FilamentResourceManager::kDefaultTexture;
        TextureHandle roughnessMap = FilamentResourceManager::kDefaultTexture;
        TextureHandle metallicMap = FilamentResourceManager::kDefaultTexture;
        TextureHandle reflectanceMap = FilamentResourceManager::kDefaultTexture;
        TextureHandle clearCoatMap = FilamentResourceManager::kDefaultTexture;
        TextureHandle clearCoatRoughnessMap =
                FilamentResourceManager::kDefaultTexture;
        TextureHandle anisotropyMap = FilamentResourceManager::kDefaultTexture;
    };

    struct Materials {
        LitMaterial lit;
        UnlitMaterial unlit;
        TextureMaps maps;
    };

    std::map<std::string, LitMaterial> prefabMaterials = {
            {kDefaultMaterialName, {}},
            {"Metal (rougher)",
             {visualization::MaterialInstanceHandle::kBad,
              {1.0f, 1.0f, 1.0f},
              1.0f,
              0.5f,
              0.9f,
              0.0f,
              0.0f,
              0.0f,
              3.0f}},
            {"Metal (smoother)",
             {visualization::MaterialInstanceHandle::kBad,
              {1.0f, 1.0f, 1.0f},
              1.0f,
              0.3f,
              0.9f,
              0.0f,
              0.0f,
              0.0f,
              3.0f}},
            {"Plastic",
             {visualization::MaterialInstanceHandle::kBad,
              {1.0f, 1.0f, 1.0f},
              0.0f,
              0.5f,
              0.5f,
              0.5f,
              0.2f,
              0.0f,
              3.0f}},
            {"Glazed ceramic",
             {visualization::MaterialInstanceHandle::kBad,
              {1.0f, 1.0f, 1.0f},
              0.0f,
              0.5f,
              0.9f,
              1.0f,
              0.1f,
              0.0f,
              3.0f}},
            {"Clay",
             {visualization::MaterialInstanceHandle::kBad,
              {0.7725f, 0.7725f, 0.7725f},
              0.0f,
              1.0f,
              0.5f,
              0.1f,
              0.287f,
              0.0f,
              3.0f}},
    };

    visualization::MaterialHandle hLitMaterial;
    visualization::MaterialHandle hUnlitMaterial;

    struct Settings {
        visualization::IndirectLightHandle hIbl;
        visualization::SkyboxHandle hSky;
        visualization::TextureHandle hSkyTexture;
        visualization::LightHandle hDirectionalLight;
        visualization::GeometryHandle hAxes;

        std::shared_ptr<gui::Vert> wgtBase;
        std::shared_ptr<gui::Checkbox> wgtShowAxes;
        std::shared_ptr<gui::ColorEdit> wgtBGColor;
        std::shared_ptr<gui::Button> wgtMouseArcball;
        std::shared_ptr<gui::Button> wgtMouseFly;
        std::shared_ptr<gui::Button> wgtMouseSun;
        std::shared_ptr<gui::Button> wgtMouseIBL;
        std::shared_ptr<gui::Button> wgtMouseModel;
        std::shared_ptr<gui::Combobox> wgtLightingProfile;
        std::shared_ptr<gui::CollapsableVert> wgtAdvanced;
        std::shared_ptr<gui::Checkbox> wgtIBLEnabled;
        std::shared_ptr<gui::Checkbox> wgtSkyEnabled;
        std::shared_ptr<gui::Checkbox> wgtDirectionalEnabled;
        std::shared_ptr<gui::Combobox> wgtIBLs;
        std::shared_ptr<gui::Button> wgtLoadSky;
        std::shared_ptr<gui::Slider> wgtIBLIntensity;
        std::shared_ptr<gui::Slider> wgtSunIntensity;
        std::shared_ptr<gui::VectorEdit> wgtSunDir;
        std::shared_ptr<gui::ColorEdit> wgtSunColor;

        enum MaterialType {
            LIT = 0,
            UNLIT,
            NORMAL_MAP,
            DEPTH,
        };

        MaterialType selectedType = LIT;
        Materials currentMaterials;
        std::shared_ptr<gui::Combobox> wgtMaterialType;

        std::shared_ptr<gui::Combobox> wgtPrefabMaterial;
        std::shared_ptr<gui::ColorEdit> wgtMaterialColor;
        std::shared_ptr<gui::Button> wgtResetMaterialColor;
        std::shared_ptr<gui::Slider> wgtPointSize;

        bool userHasChangedColor = false;
        bool userHasChangedLighting = false;

        void SetCustomProfile() {
            wgtLightingProfile->SetSelectedIndex(gLightingProfiles.size());
            userHasChangedLighting = true;
        }
    } settings;

    void SetMaterialsToDefault(visualization::Renderer &renderer) {
        Materials defaults;
        if (this->settings.userHasChangedColor) {
            defaults.unlit.baseColor =
                    this->settings.currentMaterials.unlit.baseColor;
            defaults.lit.baseColor =
                    this->settings.currentMaterials.lit.baseColor;
        }
        defaults.maps.albedoMap = FilamentResourceManager::kDefaultTexture;
        defaults.maps.normalMap = FilamentResourceManager::kDefaultNormalMap;
        defaults.maps.ambientOcclusionMap =
                FilamentResourceManager::kDefaultTexture;
        defaults.maps.roughnessMap = FilamentResourceManager::kDefaultTexture;
        defaults.maps.metallicMap = FilamentResourceManager::kDefaultTexture;
        defaults.maps.reflectanceMap = FilamentResourceManager::kDefaultTexture;
        defaults.maps.clearCoatMap = FilamentResourceManager::kDefaultTexture;
        defaults.maps.clearCoatRoughnessMap =
                FilamentResourceManager::kDefaultTexture;
        defaults.maps.anisotropyMap = FilamentResourceManager::kDefaultTexture;
        this->settings.currentMaterials = defaults;

        auto hLit = renderer.AddMaterialInstance(this->hLitMaterial);
        this->settings.currentMaterials.lit.handle =
                renderer.ModifyMaterial(hLit)
                        .SetColor("baseColor", defaults.lit.baseColor)
                        .SetParameter("baseRoughness", defaults.lit.roughness)
                        .SetParameter("baseMetallic", defaults.lit.metallic)
                        .SetParameter("reflectance", defaults.lit.reflectance)
                        .SetParameter("clearCoat", defaults.lit.clearCoat)
                        .SetParameter("clearCoatRoughness",
                                      defaults.lit.clearCoatRoughness)
                        .SetParameter("anisotropy", defaults.lit.anisotropy)
                        .SetParameter("pointSize", defaults.lit.pointSize)
                        .SetTexture("albedo", defaults.maps.albedoMap,
                                    TextureSamplerParameters::Pretty())
                        .SetTexture("albedo", defaults.maps.albedoMap,
                                    TextureSamplerParameters::Pretty())
                        .SetTexture("normalMap", defaults.maps.normalMap,
                                    TextureSamplerParameters::Pretty())
                        .SetTexture("ambientOcclusionMap",
                                    defaults.maps.ambientOcclusionMap,
                                    TextureSamplerParameters::Pretty())
                        .SetTexture("roughnessMap", defaults.maps.roughnessMap,
                                    TextureSamplerParameters::Pretty())
                        .SetTexture("metallicMap", defaults.maps.metallicMap,
                                    TextureSamplerParameters::Pretty())
                        .SetTexture("reflectanceMap",
                                    defaults.maps.reflectanceMap,
                                    TextureSamplerParameters::Pretty())
                        .SetTexture("clearCoatMap", defaults.maps.clearCoatMap,
                                    TextureSamplerParameters::Pretty())
                        .SetTexture("clearCoatRoughnessMap",
                                    defaults.maps.clearCoatRoughnessMap,
                                    TextureSamplerParameters::Pretty())
                        .SetTexture("anisotropyMap",
                                    defaults.maps.anisotropyMap,
                                    TextureSamplerParameters::Pretty())
                        .Finish();

        auto hUnlit = renderer.AddMaterialInstance(this->hUnlitMaterial);
        this->settings.currentMaterials.unlit.handle =
                renderer.ModifyMaterial(hUnlit)
                        .SetColor("baseColor", defaults.unlit.baseColor)
                        .SetParameter("pointSize", defaults.unlit.pointSize)
                        .SetTexture("albedo", defaults.maps.albedoMap,
                                    TextureSamplerParameters::Pretty())
                        .Finish();

        if (this->settings.wgtPrefabMaterial) {
            this->settings.wgtPrefabMaterial->SetSelectedValue(
                    kDefaultMaterialName.c_str());
        }
        if (this->settings.wgtMaterialColor) {
            Eigen::Vector3f color =
                    (this->settings.selectedType == Settings::MaterialType::LIT
                             ? defaults.lit.baseColor
                             : defaults.unlit.baseColor);
            this->settings.wgtMaterialColor->SetValue(color.x(), color.y(),
                                                      color.z());
        }
    }

    void SetMaterialsToCurrentSettings(visualization::Renderer &renderer,
                                       LitMaterial material,
                                       TextureMaps maps) {
        // Update the material settings
        this->settings.currentMaterials.lit.baseColor = material.baseColor;
        this->settings.currentMaterials.lit.roughness = material.roughness;
        this->settings.currentMaterials.lit.metallic = material.metallic;
        this->settings.currentMaterials.unlit.baseColor = material.baseColor;

        // Update maps
        this->settings.currentMaterials.maps = maps;

        // update materials
        this->settings.currentMaterials.lit.handle =
                renderer.ModifyMaterial(
                                this->settings.currentMaterials.lit.handle)
                        .SetColor("baseColor", material.baseColor)
                        .SetParameter("baseRoughness", material.roughness)
                        .SetParameter("baseMetallic", material.metallic)
                        .SetTexture("albedo", maps.albedoMap,
                                    TextureSamplerParameters::Pretty())
                        .SetTexture("normalMap", maps.normalMap,
                                    TextureSamplerParameters::Pretty())
                        .SetTexture("ambientOcclusionMap",
                                    maps.ambientOcclusionMap,
                                    TextureSamplerParameters::Pretty())
                        .SetTexture("roughnessMap", maps.roughnessMap,
                                    TextureSamplerParameters::Pretty())
                        .SetTexture("metallicMap", maps.metallicMap,
                                    TextureSamplerParameters::Pretty())
                        .SetTexture("reflectanceMap", maps.reflectanceMap,
                                    TextureSamplerParameters::Pretty())
                        .SetTexture("clearCoatMap", maps.clearCoatMap,
                                    TextureSamplerParameters::Pretty())
                        .SetTexture("clearCoatRoughnessMap",
                                    maps.clearCoatRoughnessMap,
                                    TextureSamplerParameters::Pretty())
                        .SetTexture("anisotropyMap", maps.anisotropyMap,
                                    TextureSamplerParameters::Pretty())
                        .Finish();
        this->settings.currentMaterials.unlit.handle =
                renderer.ModifyMaterial(
                                this->settings.currentMaterials.unlit.handle)
                        .SetColor("baseColor", material.baseColor)
                        .SetTexture("albedo", maps.albedoMap,
                                    TextureSamplerParameters::Pretty())
                        .Finish();
    }

    void SetMaterialType(Impl::Settings::MaterialType type) {
        using MaterialType = Impl::Settings::MaterialType;
        using ViewMode = visualization::View::Mode;

        auto renderScene = scene->GetScene();
        auto view = scene->GetView();
        this->settings.selectedType = type;
        this->settings.wgtMaterialType->SetSelectedIndex(int(type));

        bool isLit = (type == MaterialType::LIT);
        this->settings.wgtPrefabMaterial->SetEnabled(isLit);

        switch (type) {
            case MaterialType::LIT: {
                view->SetMode(ViewMode::Color);
                for (const auto &handle : this->geometryHandles) {
                    auto mat = this->settings.currentMaterials.lit.handle;
                    renderScene->AssignMaterial(handle, mat);
                }
                this->settings.wgtMaterialColor->SetEnabled(true);
                gui::Color color(
                        this->settings.currentMaterials.lit.baseColor.x(),
                        this->settings.currentMaterials.lit.baseColor.y(),
                        this->settings.currentMaterials.lit.baseColor.z());
                this->settings.wgtMaterialColor->SetValue(color);
                this->settings.wgtResetMaterialColor->SetEnabled(
                        this->settings.userHasChangedColor);
                break;
            }
            case MaterialType::UNLIT: {
                view->SetMode(ViewMode::Color);
                for (const auto &handle : this->geometryHandles) {
                    auto mat = this->settings.currentMaterials.unlit.handle;
                    renderScene->AssignMaterial(handle, mat);
                }
                this->settings.wgtMaterialColor->SetEnabled(true);
                gui::Color color(
                        this->settings.currentMaterials.unlit.baseColor.x(),
                        this->settings.currentMaterials.unlit.baseColor.y(),
                        this->settings.currentMaterials.unlit.baseColor.z());
                this->settings.wgtMaterialColor->SetValue(color);
                this->settings.wgtResetMaterialColor->SetEnabled(
                        this->settings.userHasChangedColor);
                break;
            }
            case MaterialType::NORMAL_MAP:
                view->SetMode(ViewMode::Normals);
                this->settings.wgtMaterialColor->SetEnabled(false);
                this->settings.wgtResetMaterialColor->SetEnabled(false);
                break;
            case MaterialType::DEPTH:
                view->SetMode(ViewMode::Depth);
                this->settings.wgtMaterialColor->SetEnabled(false);
                this->settings.wgtResetMaterialColor->SetEnabled(false);
                break;
        }
    }

    visualization::MaterialInstanceHandle CreateUnlitMaterial(
            visualization::Renderer &renderer,
            visualization::MaterialInstanceHandle mat) {
        auto color = settings.wgtMaterialColor->GetValue();
        Eigen::Vector3f color3(color.GetRed(), color.GetGreen(),
                               color.GetBlue());
        float pointSize = settings.wgtPointSize->GetDoubleValue();
        return renderer.ModifyMaterial(mat)
                .SetColor("baseColor", color3)
                .SetParameter("pointSize", pointSize)
                .Finish();
    }

    visualization::MaterialInstanceHandle CreateLitMaterial(
            visualization::Renderer &renderer,
            visualization::MaterialInstanceHandle mat,
            const LitMaterial &prefab) {
        Eigen::Vector3f color;
        if (settings.userHasChangedColor) {
            auto c = settings.wgtMaterialColor->GetValue();
            color = Eigen::Vector3f(c.GetRed(), c.GetGreen(), c.GetBlue());
        } else {
            color = prefab.baseColor;
        }
        float pointSize = settings.wgtPointSize->GetDoubleValue();
        return renderer.ModifyMaterial(mat)
                .SetColor("baseColor", color)
                .SetParameter("baseRoughness", prefab.roughness)
                .SetParameter("baseMetallic", prefab.metallic)
                .SetParameter("reflectance", prefab.reflectance)
                .SetParameter("clearCoat", prefab.clearCoat)
                .SetParameter("clearCoatRoughness", prefab.clearCoatRoughness)
                .SetParameter("anisotropy", prefab.anisotropy)
                .SetParameter("pointSize", pointSize)
                .Finish();
    }

    void SetMaterialByName(visualization::Renderer &renderer,
                           const std::string &name) {
        auto prefabIt = this->prefabMaterials.find(name);
        if (prefabIt != this->prefabMaterials.end()) {
            auto &prefab = prefabIt->second;
            if (!this->settings.userHasChangedColor) {
                this->settings.currentMaterials.lit.baseColor =
                        prefab.baseColor;
                this->settings.wgtMaterialColor->SetValue(prefab.baseColor.x(),
                                                          prefab.baseColor.y(),
                                                          prefab.baseColor.z());
            }
            auto mat = this->settings.currentMaterials.lit.handle;
            mat = this->CreateLitMaterial(renderer, mat, prefab);
            for (const auto &handle : this->geometryHandles) {
                this->scene->GetScene()->AssignMaterial(handle, mat);
            }
            this->settings.currentMaterials.lit.handle = mat;
        }
    }
    void SetLightingProfile(visualization::Renderer &renderer,
                            const std::string &name) {
        for (size_t i = 0; i < gLightingProfiles.size(); ++i) {
            if (gLightingProfiles[i].name == name) {
                SetLightingProfile(renderer, gLightingProfiles[i]);
                this->settings.wgtLightingProfile->SetSelectedValue(
                        name.c_str());
                return;
            }
        }
        utility::LogWarning("Could not find lighting profile '{}'", name);
    }

    void SetLightingProfile(visualization::Renderer &renderer,
                            const LightingProfile &profile) {
        auto *renderScene = this->scene->GetScene();
        if (profile.useDefaultIBL) {
            this->SetIBL(renderer, nullptr);
            this->settings.wgtIBLs->SetSelectedValue(kDefaultIBL.c_str());
        }
        if (profile.iblEnabled) {
            renderScene->SetIndirectLight(this->settings.hIbl);
        } else {
            renderScene->SetIndirectLight(IndirectLightHandle());
        }
        renderScene->SetIndirectLightIntensity(profile.iblIntensity);
        renderScene->SetIndirectLightRotation(profile.iblRotation);
        renderScene->SetSkybox(SkyboxHandle());
        renderScene->SetEntityEnabled(this->settings.hDirectionalLight,
                                      profile.sunEnabled);
        renderScene->SetLightIntensity(this->settings.hDirectionalLight,
                                       profile.sunIntensity);
        renderScene->SetLightDirection(this->settings.hDirectionalLight,
                                       profile.sunDir);
        renderScene->SetLightColor(this->settings.hDirectionalLight,
                                   profile.sunColor);
        this->settings.wgtIBLEnabled->SetChecked(profile.iblEnabled);
        this->settings.wgtSkyEnabled->SetChecked(false);
        this->settings.wgtDirectionalEnabled->SetChecked(profile.sunEnabled);
        this->settings.wgtIBLs->SetSelectedValue(kDefaultIBL.c_str());
        this->settings.wgtIBLIntensity->SetValue(profile.iblIntensity);
        this->settings.wgtSunIntensity->SetValue(profile.sunIntensity);
        this->settings.wgtSunDir->SetValue(profile.sunDir);
        this->settings.wgtSunColor->SetValue(gui::Color(
                profile.sunColor[0], profile.sunColor[1], profile.sunColor[2]));
    }

    bool SetIBL(visualization::Renderer &renderer, const char *path) {
        visualization::IndirectLightHandle newIBL;
        std::string iblPath;
        if (path) {
            newIBL = renderer.AddIndirectLight(ResourceLoadRequest(path));
            iblPath = path;
        } else {
            iblPath =
                    std::string(
                            gui::Application::GetInstance().GetResourcePath()) +
                    "/" + kDefaultIBL + "_ibl.ktx";
            newIBL = renderer.AddIndirectLight(
                    ResourceLoadRequest(iblPath.c_str()));
        }
        if (newIBL) {
            auto *renderScene = this->scene->GetScene();
            this->settings.hIbl = newIBL;
            auto intensity = renderScene->GetIndirectLightIntensity();
            renderScene->SetIndirectLight(newIBL);
            renderScene->SetIndirectLightIntensity(intensity);

            auto skyboxPath = std::string(iblPath);
            if (skyboxPath.find("_ibl.ktx") != std::string::npos) {
                skyboxPath = skyboxPath.substr(0, skyboxPath.size() - 8);
                skyboxPath += "_skybox.ktx";
                this->settings.hSky = renderer.AddSkybox(
                        ResourceLoadRequest(skyboxPath.c_str()));
                if (!this->settings.hSky) {
                    this->settings.hSky = renderer.AddSkybox(
                            ResourceLoadRequest(iblPath.c_str()));
                }
                bool isOn = this->settings.wgtSkyEnabled->IsChecked();
                if (isOn) {
                    this->scene->GetScene()->SetSkybox(this->settings.hSky);
                }
                this->scene->SetSkyboxHandle(this->settings.hSky, isOn);
            }
            return true;
        }
        return false;
    }

    void SetMouseControls(gui::Window &window,
                          gui::SceneWidget::Controls mode) {
        using Controls = gui::SceneWidget::Controls;
        this->scene->SetViewControls(mode);
        window.SetFocusWidget(this->scene.get());
        this->settings.wgtMouseArcball->SetOn(mode == Controls::ROTATE_OBJ);
        this->settings.wgtMouseFly->SetOn(mode == Controls::FLY);
        this->settings.wgtMouseModel->SetOn(mode == Controls::ROTATE_MODEL);
        this->settings.wgtMouseSun->SetOn(mode == Controls::ROTATE_SUN);
        this->settings.wgtMouseIBL->SetOn(mode == Controls::ROTATE_IBL);
    }
};

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
    auto &app = gui::Application::GetInstance();
    auto &theme = GetTheme();

    // Create menu
    if (!gui::Application::GetInstance().GetMenubar()) {
        auto fileMenu = std::make_shared<gui::Menu>();
        fileMenu->AddItem("Open...", FILE_OPEN, gui::KEY_O);
        fileMenu->AddItem("Export Current Image...", FILE_EXPORT_RGB);
        fileMenu->AddSeparator();
#if WIN32
        fileMenu->AddItem("Exit", FILE_QUIT);
#else
        fileMenu->AddItem("Quit", FILE_QUIT, gui::KEY_Q);
#endif
        auto helpMenu = std::make_shared<gui::Menu>();
        helpMenu->AddItem("Show Controls", HELP_KEYS);
        helpMenu->AddItem("Show Camera Info", HELP_CAMERA);
        helpMenu->AddSeparator();
        helpMenu->AddItem("About", HELP_ABOUT);
        helpMenu->AddItem("Contact", HELP_CONTACT);
        auto settingsMenu = std::make_shared<gui::Menu>();
        settingsMenu->AddItem("Lighting & Materials",
                              SETTINGS_LIGHT_AND_MATERIALS);
        settingsMenu->SetChecked(SETTINGS_LIGHT_AND_MATERIALS, true);
        auto menu = std::make_shared<gui::Menu>();
        menu->AddMenu("File", fileMenu);
        menu->AddMenu("Settings", settingsMenu);
#if defined(__APPLE__) && GUI_USE_NATIVE_MENUS
        // macOS adds a special search item to menus named "Help",
        // so add a space to avoid that.
        menu->AddMenu("Help ", helpMenu);
#else
        menu->AddMenu("Help", helpMenu);
#endif
        gui::Application::GetInstance().SetMenubar(menu);
    }

    // Create scene
    auto sceneId = GetRenderer().CreateScene();
    auto scene = std::make_shared<gui::SceneWidget>(
            *GetRenderer().GetScene(sceneId));
    auto renderScene = scene->GetScene();
    impl_->scene = scene;
    scene->SetBackgroundColor(gui::Color(1.0, 1.0, 1.0));

    // Create light
    const int defaultLightingProfileIdx = 0;
    auto &lightingProfile = gLightingProfiles[defaultLightingProfileIdx];
    visualization::LightDescription lightDescription;
    lightDescription.intensity = lightingProfile.sunIntensity;
    lightDescription.direction = lightingProfile.sunDir;
    lightDescription.castShadows = true;
    lightDescription.customAttributes["custom_type"] = "SUN";

    impl_->settings.hDirectionalLight =
            scene->GetScene()->AddLight(lightDescription);

    auto &settings = impl_->settings;
    std::string rsrcPath = app.GetResourcePath();
    auto iblPath = rsrcPath + "/default_ibl.ktx";
    settings.hIbl =
            GetRenderer().AddIndirectLight(ResourceLoadRequest(iblPath.data()));
    renderScene->SetIndirectLight(settings.hIbl);
    renderScene->SetIndirectLightIntensity(lightingProfile.iblIntensity);
    renderScene->SetIndirectLightRotation(lightingProfile.iblRotation);

    auto skyPath = rsrcPath + "/" + kDefaultIBL + "_skybox.ktx";
    settings.hSky =
            GetRenderer().AddSkybox(ResourceLoadRequest(skyPath.data()));
    scene->SetSkyboxHandle(settings.hSky, kDefaultShowSkybox);

    // Create materials
    auto litPath = rsrcPath + "/defaultLit.filamat";
    impl_->hLitMaterial = GetRenderer().AddMaterial(
            visualization::ResourceLoadRequest(litPath.data()));

    auto unlitPath = rsrcPath + "/defaultUnlit.filamat";
    impl_->hUnlitMaterial = GetRenderer().AddMaterial(
            visualization::ResourceLoadRequest(unlitPath.data()));

    impl_->SetMaterialsToDefault(GetRenderer());

    // Setup UI
    const auto em = theme.fontSize;
    const int lm = std::ceil(0.5 * em);
    const int gridSpacing = std::ceil(0.25 * em);

    AddChild(scene);

    // Add settings widget
    const int separationHeight = std::ceil(0.75 * em);
    // (we don't want as much left margin because the twisty arrow is the
    // only thing there, and visually it looks larger than the right.)
    const gui::Margins baseMargins(0.5 * lm, lm, lm, lm);
    settings.wgtBase = std::make_shared<gui::Vert>(0, baseMargins);

    gui::Margins indent(em, 0, 0, 0);
    auto viewCtrls =
            std::make_shared<gui::CollapsableVert>("View controls", 0, indent);

    // ... view manipulator buttons
    settings.wgtMouseArcball = std::make_shared<SmallToggleButton>("Arcball");
    this->impl_->settings.wgtMouseArcball->SetOn(true);
    settings.wgtMouseArcball->SetOnClicked([this]() {
        impl_->SetMouseControls(*this, gui::SceneWidget::Controls::ROTATE_OBJ);
    });
    settings.wgtMouseFly = std::make_shared<SmallToggleButton>("Fly");
    settings.wgtMouseFly->SetOnClicked([this]() {
        impl_->SetMouseControls(*this, gui::SceneWidget::Controls::FLY);
    });
    settings.wgtMouseModel = std::make_shared<SmallToggleButton>("Model");
    settings.wgtMouseModel->SetOnClicked([this]() {
        impl_->SetMouseControls(*this,
                                gui::SceneWidget::Controls::ROTATE_MODEL);
    });
    settings.wgtMouseSun = std::make_shared<SmallToggleButton>("Sun");
    settings.wgtMouseSun->SetOnClicked([this]() {
        impl_->SetMouseControls(*this, gui::SceneWidget::Controls::ROTATE_SUN);
    });
    settings.wgtMouseIBL = std::make_shared<SmallToggleButton>("Environment");
    settings.wgtMouseIBL->SetOnClicked([this]() {
        impl_->SetMouseControls(*this, gui::SceneWidget::Controls::ROTATE_IBL);
    });

    auto resetCamera = std::make_shared<SmallButton>("Reset camera");
    resetCamera->SetOnClicked([this]() {
        impl_->scene->GoToCameraPreset(gui::SceneWidget::CameraPreset::PLUS_Z);
    });

    auto cameraControls1 = std::make_shared<gui::Horiz>(gridSpacing);
    cameraControls1->AddStretch();
    cameraControls1->AddChild(settings.wgtMouseArcball);
    cameraControls1->AddChild(settings.wgtMouseFly);
    cameraControls1->AddChild(settings.wgtMouseModel);
    cameraControls1->AddStretch();
    auto cameraControls2 = std::make_shared<gui::Horiz>(gridSpacing);
    cameraControls2->AddStretch();
    cameraControls2->AddChild(settings.wgtMouseSun);
    cameraControls2->AddChild(settings.wgtMouseIBL);
    cameraControls2->AddStretch();
    viewCtrls->AddChild(std::make_shared<gui::Label>("Mouse Controls"));
    viewCtrls->AddChild(cameraControls1);
    viewCtrls->AddFixed(0.25 * em);
    viewCtrls->AddChild(cameraControls2);
    viewCtrls->AddFixed(separationHeight);
    viewCtrls->AddChild(gui::Horiz::MakeCentered(resetCamera));

    // ... background
    settings.wgtSkyEnabled = std::make_shared<gui::Checkbox>("Show skymap");
    settings.wgtSkyEnabled->SetChecked(kDefaultShowSkybox);
    settings.wgtSkyEnabled->SetOnChecked([this, renderScene](bool checked) {
        if (checked) {
            renderScene->SetSkybox(impl_->settings.hSky);
        } else {
            renderScene->SetSkybox(SkyboxHandle());
        }
        impl_->scene->SetSkyboxHandle(impl_->settings.hSky, checked);
        impl_->settings.wgtBGColor->SetEnabled(!checked);
    });

    impl_->settings.wgtBGColor = std::make_shared<gui::ColorEdit>();
    impl_->settings.wgtBGColor->SetValue({1, 1, 1});
    impl_->settings.wgtBGColor->SetOnValueChanged(
            [scene](const gui::Color &newColor) {
                scene->SetBackgroundColor(newColor);
            });
    auto bgLayout = std::make_shared<gui::VGrid>(2, gridSpacing);
    bgLayout->AddChild(std::make_shared<gui::Label>("BG Color"));
    bgLayout->AddChild(impl_->settings.wgtBGColor);

    viewCtrls->AddFixed(separationHeight);
    viewCtrls->AddChild(settings.wgtSkyEnabled);
    viewCtrls->AddFixed(0.25 * em);
    viewCtrls->AddChild(bgLayout);

    // ... show axes
    settings.wgtShowAxes = std::make_shared<gui::Checkbox>("Show axes");
    settings.wgtShowAxes->SetChecked(kDefaultShowAxes);
    settings.wgtShowAxes->SetOnChecked([this, renderScene](bool isChecked) {
        renderScene->SetEntityEnabled(this->impl_->settings.hAxes, isChecked);
    });
    viewCtrls->AddFixed(separationHeight);
    viewCtrls->AddChild(settings.wgtShowAxes);

    // ... lighting profiles
    settings.wgtLightingProfile = std::make_shared<gui::Combobox>();
    for (size_t i = 0; i < gLightingProfiles.size(); ++i) {
        settings.wgtLightingProfile->AddItem(gLightingProfiles[i].name.c_str());
    }
    settings.wgtLightingProfile->AddItem("Custom");
    settings.wgtLightingProfile->SetSelectedIndex(defaultLightingProfileIdx);
    settings.wgtLightingProfile->SetOnValueChanged(
            [this](const char *, int index) {
                if (index < int(gLightingProfiles.size())) {
                    this->impl_->SetLightingProfile(this->GetRenderer(),
                                                    gLightingProfiles[index]);
                    this->impl_->settings.userHasChangedLighting = true;
                } else {
                    this->impl_->settings.wgtAdvanced->SetIsOpen(true);
                    this->SetNeedsLayout();
                }
            });

    auto profileLayout = std::make_shared<gui::Vert>();
    profileLayout->AddChild(std::make_shared<gui::Label>("Lighting profiles"));
    profileLayout->AddChild(settings.wgtLightingProfile);
    viewCtrls->AddFixed(separationHeight);
    viewCtrls->AddChild(profileLayout);

    settings.wgtBase->AddChild(viewCtrls);
    settings.wgtBase->AddFixed(separationHeight);

    // ... advanced lighting
    settings.wgtAdvanced = std::make_shared<gui::CollapsableVert>(
            "Advanced lighting", 0, indent);
    settings.wgtAdvanced->SetIsOpen(false);
    settings.wgtBase->AddChild(settings.wgtAdvanced);

    // ....... lighting on/off
    settings.wgtAdvanced->AddChild(
            std::make_shared<gui::Label>("Light sources"));
    auto checkboxes = std::make_shared<gui::Horiz>();
    settings.wgtIBLEnabled = std::make_shared<gui::Checkbox>("HDR map");
    settings.wgtIBLEnabled->SetChecked(true);
    settings.wgtIBLEnabled->SetOnChecked([this, renderScene](bool checked) {
        impl_->settings.SetCustomProfile();
        if (checked) {
            renderScene->SetIndirectLight(impl_->settings.hIbl);
        } else {
            renderScene->SetIndirectLight(IndirectLightHandle());
        }
        this->impl_->settings.userHasChangedLighting = true;
    });
    checkboxes->AddChild(settings.wgtIBLEnabled);
    settings.wgtDirectionalEnabled = std::make_shared<gui::Checkbox>("Sun");
    settings.wgtDirectionalEnabled->SetChecked(true);
    settings.wgtDirectionalEnabled->SetOnChecked(
            [this, renderScene](bool checked) {
                impl_->settings.SetCustomProfile();
                renderScene->SetEntityEnabled(impl_->settings.hDirectionalLight,
                                              checked);
            });
    checkboxes->AddChild(settings.wgtDirectionalEnabled);
    settings.wgtAdvanced->AddChild(checkboxes);

    settings.wgtAdvanced->AddFixed(separationHeight);

    // ....... IBL
    settings.wgtIBLs = std::make_shared<gui::Combobox>();
    std::vector<std::string> resourceFiles;
    utility::filesystem::ListFilesInDirectory(rsrcPath, resourceFiles);
    std::sort(resourceFiles.begin(), resourceFiles.end());
    int n = 0;
    for (auto &f : resourceFiles) {
        if (f.find("_ibl.ktx") == f.size() - 8) {
            auto name = utility::filesystem::GetFileNameWithoutDirectory(f);
            name = name.substr(0, name.size() - 8);
            settings.wgtIBLs->AddItem(name.c_str());
            if (name == kDefaultIBL) {
                settings.wgtIBLs->SetSelectedIndex(n);
            }
            n++;
        }
    }
    settings.wgtIBLs->AddItem("Custom KTX file...");
    settings.wgtIBLs->SetOnValueChanged([this](const char *name, int) {
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
                this->impl_->settings.SetCustomProfile();
            });
            ShowDialog(dlg);
        }
    });

    settings.wgtIBLIntensity = MakeSlider(gui::Slider::INT, 0.0, 150000.0,
                                          lightingProfile.iblIntensity);
    settings.wgtIBLIntensity->SetOnValueChanged(
            [this, renderScene](double newValue) {
                renderScene->SetIndirectLightIntensity(newValue);
                this->impl_->settings.SetCustomProfile();
            });

    auto ambientLayout = std::make_shared<gui::VGrid>(2, gridSpacing);
    ambientLayout->AddChild(std::make_shared<gui::Label>("HDR map"));
    ambientLayout->AddChild(settings.wgtIBLs);
    ambientLayout->AddChild(std::make_shared<gui::Label>("Intensity"));
    ambientLayout->AddChild(settings.wgtIBLIntensity);
    // ambientLayout->AddChild(std::make_shared<gui::Label>("Skybox"));
    // ambientLayout->AddChild(settings.wgtLoadSky);

    settings.wgtAdvanced->AddChild(std::make_shared<gui::Label>("Environment"));
    settings.wgtAdvanced->AddChild(ambientLayout);
    settings.wgtAdvanced->AddFixed(separationHeight);

    // ... directional light (sun)
    settings.wgtSunIntensity = MakeSlider(gui::Slider::INT, 0.0, 500000.0,
                                          lightingProfile.sunIntensity);
    settings.wgtSunIntensity->SetOnValueChanged(
            [this, renderScene](double newValue) {
                renderScene->SetLightIntensity(
                        impl_->settings.hDirectionalLight, newValue);
                this->impl_->settings.SetCustomProfile();
            });

    auto setSunDir = [this, renderScene](const Eigen::Vector3f &dir) {
        this->impl_->settings.wgtSunDir->SetValue(dir);
        renderScene->SetLightDirection(impl_->settings.hDirectionalLight,
                                       dir.normalized());
        this->impl_->settings.SetCustomProfile();
    };

    this->impl_->scene->SelectDirectionalLight(
            settings.hDirectionalLight, [this](const Eigen::Vector3f &newDir) {
                impl_->settings.wgtSunDir->SetValue(newDir);
                this->impl_->settings.SetCustomProfile();
            });

    settings.wgtSunDir = std::make_shared<gui::VectorEdit>();
    settings.wgtSunDir->SetValue(lightDescription.direction);
    settings.wgtSunDir->SetOnValueChanged(setSunDir);

    settings.wgtSunColor = std::make_shared<gui::ColorEdit>();
    settings.wgtSunColor->SetValue({1, 1, 1});
    settings.wgtSunColor->SetOnValueChanged(
            [this, renderScene](const gui::Color &newColor) {
                this->impl_->settings.SetCustomProfile();
                renderScene->SetLightColor(
                        impl_->settings.hDirectionalLight,
                        {newColor.GetRed(), newColor.GetGreen(),
                         newColor.GetBlue()});
            });

    auto sunLayout = std::make_shared<gui::VGrid>(2, gridSpacing);
    sunLayout->AddChild(std::make_shared<gui::Label>("Intensity"));
    sunLayout->AddChild(settings.wgtSunIntensity);
    sunLayout->AddChild(std::make_shared<gui::Label>("Direction"));
    sunLayout->AddChild(settings.wgtSunDir);
    sunLayout->AddChild(std::make_shared<gui::Label>("Color"));
    sunLayout->AddChild(settings.wgtSunColor);

    settings.wgtAdvanced->AddChild(
            std::make_shared<gui::Label>("Sun (Directional light)"));
    settings.wgtAdvanced->AddChild(sunLayout);

    // materials settings
    auto materials = std::make_shared<gui::CollapsableVert>("Material settings",
                                                            0, indent);

    auto matGrid = std::make_shared<gui::VGrid>(2, gridSpacing);
    matGrid->AddChild(std::make_shared<gui::Label>("Type"));
    settings.wgtMaterialType.reset(
            new gui::Combobox({"Lit", "Unlit", "Normal map", "Depth"}));
    settings.wgtMaterialType->SetOnValueChanged([this](const char *,
                                                       int selectedIdx) {
        impl_->SetMaterialType(Impl::Settings::MaterialType(selectedIdx));
    });
    matGrid->AddChild(settings.wgtMaterialType);

    settings.wgtPrefabMaterial = std::make_shared<gui::Combobox>();
    for (auto &prefab : impl_->prefabMaterials) {
        settings.wgtPrefabMaterial->AddItem(prefab.first.c_str());
    }
    settings.wgtPrefabMaterial->SetSelectedValue(kDefaultMaterialName.c_str());
    settings.wgtPrefabMaterial->SetOnValueChanged(
            [this](const char *name, int) {
                auto &renderer = this->GetRenderer();
                impl_->SetMaterialByName(renderer, name);
            });

    matGrid->AddChild(std::make_shared<gui::Label>("Material"));
    matGrid->AddChild(settings.wgtPrefabMaterial);

    settings.wgtMaterialColor = std::make_shared<gui::ColorEdit>();
    settings.wgtMaterialColor->SetOnValueChanged(
            [this, renderScene](const gui::Color &color) {
                auto &renderer = this->GetRenderer();
                auto &settings = impl_->settings;
                Eigen::Vector3f color3(color.GetRed(), color.GetGreen(),
                                       color.GetBlue());
                if (settings.selectedType == Impl::Settings::LIT) {
                    settings.currentMaterials.lit.baseColor = color3;
                } else {
                    settings.currentMaterials.unlit.baseColor = color3;
                }
                settings.userHasChangedColor = true;
                settings.wgtResetMaterialColor->SetEnabled(true);

                visualization::MaterialInstanceHandle mat;
                if (settings.selectedType == Impl::Settings::UNLIT) {
                    mat = settings.currentMaterials.unlit.handle;
                } else {
                    mat = settings.currentMaterials.lit.handle;
                }
                mat = renderer.ModifyMaterial(mat)
                              .SetColor("baseColor", color3)
                              .Finish();
                for (const auto &handle : impl_->geometryHandles) {
                    renderScene->AssignMaterial(handle, mat);
                }
            });
    settings.wgtResetMaterialColor = std::make_shared<SmallButton>("Reset");
    settings.wgtResetMaterialColor->SetEnabled(
            impl_->settings.userHasChangedColor);
    settings.wgtResetMaterialColor->SetOnClicked([this]() {
        auto &renderer = this->GetRenderer();
        impl_->settings.userHasChangedColor = false;
        impl_->settings.wgtResetMaterialColor->SetEnabled(false);
        impl_->SetMaterialByName(
                renderer,
                impl_->settings.wgtPrefabMaterial->GetSelectedValue());
    });

    matGrid->AddChild(std::make_shared<gui::Label>("Color"));
    auto colorLayout = std::make_shared<gui::Horiz>();
    colorLayout->AddChild(settings.wgtMaterialColor);
    colorLayout->AddFixed(0.25 * em);
    colorLayout->AddChild(impl_->settings.wgtResetMaterialColor);
    matGrid->AddChild(colorLayout);

    matGrid->AddChild(std::make_shared<gui::Label>("Point size"));
    settings.wgtPointSize = MakeSlider(gui::Slider::INT, 1.0, 10.0, 3);
    settings.wgtPointSize->SetOnValueChanged([this](double value) {
        float size = float(value);
        impl_->settings.currentMaterials.unlit.pointSize = size;
        auto &renderer = GetRenderer();
        renderer.ModifyMaterial(impl_->settings.currentMaterials.lit.handle)
                .SetParameter("pointSize", size)
                .Finish();
        renderer.ModifyMaterial(impl_->settings.currentMaterials.unlit.handle)
                .SetParameter("pointSize", size)
                .Finish();
        renderer.ModifyMaterial(FilamentResourceManager::kDepthMaterial)
                .SetParameter("pointSize", size)
                .Finish();
        renderer.ModifyMaterial(FilamentResourceManager::kNormalsMaterial)
                .SetParameter("pointSize", size)
                .Finish();
    });
    matGrid->AddChild(settings.wgtPointSize);
    materials->AddChild(matGrid);

    settings.wgtBase->AddFixed(separationHeight);
    settings.wgtBase->AddChild(materials);

    AddChild(settings.wgtBase);

    // Other items
    impl_->helpKeys = createHelpDisplay(this);
    impl_->helpKeys->SetVisible(false);
    AddChild(impl_->helpKeys);
    impl_->helpCamera = createCameraDisplay(this);
    impl_->helpCamera->SetVisible(false);
    AddChild(impl_->helpCamera);

    // Set the actual geometries
    SetGeometry(geometries);  // also updates the camera
}

GuiVisualizer::~GuiVisualizer() {}

void GuiVisualizer::SetTitle(const std::string &title) {
    Super::SetTitle(title.c_str());
}

void GuiVisualizer::SetGeometry(
        const std::vector<std::shared_ptr<const geometry::Geometry>>
                &geometries) {
    const std::size_t kMinPointCloudPointsForDecimation = 6000000;

    gui::SceneWidget::ModelDescription desc;

    auto *scene3d = impl_->scene->GetScene();
    if (impl_->settings.hAxes) {
        scene3d->RemoveGeometry(impl_->settings.hAxes);
    }
    for (auto &h : impl_->geometryHandles) {
        scene3d->RemoveGeometry(h);
    }
    impl_->geometryHandles.clear();

    impl_->SetMaterialsToDefault(GetRenderer());

    std::size_t nPointClouds = 0;
    std::size_t nPointCloudPoints = 0;
    for (auto &g : geometries) {
        if (g->GetGeometryType() ==
            geometry::Geometry::GeometryType::PointCloud) {
            nPointClouds++;
            auto cloud =
                    std::static_pointer_cast<const geometry::PointCloud>(g);
            nPointCloudPoints += cloud->points_.size();
        }
    }

    geometry::AxisAlignedBoundingBox bounds;
    std::size_t nUnlit = 0;
    for (size_t i = 0; i < geometries.size(); ++i) {
        std::shared_ptr<const geometry::Geometry> g = geometries[i];
        Impl::Materials materials = impl_->settings.currentMaterials;

        visualization::MaterialInstanceHandle selectedMaterial;

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
                    selectedMaterial = materials.unlit.handle;
                    nUnlit += 1;
                } else {
                    selectedMaterial = materials.lit.handle;
                }
            } break;
            case geometry::Geometry::GeometryType::LineSet: {
                selectedMaterial = materials.unlit.handle;
                nUnlit += 1;
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
                    material.baseColor.x() = mesh_material.baseColor.r;
                    material.baseColor.y() = mesh_material.baseColor.g;
                    material.baseColor.z() = mesh_material.baseColor.b;
                    material.roughness = mesh_material.baseRoughness;

                    auto is_map_valid =
                            [](std::shared_ptr<geometry::Image> map) -> bool {
                        return map && map->HasData();
                    };

                    if (is_map_valid(mesh_material.albedo)) {
                        maps.albedoMap =
                                GetRenderer().AddTexture(mesh_material.albedo);
                    }
                    if (is_map_valid(mesh_material.normalMap)) {
                        maps.normalMap = GetRenderer().AddTexture(
                                mesh_material.normalMap);
                        albedo_only = false;
                    }
                    if (is_map_valid(mesh_material.ambientOcclusion)) {
                        maps.ambientOcclusionMap = GetRenderer().AddTexture(
                                mesh_material.ambientOcclusion);
                        albedo_only = false;
                    }
                    if (is_map_valid(mesh_material.roughness)) {
                        maps.roughnessMap = GetRenderer().AddTexture(
                                mesh_material.roughness);
                        albedo_only = false;
                    }
                    if (is_map_valid(mesh_material.metallic)) {
                        material.metallic = 1.f;
                        maps.metallicMap = GetRenderer().AddTexture(
                                mesh_material.metallic);
                        albedo_only = false;
                    } else {
                        material.metallic = 0.f;
                    }
                    if (is_map_valid(mesh_material.reflectance)) {
                        maps.reflectanceMap = GetRenderer().AddTexture(
                                mesh_material.reflectance);
                        albedo_only = false;
                    }
                    if (is_map_valid(mesh_material.clearCoat)) {
                        maps.clearCoatMap = GetRenderer().AddTexture(
                                mesh_material.clearCoat);
                        albedo_only = false;
                    }
                    if (is_map_valid(mesh_material.clearCoatRoughness)) {
                        maps.clearCoatRoughnessMap = GetRenderer().AddTexture(
                                mesh_material.clearCoatRoughness);
                        albedo_only = false;
                    }
                    if (is_map_valid(mesh_material.anisotropy)) {
                        maps.anisotropyMap = GetRenderer().AddTexture(
                                mesh_material.anisotropy);
                        albedo_only = false;
                    }
                    impl_->SetMaterialsToCurrentSettings(GetRenderer(),
                                                         material, maps);
                }

                if ((mesh->HasVertexColors() && !MeshHasUniformColor(*mesh)) ||
                    (mesh->HasMaterials() && albedo_only)) {
                    selectedMaterial = materials.unlit.handle;
                    nUnlit += 1;
                } else {
                    selectedMaterial = materials.lit.handle;
                }
            } break;
            default:
                utility::LogWarning("Geometry type {} not supported!",
                                    (int)g->GetGeometryType());
                break;
        }

        auto g3 = std::static_pointer_cast<const geometry::Geometry3D>(g);
        auto handle = scene3d->AddGeometry(*g3, selectedMaterial);
        bounds += scene3d->GetEntityBoundingBox(handle);
        impl_->geometryHandles.push_back(handle);

        if (g->GetGeometryType() ==
            geometry::Geometry::GeometryType::PointCloud) {
            desc.pointClouds.push_back(handle);
            auto pcd = std::static_pointer_cast<const geometry::PointCloud>(g);
            if (nPointCloudPoints > kMinPointCloudPointsForDecimation) {
                int sampleRate = nPointCloudPoints /
                                 (kMinPointCloudPointsForDecimation / 2);
                auto small = pcd->UniformDownSample(sampleRate);
                handle = scene3d->AddGeometry(*small, selectedMaterial);
                desc.fastPointClouds.push_back(handle);
                impl_->geometryHandles.push_back(handle);
            }
        } else {
            desc.meshes.push_back(handle);
        }
    }

    if (!geometries.empty()) {
        auto viewMode = impl_->scene->GetView()->GetMode();
        if (viewMode == visualization::View::Mode::Normals) {
            impl_->SetMaterialType(Impl::Settings::NORMAL_MAP);
        } else if (viewMode == visualization::View::Mode::Depth) {
            impl_->SetMaterialType(Impl::Settings::DEPTH);
        } else {
            if (nUnlit == geometries.size()) {
                impl_->SetMaterialType(Impl::Settings::UNLIT);
            } else {
                impl_->SetMaterialType(Impl::Settings::LIT);
            }
        }

        if (nPointClouds == geometries.size() &&
            !impl_->settings.userHasChangedLighting) {
            impl_->SetLightingProfile(GetRenderer(), kPointCloudProfileName);
        }
        impl_->settings.wgtPointSize->SetEnabled(nPointClouds > 0);
    }

    // Add axes. Axes length should be the longer of the bounds extent
    // or 25% of the distance from the origin. The latter is necessary
    // so that the axis is big enough to be visible even if the object
    // is far from the origin. See caterpillar.ply from Tanks & Temples.
    auto axisLength = bounds.GetMaxExtent();
    if (axisLength < 0.001) {  // avoid div by zero errors in CreateAxes()
        axisLength = 1.0;
    }
    axisLength = std::max(axisLength, 0.25 * bounds.GetCenter().norm());
    auto axes = CreateAxes(axisLength);
    impl_->settings.hAxes = scene3d->AddGeometry(*axes);
    scene3d->SetGeometryShadows(impl_->settings.hAxes, false, false);
    scene3d->SetEntityEnabled(impl_->settings.hAxes,
                              impl_->settings.wgtShowAxes->IsChecked());
    desc.axes = impl_->settings.hAxes;
    impl_->scene->SetModel(desc);

    impl_->scene->SetupCamera(60.0, bounds, bounds.GetCenter().cast<float>());
}

void GuiVisualizer::Layout(const gui::Theme &theme) {
    auto r = GetContentRect();
    const auto em = theme.fontSize;
    impl_->scene->SetFrame(r);

    // Draw help keys HUD in upper left
    const auto pref = impl_->helpKeys->CalcPreferredSize(theme);
    impl_->helpKeys->SetFrame(gui::Rect(0, r.y, pref.width, pref.height));
    impl_->helpKeys->Layout(theme);

    // Draw camera HUD in lower left
    const auto prefcam = impl_->helpCamera->CalcPreferredSize(theme);
    impl_->helpCamera->SetFrame(gui::Rect(0, r.height + r.y - prefcam.height,
                                          prefcam.width, prefcam.height));
    impl_->helpCamera->Layout(theme);

    // Settings in upper right
    const auto kLightSettingsWidth = 18 * em;
    auto lightSettingsSize = impl_->settings.wgtBase->CalcPreferredSize(theme);
    gui::Rect lightSettingsRect(r.width - kLightSettingsWidth, r.y,
                                kLightSettingsWidth, lightSettingsSize.height);
    impl_->settings.wgtBase->SetFrame(lightSettingsRect);

    Super::Layout(theme);
}

bool GuiVisualizer::SetIBL(const char *path) {
    auto result = impl_->SetIBL(GetRenderer(), path);
    PostRedraw();
    return result;
}

bool GuiVisualizer::LoadGeometry(const std::string &path) {
    auto geometry = std::shared_ptr<geometry::Geometry3D>();

    auto geometryType = io::ReadFileGeometryType(path);

    auto mesh = std::make_shared<geometry::TriangleMesh>();
    bool meshSuccess = false;
    if (geometryType & io::CONTAINS_TRIANGLES) {
        try {
            meshSuccess = io::ReadTriangleMesh(path, *mesh);
        } catch (...) {
            meshSuccess = false;
        }
    }
    if (meshSuccess) {
        if (mesh->triangles_.size() == 0) {
            utility::LogWarning(
                    "Contains 0 triangles, will read as point cloud");
            mesh.reset();
        } else {
            mesh->ComputeVertexNormals();
            if (mesh->vertex_colors_.empty()) {
                mesh->PaintUniformColor({1, 1, 1});
            }
            geometry = mesh;
        }
        // Make sure the mesh has texture coordinates
        if (!mesh->HasTriangleUvs()) {
            mesh->triangle_uvs_.resize(mesh->triangles_.size() * 3, {0.0, 0.0});
        }
    } else {
        // LogError throws an exception, which we don't want, because this might
        // be a point cloud.
        utility::LogInfo("{} appears to be a point cloud", path.c_str());
        mesh.reset();
    }

    if (!geometry) {
        auto cloud = std::make_shared<geometry::PointCloud>();
        bool success = false;
        try {
            success = io::ReadPointCloud(path, *cloud);
        } catch (...) {
            success = false;
        }
        if (success) {
            utility::LogInfo("Successfully read {}", path.c_str());
            if (!cloud->HasNormals()) {
                cloud->EstimateNormals();
            }
            cloud->NormalizeNormals();
            geometry = cloud;
        } else {
            utility::LogWarning("Failed to read points {}", path.c_str());
            cloud.reset();
        }
    }

    if (geometry) {
        SetGeometry({geometry});
    }
    return (geometry != nullptr);
}

void GuiVisualizer::ExportCurrentImage(int width,
                                       int height,
                                       const std::string &path) {
    GetRenderer().RenderToImage(
            width, height, impl_->scene->GetView(), impl_->scene->GetScene(),
            [this, path](std::shared_ptr<geometry::Image> image) mutable {
                if (!io::WriteImage(path, *image)) {
                    this->ShowMessageBox(
                            "Error", (std::string("Could not write image to ") +
                                      path + ".")
                                             .c_str());
                }
            });
}

void GuiVisualizer::OnMenuItemSelected(gui::Menu::ItemId itemId) {
    auto menuId = MenuId(itemId);
    switch (menuId) {
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
            auto visibility = !impl_->settings.wgtBase->IsVisible();
            impl_->settings.wgtBase->SetVisible(visibility);
            auto menubar = gui::Application::GetInstance().GetMenubar();
            menubar->SetChecked(SETTINGS_LIGHT_AND_MATERIALS, visibility);

            // We need relayout because materials settings pos depends on light
            // settings visibility
            Layout(GetTheme());

            break;
        }
        case HELP_KEYS: {
            bool isVisible = !impl_->helpKeys->IsVisible();
            impl_->helpKeys->SetVisible(isVisible);
            auto menubar = gui::Application::GetInstance().GetMenubar();
            menubar->SetChecked(HELP_KEYS, isVisible);
            break;
        }
        case HELP_CAMERA: {
            bool isVisible = !impl_->helpCamera->IsVisible();
            impl_->helpCamera->SetVisible(isVisible);
            auto menubar = gui::Application::GetInstance().GetMenubar();
            menubar->SetChecked(HELP_CAMERA, isVisible);
            if (isVisible) {
                impl_->scene->SetCameraChangedCallback(
                        [this](visualization::Camera *cam) {
                            auto children =
                                    this->impl_->helpCamera->GetChildren();
                            auto set_text = [](const Eigen::Vector3f &v,
                                               std::shared_ptr<gui::Widget>
                                                       label) {
                                auto l = std::dynamic_pointer_cast<gui::Label>(
                                        label);
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
                impl_->scene->SetCameraChangedCallback(
                        std::function<void(visualization::Camera *)>());
            }
            break;
        }
        case HELP_ABOUT: {
            auto dlg = createAboutDialog(this);
            ShowDialog(dlg);
            break;
        }
        case HELP_CONTACT: {
            auto dlg = createContactDialog(this);
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
    if (!vis->LoadGeometry(path)) {
        auto err = std::string("Error reading geometry file '") + path + "'";
        vis->ShowMessageBox("Error loading geometry", err.c_str());
    }
    PostRedraw();
}

}  // namespace visualization
}  // namespace open3d
