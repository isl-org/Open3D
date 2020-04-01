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
#include "Open3D/IO/ClassIO/ImageIO.h"
#include "Open3D/IO/ClassIO/PointCloudIO.h"
#include "Open3D/IO/ClassIO/TriangleMeshIO.h"
#include "Open3D/Open3DConfig.h"
#include "Open3D/Utility/Console.h"
#include "Open3D/Utility/FileSystem.h"
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
            "Copyright (c) 2018 www.open3d.org\n\n"

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
    layout->AddChild(gui::Horiz::MakeFixed(theme.fontSize));
    layout->AddChild(text);
    layout->AddChild(gui::Horiz::MakeFixed(theme.fontSize));
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
    addRow("Shift + left-drag    ", "Forward/backward");

#if defined(__APPLE__)
    addLabel("Cmd + left-drag");
#else
    addLabel("Ctrl + left-drag");
#endif  // __APPLE__
    addLabel("Pan camera");

#if defined(__APPLE__)
    addLabel("Opt + left-drag");
#else
    addLabel("Win + left-drag");
#endif  // __APPLE__
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
    addRow("W", "Forward");
    addRow("S", "Backward");
    addRow("A", "Step left");
    addRow("D", "Step right");
    addRow("Q", "Step up");
    addRow("Z", "Step down");
    addRow("Up", "Look up");
    addRow("Down", "Look down");
    addRow("Left", "Look left");
    addRow("Right", "Look right");

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
    layout->AddChild(gui::Horiz::MakeFixed(em));

    auto columns = std::make_shared<gui::Horiz>(em, gui::Margins());
    columns->AddChild(leftCol);
    columns->AddChild(rightCol);
    layout->AddChild(columns);

    layout->AddChild(gui::Horiz::MakeFixed(em));
    layout->AddChild(gui::Horiz::MakeCentered(ok));
    dlg->AddChild(layout);

    return dlg;
}

struct SmartMode {
    static bool PointCloudHasUniformColor(const geometry::PointCloud &pcd) {
        if (!pcd.HasColors()) {
            return true;
        }

        static const double e = 1.0 / 255.0;
        static const double kSqEpsilon = Eigen::Vector3d(e, e, e).squaredNorm();
        const auto &color = pcd.colors_[0];

        for (const auto &c : pcd.colors_) {
            if ((color - c).squaredNorm() > kSqEpsilon) {
                return false;
            }
        }

        return true;
    }
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
    bool sunEnabled = true;
};

static const std::string kPointCloudProfileName = "Cloudy day (no direct sun)";

static const std::vector<LightingProfile> gLightingProfiles = {
        {.name = "Bright day with sun at +Y [default]",
         .iblIntensity = 100000,
         .sunIntensity = 100000,
         .sunDir = {0.577f, -0.577f, -0.577f}},
        {.name = "Bright day with sun at -Y",
         .iblIntensity = 100000,
         .sunIntensity = 100000,
         .sunDir = {0.577f, 0.577f, 0.577f},
         .sunColor = {1.0f, 1.0f, 1.0f},
         .iblRotation = Scene::Transform(
                 Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitX()))},
        {.name = "Bright day with sun at +Z",
         .iblIntensity = 100000,
         .sunIntensity = 100000,
         .sunDir = {0.577f, 0.577f, -0.577f}},
        {.name = "Less bright day with sun at +Y",
         .iblIntensity = 75000,
         .sunIntensity = 100000,
         .sunDir = {0.577f, -0.577f, -0.577f}},
        {.name = "Less bright day with sun at -Y",
         .iblIntensity = 75000,
         .sunIntensity = 100000,
         .sunDir = {0.577f, 0.577f, 0.577f},
         .sunColor = {1.0f, 1.0f, 1.0f},
         .iblRotation = Scene::Transform(
                 Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitX()))},
        {.name = "Less bright day with sun at +Z",
         .iblIntensity = 75000,
         .sunIntensity = 100000,
         .sunDir = {0.577f, 0.577f, -0.577f}},
        {.name = kPointCloudProfileName,
         .iblIntensity = 60000,
         .sunIntensity = 100000,
         .sunDir = {0.577f, -0.577f, -0.577f},
         .sunColor = {1.0f, 1.0f, 1.0f},
         .iblRotation = Scene::Transform::Identity(),
         .iblEnabled = true,
         .sunEnabled = false}};

enum MenuId {
    FILE_OPEN,
    FILE_EXPORT_RGB,
    FILE_CLOSE,
    SETTINGS_LIGHT_AND_MATERIALS,
    HELP_KEYS,
    HELP_ABOUT,
    HELP_CONTACT
};

struct GuiVisualizer::Impl {
    std::vector<visualization::GeometryHandle> geometryHandles;

    std::shared_ptr<gui::SceneWidget> scene;
    std::shared_ptr<gui::VGrid> helpKeys;

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
        Eigen::Vector3f baseColor = {1.f, 1.f, 1.f};
        float pointSize = 5.f;
    };

    struct Materials {
        LitMaterial lit;
        UnlitMaterial unlit;
    };

    std::map<std::string, LitMaterial> prefabMaterials = {
            {"Default", {}},
            {"Aluminum",
             {visualization::MaterialInstanceHandle::kBad,
              {0.913f, 0.921f, 0.925f},
              1.0f,
              0.5f,
              0.9f,
              0.0f,
              0.0f,
              0.0f,
              3.0f}},
            {"Gold",
             {visualization::MaterialInstanceHandle::kBad,
              {1.000f, 0.766f, 0.336f},
              1.0f,
              0.3f,
              0.9f,
              0.0f,
              0.0f,
              0.0f,
              3.0f}},
            {"Copper",
             {visualization::MaterialInstanceHandle::kBad,
              {0.955f, 0.637f, 0.538f},
              1.0f,
              0.3f,
              0.9f,
              0.0f,
              0.0f,
              0.0f,
              3.0f}},
            {"Iron",
             {visualization::MaterialInstanceHandle::kBad,
              {0.560f, 0.570f, 0.580f},
              1.0f,
              0.5f,
              0.9f,
              0.0f,
              0.0f,
              0.0f,
              3.0f}},
            {"Plastic (white)",
             {visualization::MaterialInstanceHandle::kBad,
              {1.0f, 1.0f, 1.0f},
              0.0f,
              0.5f,
              0.5f,
              0.5f,
              0.2f,
              0.0f,
              3.0f}},
            {"Ceramic (white)",
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

    std::unordered_map<visualization::REHandle_abstract, Materials>
            geometryMaterials;

    visualization::MaterialHandle hLitMaterial;
    visualization::MaterialHandle hUnlitMaterial;

    struct Settings {
        visualization::IndirectLightHandle hIbl;
        visualization::SkyboxHandle hSky;
        visualization::TextureHandle hSkyTexture;
        visualization::LightHandle hDirectionalLight;

        std::shared_ptr<gui::Widget> wgtBase;
        std::shared_ptr<gui::Button> wgtMouseArcball;
        std::shared_ptr<gui::Button> wgtMouseFly;
        std::shared_ptr<gui::Button> wgtMouseSun;
        std::shared_ptr<gui::Button> wgtMouseIBL;
        std::shared_ptr<gui::Combobox> wgtLightingProfile;
        std::shared_ptr<gui::CollapsableVert> wgtAdvanced;
        std::shared_ptr<gui::Checkbox> wgtAmbientEnabled;
        std::shared_ptr<gui::Checkbox> wgtSkyEnabled;
        std::shared_ptr<gui::Checkbox> wgtDirectionalEnabled;
        std::shared_ptr<gui::Combobox> wgtAmbientIBLs;
        std::shared_ptr<gui::Button> wgtLoadSky;
        std::shared_ptr<gui::Slider> wgtAmbientIntensity;
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
        std::shared_ptr<gui::Combobox> wgtMaterialType;

        std::shared_ptr<gui::Combobox> wgtPrefabMaterial;
        std::shared_ptr<gui::Slider> wgtPointSize;

        void SetCustomProfile() {
            wgtLightingProfile->SetSelectedIndex(gLightingProfiles.size());
        }

        void SetMaterialSelected(const MaterialType type) {
            wgtMaterialType->SetSelectedIndex(type);
        }
    } settings;

    static void SetMaterialsDefaults(Materials &materials,
                                     visualization::Renderer &renderer) {
        materials.lit.handle =
                renderer.ModifyMaterial(materials.lit.handle)
                        .SetColor("baseColor", materials.lit.baseColor)
                        .SetParameter("roughness", materials.lit.roughness)
                        .SetParameter("metallic", materials.lit.metallic)
                        .SetParameter("reflectance", materials.lit.reflectance)
                        .SetParameter("clearCoat", materials.lit.clearCoat)
                        .SetParameter("clearCoatRoughness",
                                      materials.lit.clearCoatRoughness)
                        .SetParameter("anisotropy", materials.lit.anisotropy)
                        .SetParameter("pointSize", materials.lit.pointSize)
                        .Finish();

        materials.unlit.handle =
                renderer.ModifyMaterial(materials.unlit.handle)
                        .SetColor("baseColor", materials.unlit.baseColor)
                        .SetParameter("pointSize", materials.unlit.pointSize)
                        .Finish();
    }

    void SetLightingProfile(const std::string &name) {
        for (size_t i = 0; i < gLightingProfiles.size(); ++i) {
            if (gLightingProfiles[i].name == name) {
                SetLightingProfile(gLightingProfiles[i]);
                this->settings.wgtLightingProfile->SetSelectedValue(
                        name.c_str());
                return;
            }
        }
        utility::LogWarning("Could not find lighting profile '{}'", name);
    }

    void SetLightingProfile(const LightingProfile &profile) {
        auto *renderScene = this->scene->GetScene();
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
        this->settings.wgtAmbientEnabled->SetChecked(profile.iblEnabled);
        this->settings.wgtSkyEnabled->SetChecked(false);
        this->settings.wgtDirectionalEnabled->SetChecked(profile.sunEnabled);
        this->settings.wgtAmbientIBLs->SetSelectedValue("default");
        this->settings.wgtAmbientIntensity->SetValue(profile.iblIntensity);
        this->settings.wgtSunIntensity->SetValue(profile.sunIntensity);
        this->settings.wgtSunDir->SetValue(profile.sunDir);
        this->settings.wgtSunColor->SetValue(gui::Color(
                profile.sunColor[0], profile.sunColor[1], profile.sunColor[2]));
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
        fileMenu->AddItem("Open...", "Ctrl-O", FILE_OPEN);
        fileMenu->AddItem("Export Current Image...", nullptr, FILE_EXPORT_RGB);
        fileMenu->AddSeparator();
        fileMenu->AddItem("Close", "Ctrl-W", FILE_CLOSE);
        auto helpMenu = std::make_shared<gui::Menu>();
        helpMenu->AddItem("Show Controls", nullptr, HELP_KEYS);
        helpMenu->AddSeparator();
        helpMenu->AddItem("About", nullptr, HELP_ABOUT);
        helpMenu->AddItem("Contact", nullptr, HELP_CONTACT);
        auto settingsMenu = std::make_shared<gui::Menu>();
        settingsMenu->AddItem("Lighting & Materials", nullptr,
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

    // Create materials
    auto skyPath = rsrcPath + "/default_sky.ktx";
    settings.hSky =
            GetRenderer().AddSkybox(ResourceLoadRequest(skyPath.data()));

    auto litPath = rsrcPath + "/defaultLit.filamat";
    impl_->hLitMaterial = GetRenderer().AddMaterial(
            visualization::ResourceLoadRequest(litPath.data()));

    auto unlitPath = rsrcPath + "/defaultUnlit.filamat";
    impl_->hUnlitMaterial = GetRenderer().AddMaterial(
            visualization::ResourceLoadRequest(unlitPath.data()));

    // Setup UI
    const auto em = theme.fontSize;
    const int lm = std::ceil(0.5 * em);
    const int gridSpacing = std::ceil(0.25 * em);

    auto drawTimeLabel = std::make_shared<DrawTimeLabel>(this);
    drawTimeLabel->SetTextColor(gui::Color(0.5, 0.5, 0.5));

    AddChild(scene);

    // Add settings widget
    const int separationHeight = std::ceil(em);
    settings.wgtBase = std::make_shared<gui::Vert>(0, gui::Margins(lm));

    settings.wgtLoadSky = std::make_shared<SmallButton>("Load skybox");
    settings.wgtLoadSky->SetOnClicked([this, renderScene]() {
        auto dlg = std::make_shared<gui::FileDialog>(
                gui::FileDialog::Type::OPEN, "Open skybox", GetTheme());
        dlg->AddFilter(".ktx", "Khronos Texture (.ktx)");
        dlg->SetOnCancel([this]() { this->CloseDialog(); });
        dlg->SetOnDone([this, renderScene](const char *path) {
            this->CloseDialog();
            auto newSky = GetRenderer().AddSkybox(ResourceLoadRequest(path));
            if (newSky) {
                impl_->settings.hSky = newSky;
                impl_->settings.wgtSkyEnabled->SetChecked(true);
                impl_->settings.SetCustomProfile();

                renderScene->SetSkybox(newSky);
            }
        });
        ShowDialog(dlg);
    });

    gui::Margins indent(em, 0, 0, 0);
    auto viewCtrls =
            std::make_shared<gui::CollapsableVert>("View controls", 0, indent);

    // ... view manipulator buttons
    settings.wgtMouseArcball = std::make_shared<SmallToggleButton>("Arcball");
    this->impl_->settings.wgtMouseArcball->SetOn(true);
    settings.wgtMouseArcball->SetOnClicked([this]() {
        this->impl_->scene->SetViewControls(
                gui::SceneWidget::Controls::ROTATE_OBJ);
        this->SetTickEventsEnabled(false);
        this->impl_->settings.wgtMouseArcball->SetOn(true);
        this->impl_->settings.wgtMouseFly->SetOn(false);
        this->impl_->settings.wgtMouseSun->SetOn(false);
        this->impl_->settings.wgtMouseIBL->SetOn(false);
    });
    settings.wgtMouseFly = std::make_shared<SmallToggleButton>("Fly");
    settings.wgtMouseFly->SetOnClicked([this]() {
        this->impl_->scene->SetViewControls(gui::SceneWidget::Controls::FPS);
        this->SetFocusWidget(this->impl_->scene.get());
        this->SetTickEventsEnabled(true);
        this->impl_->settings.wgtMouseArcball->SetOn(false);
        this->impl_->settings.wgtMouseFly->SetOn(true);
        this->impl_->settings.wgtMouseSun->SetOn(false);
        this->impl_->settings.wgtMouseIBL->SetOn(false);
    });
    settings.wgtMouseSun = std::make_shared<SmallToggleButton>("Sun");
    settings.wgtMouseSun->SetOnClicked([this]() {
        this->impl_->scene->SetViewControls(
                gui::SceneWidget::Controls::ROTATE_SUN);
        this->SetTickEventsEnabled(false);
        this->impl_->settings.wgtMouseArcball->SetOn(false);
        this->impl_->settings.wgtMouseFly->SetOn(false);
        this->impl_->settings.wgtMouseSun->SetOn(true);
        this->impl_->settings.wgtMouseIBL->SetOn(false);
    });
    settings.wgtMouseIBL = std::make_shared<SmallToggleButton>("IBL");
    settings.wgtMouseIBL->SetOnClicked([this]() {
        this->impl_->scene->SetViewControls(
                gui::SceneWidget::Controls::ROTATE_IBL);
        this->SetTickEventsEnabled(false);
        this->impl_->settings.wgtMouseArcball->SetOn(false);
        this->impl_->settings.wgtMouseFly->SetOn(false);
        this->impl_->settings.wgtMouseSun->SetOn(false);
        this->impl_->settings.wgtMouseIBL->SetOn(true);
    });

    auto cameraControls = std::make_shared<gui::Horiz>(gridSpacing);
    cameraControls->AddChild(gui::Horiz::MakeStretch());
    cameraControls->AddChild(settings.wgtMouseArcball);
    cameraControls->AddChild(settings.wgtMouseFly);
    cameraControls->AddChild(gui::Horiz::MakeFixed(em));
    cameraControls->AddChild(settings.wgtMouseSun);
    cameraControls->AddChild(settings.wgtMouseIBL);
    cameraControls->AddChild(gui::Horiz::MakeStretch());
    viewCtrls->AddChild(std::make_shared<gui::Label>("Mouse Controls"));
    viewCtrls->AddChild(cameraControls);

    // ... background colors
    auto bgcolor = std::make_shared<gui::ColorEdit>();
    bgcolor->SetValue({1, 1, 1});
    bgcolor->OnValueChanged = [scene](const gui::Color &newColor) {
        scene->SetBackgroundColor(newColor);
    };
    auto bgcolorLayout = std::make_shared<gui::VGrid>(2, gridSpacing);
    bgcolorLayout->AddChild(std::make_shared<gui::Label>("BG Color"));
    bgcolorLayout->AddChild(bgcolor);
    viewCtrls->AddChild(gui::Horiz::MakeFixed(separationHeight));
    viewCtrls->AddChild(bgcolorLayout);

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
                    this->impl_->SetLightingProfile(gLightingProfiles[index]);
                } else {
                    this->impl_->settings.wgtAdvanced->SetIsOpen(true);
                    this->SetNeedsLayout();
                }
            });

    auto profileLayout = std::make_shared<gui::Vert>();
    profileLayout->AddChild(std::make_shared<gui::Label>("Lighting profiles"));
    profileLayout->AddChild(settings.wgtLightingProfile);
    viewCtrls->AddChild(gui::Horiz::MakeFixed(separationHeight));
    viewCtrls->AddChild(profileLayout);

    settings.wgtBase->AddChild(viewCtrls);
    settings.wgtBase->AddChild(gui::Horiz::MakeFixed(separationHeight));

    // ... advanced lighting
    settings.wgtAdvanced = std::make_shared<gui::CollapsableVert>(
            "Advanced lighting", 0, indent);
    settings.wgtAdvanced->SetIsOpen(false);
    settings.wgtBase->AddChild(settings.wgtAdvanced);

    // ....... lighting on/off
    settings.wgtAdvanced->AddChild(
            std::make_shared<gui::Label>("Light sources"));
    auto checkboxes = std::make_shared<gui::Horiz>();
    settings.wgtAmbientEnabled = std::make_shared<gui::Checkbox>("Ambient");
    settings.wgtAmbientEnabled->SetChecked(true);
    settings.wgtAmbientEnabled->SetOnChecked([this, renderScene](bool checked) {
        impl_->settings.SetCustomProfile();
        if (checked) {
            renderScene->SetIndirectLight(impl_->settings.hIbl);
        } else {
            renderScene->SetIndirectLight(IndirectLightHandle());
        }
    });
    checkboxes->AddChild(settings.wgtAmbientEnabled);
    settings.wgtSkyEnabled = std::make_shared<gui::Checkbox>("Sky");
    settings.wgtSkyEnabled->SetChecked(false);
    settings.wgtSkyEnabled->SetOnChecked([this, renderScene](bool checked) {
        impl_->settings.SetCustomProfile();
        if (checked) {
            renderScene->SetSkybox(impl_->settings.hSky);
        } else {
            renderScene->SetSkybox(SkyboxHandle());
        }
        impl_->scene->SetSkyboxHandle(impl_->settings.hSky, checked);
    });
    checkboxes->AddChild(settings.wgtSkyEnabled);
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

    settings.wgtAdvanced->AddChild(gui::Horiz::MakeFixed(separationHeight));

    // ....... ambient light (IBL)
    settings.wgtAmbientIBLs = std::make_shared<gui::Combobox>();
    std::vector<std::string> resourceFiles;
    utility::filesystem::ListFilesInDirectory(rsrcPath, resourceFiles);
    std::sort(resourceFiles.begin(), resourceFiles.end());
    int n = 0;
    for (auto &f : resourceFiles) {
        if (f.find("_ibl.ktx") == f.size() - 8) {
            auto name = utility::filesystem::GetFileNameWithoutDirectory(f);
            name = name.substr(0, name.size() - 8);
            settings.wgtAmbientIBLs->AddItem(name.c_str());
            if (name == "default") {
                settings.wgtAmbientIBLs->SetSelectedIndex(n);
            }
            n++;
        }
    }
    settings.wgtAmbientIBLs->AddItem("Custom...");
    settings.wgtAmbientIBLs->SetOnValueChanged([this](const char *name, int) {
        std::string path = gui::Application::GetInstance().GetResourcePath();
        path += std::string("/") + name + "_ibl.ktx";
        if (!this->SetIBL(path.c_str())) {
            // must be the "Custom..." option
            auto dlg = std::make_shared<gui::FileDialog>(
                    gui::FileDialog::Type::OPEN, "Open IBL", GetTheme());
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

    settings.wgtAmbientIntensity = MakeSlider(gui::Slider::INT, 0.0, 150000.0,
                                              lightingProfile.iblIntensity);
    settings.wgtAmbientIntensity->OnValueChanged =
            [this, renderScene](double newValue) {
                renderScene->SetIndirectLightIntensity(newValue);
                this->impl_->settings.SetCustomProfile();
            };

    auto ambientLayout = std::make_shared<gui::VGrid>(2, gridSpacing);
    ambientLayout->AddChild(std::make_shared<gui::Label>("IBL"));
    ambientLayout->AddChild(settings.wgtAmbientIBLs);
    ambientLayout->AddChild(std::make_shared<gui::Label>("Intensity"));
    ambientLayout->AddChild(settings.wgtAmbientIntensity);
    ambientLayout->AddChild(std::make_shared<gui::Label>("Skybox"));
    ambientLayout->AddChild(settings.wgtLoadSky);

    settings.wgtAdvanced->AddChild(std::make_shared<gui::Label>("Ambient"));
    settings.wgtAdvanced->AddChild(ambientLayout);
    settings.wgtAdvanced->AddChild(gui::Horiz::MakeFixed(separationHeight));

    // ... directional light (sun)
    settings.wgtSunIntensity = MakeSlider(gui::Slider::INT, 0.0, 500000.0,
                                          lightingProfile.sunIntensity);
    settings.wgtSunIntensity->OnValueChanged = [this,
                                                renderScene](double newValue) {
        renderScene->SetLightIntensity(impl_->settings.hDirectionalLight,
                                       newValue);
        this->impl_->settings.SetCustomProfile();
    };

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
    settings.wgtSunColor->OnValueChanged = [this, renderScene](
                                                   const gui::Color &newColor) {
        this->impl_->settings.SetCustomProfile();
        renderScene->SetLightColor(
                impl_->settings.hDirectionalLight,
                {newColor.GetRed(), newColor.GetGreen(), newColor.GetBlue()});
    };

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
    settings.wgtBase->AddChild(gui::Horiz::MakeFixed(separationHeight));
    auto materials = std::make_shared<gui::CollapsableVert>("Material settings",
                                                            0, indent);

    auto matGrid = std::make_shared<gui::VGrid>(2, gridSpacing);
    matGrid->AddChild(std::make_shared<gui::Label>("Type"));
    settings.wgtMaterialType.reset(
            new gui::Combobox({"Lit", "Unlit", "Normal map", "Depth"}));
    settings.wgtMaterialType->SetOnValueChanged([this, scene, renderScene](
                                                        const char *,
                                                        int selectedIdx) {
        using MaterialType = Impl::Settings::MaterialType;
        using ViewMode = visualization::View::Mode;
        auto selected = (Impl::Settings::MaterialType)selectedIdx;

        auto view = scene->GetView();
        impl_->settings.selectedType = selected;

        switch (selected) {
            case MaterialType::LIT:
                view->SetMode(ViewMode::Color);
                for (const auto &handle : impl_->geometryHandles) {
                    auto mat = impl_->geometryMaterials[handle].lit.handle;
                    renderScene->AssignMaterial(handle, mat);
                }
                break;
            case MaterialType::UNLIT:
                view->SetMode(ViewMode::Color);
                for (const auto &handle : impl_->geometryHandles) {
                    auto mat = impl_->geometryMaterials[handle].unlit.handle;
                    renderScene->AssignMaterial(handle, mat);
                }
                break;
            case MaterialType::NORMAL_MAP:
                view->SetMode(ViewMode::Normals);
                break;
            case MaterialType::DEPTH:
                view->SetMode(ViewMode::Depth);
                break;
        }

        impl_->settings.wgtPrefabMaterial->SetEnabled(selected ==
                                                      MaterialType::LIT);
    });
    matGrid->AddChild(settings.wgtMaterialType);

    settings.wgtPrefabMaterial = std::make_shared<gui::Combobox>();
    for (auto &prefab : impl_->prefabMaterials) {
        settings.wgtPrefabMaterial->AddItem(prefab.first.c_str());
    }
    settings.wgtPrefabMaterial->SetSelectedValue("Default");
    settings.wgtPrefabMaterial->SetOnValueChanged([this, renderScene](
                                                          const char *name,
                                                          int) {
        auto &renderer = this->GetRenderer();
        auto prefabIt = this->impl_->prefabMaterials.find(name);
        if (prefabIt != this->impl_->prefabMaterials.end()) {
            auto &prefab = prefabIt->second;
            for (const auto &handle : impl_->geometryHandles) {
                auto mat = impl_->geometryMaterials[handle].lit.handle;
                mat = renderer.ModifyMaterial(mat)
                              .SetColor("baseColor", prefab.baseColor)
                              .SetParameter("roughness", prefab.roughness)
                              .SetParameter("metallic", prefab.metallic)
                              .SetParameter("reflectance", prefab.reflectance)
                              .SetParameter("clearCoat", prefab.clearCoat)
                              .SetParameter("clearCoatRoughness",
                                            prefab.clearCoatRoughness)
                              .SetParameter("anisotropy", prefab.anisotropy)
                              // Point size is part of the material for
                              // rendering reasons, and therefore
                              // prefab.pointSize exists, but conceptually (and
                              // UI-wise) it is separate. So use the current
                              // setting instead of the prefab setting for point
                              // size.
                              .SetParameter("pointSize",
                                            float(impl_->settings.wgtPointSize
                                                          ->GetDoubleValue()))
                              .Finish();
                renderScene->AssignMaterial(handle, mat);
            }
        }
    });
    matGrid->AddChild(std::make_shared<gui::Label>("Material"));
    matGrid->AddChild(settings.wgtPrefabMaterial);

    matGrid->AddChild(std::make_shared<gui::Label>("Point size"));
    settings.wgtPointSize = MakeSlider(gui::Slider::INT, 0.0, 10.0, 3);
    settings.wgtPointSize->OnValueChanged = [this](double value) {
        auto &renderer = GetRenderer();
        for (const auto &pair : impl_->geometryMaterials) {
            renderer.ModifyMaterial(pair.second.lit.handle)
                    .SetParameter("pointSize", (float)value)
                    .Finish();
            renderer.ModifyMaterial(pair.second.unlit.handle)
                    .SetParameter("pointSize", (float)value)
                    .Finish();
        }

        renderer.ModifyMaterial(FilamentResourceManager::kDepthMaterial)
                .SetParameter("pointSize", (float)value)
                .Finish();
        renderer.ModifyMaterial(FilamentResourceManager::kNormalsMaterial)
                .SetParameter("pointSize", (float)value)
                .Finish();
    };
    matGrid->AddChild(settings.wgtPointSize);
    materials->AddChild(matGrid);

    settings.wgtBase->AddChild(materials);

    AddChild(settings.wgtBase);

    // Other items
    impl_->helpKeys = createHelpDisplay(this);
    impl_->helpKeys->SetVisible(false);
    AddChild(impl_->helpKeys);

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
    auto *scene3d = impl_->scene->GetScene();
    for (auto &h : impl_->geometryHandles) {
        scene3d->RemoveGeometry(h);
    }
    impl_->geometryHandles.clear();

    auto &renderer = GetRenderer();
    for (const auto &pair : impl_->geometryMaterials) {
        renderer.RemoveMaterialInstance(pair.second.unlit.handle);
        renderer.RemoveMaterialInstance(pair.second.lit.handle);
    }
    impl_->geometryMaterials.clear();

    geometry::AxisAlignedBoundingBox bounds;

    std::size_t nPointClouds = 0;
    for (auto &g : geometries) {
        Impl::Materials materials;
        materials.lit.handle =
                GetRenderer().AddMaterialInstance(impl_->hLitMaterial);
        materials.unlit.handle =
                GetRenderer().AddMaterialInstance(impl_->hUnlitMaterial);
        Impl::SetMaterialsDefaults(materials, GetRenderer());

        visualization::MaterialInstanceHandle selectedMaterial;

        switch (g->GetGeometryType()) {
            case geometry::Geometry::GeometryType::PointCloud: {
                nPointClouds++;
                auto pcd =
                        std::static_pointer_cast<const geometry::PointCloud>(g);

                if (pcd->HasColors()) {
                    selectedMaterial = materials.unlit.handle;

                    if (SmartMode::PointCloudHasUniformColor(*pcd)) {
                        selectedMaterial = materials.lit.handle;
                    }
                } else {
                    selectedMaterial = materials.lit.handle;
                }
            } break;
            case geometry::Geometry::GeometryType::LineSet: {
                selectedMaterial = materials.unlit.handle;
            } break;
            case geometry::Geometry::GeometryType::TriangleMesh: {
                auto mesh =
                        std::static_pointer_cast<const geometry::TriangleMesh>(
                                g);

                if (mesh->HasVertexColors()) {
                    selectedMaterial = materials.unlit.handle;
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

        auto viewMode = impl_->scene->GetView()->GetMode();
        if (viewMode == visualization::View::Mode::Normals) {
            impl_->settings.SetMaterialSelected(Impl::Settings::NORMAL_MAP);
        } else if (viewMode == visualization::View::Mode::Depth) {
            impl_->settings.SetMaterialSelected(Impl::Settings::DEPTH);
        } else {
            if (selectedMaterial == materials.unlit.handle) {
                impl_->settings.SetMaterialSelected(Impl::Settings::UNLIT);
            } else {
                impl_->settings.SetMaterialSelected(Impl::Settings::LIT);
            }
        }

        if (nPointClouds == geometries.size()) {
            impl_->SetLightingProfile(kPointCloudProfileName);
        }

        impl_->geometryMaterials.emplace(handle, materials);
    }

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

    // Settings in upper right
    const auto kLightSettingsWidth = 18 * em;
    auto lightSettingsSize = impl_->settings.wgtBase->CalcPreferredSize(theme);
    gui::Rect lightSettingsRect(r.width - kLightSettingsWidth, r.y,
                                kLightSettingsWidth, lightSettingsSize.height);
    impl_->settings.wgtBase->SetFrame(lightSettingsRect);

    Super::Layout(theme);
}

bool GuiVisualizer::SetIBL(const char *path) {
    auto newIBL = GetRenderer().AddIndirectLight(ResourceLoadRequest(path));
    if (newIBL) {
        auto *scene = impl_->scene->GetScene();
        impl_->settings.hIbl = newIBL;
        auto intensity = scene->GetIndirectLightIntensity();
        scene->SetIndirectLight(newIBL);
        scene->SetIndirectLightIntensity(intensity);

        std::string pathStr(path);
        auto idx = pathStr.find("ibl.ktx");
        if (idx != std::string::npos) {
            std::string skyboxPath = pathStr.substr(0, idx);
            skyboxPath += "skybox.ktx";
            auto skybox = GetRenderer().AddSkybox(
                    ResourceLoadRequest(skyboxPath.c_str()));
            impl_->scene->SetSkyboxHandle(
                    skybox, impl_->settings.wgtSkyEnabled->IsChecked());
            impl_->settings.hSky = skybox;
        } else {
        }
        return true;
    }
    return false;
}

bool GuiVisualizer::LoadGeometry(const std::string &path) {
    auto geometry = std::shared_ptr<geometry::Geometry3D>();

    auto mesh = std::make_shared<geometry::TriangleMesh>();
    bool meshSuccess = false;
    try {
        meshSuccess = io::ReadTriangleMesh(path, *mesh);
    } catch (...) {
        meshSuccess = false;
    }
    if (meshSuccess) {
        if (mesh->triangles_.size() == 0) {
            utility::LogWarning(
                    "Contains 0 triangles, will read as point cloud");
            mesh.reset();
        } else {
            mesh->ComputeVertexNormals();
            geometry = mesh;
        }
    } else {
        // LogError throws an exception, which we don't want, because this might
        // be a point cloud.
        utility::LogWarning("Failed to read %s", path.c_str());
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
            utility::LogInfof("Successfully read %s", path.c_str());
            if (!cloud->HasNormals()) {
                cloud->EstimateNormals();
            }
            cloud->NormalizeNormals();
            geometry = cloud;
        } else {
            utility::LogWarning("Failed to read points %s", path.c_str());
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
                    gui::FileDialog::Type::OPEN, "Open Geometry", GetTheme());
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
                    gui::FileDialog::Type::SAVE, "Save File", GetTheme());
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
        case FILE_CLOSE:
            this->Close();
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
