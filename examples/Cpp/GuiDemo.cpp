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

#include "Open3D/Open3D.h"

#include "Open3D/Visualization/Rendering/AbstractRenderInterface.h"
#include "Open3D/Visualization/Rendering/Camera.h"
#include "Open3D/Visualization/Rendering/CameraManipulator.h"
#include "Open3D/Visualization/Rendering/RendererStructs.h"
#include "Open3D/Visualization/Rendering/Scene.h"

#if !defined(WIN32)
#    include <unistd.h>
#else
#    include <io.h>
#endif
#include <fcntl.h>

using namespace open3d;

namespace {

static bool readBinaryFile(const std::string& path, std::vector<char> *bytes, std::string *errorStr)
{
    bytes->clear();
    if (errorStr) {
        *errorStr = "";
    }

    // Open file
    int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) {
//        if (errorStr) {
//            *errorStr = getIOErrorString(errno);
//        }
        return false;
    }

    // Get file size
    size_t filesize = (size_t)lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);  // reset file pointer back to beginning

    // Read data
    bytes->resize(filesize);
    read(fd, bytes->data(), filesize);

    // We're done, close and return
    close(fd);
    return true;
}

}

class DemoWindow : public gui::Window {
    using Super = Window;

    enum MenuIds {
        FILE_OPEN, FILE_SAVE, FILE_CLOSE,
        VIEW_POINTS, VIEW_WIREFRAME, VIEW_MESH,
        DEBUG_VOXINATED, DEBUG_SELUNA,
        HELP_ABOUT, HELP_CONTACT
    };

public:
    DemoWindow() : gui::Window("GuiDemo", 640, 480) {
        auto &theme = GetTheme();
        int spacing = theme.defaultLayoutSpacing;
        gui::Margins noMargins(0);
        gui::Margins margins(theme.defaultMargin);

        // Menu
        menubar_ = std::make_shared<gui::Menu>();
        auto fileMenu = std::make_shared<gui::Menu>();
        fileMenu->AddItem("Open", "Ctrl-O", FILE_OPEN);
        fileMenu->AddItem("Save", "Ctrl-S", FILE_SAVE);
        fileMenu->AddSeparator();
        fileMenu->AddItem("Close", "Ctrl-W", FILE_CLOSE);  // Ctrl-C is copy...
        menubar_->AddMenu("File", fileMenu);
        auto viewMenu = std::make_shared<gui::Menu>();
        viewMenu->AddItem("Points", "", VIEW_POINTS);
        viewMenu->AddItem("Wireframe", "", VIEW_WIREFRAME);
        viewMenu->AddItem("Mesh", "", VIEW_MESH);
        auto debugSubmenu = std::make_shared<gui::Menu>();
        debugSubmenu->AddItem("Voxinated", "", DEBUG_VOXINATED);
        debugSubmenu->AddItem("Seluna", "", DEBUG_SELUNA);
        viewMenu->AddMenu("Debug", debugSubmenu);
        menubar_->AddMenu("View", viewMenu);
        auto helpMenu = std::make_shared<gui::Menu>();
        helpMenu->AddItem("About", "", HELP_ABOUT);
        helpMenu->AddItem("Contact", "", HELP_CONTACT);
        menubar_->AddMenu("Help", helpMenu);
        SetMenubar(menubar_);
        this->OnMenuItemSelected = [this](gui::Menu::ItemId id) { this->OnMenuItem(id); };

        // Button grid (left panel)
        toolGrid_ = std::make_shared<gui::VGrid>(2, spacing, margins);
        AddChild(toolGrid_);

        MakeToolButton(toolGrid_, "B1", []() { std::cout << "B1 clicked" << std::endl; });
        MakeToolButton(toolGrid_, "B2", []() { std::cout << "B2 clicked" << std::endl; });
        MakeToolButton(toolGrid_, "B3", []() { std::cout << "B3 clicked" << std::endl; });
        MakeToolButton(toolGrid_, "B4", []() { std::cout << "B4 clicked" << std::endl; });
        MakeToolButton(toolGrid_, "B5", []() { std::cout << "B5 clicked" << std::endl; });
        MakeToolButton(toolGrid_, "B6", []() { std::cout << "B6 clicked" << std::endl; });

        // Right panel's tab control
        auto tabs = std::make_shared<gui::TabControl>();
        auto showBG = std::make_shared<gui::Checkbox>("Show background");
        auto useShadows = std::make_shared<gui::Checkbox>("Shadows");
        auto angle = std::make_shared<gui::Slider>(gui::Slider::DOUBLE);
        angle->SetLimits(0, 360);
        angle->SetValue(0);
        auto angle2 = std::make_shared<gui::Slider>(gui::Slider::INT);
        angle2->SetLimits(0, 360);
        angle2->SetValue(0);
        auto aaCombo = std::shared_ptr<gui::Combobox>(new gui::Combobox({"FSAA", "Quincuz", "None"}));
        useShadows->OnChecked =  [](bool isChecked) { std::cout << "Shadows: " << isChecked << std::endl; };
        auto viewPanel = std::make_shared<gui::Vert>(spacing, margins,
                                                     std::vector<std::shared_ptr<gui::Widget>>({
            aaCombo,
            showBG,
            useShadows,
            angle,
            angle2
        }));
        auto models = std::shared_ptr<gui::Combobox>(new gui::Combobox({"Teapot", "Sphere", "Cube"}));
        models->SetSelectedIndex(1);
        auto modelsPanel = std::make_shared<gui::Vert>(spacing, margins,
                                                       std::vector<std::shared_ptr<gui::Widget>>({
            models
        }));
/*        auto materials = std::make_shared<gui::Combobox>();
        materials->AddItem("Spiffy");
        materials->AddItem("Gold");
        materials->AddItem("Red plastic");
        materials->AddItem("Blue ceramic"); */
        auto materials = std::shared_ptr<gui::Combobox>(new gui::Combobox({"Gold", "Red plastic", "Blue ceramic"}));
        materials->OnValueChanged = [](const char *newValue) {
            std::cout << "New material: " << newValue << std::endl;
        };
        auto materials2 = std::shared_ptr<gui::Combobox>(new gui::Combobox({"One", "Two", "Three"}));
//        auto materials3 = std::shared_ptr<gui::Combobox>(new gui::Combobox({"一", "二", "三"}));
        auto matPanel = std::make_shared<gui::Vert>(spacing, margins,
                                                    std::vector<std::shared_ptr<gui::Widget>>({
            materials,
            materials2,
//            materials3
        }));
        tabs->AddTab("View", viewPanel);
        tabs->AddTab("Models", modelsPanel);
        tabs->AddTab("Materials", matPanel);

        // Right panel
        auto title = std::make_shared<gui::Label>("Info");
        auto nVerts = std::make_shared<gui::Label>("248572 vertices");
        auto textEdit = std::make_shared<gui::TextEdit>();
        textEdit->SetPlaceholderText("Edit some text");
        textEdit->OnTextChanged = [](const char *text) { std::cout << "Text changed: '" << text << "'" << std::endl; };
        textEdit->OnValueChanged = [](const char *newValue) { std::cout << "Text value changed: '" << newValue << "'" << std::endl; };
        rightPanel_ = std::make_shared<gui::Vert>(0, noMargins,
                                                  std::vector<std::shared_ptr<gui::Widget>>(
        {
            std::make_shared<gui::Vert>(spacing, margins,
                                        std::vector<std::shared_ptr<gui::Widget>>(
            {
                std::make_shared<gui::Horiz>(0, gui::Margins(0),
                                             std::vector<std::shared_ptr<gui::Widget>>(
                   { gui::Horiz::MakeStretch(), title, gui::Horiz::MakeStretch() })),
                nVerts,
                textEdit
            })),
            gui::Vert::MakeStretch(),
            tabs
        }));
        AddChild(rightPanel_);

        // Create materials
        visualization::MaterialHandle nonmetal;

        std::string errorStr;
        std::vector<char> materialData;
        std::string resourcePath = gui::Application::GetInstance().GetResourcePath();

        if (!readBinaryFile(resourcePath + "/nonmetal.filamat", &materialData, &errorStr)) {
            std::cout << "WARNING: Could not read non metal material" << "(" << errorStr << ")."
                      << "Will use default material instead." << std::endl;
        } else {
            nonmetal = GetRenderer().AddMaterial(materialData.data(), materialData.size());
        }

        auto redPlastic = GetRenderer().ModifyMaterial(nonmetal)
                .SetColor("baseColor", {0.8, 0.0, 0.0})
                .SetParameter("roughness", 0.5f)
                .SetParameter("clearCoat", 1.f)
                .SetParameter("clearCoatRoughness", 0.3f)
                .Finish();

        auto blueCeramic = GetRenderer().ModifyMaterial(nonmetal)
                .SetColor("baseColor", {0.537, 0.812, 0.941})
                .SetParameter("roughness", 0.5f)
                .SetParameter("clearCoat", 1.f)
                .SetParameter("clearCoatRoughness", 0.01f)
                .Finish();

        auto green = GetRenderer().ModifyMaterial(nonmetal)
                .SetColor("baseColor", {0.537, 0.941, 0.6})
                .SetParameter("roughness", 0.25f)
                .SetParameter("clearCoat", 0.f)
                .SetParameter("clearCoatRoughness", 0.01f)
                .Finish();

        auto white = GetRenderer().ModifyMaterial(nonmetal)
                .SetColor("baseColor", {1.0, 1.0, 1.0})
                .SetParameter("roughness", 0.5f)
                .SetParameter("clearCoat", 1.f)
                .SetParameter("clearCoatRoughness", 0.3f)
                .Finish();

        // Create scene
        auto sceneId = GetRenderer().CreateScene();
        sceneWidget_ = std::make_shared<gui::SceneWidget>(*GetRenderer().GetScene(sceneId));
        sceneWidget_->SetBackgroundColor(gui::Color(0.5, 0.5, 1.0));

        sceneWidget_->GetCameraManipulator()->SetFov(90.0f);
        sceneWidget_->GetCameraManipulator()->SetNearPlane(0.1f);
        sceneWidget_->GetCameraManipulator()->SetFarPlane(50.0f);
        sceneWidget_->GetCameraManipulator()->LookAt({0, 0, 0},   {-2, 10, 10});

        // Create light
        visualization::LightDescription lightDescription;
        lightDescription.intensity = 100000;
        lightDescription.direction = { -0.707, -.707, 0.0 };
        lightDescription.customAttributes["custom_type"] = "SUN";

        sceneWidget_->GetScene()->AddLight(lightDescription);

        auto sphere = geometry::TriangleMesh::CreateBox(2,2,2);
        sphere->ComputeVertexNormals();

        // Add geometry
        {
            using Transform = visualization::Scene::Transform;

            Transform t = Transform::Identity();
            auto whiteSphere = sceneWidget_->GetScene()->AddGeometry(*sphere, white);
            sceneWidget_->GetScene()->SetEntityTransform(whiteSphere, t);

            t = Transform::Identity();
            t.translate(Eigen::Vector3f(2.f, 0, 0));
            auto redSphere = sceneWidget_->GetScene()->AddGeometry(*sphere, redPlastic);
            sceneWidget_->GetScene()->SetEntityTransform(redSphere, t);

            t = Transform::Identity();
            t.translate(Eigen::Vector3f(0, 2.f, 0));
            auto greenSphere = sceneWidget_->GetScene()->AddGeometry(*sphere, green);
            sceneWidget_->GetScene()->SetEntityTransform(greenSphere, t);

            t = Transform::Identity();
            t.translate(Eigen::Vector3f(0, 0, 2.f));
            auto blueSphere = sceneWidget_->GetScene()->AddGeometry(*sphere, blueCeramic);
            sceneWidget_->GetScene()->SetEntityTransform(blueSphere, t);
        }

        AddChild(sceneWidget_);
    }

    void OnMenuItem(gui::Menu::ItemId id) {
        switch (id) {
            case FILE_CLOSE:
                this->Close(); break;
            case DEBUG_VOXINATED:
            case DEBUG_SELUNA:
            {
                bool checked = menubar_->IsChecked(id);
                menubar_->SetChecked(id, !checked);
                break;
            }
            default:
                break;
        }
    }

protected:
    void Layout(const gui::Theme& theme) override {
        auto contentRect = GetContentRect();

        gui::Rect leftRect(contentRect.x, contentRect.y,
                           toolGrid_->CalcPreferredSize(theme).width,
                           contentRect.height);
        toolGrid_->SetFrame(leftRect);

        auto rightSize = rightPanel_->CalcPreferredSize(theme);
        gui::Rect rightRect(contentRect.width - rightSize.width,
                            contentRect.y,
                            rightSize.width,
                            contentRect.height);
        rightPanel_->SetFrame(rightRect);

        sceneWidget_->SetFrame(gui::Rect(leftRect.GetRight(), contentRect.y,
                                   rightRect.x - leftRect.GetRight(),
                                   contentRect.height));

        Super::Layout(theme);
    }

private:
    void MakeToolButton(std::shared_ptr<gui::Widget> parent,
                        const char *name, std::function<void()> onClicked) {
        std::shared_ptr<gui::Button> b;
        b = std::make_shared<gui::Button>(name);
        b->OnClicked = onClicked;
        parent->AddChild(b);
    }

private:
    std::shared_ptr<gui::Menu> menubar_;
    std::shared_ptr<gui::SceneWidget> sceneWidget_;
    std::shared_ptr<gui::VGrid> toolGrid_;
    std::shared_ptr<gui::Vert> rightPanel_;
};

int main(int argc, const char *argv[]) {
    auto &app = gui::Application::GetInstance();
    app.Initialize(argc, argv);

    auto w = std::make_shared<DemoWindow>();
    app.AddWindow(w);
    w->Show();

    app.Run();    
    return 0;
}
