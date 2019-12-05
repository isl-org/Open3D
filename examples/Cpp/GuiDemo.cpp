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

using namespace open3d;

namespace {

struct Geometry {
    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<float> indices;

    void AddVertex(float x, float y, float z) {
        vertices.push_back(x);
        vertices.push_back(y);
        vertices.push_back(z);
    }
    void AddNormal(float x, float y, float z) {
        normals.push_back(x);
        normals.push_back(y);
        normals.push_back(z);
    }
};

enum SphereDetail { NORMAL = 40, MILLION = 1000, TEN_MILLION = 3162 };
gui::Renderer::GeometryId createSphereGeometry(gui::Renderer& renderer,
                                               SphereDetail detail = NORMAL) {
    int N = int(detail);
    float R = 1.0;
    // We could use fewer vertices by reusing the north and south poles to
    // be triangle fans, and reusing the vertices where the strips join at
    // longitude 0, but that would make the code more complex, and this is
    // just exploration code.
    int nVerts = (N + 1) * (N + 1);
    int nTris = 2 * N * N;

    Geometry g;
    g.vertices.reserve(3 * nVerts);
    g.normals.reserve(3 * nVerts);
    for (int y = 0;  y <= N;  ++y) {
        for (int i = 0;  i <= N;  ++i) {
            float lat = -M_PI / 2.0 + M_PI * (float(y) / float(N));
            float lng = 2.0 * M_PI * (float(i) / float(N));
            float x = std::cos(lng) * std::cos(lat);
            float y = std::sin(lat);
            float z = std::sin(lng) * std::cos(lat);
            g.AddVertex(R * x, R * y, R * z);
            g.AddNormal(x, y, z);
        }
    }

    std::vector<uint32_t> indices;
    indices.reserve(3 * nTris);
    for (int y = 0;  y < N;  ++y) {
        int latStartVIdx = y * (N + 1);
        int nextLatStartVIdx = (y + 1) * (N + 1);
        for (int i = 0;  i < N;  ++i) {
            indices.push_back(latStartVIdx + i);
            indices.push_back(nextLatStartVIdx + i);
            indices.push_back(latStartVIdx + i + 1);

            indices.push_back(latStartVIdx + i + 1);
            indices.push_back(nextLatStartVIdx + i);
            indices.push_back(nextLatStartVIdx + i + 1);
        }
    }

    return renderer.CreateGeometry(g.vertices, g.normals, indices,
                                   gui::BoundingBox(0, 0, 0, R));
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
        auto redPlastic = GetRenderer().CreateNonMetal(gui::Color(0.8, 0.0, 0.0),
                                                       0.5f, // roughness
                                                       1.0f, // clear coat
                                                       0.3f);// clear coat roughness

        auto blueCeramic = GetRenderer().CreateNonMetal(gui::Color(0.537, 0.812, 0.941),
                                                        0.5f, 1.0f, 0.01f);

        auto green = GetRenderer().CreateNonMetal(gui::Color(0.537, 0.941, 0.6),
                                                  0.25f, 0.0f, 0.01f);

        auto white = GetRenderer().CreateNonMetal(gui::Color(1.0, 1.0, 1.0),
                                                       0.5f, // roughness
                                                       1.0f, // clear coat
                                                       0.3f);// clear coat roughness

        // Create scene
        scene_ = std::make_shared<gui::SceneWidget>(GetRenderer());
        scene_->SetBackgroundColor(gui::Color(0.5, 0.5, 1.0));

        const float near = 0.1f;
        const float far = 50.0f;
        const float fov = 90.0f;
        scene_->GetCamera().SetProjection(near, far, fov);
        scene_->GetCamera().LookAt(0, 0, 5,   0, 0, 0,   0, 1, 0);

        auto sun = GetRenderer().CreateSunLight(-0.707, -.707, 0.0);
        scene_->AddLight(sun);

        // Add geometry
        auto geometry = createSphereGeometry(GetRenderer(), NORMAL);
        auto mesh = GetRenderer().CreateMesh(geometry, white);
        scene_->AddMesh(mesh);

        mesh = GetRenderer().CreateMesh(geometry, redPlastic);
        scene_->AddMesh(mesh, 2, 0, 0);

        mesh = GetRenderer().CreateMesh(geometry, green);
        scene_->AddMesh(mesh, 0, 2, 0);

        mesh = GetRenderer().CreateMesh(geometry, blueCeramic);
        scene_->AddMesh(mesh, 0, 0, 2);

        AddChild(scene_);
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

        scene_->SetFrame(gui::Rect(leftRect.GetRight(), contentRect.y,
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
    std::shared_ptr<gui::SceneWidget> scene_;
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
