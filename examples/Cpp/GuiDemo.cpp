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
public:
    DemoWindow() : gui::Window("GuiDemo", 640, 480) {
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

        const float frustumNear = 0.1f;
        const float frustumFar = 50.0f;
        const float frustumFov = 90.0f;
        scene_->GetCamera().SetProjection(frustumNear, frustumFar,
                                          frustumFov);
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

protected:
    void Layout() override {
        auto windowSize = GetSize();
        scene_->SetFrame(gui::Rect(0, 0, windowSize.width, windowSize.height));
    }

private:
    std::shared_ptr<gui::SceneWidget> scene_;
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
