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

#include "open3d/Open3D.h"
#include "open3d/visualization/rendering/Camera.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/FilamentRenderer.h"

using namespace open3d;
using namespace open3d::visualization::gui;
using namespace open3d::visualization::rendering;

// Headless rendering requires Open3D to be compiled with OSMesa support.
// Add -DENABLE_HEADLESS_RENDERING=ON when you run CMake.
static const bool kUseHeadless = false;

static const std::string kOutputFilename = "offscreen.png";

int main(int argc, const char *argv[]) {
    const int width = 640;
    const int height = 480;

    auto &app = Application::GetInstance();
    app.Initialize(argc, argv);
    if (kUseHeadless) {
        EngineInstance::EnableHeadless();
    }

    auto *renderer =
            new FilamentRenderer(EngineInstance::GetInstance(), width, height,
                                 EngineInstance::GetResourceManager());
    auto *scene = new Open3DScene(*renderer);

    Material mat;
    mat.shader = "defaultLit";
    auto torus = open3d::geometry::TriangleMesh::CreateTorus();
    torus->ComputeVertexNormals();
    torus->PaintUniformColor({1.0f, 1.0f, 0.0f});
    scene->AddGeometry("torus", torus.get(), mat);
    scene->ShowAxes(true);
    scene->GetCamera()->SetProjection(60.0f, float(width) / float(height), 0.1f,
                                      10.0f, Camera::FovType::Vertical);
    scene->GetCamera()->LookAt({0.0f, 0.0f, 0.0f}, {3.0f, 3.0f, 3.0f},
                               {0.0f, 1.0f, 0.0f});

    // This example demonstrates rendering to an image without a window.
    // If you want to render to an image from within a window you should use
    // scene->GetScene()->RenderToImage() instead.
    Application::EnvUnlocker unlocker;
    auto img = app.RenderToImage(unlocker, scene->GetView(), scene->GetScene(),
                                 width, height);
    std::cout << "Writing file to " << kOutputFilename << std::endl;
    io::WriteImage(kOutputFilename, *img);

    // We manually delete these because Filament requires that things get
    // destructed in the right order.
    delete scene;
    delete renderer;

    // Cleanup Filament. Normally this is done by app.Run(), but since we not
    // using that we need to do it ourselves.
    app.OnTerminate();
}
