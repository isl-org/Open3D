// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
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
static const bool kUseHeadless [[maybe_unused]] = false;

static const std::string kOutputFilename = "offscreen.png";

int main(int argc, const char *argv[]) {
    const int width = 640;
    const int height = 480;

    auto &app = Application::GetInstance();
    app.Initialize(argc, argv);

    auto *renderer =
            new FilamentRenderer(EngineInstance::GetInstance(), width, height,
                                 EngineInstance::GetResourceManager());
    auto *scene = new Open3DScene(*renderer);

    MaterialRecord mat;
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
    auto img = app.RenderToImage(*renderer, scene->GetView(), scene->GetScene(),
                                 width, height);
    std::cout << "Writing file to " << kOutputFilename << std::endl;
    io::WriteImage(kOutputFilename, *img);

    // We manually delete these because Filament requires that things get
    // destructed in the right order.
    delete scene;
    delete renderer;

    // Cleanup Filament. Normally this is done by app.Run(), but since we are
    // not using that we need to do it ourselves.
    app.OnTerminate();
}
