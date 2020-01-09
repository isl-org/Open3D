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

#include "Open3D/GUI/Native.h"
#include "Open3D/Open3D.h"
#include "Open3D/Visualization/Rendering/Camera.h"
#include "Open3D/Visualization/Rendering/Filament/FilamentEngine.h"
#include "Open3D/Visualization/Rendering/Filament/FilamentRenderer.h"
#include "Open3D/Visualization/Rendering/Filament/FilamentResourceManager.h"
#include "Open3D/Visualization/Rendering/RendererHandle.h"
#include "Open3D/Visualization/Rendering/Scene.h"
#include "Open3D/Visualization/Rendering/View.h"

#include <SDL.h>
#if !defined(WIN32)
#include <unistd.h>
#else
#include <io.h>
#endif
#include <fcntl.h>

static bool readBinaryFile(const std::string& path,
                           std::vector<char>* bytes,
                           std::string* errorStr) {
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

void PrintHelp() {
    using namespace open3d;
    utility::LogInfo("Usage :");
    utility::LogInfo("    > FilamentDemo sphere <material file>");
    utility::LogInfo("    > FilamentDemo mesh <mesh file> <material file>");
}

int main(int argc, char* argv[]) {
    using namespace open3d;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc < 2) {
        PrintHelp();
        return 1;
    }

    std::shared_ptr<geometry::TriangleMesh> mesh;
    int materialPathIndex = 2;

    std::string option(argv[1]);
    if (option == "sphere") {
        mesh = geometry::TriangleMesh::CreateSphere(42);
        mesh->ComputeVertexNormals();
    } else if (option == "mesh") {
        if (argc < 3) {
            std::cout << "ERROR: You need to provide path to mesh file"
                      << std::endl;
            return 2;
        }

        mesh = std::make_shared<geometry::TriangleMesh>();
        if (!io::ReadTriangleMesh(argv[2], *mesh)) {
            std::cout << "ERROR: Failed to load mesh from " << argv[2]
                      << std::endl;
            return 2;
        }

        materialPathIndex = 3;
    } else {
        std::cout << "ERROR: Unknown option \'" << option << "\'" << std::endl;
        PrintHelp();
        return 1;
    }

    mesh->ComputeVertexNormals();

    bool materialDataLoaded = false;
    std::vector<char> materialData;

    if (argc > materialPathIndex) {
        const std::string pathToMaterial = argv[materialPathIndex];

        std::string errorStr;
        materialDataLoaded =
                readBinaryFile(pathToMaterial, &materialData, &errorStr);
        if (!materialDataLoaded) {
            std::cout << "WARNING: Could not read " << pathToMaterial << "("
                      << errorStr << ")."
                      << "Will use default material instead." << std::endl;
        }
    } else {
        std::cout << "WARNING: No path to material provided, using default "
                     "material..."
                  << std::endl;
    }

    const int x = SDL_WINDOWPOS_CENTERED;
    const int y = SDL_WINDOWPOS_CENTERED;
    const int w = 1280;
    const int h = 720;
    uint32_t flags =
            SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI;
    auto* window =
            SDL_CreateWindow("triangle mesh filament", x, y, w, h, flags);

    auto& engineInstance = visualization::EngineInstance::GetInstance();
    auto& resourceManager = visualization::EngineInstance::GetResourceManager();
    auto renderer = new visualization::FilamentRenderer(
            engineInstance, (void*)open3d::gui::GetNativeDrawable(window),
            resourceManager);

    SDL_ShowWindow(window);

    SDL_Init(SDL_INIT_EVENTS);

    auto sceneId = renderer->CreateScene();
    auto scene = renderer->GetScene(sceneId);

    auto viewId = scene->AddView(0, 0, w, h);
    auto view = scene->GetView(viewId);
    view->SetClearColor({0.5f, 0.5f, 1.f});

    auto camera = view->GetCamera();
    camera->SetProjection(90, float(w) / float(h), 0.01, 1000,
                          visualization::Camera::FovType::Horizontal);
    camera->LookAt({0, 0, 0}, {80, 80, 80}, {0, 1, 0});

    visualization::MaterialInstanceHandle matInstance;
    if (materialDataLoaded) {
        visualization::MaterialHandle matId =
                renderer->AddMaterial(materialData.data(), materialData.size());

        matInstance = renderer->ModifyMaterial(matId)
                              .SetParameter("roughness", 0.5f)
                              .SetParameter("clearCoat", 1.0f)
                              .SetParameter("clearCoatRoughness", 0.3f)
                              .SetColor("baseColor", {1.f, 0.f, 0.f})
                              .Finish();
    }

    visualization::LightDescription lightDescription;
    lightDescription.intensity = 100000;
    lightDescription.direction = {-0.707, -.707, 0.0};
    lightDescription.customAttributes["custom_type"] = "SUN";

    scene->AddLight(lightDescription);

    scene->AddGeometry(*mesh, matInstance);

    SDL_EventState(SDL_DROPFILE, SDL_ENABLE);
    while (true) {
        bool isDone = false;

        constexpr int kMaxEvents = 16;
        SDL_Event events[kMaxEvents];
        int nevents = 0;
        while (nevents < kMaxEvents && SDL_PollEvent(&events[nevents]) != 0) {
            const SDL_Event& event = events[nevents];
            switch (event.type) {
                case SDL_QUIT:  // sent after last window closed
                    isDone = true;
                    break;
            }

            ++nevents;
        }

        renderer->BeginFrame();
        renderer->Draw();
        renderer->EndFrame();

        SDL_Delay(10);  // millisec

        if (isDone) break;
    }

    delete renderer;
    resourceManager.DestroyAll();

    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}