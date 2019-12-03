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
#include "Open3D/Visualization/Rendering/Filament/FilamentRenderer.h"
#include "Open3D/Visualization/Rendering/RendererHandle.h"
#include "Open3D/GUI/Native.h"

#include <SDL.h>
#if !defined(WIN32)
#    include <unistd.h>
#else
#    include <io.h>
#endif
#include <sys/errno.h>
#include <fcntl.h>

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

void PrintHelp() {
    using namespace open3d;
    utility::LogInfo("Usage :");
    utility::LogInfo("    > FilamentDemo sphere <material file>");
    utility::LogInfo("    > FilamentDemo mesh <mesh file> <material file>");
}

int main(int argc, char *argv[]) {
    using namespace open3d;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc < 2) {
        PrintHelp();
        return 1;
    }

    std::string option(argv[1]);
    if (option == "sphere") {
        // auto mesh = geometry::TriangleMesh::CreateSphere(1);
        // mesh->ComputeVertexNormals();

        auto mesh = std::make_shared<geometry::TriangleMesh>();
        if (io::ReadTriangleMesh(
                    "/home/maks/Work/code/Open3D/examples/TestData/knot.ply",
                    *mesh)) {
        } else {
            return 1;
        }
        mesh->ComputeVertexNormals();

        std::string pathToMaterial;
        if (argc > 2) {
            pathToMaterial = argv[2];
        }

        std::vector<char> bytes;
        std::string errorStr;
        if (!readBinaryFile(pathToMaterial, &bytes, &errorStr)) {
            std::cout << "[WARNING] Could not read " << pathToMaterial << "(" << errorStr << ")."
                      << "Will use default material instead" << std::endl;
        }

        visualization::MaterialHandle matId = visualization::TheRenderer->AddMaterial(bytes.data(), bytes.size());
        auto matInstance = visualization::TheRenderer->ModifyMaterial(matId)
            .SetParameter("metallic", 1.0f)
            .SetParameter("roughness", 1.0f)
            .SetParameter("anisotropy", 1.0f)
            .SetColor("baseColor", {1.f, 0.f, 0.f})
            .Finish();

        const int x = SDL_WINDOWPOS_CENTERED;
        const int y = SDL_WINDOWPOS_CENTERED;
        uint32_t flags = SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE |
                         SDL_WINDOW_ALLOW_HIGHDPI;
        auto* window = SDL_CreateWindow("triangle mesh filament", x, y, 1280,
                                        720, flags);

        visualization::FilamentRenderer::InitGlobal(
                (void*)open3d::gui::GetNativeDrawable(window));

        SDL_ShowWindow(window);

        SDL_Init(SDL_INIT_EVENTS);

        visualization::TheRenderer->AddGeometry(*mesh, matInstance);

        SDL_EventState(SDL_DROPFILE, SDL_ENABLE);
        while (true) {
            bool isDone = false;

            constexpr int kMaxEvents = 16;
            SDL_Event events[kMaxEvents];
            int nevents = 0;
            while (nevents < kMaxEvents &&
                   SDL_PollEvent(&events[nevents]) != 0) {
                const SDL_Event& event = events[nevents];
                switch (event.type) {
                    case SDL_QUIT:  // sent after last window closed
                        isDone = true;
                        break;
                }

                ++nevents;
            }

            visualization::TheRenderer->Draw();

            SDL_Delay(10);  // millisec

            if (isDone) break;
        }

        visualization::FilamentRenderer::ShutdownGlobal();

        SDL_DestroyWindow(window);
        SDL_Quit();
    }

    return 0;
}