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

#include "Open3DViewer.h"

#include <string>

#include "open3d/Open3D.h"
#include "open3d/visualization/gui/Native.h"
#include "open3d/visualization/visualizer/O3DVisualizer.h"

using namespace open3d;
using namespace open3d::geometry;
using namespace open3d::visualization;

namespace {
static const std::string gUsage = "Usage: Open3DViewer [meshfile|pointcloud]";
static const int FILE_OPEN = 100;

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

}  // namespace

Open3DViewer::Open3DViewer(const std::string &title, int width, int height)
    : visualizer::O3DVisualizer(title, width, height) {
    ShowSettings(true);

    auto file_menu = GetFileMenu();
    file_menu.menu->InsertItem(file_menu.insertion_idx++, "Open...", FILE_OPEN,
                               gui::KEY_O);
}

void Open3DViewer::LoadGeometry(const std::string &path) {
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

        ClearGeometry();

        auto geometry_type = io::ReadFileGeometryType(path);

        auto model = std::make_shared<rendering::TriangleMeshModel>();
        bool model_success = false;
        if (geometry_type & io::CONTAINS_TRIANGLES) {
            const float ioProgressAmount = 1.0f;
            try {
                io::ReadTriangleModelOptions opt;
                opt.update_progress = [ioProgressAmount,
                                       UpdateProgress](double percent) -> bool {
                    UpdateProgress(ioProgressAmount * float(percent / 100.0));
                    return true;
                };
                model_success = io::ReadTriangleModel(path, *model, opt);
            } catch (...) {
                model_success = false;
            }
        }
        if (!model_success) {
            utility::LogInfo("{} appears to be a point cloud", path.c_str());
        }

        auto geometry = std::shared_ptr<geometry::Geometry3D>();
        if (!model_success) {
            auto cloud = std::make_shared<geometry::PointCloud>();
            bool success = false;
            const float ioProgressAmount = 0.5f;
            try {
                io::ReadPointCloudOption opt;
                opt.update_progress = [ioProgressAmount,
                                       UpdateProgress](double percent) -> bool {
                    UpdateProgress(ioProgressAmount * float(percent / 100.0));
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
                UpdateProgress(0.666f);
                cloud->NormalizeNormals();
                UpdateProgress(0.75f);
                geometry = cloud;
            } else {
                utility::LogWarning("Failed to read points {}", path.c_str());
                cloud.reset();
            }
        }

        if (model_success || geometry) {
            gui::Application::GetInstance().PostToMainThread(
                    this, [this, path, model_success, model, geometry]() {
                        auto model_name = utility::filesystem::
                                GetFileNameWithoutDirectory(path);
                        auto scene = GetScene();
                        scene->ClearGeometry();
                        SetModelUp(rendering::Open3DScene::UpDir::PLUS_Y);
                        if (model_success) {
                            AddGeometry(model_name, model);
                        } else {
                            // If a point cloud has colors, assume that these
                            // colors are the lit values (since they probably
                            // come from an actual photograph), and make this
                            // unlit so that we don't re-light pixels. (Note
                            // that 'geometry' should always be a point cloud
                            // here.) But if the vertices are all white, use the
                            // normals if they exist, otherwise the cloud will
                            // be white on white and effectively invisible.
                            auto cloud = std::dynamic_pointer_cast<
                                    geometry::PointCloud>(geometry);
                            if (cloud && cloud->HasNormals() &&
                                !PointCloudHasUniformColor(*cloud)) {
                                rendering::Material mat;
                                mat.shader = "defaultUnlit";
                                AddGeometry(model_name, cloud, &mat);
                            } else {
                                AddGeometry(model_name, geometry);
                            }
                            if (cloud) {
                                SetLightingProfile(
                                        rendering::Open3DScene::
                                                LightingProfile::NO_SHADOWS);
                            }
                        }
                        ResetCameraToDefault();
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

void Open3DViewer::OnMenuItemSelected(gui::Menu::ItemId item_id) {
    if (item_id == FILE_OPEN) {
        OnFileOpen();
    } else {
        Super::OnMenuItemSelected(item_id);
    }
}

void Open3DViewer::OnFileOpen() {
    auto dlg = std::make_shared<gui::FileDialog>(gui::FileDialog::Mode::OPEN,
                                                 "Open Geometry", GetTheme());
    dlg->AddFilter(".ply .stl .fbx .obj .off .gltf .glb",
                   "Triangle mesh files (.ply, .stl, .fbx, .obj, .off, "
                   ".gltf, .glb)");
    dlg->AddFilter(".xyz .xyzn .xyzrgb .ply .pcd .pts",
                   "Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, "
                   ".pcd, .pts)");
    dlg->AddFilter(".ply", "Polygon files (.ply)");
    dlg->AddFilter(".stl", "Stereolithography files (.stl)");
    dlg->AddFilter(".fbx", "Autodesk Filmbox files (.fbx)");
    dlg->AddFilter(".obj", "Wavefront OBJ files (.obj)");
    dlg->AddFilter(".off", "Object file format (.off)");
    dlg->AddFilter(".gltf", "OpenGL transfer files (.gltf)");
    dlg->AddFilter(".glb", "OpenGL binary transfer files (.glb)");
    dlg->AddFilter(".xyz", "ASCII point cloud files (.xyz)");
    dlg->AddFilter(".xyzn", "ASCII point cloud with normals (.xyzn)");
    dlg->AddFilter(".xyzrgb", "ASCII point cloud files with colors (.xyzrgb)");
    dlg->AddFilter(".pcd", "Point Cloud Data files (.pcd)");
    dlg->AddFilter(".pts", "3D Points files (.pts)");
    dlg->AddFilter("", "All files");
    dlg->SetOnCancel([this]() { this->CloseDialog(); });
    dlg->SetOnDone([this](const char *path) {
        this->CloseDialog();
        OnDragDropped(path);
    });
    ShowDialog(dlg);
}

void Open3DViewer::OnDragDropped(const char *path) {
    auto title = std::string("Open3D - ") + path;
    SetTitle(title.c_str());
    LoadGeometry(path);
}

int Run(int argc, const char *argv[]) {
    const char *path = nullptr;
    if (argc > 1) {
        path = argv[1];
        if (argc > 2) {
            utility::LogWarning(gUsage.c_str());
        }
    }

    auto &app = gui::Application::GetInstance();
    app.Initialize(argc, argv);

    auto vis = std::make_shared<Open3DViewer>("Open3D", WIDTH, HEIGHT);
    bool is_path_valid = (path && path[0] != '\0');
    if (is_path_valid) {
        vis->LoadGeometry(path);
    }
    gui::Application::GetInstance().AddWindow(vis);
    // when Run() ends, Filament will be stopped, so we can't be holding on
    // to any GUI objects.
    vis.reset();

    app.Run();

    return 0;
}

#if __APPLE__
// Open3DViewer_mac.mm
#else
int main(int argc, const char *argv[]) { return Run(argc, argv); }
#endif  // __APPLE__

// Testing:
// There is a fair a number of different outcomes that are expected, based on
// the mesh type:
//
// Point clouds:
// - has colors: unlit (assumes lighting is backed into vertex colors),
//               low contrast lighting. [caterpillar.ply]
// - no colors: lit, low contrast lighting. [office.ply]
//
// Triangle meshes
// - normals, no colors:  lit, automatically colored white
// - normals, colors: lit
//
// Stanford meshs:
//   fountain.ply, cactusgarden.ply, readingroom.ply, stonewall.ply
//   These should "look good" be default. The difficulty is that the up
//   direction is -y, which few other meshes seem to have.
