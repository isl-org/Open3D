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

#include <vector>

#include "Open3D/Open3D.h"

int main(int argc, char *argv[]) {
    using namespace open3d;
    using namespace open3d::utility::filesystem;
    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc != 2) {
        utility::LogInfo("Usage :\n");
        utility::LogInfo(">    ColorMapOptimization data_dir\n");
        return 1;
    }
    // Read RGBD images
    std::string data_path = argv[1];
    std::vector<std::string> depth_filenames, color_filenames;
    ListFilesInDirectoryWithExtension(data_path + "/depth/", "png",
                                      depth_filenames);
    ListFilesInDirectoryWithExtension(data_path + "/image/", "jpg",
                                      color_filenames);
    assert(depth_filenames.size() == color_filenames.size());
    std::vector<std::shared_ptr<geometry::RGBDImage>> rgbd_images;
    for (size_t i = 0; i < depth_filenames.size(); i++) {
        utility::LogDebug("reading {}...\n", depth_filenames[i]);
        auto depth = io::CreateImageFromFile(depth_filenames[i]);
        utility::LogDebug("reading {}...\n", color_filenames[i]);
        auto color = io::CreateImageFromFile(color_filenames[i]);
        auto rgbd_image = geometry::RGBDImage::CreateFromColorAndDepth(
                *color, *depth, 1000.0, 3.0, false);
        rgbd_images.push_back(rgbd_image);
    }
    auto camera = io::CreatePinholeCameraTrajectoryFromFile(data_path +
                                                            "/scene/key.log");
    auto mesh = io::CreateMeshFromFile(data_path + "/scene/integrated.ply");

    // Optimize texture and save the mesh as texture_mapped.ply
    // This is implementation of following paper
    // Q.-Y. Zhou and V. Koltun,
    // Color Map Optimization for 3D Reconstruction with Consumer Depth Cameras,
    // SIGGRAPH 2014
    color_map::ColorMapOptimizationOption option;
    option.maximum_iteration_ = 300;
    option.non_rigid_camera_coordinate_ = true;
    color_map::ColorMapOptimization(*mesh, rgbd_images, *camera, option);
    io::WriteTriangleMesh("color_map_after_optimization.ply", *mesh);

    return 0;
}
