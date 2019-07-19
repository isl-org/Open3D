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

#include "Open3D/Integration/UniformTSDFVolume.h"
#include "Open3D/Camera/PinholeCameraIntrinsic.h"
#include "Open3D/Geometry/RGBDImage.h"
#include "Open3D/IO/ClassIO/ImageIO.h"
#include "Open3D/Visualization/Utility/DrawGeometry.h"
#include "TestUtility/UnitTest.h"

#include <sstream>

using namespace open3d;
using namespace unit_test;

bool ReadPoses(const std::string& trajectory_path,
               std::vector<Eigen::Matrix4d>& poses) {
    FILE* f = fopen(trajectory_path.c_str(), "r");
    if (f == NULL) {
        utility::LogWarning("Read poses failed: unable to open file: {}\n",
                            trajectory_path);
        return false;
    }
    char line_buffer[DEFAULT_IO_BUFFER_SIZE];
    Eigen::Matrix4d pose;

    auto read_pose = [&pose, &line_buffer, f]() -> bool {
        // Read meta line
        if (!fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f)) {
            return false;
        }
        // Read 4x4 matrix
        for (size_t row = 0; row < 4; ++row) {
            if (!fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f)) {
                return false;
            }
            if (sscanf(line_buffer, "%lf %lf %lf %lf", &pose(row, 0),
                       &pose(row, 1), &pose(row, 2), &pose(row, 3)) != 4) {
                return false;
            }
        }
        return true;
    };

    while (read_pose()) {
        // Copy to poses
        poses.push_back(pose);
    }

    fclose(f);
    return true;
}

TEST(UniformTSDFVolume, Constructor) {
    double length = 4.0;
    int resolution = 128;
    double sdf_trunc = 0.04;
    auto color_type = integration::TSDFVolumeColorType::RGB8;
    integration::UniformTSDFVolume tsdf_volume(
            length, resolution, sdf_trunc,
            integration::TSDFVolumeColorType::RGB8);

    // TSDFVolume base class attributes
    EXPECT_EQ(tsdf_volume.voxel_length_, length / resolution);
    EXPECT_EQ(tsdf_volume.sdf_trunc_, sdf_trunc);
    EXPECT_EQ(tsdf_volume.color_type_, color_type);

    // UniformTSDFVolume attributes
    ExpectEQ(tsdf_volume.origin_, Eigen::Vector3d(0, 0, 0));
    EXPECT_EQ(tsdf_volume.length_, length);
    EXPECT_EQ(tsdf_volume.resolution_, resolution);
    EXPECT_EQ(tsdf_volume.voxel_num_, resolution * resolution * resolution);
    EXPECT_EQ(int(tsdf_volume.voxels_.size()), tsdf_volume.voxel_num_);
}

TEST(UniformTSDFVolume, RealData) {
    std::string test_data_dir = std::string(TEST_DATA_DIR);

    // Poses
    std::string trajectory_path = test_data_dir + "/RGBD/odometry.log";
    std::vector<Eigen::Matrix4d> poses;
    if (!ReadPoses(trajectory_path, poses)) {
        throw std::runtime_error("Cannot read trajectory file");
    }

    // Extrinsics
    std::vector<Eigen::Matrix4d> extrinsics;
    for (const auto& pose : poses) {
        extrinsics.push_back(pose.inverse());
    }

    // Intrinsics
    camera::PinholeCameraIntrinsic intrinsic(
            camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);

    // TSDF init
    integration::UniformTSDFVolume tsdf_volume(
            4.0, 100, 0.04, integration::TSDFVolumeColorType::RGB8);

    // Integrate RGBD frames
    for (size_t i = 0; i < poses.size(); ++i) {
        // Color
        geometry::Image im_color;
        std::ostringstream im_color_path;
        im_color_path << TEST_DATA_DIR << "/RGBD/color/" << std::setfill('0')
                      << std::setw(5) << i << ".jpg";
        io::ReadImage(im_color_path.str(), im_color);

        // Depth
        geometry::Image im_depth;
        std::ostringstream im_depth_path;
        im_depth_path << TEST_DATA_DIR << "/RGBD/depth/" << std::setfill('0')
                      << std::setw(5) << i << ".png";
        io::ReadImage(im_depth_path.str(), im_depth);

        // Ingegrate
        std::shared_ptr<geometry::RGBDImage> im_rgbd =
                geometry::RGBDImage::CreateFromColorAndDepth(
                        im_color, im_depth, /*depth_scale*/ 1000.0,
                        /*depth_func*/ 4.0, /*convert_rgb_to_intensity*/ false);
        tsdf_volume.Integrate(*im_rgbd, intrinsic, extrinsics[i]);
    }

    // These hard-coded values are for unit test only. They are used to make
    // sure that after code refactoring, the numerical values still stay the
    // same. However, using different parameters or algorithmtic improvements
    // could invalidate these reference values. We use a custom threshold 0.1
    // to account for acccumulative floating point errors.

    // Extract mesh
    std::shared_ptr<geometry::TriangleMesh> mesh =
            tsdf_volume.ExtractTriangleMesh();
    EXPECT_EQ(mesh->vertices_.size(), 3198u);
    EXPECT_EQ(mesh->triangles_.size(), 4402u);
    Eigen::Vector3d color_sum(0, 0, 0);
    for (const Eigen::Vector3d& color : mesh->vertex_colors_) {
        color_sum += color;
    }
    ExpectEQ(color_sum, Eigen::Vector3d(2703.841944, 2561.480949, 2481.503805),
             /*threshold*/ 0.1);
    // Uncomment to visualize
    // visualization::DrawGeometries({mesh});

    // Extract point cloud
    std::shared_ptr<geometry::PointCloud> pcd = tsdf_volume.ExtractPointCloud();
    EXPECT_EQ(pcd->points_.size(), 2227u);
    EXPECT_EQ(pcd->colors_.size(), 2227u);
    color_sum << 0, 0, 0;
    for (const Eigen::Vector3d& color : pcd->colors_) {
        color_sum += color;
    }
    ExpectEQ(color_sum, Eigen::Vector3d(1877.673116, 1862.126057, 1862.190616),
             /*threshold*/ 0.1);
    Eigen::Vector3d normal_sum(0, 0, 0);
    for (const Eigen::Vector3d& normal : pcd->normals_) {
        normal_sum += normal;
    }
    ExpectEQ(normal_sum, Eigen::Vector3d(-161.569098, -95.969433, -1783.167177),
             /*threshold*/ 0.1);

    // Extract voxel cloud
    std::shared_ptr<geometry::PointCloud> voxel_pcd =
            tsdf_volume.ExtractVoxelPointCloud();
    EXPECT_EQ(voxel_pcd->points_.size(), 4488u);
    EXPECT_EQ(voxel_pcd->colors_.size(), 4488u);
    color_sum << 0, 0, 0;
    for (const Eigen::Vector3d& color : voxel_pcd->colors_) {
        color_sum += color;
    }
    ExpectEQ(color_sum, Eigen::Vector3d(2096.428416, 2096.428416, 2096.428416),
             /*threshold*/ 0.1);
}

TEST(UniformTSDFVolume, DISABLED_Destructor) {}

TEST(UniformTSDFVolume, DISABLED_MemberData) {}

TEST(UniformTSDFVolume, DISABLED_Reset) {}

TEST(UniformTSDFVolume, DISABLED_Integrate) {}

TEST(UniformTSDFVolume, DISABLED_ExtractPointCloud) {}

TEST(UniformTSDFVolume, DISABLED_ExtractTriangleMesh) {}

TEST(UniformTSDFVolume, DISABLED_ExtractVoxelPointCloud) {}

TEST(UniformTSDFVolume, DISABLED_IntegrateWithDepthToCameraDistanceMultiplier) {
}

TEST(UniformTSDFVolume, DISABLED_IndexOf) {}

TEST(UniformTSDFVolume, DISABLED_GetNormalAt) {}

TEST(UniformTSDFVolume, DISABLED_GetTSDFAt) {}
